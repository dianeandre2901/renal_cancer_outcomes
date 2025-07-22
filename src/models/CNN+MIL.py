# load libraries
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset
from collections import defaultdict
import random
from PIL import Image
import time as time_module
import pandas as pd
from torchvision import transforms

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
import os
import numpy as np

os.makedirs("results/plots", exist_ok=True)
#
start_time = time_module.time()

# Load clinical metadata
slide_info_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
slide_info_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")

df_train = slide_info_train
df_val = slide_info_val
df_train = df_train[["slide_id", "event", "os_days"]].copy()
df_val = df_val[["slide_id", "event", "os_days"]].copy()

# Define feature directories
features_train_dir = "/rds/general/user/dla24/home/thesis/src/scripts/features_train/"
features_val_dir = "/rds/general/user/dla24/home/thesis/src/scripts/features_train_validation/"

# Get slide IDs that have corresponding .npy features
train_slide_ids = [f.replace(".npy", "") for f in os.listdir(features_train_dir) if f.endswith(".npy")]
val_slide_ids = [f.replace(".npy", "") for f in os.listdir(features_val_dir) if f.endswith(".npy")]

slide_info_train = slide_info_train[slide_info_train["slide_id"].isin(train_slide_ids)].copy()
slide_info_val = slide_info_val[slide_info_val["slide_id"].isin(val_slide_ids)].copy()

# Drop missing os_days or event entries
slide_info_train = slide_info_train.dropna(subset=["os_days", "event"])
slide_info_val = slide_info_val.dropna(subset=["os_days", "event"])

# Ensure correct types
slide_info_train["slide_id"] = slide_info_train["slide_id"].astype(str)
slide_info_val["slide_id"] = slide_info_val["slide_id"].astype(str)

print("Found .npy train features for:", train_slide_ids[:5])
print("Found .npy val features for:", val_slide_ids[:5])
print("Before filtering:", slide_info_train.shape, slide_info_val.shape)

# Filter slide_info to only include those with corresponding features
slide_info_train = slide_info_train[slide_info_train["slide_id"].isin(train_slide_ids)]
slide_info_val = slide_info_val[slide_info_val["slide_id"].isin(val_slide_ids)]

# Select top 20 valid slide IDs with event==1
valid_train_ids = slide_info_train.query("event == 1")["slide_id"].unique()[:20]
if len(valid_train_ids) == 0:
    print("ERROR: No slides with event==1 found in training set after filtering.")
    exit(1)
valid_val_ids = slide_info_val["slide_id"].unique()[:20]

df_subset = slide_info_train[slide_info_train["slide_id"].isin(valid_train_ids)]
val_subset = slide_info_val[slide_info_val["slide_id"].isin(valid_val_ids)]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PrecomputedNumpyDataset(Dataset):
    def __init__(self, slide_info_df, features_dir, max_patches=None):
        self.slide_info_df = slide_info_df.reset_index(drop=True)
        self.features_dir = features_dir
        self.max_patches = max_patches

    def __len__(self):
        return len(self.slide_info_df)

    def __getitem__(self, idx):
        row = self.slide_info_df.iloc[idx]
        slide_id = row["slide_id"]
        feature_path = os.path.join(self.features_dir, f"{slide_id}.npy")
        features = np.load(feature_path)  # shape (N_patches, 512)
        if self.max_patches:
            if features.shape[0] > self.max_patches:
                selected_indices = random.sample(range(features.shape[0]), self.max_patches)
                features = features[selected_indices]
        patches = torch.tensor(features, dtype=torch.float32)
        os_days = torch.tensor(row["os_days"], dtype=torch.float32)
        event = torch.tensor(row["event"], dtype=torch.float32)
        return patches, os_days, event, slide_id
    

class TransformerMIL(nn.Module):
    def __init__(self, input_dim, embed_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        x: Tensor of shape (1, N_patches, input_dim)
        """
        x = self.embedding(x)  # (1, N_patches, embed_dim)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        x = self.transformer(x)  # (1, N_patches, embed_dim)
        attention_weights = torch.softmax(self.attention_head(x), dim=1)  # (1, N_patches, 1)
        self.last_attention = attention_weights.detach().cpu()
        weighted_avg = torch.sum(attention_weights * x, dim=1)  # (1, embed_dim)
        output = torch.sum(weighted_avg, dim=1, keepdim=True)  # (1, 1)
        return output

class MILTransformerModel(nn.Module):
    def __init__(self, 
                 embed_dim=512, 
                 num_heads=8, 
                 num_layers=2, 
                 dropout=0.1):
        super().__init__()


        # CNN patch encoder (EfficientNet backbone)
        self.feature_extractor = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.feature_extractor.classifier[1].in_features
        self.feature_extractor.classifier = nn.Identity()

        #  Transformer MIL head 
        self.mil_head = TransformerMIL(
            input_dim=in_features,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x):  
        """
        x: Tensor of shape (N_patches, 3, H, W)  ← patches from one slide
        """
        N = x.size(0)
        x = self.feature_extractor(x)             # (N_patches, input_dim)
        x = x.unsqueeze(0)                        # (1, N_patches, input_dim)
        out = self.mil_head(x)                    # (1, 1)
        return out.squeeze(0).squeeze(-1)   

# Autoencoder for patches
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (B, 64, H/4, W/4)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, latent_dim)  # assuming input patch size is 256x256
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 64 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (64, 64, 64)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # (B, 3, H, W)
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    

# Simple Cox Loss 
def cox_loss(risk_scores, os_days, event, epsilon=1e-8):
    # sort by descending survival time
    sorted_idx = torch.argsort(os_days, descending=True)
    risk_scores = risk_scores[sorted_idx]
    event = event[sorted_idx]
    hazard_ratio = torch.exp(risk_scores)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + epsilon)
    uncensored_likelihood = risk_scores - log_risk
    if torch.sum(event) == 0:
        return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)
    loss = -torch.mean(uncensored_likelihood[event == 1])
    return loss

#  Training Config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MILTransformerModel().to(device)
for param in model.parameters():
    assert param.requires_grad
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
n_epochs = 20
print("Filtered slide_info_train shape:", slide_info_train.shape)
 # Select 10 uncensored (event==1) + 10 censored (event==0) for training
uncensored = slide_info_train.query("event == 1").sample(n=10, random_state=42)
censored = slide_info_train.query("event == 0").sample(n=10, random_state=42)
df_subset = pd.concat([uncensored, censored]).reset_index(drop=True)

print("Training set events:", df_subset["event"].sum())

train_dataset = PrecomputedNumpyDataset(df_subset, features_train_dir, max_patches=100)
# Select 10 uncensored + 10 censored for validation
val_uncensored = slide_info_val.query("event == 1").sample(n=10, random_state=42)
val_censored = slide_info_val.query("event == 0").sample(n=10, random_state=42)
val_subset = pd.concat([val_uncensored, val_censored]).reset_index(drop=True)
val_dataset = PrecomputedNumpyDataset(val_subset, features_val_dir, max_patches=100)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

train_losses, val_losses = [], []
c_index_list = []

for epoch in range(n_epochs):
    model.train()
    epoch_train_loss = 0
    for patches, os_days, event, _ in train_loader:
        patches, os_days, event = patches.to(device), os_days.to(device), event.to(device)
        optimizer.zero_grad()
        risk_score = model(patches.squeeze(0)) 
        loss = cox_loss(risk_score.unsqueeze(0), os_days, event)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item()

    model.eval()
    epoch_val_loss = 0
    all_risk_scores = []
    all_events = []
    all_os_days = []
    with torch.no_grad():
        for patches, os_days, event, slide_id in val_loader:
            patches, os_days, event = patches.to(device), os_days.to(device), event.to(device)
            risk_score = torch.tanh(model(patches.squeeze(0)))
            loss = cox_loss(risk_score.unsqueeze(0), os_days, event)
            epoch_val_loss += loss.item()
            all_risk_scores.append(risk_score.item())
            all_events.append(event.item())
            all_os_days.append(os_days.item())

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_val_loss = epoch_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    c_index = concordance_index(all_os_days, [-r for r in all_risk_scores], all_events)
    c_index_list.append(c_index)
    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - C-index: {c_index:.4f}")


# --- Plot Loss Curve ---
plt.figure()
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Cox Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/cnn_mil_loss_curve_20trainslides.png")
plt.close()

# --- Plot C-index Performance Curve ---
plt.figure()
plt.plot(c_index_list, label="C-index", marker='o', color='green')
plt.xlabel("Epoch")
plt.ylabel("C-index")
plt.title("C-index over Epochs")
plt.grid(True)
plt.savefig("results/plots/cnn_mil_cindex_curve_20trainslides.png")
plt.close()

print("Done.")
end_time = time_module.time()
elapsed = end_time - start_time
# Print to stdout 
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save file
with open("results/plots/cnn_mil_20trainslides_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")
