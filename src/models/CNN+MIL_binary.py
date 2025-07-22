# load libraries
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset
from collections import defaultdict
import openslide
import random
from PIL import Image
import time
import pandas as pd
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

os.makedirs("results/plots", exist_ok=True)
#
start_time = time.time()

# Load patch coordinates
patch_coords_train = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_val.csv")
patch_coords_val = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv")
# Load clinical metadata
slide_info_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
slide_info_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")

# Merge patch coordinates with clinical info
df_train = pd.merge(patch_coords_train, slide_info_train[["slide_id", "vital_status","age_at_diagnosis_years", "tumour_grade", "tumour_stage"]], on="slide_id", how="inner")
df_val = pd.merge(patch_coords_val, slide_info_val[["slide_id","vital_status","age_at_diagnosis_years", "tumour_grade", "tumour_stage"]], on="slide_id", how="inner")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class PrecomputedPatchDataset(Dataset):
    def __init__(self, base_df, transform=None, max_patches=None):
        self.grouped = defaultdict(list)
        for _, row in base_df.iterrows():
            self.grouped[row["slide_id"]].append(row)
        self.slide_ids = list(self.grouped.keys())
        self.transform = transform
        self.max_patches = max_patches

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        rows = self.grouped[slide_id]
        if self.max_patches:
            rows = random.sample(rows, min(self.max_patches, len(rows)))
        patches = []
        for row in rows:
            slide = openslide.OpenSlide(row["slide_path"])
            patch = slide.read_region((int(row["x"]), int(row["y"])), 0, (int(row["patch_px"]), int(row["patch_px"]))).convert("RGB")
            patch = patch.resize((224, 224), resample=Image.BILINEAR)
            if self.transform:
                patch = self.transform(patch)
            else:
                patch = transforms.ToTensor()(patch)
            patches.append(patch)
        patches = torch.stack(patches)  # (N_patches, 3, 224, 224)
        label = 1.0 if rows[0]["vital_status"] == "Dead" else 0.0
        return patches, torch.tensor(label, dtype=torch.float32), slide_id
    
    

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
    
# BCEWithLogitsLoss for binary classification
criterion = nn.BCEWithLogitsLoss()

#  Training Config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MILTransformerModel().to(device)
for param in model.parameters():
    assert param.requires_grad
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
n_epochs = 20

# Select first 10 unique slides instead of first 10 rows
first_20_slides = df_train["slide_id"].unique()[:20]
df_subset = df_train[df_train["slide_id"].isin(first_20_slides)]

num_dead = (df_subset["vital_status"] == "Dead").sum()
print(f"Training set positive samples (Dead): {num_dead}")
train_dataset = PrecomputedPatchDataset(df_subset, transform=transform, max_patches=100)
val_subset = df_val[df_val["slide_id"].isin(df_val["slide_id"].unique()[:100])]
val_dataset = PrecomputedPatchDataset(val_subset, transform=transform, max_patches=100)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

train_losses, val_losses = [], []
auc_list = []
train_auc_list = []

for epoch in range(n_epochs):
    model.train()
    epoch_train_loss = 0
    all_train_logits = []
    all_train_labels = []
    for patches, vital_status, _ in train_loader:
        patches, vital_status = patches.to(device), vital_status.to(device)
        optimizer.zero_grad()
        logits = model(patches.squeeze(0))  # raw logits
        logits = logits.view(-1)
        vital_status = vital_status.view(-1)
        loss = criterion(logits.view(-1), vital_status.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item()
        all_train_logits.append(logits.item())
        all_train_labels.append(vital_status.item())

    model.eval()
    epoch_val_loss = 0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for patches, vital_status, slide_id in val_loader:
            patches, vital_status = patches.to(device), vital_status.to(device)
            logits = model(patches.squeeze(0))
            # Ensure both logits and vital_status are shaped [1] for BCEWithLogitsLoss
            loss = criterion(logits.view(-1), vital_status.view(-1))
            epoch_val_loss += loss.item()
            all_logits.append(logits.item())
            all_labels.append(vital_status.item())

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_val_loss = epoch_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    try:
        auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        auc = float('nan')
    auc_list.append(auc)
    try:
        train_auc = roc_auc_score(all_train_labels, all_train_logits)
    except ValueError:
        train_auc = float('nan')
    train_auc_list.append(train_auc)
    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Train AUC: {train_auc:.4f} - Val AUC: {auc:.4f}")


# After final training epoch, plot the last attention weights
if hasattr(model.mil_head, 'last_attention'):
    plt.figure()
    plt.plot(model.mil_head.last_attention.squeeze().numpy())
    plt.title("Final Attention Weights")
    plt.savefig("results/plots/final_attention_weights_binary_cnnnmil.png")
    plt.close()

# --- Plot Loss Curve ---
plt.figure()
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Training & Validation Loss - Binary Classification")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/cnn_mil_loss_curve_20trainslides_binary.png")
plt.close()

# --- Plot Accuracy (AUC) Curve ---
plt.figure()
plt.plot(train_auc_list, label="Train AUC", marker='o')
plt.plot(auc_list, label="Validation AUC", marker='o')
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Training & Validation AUC - Binary Classification")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/cnn_mil_auc_curve_20trainslides_binary.png")
plt.close()

from sklearn.metrics import roc_curve

# ROC Curve (Final Epoch)
fpr, tpr, _ = roc_curve(all_labels, all_logits)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Final Epoch")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/cnn_mil_final_roc_curve.png")
plt.close()

# Calibration Scatter Plot - Final Epoch
plt.figure()
plt.scatter(all_logits, all_labels, alpha=0.4)
plt.xlabel("Predicted Logits")
plt.ylabel("True Label")
plt.title("Validation Calibration - Predicted vs True Labels")
plt.grid(True)
plt.savefig("results/plots/cnn_mil_val_calibration.png")
plt.close()

# For training set
model.eval()
all_train_logits, all_train_labels = [], []
with torch.no_grad():
    for patches, vital_status, _ in train_loader:
        patches, vital_status = patches.to(device), vital_status.to(device)
        logit = model(patches.squeeze(0))
        all_train_logits.append(logit.item())
        all_train_labels.append(vital_status.item())

plt.figure()
plt.scatter(all_train_logits, all_train_labels, alpha=0.4)
plt.xlabel("Predicted Logits")
plt.ylabel("True Label")
plt.title("Train Calibration - Predicted vs True Labels")
plt.grid(True)
plt.savefig("results/plots/cnn_mil_train_calibration.png")
plt.close()

print("Done.")
end_time = time.time()
elapsed = end_time - start_time
# Print to stdout 
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save file
with open("results/plots/cnn_mil_20trainslides_binary_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")
