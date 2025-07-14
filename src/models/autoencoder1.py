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
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#
start_time = time.time()

# Load patch coordinates
patch_coords_train = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv")
patch_coords_val = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_val.csv")
# Load clinical metadata
slide_info_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")
slide_info_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")

# Merge patch coordinates with clinical info
df_train = pd.merge(patch_coords_train, slide_info_train[["slide_id", "os_days", "event","age_at_diagnosis_years"]], on="slide_id", how="inner")
df_val = pd.merge(patch_coords_val, slide_info_val[["slide_id", "os_days", "event","age_at_diagnosis_years"]], on="slide_id", how="inner")

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
        return patches, torch.tensor(rows[0]["os_days"], dtype=torch.float32), torch.tensor(rows[0]["event"], dtype=torch.float32), slide_id
    

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
        x = self.transformer(x)  # (1, N_patches, embed_dim)
        attention_weights = torch.softmax(self.attention_head(x), dim=1)  # (1, N_patches, 1)
        weighted_avg = torch.sum(attention_weights * x, dim=1)  # (1, embed_dim)
        output = torch.sum(weighted_avg, dim=1, keepdim=True)  # (1, 1)
        return output

class MILTransformerModel(nn.Module):
    def __init__(self, 
                 resnet_type='resnet18',
                 input_dim=512, 
                 embed_dim=512, 
                 num_heads=8, 
                 num_layers=2, 
                 dropout=0.1):
        super().__init__()

        # --- CNN patch encoder (ResNet backbone) ---
        if resnet_type == 'resnet18':
            self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
            in_features = 512
        elif resnet_type == 'resnet50':
            self.feature_extractor = models.resnet50(pretrained=True)
            in_features = 2048
        else:
            raise ValueError("Unsupported backbone")

        self.feature_extractor.fc = nn.Identity()  # Remove classification head

        # --- Transformer MIL head ---
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
        return out.squeeze(0).squeeze(-1)         # scalar prediction


# ------------------- Autoencoder for 2D image patches -------------------
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
    


end_time = time.time()
elapsed = end_time - start_time
# Print to stdout (it will appear in your .log)
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to a file for easy access
with open("results/plots/MIL+autoencoder_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")


import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# --- Simple Cox Loss ---
def cox_loss(risk_scores, os_days, event, epsilon=1e-8):
    # sort by descending survival time
    sorted_idx = torch.argsort(os_days, descending=True)
    risk_scores = risk_scores[sorted_idx]
    event = event[sorted_idx]
    hazard_ratio = torch.exp(risk_scores)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + epsilon)
    uncensored_likelihood = risk_scores - log_risk
    loss = -torch.mean(uncensored_likelihood[event == 1])
    return loss

# --- Training Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MILTransformerModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
n_epochs = 10

# Select first 10 unique slides instead of first 10 rows
first_10_slides = df_train["slide_id"].unique()[:10]
df_subset = df_train[df_train["slide_id"].isin(first_10_slides)]
train_dataset = PrecomputedPatchDataset(df_subset, transform=None, max_patches=50)
val_dataset = PrecomputedPatchDataset(df_subset, transform=None, max_patches=50)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

train_losses, val_losses = [], []

for epoch in range(n_epochs):
    model.train()
    epoch_train_loss = 0
    for patches, os_days, event, _ in train_loader:
        patches, os_days, event = patches.to(device), os_days.to(device), event.to(device)
        optimizer.zero_grad()
        risk_score = model(patches.squeeze(0).to(device))  # scalar
        loss = cox_loss(risk_score.unsqueeze(0), os_days, event)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for patches, os_days, event, _ in val_loader:
            patches, os_days, event = patches.to(device), os_days.to(device), event.to(device)
            risk_score = model(patches.squeeze(0).to(device))
            loss = cox_loss(risk_score.unsqueeze(0), os_days, event)
            epoch_val_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_val_loss = epoch_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

# --- Plot Loss Curve ---
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig("results/plots/survival_training_loss.png")
plt.close()

print("Done.")
end_time = time.time()
elapsed = end_time - start_time
# Print to stdout 
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save file
with open("results/plots/autoencoder1_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")