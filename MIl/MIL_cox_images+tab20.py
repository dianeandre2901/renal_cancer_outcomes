"""
Survival MIL Cox Model: MIL Attention Aggregation over Image Patches (EfficientNet-B0) + Tabular MLP
20 slides for training - 20 for validation
Trains a multimodal model producing a scalar risk score for Cox proportional hazards regression.
Reports Cox loss & C-index, plots loss calibration and C-statistic style plots, early stopping on best C-index.
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import openslide
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from lifelines.utils import concordance_index

# --- Setup ---
os.makedirs("results/plots", exist_ok=True)
start_time = time.time()

# --- Tabular features ---
tabular_features = ["age_at_diagnosis_years", "tumour_grade", "tumour_stage"]

# --- Load slide-level data ---
df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")
df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
df_train["slide_id"] = df_train["slide_id"].astype(str)
df_val["slide_id"] = df_val["slide_id"].astype(str)

# --- Patch-level coordinates ---
train_patches = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv")
val_patches = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_val.csv")

# --- Merge tabular features into patch DataFrames ---
df_train = df_train.rename(columns={"os_days": "time"})
df_val = df_val.rename(columns={"os_days": "time"})
tabular_features_train = df_train[["slide_id"] + tabular_features + ["time", "event"]]
tabular_features_val = df_val[["slide_id"] + tabular_features + ["time", "event"]]
train_patches = train_patches.merge(tabular_features_train, on="slide_id", how="left")
val_patches = val_patches.merge(tabular_features_val, on="slide_id", how="left")

# --- Restrict to 20 train/20 val slides (all their patches) ---
first_20_train = train_patches[train_patches['slide_id'].isin(train_patches['slide_id'].unique()[:20])]
first_20_val   = val_patches[val_patches['slide_id'].isin(val_patches['slide_id'].unique()[:20])]
patch_cap = 100  # Max patches per slide (or None for all)

# --- Dataset ---
class MultiModalPatchDataset(Dataset):
    def __init__(self, patch_df, transform=None, max_patches_per_slide=None):
        if max_patches_per_slide:
            patch_df = patch_df.groupby("slide_id", group_keys=False).apply(
                lambda g: g.sample(min(max_patches_per_slide, len(g)), random_state=42)
            ).reset_index(drop=True)
        self.df = patch_df.reset_index(drop=True)
        self.transform = transform
        self.slide_ids = self.df["slide_id"].unique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide = openslide.OpenSlide(row["slide_path"])
        patch = slide.read_region((int(row["x"]), int(row["y"])), 0,
                                  (int(row["patch_px"]), int(row["patch_px"]))).convert("RGB")
        patch = patch.resize((224, 224), resample=Image.BILINEAR)
        if self.transform:
            patch = self.transform(patch)
        # Tabular features
        tabular = torch.tensor([float(row[f]) for f in tabular_features], dtype=torch.float32)
        # Survival targets
        time_val = float(row["time"])
        event_val = float(row["event"])
        slide_id = row["slide_id"]
        return patch, tabular, time_val, event_val, slide_id

# --- Augmentation ---
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --- DataLoader ---
train_dataset = MultiModalPatchDataset(first_20_train, transform=transform, max_patches_per_slide=patch_cap)
val_dataset   = MultiModalPatchDataset(first_20_val, transform=transform, max_patches_per_slide=patch_cap)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

# --- Model ---
class AttentionMILCox(nn.Module):
    def __init__(self, base_model, feature_dim=1280, tabular_dim=3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            base_model.features,
            base_model.avgpool,
            nn.Flatten()
        )
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.tabular_fc = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU()
        )
        self.final_fc = nn.Linear(feature_dim + 128, 1)  # Scalar risk score

    def forward(self, x, tabular):
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)  # (B*N, C, H, W)
        features = self.feature_extractor(x)  # (B*N, D)
        features = features.view(B, N, -1)  # (B, N, D)
        attn_scores = self.attention(features)  # (B, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, N, 1)
        weighted_feat = torch.sum(attn_weights * features, dim=1)  # (B, D)
        tab = self.tabular_fc(tabular)  # (B, 128)
        combined = torch.cat([weighted_feat, tab], dim=1)  # (B, D+128)
        risk = self.final_fc(combined).squeeze(1)  # (B,)
        return risk

# --- Instantiate model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model = AttentionMILCox(base_model).to(device)
# Freeze all blocks except 6 and 7
for name, param in model.feature_extractor[0].named_parameters():
    if "6" in name or "7" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# --- Cox loss ---
def cox_ph_loss(risk_scores, times, events):
    # risk_scores: (B,) higher = higher risk
    # times, events: (B,) float, float
    # Compute partial likelihood loss (neg log likelihood)
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]
    loss = 0.
    n_events = 0
    max_clip = 80  # for stability in exp
    for i in range(risk_scores.shape[0]):
        if events[i] == 1:
            n_events += 1
            denom = torch.logsumexp(risk_scores[i:], dim=0)
            loss += (risk_scores[i] - denom)
    if n_events == 0:
        return torch.tensor(0., device=risk_scores.device)
    return -loss / n_events

# --- Print patch/slide summary ---
def print_patch_summary(dataset, name):
    patch_counts = dataset.df['slide_id'].value_counts()
    counts = patch_counts.values
    if len(counts) == 0:
        print(f"{name} set: 0 slides, 0 patches (EMPTY)")
        return
    print(f"{name} set: {len(patch_counts)} slides, {len(dataset)} patches")
    print(f"  Avg patches/slide: {np.mean(counts):.1f}, min: {np.min(counts)}, max: {np.max(counts)}")

print_patch_summary(train_dataset, "Train")
print_patch_summary(val_dataset, "Val")

# --- Optimizer ---
optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-5, weight_decay=1e-4)
epochs = 40
patience = 5
best_val_cindex = 0
epochs_since_improvement = 0
best_model_state = None

# --- Training Loop ---
train_loss_list = []
val_loss_list = []
val_cindex_list = []
best_epoch = 0
for epoch in range(epochs):
    model.train()
    running_loss, total = 0, 0
    slide_risks = defaultdict(list)
    slide_times = {}
    slide_events = {}
    # --- TRAIN ---
    for imgs, tabular, times, events, slide_ids in train_loader:
        imgs = imgs.view(imgs.size(0), -1, 3, 224, 224)
        imgs = imgs.to(device)
        tabular = tabular.to(device)
        times = times.to(device)
        events = events.to(device)
        optimizer.zero_grad()
        risk_scores = model(imgs, tabular)
        loss = cox_ph_loss(risk_scores, times, events)
        loss = loss.requires_grad_()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        # For slide-level aggregation
        for i, sid in enumerate(slide_ids):
            slide_risks[sid].append(risk_scores[i].detach().cpu().item())
            slide_times[sid] = times[i].cpu().item()
            slide_events[sid] = events[i].cpu().item()
    train_loss = running_loss / total if total > 0 else float("nan")
    train_loss_list.append(train_loss)

    # --- VAL ---
    model.eval()
    val_running_loss, val_total = 0, 0
    val_slide_risks = defaultdict(list)
    val_slide_times = {}
    val_slide_events = {}
    with torch.no_grad():
        for imgs, tabular, times, events, slide_ids in val_loader:
            imgs = imgs.view(imgs.size(0), -1, 3, 224, 224)
            imgs = imgs.to(device)
            tabular = tabular.to(device)
            times = times.to(device)
            events = events.to(device)
            risk_scores = model(imgs, tabular)
            loss = cox_ph_loss(risk_scores, times, events)
            loss = loss.requires_grad_()
            val_running_loss += loss.item() * imgs.size(0)
            val_total += imgs.size(0)
            for i, sid in enumerate(slide_ids):
                val_slide_risks[sid].append(risk_scores[i].cpu().item())
                val_slide_times[sid] = times[i].cpu().item()
                val_slide_events[sid] = events[i].cpu().item()
    val_loss = val_running_loss / val_total if val_total > 0 else float("nan")
    val_loss_list.append(val_loss)

    # --- Slide-level aggregation (mean risk per slide) ---
    val_slide_ids = list(val_slide_risks.keys())
    slide_mean_risks = np.array([np.mean(val_slide_risks[sid]) for sid in val_slide_ids])
    slide_times = np.array([val_slide_times[sid] for sid in val_slide_ids])
    slide_events = np.array([val_slide_events[sid] for sid in val_slide_ids])
    # --- C-index ---
    try:
        val_cindex = concordance_index(slide_times, -slide_mean_risks, slide_events)  # -risk: higher risk = shorter time
    except Exception:
        val_cindex = float('nan')
    val_cindex_list.append(val_cindex)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val C-index: {val_cindex:.4f}")

    # --- Early stopping on best C-index ---
    if val_cindex > best_val_cindex:
        best_val_cindex = val_cindex
        best_model_state = model.state_dict()
        best_epoch = epoch
        epochs_since_improvement = 0
        torch.save(model.state_dict(), "results/plots/MIL_cox_images+tab20_best.pth")
    else:
        epochs_since_improvement += 1
    if epochs_since_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

# --- Plot loss curves ---
epochs_range = range(1, len(train_loss_list)+1)
plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_loss_list, label="Train Loss", marker='o', color='red')
plt.plot(epochs_range, val_loss_list, label="Val Loss", marker='o', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Cox Loss")
plt.title("MIL Cox: Train/Val Loss per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/MIL_cox_images+tab20_loss_curve.png")
plt.close()
print("Saved loss calibration plot to results/plots/MIL_cox_images+tab20_loss_curve.png")

# --- Plot C-index curve ---
plt.figure(figsize=(8,6))
plt.plot(epochs_range, val_cindex_list, label="Val C-index", marker='s', color='green')
plt.xlabel("Epoch")
plt.ylabel("C-index")
plt.title("MIL Cox: Validation C-index per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/MIL_cox_images+tab20_cindex_curve.png")
plt.close()
print("Saved C-index curve to results/plots/MIL_cox_images+tab20_cindex_curve.png")

# --- Final C-statistic style plot (sorted risk vs survival) ---
# Reload best model
model.load_state_dict(torch.load("results/plots/MIL_cox_images+tab20_best.pth"))
model.eval()
val_slide_risks = defaultdict(list)
val_slide_times = {}
val_slide_events = {}
with torch.no_grad():
    for imgs, tabular, times, events, slide_ids in val_loader:
        imgs = imgs.view(imgs.size(0), -1, 3, 224, 224)
        imgs = imgs.to(device)
        tabular = tabular.to(device)
        risk_scores = model(imgs, tabular)
        for i, sid in enumerate(slide_ids):
            val_slide_risks[sid].append(risk_scores[i].cpu().item())
            val_slide_times[sid] = float(times[i])
            val_slide_events[sid] = float(events[i])
slide_ids = list(val_slide_risks.keys())
slide_mean_risks = np.array([np.mean(val_slide_risks[sid]) for sid in slide_ids])
slide_times = np.array([val_slide_times[sid] for sid in slide_ids])
slide_events = np.array([val_slide_events[sid] for sid in slide_ids])
# Sort by risk
sort_idx = np.argsort(slide_mean_risks)[::-1]  # descending risk
plt.figure(figsize=(10,5))
plt.scatter(np.arange(len(sort_idx)), slide_times[sort_idx], c=slide_events[sort_idx], cmap="coolwarm", s=60, edgecolor='k')
plt.xlabel("Slides (sorted by predicted risk)")
plt.ylabel("Survival Time (days)")
plt.title("MIL Cox: Sorted Risk vs Survival (Val set)\nRed=Event, Blue=Censored")
cbar = plt.colorbar()
cbar.set_label("Event (1=event, 0=censored)")
plt.tight_layout()
plt.savefig("results/plots/MIL_cox_images+tab20_cstat_scatter.png")
plt.close()
print("Saved C-statistic style plot to results/plots/MIL_cox_images+tab20_cstat_scatter.png")

# --- Save runtime ---
end_time = time.time()
elapsed = end_time - start_time
with open("results/plots/MIL_cox_images+tab20_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")