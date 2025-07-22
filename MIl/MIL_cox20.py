"""
Binary Classifier with MIL Attention Aggregation over Patches
20 slides for training - 100 for validation
This model trains an EfficientNet-B0 with MIL attention to classify WSI slides as Alive/Dead,
based on precomputed tissue coordinates. Patches inherit slide-level labels. At evaluation,
patch features are aggregated with attention to derive a slide-level prediction.
Includes patch- and slide-level accuracy, confusion matrices, and loss/accuracy curves.
"""
from lifelines.utils import concordance_index
from scipy.interpolate import make_interp_spline
from matplotlib import pyplot as plt
from seaborn import heatmap, set_style
import openslide
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
from torchvision import transforms
import pandas as pd
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
import random
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from collections import Counter
import time
# 
start_time = time.time()



# Load data
df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")
df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
df_train['slide_id'] = df_train['slide_id'].astype(str)
df_train = df_train[["slide_id", "event", "os_days"]]
df_val = df_val[["slide_id", "event", "os_days"]]
class PrecomputedPatchDataset(Dataset):
    def __init__(self, patch_csv, transform=None, max_patches_per_slide=None):
        if isinstance(patch_csv, pd.DataFrame):
           self.df = patch_csv.reset_index(drop=True)
           self.transform = transform 
        else:
           self.df = pd.read_csv(patch_csv)
           self.transform = transform 
        if max_patches_per_slide is not None:
            # Optionally subsample to avoid OOM
            self.df = self.df.groupby("slide_id").apply(
                lambda g: g.sample(min(max_patches_per_slide, len(g)), random_state=42)
            ).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        assert "event" in row and "os_days" in row, "Missing event or os_days in dataset row"
        slide = openslide.OpenSlide(row["slide_path"])
        patch = slide.read_region((int(row["x"]), int(row["y"])), 0, (int(row["patch_px"]), int(row["patch_px"]))).convert("RGB")
        patch = patch.resize((224, 224), resample=Image.BILINEAR)
        label = row["label"]
        slide_id = row["slide_id"]
        if self.transform:
            patch = self.transform(patch)
        return patch, label, slide_id

#  augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


# Load and filter CSV
train_patches = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv")
val_patches   = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_val.csv")

# Merge survival info into patch DataFrames
train_patches = train_patches.merge(df_train[["slide_id", "event", "os_days"]], on="slide_id", how="inner")
val_patches = val_patches.merge(df_val[["slide_id", "event", "os_days"]], on="slide_id", how="inner")

# Ensure both event==1 and event==0 slides are included in train set
evented_train_slides = df_train[df_train["event"] == 1]["slide_id"].unique()
censored_train_slides = df_train[df_train["event"] == 0]["slide_id"].unique()

# Take up to 10 of each if available
selected_train_slides = list(evented_train_slides[:10]) + list(censored_train_slides[:10])
first_20_train = train_patches[train_patches["slide_id"].isin(selected_train_slides)]

# Same logic for validation
evented_val_slides = df_val[df_val["event"] == 1]["slide_id"].unique()
censored_val_slides = df_val[df_val["event"] == 0]["slide_id"].unique()
selected_val_slides = list(evented_val_slides[:10]) + list(censored_val_slides[:10])
first_100_val = val_patches[val_patches["slide_id"].isin(selected_val_slides)]

print("Train set event counts:")
print(df_train[df_train["slide_id"].isin(selected_train_slides)]["event"].value_counts())

print("Val set event counts:")
print(df_val[df_val["slide_id"].isin(selected_val_slides)]["event"].value_counts())

patch_cap = 100 # or None for all

train_dataset = PrecomputedPatchDataset(first_20_train, transform=transform, max_patches_per_slide=patch_cap)
val_dataset   = PrecomputedPatchDataset(first_100_val, transform=transform, max_patches_per_slide=patch_cap)

#train_dataset = PrecomputedPatchDataset("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv", transform=transform, max_patches_per_slide=patch_cap)
#val_dataset   = PrecomputedPatchDataset("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_val.csv", transform=transform, max_patches_per_slide=patch_cap)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionMIL(nn.Module):
    def __init__(self, base_model, feature_dim=1280):
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
        self.classifier = nn.Linear(feature_dim, 1)  # Output risk score

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)  # (B*N, C, H, W)
        features = self.feature_extractor(x)  # (B*N, D)
        features = features.view(B, N, -1)  # (B, N, D)
        attn_scores = self.attention(features)  # (B, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, N, 1)
        weighted_feat = torch.sum(attn_weights * features, dim=1)  # (B, D)
        out = self.classifier(weighted_feat)  # (B, 1)
        return out.squeeze(-1)  # (B,)

base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model = AttentionMIL(base_model).to(device)
# Freeze all blocks except 6 and 7
for name, param in model.feature_extractor[0].named_parameters():
    if "6" in name or "7" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Cox proportional hazards loss
def cox_ph_loss(risk_scores, times, events):
    # risk_scores: (B,) predicted risk (higher = higher risk)
    # times: (B,) observed times
    # events: (B,) 1 if event occurred, 0 if censored
    # Sort by descending time (largest time first)
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    times = times[order]
    events = events[order]
    log_cumsum_exp = torch.logcumsumexp(risk_scores, dim=0)
    diff = risk_scores - log_cumsum_exp
    loss = -torch.sum(diff * events) / (torch.sum(events) + 1e-8)
    return loss


optimizer = torch.optim.Adam(model.parameters(), lr=2.4844087551934078e-05, weight_decay=0.00011136085331748044 )
# Early stopping
best_val_cindex = 0
patience = 2
epochs_since_improvement = 0
best_model_state = None

epochs = 20
def print_patch_summary(dataset, name):
    patch_counts = dataset.df['slide_id'].value_counts()
    counts = patch_counts.values
    if len(counts) == 0:
        print(f"{name} set: 0 slides, 0 patches (EMPTY)")
        return
    print(f"{name} set: {len(patch_counts)} slides, {len(dataset)} patches")
    print(f"  Avg patches/slide: {np.mean(counts):.1f}, min: {np.min(counts)}, max: {np.max(counts)}")

# Collate function to skip batches with no events (placeholder logic)
def skip_censored_collate_fn(batch):
    imgs, labels, slide_ids = zip(*batch)
    events = [float(train_dataset.df[train_dataset.df['slide_id'] == sid]['event'].iloc[0]) for sid in slide_ids]
    imgs = torch.stack(imgs)
    return imgs, labels, slide_ids if sum(events) > 0 else None

print_patch_summary(train_dataset, "Train")
print_patch_summary(val_dataset, "Val")

train_loss_list = []
val_loss_list = []
train_cindex_list = []
val_cindex_list = []
for epoch in range(epochs):
    # --- TRAIN ---
    model.train()
    running_loss, total = 0, 0
    all_train_risk = []
    all_train_time = []
    all_train_event = []
    for imgs, labels, slide_ids in train_loader:
        # For CoxPH, labels should be event indicators (1/0), and  survival times and events
        
        # For now, let's assume they are available in train_dataset.df
        idxs = [train_dataset.df.index[train_dataset.df['slide_id'] == sid][0] for sid in slide_ids]
        # Use slide-level time/event (all patches from slide share same time/event)
        times = torch.tensor([float(train_dataset.df.iloc[idx]['os_days']) for idx in idxs], dtype=torch.float32).to(device)
        events = torch.tensor([float(train_dataset.df.iloc[idx]['event']) for idx in idxs], dtype=torch.float32).to(device)
        imgs = imgs.view(imgs.size(0), -1, 3, 224, 224)
        imgs = imgs.to(device)
        optimizer.zero_grad()
        risk_scores = model(imgs)
        loss = cox_ph_loss(risk_scores, times, events)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        all_train_risk.extend(risk_scores.detach().cpu().numpy())
        all_train_time.extend(times.cpu().numpy())
        all_train_event.extend(events.cpu().numpy())
    train_loss = running_loss / total if total > 0 else float("nan")
    # Compute C-index for training set
    train_cindex = concordance_index(all_train_time, -np.array(all_train_risk), all_train_event)

    # --- VALIDATION ---
    model.eval()
    val_loss, val_total = 0, 0
    all_val_risk = []
    all_val_time = []
    all_val_event = []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            imgs, labels, slide_ids = batch
            idxs = [val_dataset.df.index[val_dataset.df['slide_id'] == sid][0] for sid in slide_ids]
            times = torch.tensor([float(val_dataset.df.iloc[idx]['os_days']) for idx in idxs], dtype=torch.float32).to(device)
            events = torch.tensor([float(val_dataset.df.iloc[idx]['event']) for idx in idxs], dtype=torch.float32).to(device)
            imgs = imgs.view(imgs.size(0), -1, 3, 224, 224)
            imgs = imgs.to(device)
            risk_scores = model(imgs)
            loss = cox_ph_loss(risk_scores, times, events)
            val_loss += loss.item() * imgs.size(0)
            val_total += imgs.size(0)
            all_val_risk.extend(risk_scores.cpu().numpy())
            all_val_time.extend(times.cpu().numpy())
            all_val_event.extend(events.cpu().numpy())
    val_loss = val_loss / val_total if val_total > 0 else float("nan")
    val_cindex = concordance_index(all_val_time, -np.array(all_val_risk), all_val_event)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_cindex_list.append(train_cindex)
    val_cindex_list.append(val_cindex)

    print(f"Epoch {epoch + 1}/{epochs} | "
          f"Train Loss: {train_loss:.4f} | Train C-index: {train_cindex:.3f} | "
          f"Val Loss: {val_loss:.4f} | Val C-index: {val_cindex:.3f}")

    if val_cindex > best_val_cindex:
        best_val_cindex = val_cindex
        epochs_since_improvement = 0
        best_model_state = model.state_dict()
        torch.save(model.state_dict(), "model_MIL_cox_images20.pth")
    else:
        epochs_since_improvement += 1
    if epochs_since_improvement >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

epochs_range = range(1, len(train_loss_list) + 1)
plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_loss_list, label="Train Loss", marker='o', color='red')
plt.plot(epochs_range, val_loss_list, label="Val Loss", marker='o', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss per Epoch (CoxPH)")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/MIL_cox_images20_train_val_loss.png")
plt.close()
print("Saved loss plot to results/plots/MIL_cox_images20_train_val_loss.png")

# --- C-index Plot ---
plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_cindex_list, label="Train C-index", marker='s', color='blue')
plt.plot(epochs_range, val_cindex_list, label="Val C-index", marker='s', color='green')
plt.xlabel("Epoch")
plt.ylabel("C-index")
plt.title("Train vs Validation C-index per Epoch (CoxPH)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/MIL_cox_images20_train_val_cindex.png")
plt.close()
print("Saved C-index plot to results/plots/MIL_cox_images20_train_val_cindex.png")

# --- Calibration Curve (Loss Only) ---
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, train_loss_list, label="Train Loss", marker='o', linestyle='-')
plt.plot(epochs_range, val_loss_list, label="Validation Loss", marker='o', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Calibration Plot: Loss Curve")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/MIL_cox_images20_calibration_loss_curve.png")
plt.close()
print("Saved calibration plot to results/plots/MIL_cox_images20_calibration_loss_curve.png")

# --- C-Statistic Visual ROC-style ---
try:
    sorted_idx = np.argsort(-np.array(all_val_risk))
    sorted_times = np.array(all_val_time)[sorted_idx]
    sorted_events = np.array(all_val_event)[sorted_idx]
    norm_times = (sorted_times - np.min(sorted_times)) / (np.max(sorted_times) - np.min(sorted_times) + 1e-8)

    plt.figure(figsize=(8, 6))
    plt.plot(norm_times, label="Normalized Survival Time", color="blue", lw=2)
    plt.plot(sorted_events, label="Event Observed (1=Yes)", color="orange", linestyle="--")
    plt.title("C-Statistic Visual Curve (ROC-style)")
    plt.xlabel("Samples (Sorted by Decreasing Risk)")
    plt.ylabel("Normalized Time / Event")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/MIL_cox_images20_cstat_visual_curve.png")
    plt.close()
    print("Saved C-Statistic ROC-style visual curve to results/plots/MIL_cox_images20_cstat_visual_curve.png")
except Exception as e:
    print("Failed to generate C-Statistic visual curve:", str(e))

print(f"Total train patches: {len(train_dataset)}")
print(f"Total val patches:   {len(val_dataset)}")

end_time = time.time()
elapsed = end_time - start_time
# Print to stdout 
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to file
with open("results/plots/MIL_cox_images20_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")
