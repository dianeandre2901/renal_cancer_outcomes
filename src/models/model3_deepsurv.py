"""
Model 3 for Survival: Tissue-Driven Patch Extraction + DeepSurv with Cox Loss

This script uses the Model 3 'TissueWSIPatchDataset' to extract all tissue-covering patches per slide,
and fits a survival model (EfficientNet-B0 + Cox proportional hazards loss).
Outputs patient/slide-level C-index and metrics, and saves PNG plots for training/validation loss, C-index, and confusion matrix.
"""

import os
import openslide
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
from skimage.filters import threshold_otsu
from torchvision import transforms
import pandas as pd
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
from pycox.models.loss import CoxPHLoss
from lifelines.utils import concordance_index
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import time
# 
start_time = time.time()

print("Dataset loading started", flush=True)

# ---- Load slide metadata ----
df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")
df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
df_train['slide_id'] = df_train['slide_id'].astype(str)
df_val['slide_id']   = df_val['slide_id'].astype(str)

df_train = df_train.drop(columns=["vital_status"])
df_val = df_val.drop(columns=["vital_status"])


class TissueWSIPatchDataset(Dataset):
    def __init__(self, df, area_um=256, out_px=224, tissue_downsample=32, tissue_thresh=0.8, transform=None, max_patches_per_slide=None):
        self.df = df.reset_index(drop=True)
        self.area_um = area_um
        self.out_px = out_px
        self.tissue_downsample = tissue_downsample
        self.tissue_thresh = tissue_thresh
        self.transform = transform
        self.max_patches_per_slide = max_patches_per_slide
        self.all_patch_info = []
        for idx, row in self.df.iterrows():
            slide_path = row["slide_path"]
            mpp = float(row["mpp_x"])
            event = int(row.get("event", 0))
            os_days = float(row.get("os_days", 0))
            slide_id = row.get("slide_id", f"slide_{idx}")
            patch_px = int(round(self.area_um / mpp))
            coords = self._find_all_tissue_coords(slide_path, patch_px)
            print(f"{slide_id}: {len(coords)} patches", flush=True)  # printing to see where it fails 
            if self.max_patches_per_slide is not None and len(coords) > self.max_patches_per_slide:
                coords = random.sample(coords, self.max_patches_per_slide)
            for (X, Y) in coords:
                self.all_patch_info.append((slide_path, patch_px, X, Y, event, os_days, slide_id))

    @staticmethod
    def _find_all_tissue_coords(slide_path, patch_px, tissue_downsample=32, tissue_thresh=0.8):
        slide = openslide.OpenSlide(slide_path)
        thumb = slide.get_thumbnail((slide.dimensions[0] // tissue_downsample, slide.dimensions[1] // tissue_downsample))
        gray = np.array(thumb.convert("L"))
        try:
            otsu_val = threshold_otsu(gray)
        except Exception:
            otsu_val = 220
        mask = gray < otsu_val
        H, W = slide.dimensions
        mask_full = np.kron(mask, np.ones((tissue_downsample, tissue_downsample), dtype=bool))
        mask_full = mask_full[:H, :W]
        h, w = mask_full.shape
        coords = []
        for y in range(0, h - patch_px + 1, patch_px):
            for x in range(0, w - patch_px + 1, patch_px):
                patch_mask = mask_full[y:y+patch_px, x:x+patch_px]
                if np.mean(patch_mask) > tissue_thresh:
                    coords.append((x, y))
        if len(coords) == 0:
            coords = [(0,0)]
        return coords

    def __len__(self):
        return len(self.all_patch_info)

    def __getitem__(self, idx):
        slide_path, patch_px, X, Y, event, os_days, slide_id = self.all_patch_info[idx]
        slide = openslide.OpenSlide(slide_path)
        patch = slide.read_region((X, Y), 0, (patch_px, patch_px)).convert("RGB")
        patch = patch.resize((self.out_px, self.out_px), resample=Image.BILINEAR)
        print(f"Loading patch idx: {idx}", flush=True)
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = transforms.ToTensor()(patch)
        return patch, torch.tensor(os_days, dtype=torch.float32), torch.tensor(event, dtype=torch.float32), slide_id

# data aug

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
patch_cap = None


train_dataset = TissueWSIPatchDataset(df_train.head(20), area_um=124, out_px=224, transform=transform, max_patches_per_slide=patch_cap)
val_dataset   = TissueWSIPatchDataset(df_val.head(20),   area_um=124, out_px=224, transform=transform, max_patches_per_slide=patch_cap)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model load
class MyEffNetSurv(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.dropout = nn.Dropout(0.47)
        self.classifier = nn.Linear(base_model.classifier[1].in_features, 1)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x.squeeze(-1)

base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model = MyEffNetSurv(base_model).to(device)
criterion = CoxPHLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

best_val_acc = 0
patience = 2
epochs_since_improvement = 0
best_model_state = None

train_losses = []
val_losses = []
val_cindices = []
epochs = 10


def evaluate_cindex(model, loader, device):
    model.eval()
    all_risks, all_times, all_events = [], [], []
    val_loss, total = 0.0, 0
    with torch.no_grad():
        for imgs, times, events, _ in loader:
            imgs, times, events = imgs.to(device), times.to(device), events.to(device)
            risk = model(imgs)
            loss = criterion(risk, times, events)
            val_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
            all_risks.append(risk.cpu().numpy())
            all_times.append(times.cpu().numpy())
            all_events.append(events.cpu().numpy())
    risks = np.concatenate(all_risks)
    times = np.concatenate(all_times)
    events = np.concatenate(all_events)
    val_loss = val_loss / total if total > 0 else float('nan')
    cidx = concordance_index(-risks, times, events) if np.unique(times).size > 1 else float('nan')
    return val_loss, cidx

def train_deepsurv(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=40, patience=5, device='cpu'):
    best_cidx = -np.inf
    epochs_no_improve = 0
    train_losses, val_losses, val_cindices = [], [], []
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss, n = 0.0, 0
        for imgs, times, events, _ in train_loader:
            imgs, times, events = imgs.to(device), times.to(device), events.to(device)
            optimizer.zero_grad()
            risk = model(imgs)
            loss = criterion(risk, times, events)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss = running_loss / n if n > 0 else float("nan")
        train_losses.append(train_loss)

        val_loss, val_cidx = evaluate_cindex(model, val_loader, device)
        val_losses.append(val_loss)
        val_cindices.append(val_cidx)

        if scheduler:
            scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val C-idx: {val_cidx:.4f}")

        if val_cidx > best_cidx:
            best_cidx = val_cidx
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), "best_deepsurv_model3.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs. Best Val C-idx: {best_cidx:.4f}")
                break

    return train_losses, val_losses, val_cindices


train_losses, val_losses, val_cindices = train_deepsurv(
    model, train_loader, val_loader, criterion, optimizer,
    scheduler=None, epochs=10, patience=3, device=device
)

# Plotting (loss + c-index)
import matplotlib.pyplot as plt
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.plot(epochs_range, val_cindices, label="Val C-index")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("DeepSurv Training Curves")
plt.legend()
plt.tight_layout()
plt.savefig("deepsurv_training_curves.pdf")
plt.close()


print("Done.")
end_time = time.time()
elapsed = end_time - start_time
# Print to stdout (it will appear in your .log)
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to a file for easy access
with open("results/plots/model3_deepsurv_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")