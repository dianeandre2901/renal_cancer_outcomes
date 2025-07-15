"""
Model 3 for Survival: Tissue-Driven Patch Extraction + DeepSurv with Cox Loss

This script uses the Model 3 'TissueWSIPatchDataset' to extract all tissue-covering patches per slide,
and fits a survival model (EfficientNet-B0 + Cox proportional hazards loss).
Outputs patient/slide-level C-index and metrics.
"""

import openslide
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
from collections import defaultdict
import time
# 
start_time = time.time()

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
            # --- Survival label (must have event and os_days) ---
            event = int(row.get("event", 0))
            os_days = float(row.get("os_days", 0))
            slide_id = row.get("slide_id", f"slide_{idx}")
            patch_px = int(round(self.area_um / mpp))
            coords = self._find_all_tissue_coords(slide_path, patch_px)
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
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = transforms.ToTensor()(patch)
        return patch, torch.tensor(os_days, dtype=torch.float32), torch.tensor(event, dtype=torch.float32), slide_id

# Data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

patch_cap = None  # Use None to get all tissue patches
train_dataset = TissueWSIPatchDataset(df_train.head(20), area_um=256, out_px=224, transform=transform, max_patches_per_slide=patch_cap)
val_dataset   = TissueWSIPatchDataset(df_val.head(20),   area_um=256, out_px=224, transform=transform, max_patches_per_slide=patch_cap)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyEffNetSurv(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.dropout = nn.Dropout(0.7)
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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training loop
from tqdm import tqdm

train_loss_list = []
val_loss_list = []

epochs = 10
for epoch in range(epochs):
    # --- TRAIN ---
    model.train()
    running_loss, total = 0, 0
    for imgs, times, events, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
        imgs, times, events = imgs.to(device), times.to(device), events.to(device)
        optimizer.zero_grad()
        risk = model(imgs)
        idx = torch.argsort(times, descending=True)
        risk = risk[idx]
        times = times[idx]
        events = events[idx]
        loss = criterion(risk, times, events)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        if torch.sum(events) == 0:
            continue  # skip batch with no events
    train_loss = running_loss / total if total > 0 else float("nan")
    train_loss_list.append(train_loss)
    print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

    # --- EVAL ---
    model.eval()
    val_loss, n = 0, 0
    risks, times, events, slide_ids = [], [], [], []
    with torch.no_grad():
        for imgs, time, event, sid in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]"):
            imgs, time, event = imgs.to(device), time.to(device), event.to(device)
            risk = model(imgs).cpu()
            batch_loss = criterion(risk, time, event)
            val_loss += batch_loss.item() * imgs.size(0)
            n += imgs.size(0)
            risks.append(risk.numpy())
            times.append(time.cpu().numpy())
            events.append(event.cpu().numpy())
            slide_ids.extend(sid)
    val_loss = val_loss / n if n > 0 else float("nan")
    val_loss_list.append(val_loss)
    risks = np.concatenate(risks)
    times = np.concatenate(times)
    events = np.concatenate(events)
    # Slide-level aggregation
    slide_risk = defaultdict(list)
    slide_event = {}
    slide_time = {}
    for r, t, e, sid in zip(risks, times, events, slide_ids):
        slide_risk[sid].append(r)
        slide_event[sid] = e
        slide_time[sid] = t
    slide_mean_risk = np.array([np.mean(slide_risk[sid]) for sid in slide_risk])
    slide_events = np.array([slide_event[sid] for sid in slide_risk])
    slide_times = np.array([slide_time[sid] for sid in slide_risk])

    # Compute C-index at slide-level
    slide_mean_risk = (slide_mean_risk - np.mean(slide_mean_risk)) / np.std(slide_mean_risk)
    cindex = concordance_index(-slide_mean_risk, slide_times, slide_events)
    print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f} | Slide-level C-index: {cindex:.3f}")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plot Loss Curves
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_loss_list, label='Train Loss', marker='o')
plt.plot(range(1, epochs+1), val_loss_list, label='Val Loss', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/model3_deepsurv_loss_curves.png")
plt.close()

# Final Slide-level Confusion Matrix
slide_pred = (slide_mean_risk > 0).astype(int)  # convert to binary: risk > 0 = high-risk
slide_true = slide_events.astype(int)
cm = confusion_matrix(slide_true, slide_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Alive", "Dead"])
disp.plot(cmap='Blues')
plt.title("Final Slide-level Confusion Matrix")
plt.savefig("results/plots/model3_deepsurv_confusion_matrix.png")
plt.close()


print("Done.")


end_time = time.time()
elapsed = end_time - start_time
# Print to stdout 
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to file
with open("results/plots/model3_deepsurv_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")