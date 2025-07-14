""""
Model 2 — Tissue-Filtered Patch-Based Binary Classification
This script implements a patch-based deep learning model for binary classification (Alive vs Dead)
on renal cancer WSIs using EfficientNet-B0.
"""

import pandas as pd
import numpy as np 
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn
import openslide
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from collections import defaultdict
from matplotlib import pyplot as plt 
import seaborn as sns
import skimage
from torch.utils.data import Dataset
from PIL import Image
import random
from skimage.filters import threshold_otsu

# --- DATASET ---

df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")
df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
df_train['slide_id'] = df_train['slide_id'].astype(str)
df_val['slide_id'] = df_val['slide_id'].astype(str)
df_train = df_train.drop(columns=["event"])
df_val = df_val.drop(columns=["event"])

class TissueWSIPatchDataset(Dataset):
    def __init__(self, df, area_um=256, out_px=224, n_patches_per_slide=100, transform=None, tissue_downsample=32, random_seed=None):
        self.df = df.reset_index(drop=True)
        self.area_um = area_um
        self.out_px = out_px
        self.n_patches_per_slide = n_patches_per_slide
        self.transform = transform
        self.tissue_downsample = tissue_downsample
        self.random_seed = random_seed

        self.slide_patch_coords = []
        self.slide_labels = []
        self.slide_ids = []
        self.slide_paths = []

        for idx, row in self.df.iterrows():
            slide_path = row["slide_path"]
            mpp = float(row["mpp_x"])
            label = 1 if row.get("vital_status", "Alive") == "Dead" else 0
            slide_id = row.get("slide_id", f"slide_{idx}")
            patch_px = int(round(self.area_um / mpp))
            slide = openslide.OpenSlide(slide_path)
            coords = self.get_tissue_coords(slide, patch_px)
            n_sample = min(len(coords), self.n_patches_per_slide)
            sampled_coords = random.sample(coords, n_sample) if n_sample > 0 else [(0,0)]
            for c in sampled_coords:
                self.slide_patch_coords.append((slide_path, patch_px, c))
                self.slide_labels.append(label)
                self.slide_ids.append(slide_id)
                self.slide_paths.append(slide_path)

    def get_tissue_coords(self, slide, patch_px):
        thumb = slide.get_thumbnail((slide.dimensions[0]//self.tissue_downsample, slide.dimensions[1]//self.tissue_downsample))
        gray = np.array(thumb.convert("L"))
        try:
            otsu_val = threshold_otsu(gray)
        except:
            otsu_val = 220
        mask = gray < otsu_val
        ys, xs = np.where(mask)
        coords = []
        for y, x in zip(ys, xs):
            X = int(x * self.tissue_downsample)
            Y = int(y * self.tissue_downsample)
            if X + patch_px < slide.dimensions[0] and Y + patch_px < slide.dimensions[1]:
                coords.append((X, Y))
        if len(coords) == 0:
            coords = [(0,0)]
        return coords

    def __len__(self):
        return len(self.slide_patch_coords)

    def __getitem__(self, idx):
        slide_path, patch_px, (X, Y) = self.slide_patch_coords[idx]
        label = self.slide_labels[idx]
        slide_id = self.slide_ids[idx]
        slide = openslide.OpenSlide(slide_path)
        patch = slide.read_region((X, Y), 0, (patch_px, patch_px)).convert("RGB")
        patch = patch.resize((self.out_px, self.out_px), resample=Image.BILINEAR)
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = transforms.ToTensor()(patch)
        return patch, label, slide_id

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.05),
    transforms.ToTensor(),
])

train_dataset = TissueWSIPatchDataset(df_train.head(30), area_um=224, out_px=224, n_patches_per_slide=50, transform=transform)
val_dataset   = TissueWSIPatchDataset(df_val.head(30), area_um=224, out_px=224, n_patches_per_slide=50, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

# --- MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001378, weight_decay=1.005e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

best_val_acc = 0.0

# For plots
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(20):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for imgs, labels, slide_ids in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
    train_loss /= train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_probs, all_labels, all_slide_ids = [], [], []
    with torch.no_grad():
        for imgs, labels, slide_ids in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            probs = torch.softmax(outputs, dim=1)[:,1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_slide_ids.extend(slide_ids)
    val_loss /= val_total
    val_acc = val_correct / val_total

    auc_patch = roc_auc_score(all_labels, all_probs)
    patch_preds = [1 if p > 0.5 else 0 for p in all_probs]
    patch_cm = confusion_matrix(all_labels, patch_preds)
    slide_cm = confusion_matrix(list(slide_true_labels.values()), list(slide_pred_classes.values()))

    slide_probs = defaultdict(list)
    slide_labels = {}
    for prob, label, slide_id in zip(all_probs, all_labels, all_slide_ids):
        slide_probs[slide_id].append(prob)
        slide_labels[slide_id] = label
    slide_pred_probs = {sid: np.mean(probs) for sid, probs in slide_probs.items()}
    slide_pred_classes = {sid: int(np.mean(probs) > 0.5) for sid, probs in slide_probs.items()}
    slide_true_labels = {sid: slide_labels[sid] for sid in slide_probs}
    slide_acc = np.mean([slide_pred_classes[sid] == slide_true_labels[sid] for sid in slide_probs])

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch {epoch+1:02d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
          f"Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f} | Patch AUC {auc_patch:.3f} | Slide Acc {slide_acc:.3f}")
    print("Patch-level Confusion Matrix:\n", patch_cm)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

print(classification_report(all_labels, patch_preds, target_names=["Alive", "Dead"]))

# --- PLOT: Training Curves ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Val Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


# --- PLOT: Final Patch-Level Confusion Matrix ---
plt.figure(figsize=(5, 4))
sns.heatmap(patch_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Alive", "Dead"], yticklabels=["Alive", "Dead"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Final Patch-Level Confusion Matrix")
plt.tight_layout()
plt.show()

# --- PLOT: Slide-Level Confusion Matrix ---
slide_true = list(slide_true_labels.values())
slide_pred = list(slide_pred_classes.values())
slide_cm = confusion_matrix(slide_true, slide_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(slide_cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=["Alive", "Dead"], yticklabels=["Alive", "Dead"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Final Slide-Level Confusion Matrix")
plt.tight_layout()
plt.show()