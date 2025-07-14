"""

Model 3: Tissue-Driven Patch Extraction with Slide-Level Aggregation
This script implements a patch-based binary classification pipeline for WSIs (Whole Slide Images),
 make sure the full tissues regions is covered by patches using EfficientNet-B0.
 Key features:
- **Tissue-aware patch extraction:** 
- **Fully automated patch sizing:**  
- **Strong data augmentation:**  
- **Slide-level prediction via patch aggregation:**  
"""
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
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import time
# 
start_time = time.time()

# Load
df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")
df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
df_train['slide_id'] = df_train['slide_id'].astype(str)
df_train = df_train.drop(columns=["event"])
df_val = df_val.drop(columns=["event"])

slide_ids = df_train['slide_id'].unique()

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
            label = 1 if row.get("vital_status", "Alive") == "Dead" else 0
            slide_id = row.get("slide_id", f"slide_{idx}")
            patch_px = int(round(self.area_um / mpp))
            coords = self._find_all_tissue_coords(slide_path, patch_px)
            if self.max_patches_per_slide is not None and len(coords) > self.max_patches_per_slide:
                coords = random.sample(coords, self.max_patches_per_slide)
            for (X, Y) in coords:
                self.all_patch_info.append((slide_path, patch_px, X, Y, label, slide_id))

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
        slide_path, patch_px, X, Y, label, slide_id = self.all_patch_info[idx]
        slide = openslide.OpenSlide(slide_path)
        patch = slide.read_region((X, Y), 0, (patch_px, patch_px)).convert("RGB")
        patch = patch.resize((self.out_px, self.out_px), resample=Image.BILINEAR)
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = transforms.ToTensor()(patch)
        return patch, label, slide_id

# augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


patch_cap = None # or None for all

train_dataset = TissueWSIPatchDataset(df_train.head(2), area_um=128, out_px=224, transform=transform, max_patches_per_slide=patch_cap)
val_dataset   = TissueWSIPatchDataset(df_val.head(2),   area_um=128, out_px=224, transform=transform, max_patches_per_slide=patch_cap)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyEffNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.dropout = nn.Dropout(0.47)
        self.classifier = nn.Linear(base_model.classifier[1].in_features, 2)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model = MyEffNet(base_model).to(device)

# class weights
train_patch_labels = [label for _, _, _, _, label, _ in train_dataset.all_patch_info]
counts = np.bincount(train_patch_labels)
if len(counts) < 2:  # In case only one class is present
    counts = np.pad(counts, (0, 2-len(counts)), constant_values=0)
class_weights = torch.tensor([1.0 / (counts[i] + 1e-6) for i in range(2)], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
print(f"Class weights (Alive, Dead): {class_weights.cpu().numpy()}")


optimizer = torch.optim.Adam(model.parameters(), lr=4.09e-05, weight_decay=2.26e-06 )
# Early stopping
best_val_acc = 0
patience = 2
epochs_since_improvement = 0
best_model_state = None

epochs = 10
slide_count = Counter([sid for *_, sid in train_dataset.all_patch_info])
print("Patch count per slide in train:", slide_count)
print("Patch count per slide in val:", Counter([sid for *_, sid in val_dataset.all_patch_info]))


train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

for epoch in range(epochs):
    # TRAIN 
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total if total > 0 else float("nan")
    train_acc = correct / total if total > 0 else float("nan")

   
    # VALIDATION
    model.eval()
    all_val_probs = []
    all_val_labels = []
    all_val_slideids = []
    val_loss_sum, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels, slide_ids in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()  # Probability "Dead"
            all_val_probs.extend(probs)
            all_val_labels.extend(labels.cpu().numpy())
            all_val_slideids.extend(slide_ids)
            val_loss_sum += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_loss = val_loss_sum / val_total if val_total > 0 else float("nan")
    val_acc = val_correct / val_total if val_total > 0 else float("nan")


    # --- SLIDE-LEVEL AGGREGATION ---
    slide_probs = defaultdict(list)
    slide_labels = {}
    for prob, label, sid in zip(all_val_probs, all_val_labels, all_val_slideids):
        slide_probs[sid].append(prob)
        slide_labels[sid] = label
    slide_pred_classes = {sid: int(np.mean(probs) > 0.5) for sid, probs in slide_probs.items()}
    slide_true_labels = {sid: slide_labels[sid] for sid in slide_probs}
    slide_preds = [slide_pred_classes[sid] for sid in slide_probs]
    slide_trues = [slide_true_labels[sid] for sid in slide_probs]
    slide_acc = np.mean(np.array(slide_preds) == np.array(slide_trues)) if len(slide_trues) > 0 else float("nan")
    print(f"Slide-level accuracy: {slide_acc:.3f}")

    # --- slide level cm ---
    if len(slide_trues) > 0:
        slide_cm = confusion_matrix(slide_trues, slide_preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(slide_cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred_Alive", "Pred_Dead"],
                    yticklabels=["True_Alive", "True_Dead"], ax=ax)
        plt.title(f"Slide-level Confusion Matrix (Epoch {epoch + 1})")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(f"results/plots/model3_slide_confusion_matrix_epoch{epoch + 1}.png")
        plt.savefig(f"results/plots/model3_slide_confusion_matrix_epoch{epoch + 1}.pdf")
        plt.close(fig)
        print(f"Saved confusion matrix as results/plots/model3_slide_confusion_matrix_epoch{epoch + 1}.png/pdf")
        print(f"Slide-level Confusion Matrix:\n{slide_cm}")
        print("Slide-level classification report:")
        print(classification_report(slide_trues, slide_preds, target_names=["Alive", "Dead"]))
    else:
        print("[WARNING] No slide-level predictions available for confusion matrix this epoch.")

    print(f"Epoch {epoch + 1}/{epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
          f"Slide Acc: {slide_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_since_improvement = 0
        best_model_state = model.state_dict()
        torch.save(model.state_dict(), "best_model_overfit.pth")
    else:
        epochs_since_improvement += 1
    if epochs_since_improvement >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break



print("Done.")
end_time = time.time()
elapsed = end_time - start_time
# Print to stdout
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to file
with open("results/plots/model3_deepsurv_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")