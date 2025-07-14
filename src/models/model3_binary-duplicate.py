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



# ---- Load slide metadata ----
df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")
df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
df_train['slide_id'] = df_train['slide_id'].astype(str)
df_train = df_train.drop(columns=["event"])
df_val = df_val.drop(columns=["event"])

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


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

patch_cap = None # or None for all

train_dataset = TissueWSIPatchDataset(df_train.head(20), area_um=128, out_px=224, transform=transform, max_patches_per_slide=patch_cap)
val_dataset   = TissueWSIPatchDataset(df_val.head(20),   area_um=128, out_px=224, transform=transform, max_patches_per_slide=patch_cap)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

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
total = sum(counts)
weights = [total / (2 * c) for c in counts]

# Normalize to have mean=1
weights = np.array(weights)
weights = weights / np.mean(weights)

class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
print(f"Class weights (Alive, Dead): {class_weights.cpu().numpy()}")


optimizer = torch.optim.Adam(model.parameters(), lr=4.09e-05, weight_decay=2.26e-06 )
# Early stopping
best_val_acc = 0
patience = 2
epochs_since_improvement = 0
best_model_state = None

epochs = 10

def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else float("nan")

def preds_labels(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10,
                save_model_path='best_model3binary.pt', early_stopping=True, patience=2, device='cpu'):
    model.to(device)
    best_acc, epochs_no_improve = 0, 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total if total > 0 else float("nan")
        val_acc = evaluate_model(model, val_loader, device)

        train_losses.append(running_loss / total if total > 0 else float("nan"))
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Validation loss
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * images.size(0)
                val_total += labels.size(0)
        val_losses.append(val_loss / val_total if val_total > 0 else float("nan"))

        if scheduler:
            scheduler.step(val_losses[-1])

        print(f"Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_model_path)
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs. Best Val Acc: {best_acc:.4f}")
                break

    return train_losses, val_losses, train_accs, val_accs

#Training 

train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, criterion, optimizer,
    scheduler=None, epochs=10, save_model_path="best_model3binary.pt",
    early_stopping=True, patience=2, device=device
)

# Evaluation & CM
all_preds, all_labels, all_probs = preds_labels(model, val_loader, device)
cm = confusion_matrix(all_labels, all_preds)
print("Validation Confusion Matrix:\n", cm)
print(classification_report(all_labels, all_preds, target_names=["Alive", "Dead"], zero_division=0))

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Alive", "Dead"], yticklabels=["Alive", "Dead"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Validation Confusion Matrix")
plt.tight_layout()
plt.savefig("results/plots/model3_val_confmat.png")
plt.close()

#  Training Progress Plot 
def plot_training_progress(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, train_losses, label='Train Loss', color='red', marker='o')
    ax1.plot(epochs, val_losses, label='Val Loss', color='orange', marker='o')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(epochs, train_accs, label='Train Acc', color='blue', marker='s')
    ax2.plot(epochs, val_accs, label='Val Acc', color='green', marker='s')
    fig.legend(loc='upper right')
    plt.title('Training and Validation Loss/Accuracy')
    plt.tight_layout()
    plt.savefig("results/plots/model3_train_val_progress.png")
    plt.close()

plot_training_progress(train_losses, val_losses, train_accs, val_accs)
print("Saved all plots to results/plots/")

end_time = time.time()
elapsed = end_time - start_time
# Print to stdout
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to file
with open("results/plots/model3_binaryduplic_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")