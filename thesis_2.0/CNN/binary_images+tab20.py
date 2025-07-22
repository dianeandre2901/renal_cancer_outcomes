"""
 Patch-Level Binary Classifier with tissue coordinates pre computed - only images passed to model 
20 slides for training - 100 for validation
This model trains an EfficientNet-B0 to classify individual WSI patches as Alive/Dead,
based on precomputed tissue coordinates. Patches inherit slide-level labels. At evaluation,
patch predictions are aggregated (mean probability) to derive a slide-level prediction.
Includes patch- and slide-level accuracy, confusion matrices, and loss/accuracy curves.
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
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import make_interp_spline
from collections import Counter
import time
# 
start_time = time.time()



# Load data
df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")
df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
df_train['slide_id'] = df_train['slide_id'].astype(str)
df_train = df_train.drop(columns=["event"])
df_val = df_val.drop(columns=["event"])
tabular_features = ["age_at_diagnosis_years", "tumour_grade", "tumour_stage"]

class MultiModalPatchDataset(Dataset):
    def __init__(self, patch_df, transform=None):
        self.df = patch_df.reset_index(drop=True)
        self.transform = transform

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

        label = row["label"]
        slide_id = row["slide_id"]
        tabular = torch.tensor([row[f] for f in tabular_features], dtype=torch.float32)

        return patch, tabular, label, slide_id
    

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

# Take the first 20 unique slides (and all their patches)
first_20_train = train_patches[train_patches['slide_id'].isin(train_patches['slide_id'].unique()[:20])]
first_100_val   = val_patches[val_patches['slide_id'].isin(val_patches['slide_id'].unique()[:100])]
patch_cap = 100 # or None for all

train_dataset = MultiModalPatchDataset(first_20_train, transform=transform, max_patches_per_slide=patch_cap)
val_dataset   = MultiModalPatchDataset(first_100_val, transform=transform, max_patches_per_slide=patch_cap)

#train_dataset = PrecomputedPatchDataset("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv", transform=transform, max_patches_per_slide=patch_cap)
#val_dataset   = PrecomputedPatchDataset("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_val.csv", transform=transform, max_patches_per_slide=patch_cap)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNWithTabular(nn.Module):
    def __init__(self, base_model, tabular_dim):
        super().__init__()
        self.cnn = base_model.features
        self.pool = base_model.avgpool
        self.dropout = nn.Dropout(0.5)
        cnn_out_dim = base_model.classifier[1].in_features
        self.tabular_fc = nn.Linear(tabular_dim, 32)
        self.classifier = nn.Linear(cnn_out_dim + 32, 2)

    def forward(self, img, tabular):
        x = self.cnn(img)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        tab = torch.relu(self.tabular_fc(tabular))
        x = torch.cat([x, tab], dim=1)
        x = self.dropout(x)
        return self.classifier(x)

base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model = CNNWithTabular(base_model).to(device)
# Freeze all blocks except 6 and 7
for name, param in model.features.named_parameters():
    if "6" in name or "7" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

        
train_patch_labels = train_dataset.df['label'].to_numpy()
counts = np.bincount(train_patch_labels)
if len(counts) < 2:
    counts = np.pad(counts, (0, 2-len(counts)), constant_values=0)
class_weights = torch.tensor([1.0 / (counts[i] + 1e-6) for i in range(2)], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
print(f"Class weights (Alive, Dead): {class_weights.cpu().numpy()}")


optimizer = torch.optim.Adam(model.parameters(), lr=2.4844087551934078e-05, weight_decay=0.00011136085331748044 )
# Early stopping
best_val_acc = 0
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

print_patch_summary(train_dataset, "Train")
print_patch_summary(val_dataset, "Val")

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
for epoch in range(epochs):
    # --- TRAIN ---
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, tabular, labels, slide_ids in train_loader:
        imgs, tabular, labels = imgs.to(device), tabular.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, tabular)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total if total > 0 else float("nan")
    train_acc = correct / total if total > 0 else float("nan")

    # --- VALIDATION & AGGREGATION ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_val_probs = []
    all_val_labels = []
    all_val_slideids = []
    with torch.no_grad():
        for imgs, labels, tabular, slide_ids in val_loader:
            imgs, labels, tabular = imgs.to(device), labels.to(device), tabular.to(device)
            outputs = model(imgs, tabular)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_val_probs.extend(probs)
            all_val_labels.extend(labels.cpu().numpy())
            all_val_slideids.extend(slide_ids)
    val_loss = val_loss / val_total if val_total > 0 else float("nan")
    val_acc = val_correct / val_total if val_total > 0 else float("nan")
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    # --- PATCH-LEVEL CONFUSION MATRIX ---
    patch_preds = [1 if p > 0.5 else 0 for p in all_val_probs]
    patch_cm = confusion_matrix(all_val_labels, patch_preds)
    print(f"Patch-level Confusion Matrix:\n{patch_cm}")

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
    slide_acc = np.mean(np.array(slide_preds) == np.array(slide_trues))
    print(f"Slide-level accuracy: {slide_acc:.3f}")

    # --- SLIDE-LEVEL CONFUSION MATRIX & PLOT ---
    slide_cm = confusion_matrix(slide_trues, slide_preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(slide_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred_Alive", "Pred_Dead"],
                yticklabels=["True_Alive", "True_Dead"], ax=ax)
    plt.title(f"Slide-level Confusion Matrix (Epoch {epoch + 1})")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"results/plots/CNN_binary_images20+tab_slide_confusion_matrix_epoch{epoch + 1}.png")
    plt.close(fig)
    print(f"Saved confusion matrix as results/plots/CNN_binary_images+tab20_slide_confusion_matrix_epoch{epoch + 1}.png/pdf")
    print(f"Slide-level Confusion Matrix:\n{slide_cm}")
    print("Slide-level classification report:")
    print(classification_report(slide_trues, slide_preds, target_names=["Alive", "Dead"]))

    # --- ROC Curve (Slide-level) ---
    # Get mean probs per slide (same as in slide_pred_classes)
    slide_mean_probs = [np.mean(slide_probs[sid]) for sid in slide_trues]

    # Compute ROC
    fpr, tpr, _ = roc_curve(slide_trues, slide_mean_probs)
    roc_auc = auc(fpr, tpr)

    # Smooth ROC curve
    fpr_unique, idx_unique = np.unique(fpr, return_index=True)
    tpr_unique = tpr[idx_unique]
    fpr_smooth = np.linspace(0, 1, 200)
    tpr_spline = make_interp_spline(fpr_unique, tpr_unique, k=2)
    tpr_smooth = tpr_spline(fpr_smooth)

    plt.figure()
    plt.plot(fpr_smooth, tpr_smooth, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Slide-level ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"results/plots/CNN_binary_images+tab20_slide_ROC_epoch{epoch + 1}.png")
    plt.close()
    print(f"Saved ROC curve as results/plots/CNN_binary_images+tab20_slide_ROC_epoch{epoch + 1}.png")

    print(f"Epoch {epoch + 1}/{epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
          f"Slide Acc: {slide_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_since_improvement = 0
        best_model_state = model.state_dict()
        torch.save(model.state_dict(), "model_CNN_binary_images+tab20.pth")
    else:
        epochs_since_improvement += 1
    if epochs_since_improvement >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

epochs_range = range(1, len(train_loss_list) + 1)
plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_loss_list, label="Train Loss", marker='o', color='red')
plt.plot(epochs_range, val_loss_list, label="Val Loss", marker='o', color='orange')
plt.plot(epochs_range, train_acc_list, label="Train Acc", marker='s', color='blue')
plt.plot(epochs_range, val_acc_list, label="Val Acc", marker='s', color='green')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Train/Val Loss and Accuracy per Epoch")
plt.legend(loc="center right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/CNN_binary_images+tab20_train_val_loss_acc.png")
plt.close()
print("Saved train/val loss & acc plot to results/plots/CNN_binary_images20_train_val_loss_acc.png")


# --- Separate Loss Plot ---
plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_loss_list, label="Train Loss", marker='o', color='red')
plt.plot(epochs_range, val_loss_list, label="Val Loss", marker='o', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss per Epoch")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/CNN_binary_images+tab20_train_val_loss_separate.png")
plt.close()
print("Saved loss calibration plot to results/plots/CNN_binary_images+tab20_train_val_loss_separate.png")

# --- Separate Accuracy Plot ---
plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_acc_list, label="Train Accuracy", marker='s', color='blue')
plt.plot(epochs_range, val_acc_list, label="Val Accuracy", marker='s', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy per Epoch")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/CNN_binary_images+tab20_train_val_acc_separate.png")
plt.close()
print("Saved accuracy calibration plot to results/plots/CNN_binary_images+tab20_train_val_acc_separate.png")

print(f"Total train patches: {len(train_dataset)}")
print(f"Total val patches:   {len(val_dataset)}")

end_time = time.time()
elapsed = end_time - start_time
# Print to stdout 
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to file
with open("results/plots/CNN_binary_images+tab20_runtime_aftertuning.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")


