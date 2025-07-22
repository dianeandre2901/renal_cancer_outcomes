"""
Model 4: Patch-Level Classifier with tissue coordinates pre computed

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

# Take the first 20 unique slides (and all their patches)
first_60_train = train_patches[train_patches['slide_id'].isin(train_patches['slide_id'].unique()[:60])]
first_60_val   = val_patches[val_patches['slide_id'].isin(val_patches['slide_id'].unique()[:30])]
patch_cap = 100 # or None for all

train_dataset = PrecomputedPatchDataset(first_60_train, transform=transform, max_patches_per_slide=patch_cap)
val_dataset   = PrecomputedPatchDataset(first_60_val, transform=transform, max_patches_per_slide=patch_cap)

#train_dataset = PrecomputedPatchDataset("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv", transform=transform, max_patches_per_slide=patch_cap)
#val_dataset   = PrecomputedPatchDataset("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_val.csv", transform=transform, max_patches_per_slide=patch_cap)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyEffNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.dropout = nn.Dropout(0.5594)
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

epochs = 10
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

    # --- VALIDATION & AGGREGATION ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_val_probs = []
    all_val_labels = []
    all_val_slideids = []
    with torch.no_grad():
        for imgs, labels, slide_ids in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
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
    plt.savefig(f"results/plots/model4_slide_confusion_matrix_60slides_aftertuning_epoch{epoch + 1}.png")
    plt.savefig(f"results/plots/model4_slide_confusion_matrix_60slides_aftertuning_epoch{epoch + 1}.pdf")
    plt.close(fig)
    print(f"Saved confusion matrix as results/plots/model3_slide_confusion_matrix_60slides_epoch{epoch + 1}.png/pdf")
    print(f"Slide-level Confusion Matrix:\n{slide_cm}")
    print("Slide-level classification report:")
    print(classification_report(slide_trues, slide_preds, target_names=["Alive", "Dead"]))

    print(f"Epoch {epoch + 1}/{epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
          f"Slide Acc: {slide_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_since_improvement = 0
        best_model_state = model.state_dict()
        torch.save(model.state_dict(), "best_model_overfit_60slides_aftertuning.pth")
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
plt.title("Model 4: Train/Val Loss and Accuracy per Epoch")
plt.legend(loc="center right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/model4_train_val_loss_acc_60slides_aftertuning.png")
plt.savefig("results/plots/model4_train_val_loss_acc_60slides_aftertuning.pdf")
plt.close()
print("Saved train/val loss & acc plot to results/plots/model4_train_val_loss_acc_60slides_aftertuning.png/pdf")
print(f"Total train patches: {len(train_dataset)}")
print(f"Total val patches:   {len(val_dataset)}")

end_time = time.time()
elapsed = end_time - start_time
# Print to stdout 
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to file
with open("results/plots/model4_60slides_runtime_aftertuning.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")


