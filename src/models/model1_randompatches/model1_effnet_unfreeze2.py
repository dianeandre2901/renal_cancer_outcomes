import os
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np
import openslide
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from skimage.filters import threshold_otsu

# load Data
df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_clean.csv")
df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_clean.csv")
df_train = df_train.drop(columns = "event")
df_val = df_val.drop(columns = "event")
df_train['slide_id'] = df_train['slide_id'].astype(str)
df_val['slide_id'] = df_val['slide_id'].astype(str)

# Define function to extract tissues coordinates regions from slides

def get_tissue_coords(slide, patch_size=256, downsample=32, threshold=220):
    thumb = slide.get_thumbnail((slide.dimensions[0]//downsample, slide.dimensions[1]//downsample))
    gray = np.array(thumb.convert("L"))
    try:
        otsu_val = threshold_otsu(gray)
    except:
        otsu_val = threshold
    mask = gray < otsu_val
    ys, xs = np.where(mask)
    coords = []
    for y, x in zip(ys, xs):
        X = int(x * downsample)
        Y = int(y * downsample)
        if X + patch_size < slide.dimensions[0] and Y + patch_size < slide.dimensions[1]:
            coords.append((X, Y))
    if len(coords) == 0:
        coords = [(0,0)]
    return coords

class OnTheFlyPatchDataset(Dataset):
    def __init__(self, df, patch_size=256, n_patches_per_slide=8, transform=None):
        self.df = df.reset_index(drop=True)
        self.patch_size = patch_size
        self.n_patches_per_slide = n_patches_per_slide
        self.transform = transform

    def __len__(self):
        return len(self.df) * self.n_patches_per_slide

    def __getitem__(self, idx):
        slide_idx = idx // self.n_patches_per_slide
        row = self.df.iloc[slide_idx]
        slide_path = row['slide_path']
        label = 1 if row['vital_status'] == "Dead" else 0
        slide_id = row['slide_id']
        slide = openslide.OpenSlide(slide_path)
        coords = get_tissue_coords(slide, patch_size=self.patch_size)
        X, Y = coords[np.random.randint(len(coords))]
        patch = slide.read_region((X, Y), 0, (self.patch_size, self.patch_size)).convert("RGB")
        if self.transform:
            patch = self.transform(patch)
        return patch, label, slide_id

#  Data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# DataLoaders 
n_patches_per_slide = 8
train_ds = OnTheFlyPatchDataset(df_train, patch_size=256, n_patches_per_slide=n_patches_per_slide, transform=transform)
val_ds = OnTheFlyPatchDataset(df_val, patch_size=256, n_patches_per_slide=n_patches_per_slide, transform=transform)
##train_ds = OnTheFlyPatchDataset(df_train.head(4), patch_size=256, n_patches_per_slide=n_patches_per_slide, transform=transform). #to debug 
#val_ds = OnTheFlyPatchDataset(df_val.head(4), patch_size=256, n_patches_per_slide=n_patches_per_slide, transform=transform)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=8, num_workers=2)

# Model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True
for param in model.features[7].parameters():
    param.requires_grad = True
model = model.to(device)

#  Loss and optimizer
class_weights = compute_class_weight("balanced", classes=np.array([0,1]), y=df_train['vital_status'].map({'Dead': 1, 'Alive': 0}))
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

history = {"train_loss": [], "val_loss": [], "train_slide_acc": [], "val_slide_acc": []}
best_val_acc = 0.0

# training epochs
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels, _ in loader:
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
    return running_loss/total, correct/total

def eval_epoch_slide_agg(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_slide_ids = []
    with torch.no_grad():
        for imgs, labels, slide_ids in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
            preds = np.argmax(outputs.cpu().numpy(), axis=1)
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_slide_ids.extend(slide_ids)
    slide_probs = defaultdict(list)
    slide_labels = {}
    for prob, label, slide_id in zip(all_preds, all_labels, all_slide_ids):
        slide_probs[slide_id].append(prob)
        slide_labels[slide_id] = label
    slide_pred_probs = {sid: np.mean(probs) for sid, probs in slide_probs.items()}
    slide_pred_classes = {sid: int(np.mean(probs) > 0.5) for sid, probs in slide_probs.items()}
    slide_true_labels = {sid: slide_labels[sid] for sid in slide_probs}
    slide_acc = np.mean([slide_pred_classes[sid] == slide_true_labels[sid] for sid in slide_probs])
    return slide_acc, slide_pred_probs, slide_true_labels

num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} started at {time.ctime()}")
    train_loss, _ = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, _ = train_epoch(model, val_loader, criterion, optimizer, device)
    train_slide_acc, _, _ = eval_epoch_slide_agg(model, train_loader, device)
    val_slide_acc, _, _ = eval_epoch_slide_agg(model, val_loader, device)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_slide_acc"].append(train_slide_acc)
    history["val_slide_acc"].append(val_slide_acc)
    print(f"Epoch {epoch+1}/{num_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Train slide acc: {train_slide_acc:.3f} | Val slide acc: {val_slide_acc:.3f}")
    if val_slide_acc > best_val_acc:
        best_val_acc = val_slide_acc
        torch.save(model.state_dict(), "efficientnet_b0_patch_classifier_best.pth")

#  Plot
epochs = range(1, len(history["train_loss"]) + 1)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Patch-level Loss")
plt.subplot(1,2,2)
plt.plot(epochs, history["train_slide_acc"], label="Train Slide Acc")
plt.plot(epochs, history["val_slide_acc"], label="Val Slide Acc")
plt.xlabel("Epochs")
plt.ylabel("Slide-level Accuracy")
plt.legend()
plt.title("Slide-level Accuracy")
plt.tight_layout()
plt.savefig("train_val_slide_metrics.png", dpi=200)
plt.show()