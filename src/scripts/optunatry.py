"""
Optuna Hyperparameter Tuning for Patch-Based Tissue Masked EfficientNet Model (Model 2)
This script uses Optuna to automate hyperparameter search for a weakly supervised classification model 
using WSIs in kidney cancer. The model is based on EfficientNet-B0, trained on tissue-only patches 
extracted from each slide using an Otsu-threshold tissue mask 
Hyperparameters tuned:
    - Learning rate 
    - Batch size
    - Number of patches per slide
    - Patch size in microns 
    - Optimizer type (Adam/SGD)
    - Weight decay

Workflow:
    1. Loads train/val split CSVs (with slide paths and mpp info).
    2. For each trial, creates new dataloaders with specified patch parameters and data augmentations.
    3. Instantiates EfficientNet-B0, replaces classifier head for 2-class output.
    4. Trains the model for 2 quick epochs on the training patches.
    5. Evaluates performance by computing ROC AUC on the validation patches.
    6. Reports the best hyperparameter set found.
"""

import optuna
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import defaultdict
import openslide
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from skimage.filters import threshold_otsu
from torchvision import transforms


# 1. Data loading (use small number for fast tuning!)
df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv").head(30)
df_val   = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv").head(30)
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

        # Precompute all valid tissue patch coords for all slides!
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
            # Instead of all possible, sample up to n_patches_per_slide
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
    



def objective(trial):
    # --- Hyperparameters to search ---
    lr =     trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    n_patches_per_slide = trial.suggest_categorical('n_patches', [32, 50, 75, 100])
    area_um = trial.suggest_categorical('area_um', [128, 224, 256, 384])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.7)


    print("="*50)
    print(f"TRIAL {trial.number} | optimizer_name={optimizer_name} | n_patches_per_slide ={n_patches_per_slide} | area_um={area_um} | batch_size={batch_size} | lr={lr:.2e} | wd={weight_decay:.2e} | dropout={dropout:.2f}")


    # --- Dataset ---
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.ToTensor(),
    ])
    train_dataset = TissueWSIPatchDataset(
        df_train, area_um=area_um, out_px=224,
        n_patches_per_slide=n_patches_per_slide,
        transform=transform)
    val_dataset = TissueWSIPatchDataset(
        df_val, area_um=area_um, out_px=224,
        n_patches_per_slide=n_patches_per_slide,
        transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # --- Model ---
    class MyEffNet(nn.Module):
        def __init__(self, dropout):
            super().__init__()
            base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.features = base.features
            self.avgpool = base.avgpool
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(base.classifier[1].in_features, 2)
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.classifier(x)
            return x
    model = MyEffNet(dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyEffNet(dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    # --- Training Loop (small number of epochs for speed!) ---
    n_epochs = 7
    best_val_acc = 0.0
    for epoch in range(n_epochs):
        # --- Train ---
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
        train_loss = running_loss/total if total > 0 else float("nan")
        train_acc = correct/total if total > 0 else float("nan")
        
        # --- Validation and slide-level aggregation ---
        model.eval()
        all_probs, all_labels, all_slide_ids = [], [], []
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels, slide_ids in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                all_slide_ids.extend(slide_ids)
        val_loss = val_loss/val_total if val_total > 0 else float("nan")
        val_acc = val_correct/val_total if val_total > 0 else float("nan")
        # Patch-level AUC & confusion matrix
        try:
            auc_patch = roc_auc_score(all_labels, all_probs)
        except:
            auc_patch = float('nan')
        patch_preds = [1 if p > 0.5 else 0 for p in all_probs]
        patch_cm = confusion_matrix(all_labels, patch_preds)
        # Slide-level aggregation
        slide_probs = defaultdict(list)
        slide_labels = {}
        for prob, label, slide_id in zip(all_probs, all_labels, all_slide_ids):
            slide_probs[slide_id].append(prob)
            slide_labels[slide_id] = label
        slide_pred_classes = {sid: int(np.mean(probs) > 0.5) for sid, probs in slide_probs.items()}
        slide_true_labels = {sid: slide_labels[sid] for sid in slide_probs}
        slide_acc = np.mean([slide_pred_classes[sid] == slide_true_labels[sid] for sid in slide_probs])
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Slide Acc: {slide_acc:.3f}, Patch AUC: {auc_patch:.3f}")
        print("Patch Confusion Matrix:\n", patch_cm)
        # print warning if always one class
        maj_pred = np.mean(patch_preds)
        if maj_pred < 0.05 or maj_pred > 0.95:
            print("WARNING: Model is predicting almost all one class at patch level!")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    print("-"*40)
    print(f"Trial {trial.number} done: Val Slide Acc = {slide_acc:.4f}")
    return slide_acc  # or val_acc or auc_patch

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)  # increase n_trials for real search!
    print("="*70)
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")