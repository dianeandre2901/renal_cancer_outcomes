print("JOB STARTED", flush=True)
import optuna
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
import openslide
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from skimage.filters import threshold_otsu
from torchvision import transforms
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from collections import defaultdict

def objective(trial):
    # Hyperparameters to tune
    area_um = trial.suggest_categorical("area_um", [128, 192, 256, 384])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.7)
    
    print("="*50)
    print(f"TRIAL {trial.number} | area_um={area_um} | batch_size={batch_size} | lr={lr:.2e} | wd={weight_decay:.2e} | dropout={dropout:.2f}")
    
    # Data
    df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")
    df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
    df_train['slide_id'] = df_train['slide_id'].astype(str)
    df_val['slide_id'] = df_val['slide_id'].astype(str)
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
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    train_dataset = TissueWSIPatchDataset(df_train.head(20), area_um=area_um, out_px=224, transform=transform, max_patches_per_slide=None)
    val_dataset = TissueWSIPatchDataset(df_val.head(20), area_um=area_um, out_px=224, transform=transform, max_patches_per_slide=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train patches: {len(train_dataset)}, Val patches: {len(val_dataset)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
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
    
    best_val_acc = 0.0
    n_epochs = 5 # for quick Optuna, increase for final!
    
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
        slide_preds = list(slide_pred_classes.values())
        print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Slide Acc: {slide_acc:.3f}")
        maj_pred = np.mean(slide_preds)
        if maj_pred < 0.05 or maj_pred > 0.95:
            print("WARNING: Model is predicting almost all one class at slide level!")
        
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