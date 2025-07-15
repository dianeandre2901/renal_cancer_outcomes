import optuna
import torch
import numpy as np
import pandas as pd
import openslide
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
from PIL import Image
from collections import defaultdict

# Load
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


# augmentation
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
first_20_val   = val_patches[val_patches['slide_id'].isin(val_patches['slide_id'].unique()[:20])]
patch_cap = 100 # or None for all

train_dataset = PrecomputedPatchDataset(first_20_train, transform=transform, max_patches_per_slide=patch_cap)
val_dataset   = PrecomputedPatchDataset(first_20_val, transform=transform, max_patches_per_slide=patch_cap)

#train_dataset = PrecomputedPatchDataset("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv", transform=transform, max_patches_per_slide=patch_cap)
#val_dataset   = PrecomputedPatchDataset("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_val.csv", transform=transform, max_patches_per_slide=patch_cap)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # --- Hyperparameters to tune ---
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.7)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    max_patches = trial.suggest_categorical('max_patches_per_slide', [50, 100, 200, None])
    area_um = trial.suggest_categorical('area_um', [128, 192, 224, 256, 384])
    # --- Print trial start info ---
    print("="*50)
    print(f"TRIAL {trial.number} | area_um={area_um} | batch_size={batch_size} | lr={lr:.2e} | wd={weight_decay:.2e} | dropout={dropout:.2f}")
    # --- Data Preparation  ---
    
    train_patches = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv")
    val_patches   = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_val.csv")

    first_n = 20 # first 20 patches ?
    train_df = train_patches[train_patches['slide_id'].isin(train_patches['slide_id'].unique()[:first_n])]
    val_df   = val_patches[val_patches['slide_id'].isin(val_patches['slide_id'].unique()[:first_n])]
    train_dataset = PrecomputedPatchDataset(train_df, transform=transform, max_patches_per_slide=max_patches)
    val_dataset   = PrecomputedPatchDataset(val_df,   transform=transform, max_patches_per_slide=max_patches)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- Model & Optimizer ---
    base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    class MyEffNet(nn.Module):
        def __init__(self, base_model, dropout):
            super().__init__()
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(base_model.classifier[1].in_features, 2)
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.classifier(x)
            return x
    model = MyEffNet(base_model, dropout=dropout).to(device)

    train_patch_labels = train_dataset.df['label'].to_numpy()
    counts = np.bincount(train_patch_labels)
    if len(counts) < 2:
        counts = np.pad(counts, (0, 2-len(counts)), constant_values=0)
    class_weights = torch.tensor([1.0 / (counts[i] + 1e-6) for i in range(2)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop mini version just 3 epochs for speed
    best_val_acc = 0
    for epoch in range(3):
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
        train_acc = correct / total if total > 0 else float("nan")
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_val_probs, all_val_labels, all_val_slideids = [], [], []
        with torch.no_grad():
            for imgs, labels, slide_ids in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
                all_val_probs.extend(probs)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_slideids.extend(slide_ids)
        val_acc = val_correct / val_total if val_total > 0 else float("nan")
        # Slide-level aggregation
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
        if slide_acc > best_val_acc:
            best_val_acc = slide_acc
        print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Slide Acc: {slide_acc:.3f}")
        maj_pred = np.mean(slide_preds)
        if maj_pred < 0.05 or maj_pred > 0.95:
            print("WARNING: Model is predicting almost all one class at slide level!")

    # Report slide-level accuracy as the optimization metric 
    print("-"*40)
    print(f"Trial {trial.number} done: Val Slide Acc = {slide_acc:.4f}")
    print(f"[Optuna Trial] area_um={area_um} | batch_size={batch_size} | lr={lr:.2e} | wd={weight_decay:.2e} | dropout={dropout:.2f} | slide_acc={best_val_acc:.4f}")
    return best_val_acc

# Run the optimization
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)  # Increase for real runs!
    print("="*70)
    print("Best trial:")
    best = study.best_trial
    print(f"Slide-level Acc: {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")