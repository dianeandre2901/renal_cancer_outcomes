"""
CNN Cox Model with images only. 20 training slides - 100 validation slides
This model trains an EfficientNet-B0 to estimate patch-level survival risk scores
from precomputed tissue coordinates. Patches inherit slide-level event labels.
At evaluation, patch-level risk scores are averaged to produce slide-level estimates.
The model is using Cox proportional hazards loss.
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
from collections import defaultdict
from collections import Counter
import time
from lifelines.utils import concordance_index
# 
start_time = time.time()



# Load data
df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")
df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
df_train['slide_id'] = df_train['slide_id'].astype(str)
df_val['slide_id'] = df_val['slide_id'].astype(str)
df_train = df_train[['slide_id', 'os_days', 'event']]
df_val = df_val[['slide_id', 'os_days', 'event']]

class PrecomputedPatchDataset(Dataset):
    def __init__(self, patch_csv, clinical_df, transform=None, max_patches_per_slide=None):
        if isinstance(patch_csv, pd.DataFrame):
           self.df = patch_csv.reset_index(drop=True)
        else:
           self.df = pd.read_csv(patch_csv)
        # Merge clinical data to get os_days and event per patch
        self.df = self.df.merge(clinical_df, on='slide_id', how='left')
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
        slide_id = row["slide_id"]
        time = row["os_days"]
        event = row["event"]
        if self.transform:
            patch = self.transform(patch)
        return patch, slide_id, time, event

#  augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def skip_censored_collate_fn(batch):
    imgs, slide_ids, times, events = zip(*batch)
    events_tensor = torch.tensor(events)
    if (events_tensor == 1).sum() == 0:
        return None  # This batch is bad, will be skipped
    imgs = torch.stack(imgs)
    times_tensor = torch.tensor(times)
    return imgs, slide_ids, times_tensor, events_tensor

# Load and filter CSV
train_patches = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv")
val_patches   = pd.read_csv("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_val.csv")

# Take the first 20 unique slides (and all their patches)
first_20_train = train_patches[train_patches['slide_id'].isin(train_patches['slide_id'].unique()[:20])]
first_100_val   = val_patches[val_patches['slide_id'].isin(val_patches['slide_id'].unique()[:100])]
patch_cap = None # or None for all

train_dataset = PrecomputedPatchDataset(first_20_train, df_train, transform=transform, max_patches_per_slide=patch_cap)
val_dataset   = PrecomputedPatchDataset(first_100_val, df_val, transform=transform, max_patches_per_slide=patch_cap)

#train_dataset = PrecomputedPatchDataset("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv", df_train, transform=transform, max_patches_per_slide=patch_cap)
#val_dataset   = PrecomputedPatchDataset("/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_val.csv", df_val, transform=transform, max_patches_per_slide=patch_cap)
train_loader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=2,
    collate_fn=skip_censored_collate_fn
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=16, 
    shuffle=False, 
    num_workers=2,
    collate_fn=skip_censored_collate_fn
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyEffNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.dropout = nn.Dropout(0.5594110)
        self.classifier = nn.Linear(base_model.classifier[1].in_features, 1)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x).squeeze(-1)

base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model = MyEffNet(base_model).to(device)

# Freeze all layers
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze specific blocks
for name, param in model.features.named_parameters():
    if "6" in name or "7" in name:
        param.requires_grad = True

        
def cox_loss(risk, time, event, eps=1e-8):
    idx = torch.argsort(time, descending=True)
    risk, time, event = risk[idx], time[idx], event[idx]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss = -torch.mean((risk - log_cumsum)[event == 1])
    return loss

optimizer = torch.optim.Adam(model.parameters(), lr=2.4844087551934078e-05, weight_decay=0.00011136085331748)
# Early stopping
best_val_cindex = 0
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

for epoch in range(epochs):
    # --- TRAIN ---
    model.train()
    running_loss = 0
    total = 0
    for batch in train_loader:
        if batch is None:
           continue  # Skip batches with no events
        imgs, slide_ids, times, events = batch
        imgs, times, events = imgs.to(device), times.to(device), events.to(device)
        optimizer.zero_grad()
        risk_scores = model(imgs)
        loss = cox_loss(risk_scores, times.float(), events.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        total += events.size(0)
    train_loss = running_loss / total if total > 0 else float("nan")
    

    # --- VALIDATION & AGGREGATION ---
    model.eval()
    val_loss = 0
    val_total = 0
    all_val_risks = []
    all_val_slideids = []
    all_val_times = []
    all_val_events = []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
               continue  # Skip batches with no events
            imgs, slide_ids, times, events = batch
            imgs, times, events = imgs.to(device), times.to(device), events.to(device)
            risk_scores = model(imgs)
            loss = cox_loss(risk_scores, times.float(), events.float())
            val_loss += loss.item() * imgs.size(0)
            val_total += events.size(0)
            all_val_risks.extend(risk_scores.cpu().numpy())
            all_val_slideids.extend(slide_ids)
            all_val_times.extend(times.cpu().numpy())
            all_val_events.extend(events.cpu().numpy())
    val_loss = val_loss / val_total if val_total > 0 else float("nan")
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

 

    # --- SLIDE-LEVEL AGGREGATION ---
    slide_risks_dict = defaultdict(list)
    slide_times_dict = {}
    slide_events_dict = {}
    for risk, sid, time, event in zip(all_val_risks, all_val_slideids, all_val_times, all_val_events):
        slide_risks_dict[sid].append(risk)
        slide_times_dict[sid] = time
        slide_events_dict[sid] = event
    slide_risks = np.array([np.mean(risks) for risks in slide_risks_dict.values()])
    slide_events = np.array([slide_events_dict[sid] for sid in slide_risks_dict.keys()])
    slide_times = np.array([slide_times_dict[sid] for sid in slide_risks_dict.keys()])
    cindex = concordance_index(-slide_risks, slide_times, slide_events)
    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Validation C-index: {cindex:.4f}")

    if cindex > best_val_cindex:
        best_val_cindex = cindex
        epochs_since_improvement = 0
        best_model_state = model.state_dict()
        torch.save(model.state_dict(), "cox_images20.pth")
    else:
        epochs_since_improvement += 1
    if epochs_since_improvement >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

epochs_range = range(1, len(train_loss_list) + 1)
plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_loss_list, label="Train Loss", marker='o', color='red')
plt.plot(epochs_range, val_loss_list, label="Val Loss", marker='o', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Val Loss per Epoch")
plt.legend(loc="center right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/train_val_loss_acc_cox_images20.pdf")
plt.close()
print("Saved train/val loss & acc plot to results/plots/train_val_loss_cox_images20.png")
print(f"Total train patches: {len(train_dataset)}")
print(f"Total val patches:   {len(val_dataset)}")


# Final evaluation of best model on validation set
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    model.eval()
    all_val_risks = []
    all_val_slideids = []
    all_val_times = []
    all_val_events = []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            imgs, slide_ids, times, events = batch
            imgs, times, events = imgs.to(device), times.to(device), events.to(device)
            risk_scores = model(imgs)
            all_val_risks.extend(risk_scores.cpu().numpy())
            all_val_slideids.extend(slide_ids)
            all_val_times.extend(times.cpu().numpy())
            all_val_events.extend(events.cpu().numpy())
    slide_risks_dict = defaultdict(list)
    slide_times_dict = {}
    slide_events_dict = {}
    for risk, sid, time, event in zip(all_val_risks, all_val_slideids, all_val_times, all_val_events):
        slide_risks_dict[sid].append(risk)
        slide_times_dict[sid] = time
        slide_events_dict[sid] = event
    slide_risks = np.array([np.mean(risks) for risks in slide_risks_dict.values()])
    slide_events = np.array([slide_events_dict[sid] for sid in slide_risks_dict.keys()])
    slide_times = np.array([slide_times_dict[sid] for sid in slide_risks_dict.keys()])
    cindex = concordance_index(-slide_risks, slide_times, slide_events)
    print(f"Validation C-index: {cindex:.4f}")

        # --- C-statistic Visual Curve ---

    # Sort by predicted risk (descending)
    sorted_indices = np.argsort(-slide_risks)
    sorted_times = slide_times[sorted_indices]
    sorted_events = slide_events[sorted_indices]
    sorted_risks = slide_risks[sorted_indices]

    # Normalize time for plotting
    norm_times = (sorted_times - sorted_times.min()) / (sorted_times.max() - sorted_times.min())

    plt.figure(figsize=(8, 6))
    plt.plot(norm_times, label="Normalized Survival Time", color="blue", lw=2)
    plt.plot(sorted_events, label="Event (1=Observed)", color="orange", linestyle="--")
    plt.title(f"C-Statistic Visual Curve (C-index = {cindex:.2f})")
    plt.xlabel("Samples (Sorted by Decreasing Risk)")
    plt.ylabel("Normalized Survival Time / Event")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/cox_images20_cstat_curve.png")
    plt.close()
    print("Saved C-statistic performance curve to results/plots/cox_images20_cstat_curve.png")


end_time = time.time()
elapsed = end_time - start_time
# Print to stdout 
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to file
with open("results/plots/cox_images20_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")
