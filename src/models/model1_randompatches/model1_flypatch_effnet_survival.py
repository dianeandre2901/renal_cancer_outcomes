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
from skimage.filters import threshold_otsu
from pycox.models.loss import CoxPHLoss
from lifelines.utils import concordance_index

# 1. Data loading
df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_clean.csv")
df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_clean.csv")
df_train['slide_id'] = df_train['slide_id'].astype(str)
df_val['slide_id'] = df_val['slide_id'].astype(str)

# 2. Patch extraction helpers
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

class WSI_PatientDataset(Dataset):
    def __init__(self, df, patch_size=256, n_patches=8, transform=None, debug_save=False):
        self.df = df.reset_index(drop=True)
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.transform = transform
        self.debug_save = debug_save

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_path = row['slide_path']
        time = float(row['os_days'])
        event = int(row['event'])
        slide_id = row['slide_id']
        slide = openslide.OpenSlide(slide_path)
        coords = get_tissue_coords(slide, patch_size=self.patch_size)
        n_found = 0
        patches = []
        used_coords = []
        tries = 0
        while len(patches) < self.n_patches and tries < self.n_patches * 10:
            X, Y = coords[np.random.randint(len(coords))]
            patch = slide.read_region((X, Y), 0, (self.patch_size, self.patch_size)).convert("RGB")
            used_coords.append((X, Y))
            if self.transform:
                patch = self.transform(patch)
            patches.append(patch)
            n_found += 1
            tries += 1
        # Pad if needed
        if len(patches) < self.n_patches:
            pad_count = self.n_patches - len(patches)
            patches.extend([patches[-1]] * pad_count)
        elif len(patches) > self.n_patches:
            patches = patches[:self.n_patches]

        # Debug: save first patch per slide for visual inspection
        if self.debug_save and idx < 10:
            patch_np = patches[0].detach().cpu().numpy().transpose(1,2,0)
            patch_np = np.clip((patch_np * 255), 0, 255).astype(np.uint8)
            im = Image.fromarray(patch_np)
            im.save(f"debug_patch_{slide_id}.png")

        patches = torch.stack(patches)
        return patches, torch.tensor(time, dtype=torch.float32), torch.tensor(event, dtype=torch.float32), slide_id

# 3. Augmentation & Loader
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.05),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
n_patches = 8
batch_size = 1  # for stability

train_ds = WSI_PatientDataset(df_train, patch_size=256, n_patches=n_patches, transform=transform, debug_save=True)
val_ds   = WSI_PatientDataset(df_val,   patch_size=256, n_patches=n_patches, transform=transform, debug_save=False)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=1)

# 4. Model
class DeepSurvWSI(nn.Module):
    def __init__(self, base_model=None):
        super().__init__()
        if base_model is None:
            base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(base_model.classifier[1].in_features, 1)
    def forward(self, x):  # x: [N, 3, 224, 224]
        feats = self.features(x)
        pooled = self.pool(feats).view(x.shape[0], -1)
        risk = self.fc(pooled)
        return risk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepSurvWSI()
for param in model.fc.parameters():
    param.requires_grad = True
for param in model.features[7].parameters():
    param.requires_grad = True
model = model.to(device)
criterion = CoxPHLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 5. Train/Val Loop (patches -> mean risk per slide)
def train_epoch(model, loader, crit, opt, dev):
    model.train()
    total_loss, n = 0.0, 0
    all_risks, all_times, all_events = [], [], []
    for patches, time, event, _ in loader:
        batch_size, n_patches, C, H, W = patches.shape
        patches = patches.view(batch_size * n_patches, C, H, W).to(dev)
        time = time.to(dev)
        event = event.to(dev)
        opt.zero_grad()
        risk_patch = model(patches)
        risk_patch = risk_patch.view(batch_size, n_patches, -1)
        risk_slide = risk_patch.mean(1).squeeze(1)  # [batch_size]
        all_risks.append(risk_slide)
        all_times.append(time)
        all_events.append(event)
    all_risks = torch.cat(all_risks)
    all_times = torch.cat(all_times)
    all_events = torch.cat(all_events)
    if all_events.sum() == 0:
        print("All censored in this epoch, skipping loss.")
        return np.nan
    loss = crit(all_risks, all_times, all_events)
    if torch.isnan(loss) or torch.isinf(loss):
        print("NaN/Inf in epoch loss, skipping epoch.")
        return np.nan
    loss.backward()
    opt.step()
    return loss.item()

@torch.no_grad()
def eval_epoch(model, loader, crit, dev):
    model.eval()
    total_loss, n = 0.0, 0
    risks, times, events, slide_ids = [], [], [], []
    for patches, time, event, slide_id in loader:
        batch_size, n_patches, C, H, W = patches.shape
        patches = patches.view(batch_size * n_patches, C, H, W).to(dev)
        time = time.cpu()
        event = event.cpu()
        risk_patch = model(patches)
        risk_patch = risk_patch.view(batch_size, n_patches, -1)
        risk_slide = risk_patch.mean(1).squeeze(1).cpu().numpy()
        loss = crit(torch.tensor(risk_slide), time, event)
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN/Inf in val loss, skipping batch.")
            continue
        total_loss += loss.item()
        n += batch_size
        risks.append(risk_slide)
        times.append(time.numpy())
        events.append(event.numpy())
        slide_ids.extend([s for s in slide_id])
    if n == 0:
        return np.nan, np.array([]), np.array([]), np.array([]), np.array([])
    risks = np.concatenate(risks)
    times = np.concatenate(times)
    events = np.concatenate(events)
    return total_loss / n, risks, times, events, np.array(slide_ids)

# 6. Training
history = {"train_loss": [], "val_loss": [], "slide_c_index": []}
num_epochs = 10
for epoch in range(num_epochs):
    tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, risks, times, events, slide_ids = eval_epoch(model, val_loader, criterion, device)
    if risks.size > 0:
        slide_c_index = concordance_index(-risks, times, events)
    else:
        slide_c_index = np.nan
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["slide_c_index"].append(slide_c_index)
    print(f"Epoch {epoch+1:02d} | Train loss {tr_loss:.4f} | Val loss {val_loss:.4f} | Slide CI {slide_c_index:.3f}")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history["train_loss"], label="Train")
plt.plot(history["val_loss"],   label="Val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")
plt.subplot(1,2,2)
plt.plot(history["slide_c_index"], label="Slide")
plt.xlabel("Epoch"); plt.ylabel("C-index"); plt.legend(); plt.title("C-index")
plt.tight_layout()
plt.savefig("train_val_surv_metrics.png", dpi=200)
plt.close()