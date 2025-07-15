import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn
from skimage.filters import threshold_otsu
from pycox.models.loss import CoxPHLoss
from lifelines.utils import concordance_index
import openslide
import random
from PIL import Image
import matplotlib.pyplot as plt

# --- Data loading ---
df_train = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv")
df_val = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")
df_train['slide_id'] = df_train['slide_id'].astype(str)
df_val['slide_id'] = df_val['slide_id'].astype(str)

df_train= df_train.drop(columns=["vital_status"])
df_val = df_val.drop(columns=["vital_status"])


# --- Dataset class ---
class TissueWSIPatchDataset(Dataset):
    def __init__(self, df, area_um=256, out_px=224, n_patches_per_slide=100, transform=None, tissue_downsample=32, tissue_thresh=0.8, random_seed=None):
        self.df = df.reset_index(drop=True)
        self.area_um = area_um
        self.out_px = out_px
        self.n_patches_per_slide = n_patches_per_slide
        self.transform = transform
        self.tissue_downsample = tissue_downsample
        self.tissue_thresh = tissue_thresh
        self.random_seed = random_seed
        self.slide_patch_coords = []
        self.slide_times = []
        self.slide_events = []
        self.slide_ids = []
        for idx, row in self.df.iterrows():
            slide_path = row["slide_path"]
            mpp = float(row["mpp_x"])
            time = float(row["os_days"])
            event = int(row["event"])
            slide_id = row.get("slide_id", f"slide_{idx}")
            patch_px = int(round(self.area_um / mpp))
            slide = openslide.OpenSlide(slide_path)
            coords = self.get_tissue_coords(slide, patch_px)
            n_sample = min(len(coords), self.n_patches_per_slide)
            sampled_coords = random.sample(coords, n_sample) if n_sample > 0 else [(0,0)]
            for c in sampled_coords:
                self.slide_patch_coords.append((slide_path, patch_px, c))
                self.slide_times.append(time)
                self.slide_events.append(event)
                self.slide_ids.append(slide_id)

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
        time = self.slide_times[idx]
        event = self.slide_events[idx]
        slide_id = self.slide_ids[idx]
        slide = openslide.OpenSlide(slide_path)
        patch = slide.read_region((X, Y), 0, (patch_px, patch_px)).convert("RGB")
        patch = patch.resize((self.out_px, self.out_px), resample=Image.BILINEAR)
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = transforms.ToTensor()(patch)
        return patch, torch.tensor(time, dtype=torch.float32), torch.tensor(event, dtype=torch.float32), slide_id

# --- Transforms ---
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.05),
    transforms.ToTensor(),
])

# --- DataLoaders ---
train_dataset = TissueWSIPatchDataset(df_train.head(5), area_um=256, out_px=224, n_patches_per_slide=100, transform=transform)
val_dataset   = TissueWSIPatchDataset(df_val.head(5),   area_um=256, out_px=224, n_patches_per_slide=50,  transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)  # single output for risk
model = model.to(device)

criterion = CoxPHLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

history = {"train_loss": [], "val_loss": [], "slide_cindex": []}
best_val_cindex = 0.0

for epoch in range(10):
    # TRAIN
    model.train()
    train_loss, train_risk, train_time, train_event, train_slide_ids = 0, [], [], [], []
    for imgs, times, events, slide_ids in train_loader:
        imgs, times, events = imgs.to(device), times.to(device), events.to(device)
        optimizer.zero_grad()
        risk = model(imgs).squeeze(1)  # [batch]
        loss = criterion(risk, times, events)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        train_risk.append(risk.detach().cpu().numpy())
        train_time.append(times.cpu().numpy())
        train_event.append(events.cpu().numpy())
        train_slide_ids.extend(slide_ids)
    train_loss /= len(train_loader.dataset)
    train_risk = np.concatenate(train_risk)
    train_time = np.concatenate(train_time)
    train_event = np.concatenate(train_event)
    train_cindex = concordance_index(-train_risk, train_time, train_event)

    # VALIDATION
    model.eval()
val_loss, n_valid = 0, 0
val_risk, val_time, val_event, val_slide_ids = [], [], [], []
with torch.no_grad():
    for imgs, times, events, slide_ids in val_loader:
        imgs, times, events = imgs.to(device), times.to(device), events.to(device)
        risk = model(imgs).squeeze(1)
        loss = criterion(risk, times, events)
        # Checking for nan
        if (torch.isnan(times).any() or torch.isinf(times).any() or
            torch.isnan(events).any() or torch.isinf(events).any() or
            torch.isnan(risk).any() or torch.isinf(risk).any() or
            torch.isnan(loss) or torch.isinf(loss)):
            print(f"[Val] Skipping batch (NaN/Inf detected). Slide IDs: {slide_ids}")
            continue
        val_loss += loss.item() * imgs.size(0)
        n_valid += imgs.size(0)
        val_risk.append(risk.cpu().numpy())
        val_time.append(times.cpu().numpy())
        val_event.append(events.cpu().numpy())
        val_slide_ids.extend(slide_ids)

val_loss = val_loss / n_valid if n_valid > 0 else float('nan')
if len(val_risk) > 0:
    val_risk = np.concatenate(val_risk)
    val_time = np.concatenate(val_time)
    val_event = np.concatenate(val_event)
    val_cindex = concordance_index(-val_risk, val_time, val_event)
else:
    val_cindex = float('nan')

    scheduler.step(val_loss)
    if val_cindex > best_val_cindex:
        best_val_cindex = val_cindex
        torch.save(model.state_dict(), "best_deepsurv_model.pth")

    print(f"Epoch {epoch+1:02d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
          f"Train CI {train_cindex:.3f} | Val CI {val_cindex:.3f}")

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["slide_cindex"].append(val_cindex)

# --- Plotting ---
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.plot(history["slide_cindex"], label="Val C-index")
plt.legend()
plt.show()