import os
import torch
import pandas as pd
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import openslide
from tqdm import tqdm


# ==== CONFIG ====
PATCH_COORDS_CSV = "/rds/general/user/dla24/home/thesis/src/scripts/results/patch_coords_train.csv"  # or patch_coords_val.csv
OUTPUT_DIR = "features_train"  # or features_val
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
PATCH_SIZE = 224

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== MODEL ====
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove FC
resnet.eval().to(DEVICE)

# ==== DATASET ====
class SlidePatchDataset(Dataset):
    def __init__(self, patch_df, transform=None):
        self.patch_df = patch_df
        self.transform = transform

    def __len__(self):
        return len(self.patch_df)

    def __getitem__(self, idx):
        row = self.patch_df.iloc[idx]
        slide_path = row["slide_path"]
        x, y, patch_px = int(row["x"]), int(row["y"]), int(row["patch_px"])
        slide = openslide.OpenSlide(slide_path)
        patch = slide.read_region((x, y), 0, (patch_px, patch_px)).convert("RGB")
        patch = patch.resize((PATCH_SIZE, PATCH_SIZE), resample=Image.BILINEAR)
        if self.transform:
            patch = self.transform(patch)
        return patch

# ==== FEATURE EXTRACTION ====
def extract_features_for_slide(slide_id, slide_df, output_dir):
    dataset = SlidePatchDataset(slide_df, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    features = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            feats = resnet(batch).squeeze(-1).squeeze(-1).cpu().numpy()
            features.append(feats)

    features = np.concatenate(features, axis=0)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{slide_id}.npy"), features)

# ==== MAIN LOOP ====
def main():
    df = pd.read_csv(PATCH_COORDS_CSV)
    slide_ids = df["slide_id"].unique()

    for slide_id in tqdm(slide_ids, desc="Extracting features"):
        slide_df = df[df["slide_id"] == slide_id].reset_index(drop=True)
        extract_features_for_slide(slide_id, slide_df, OUTPUT_DIR)

if __name__ == "__main__":
    main()
