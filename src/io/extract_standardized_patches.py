import openslide
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from skimage.filters import threshold_otsu
from torchvision import transforms

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
    

