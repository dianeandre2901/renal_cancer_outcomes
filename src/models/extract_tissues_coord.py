import pandas as pd
import numpy as np
import openslide
import os
from skimage.filters import threshold_otsu
import time
# 
start_time = time.time()


def extract_patch_indices(slide_path, mpp, area_um=256, tissue_downsample=32, tissue_thresh=0.8):
    patch_px = int(round(area_um / mpp))
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
                coords.append((x, y, patch_px))
    return coords


for split, path in [("train", "/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv"),
                    ("val",   "/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv")]:
    df = pd.read_csv(path)
    df['slide_id'] = df['slide_id'].astype(str)
    out_rows = []
    for _, row in df.iterrows():
        slide_path = row["slide_path"]
        mpp = float(row["mpp_x"])
        label = 1 if row.get("vital_status", "Alive") == "Dead" else 0
        slide_id = row["slide_id"]
        coords = extract_patch_indices(slide_path, mpp)
        for x, y, patch_px in coords:
            out_rows.append({
                "slide_path": slide_path,
                "slide_id": slide_id,
                "x": x,
                "y": y,
                "patch_px": patch_px,
                "label": label
            })
    pd.DataFrame(out_rows).to_csv(f"results/patch_coords_{split}.csv", index=False)
    print(f"Saved patch coords for {split} to results/patch_coords_{split}.csv")




print("Done.")
end_time = time.time()
elapsed = end_time - start_time
# Print to stdout
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to file
with open("results/plots/extractpatchescoords.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")