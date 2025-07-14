import os
import openslide
from PIL import Image
import numpy as np
from joblib import Parallel, delayed

slide_dir = "/rds/general/user/dla24/home/thesis/TGCA_dataset/all_slides"
patch_dir = "/rds/general/user/dla24/home/thesis/TGCA_dataset/patches_512"
thumb_dir = "/rds/general/user/dla24/home/thesis/TGCA_dataset/thumbnails"
os.makedirs(patch_dir, exist_ok=True)
os.makedirs(thumb_dir, exist_ok=True)

PATCH_SIZE = 512  # Increased patch size
THUMB_SIZE = 2048

def get_tissue_mask(thumb, threshold=210):
    arr = np.array(thumb)
    return arr < threshold, (arr < threshold).mean()

def save_patch(slide_path, x, y, out_folder):
    slide = openslide.OpenSlide(slide_path)
    patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
    patch.save(os.path.join(out_folder, f"patch_{x}_{y}.png"))

for slidefile in os.listdir(slide_dir):
    if not slidefile.endswith(".svs"):
        continue

    slide_path = os.path.join(slide_dir, slidefile)
    slide_id = slidefile.split(".")[0]
    out_folder = os.path.join(patch_dir, slide_id)
    os.makedirs(out_folder, exist_ok=True)

    try:
        slide = openslide.OpenSlide(slide_path)
        width, height = slide.dimensions
        print(f"{slide_id}: dimensions {width}x{height}")
    except Exception as e:
        print(f"Error opening {slidefile}: {e}")
        continue

    scale = THUMB_SIZE / max(width, height)
    thumb_size = (int(width * scale), int(height * scale))
    thumbnail = slide.get_thumbnail(thumb_size).convert("L")
    thumbnail.save(os.path.join(thumb_dir, f"{slide_id}_thumb.png"))

    tissue_mask, frac_tissue = get_tissue_mask(thumbnail)
    print(f"Detected tissue fraction in thumbnail: {frac_tissue:.2f}")

    if frac_tissue < 0.01:
        print(f"Almost no tissue found for {slide_id}, skipping.")
        continue

    tissue_mask = np.array(Image.fromarray(tissue_mask.astype(np.uint8)).resize(
        (width // PATCH_SIZE, height // PATCH_SIZE), Image.NEAREST))

    coords = [
        (xi * PATCH_SIZE, yi * PATCH_SIZE)
        for xi in range(tissue_mask.shape[0])
        for yi in range(tissue_mask.shape[1])
        if tissue_mask[xi, yi]
    ]

    Parallel(n_jobs=16)(
        delayed(save_patch)(slide_path, x, y, out_folder) for x, y in coords
    )

    print(f"{slide_id}: Saved {len(coords)} tissue patches.")

print("All slides processed.")
