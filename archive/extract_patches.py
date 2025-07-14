import os
import openslide
from PIL import Image
import numpy as np

slide_dir = "/rds/general/user/dla24/home/thesis/TGCA_dataset/all_slides"
patch_dir = "/rds/general/user/dla24/home/thesis/TGCA_dataset/patches_256"
thumb_dir = "/rds/general/user/dla24/home/thesis/TGCA_dataset/thumbnails"
os.makedirs(patch_dir, exist_ok=True)
os.makedirs(thumb_dir, exist_ok=True)

PATCH_SIZE = 256
THUMB_SIZE = 2048  # Thumbnail max side in pixels

def get_tissue_mask(thumb, threshold=220, min_frac=0.05):
    # thumb: PIL Image in grayscale
    arr = np.array(thumb)
    tissue_mask = arr < threshold
    frac_tissue = tissue_mask.mean()
    return tissue_mask, frac_tissue

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

    # 1. Generate thumbnail
    scale = THUMB_SIZE / max(width, height)
    thumb_size = (int(width * scale), int(height * scale))
    thumbnail = slide.get_thumbnail(thumb_size).convert("L")
    thumb_path = os.path.join(thumb_dir, f"{slide_id}_thumb.png")
    thumbnail.save(thumb_path)

    # 2. Tissue detection on thumbnail
    tissue_mask, frac_tissue = get_tissue_mask(thumbnail, threshold=220, min_frac=0.05)
    print(f"  Detected tissue fraction in thumbnail: {frac_tissue:.2f}")

    if frac_tissue < 0.01:
        print(f" Almost no tissue found in thumbnail for {slide_id}. Skipping this slide.")
        continue

    # 3. Upscale tissue mask to WSI size (for patch selection)
    tissue_mask = np.array(Image.fromarray(tissue_mask.astype(np.uint8)).resize((width // PATCH_SIZE, height // PATCH_SIZE), Image.NEAREST))

    # 4. Extract patches only from tissue areas
    count = 0
    for xi in range(tissue_mask.shape[0]):
        for yi in range(tissue_mask.shape[1]):
            if tissue_mask[xi, yi]:
                x = xi * PATCH_SIZE
                y = yi * PATCH_SIZE
                patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                patch.save(os.path.join(out_folder, f"patch_{x}_{y}.png"))
                count += 1
    print(f"{slide_id}: Saved {count} tissue patches.")

print("All slides done.")