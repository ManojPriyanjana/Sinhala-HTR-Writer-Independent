from pathlib import Path
import cv2
import numpy as np

ROOT = Path("data/segmented_lines")
MIN_H, MAX_H = 30, 260        # too small = noise, too tall = merged paragraph
MIN_INK_RATIO = 0.002         # how much text pixels must exist (tune)
MIN_DARK_PIXELS = 250         # absolute minimum ink pixels

deleted = 0
kept = 0

for img_path in ROOT.rglob("*.png"):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    h, w = img.shape[:2]
    if h < MIN_H or h > MAX_H:
        img_path.unlink(missing_ok=True)
        deleted += 1
        continue

    # binarize (ink = 1)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (th > 0).sum()
    ink_ratio = ink / (h * w)

    if ink < MIN_DARK_PIXELS or ink_ratio < MIN_INK_RATIO:
        img_path.unlink(missing_ok=True)
        deleted += 1
    else:
        kept += 1

print("Done")
print("Kept:", kept)
print("Deleted:", deleted)
