from pathlib import Path
import cv2
import numpy as np

ROOT = Path("data/segmented_lines")

# Height filter (tune if needed)
MIN_H, MAX_H = 30, 260

# Ink filters (tune)
MIN_INK_RATIO = 0.0025
MIN_DARK_PIXELS = 350

# Ruled-line-only detector
# if most ink is inside a very small number of rows, it's likely just a horizontal line
RULE_BAND_RATIO = 0.70     # 70% of ink in a thin band
RULE_BAND_ROWS = 8         # thin band height

deleted = 0
kept = 0

for img_path in ROOT.rglob("*.png"):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    h, w = img.shape
    if h < MIN_H or h > MAX_H:
        img_path.unlink(missing_ok=True)
        deleted += 1
        continue

    # Binarize: ink = white (255) in th_inv
    _, th_inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (th_inv > 0).sum()
    ink_ratio = ink / (h * w)

    # 1) Too little ink => delete
    if ink < MIN_DARK_PIXELS or ink_ratio < MIN_INK_RATIO:
        img_path.unlink(missing_ok=True)
        deleted += 1
        continue

    # 2) Ruled-line-only check: ink concentrated in a thin band
    row_sum = (th_inv > 0).sum(axis=1)  # ink pixels per row
    if ink > 0:
        # sliding window: maximum ink inside any RULE_BAND_ROWS band
        window = np.ones(RULE_BAND_ROWS, dtype=np.int32)
        band_ink = np.convolve(row_sum, window, mode="same").max()
        if (band_ink / ink) > RULE_BAND_RATIO:
            img_path.unlink(missing_ok=True)
            deleted += 1
            continue

    kept += 1

print("Done")
print("Kept:", kept)
print("Deleted:", deleted)
