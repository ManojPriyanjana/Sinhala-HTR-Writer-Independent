from pathlib import Path
import cv2
import numpy as np

RAW_DIR = Path("data/raw_pages")
OUT_DIR = Path("data/segmented_lines")
DBG_DIR = Path("results/figures/seg_debug")

OUT_DIR.mkdir(parents=True, exist_ok=True)
DBG_DIR.mkdir(parents=True, exist_ok=True)

# Pilot first
PILOT_PER_GRADE = None      # set None later to run all
PAD_Y = 8
MIN_LINE_H = 18

# ---------- helpers ----------
def auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def remove_long_ruled_lines(bgr):
    """
    Remove long horizontal ruled lines only (usually blue) while keeping handwriting (blue/black).
    Uses binarization -> horizontal morphology -> inpaint on grayscale.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    bin_inv = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 15
    )

    h, w = bin_inv.shape
    klen = max(30, w // 20)  # bigger = detects longer lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    horiz = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

    # keep only very long components (filter out short strokes / handwriting)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    mask = np.zeros_like(horiz)
    min_len = int(0.45 * w)  # must span at least 45% of width to be considered ruled line
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if ww >= min_len and hh <= 6:
            mask[labels == i] = 255

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    cleaned = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
    return cleaned, bin_inv, mask

def deskew_if_needed(gray):
    """
    Light deskew using Hough lines on edges. If angle is small, rotate.
    """
    edges = auto_canny(gray)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=200)
    if lines is None:
        return gray, 0.0

    angles = []
    for rho, theta in lines[:,0]:
        ang = (theta - np.pi/2) * 180/np.pi  # around 0 for horizontal
        if -10 < ang < 10:
            angles.append(ang)

    if len(angles) < 5:
        return gray, 0.0

    angle = float(np.median(angles))
    if abs(angle) < 0.4:
        return gray, angle

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rot, angle

def segment_lines(gray_clean):
    """
    Robust line segmentation:
    - binarize
    - remove small noise
    - use horizontal projection
    - refine band boundaries using connected components
    """
    blur = cv2.GaussianBlur(gray_clean, (3, 3), 0)
    bin_inv = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 15
    )

    # remove tiny noise
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    proj = (bin_inv.sum(axis=1) / 255).astype(np.int32)
    proj_s = np.convolve(proj, np.ones(17)/17, mode="same")

    thresh = max(10, 0.12 * np.max(proj_s))
    text_rows = proj_s > thresh

    bands = []
    in_band = False
    start = 0
    for y, v in enumerate(text_rows):
        if v and not in_band:
            in_band = True
            start = y
        elif not v and in_band:
            in_band = False
            end = y
            if end - start >= MIN_LINE_H:
                bands.append((start, end))
    if in_band:
        end = len(text_rows) - 1
        if end - start >= MIN_LINE_H:
            bands.append((start, end))

    # merge very close bands (split lines)
    merged = []
    for y1, y2 in bands:
        if not merged:
            merged.append([y1, y2]); continue
        py1, py2 = merged[-1]
        if y1 - py2 < 14:
            merged[-1][1] = y2
        else:
            merged.append([y1, y2])

    return bin_inv, [(a, b) for a, b in merged]

def crop_and_save(gray, bands, out_dir, stem):
    h, w = gray.shape
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (y1, y2) in enumerate(bands, start=1):
        yy1 = max(0, y1 - PAD_Y)
        yy2 = min(h, y2 + PAD_Y)
        crop = gray[yy1:yy2, :]
        if crop.shape[0] < MIN_LINE_H:
            continue
        cv2.imwrite(str(out_dir / f"{stem}_L{i:02d}.png"), crop)

def debug_save(stem, grade, gray, bin_inv, mask, line_bin, bands):
    # save debug images for the first few pages
    dbg_base = DBG_DIR / grade
    dbg_base.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(dbg_base / f"{stem}_1_gray_clean.png"), gray)
    cv2.imwrite(str(dbg_base / f"{stem}_2_bin_inv.png"), bin_inv)
    cv2.imwrite(str(dbg_base / f"{stem}_3_ruled_mask.png"), mask)
    cv2.imwrite(str(dbg_base / f"{stem}_4_line_bin.png"), line_bin)

    # overlay bands
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (y1, y2) in bands:
        cv2.rectangle(vis, (0, y1), (vis.shape[1]-1, y2), (0, 255, 0), 2)
    cv2.imwrite(str(dbg_base / f"{stem}_5_bands_overlay.png"), vis)

def main():
    grade_dirs = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    if not grade_dirs:
        raise FileNotFoundError(f"No grade folders found inside {RAW_DIR}")

    for grade_dir in sorted(grade_dirs):
        grade = grade_dir.name
        imgs = sorted(list(grade_dir.glob("*.jpg")) + list(grade_dir.glob("*.png")))

        if PILOT_PER_GRADE is not None:
            imgs = imgs[:PILOT_PER_GRADE]

        print(f"\n[{grade}] pages: {len(imgs)}")

        out_grade = OUT_DIR / grade
        for idx, img_path in enumerate(imgs, start=1):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print("  skip unreadable:", img_path.name)
                continue

            gray_clean, bin_inv, ruled_mask = remove_long_ruled_lines(bgr)
            gray_clean, angle = deskew_if_needed(gray_clean)

            line_bin, bands = segment_lines(gray_clean)
            print(f"  {img_path.name} -> {len(bands)} lines (deskew angle {angle:.2f}°)")

            crop_and_save(gray_clean, bands, out_grade, img_path.stem)

            # save debug images for first 2 pages per grade
            if idx <= 2:
                debug_save(img_path.stem, grade, gray_clean, bin_inv, ruled_mask, line_bin, bands)

    print("\nDone.")
    print("Line crops: data/segmented_lines/")
    print("Debug images: results/figures/seg_debug/")

if __name__ == "__main__":
    main()
