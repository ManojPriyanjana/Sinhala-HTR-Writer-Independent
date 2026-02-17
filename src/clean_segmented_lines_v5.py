import os
import cv2
import shutil
import argparse
import numpy as np
from pathlib import Path

# -----------------------------
# Helpers
# -----------------------------
def read_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    # Ink = 1, Background = 0
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (th == 0).astype(np.uint8)
    return ink

def ruled_line_band_ratio(ink: np.ndarray, band_rows: int = 8) -> float:
    total = int(ink.sum())
    if total == 0:
        return 1.0
    row_sum = ink.sum(axis=1).astype(np.int64)
    if ink.shape[0] <= band_rows:
        band_max = int(row_sum.sum())
    else:
        window = np.ones(band_rows, dtype=np.int64)
        conv = np.convolve(row_sum, window, mode="valid")
        band_max = int(conv.max()) if conv.size else int(row_sum.sum())
    return band_max / total

def largest_connected_component_area(ink: np.ndarray) -> int:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((ink * 255).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(areas.max()) if areas.size else 0

def ink_bbox_metrics(ink: np.ndarray):
    ys, xs = np.where(ink > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return (x0, y0, x1, y1, (x1 - x0 + 1), (y1 - y0 + 1))

def should_delete(
    gray: np.ndarray,
    min_dark_pixels: int,
    min_ink_ratio: float,
    rule_band_rows: int,
    rule_band_ratio_thr: float,
    min_lcc_ratio: float,
    # NEW thresholds
    min_bbox_h_ratio: float,
    min_ink_rows_ratio: float,
    min_ink_cols_ratio: float,
) -> tuple[bool, str]:
    h, w = gray.shape[:2]
    if h < 10 or w < 50:
        return True, "too_small"

    ink = binarize_otsu(gray)
    total_ink = int(ink.sum())
    area = h * w
    ink_ratio = total_ink / max(1, area)

    # 1) empty / almost empty
    if total_ink < min_dark_pixels:
        return True, f"low_ink_pixels({total_ink})"
    if ink_ratio < min_ink_ratio:
        return True, f"low_ink_ratio({ink_ratio:.6f})"

    # 2) ruled line detector (thin band dominates ink)
    rb = ruled_line_band_ratio(ink, band_rows=rule_band_rows)
    if rb >= rule_band_ratio_thr:
        return True, f"ruled_band({rb:.3f})"

    # 3) bbox + occupancy tests (catch thin strips with tiny marks)
    bbox = ink_bbox_metrics(ink)
    if bbox is not None:
        x0, y0, x1, y1, bw, bh = bbox
        bh_ratio = bh / max(1, h)
        if bh_ratio < min_bbox_h_ratio:
            return True, f"thin_bbox_h({bh_ratio:.4f})"

    # how many rows/cols contain any ink
    ink_rows = (ink.sum(axis=1) > 0).sum()
    ink_cols = (ink.sum(axis=0) > 0).sum()
    if (ink_rows / max(1, h)) < min_ink_rows_ratio:
        return True, f"few_ink_rows({ink_rows}/{h})"
    if (ink_cols / max(1, w)) < min_ink_cols_ratio:
        return True, f"few_ink_cols({ink_cols}/{w})"

    # 4) largest CC area too small (tiny dust/noise)
    lcc = largest_connected_component_area(ink)
    if lcc < int(min_lcc_ratio * area):
        return True, f"small_lcc({lcc})"

    return False, "keep"

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/segmented_lines", help="Folder with Grade6/Grade7/... subfolders")
    ap.add_argument("--move_deleted_to", default="data/deleted_lines", help="Where to move deleted images (safer). Use '' to permanently delete.")
    ap.add_argument("--exts", default=".png,.jpg,.jpeg", help="Comma-separated extensions")

    # Existing thresholds
    ap.add_argument("--min_dark_pixels", type=int, default=450)
    ap.add_argument("--min_ink_ratio", type=float, default=0.00020)
    ap.add_argument("--rule_band_rows", type=int, default=8)
    ap.add_argument("--rule_band_ratio", type=float, default=0.75)
    ap.add_argument("--min_lcc_ratio", type=float, default=0.0008)

    # NEW thresholds (start values — tune if needed)
    ap.add_argument("--min_bbox_h_ratio", type=float, default=0.06, help="If ink bbox height < this * image height => delete (thin strip)")
    ap.add_argument("--min_ink_rows_ratio", type=float, default=0.05, help="If ink exists in < this fraction of rows => delete")
    ap.add_argument("--min_ink_cols_ratio", type=float, default=0.10, help="If ink exists in < this fraction of cols => delete")

    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--print_examples", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    exts = tuple([e.strip().lower() for e in args.exts.split(",")])

    deleted_root = Path(args.move_deleted_to) if args.move_deleted_to.strip() else None
    if deleted_root and (not args.dry_run):
        deleted_root.mkdir(parents=True, exist_ok=True)

    kept = 0
    deleted = 0
    examples = []

    files = []
    for root, _, fnames in os.walk(input_dir):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(Path(root) / f)

    for p in files:
        gray = read_grayscale(str(p))

        drop, reason = should_delete(
            gray,
            min_dark_pixels=args.min_dark_pixels,
            min_ink_ratio=args.min_ink_ratio,
            rule_band_rows=args.rule_band_rows,
            rule_band_ratio_thr=args.rule_band_ratio,
            min_lcc_ratio=args.min_lcc_ratio,
            min_bbox_h_ratio=args.min_bbox_h_ratio,
            min_ink_rows_ratio=args.min_ink_rows_ratio,
            min_ink_cols_ratio=args.min_ink_cols_ratio,
        )

        if drop:
            deleted += 1
            if len(examples) < 40:
                examples.append((str(p), reason))

            if not args.dry_run:
                if deleted_root:
                    rel = p.relative_to(input_dir)
                    target = deleted_root / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(target))
                else:
                    p.unlink(missing_ok=True)
        else:
            kept += 1

    print("Done.")
    print(f"Kept:    {kept}")
    print(f"Deleted: {deleted}")

    if args.print_examples:
        print("\nExamples of deleted:")
        for fp, rsn in examples:
            print(f"  {fp}  -> {rsn}")

if __name__ == "__main__":
    main()
