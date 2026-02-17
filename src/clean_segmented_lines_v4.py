import os
import cv2
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# -----------------------------
# Helpers
# -----------------------------
def read_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Returns ink mask as 0/1 (1=ink, 0=background)
    """
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (th == 0).astype(np.uint8)  # black pixels -> ink
    return ink


def ruled_line_band_ratio(ink: np.ndarray, band_rows: int = 10) -> float:
    """
    If most ink is concentrated in a thin horizontal band => likely a ruled line crop.
    Returns max_band_sum / total_ink
    """
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
    """
    ink: 0/1 image (1=ink). returns largest CC area in pixels.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((ink * 255).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    return int(areas.max()) if areas.size else 0


def text_row_coverage(ink: np.ndarray, min_row_ink: int = 8) -> float:
    """
    Fraction of rows that contain at least min_row_ink ink pixels.
    Ruled-line-only crops usually have only 1–2 rows with ink.
    """
    row_sum = ink.sum(axis=1)
    rows_with_ink = int((row_sum >= min_row_ink).sum())
    return rows_with_ink / max(1, ink.shape[0])


def should_delete(
    gray: np.ndarray,
    min_dark_pixels: int,
    min_ink_ratio: float,
    rule_band_rows: int,
    rule_band_ratio_thr: float,
    min_lcc_ratio: float,
    min_text_row_coverage: float,
    min_row_ink: int
) -> Tuple[bool, str]:
    """
    Returns (delete?, reason)
    """
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

    # 2) ruled line detector (thin horizontal band dominates ink)
    rb = ruled_line_band_ratio(ink, band_rows=rule_band_rows)
    if rb >= rule_band_ratio_thr:
        return True, f"ruled_band({rb:.3f})"

    # 3) tiny/noise only (largest CC too small)
    lcc = largest_connected_component_area(ink)
    if lcc < int(min_lcc_ratio * area):
        return True, f"small_lcc({lcc})"

    # 4) “no text structure”: very few rows contain ink (common for line-only crops)
    cov = text_row_coverage(ink, min_row_ink=min_row_ink)
    if cov < min_text_row_coverage:
        return True, f"low_row_coverage({cov:.3f})"

    return False, "keep"


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/segmented_lines", help="Folder with Grade6/Grade7/... subfolders")
    ap.add_argument("--move_deleted_to", default="data/deleted_lines", help="Where to move deleted images. Use '' to permanently delete.")
    ap.add_argument("--exts", default=".png,.jpg,.jpeg", help="Comma-separated extensions")

    # Tunable thresholds (good starting values for your crops)
    ap.add_argument("--min_dark_pixels", type=int, default=650, help="Minimum ink pixels to keep")
    ap.add_argument("--min_ink_ratio", type=float, default=0.00025, help="Minimum ink_ratio to keep")

    ap.add_argument("--rule_band_rows", type=int, default=10, help="Band height (rows) to detect ruled line")
    ap.add_argument("--rule_band_ratio", type=float, default=0.70, help="If band contains >= this of ink => delete")

    ap.add_argument("--min_lcc_ratio", type=float, default=0.0010, help="Largest CC must be at least this fraction of image area")

    ap.add_argument("--min_text_row_coverage", type=float, default=0.06, help="Min fraction of rows that contain ink (text-like)")
    ap.add_argument("--min_row_ink", type=int, default=8, help="A row counts as text-row if it has >= this many ink pixels")

    ap.add_argument("--dry_run", action="store_true", help="Do not delete/move, just report")
    ap.add_argument("--print_examples", action="store_true", help="Print some deleted file paths with reasons")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    exts = tuple([e.strip().lower() for e in args.exts.split(",")])

    deleted_root: Optional[Path] = Path(args.move_deleted_to) if args.move_deleted_to.strip() else None
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
            min_text_row_coverage=args.min_text_row_coverage,
            min_row_ink=args.min_row_ink
        )

        if drop:
            deleted += 1
            if len(examples) < 30:
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
