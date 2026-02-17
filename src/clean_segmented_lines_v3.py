import os
import cv2
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List


# -----------------------------
# Helpers
# -----------------------------
def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def binarize_inv_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Returns binary image with foreground (ink/lines) = 255, background = 0
    """
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def remove_long_horizontal_lines(bin_inv: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect long horizontal lines (ruling lines) and remove them.
    Returns: (text_only, detected_hlines)
    """
    # wide, short kernel to catch long horizontal strokes
    k = max(40, int(width))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    hlines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    text_only = cv2.subtract(bin_inv, hlines)
    return text_only, hlines


def count_fg_pixels(bin_img: np.ndarray) -> int:
    return int(np.count_nonzero(bin_img))


def largest_cc_area(bin_img: np.ndarray) -> int:
    """
    bin_img: 0/255 foreground mask. returns largest connected component area (excluding background).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if num_labels <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(areas.max()) if areas.size else 0


def should_delete(
    gray: np.ndarray,
    min_fg_pixels: int,
    min_fg_ratio: float,
    min_lcc_pixels: int,
    hline_kernel_ratio: float,
    keep_if_has_words: bool = True,
) -> Tuple[bool, str]:
    """
    Decide if a crop should be deleted.
    Strategy:
      1) Binarize (foreground = 255)
      2) Remove long horizontal ruling line(s)
      3) Evaluate remaining foreground pixels + largest CC
    """
    h, w = gray.shape[:2]
    if h < 12 or w < 80:
        return True, "too_small"

    bin_inv = binarize_inv_otsu(gray)

    # kernel width based on image width (works across different scans)
    hline_kernel_w = max(60, int(w * hline_kernel_ratio))
    text_only, hlines = remove_long_horizontal_lines(bin_inv, hline_kernel_w)

    fg = count_fg_pixels(text_only)
    area = h * w
    fg_ratio = fg / max(1, area)

    # if almost nothing remains after removing ruling line
    if fg < min_fg_pixels:
        return True, f"low_fg_after_hline({fg})"

    if fg_ratio < min_fg_ratio:
        return True, f"low_fg_ratio_after_hline({fg_ratio:.6f})"

    lcc = largest_cc_area(text_only)
    if lcc < min_lcc_pixels:
        return True, f"small_lcc({lcc})"

    return False, "keep"


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/segmented_lines", help="Folder with Grade6/Grade7/... subfolders")
    ap.add_argument("--move_deleted_to", default="data/deleted_lines_v3", help="Move deleted images here (safer). Use '' to permanently delete.")
    ap.add_argument("--exts", default=".png,.jpg,.jpeg", help="Comma-separated extensions")

    # Tunable thresholds (good starting values for your dataset)
    ap.add_argument("--min_fg_pixels", type=int, default=220, help="Minimum remaining ink pixels (after removing ruling line)")
    ap.add_argument("--min_fg_ratio", type=float, default=0.00008, help="Minimum remaining ink ratio (after removing ruling line)")
    ap.add_argument("--min_lcc_pixels", type=int, default=90, help="Largest CC must be at least this many pixels")
    ap.add_argument("--hline_kernel_ratio", type=float, default=0.25, help="Kernel width ratio vs image width for removing long horizontal lines")

    ap.add_argument("--dry_run", action="store_true", help="Do not delete/move, just report")
    ap.add_argument("--print_examples", action="store_true", help="Print some deleted file paths with reasons")
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
    examples: List[Tuple[str, str]] = []

    files = []
    for root, _, fnames in os.walk(input_dir):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(Path(root) / f)

    for p in files:
        gray = read_gray(str(p))

        drop, reason = should_delete(
            gray,
            min_fg_pixels=args.min_fg_pixels,
            min_fg_ratio=args.min_fg_ratio,
            min_lcc_pixels=args.min_lcc_pixels,
            hline_kernel_ratio=args.hline_kernel_ratio,
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
