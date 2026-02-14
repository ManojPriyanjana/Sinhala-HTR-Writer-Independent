import json
import shutil
from pathlib import Path

SPLIT_JSON = Path("data/splits/writer_split.json")
SEG_ROOT = Path("data/segmented_lines")
OUT_ROOT = Path("data/final_splits")

def load_split():
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dirs():
    for s in ["train", "val", "test"]:
        (OUT_ROOT / s).mkdir(parents=True, exist_ok=True)

def find_lines_for_writer(writer_id: str):
    # Example line file: G9_0001_L02.png  -> writer_id = G9_0001
    # Search all grades under data/segmented_lines/
    files = []
    for grade_dir in SEG_ROOT.glob("Grade*"):
        if not grade_dir.is_dir():
            continue
        files.extend(sorted(grade_dir.glob(f"{writer_id}_L*.png")))
    return files

def copy_lines(writer_ids, split_name):
    out_dir = OUT_ROOT / split_name
    count = 0
    for wid in writer_ids:
        line_files = find_lines_for_writer(wid)
        if len(line_files) == 0:
            print(f"[WARN] No lines found for {wid}")
            continue

        # Keep grade info by subfolder (optional):
        # We'll detect grade from filename prefix: G6_, G7_, G8_, G9_
        for f in line_files:
            prefix = f.name.split("_")[0]  # G6/G7/G8/G9
            (out_dir / prefix).mkdir(exist_ok=True)
            shutil.copy2(f, out_dir / prefix / f.name)
            count += 1
    print(f"{split_name}: copied {count} line images")

def main():
    ensure_dirs()
    split = load_split()

    train_ids = split["train"]
    val_ids = split["val"]
    test_ids = split["test"]

    print("Writers:", len(train_ids) + len(val_ids) + len(test_ids))
    print("Train/Val/Test:", len(train_ids), len(val_ids), len(test_ids))

    copy_lines(train_ids, "train")
    copy_lines(val_ids, "val")
    copy_lines(test_ids, "test")

    print("Done. Output at:", OUT_ROOT)

if __name__ == "__main__":
    main()
