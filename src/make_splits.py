import random
from pathlib import Path
import json

SEED = 42
TRAIN = 0.70
VAL = 0.15
TEST = 0.15

RAW_DIR = Path("data/raw_pages")
SPLIT_DIR = Path("data/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

def list_writers():
    writers = []
    for grade_dir in sorted(RAW_DIR.iterdir()):
        if not grade_dir.is_dir():
            continue
        grade = grade_dir.name  # e.g., "Grade9" or "grade9"
        for img in sorted(grade_dir.glob("*.jpg")):
            writer_id = img.stem  # e.g., "G9_0001"
            writers.append({"writer_id": writer_id, "grade": grade, "path": str(img)})
    return writers

def main():
    random.seed(SEED)
    writers = list_writers()

    # Split by writer_id (each image is one writer)
    writer_ids = sorted({w["writer_id"] for w in writers})
    random.shuffle(writer_ids)

    n = len(writer_ids)
    n_train = int(n * TRAIN)
    n_val = int(n * VAL)
    train_ids = set(writer_ids[:n_train])
    val_ids = set(writer_ids[n_train:n_train + n_val])
    test_ids = set(writer_ids[n_train + n_val:])

    split = {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}
    (SPLIT_DIR / "writer_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    print("Total writers:", n)
    print("Train:", len(train_ids), "Val:", len(val_ids), "Test:", len(test_ids))
    print("Saved:", SPLIT_DIR / "writer_split.json")

if __name__ == "__main__":
    main()
