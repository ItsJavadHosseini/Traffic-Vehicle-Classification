import os
from pathlib import Path
from collections import Counter
import json


# Root
ROOT = Path("/home/javad/Desktop/Traffic-Vehicle-Classification/data/raw")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def count_class(split_dir: Path):
    """
    count img of classes
    """
    counts = {}
    total = 0
    if not split_dir.exists():
        return counts, total
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        n = sum( 1 for f in cls_dir.iterdir() if is_image)
        counts[cls_dir.name] = n
        total += n
    return counts, total


print(f"ROOT: {ROOT}")
assert ROOT.exists(), "Root not found"

summary = {}
for split in ["train", "test", "unclean"]:
    counts, total = count_class(ROOT / split)
    summary[split] = {"counts": counts, total: total}
    
    print(f"\n{"="*15} {split} {"="*15}")
    for cls, n in counts.items():
        print(f" {cls:15s} : {n:5d}")
    print(f" {"TOTAL":15s} : {total:5d}")
    
Path("reports").mkdir(exist_ok=True)
with open("reports/Class_count.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\n Saved: Reports/Class_counts.json")


train_class = set(summary["train"]["counts"])
test_classes = set(summary["test"]["counts"])
unclean_clases = set(summary["unclean"]["counts"])

print(f" train ({len(train_class)}): {sorted(train_class)}")
print(f" test ({len(test_classes)}): {sorted(test_classes)}")
print(f"unclean ({len(unclean_clases)}): {sorted(unclean_clases)}")

if train_class == test_classes:
    print("/n test and train is iqual")
missing = train_class - unclean_clases
if missing:
    print(f"missing classes: {sorted(missing)}")