"""
ZeroDefect AI — Phase 1: Dataset Preparation
Converts Kaggle Casting Product Defect dataset to YOLO format.
Handles both classification-style folders (ok_front/def_front)
and already-split structures.

Run: python phase1_prepare_dataset.py --data_path ./casting_data
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from PIL import Image
import yaml
import json

# ── CONFIG ──────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10
SEED        = 42
IMG_SIZE    = 512   # casting dataset native size

# Defect classes — index 0 = defective casting
# We do binary detection: defect vs ok
CLASS_NAMES = ["defect", "ok"]

# ── HELPERS ─────────────────────────────────────────────────────────────────

def make_dirs(base: Path):
    """Create YOLO directory tree."""
    for split in ["train", "val", "test"]:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created YOLO directory tree at {base}")


def full_image_label(class_id: int) -> str:
    """
    YOLO label where the bounding box = entire image.
    Format: class cx cy w h (all normalised 0-1)
    For a whole-image box: 0.5 0.5 1.0 1.0
    """
    return f"{class_id} 0.500000 0.500000 1.000000 1.000000\n"


def copy_and_label(src_path: Path, dst_img: Path, dst_lbl: Path, class_id: int):
    """Copy image + write matching YOLO label file."""
    shutil.copy2(src_path, dst_img)
    dst_lbl.write_text(full_image_label(class_id))


def split_files(files: list, seed: int = SEED):
    """Return (train, val, test) lists."""
    random.seed(seed)
    random.shuffle(files)
    n = len(files)
    t1 = int(n * TRAIN_RATIO)
    t2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    return files[:t1], files[t1:t2], files[t2:]


def write_yaml(out_dir: Path, dataset_root: Path):
    """Write data.yaml for YOLOv8 training."""
    cfg = {
        "path"  : str(dataset_root.resolve()),
        "train" : "images/train",
        "val"   : "images/val",
        "test"  : "images/test",
        "nc"    : len(CLASS_NAMES),
        "names" : CLASS_NAMES,
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[OK] Wrote {yaml_path}")
    return yaml_path


def write_stats(out_dir: Path, stats: dict):
    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Stats → {stats_path}")
    print(json.dumps(stats, indent=2))


# ── MAIN ─────────────────────────────────────────────────────────────────────

def prepare_classification_style(data_path: Path, out_dir: Path):
    """
    Input layout:
        data_path/
            ok_front/    ← good castings
            def_front/   ← defective castings
    OR similar names containing 'ok' / 'def' / 'defect'
    """
    # Auto-detect folder names
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    ok_dirs  = [d for d in subdirs if "ok" in d.name.lower() and "def" not in d.name.lower()]
    def_dirs = [d for d in subdirs if "def" in d.name.lower()]

    if not ok_dirs or not def_dirs:
        raise ValueError(
            f"Cannot find ok/def folders in {data_path}.\n"
            f"Found: {[d.name for d in subdirs]}\n"
            "Expected folders containing 'ok' and 'def' in their names."
        )

    print(f"[INFO] OK  folder(s): {[d.name for d in ok_dirs]}")
    print(f"[INFO] DEF folder(s): {[d.name for d in def_dirs]}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def get_images(dirs):
        imgs = []
        for d in dirs:
            imgs += [f for f in d.rglob("*") if f.suffix.lower() in exts]
        return imgs

    ok_files  = get_images(ok_dirs)
    def_files = get_images(def_dirs)

    print(f"[INFO] Found {len(ok_files)} OK images, {len(def_files)} DEFECT images")

    # Split each class independently (stratified)
    ok_tr,  ok_va,  ok_te  = split_files(ok_files)
    def_tr, def_va, def_te = split_files(def_files)

    make_dirs(out_dir)
    stats = {"train": {}, "val": {}, "test": {}}

    for split_name, ok_split, def_split in [
        ("train", ok_tr, def_tr),
        ("val",   ok_va, def_va),
        ("test",  ok_te, def_te),
    ]:
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        count_ok, count_def = 0, 0

        for src in ok_split:
            dst_img = img_dir / src.name
            dst_lbl = lbl_dir / (src.stem + ".txt")
            copy_and_label(src, dst_img, dst_lbl, class_id=1)  # class 1 = ok
            count_ok += 1

        for src in def_split:
            # Prefix with 'def_' to avoid name collisions
            dst_img = img_dir / ("def_" + src.name)
            dst_lbl = lbl_dir / ("def_" + src.stem + ".txt")
            copy_and_label(src, dst_img, dst_lbl, class_id=0)  # class 0 = defect
            count_def += 1

        stats[split_name] = {"ok": count_ok, "defect": count_def, "total": count_ok + count_def}
        print(f"[OK] {split_name:5s}: {count_def} defects + {count_ok} ok = {count_ok+count_def} images")

    yaml_path = write_yaml(out_dir, out_dir)
    write_stats(out_dir, stats)
    return yaml_path, stats


def prepare_presplit_style(data_path: Path, out_dir: Path):
    """
    Input layout already has train/val/test splits:
        data_path/
            train/ok_front/ + train/def_front/
            val/ok_front/   + val/def_front/
            test/ok_front/  + test/def_front/
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    make_dirs(out_dir)
    stats = {}

    for split_name in ["train", "val", "test"]:
        split_path = data_path / split_name
        if not split_path.exists():
            print(f"[WARN] {split_path} not found, skipping.")
            continue

        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        count_ok, count_def = 0, 0

        for img_file in split_path.rglob("*"):
            if img_file.suffix.lower() not in exts:
                continue
            parent = img_file.parent.name.lower()
            if "def" in parent:
                class_id = 0
                dst_name = "def_" + img_file.name
                count_def += 1
            else:
                class_id = 1
                dst_name = img_file.name
                count_ok += 1
            copy_and_label(img_file, img_dir / dst_name, lbl_dir / (Path(dst_name).stem + ".txt"), class_id)

        stats[split_name] = {"ok": count_ok, "defect": count_def, "total": count_ok + count_def}
        print(f"[OK] {split_name:5s}: {count_def} defects + {count_ok} ok = {count_ok+count_def} images")

    yaml_path = write_yaml(out_dir, out_dir)
    write_stats(out_dir, stats)
    return yaml_path, stats


def verify_dataset(out_dir: Path):
    """Quick sanity check — make sure every image has a label."""
    errors = []
    for split in ["train", "val", "test"]:
        imgs = list((out_dir / "images" / split).glob("*"))
        lbls = list((out_dir / "labels" / split).glob("*.txt"))
        if len(imgs) != len(lbls):
            errors.append(f"{split}: {len(imgs)} images but {len(lbls)} labels!")
        else:
            print(f"[OK] {split}: {len(imgs)} image-label pairs verified")
    if errors:
        print("[WARN] Mismatches found:")
        for e in errors: print(f"  {e}")
    else:
        print("[OK] Dataset integrity check passed!")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare casting dataset for YOLOv8")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to downloaded Kaggle casting dataset folder")
    parser.add_argument("--out_dir", type=str, default="./yolo_dataset",
                        help="Output directory for YOLO-formatted dataset")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "classification", "presplit"],
                        help="Dataset layout mode")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir   = Path(args.out_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    print(f"\n{'='*60}")
    print("  ZeroDefect AI — Phase 1: Dataset Preparation")
    print(f"{'='*60}")
    print(f"  Source : {data_path}")
    print(f"  Output : {out_dir}")
    print(f"{'='*60}\n")

    # Auto-detect layout
    mode = args.mode
    if mode == "auto":
        subdirs = [d.name.lower() for d in data_path.iterdir() if d.is_dir()]
        if any(s in subdirs for s in ["train", "val", "test"]):
            mode = "presplit"
        else:
            mode = "classification"
        print(f"[AUTO] Detected layout: {mode}")

    if mode == "classification":
        yaml_path, stats = prepare_classification_style(data_path, out_dir)
    else:
        yaml_path, stats = prepare_presplit_style(data_path, out_dir)

    verify_dataset(out_dir)

    print(f"\n{'='*60}")
    print("  NEXT STEP — copy this path for Phase 2 training:")
    print(f"  data.yaml → {yaml_path.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
