# NOTE: AI Generated
# It's an extremely simple script, but I was too lazy to write it myself - James

"""Prepare dataset into train/ val/ test/ ImageFolder structure.

Source layout (basically just the uncompressed Figshare Dataset as is):
  data_root/
    pituitary_tumor/*.jpg
    glioma_tumor/*.jpg
    meningioma_tumor/*.jpg
    no_tumor/*.jpg

Target layout (ImageFolder expected by training script):
  output_root/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
    test/<class_name>/*.jpg

Class name mapping:
  pituitary_tumor -> pituitary
  glioma_tumor -> glioma
  meningioma_tumor -> meningioma
  no_tumor -> negative

Split ratios default to 0.8 / 0.1 / 0.1. Random but reproducible via --seed.

Example:
  python scripts/prepare_dataset.py --source data --dest prepared_data --seed 42 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import argparse
from pathlib import Path
import shutil
import random

MAPPING: dict[str, str] = {
    'pituitary_tumor': 'pituitary',
    'glioma_tumor': 'glioma',
    'meningioma_tumor': 'meningioma',
    'no_tumor': 'negative',
}

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def list_images(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]


def split_list(items: list[Path], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    rng = random.Random(seed)
    items_copy = items[:]
    rng.shuffle(items_copy)
    n = len(items_copy)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_items = items_copy[:n_train]
    val_items = items_copy[n_train:n_train + n_val]
    test_items = items_copy[n_train + n_val:]
    return train_items, val_items, test_items


def copy_files(files: list[Path], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_dir / f.name)


def prepare(source: Path, dest: Path, train_ratio: float, val_ratio: float, test_ratio: float, seed: int, dry_run: bool = False):
    if not source.exists():
        raise SystemExit(f"Source root does not exist: {source}")
    dest.mkdir(parents=True, exist_ok=True)

    summary = []
    for raw_name, mapped in MAPPING.items():
        class_dir = source / raw_name
        if not class_dir.exists():
            print(f"[WARN] Missing source class folder: {class_dir}")
            continue
        images = list_images(class_dir)
        if not images:
            print(f"[WARN] No images found in {class_dir}")
            continue
        tr, va, te = split_list(images, train_ratio, val_ratio, test_ratio, seed)
        summary.append((mapped, len(tr), len(va), len(te)))
        if not dry_run:
            copy_files(tr, dest / 'train' / mapped)
            copy_files(va, dest / 'val' / mapped)
            copy_files(te, dest / 'test' / mapped)
        print(f"Class {mapped}: train={len(tr)} val={len(va)} test={len(te)}")

    print("\nSummary:")
    for mapped, a, b, c in summary:
        print(f"  {mapped}: train={a} val={b} test={c}")


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare dataset into train/val/test folders with mapping.")
    ap.add_argument('--source', type=Path, required=True, help='Root containing original class folders')
    ap.add_argument('--dest', type=Path, required=True, help='Destination root for ImageFolder structure')
    ap.add_argument('--train-ratio', type=float, default=0.8)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--test-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry-run', action='store_true', help='Compute & print split counts without copying files')
    return ap.parse_args()


def main():
    args = parse_args()
    prepare(args.source, args.dest, args.train_ratio, args.val_ratio, args.test_ratio, args.seed, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
