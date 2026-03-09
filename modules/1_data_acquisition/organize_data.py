"""Module 1 — Show statistics and optionally filter the clean dataset.

Usage:
    python modules/1_data_acquisition/organize_data.py
    python modules/1_data_acquisition/organize_data.py --classes airplane dog
    python modules/1_data_acquisition/organize_data.py --classes airplane dog --out data/cifar10/clean_subset
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def show_stats(folder: Path) -> dict:
    if not folder.exists():
        print(f"Folder not found: {folder}")
        print("Run download_cifar.py first.")
        return {}

    files = sorted(f for f in folder.iterdir()
                   if f.suffix.lower() == ".png")
    if not files:
        print(f"No PNG files found in {folder}")
        return {}

    counts = {}
    for f in files:
        # Filename: img_00000_classname.png
        parts = f.stem.split("_", 2)
        cls = parts[2] if len(parts) >= 3 else "unknown"
        counts[cls] = counts.get(cls, 0) + 1

    print(f"\nData statistics: {folder}")
    print(f"Total images: {len(files)}")
    print("-" * 35)
    for cls, cnt in sorted(counts.items()):
        bar = "█" * (cnt // 5)
        print(f"  {cls:<12} {cnt:>4}  {bar}")
    return counts


def filter_by_class(source: Path, dest: Path, keep_classes: list) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in sorted(source.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        parts = f.stem.split("_", 2)
        cls = parts[2] if len(parts) >= 3 else ""
        if cls in keep_classes:
            shutil.copy2(f, dest / f.name)
            copied += 1

    print(f"\nCopied {copied} images from {source} -> {dest}")
    print(f"Filtered to classes: {keep_classes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect and optionally filter CIFAR-10 clean set")
    parser.add_argument("--dir",     type=str, default=None,
                        help="Folder to inspect (default: data/cifar10/clean)")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="If provided, filter to these classes and save to --out")
    parser.add_argument("--out",     type=str, default=None,
                        help="Output folder for filtered subset")
    args = parser.parse_args()

    source = Path(args.dir) if args.dir else ROOT / "data" / "cifar10" / "clean"
    counts = show_stats(source)

    if args.classes and counts:
        dest = Path(args.out) if args.out else ROOT / "data" / "cifar10" / "clean_subset"
        filter_by_class(source, dest, args.classes)
        show_stats(dest)
