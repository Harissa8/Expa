"""Module 6 — Validate AlexNet attacks and split into fooled / not_fooled.

Checks each patched image: did AlexNet predict the target class (banana=954)?
Saves patched images (no clean copies) to data2/fooled/ and data2/not_fooled/.

Usage (from adversarial_testing/ root):
    python modules/6_alexnet_data/save_splits_alexnet.py
"""

import argparse
import shutil
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_alexnet():
    model = torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    )
    model.eval()
    return model


def normalize(tensor, mean, std):
    m = torch.tensor(mean, dtype=tensor.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std,  dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - m) / s


def run_splits(clean_dir, patched_dir, fooled_dir, not_fooled_dir, target_class):
    clean_dir     = Path(clean_dir)
    patched_dir   = Path(patched_dir)
    fooled_dir    = Path(fooled_dir)
    not_fooled_dir= Path(not_fooled_dir)

    fooled_dir.mkdir(parents=True, exist_ok=True)
    not_fooled_dir.mkdir(parents=True, exist_ok=True)

    print("Loading AlexNet ...")
    model     = load_alexnet()
    to_tensor = T.ToTensor()

    patched_files = sorted(f for f in patched_dir.iterdir()
                           if f.suffix.lower() == ".png" and not f.name.startswith("_"))

    fooled = not_fooled = skipped = 0

    print(f"\nValidating {len(patched_files)} patched images ...")
    print(f"Target class: {target_class} (banana)")
    print(f"{'Filename':<35} {'Clean pred':>12} {'Patched pred':>13} {'Fooled':>8}")
    print("-" * 72)

    for f in patched_files:
        clean_f = clean_dir / f.name
        if not clean_f.exists():
            skipped += 1
            continue

        clean_img   = to_tensor(Image.open(clean_f).convert("RGB"))
        patched_img = to_tensor(Image.open(f).convert("RGB"))

        with torch.no_grad():
            clean_pred   = model(normalize(clean_img.unsqueeze(0),   MEAN, STD)).argmax().item()
            patched_pred = model(normalize(patched_img.unsqueeze(0), MEAN, STD)).argmax().item()

        if patched_pred == target_class:
            shutil.copy2(f, fooled_dir / f.name)
            fooled += 1
            tag = "YES"
        else:
            shutil.copy2(f, not_fooled_dir / f.name)
            not_fooled += 1
            tag = "no"

        print(f"  {f.name:<33} {clean_pred:>12} {patched_pred:>13} {tag:>8}")

    total = fooled + not_fooled
    print(f"\n{'='*55}")
    print(f"  Total tested  : {total}")
    print(f"  Fooled        : {fooled}  ({fooled/max(1,total)*100:.1f}%)")
    print(f"  Not fooled    : {not_fooled}  ({not_fooled/max(1,total)*100:.1f}%)")
    print(f"  Clean dir     : {clean_dir}")
    print(f"  Fooled dir    : {fooled_dir}")
    print(f"  Not-fooled dir: {not_fooled_dir}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean",       type=str, default=None)
    parser.add_argument("--patched",     type=str, default=None)
    parser.add_argument("--fooled",      type=str, default=None)
    parser.add_argument("--not-fooled",  type=str, default=None)
    parser.add_argument("--patch-size",  type=int, default=28,
                        help="Patch size — used to resolve default paths (default: 28)")
    parser.add_argument("--target",      type=int, default=954)
    args = parser.parse_args()

    ps             = args.patch_size
    clean_dir      = Path(args.clean)      if args.clean      else ROOT / "data2" / "stl10" / "clean"
    patched_dir    = Path(args.patched)    if args.patched    else ROOT / "data2" / "stl10" / f"patched_{ps}" / "raw"
    fooled_dir     = Path(args.fooled)     if args.fooled     else ROOT / "data2" / "stl10" / f"patched_{ps}" / "fooled"
    not_fooled_dir = Path(args.not_fooled) if args.not_fooled else ROOT / "data2" / "stl10" / f"patched_{ps}" / "not_fooled"

    run_splits(clean_dir, patched_dir, fooled_dir, not_fooled_dir, args.target)


if __name__ == "__main__":
    main()
