"""Module 6 — Download STL-10 dataset and AlexNet model for 224×224 testing.

STL-10:
  - 96×96 RGB images, 10 classes (airplane, bird, car, cat, deer,
    dog, horse, monkey, ship, truck)
  - Resized to 224×224 for AlexNet
  - Downloaded via torchvision (free, no registration needed)

AlexNet:
  - Pretrained on ImageNet (torchvision)
  - Downloaded and cached automatically on first use

Output:
    data2/stl10/
    ├── clean/    ← 224×224 PNG images  (img_XXXXX_classname.png)
    └── raw/      ← torchvision raw download cache

Usage (from adversarial_testing/ root):
    python modules/6_alexnet_data/download_stl10.py
    python modules/6_alexnet_data/download_stl10.py --n 500 --split test
"""

import argparse
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent

CLASSES = ["airplane", "bird", "car", "cat", "deer",
           "dog", "horse", "monkey", "ship", "truck"]

# AlexNet expects 224×224, normalized with ImageNet stats
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def download_alexnet():
    """Download and cache AlexNet pretrained model."""
    print("Downloading AlexNet (pretrained on ImageNet) ...")
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    model.eval()
    print("AlexNet ready.")
    return model


def download_stl10(out_dir: Path, n: int = 500, split: str = "test"):
    """Download STL-10, resize to 224×224, save as PNG.

    Args:
        out_dir: output folder (data/stl10/)
        n:       max number of images to save (0 = all)
        split:   'test' (8000 images) or 'train' (5000 images)
    """
    raw_dir   = out_dir / "raw"
    clean_dir = out_dir / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading STL-10 ({split} split) ...")
    dataset = torchvision.datasets.STL10(
        root=str(raw_dir),
        split=split,
        download=True,
        transform=T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
    )

    total = len(dataset) if n == 0 else min(n, len(dataset))
    print(f"Saving {total} images to {clean_dir} ...")

    to_pil = T.ToPILImage()
    for i in range(total):
        img_tensor, label = dataset[i]
        class_name = CLASSES[label]
        filename   = f"img_{i:05d}_{class_name}.png"
        to_pil(img_tensor).save(clean_dir / filename)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} saved ...", end="\r")

    print(f"\nDone. {total} images saved to {clean_dir}")
    print(f"\nClasses: {', '.join(CLASSES)}")
    print(f"Image size: 224×224 (resized from 96×96)")
    print(f"\nNext steps:")
    print(f"  python modules/6_alexnet_data/generate_patches_alexnet.py")


def main():
    parser = argparse.ArgumentParser(description="Download STL-10 for AlexNet testing")
    parser.add_argument("--n",     type=int, default=500,
                        help="Number of images (0 = all, default: 500)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "train"],
                        help="Dataset split (default: test)")
    parser.add_argument("--out",   type=str, default=None,
                        help="Output folder (default: data/stl10)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "data2" / "stl10"

    # Download AlexNet first (caches it)
    download_alexnet()

    # Download and save STL-10
    download_stl10(out_dir, n=args.n, split=args.split)


if __name__ == "__main__":
    main()
