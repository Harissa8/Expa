"""Module 1 — Download CIFAR-10 and save images as PNG files.

Usage (from adversarial_testing/ root):
    python modules/1_data_acquisition/download_cifar.py
    python modules/1_data_acquisition/download_cifar.py --n 200
    python modules/1_data_acquisition/download_cifar.py --n 500 --out data/cifar10/clean

Output:
    data/cifar10/clean/img_00000_airplane.png
    data/cifar10/clean/img_00001_automobile.png
    ...
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T

# ── Allow running from adversarial_testing/ OR from this file's folder ─────────
ROOT = Path(__file__).resolve().parent.parent.parent  # adversarial_testing/
sys.path.insert(0, str(ROOT))

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

TRANSFORM = T.ToTensor()   # keeps [0,1] float; PIL→tensor for saving


def download_and_save(n_images: int, output_dir: Path, data_root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading CIFAR-10 test set to {data_root} ...")
    dataset = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=True,
        transform=TRANSFORM,
    )

    n = min(n_images, len(dataset))
    print(f"Saving {n} images to {output_dir} ...")

    class_counts = {c: 0 for c in CLASSES}
    to_pil = T.ToPILImage()

    for idx in range(n):
        tensor, label = dataset[idx]
        class_name = CLASSES[label]
        filename = f"img_{idx:05d}_{class_name}.png"
        path = output_dir / filename
        to_pil(tensor).save(path)
        class_counts[class_name] += 1

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{n} saved...")

    print(f"\nDone. {n} images saved to: {output_dir}")
    print("\nClass distribution:")
    for cls, cnt in class_counts.items():
        bar = "█" * (cnt // 5)
        print(f"  {cls:<12} {cnt:>4}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CIFAR-10 and save as PNG")
    parser.add_argument("--n",   type=int, default=500,
                        help="Number of images to save (default: 500)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output folder (default: data/cifar10/clean/)")
    args = parser.parse_args()

    out_dir  = Path(args.out) if args.out else ROOT / "data" / "cifar10" / "clean"
    data_root = ROOT / "data" / "cifar10" / "raw"

    download_and_save(args.n, out_dir, data_root)
