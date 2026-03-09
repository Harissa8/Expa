"""Download ImageNette dataset for AlexNet testing (data3).

ImageNette: 10 easy ImageNet classes, full 224×224 resolution.
Classes: tench, english_springer, cassette_player, chain_saw, church,
         french_horn, garbage_truck, gas_pump, golf_ball, parachute

No registration needed — freely available from fastai.

Output:
    data3/imagenette/
    ├── clean/    ← 224×224 PNG images (img_XXXXX_classname.png)
    └── raw/      ← downloaded tarball + extracted folder

Usage (from adversarial_testing/ root):
    python modules/6_alexnet_data/download_imagenette.py           # all val images (~3925)
    python modules/6_alexnet_data/download_imagenette.py --n 2000
    python modules/6_alexnet_data/download_imagenette.py --split train --n 5000
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

# ImageNette WordNet ID → readable class name
CLASSES = {
    "n01440764": "tench",
    "n02102040": "english_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}


def download_imagenette(out_dir: Path, n: int = 0, split: str = "val"):
    raw_dir   = out_dir / "raw"
    clean_dir = out_dir / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    tgz_path  = raw_dir / "imagenette2-320.tgz"
    extracted = raw_dir / "imagenette2-320"

    # ── Download ────────────────────────────────────────────────────────────────
    if not tgz_path.exists():
        print(f"Downloading ImageNette (~1.5 GB) ...")
        def progress(count, block_size, total_size):
            pct = min(count * block_size / total_size * 100, 100)
            print(f"  {pct:.1f}%", end="\r")
        urllib.request.urlretrieve(URL, tgz_path, reporthook=progress)
        print("\nDownload complete.")
    else:
        print("Tarball already downloaded.")

    # ── Extract ─────────────────────────────────────────────────────────────────
    if not extracted.exists():
        print("Extracting ...")
        with tarfile.open(tgz_path) as tar:
            tar.extractall(raw_dir)
        print("Extracted.")
    else:
        print("Already extracted.")

    # ── Collect image paths ─────────────────────────────────────────────────────
    split_dir = extracted / ("train" if split == "train" else "val")
    resize    = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    to_pil    = T.ToPILImage()

    image_files = []
    for class_id, class_name in CLASSES.items():
        class_dir = split_dir / class_id
        if class_dir.exists():
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    image_files.append((f, class_name))

    total = len(image_files) if n == 0 else min(n, len(image_files))
    print(f"\nSaving {total} images to {clean_dir} ...")

    for i, (f, class_name) in enumerate(image_files[:total]):
        img        = Image.open(f).convert("RGB")
        img_tensor = resize(img)
        filename   = f"img_{i:05d}_{class_name}.png"
        to_pil(img_tensor).save(clean_dir / filename)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} saved ...", end="\r")

    print(f"\nDone. {total} images saved to {clean_dir}")
    print(f"\nNext steps:")
    print(f"  python modules/6_alexnet_data/generate_patches_alexnet.py "
          f"--input data3/imagenette/clean --size 28")
    print(f"  python modules/6_alexnet_data/save_splits_alexnet.py "
          f"--patch-size 28 --clean data3/imagenette/clean "
          f"--patched data3/imagenette/patched_28/raw "
          f"--fooled data3/imagenette/patched_28/fooled "
          f"--not-fooled data3/imagenette/patched_28/not_fooled")


def main():
    parser = argparse.ArgumentParser(description="Download ImageNette for AlexNet testing")
    parser.add_argument("--n",     type=int, default=0,
                        help="Number of images to save (0 = all, default: all ~3925 val)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"],
                        help="Dataset split: val (~3925) or train (~9469) (default: val)")
    parser.add_argument("--out",   type=str, default=None,
                        help="Output folder (default: data3/imagenette)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "data3" / "imagenette"
    download_imagenette(out_dir, n=args.n, split=args.split)


if __name__ == "__main__":
    main()
