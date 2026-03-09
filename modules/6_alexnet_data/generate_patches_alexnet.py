"""Module 6 — Generate adversarial patches for AlexNet on 224×224 images.

Same PGD approach as module 2, but for:
  - AlexNet pretrained on ImageNet (1000 classes)
  - 224×224 input images (STL-10)
  - Patch size: 56×56 (6.25% of image — same proportion as 8×8 on 32×32)
  - Target class: 954 (banana) — clearly different from all STL-10 classes

Usage (from adversarial_testing/ root):
    python modules/6_alexnet_data/generate_patches_alexnet.py
    python modules/6_alexnet_data/generate_patches_alexnet.py --iters 200 --size 56
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T 
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent


def load_alexnet():
    print("Loading AlexNet (pretrained on ImageNet) ...")
    model = torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    )
    model.eval()
    return model


def normalize(tensor, mean, std):
    m = torch.tensor(mean, dtype=tensor.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std,  dtype=tensor.dtype).view(1, 3, 1, 1)
    if tensor.dim() == 3:
        m, s = m.squeeze(0), s.squeeze(0)
    return (tensor - m) / s


def generate_patch(model, images, cfg, position="center"):
    """PGD universal adversarial patch."""
    patch_size = cfg["patch_size"]
    target     = cfg["target_class"]
    iters      = cfg["pgd_iters"]
    lr         = cfg["pgd_lr"]
    mean       = cfg["model"]["mean"]
    std        = cfg["model"]["std"]
    img_size   = cfg["model"]["input_size"]

    mid  = (img_size - patch_size) // 2
    edge = img_size - patch_size
    positions = {
        "center":        (mid,  mid),
        "top-left":      (0,    0),
        "top-right":     (0,    edge),
        "bottom-left":   (edge, 0),
        "bottom-right":  (edge, edge),
        "top-center":    (0,    mid),
        "bottom-center": (edge, mid),
        "left-center":   (mid,  0),
        "right-center":  (mid,  edge),
    }
    y, x = positions.get(position, (mid, mid))

    images_norm = normalize(images, mean, std)
    target_t    = torch.tensor([target] * images.shape[0])
    patch       = torch.rand(3, patch_size, patch_size, requires_grad=True)

    print(f"PGD patch: target={target} ({cfg['target_name']}), "
          f"size={patch_size}px on {img_size}px image, iters={iters}, position={position}")

    for i in range(iters):
        patch_n = normalize(patch.unsqueeze(0), mean, std).squeeze(0)
        imgs = images_norm.clone()
        imgs[:, :, y:y + patch_size, x:x + patch_size] = patch_n

        logits = model(imgs)
        loss   = F.cross_entropy(logits, target_t)
        loss.backward()

        with torch.no_grad():
            patch -= lr * patch.grad.sign()
            patch.clamp_(0.0, 1.0)
            patch.grad.zero_()

        if (i + 1) % 25 == 0:
            conf = torch.softmax(logits, dim=1)[:, target].mean().item()
            print(f"  iter {i+1:3d}/{iters}  avg target conf: {conf:.3f}  loss: {loss.item():.4f}")

    return patch.detach()


def apply_patch(image, patch, img_size, position="center"):
    ps   = patch.shape[1]
    mid  = (img_size - ps) // 2
    edge = img_size - ps

    positions = {
        "center":        (mid,  mid),
        "top-left":      (0,    0),
        "top-right":     (0,    edge),
        "bottom-left":   (edge, 0),
        "bottom-right":  (edge, edge),
        "top-center":    (0,    mid),
        "bottom-center": (edge, mid),
        "left-center":   (mid,  0),
        "right-center":  (mid,  edge),
    }

    y, x = positions.get(position, (mid, mid))
    result = image.clone()
    result[:, y:y + ps, x:x + ps] = patch
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial patches for AlexNet")
    parser.add_argument("--input",    type=str, default=None,
                        help="Clean images folder (default: data2/stl10/clean)")
    parser.add_argument("--output",   type=str, default=None,
                        help="Patched images folder (default: data2/stl10/patched)")
    parser.add_argument("--config",   type=str, default=None)
    parser.add_argument("--iters",       type=int, default=None)
    parser.add_argument("--size",        type=int, default=None)
    parser.add_argument("--target",      type=int, default=None,
                        help="Target class ID (overrides config). E.g. 849=teapot, 985=daisy, 440=beer bottle")
    parser.add_argument("--target-name", type=str, default=None,
                        help="Human-readable name for the target class (used in output filenames/logs)")
    parser.add_argument("--position",    type=str, default="center",
                        choices=["center", "top-left", "top-right",
                                 "bottom-left", "bottom-right",
                                 "top-center", "bottom-center",
                                 "left-center", "right-center"])
    parser.add_argument("--limit",       type=int, default=None,
                        help="Max number of images to load (e.g. 4000)")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else Path(__file__).parent / "patch_config_alexnet.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    if args.iters:       cfg["pgd_iters"]    = args.iters
    if args.size:        cfg["patch_size"]   = args.size
    if args.target:      cfg["target_class"] = args.target
    if args.target_name: cfg["target_name"]  = args.target_name
    elif args.target:    cfg["target_name"]  = f"class_{args.target}"

    patch_size = cfg["patch_size"]
    input_dir  = Path(args.input)  if args.input  else ROOT / "data2" / "stl10" / "clean"
    output_dir = Path(args.output) if args.output else ROOT / "data2" / "stl10" / f"patched_{patch_size}" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    to_tensor   = T.ToTensor()
    to_pil      = T.ToPILImage()
    image_files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() == ".png")
    if args.limit:
        image_files = image_files[: args.limit]

    if not image_files:
        print(f"No PNG files found in {input_dir}")
        return

    total = len(image_files)
    print(f"Loading {total} images from {input_dir} ...")
    images_raw = []
    for i, f in enumerate(image_files, 1):
        images_raw.append(to_tensor(Image.open(f).convert("RGB")))
        if i % 200 == 0 or i == total:
            print(f"  loaded {i}/{total}")
    all_images = torch.stack(images_raw)

    model      = load_alexnet()
    batch_size = min(cfg["pgd_batch"], len(images_raw))
    patch      = generate_patch(model, all_images[:batch_size], cfg)

    # Save patch for inspection
    patch_path = output_dir / "_patch.png"
    to_pil(patch).save(patch_path)
    print(f"\nPatch saved → {patch_path}")

    # Apply to all images
    img_size = cfg["model"]["input_size"]
    print(f"\nApplying patch to {total} images ...")
    for i, (file_path, img_tensor) in enumerate(zip(image_files, images_raw), 1):
        patched = apply_patch(img_tensor, patch, img_size, args.position)
        to_pil(patched).save(output_dir / file_path.name)
        if i % 500 == 0 or i == total:
            print(f"  applied {i}/{total}")

    print(f"Done. {total} patched images saved to {output_dir}")

    # Quick check on first image
    mean, std = cfg["model"]["mean"], cfg["model"]["std"]
    with torch.no_grad():
        orig_pred    = model(normalize(images_raw[0].unsqueeze(0), mean, std)).argmax().item()
        patched_pred = model(normalize(
            apply_patch(images_raw[0], patch, img_size).unsqueeze(0), mean, std
        )).argmax().item()

    target = cfg["target_class"]
    print(f"\nPatch check on first image:")
    print(f"  Original:  ImageNet class {orig_pred}")
    print(f"  Patched:   ImageNet class {patched_pred}")
    print(f"  Target:    {target} ({cfg['target_name']})")
    print(f"  Fooled: {'YES' if patched_pred == target else 'PARTIAL / NOT YET'}")


if __name__ == "__main__":
    main()
