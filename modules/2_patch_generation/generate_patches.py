"""Module 2 — Generate adversarial patches and apply to clean images.

Workflow:
    1. Load N clean images from --input folder
    2. Run PGD on a batch to generate a universal adversarial patch
    3. Apply the patch to every clean image (at center, top-left, random)
    4. Save patched images to --output (same filenames as clean)
    5. Save patch.png for inspection

Usage (from adversarial_testing/ root):
    python modules/2_patch_generation/generate_patches.py
    python modules/2_patch_generation/generate_patches.py --input data/cifar10/clean --output data/cifar10/patched
    python modules/2_patch_generation/generate_patches.py --iters 200 --size 10

The PGD attack optimizes a single patch that, when pasted on any image,
causes the model to predict TARGET_CLASS.  This is a universal patch —
one patch fools all images.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent  # adversarial_testing/

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(cfg: dict):
    """Load CIFAR-10 ResNet-20 from torch.hub."""
    print(f"Loading {cfg['hub_name']} from torch.hub ...")
    model = torch.hub.load(
        cfg["hub_repo"],
        cfg["hub_name"],
        pretrained=True,
        verbose=False,
    )
    model.eval()
    return model


def normalize(tensor, mean, std):
    """Normalize a (B,3,H,W) or (3,H,W) tensor in [0,1]."""
    m = torch.tensor(mean, dtype=tensor.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std,  dtype=tensor.dtype).view(1, 3, 1, 1)
    if tensor.dim() == 3:
        m, s = m.squeeze(0), s.squeeze(0)
    return (tensor - m) / s


# ── PGD patch generation ───────────────────────────────────────────────────────

def generate_patch(model, images, cfg: dict) -> torch.Tensor:
    """Generate a universal adversarial patch using PGD.

    Args:
        model:  CIFAR-10 model (eval mode, takes normalized input)
        images: (B, 3, H, W) float tensor in [0, 1]
        cfg:    patch configuration dict

    Returns:
        patch: (3, patch_size, patch_size) float tensor in [0, 1]
    """
    patch_size  = cfg["patch_size"]
    target      = cfg["target_class"]
    iters       = cfg["pgd_iters"]
    lr          = cfg["pgd_lr"]
    mean        = cfg["model"]["mean"]
    std         = cfg["model"]["std"]
    img_size    = cfg["model"]["input_size"]

    x = (img_size - patch_size) // 2
    y = (img_size - patch_size) // 2

    # Pre-normalize the background images (patch position will be overwritten)
    images_norm = normalize(images, mean, std)

    target_t = torch.tensor([target] * images.shape[0])

    patch = torch.rand(3, patch_size, patch_size, requires_grad=True)

    print(f"PGD patch generation: target={target} ({CLASSES[target]}), "
          f"size={patch_size}px, iters={iters}")

    for i in range(iters):
        # Normalize patch and paste into images
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


def apply_patch(image: torch.Tensor, patch: torch.Tensor,
                img_size: int, position: str = "center") -> torch.Tensor:
    """Paste patch onto image at given position.

    Args:
        image:    (3, H, W) float tensor in [0, 1]
        patch:    (3, ps, ps) float tensor in [0, 1]
        img_size: input size of the model (e.g. 32)
        position: "center" | "top-left" | "bottom-right"

    Returns:
        patched image (3, H, W)
    """
    ps = patch.shape[1]
    result = image.clone()

    if position == "center":
        x = y = (img_size - ps) // 2
    elif position == "top-left":
        x, y = 0, 0
    elif position == "bottom-right":
        x = y = img_size - ps
    else:
        x = y = (img_size - ps) // 2

    result[:, y:y + ps, x:x + ps] = patch
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate and apply adversarial patch")
    parser.add_argument("--input",    type=str, default=None,
                        help="Folder with clean PNG images (default: data/cifar10/clean)")
    parser.add_argument("--output",   type=str, default=None,
                        help="Folder to save patched images (default: data/cifar10/patched)")
    parser.add_argument("--config",   type=str, default=None,
                        help="Path to patch_config.json")
    parser.add_argument("--iters",    type=int, default=None,
                        help="Override pgd_iters from config")
    parser.add_argument("--size",     type=int, default=None,
                        help="Override patch_size from config")
    parser.add_argument("--position", type=str, default="center",
                        choices=["center", "top-left", "bottom-right"],
                        help="Patch placement on image")
    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.config) if args.config else Path(__file__).parent / "patch_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    if args.iters:
        cfg["pgd_iters"] = args.iters
    if args.size:
        cfg["patch_size"] = args.size

    input_dir  = Path(args.input)  if args.input  else ROOT / "data" / "cifar10" / "clean"
    output_dir = Path(args.output) if args.output else ROOT / "data" / "cifar10" / "patched"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    to_tensor = T.ToTensor()
    images_raw = []
    image_files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() == ".png")
    if not image_files:
        print(f"No PNG files found in {input_dir}")
        return

    print(f"Loading {len(image_files)} images from {input_dir} ...")
    for f in image_files:
        img = Image.open(f).convert("RGB")
        images_raw.append(to_tensor(img))

    all_images = torch.stack(images_raw)  # (N, 3, H, W)

    # Generate patch from first pgd_batch images
    batch_size = min(cfg["pgd_batch"], len(images_raw))
    batch      = all_images[:batch_size]
    model      = load_model(cfg["model"])

    patch = generate_patch(model, batch, cfg)

    # Save patch for inspection
    patch_path = output_dir / "_patch.png"
    T.ToPILImage()(patch).save(patch_path)
    print(f"\nPatch saved → {patch_path}")

    # Apply patch to all images
    to_pil   = T.ToPILImage()
    img_size = cfg["model"]["input_size"]
    position = args.position

    print(f"\nApplying patch (position={position}) to {len(image_files)} images ...")
    for file_path, img_tensor in zip(image_files, images_raw):
        patched = apply_patch(img_tensor, patch, img_size, position)
        to_pil(patched).save(output_dir / file_path.name)

    print(f"Done. {len(image_files)} patched images saved to {output_dir}")

    # Quick verification
    mean, std = cfg["model"]["mean"], cfg["model"]["std"]
    sample_norm   = normalize(images_raw[0].unsqueeze(0), mean, std)
    patched_norm  = normalize(
        apply_patch(images_raw[0], patch, img_size, position).unsqueeze(0), mean, std
    )
    with torch.no_grad():
        orig_pred    = model(sample_norm).argmax().item()
        patched_pred = model(patched_norm).argmax().item()

    target = cfg["target_class"]
    print(f"\nPatch check on first image:")
    print(f"  Original prediction: {CLASSES[orig_pred]} ({orig_pred})")
    print(f"  Patched prediction:  {CLASSES[patched_pred]} ({patched_pred})")
    print(f"  Attack target:       {CLASSES[target]} ({target})")
    print(f"  Fooled: {'YES' if patched_pred == target else 'PARTIAL / NOT YET'}")


if __name__ == "__main__":
    main()
