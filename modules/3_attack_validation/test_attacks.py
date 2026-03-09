"""Module 3 — Validate which patched images actually fool the model.

Compares predictions on clean vs patched images and prints a report.
Does NOT save files — see save_fooled.py for that.

Usage (from adversarial_testing/ root):
    python modules/3_attack_validation/test_attacks.py
    python modules/3_attack_validation/test_attacks.py --clean data/cifar10/clean --patched data/cifar10/patched
    python modules/3_attack_validation/test_attacks.py --target 9

An attack is SUCCESSFUL when:
    - original prediction ≠ target_class   (model was correct before)
    - patched  prediction == target_class   (patch fooled it)
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2470, 0.2435, 0.2616]


def load_model():
    print("Loading CIFAR-10 ResNet-20 ...")
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_resnet20",
        pretrained=True,
        verbose=False,
    )
    model.eval()
    return model


def normalize(tensor):
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    return (tensor - m) / s


def predict(model, tensor):
    """Predict class for a (3,H,W) tensor in [0,1]. Returns int class index."""
    with torch.no_grad():
        logits = model(normalize(tensor).unsqueeze(0))
        return int(logits.argmax().item())


def run_validation(clean_dir: Path, patched_dir: Path,
                   target_class: int, require_original_correct: bool = True):
    """
    Returns:
        results: list of dicts with keys:
            filename, clean_pred, patched_pred, fooled, clean_class_name
    """
    to_tensor = T.ToTensor()
    model = load_model()

    clean_files = sorted(f for f in clean_dir.iterdir() if f.suffix.lower() == ".png")
    if not clean_files:
        print(f"No PNG files in {clean_dir}")
        return []

    print(f"\nValidating {len(clean_files)} image pairs ...")
    print(f"Target class: {target_class} ({CLASSES[target_class]})")
    print(f"{'Filename':<30} {'Clean':>12} {'Patched':>12} {'Fooled':>8}")
    print("-" * 68)

    results = []
    fooled_count = 0

    for f in clean_files:
        patched_f = patched_dir / f.name
        if not patched_f.exists():
            print(f"  {f.name:<30} [no matching patched file, skipping]")
            continue

        clean_tensor   = to_tensor(Image.open(f).convert("RGB"))
        patched_tensor = to_tensor(Image.open(patched_f).convert("RGB"))

        clean_pred   = predict(model, clean_tensor)
        patched_pred = predict(model, patched_tensor)

        # Attack succeeds if patched → target AND (optionally) clean ≠ target
        original_ok = (not require_original_correct) or (clean_pred != target_class)
        fooled = original_ok and (patched_pred == target_class)

        if fooled:
            fooled_count += 1

        results.append({
            "filename":       f.name,
            "clean_pred":     clean_pred,
            "patched_pred":   patched_pred,
            "fooled":         fooled,
            "clean_class":    CLASSES[clean_pred],
            "patched_class":  CLASSES[patched_pred],
        })

        marker = " <-- FOOLED" if fooled else ""
        if fooled or len(results) % 50 == 0:
            print(f"  {f.name:<30} {CLASSES[clean_pred]:>12} {CLASSES[patched_pred]:>12} "
                  f"{'YES':>8}{marker}")

    total = len(results)
    print("\n" + "=" * 68)
    print(f"Total pairs tested : {total}")
    print(f"Successful attacks : {fooled_count} ({fooled_count/max(1,total)*100:.1f}%)")
    print(f"Failed attacks     : {total - fooled_count}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate adversarial attack success rate")
    parser.add_argument("--clean",   type=str, default=None)
    parser.add_argument("--patched", type=str, default=None)
    parser.add_argument("--target",  type=int, default=9,
                        help="Target class index (default: 9 = truck)")
    parser.add_argument("--no-require-correct", action="store_true",
                        help="Count attacks even if model was already predicting target on clean")
    args = parser.parse_args()

    clean_dir   = Path(args.clean)   if args.clean   else ROOT / "data" / "cifar10" / "clean"
    patched_dir = Path(args.patched) if args.patched else ROOT / "data" / "cifar10" / "patched"

    run_validation(clean_dir, patched_dir, args.target,
                   require_original_correct=not args.no_require_correct)


if __name__ == "__main__":
    main()
