"""Module 8 — Median Filter defense for AlexNet on 224×224 images.

Key differences from CIFAR-10 version (module 5):
  - Model:      AlexNet (ImageNet pretrained) instead of ResNet-20
  - Kernel:     5×5 instead of 3×3  (images are 7× larger)
  - Normalize:  ImageNet mean/std instead of CIFAR-10 mean/std
  - KL thresh:  1.0 (same logic, same rule: class_changed + KL > thr = ATTACK)

AlexNet layers are not used here — median filter works on the raw image,
no hooking needed.

Usage (from adversarial_testing/ root):
    python modules/8_defense_median_alexnet/test_defense_alexnet.py
    python modules/8_defense_median_alexnet/test_defense_alexnet.py --verbose
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent.parent

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

_to_tensor = T.ToTensor()


# ── Model ──────────────────────────────────────────────────────────────────────

def load_alexnet():
    print("Loading AlexNet (pretrained on ImageNet) ...")
    model = torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    )
    model.eval()
    return model


# ── Median filter + detection ──────────────────────────────────────────────────

def apply_median_filter(image_tensor, kernel_size=5):
    arr = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    pil = Image.fromarray(arr).filter(ImageFilter.MedianFilter(size=kernel_size))
    return _to_tensor(pil)


def kl_divergence(orig_probs, filt_probs):
    return float(F.kl_div(filt_probs.log(), orig_probs, reduction="sum"))


def detect_median(model, image_tensor, kernel_size=5, kl_threshold=1.0):
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)

    norm_orig = ((image_tensor - m) / s).unsqueeze(0)
    with torch.no_grad():
        orig_probs = torch.softmax(model(norm_orig), dim=1)[0].cpu()
    orig_class = int(orig_probs.argmax())
    orig_conf  = float(orig_probs.max())

    filtered  = apply_median_filter(image_tensor, kernel_size)
    norm_filt = ((filtered - m) / s).unsqueeze(0)
    with torch.no_grad():
        filt_probs = torch.softmax(model(norm_filt), dim=1)[0].cpu()
    filt_class = int(filt_probs.argmax())
    filt_conf  = float(filt_probs.max())

    kl            = kl_divergence(orig_probs, filt_probs)
    class_changed = orig_class != filt_class
    # Rule: class changed AND KL > threshold → ATTACK
    is_attack     = class_changed and (kl > kl_threshold)

    return {
        "is_attack":      is_attack,
        "kl_divergence":  kl,
        "class_changed":  class_changed,
        "original_class": orig_class,
        "filtered_class": filt_class,
        "original_conf":  orig_conf,
        "filtered_conf":  filt_conf,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def run_evaluation(clean_dir, patched_dir, kernel_size=5, kl_threshold=1.0,
                   verbose=False, collect_records=False, max_images=None):
    clean_dir   = Path(clean_dir)
    patched_dir = Path(patched_dir)

    if not clean_dir.exists() or not patched_dir.exists():
        print("clean_dir or patched_dir not found.")
        return {}

    model     = load_alexnet()
    to_tensor = T.ToTensor()

    patched_files = sorted(f for f in patched_dir.iterdir() if f.suffix.lower() == ".png")
    clean_files   = sorted(f for f in (clean_dir / n.name for n in patched_files) if f.exists())
    if max_images and len(patched_files) > max_images:
        patched_files = patched_files[:max_images]
        clean_files   = clean_files[:max_images]

    print(f"\nMedian Filter Defense Evaluation (AlexNet 224×224)")
    print(f"{'='*65}")
    print(f"Clean dir:   {clean_dir}  ({len(clean_files)} images)")
    print(f"Patched dir: {patched_dir}  ({len(patched_files)} images)")
    print(f"Kernel:      {kernel_size}×{kernel_size}  (5×5 for 224×224 images)")
    print(f"KL thresh:   {kl_threshold}")
    print(f"Rule:        class_changed AND KL > {kl_threshold}")
    print(f"{'='*65}")

    TP = TN = FP = FN = 0
    records = []

    def evaluate_set(files, is_clean_label):
        nonlocal TP, TN, FP, FN
        label_str = "CLEAN" if is_clean_label else "PATCH"
        for f in files:
            img = to_tensor(Image.open(f).convert("RGB"))
            r   = detect_median(model, img, kernel_size, kl_threshold)
            verdict = "ATTACK" if r["is_attack"] else "CLEAN"

            if   is_clean_label and not r["is_attack"]: TN += 1
            elif is_clean_label and     r["is_attack"]: FP += 1
            elif not is_clean_label and r["is_attack"]: TP += 1
            else:                                        FN += 1

            if collect_records:
                records.append({
                    "filename":      f.name,
                    "true_label":    label_str,
                    "prediction":    verdict,
                    "correct":       (is_clean_label and not r["is_attack"]) or
                                     (not is_clean_label and r["is_attack"]),
                    "kl_divergence": round(r["kl_divergence"], 4),
                    "class_changed": r["class_changed"],
                    "orig_class":    r["original_class"],
                    "filt_class":    r["filtered_class"],
                    "orig_conf":     round(r["original_conf"], 4),
                    "filt_conf":     round(r["filtered_conf"], 4),
                })

            if verbose:
                wrong = " <--" if (is_clean_label and r["is_attack"]) or \
                                  (not is_clean_label and not r["is_attack"]) else ""
                print(f"{f.name:<30} {label_str:>7} "
                      f"KL={r['kl_divergence']:>6.3f} "
                      f"changed={str(r['class_changed']):>5} "
                      f"{verdict:>8}{wrong}")

    evaluate_set(clean_files,   is_clean_label=True)
    evaluate_set(patched_files, is_clean_label=False)

    total     = TP + TN + FP + FN
    tpr       = TP / max(1, TP + FN)
    fpr       = FP / max(1, FP + TN)
    accuracy  = (TP + TN) / max(1, total)
    clean_acc = TN / max(1, TN + FP)
    patch_acc = TP / max(1, TP + FN)

    print(f"\n{'='*55}")
    print(f"  Median Filter Defense Results (AlexNet)")
    print(f"{'='*55}")
    print(f"  On CLEAN images  ({len(clean_files)} total):")
    print(f"    Correct (CLEAN)  : {TN}  → {clean_acc*100:.1f}%")
    print(f"    Wrong   (ATTACK) : {FP}  ← false positives")
    print(f"  On PATCHED images ({len(patched_files)} total):")
    print(f"    Correct (ATTACK) : {TP}  → {patch_acc*100:.1f}% detection rate")
    print(f"    Missed  (CLEAN)  : {FN}  ← missed attacks")
    print(f"  Overall accuracy : {accuracy*100:.1f}%")
    print(f"  FPR              : {fpr*100:.1f}%")
    print(f"{'='*55}")

    result = {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
              "tpr": tpr, "fpr": fpr, "accuracy": accuracy}
    if collect_records:
        result["records"] = records
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean",   type=str, default=None)
    parser.add_argument("--patched", type=str, default=None)
    parser.add_argument("--kernel",  type=int,   default=5)
    parser.add_argument("--kl-thr",  type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    clean_dir   = Path(args.clean)   if args.clean   else ROOT / "data2" / "stl10" / "clean"
    patched_dir = Path(args.patched) if args.patched else ROOT / "data2" / "fooled"

    run_evaluation(clean_dir, patched_dir,
                   kernel_size=args.kernel,
                   kl_threshold=args.kl_thr,
                   verbose=args.verbose)
