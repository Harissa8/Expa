"""Module 5 — Median Filter defense evaluation.

Loads images from fooled_images/clean/ and fooled_images/patched/,
runs the median filter defense on each, then reports TPR / FPR / accuracy.

Usage (from adversarial_testing/ root):
    python modules/5_defense_median/test_defense.py --data data/fooled_images
    python modules/5_defense_median/test_defense.py --data data/fooled_images --verbose
    python modules/5_defense_median/test_defense.py --kl-thr 0.3 --kernel 3
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
from median_filter import detect_median

ROOT = Path(__file__).resolve().parent.parent.parent

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


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


# ── Public evaluation function (called by run_test_defense2.py) ────────────────

def run_evaluation(clean_dir, patched_dir, kernel_size=3, kl_threshold=0.5,
                   require_class_change=False, verbose=False,
                   collect_records=False):
    """Evaluate median filter defense.

    Args:
        clean_dir:            Path to folder of clean PNG images
        patched_dir:          Path to folder of patched PNG images
        kernel_size:          Median filter kernel (3 for 32×32)
        kl_threshold:         KL divergence threshold for attack detection
        require_class_change: If True, require class change for detection
        verbose:              Print every result

    Returns:
        dict with TP, TN, FP, FN, tpr, fpr, accuracy
    """
    clean_dir   = Path(clean_dir)
    patched_dir = Path(patched_dir)

    if not clean_dir.exists() or not patched_dir.exists():
        print(f"clean_dir or patched_dir not found.")
        return {}

    model     = load_model()
    to_tensor = T.ToTensor()

    patched_files = sorted(f for f in patched_dir.iterdir() if f.suffix.lower() == ".png")
    # Only use clean images that have a matching patched file
    clean_files   = sorted(f for f in (clean_dir / n.name for n in patched_files)
                           if f.exists())

    print(f"\nMedian Filter Defense Evaluation")
    print(f"{'='*65}")
    print(f"Clean dir:   {clean_dir}")
    print(f"Patched dir: {patched_dir}")
    print(f"Clean:       {len(clean_files)} images")
    print(f"Patched:     {len(patched_files)} images")
    print(f"Kernel:      {kernel_size}×{kernel_size}")
    print(f"KL thresh:   {kl_threshold}")
    print(f"{'='*65}")

    if verbose:
        print(f"\n{'Image':<28} {'Label':>7} {'KL':>7} {'Changed':>8} {'Result':>8}")
        print("-" * 63)

    TP = TN = FP = FN = 0
    records = []

    def evaluate_set(files, is_clean_label):
        nonlocal TP, TN, FP, FN
        label_str = "CLEAN" if is_clean_label else "PATCH"
        for f in files:
            img = to_tensor(Image.open(f).convert("RGB"))
            r   = detect_median(model, img, kernel_size, kl_threshold,
                                require_class_change)
            verdict = "ATTACK" if r["is_attack"] else "CLEAN"

            if is_clean_label and not r["is_attack"]:
                TN += 1
            elif is_clean_label and r["is_attack"]:
                FP += 1
            elif not is_clean_label and r["is_attack"]:
                TP += 1
            else:
                FN += 1

            if collect_records:
                records.append({
                    "filename":       f.name,
                    "true_label":     label_str,
                    "prediction":     verdict,
                    "correct":        (is_clean_label and not r["is_attack"]) or
                                      (not is_clean_label and r["is_attack"]),
                    "kl_divergence":  round(r["kl_divergence"], 4),
                    "class_changed":  r["class_changed"],
                    "orig_class":     r["original_class"],
                    "filt_class":     r["filtered_class"],
                    "orig_conf":      round(r["original_conf"], 4),
                    "filt_conf":      round(r["filtered_conf"], 4),
                })

            if verbose:
                wrong = " <--" if (is_clean_label and r["is_attack"]) or \
                                  (not is_clean_label and not r["is_attack"]) else ""
                print(f"{f.name:<28} {label_str:>7} "
                      f"{r['kl_divergence']:>7.3f} "
                      f"{str(r['class_changed']):>8} "
                      f"{verdict:>8}{wrong}")

    evaluate_set(clean_files,   is_clean_label=True)
    evaluate_set(patched_files, is_clean_label=False)

    total     = TP + TN + FP + FN
    tpr       = TP / max(1, TP + FN)
    fpr       = FP / max(1, FP + TN)
    accuracy  = (TP + TN) / max(1, total)
    clean_acc = TN / max(1, TN + FP)     # accuracy on clean images only
    patch_acc = TP / max(1, TP + FN)     # accuracy on patched images only

    rule = f"class_changed AND KL > {kl_threshold}" if require_class_change \
           else f"class_same AND KL > {kl_threshold}"

    print(f"\n{'='*55}")
    print(f"  Median Filter Defense Results")
    print(f"{'='*55}")
    print(f"  Tested : {len(clean_files)} clean + {len(patched_files)} patched = {total} total")
    print(f"  Rule   : {rule}")
    print(f"")
    print(f"  On CLEAN images  ({len(clean_files)} total):")
    print(f"    Correct (CLEAN)  : {TN}  → {clean_acc*100:.1f}% clean accuracy")
    print(f"    Wrong   (ATTACK) : {FP}  ← false positives")
    print(f"")
    print(f"  On PATCHED images ({len(patched_files)} total):")
    print(f"    Correct (ATTACK) : {TP}  → {patch_acc*100:.1f}% detection rate")
    print(f"    Missed  (CLEAN)  : {FN}  ← missed attacks")
    print(f"")
    print(f"  Overall accuracy : {accuracy*100:.1f}%  (both sets combined)")
    print(f"  FPR              : {fpr*100:.1f}%  (false alarm rate on clean images)")
    print(f"{'='*55}")

    result = {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
              "tpr": tpr, "fpr": fpr, "accuracy": accuracy}
    if collect_records:
        result["records"] = records
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Median Filter defense")
    parser.add_argument("--clean",   type=str, default=None,
                        help="Path to clean images folder (default: data/cifar10/clean)")
    parser.add_argument("--patched", type=str, default=None,
                        help="Path to patched images folder (default: data/fooled)")
    parser.add_argument("--kernel",  type=int, default=3)
    parser.add_argument("--kl-thr",  type=float, default=0.5)
    parser.add_argument("--require-class-change", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    clean_dir   = Path(args.clean)   if args.clean   else ROOT / "data" / "cifar10" / "clean"
    patched_dir = Path(args.patched) if args.patched else ROOT / "data" / "fooled"

    run_evaluation(
        clean_dir, patched_dir,
        kernel_size=args.kernel,
        kl_threshold=args.kl_thr,
        require_class_change=args.require_class_change,
        verbose=args.verbose,
    )
