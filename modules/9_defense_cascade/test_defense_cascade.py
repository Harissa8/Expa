"""Module 9 — Cascade defense: Median Filter → EXPA (AlexNet 224×224).

Pipeline:
  Stage 1 — Median Filter (fast, runs on ALL images)
    → CLEAN:      accept immediately, EXPA skipped
    → SUSPICIOUS: forward to Stage 2

  Stage 2 — EXPA/EigenCAM (runs ONLY on Median-flagged images)
    → Confirms or rejects using spatial anomaly metrics
    → Final ATTACK = Median flagged AND EXPA confirmed

Advantages vs running each independently:
  - Lower FPR: two confirmations required, not one
  - Faster: EXPA only runs on ~10–30% of images
  - Single clean output per image: ATTACK or CLEAN

Metrics reported:
  - Stage 1 flags   : how many images Median Filter flagged
  - Stage 2 confirms: how many EXPA confirmed out of those
  - Final TPR / FPR / Accuracy

Usage (from adversarial_testing/ root):
    python modules/9_defense_cascade/test_defense_cascade.py
    python modules/9_defense_cascade/test_defense_cascade.py --patch-size 48
    python modules/9_defense_cascade/test_defense_cascade.py --verbose
    python modules/9_defense_cascade/test_defense_cascade.py --data-root data3/imagenette
"""

import sys
import argparse
import importlib.util
from pathlib import Path

import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
MOD4 = ROOT / "modules" / "4_defense_expa"
MOD7 = ROOT / "modules" / "7_defense_expa_alexnet"
MOD8 = ROOT / "modules" / "8_defense_median_alexnet"

# ── Load modules 7 and 8 without name collision ─────────────────────────────

def _load(path, name):
    spec   = importlib.util.spec_from_file_location(name, path)
    mod    = importlib.util.module_from_spec(spec)
    folder = str(Path(path).parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec.loader.exec_module(mod)
    return mod

_mod8 = _load(MOD8 / "test_defense_alexnet.py", "median_cascade")
_mod7 = _load(MOD7 / "test_defense_alexnet.py", "expa_cascade")

# AnomalyDetector from module 4
if str(MOD4) not in sys.path:
    sys.path.insert(0, str(MOD4))
from detector import AnomalyDetector

EXPA_CONFIG = {
    "entropy_threshold":     5.5,
    "peak_mean_threshold":   3.0,
    "topk_ratio_threshold":  0.30,
    "topk_percent":          0.05,
    "cross_layer_threshold": 0.5,
    "mask_confidence_drop":  0.10,
    "detection_mode":        "vote",
}


# ── Per-image cascade ────────────────────────────────────────────────────────

def cascade_detect(img_tensor, median_model, expa_model, detector,
                   kernel_size=5, kl_threshold=1.0, mask_thr=0.10):
    """Run Median Filter then EXPA (only if Median flags).

    Returns:
        dict with keys: final_attack, median_flagged, expa_ran, expa_confirmed,
                        kl_divergence, class_changed, expa_scores
    """
    # Stage 1 — Median Filter
    med = _mod8.detect_median(median_model, img_tensor, kernel_size, kl_threshold)
    median_flagged = med["is_attack"]

    expa_ran       = False
    expa_confirmed = False
    expa_scores    = {}

    # Stage 2 — EXPA (only if Median flagged)
    if median_flagged:
        expa_ran = True
        expa_confirmed, expa_scores = _mod7.run_expa(
            img_tensor, expa_model, detector, mask_thr
        )

    final_attack = median_flagged and expa_confirmed

    return {
        "final_attack":    final_attack,
        "median_flagged":  median_flagged,
        "expa_ran":        expa_ran,
        "expa_confirmed":  expa_confirmed,
        "kl_divergence":   med["kl_divergence"],
        "class_changed":   med["class_changed"],
        "expa_scores":     expa_scores,
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def run_evaluation(clean_dir, patched_dir, kernel_size=5, kl_threshold=1.0,
                   expa_config=None, verbose=False, collect_records=False,
                   max_images=None):
    clean_dir   = Path(clean_dir)
    patched_dir = Path(patched_dir)

    if not clean_dir.exists() or not patched_dir.exists():
        print("clean_dir or patched_dir not found.")
        return {}

    cfg          = {**EXPA_CONFIG, **(expa_config or {})}
    mask_thr     = cfg["mask_confidence_drop"]
    detector     = AnomalyDetector(cfg)
    median_model = _mod8.load_alexnet()
    expa_model   = _mod7.AlexNetWithHooks()
    to_tensor    = T.ToTensor()

    patched_files = sorted(f for f in patched_dir.iterdir() if f.suffix.lower() == ".png")
    clean_files   = sorted(f for f in (clean_dir / n.name for n in patched_files) if f.exists())
    if max_images and len(patched_files) > max_images:
        patched_files = patched_files[:max_images]
        clean_files   = clean_files[:max_images]

    print(f"\nCascade Defense — Median Filter → EXPA (AlexNet 224×224)")
    print(f"{'='*65}")
    print(f"Clean dir:    {clean_dir}  ({len(clean_files)} images)")
    print(f"Patched dir:  {patched_dir}  ({len(patched_files)} images)")
    print(f"Stage 1:      Median Filter {kernel_size}×{kernel_size},  KL > {kl_threshold}")
    print(f"Stage 2:      EXPA  mode={cfg['detection_mode']},  mask_thr={mask_thr}")
    print(f"Rule:         ATTACK = Stage1 flagged AND Stage2 confirmed")
    print(f"{'='*65}")

    TP = TN = FP = FN = 0
    total_median_flagged  = 0
    total_expa_confirmed  = 0
    records = []

    def evaluate_set(files, is_clean_label):
        nonlocal TP, TN, FP, FN, total_median_flagged, total_expa_confirmed
        label_str = "CLEAN" if is_clean_label else "PATCH"

        for f in files:
            img = to_tensor(Image.open(f).convert("RGB"))
            r   = cascade_detect(img, median_model, expa_model, detector,
                                  kernel_size, kl_threshold, mask_thr)

            if r["median_flagged"]: total_median_flagged += 1
            if r["expa_confirmed"]: total_expa_confirmed += 1

            is_attack = r["final_attack"]
            verdict   = "ATTACK" if is_attack else "CLEAN"

            if   is_clean_label and not is_attack: TN += 1
            elif is_clean_label and     is_attack: FP += 1
            elif not is_clean_label and is_attack: TP += 1
            else:                                  FN += 1

            if collect_records:
                sc = r["expa_scores"]
                records.append({
                    "filename":        f.name,
                    "true_label":      label_str,
                    "prediction":      verdict,
                    "correct":         (is_clean_label and not is_attack) or
                                       (not is_clean_label and is_attack),
                    "median_flagged":  r["median_flagged"],
                    "expa_ran":        r["expa_ran"],
                    "expa_confirmed":  r["expa_confirmed"],
                    "kl_divergence":   round(r["kl_divergence"], 4),
                    "class_changed":   r["class_changed"],
                    "entropy":         round(sc.get("entropy", 0), 4),
                    "peak_mean":       round(sc.get("peak_mean", 0), 4),
                    "topk_energy":     round(sc.get("topk_energy", 0), 4),
                    "cross_layer_corr":round(sc.get("cross_layer_corr") or 0, 4),
                    "num_flags":       sc.get("num_flags", 0),
                    "mask_conf_drop":  round(sc.get("mask_conf_drop") or 0, 4),
                })

            if verbose:
                wrong = " <--" if (is_clean_label and is_attack) or \
                                  (not is_clean_label and not is_attack) else ""
                stage2 = f"EXPA={'YES' if r['expa_confirmed'] else 'no '}" \
                         if r["expa_ran"] else "EXPA=skip"
                print(f"{f.name:<30} {label_str:>7} "
                      f"KL={r['kl_divergence']:>6.3f} "
                      f"Med={'FLAG' if r['median_flagged'] else 'ok  '} "
                      f"{stage2} → {verdict:>8}{wrong}")

    evaluate_set(clean_files,   is_clean_label=True)
    evaluate_set(patched_files, is_clean_label=False)
    expa_model.cleanup()

    total     = TP + TN + FP + FN
    tpr       = TP / max(1, TP + FN)
    fpr       = FP / max(1, FP + TN)
    accuracy  = (TP + TN) / max(1, total)
    clean_acc = TN / max(1, TN + FP)
    patch_acc = TP / max(1, TP + FN)

    n_clean   = len(clean_files)
    n_patched = len(patched_files)

    print(f"\n{'='*65}")
    print(f"  Cascade Defense Results (AlexNet 224×224)")
    print(f"{'='*65}")
    print(f"  Stage 1 — Median Filter flagged : {total_median_flagged} / {total} images")
    print(f"  Stage 2 — EXPA confirmed        : {total_expa_confirmed} / {total_median_flagged} flagged")
    print(f"  {'─'*55}")
    print(f"  On CLEAN images  ({n_clean} total):")
    print(f"    Correct (CLEAN)  : {TN}  → {clean_acc*100:.1f}%")
    print(f"    Wrong   (ATTACK) : {FP}  ← false positives")
    print(f"  On PATCHED images ({n_patched} total):")
    print(f"    Correct (ATTACK) : {TP}  → {patch_acc*100:.1f}% detection rate")
    print(f"    Missed  (CLEAN)  : {FN}  ← missed attacks")
    print(f"  {'─'*55}")
    print(f"  Overall accuracy : {accuracy*100:.1f}%")
    print(f"  TPR              : {tpr*100:.1f}%")
    print(f"  FPR              : {fpr*100:.1f}%")
    print(f"{'='*65}")

    result = {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "tpr": tpr, "fpr": fpr, "accuracy": accuracy,
        "median_flagged": total_median_flagged,
        "expa_confirmed": total_expa_confirmed,
    }
    if collect_records:
        result["records"] = records
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cascade defense: Median → EXPA")
    parser.add_argument("--clean",      type=str,   default=None)
    parser.add_argument("--patched",    type=str,   default=None)
    parser.add_argument("--patch-size", type=int,   default=28)
    parser.add_argument("--data-root",  type=str,   default="data2/stl10")
    parser.add_argument("--kernel",     type=int,   default=5)
    parser.add_argument("--kl-thr",     type=float, default=1.0)
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    ps        = args.patch_size
    data_root = ROOT / args.data_root
    clean_dir   = Path(args.clean)   if args.clean   else data_root / "clean"
    patched_dir = Path(args.patched) if args.patched else data_root / f"patched_{ps}" / "fooled"

    run_evaluation(clean_dir, patched_dir,
                   kernel_size=args.kernel,
                   kl_threshold=args.kl_thr,
                   verbose=args.verbose)


if __name__ == "__main__":
    main()
