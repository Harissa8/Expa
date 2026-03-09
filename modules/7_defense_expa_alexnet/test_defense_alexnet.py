"""Module 7 — EXPA defense for AlexNet on 224×224 images.

AlexNet layer hooks:
  Early: features.3  → 192×27×27  (much larger than ResNet-20's 16×32×32)
  Late:  features.10 → 256×13×13  (vs ResNet-20's 64×8×8)

With larger feature maps, EigenCAM can clearly detect spatial concentration
from adversarial patches → EXPA works much better at 224×224.

Usage (from adversarial_testing/ root):
    python modules/7_defense_expa_alexnet/test_defense_alexnet.py
    python modules/7_defense_expa_alexnet/test_defense_alexnet.py --verbose
"""

import sys
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image

THIS_DIR = Path(__file__).parent
ROOT     = Path(__file__).resolve().parent.parent.parent

# Reuse EigenCAM and AnomalyDetector from module 4
sys.path.insert(0, str(ROOT / "modules" / "4_defense_expa"))
from eigencam  import eigencam
from detector  import AnomalyDetector

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

EARLY_LAYER = "features.3"   # 192×27×27
LATE_LAYER  = "features.10"  # 256×13×13

DEFAULT_CONFIG = {
    "entropy_threshold":    5.5,
    "peak_mean_threshold":  3.0,
    "topk_ratio_threshold": 0.30,
    "topk_percent":         0.05,
    "cross_layer_threshold": 0.5,
    "mask_confidence_drop":  0.10,
    "detection_mode":        "vote",
}


# ── Model with hooks ───────────────────────────────────────────────────────────

class AlexNetWithHooks:
    def __init__(self):
        self._activations = {}
        self._hooks       = []
        print("Loading AlexNet (pretrained on ImageNet) ...")
        self.model = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        )
        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        def make_hook(name):
            def hook(module, input, output):
                self._activations[name] = output.detach()
            return hook

        # Navigate features.3 and features.10
        for tag, layer_str in [("early", EARLY_LAYER), ("late", LATE_LAYER)]:
            parts  = layer_str.split(".")
            module = self.model
            for p in parts:
                module = module[int(p)] if p.isdigit() else getattr(module, p)
            self._hooks.append(module.register_forward_hook(make_hook(tag)))

    def forward(self, tensor):
        m = torch.tensor(MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
        s = torch.tensor(STD,  dtype=tensor.dtype).view(1, 3, 1, 1)
        with torch.no_grad():
            logits = self.model((tensor - m) / s)
        probs = torch.softmax(logits, dim=1)
        return probs, dict(self._activations)

    def cleanup(self):
        for h in self._hooks:
            h.remove()


# ── Masking confirmation ───────────────────────────────────────────────────────

def mask_and_check(model, image, heatmap, pred_class, original_conf, topk_percent=0.05):
    h, w = image.shape[1], image.shape[2]
    hm   = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0).float(),
        size=(h, w), mode="bilinear", align_corners=False
    ).squeeze()
    threshold = torch.quantile(hm.flatten(), 1.0 - topk_percent)
    mask      = (hm >= threshold).float()
    masked    = image.clone()
    for c in range(3):
        masked[c] = image[c] * (1 - mask) + image[c].mean() * mask
    probs, _ = model.forward(masked.unsqueeze(0))
    return original_conf - float(probs[0, pred_class])


# ── Single image EXPA run ──────────────────────────────────────────────────────

def run_expa(image_tensor, model, detector, mask_threshold):
    probs, acts = model.forward(image_tensor.unsqueeze(0))
    heatmap_late  = eigencam(acts["late"])[0]
    heatmap_early = eigencam(acts["early"])[0]
    is_adv, scores = detector.detect(heatmap_late, heatmap_early)

    if is_adv:
        pred_class    = int(probs[0].argmax())
        original_conf = float(probs[0].max())
        drop = mask_and_check(model, image_tensor, heatmap_late,
                              pred_class, original_conf)
        scores["mask_conf_drop"] = drop
        if drop < mask_threshold:
            is_adv = False
    else:
        scores["mask_conf_drop"] = None

    return is_adv, scores


# ── Evaluation ─────────────────────────────────────────────────────────────────

def run_evaluation(clean_dir, patched_dir, config=None, verbose=False,
                   collect_records=False, max_images=None):
    clean_dir   = Path(clean_dir)
    patched_dir = Path(patched_dir)

    if not clean_dir.exists() or not patched_dir.exists():
        print("clean_dir or patched_dir not found.")
        return {}

    cfg      = {**DEFAULT_CONFIG, **(config or {})}
    mask_thr = cfg["mask_confidence_drop"]
    detector = AnomalyDetector(cfg)
    model    = AlexNetWithHooks()
    to_tensor = T.ToTensor()

    patched_files = sorted(f for f in patched_dir.iterdir() if f.suffix.lower() == ".png")
    clean_files   = sorted(f for f in (clean_dir / n.name for n in patched_files) if f.exists())
    if max_images and len(patched_files) > max_images:
        patched_files = patched_files[:max_images]
        clean_files   = clean_files[:max_images]

    print(f"\nEXPA Defense Evaluation (AlexNet 224×224)")
    print(f"{'='*65}")
    print(f"Clean dir:   {clean_dir}  ({len(clean_files)} images)")
    print(f"Patched dir: {patched_dir}  ({len(patched_files)} images)")
    print(f"Layers:      {EARLY_LAYER} (early), {LATE_LAYER} (late)")
    print(f"Detector:    mode={cfg['detection_mode']}, mask_thr={mask_thr}")
    print(f"{'='*65}")

    TP = TN = FP = FN = 0
    records = []

    def evaluate_set(files, is_clean_label):
        nonlocal TP, TN, FP, FN
        label_str = "CLEAN" if is_clean_label else "PATCH"
        for f in files:
            img = to_tensor(Image.open(f).convert("RGB"))
            is_adv, scores = run_expa(img, model, detector, mask_thr)
            verdict = "ATTACK" if is_adv else "CLEAN"

            if   is_clean_label and not is_adv: TN += 1
            elif is_clean_label and     is_adv: FP += 1
            elif not is_clean_label and is_adv: TP += 1
            else:                               FN += 1

            if collect_records:
                records.append({
                    "filename":         f.name,
                    "true_label":       label_str,
                    "prediction":       verdict,
                    "correct":          (is_clean_label and not is_adv) or
                                        (not is_clean_label and is_adv),
                    "entropy":          round(scores.get("entropy", 0), 4),
                    "peak_mean":        round(scores.get("peak_mean", 0), 4),
                    "topk_energy":      round(scores.get("topk_energy", 0), 4),
                    "cross_layer_corr": round(scores.get("cross_layer_corr") or 0, 4),
                    "num_flags":        scores.get("num_flags", 0),
                    "mask_conf_drop":   round(scores.get("mask_conf_drop") or 0, 4),
                })

            if verbose:
                wrong = " <--" if (is_clean_label and is_adv) or \
                                  (not is_clean_label and not is_adv) else ""
                print(f"{f.name:<30} {label_str:>7} "
                      f"E={scores.get('entropy',0):>5.2f} "
                      f"P={scores.get('peak_mean',0):>5.2f} "
                      f"flags={scores.get('num_flags',0)} "
                      f"{verdict:>8}{wrong}")

    evaluate_set(clean_files,   is_clean_label=True)
    evaluate_set(patched_files, is_clean_label=False)
    model.cleanup()

    total     = TP + TN + FP + FN
    tpr       = TP / max(1, TP + FN)
    fpr       = FP / max(1, FP + TN)
    accuracy  = (TP + TN) / max(1, total)
    clean_acc = TN / max(1, TN + FP)
    patch_acc = TP / max(1, TP + FN)

    print(f"\n{'='*55}")
    print(f"  EXPA Defense Results (AlexNet)")
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    clean_dir   = Path(args.clean)   if args.clean   else ROOT / "data2" / "stl10" / "clean"
    patched_dir = Path(args.patched) if args.patched else ROOT / "data2" / "fooled"

    run_evaluation(clean_dir, patched_dir, verbose=args.verbose)
