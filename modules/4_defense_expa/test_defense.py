"""Module 4 — EXPA (EigenCAM) defense evaluation.

Loads images from fooled_images/clean/ and fooled_images/patched/,
runs the EXPA defense on each, then reports TPR / FPR / accuracy.

Usage (from adversarial_testing/ root):
    python modules/4_defense_expa/test_defense.py --data data/fooled_images
    python modules/4_defense_expa/test_defense.py --data data/fooled_images --verbose
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# Standalone — import from same folder
THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
from eigencam import eigencam
from detector import AnomalyDetector, DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parent.parent.parent

MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2470, 0.2435, 0.2616]

# ResNet-20 layers to hook
EARLY_LAYER = "layer1"
LATE_LAYER  = "layer3"


# ── Model + hooks ──────────────────────────────────────────────────────────────

class CIFARModelWithHooks:
    """ResNet-20 with activation hooks for EigenCAM."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self._activations = {}
        self._hooks       = []

        print("Loading CIFAR-10 ResNet-20 ...")
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_resnet20",
            pretrained=True,
            verbose=False,
        ).to(self.device)
        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        def make_hook(name):
            def hook(module, input, output):
                self._activations[name] = output.detach()
            return hook

        for name, layer_name in [("early", EARLY_LAYER), ("late", LATE_LAYER)]:
            layer = dict(self.model.named_children())[layer_name]
            h = layer.register_forward_hook(make_hook(name))
            self._hooks.append(h)

    def forward(self, tensor):
        """Run model. Returns (probs, activations_dict).
        tensor: (B, 3, H, W) in [0, 1]
        """
        m = torch.tensor(MEAN, dtype=tensor.dtype).view(1, 3, 1, 1).to(self.device)
        s = torch.tensor(STD,  dtype=tensor.dtype).view(1, 3, 1, 1).to(self.device)
        normalized = (tensor.to(self.device) - m) / s
        with torch.no_grad():
            logits = self.model(normalized)
        probs = torch.softmax(logits, dim=1)
        return probs, dict(self._activations)

    def cleanup(self):
        for h in self._hooks:
            h.remove()


# ── Masking confirmation ───────────────────────────────────────────────────────

def mask_and_check(model_with_hooks, image, heatmap, pred_class, original_conf,
                   topk_percent=0.05):
    """Mask the hotspot and measure confidence drop.

    Large drop (>threshold) → patch is responsible → real attack.
    Small drop → natural object → false positive → clear it.
    """
    h, w = image.shape[1], image.shape[2]
    hm = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0).float(),
        size=(h, w), mode="bilinear", align_corners=False
    ).squeeze()

    threshold   = torch.quantile(hm.flatten(), 1.0 - topk_percent)
    mask        = (hm >= threshold).float()
    masked      = image.clone()
    for c in range(3):
        ch_mean = image[c].mean()
        masked[c] = image[c] * (1 - mask) + ch_mean * mask

    probs, _ = model_with_hooks.forward(masked.unsqueeze(0))
    masked_conf = float(probs[0, pred_class])
    return original_conf - masked_conf


# ── EXPA defense run ───────────────────────────────────────────────────────────

def run_expa(image_tensor, model_with_hooks, detector, mask_threshold):
    """Run full EXPA pipeline on a single (3,H,W) image tensor in [0,1].

    Returns:
        is_adversarial: bool
        scores: dict
    """
    probs, acts = model_with_hooks.forward(image_tensor.unsqueeze(0))

    heatmap_late  = eigencam(acts["late"])[0]   # (H, W)
    heatmap_early = eigencam(acts["early"])[0]  # (H, W)

    is_adv, scores = detector.detect(heatmap_late, heatmap_early)

    # Masking confirmation: re-check flagged images
    if is_adv:
        pred_class    = int(probs[0].argmax())
        original_conf = float(probs[0].max())
        drop = mask_and_check(
            model_with_hooks, image_tensor, heatmap_late,
            pred_class, original_conf
        )
        scores["mask_conf_drop"] = drop
        if drop < mask_threshold:
            is_adv = False   # hotspot not responsible → natural object
    else:
        scores["mask_conf_drop"] = None

    return is_adv, scores


# ── Public evaluation function (called by run_test_defense1.py) ────────────────

def run_evaluation(clean_dir, patched_dir, config: dict = None, verbose=False,
                   collect_records=False):
    """Evaluate EXPA defense.

    Args:
        clean_dir:   Path to folder of clean PNG images
        patched_dir: Path to folder of patched PNG images
        config:      Optional detector config (defaults from DEFAULT_CONFIG)
        verbose:     If True, print every image result

    Returns:
        dict with TP, TN, FP, FN, tpr, fpr, accuracy
    """
    clean_dir   = Path(clean_dir)
    patched_dir = Path(patched_dir)

    if not clean_dir.exists() or not patched_dir.exists():
        print(f"clean_dir or patched_dir not found.")
        return {}

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    mask_thr = cfg["mask_confidence_drop"]
    detector = AnomalyDetector(cfg)
    model    = CIFARModelWithHooks(device="cpu")
    to_tensor = T.ToTensor()

    patched_files = sorted(f for f in patched_dir.iterdir() if f.suffix.lower() == ".png")
    clean_files   = sorted(f for f in (clean_dir / n.name for n in patched_files)
                           if f.exists())

    print(f"\nEXPA Defense Evaluation")
    print(f"{'='*70}")
    print(f"Clean dir:   {clean_dir}")
    print(f"Patched dir: {patched_dir}")
    print(f"Clean:    {len(clean_files)} images")
    print(f"Patched:  {len(patched_files)} images")
    print(f"Detector: mode={cfg['detection_mode']}, mask_thr={mask_thr}")
    print(f"{'='*70}")

    if verbose:
        print(f"\n{'Image':<28} {'Label':>7} {'Entropy':>7} {'Peak/M':>7} "
              f"{'Top-k':>6} {'Cross':>6} {'Flags':>5} {'Result':>8}")
        print("-" * 78)

    TP = TN = FP = FN = 0
    records = []

    def evaluate_set(files, is_clean_label):
        nonlocal TP, TN, FP, FN
        label_str = "CLEAN" if is_clean_label else "PATCH"
        for f in files:
            img = to_tensor(Image.open(f).convert("RGB"))
            is_adv, scores = run_expa(img, model, detector, mask_thr)
            verdict = "ATTACK" if is_adv else "CLEAN"

            if is_clean_label and not is_adv:
                TN += 1
            elif is_clean_label and is_adv:
                FP += 1
            elif not is_clean_label and is_adv:
                TP += 1
            else:
                FN += 1

            if collect_records:
                records.append({
                    "filename":          f.name,
                    "true_label":        label_str,
                    "prediction":        verdict,
                    "correct":           (is_clean_label and not is_adv) or
                                         (not is_clean_label and is_adv),
                    "entropy":           round(scores.get("entropy", 0), 4),
                    "peak_mean":         round(scores.get("peak_mean", 0), 4),
                    "topk_energy":       round(scores.get("topk_energy", 0), 4),
                    "cross_layer_corr":  round(scores.get("cross_layer_corr") or 0, 4),
                    "num_flags":         scores.get("num_flags", 0),
                    "entropy_flag":      scores.get("entropy_flag", False),
                    "peak_mean_flag":    scores.get("peak_mean_flag", False),
                    "topk_energy_flag":  scores.get("topk_energy_flag", False),
                    "cross_layer_flag":  scores.get("cross_layer_flag", False),
                    "mask_conf_drop":    round(scores.get("mask_conf_drop") or 0, 4),
                })

            if verbose:
                flags = ""
                flags += "E" if scores.get("entropy_flag")    else "."
                flags += "P" if scores.get("peak_mean_flag")  else "."
                flags += "T" if scores.get("topk_energy_flag")else "."
                flags += "C" if scores.get("cross_layer_flag")else "."
                wrong = " <--" if (is_clean_label and is_adv) or \
                                  (not is_clean_label and not is_adv) else ""
                print(f"{f.name:<28} {label_str:>7} "
                      f"{scores.get('entropy', 0):>7.2f} "
                      f"{scores.get('peak_mean', 0):>7.2f} "
                      f"{scores.get('topk_energy', 0):>6.3f} "
                      f"{scores.get('cross_layer_corr') or 0:>6.3f} "
                      f"{flags:>5} {verdict:>8}{wrong}")

    evaluate_set(clean_files,   is_clean_label=True)
    evaluate_set(patched_files, is_clean_label=False)

    model.cleanup()

    total        = TP + TN + FP + FN
    tpr          = TP / max(1, TP + FN)
    fpr          = FP / max(1, FP + TN)
    accuracy     = (TP + TN) / max(1, total)
    clean_acc    = TN / max(1, TN + FP)      # accuracy on clean images only
    patch_acc    = TP / max(1, TP + FN)      # accuracy on patched images only

    print(f"\n{'='*55}")
    print(f"  EXPA Defense Results")
    print(f"{'='*55}")
    print(f"  Tested : {len(clean_files)} clean + {len(patched_files)} patched = {total} total")
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
    parser = argparse.ArgumentParser(description="Test EXPA defense")
    parser.add_argument("--clean",   type=str, default=None,
                        help="Path to clean images folder (default: data/cifar10/clean)")
    parser.add_argument("--patched", type=str, default=None,
                        help="Path to patched images folder (default: data/fooled)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--entropy-thr",    type=float, default=None)
    parser.add_argument("--peak-thr",       type=float, default=None)
    parser.add_argument("--topk-thr",       type=float, default=None)
    parser.add_argument("--cross-thr",      type=float, default=None)
    parser.add_argument("--mask-thr",       type=float, default=None)
    parser.add_argument("--mode",           type=str,   default=None,
                        choices=["vote", "any"])
    args = parser.parse_args()

    clean_dir   = Path(args.clean)   if args.clean   else ROOT / "data" / "cifar10" / "clean"
    patched_dir = Path(args.patched) if args.patched else ROOT / "data" / "fooled"

    override = {}
    if args.entropy_thr is not None: override["entropy_threshold"]    = args.entropy_thr
    if args.peak_thr    is not None: override["peak_mean_threshold"]  = args.peak_thr
    if args.topk_thr    is not None: override["topk_ratio_threshold"] = args.topk_thr
    if args.cross_thr   is not None: override["cross_layer_threshold"]= args.cross_thr
    if args.mask_thr    is not None: override["mask_confidence_drop"] = args.mask_thr
    if args.mode        is not None: override["detection_mode"]       = args.mode

    run_evaluation(clean_dir, patched_dir, config=override, verbose=args.verbose)
