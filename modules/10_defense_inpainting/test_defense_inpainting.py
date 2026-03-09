"""Module 10 - Patch Localization + Inpainting defense (AlexNet 224x224).

Full end-to-end adversarial patch defense:
  1. Detect attack using Cascade (Median Filter -> EXPA)
  2. Localize patch region using EigenCAM heatmap (late layer: features.10)
  3. Inpaint the detected region using surrounding pixel median fill
  4. Re-run AlexNet on inpainted image to check if class is recovered

Metrics:
  Detection rate  : cascade correctly flags attacks           (TPR)
  FPR             : clean images wrongly flagged as attack
  Localize rate   : of detected, how many were localized
  Recovery rate   : of detected, target class removed after inpainting
  Class match     : recovered prediction matches original clean prediction
  End-to-end      : overall fraction of fooled images successfully recovered

Usage (from adversarial_testing/ root):
    python modules/10_defense_inpainting/test_defense_inpainting.py
    python modules/10_defense_inpainting/test_defense_inpainting.py --patch-size 48
    python modules/10_defense_inpainting/test_defense_inpainting.py --data-root data3/imagenette
    python modules/10_defense_inpainting/test_defense_inpainting.py --verbose
"""

import sys
import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent.parent
MOD4 = ROOT / "modules" / "4_defense_expa"
MOD7 = ROOT / "modules" / "7_defense_expa_alexnet"
MOD8 = ROOT / "modules" / "8_defense_median_alexnet"

TARGET_CLASS = 954   # banana - default; overridden by --target-class CLI arg

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

EXPA_CONFIG = {
    "entropy_threshold":     5.5,
    "peak_mean_threshold":   3.0,
    "topk_ratio_threshold":  0.30,
    "topk_percent":          0.05,
    "cross_layer_threshold": 0.5,
    "mask_confidence_drop":  0.10,
    "detection_mode":        "vote",
}


# -- Load sub-modules ----------------------------------------------------------

def _load(path, name):
    spec   = importlib.util.spec_from_file_location(name, path)
    mod    = importlib.util.module_from_spec(spec)
    folder = str(Path(path).parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec.loader.exec_module(mod)
    return mod

_mod8 = _load(MOD8 / "test_defense_alexnet.py", "median_inp")
_mod7 = _load(MOD7 / "test_defense_alexnet.py", "expa_inp")

if str(MOD4) not in sys.path:
    sys.path.insert(0, str(MOD4))
from detector import AnomalyDetector
from eigencam import eigencam


# -- Helpers -------------------------------------------------------------------

def _normalize(tensor):
    m = torch.tensor(MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
    s = torch.tensor(STD,  dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - m) / s


def classify(model, img_tensor):
    """Run AlexNet and return predicted class index."""
    with torch.no_grad():
        logits = model(_normalize(img_tensor.unsqueeze(0)))
    return int(logits.argmax(dim=1).item())


# -- Localization --------------------------------------------------------------

def localize_from_heatmap(heatmap, img_size=224, topk_percent=0.10, pad=8):
    """Convert EigenCAM heatmap to a patch bounding box.

    Upscales the heatmap to img_size, thresholds the top-k% pixels,
    and returns the bounding box (y1, x1, y2, x2) padded by `pad` pixels.
    Returns None if no high-activation region is found.
    """
    hm = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0).float(),
        size=(img_size, img_size), mode="bilinear", align_corners=False
    ).squeeze()

    threshold = torch.quantile(hm.flatten(), 1.0 - topk_percent)
    mask = (hm >= threshold)

    rows = torch.where(mask.any(dim=1))[0]
    cols = torch.where(mask.any(dim=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return None

    y1 = max(0,        rows[0].item()  - pad)
    y2 = min(img_size, rows[-1].item() + pad + 1)
    x1 = max(0,        cols[0].item()  - pad)
    x2 = min(img_size, cols[-1].item() + pad + 1)

    return y1, x1, y2, x2


# -- Inpainting ----------------------------------------------------------------

def inpaint_region(image, y1, x1, y2, x2, method="smooth", border=12, inpaint_filter=5):
    """Fill patch bounding box to remove adversarial content.

    method='smooth' (default):
        Apply a median filter of size `inpaint_filter` to the full image and
        paste the filtered pixels back into the patch region.

    method='flat':
        Fill with median color of the surrounding border ring (original method).
        Fast but leaves a visible flat square.

    Args:
        image          : (3, H, W) float tensor [0, 1]
        y1, x1, y2, x2 : bounding box of patch region
        method         : 'smooth' or 'flat'
        border         : (flat only) width of surrounding ring in pixels
        inpaint_filter : (smooth only) MedianFilter kernel size (default 5, try 9 or 11)

    Returns:
        inpainted image tensor (3, H, W)
    """
    to_tensor = T.ToTensor()
    result    = image.clone()

    if method == "smooth":
        # Convert to PIL, apply median filter, paste only patch region back
        img_uint8    = (image.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
        img_pil      = Image.fromarray(img_uint8)
        filtered_pil = img_pil.filter(ImageFilter.MedianFilter(size=inpaint_filter))
        filtered     = to_tensor(filtered_pil)
        result[:, y1:y2, x1:x2] = filtered[:, y1:y2, x1:x2]

    else:  # flat - original surrounding median fill
        H, W = image.shape[1], image.shape[2]
        sy1  = max(0, y1 - border)
        sy2  = min(H, y2 + border)
        sx1  = max(0, x1 - border)
        sx2  = min(W, x2 + border)

        for c in range(3):
            region = image[c, sy1:sy2, sx1:sx2]
            inner  = torch.zeros(region.shape, dtype=torch.bool)
            inner[y1 - sy1 : y2 - sy1, x1 - sx1 : x2 - sx1] = True
            surrounding = region[~inner]
            fill = surrounding.median() if surrounding.numel() > 0 else image[c].mean()
            result[c, y1:y2, x1:x2] = fill

    return result


# -- Per-image pipeline --------------------------------------------------------

def process_image(img_tensor, clean_tensor,
                  median_model, expa_model, detector,
                  kernel_size=5, kl_threshold=1.0,
                  mask_thr=0.10, topk_percent=0.10,
                  inpaint_method="smooth", inpaint_filter=5):
    """Full Module 10 pipeline for one fooled image.

    Returns dict:
        detected       : bool - cascade flagged as attack
        localized      : bool - patch region found from heatmap
        bbox           : (y1, x1, y2, x2) or None
        pred_clean     : AlexNet class on clean image
        pred_patched   : AlexNet class on patched image (should be TARGET_CLASS)
        pred_recovered : AlexNet class after inpainting (None if not detected)
        recovered      : bool - pred_recovered != TARGET_CLASS
        class_match    : bool - pred_recovered == pred_clean
    """
    r = {
        "detected":        False,
        "median_flagged":  False,
        "localized":       False,
        "bbox":            None,
        "pred_clean":      classify(median_model, clean_tensor),
        "pred_patched":    classify(median_model, img_tensor),
        "pred_recovered":  None,
        "recovered":       False,
        "class_match":     False,
    }

    # Stage 1 - Median Filter
    med = _mod8.detect_median(median_model, img_tensor, kernel_size, kl_threshold)
    if not med["is_attack"]:
        return r
    r["median_flagged"] = True

    # Stage 2 - EXPA (also computes heatmap we need for localization)
    probs, acts   = expa_model.forward(img_tensor.unsqueeze(0))
    heatmap_late  = eigencam(acts["late"])[0]
    heatmap_early = eigencam(acts["early"])[0]
    is_adv, scores = detector.detect(heatmap_late, heatmap_early)

    if is_adv:
        pred_class    = int(probs[0].argmax())
        original_conf = float(probs[0].max())
        drop = _mod7.mask_and_check(expa_model, img_tensor, heatmap_late,
                                    pred_class, original_conf)
        if drop < mask_thr:
            is_adv = False

    if not is_adv:
        return r  # Cascade did not confirm -> not detected

    r["detected"] = True

    # Stage 3 - Localize from EigenCAM heatmap
    bbox = localize_from_heatmap(heatmap_late, img_size=224,
                                 topk_percent=topk_percent)
    if bbox is None:
        return r

    r["localized"] = True
    r["bbox"]      = bbox
    y1, x1, y2, x2 = bbox

    # Stage 4 - Inpaint + re-classify
    inpainted          = inpaint_region(img_tensor, y1, x1, y2, x2, method=inpaint_method, inpaint_filter=inpaint_filter)
    pred_rec           = classify(median_model, inpainted)
    r["pred_recovered"] = pred_rec
    r["recovered"]      = (pred_rec != TARGET_CLASS)
    r["class_match"]    = (pred_rec == r["pred_clean"])

    return r


# -- Evaluation ----------------------------------------------------------------

def run_evaluation(clean_dir, fooled_dir,
                   kernel_size=5, kl_threshold=1.0,
                   topk_percent=0.10, inpaint_method="smooth", inpaint_filter=5,
                   verbose=False, max_images=None, skip_fpr=False,
                   expa_overrides=None):
    clean_dir  = Path(clean_dir)
    fooled_dir = Path(fooled_dir)

    if not clean_dir.exists() or not fooled_dir.exists():
        print("clean_dir or fooled_dir not found.")
        return {}

    cfg = {**EXPA_CONFIG}
    if expa_overrides:
        cfg.update(expa_overrides)
    mask_thr = cfg["mask_confidence_drop"]

    detector     = AnomalyDetector(cfg)
    median_model = _mod8.load_alexnet()
    expa_model   = _mod7.AlexNetWithHooks()

    to_tensor = T.ToTensor()

    fooled_files = sorted(f for f in fooled_dir.iterdir() if f.suffix.lower() == ".png")
    clean_pairs  = [clean_dir / f.name for f in fooled_files]
    # Only keep pairs where the clean counterpart exists
    pairs = [(f, c) for f, c in zip(fooled_files, clean_pairs) if c.exists()]
    if max_images and len(pairs) > max_images:
        pairs = pairs[:max_images]

    n_total = len(pairs)

    print(f"\nModule 10 - Patch Localization + Inpainting (AlexNet 224x224)")
    print(f"{'='*68}")
    print(f"Clean dir:   {clean_dir}")
    print(f"Fooled dir:  {fooled_dir}  ({n_total} pairs)")
    print(f"Detection:   Cascade (Median {kernel_size}x{kernel_size} KL>{kl_threshold} -> EXPA)")
    print(f"Inpainting:  {inpaint_method} ({'full-image median filter' if inpaint_method == 'smooth' else 'surrounding flat fill'})")
    print(f"Localize:    EigenCAM top-{topk_percent*100:.0f}% heatmap + bounding box")
    print(f"Target cls:  {TARGET_CLASS} (banana)")
    print(f"{'='*68}\n")

    n_median_detected = 0
    n_detected        = 0
    n_localized       = 0
    n_recovered       = 0
    n_class_match     = 0

    for i, (fooled_f, clean_f) in enumerate(pairs):
        img   = to_tensor(Image.open(fooled_f).convert("RGB"))
        clean = to_tensor(Image.open(clean_f).convert("RGB"))

        r = process_image(img, clean, median_model, expa_model, detector,
                          kernel_size, kl_threshold, mask_thr, topk_percent,
                          inpaint_method, inpaint_filter)

        if r["median_flagged"]: n_median_detected += 1
        if r["detected"]:       n_detected        += 1
        if r["localized"]:      n_localized        += 1
        if r["recovered"]:      n_recovered        += 1
        if r["class_match"]:    n_class_match      += 1

        if verbose:
            bbox_str = f"bbox=({r['bbox'][0]},{r['bbox'][2]},{r['bbox'][1]},{r['bbox'][3]})" \
                       if r["bbox"] else "bbox=None"
            rec_str  = ("Y" if r["recovered"] else "N") if r["detected"] else "-"
            match_str= ("Y" if r["class_match"] else "N") if r["detected"] else "-"
            print(f"[{i+1:4d}] {fooled_f.name:<30} "
                  f"clean={r['pred_clean']:>4} patch={r['pred_patched']:>4} "
                  f"det={'Y' if r['detected'] else 'N'} "
                  f"rec={rec_str} match={match_str} {bbox_str}")
        elif (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_total} fooled images ...")

    # FPR - evaluate cascade on clean images
    if skip_fpr:
        print(f"\n  Skipping FPR check (--no-fpr flag set)")
        n_clean_total          = 0
        n_median_clean_flagged = 0
        n_clean_flagged        = 0
    else:
        print(f"\n  Checking FPR on clean images ...")
        all_clean = sorted(f for f in clean_dir.iterdir() if f.suffix.lower() == ".png")
        if max_images and len(all_clean) > max_images:
            all_clean = all_clean[:max_images]
        n_clean_total          = len(all_clean)
        n_median_clean_flagged = 0
        n_clean_flagged        = 0

        for i, clean_f in enumerate(all_clean):
            img = to_tensor(Image.open(clean_f).convert("RGB"))
            med = _mod8.detect_median(median_model, img, kernel_size, kl_threshold)
            if med["is_attack"]:
                n_median_clean_flagged += 1
                is_adv, _ = _mod7.run_expa(img, expa_model, detector, mask_thr)
                if is_adv:
                    n_clean_flagged += 1
            if (i + 1) % 100 == 0:
                print(f"  Clean: {i+1}/{n_clean_total} ...")

    expa_model.cleanup()

    # -- Metrics ---------------------------------------------------------------
    med_tpr          = n_median_detected       / max(1, n_total)        * 100
    med_fpr          = n_median_clean_flagged  / max(1, n_clean_total)  * 100
    detection_rate   = n_detected              / max(1, n_total)        * 100
    fpr              = n_clean_flagged         / max(1, n_clean_total)  * 100
    localize_rate    = n_localized   / max(1, n_detected)     * 100 if n_detected else 0.0
    recovery_rate    = n_recovered   / max(1, n_detected)     * 100 if n_detected else 0.0
    class_match_rate = n_class_match / max(1, n_detected)     * 100 if n_detected else 0.0
    end_to_end       = n_recovered   / max(1, n_total)        * 100

    print(f"\n{'='*68}")
    print(f"  Module 10 Results - Patch Localization + Inpainting")
    print(f"{'='*68}")
    print(f"  Fooled images : {n_total}    Clean images : {n_clean_total}")
    print(f"  {'-'*62}")
    print(f"  Stage 1 - Median Filter alone")
    print(f"    TPR (attacks flagged)    : {n_median_detected:>4} / {n_total:<4}  -> {med_tpr:>5.1f}%")
    print(f"    FPR (clean flagged)      : {n_median_clean_flagged:>4} / {n_clean_total:<4}  -> {med_fpr:>5.1f}%")
    print(f"  Stage 2 - Cascade (Median -> EXPA)")
    print(f"    TPR (attacks confirmed)  : {n_detected:>4} / {n_total:<4}  -> {detection_rate:>5.1f}%")
    print(f"    FPR (clean confirmed)    : {n_clean_flagged:>4} / {n_clean_total:<4}  -> {fpr:>5.1f}%")
    print(f"  Stage 3 - Localization (EigenCAM top-{topk_percent*100:.0f}%)")
    print(f"    Patch region found       : {n_localized:>4} / {n_detected:<4}  -> {localize_rate:>5.1f}%  (of detected)")
    print(f"  Stage 4 - Inpainting + Recovery")
    print(f"    Target class removed     : {n_recovered:>4} / {n_detected:<4}  -> {recovery_rate:>5.1f}%  (of detected)")
    print(f"    Original class restored  : {n_class_match:>4} / {n_detected:<4}  -> {class_match_rate:>5.1f}%  (of detected)")
    print(f"  {'-'*62}")
    print(f"  End-to-end recovery        : {n_recovered:>4} / {n_total:<4}  -> {end_to_end:>5.1f}%  (of all fooled)")
    print(f"{'='*68}")

    return {
        "n_total":          n_total,
        "n_detected":       n_detected,
        "n_localized":      n_localized,
        "n_recovered":      n_recovered,
        "n_class_match":    n_class_match,
        "detection_rate":   detection_rate,
        "localize_rate":    localize_rate,
        "recovery_rate":    recovery_rate,
        "class_match_rate": class_match_rate,
        "fpr":              fpr,
        "end_to_end":       end_to_end,
    }


# -- CLI -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Module 10: Patch Localization + Inpainting defense")
    parser.add_argument("--clean",      type=str,   default=None)
    parser.add_argument("--fooled",     type=str,   default=None)
    parser.add_argument("--patch-size", type=int,   default=28)
    parser.add_argument("--data-root",  type=str,   default="data2/stl10")
    parser.add_argument("--kernel",     type=int,   default=5)
    parser.add_argument("--kl-thr",     type=float, default=1.0)
    parser.add_argument("--topk",       type=float, default=0.10,
                        help="Top-k fraction of heatmap for localization (default: 0.10)")
    parser.add_argument("--inpaint",        type=str, default="smooth",
                        choices=["smooth", "flat"],
                        help="Inpainting method: smooth=median filter (default), flat=surrounding fill")
    parser.add_argument("--inpaint-filter", type=int, default=5,
                        help="MedianFilter kernel size for smooth inpainting (default: 5, try 9 or 11)")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--max-images", type=int, default=2000,
                        help="Max fooled images to process (default: 2000, 0=unlimited)")
    parser.add_argument("--target-class", type=int, default=None,
                        help="Attack target class ID (default: 954 banana)")
    parser.add_argument("--no-fpr", action="store_true",
                        help="Skip FPR clean-image check (reuse result from previous run)")
    # -- EXPA Stage-2 thresholds (override EXPA_CONFIG defaults) --
    parser.add_argument("--entropy-thr",    type=float, default=None,
                        help="EigenCAM entropy threshold: H < thr → concentrated heatmap (default: 5.5)")
    parser.add_argument("--peak-mean-thr",  type=float, default=None,
                        help="EigenCAM peak/mean ratio threshold: R > thr → dominant peak (default: 3.0)")
    parser.add_argument("--topk-ratio-thr", type=float, default=None,
                        help="EigenCAM top-k energy ratio threshold: E_k > thr → patch-like blob (default: 0.30)")
    parser.add_argument("--expa-topk",      type=float, default=None,
                        help="Top-k fraction used for Stage-2 energy ratio (default: 0.05)")
    parser.add_argument("--cross-layer-thr",type=float, default=None,
                        help="Cross-layer correlation threshold: rho < thr → layer disagreement (default: 0.5)")
    parser.add_argument("--mask-conf-drop", type=float, default=None,
                        help="Confidence drop threshold for Stage-4 mask validation (default: 0.10)")
    args = parser.parse_args()

    if args.target_class is not None:
        global TARGET_CLASS
        TARGET_CLASS = args.target_class

    ps        = args.patch_size
    data_root = ROOT / args.data_root
    clean_dir  = Path(args.clean)  if args.clean  else data_root / "clean"
    fooled_dir = Path(args.fooled) if args.fooled else data_root / f"patched_{ps}" / "fooled"

    # Build optional EXPA overrides (only set keys that were explicitly provided)
    expa_overrides = {}
    if args.entropy_thr    is not None: expa_overrides["entropy_threshold"]     = args.entropy_thr
    if args.peak_mean_thr  is not None: expa_overrides["peak_mean_threshold"]   = args.peak_mean_thr
    if args.topk_ratio_thr is not None: expa_overrides["topk_ratio_threshold"]  = args.topk_ratio_thr
    if args.expa_topk      is not None: expa_overrides["topk_percent"]          = args.expa_topk
    if args.cross_layer_thr is not None: expa_overrides["cross_layer_threshold"] = args.cross_layer_thr
    if args.mask_conf_drop is not None: expa_overrides["mask_confidence_drop"]  = args.mask_conf_drop

    run_evaluation(clean_dir, fooled_dir,
                   kernel_size=args.kernel,
                   kl_threshold=args.kl_thr,
                   topk_percent=args.topk,
                   inpaint_method=args.inpaint,
                   inpaint_filter=args.inpaint_filter,
                   verbose=args.verbose,
                   max_images=args.max_images or None,
                   skip_fpr=args.no_fpr,
                   expa_overrides=expa_overrides or None)


if __name__ == "__main__":
    main()
