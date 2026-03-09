"""Run Defense 1 — EXPA (EigenCAM).

Usage:
    python run_test_defense1.py                                  # fooled patches
    python run_test_defense1.py --patched data/not_fooled        # not-fooled patches
    python run_test_defense1.py --verbose
    python run_test_defense1.py --entropy-thr 4.5 --mask-thr 0.1
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "modules" / "4_defense_expa"))
from test_defense import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Test EXPA (EigenCAM) defense")
    parser.add_argument("--clean",       type=str, default=None,
                        help="Clean images folder (default: data/cifar10/clean)")
    parser.add_argument("--patched",     type=str, default=None,
                        help="Patched images folder (default: data/fooled)")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--entropy-thr", type=float, default=None)
    parser.add_argument("--peak-thr",    type=float, default=None)
    parser.add_argument("--topk-thr",    type=float, default=None)
    parser.add_argument("--cross-thr",   type=float, default=None)
    parser.add_argument("--mask-thr",    type=float, default=None)
    parser.add_argument("--mode",        type=str,   default=None, choices=["vote", "any"])
    args = parser.parse_args()

    clean_dir   = Path(args.clean)   if args.clean   else ROOT / "data" / "cifar10" / "clean"
    patched_dir = Path(args.patched) if args.patched else ROOT / "data" / "fooled"

    overrides = {}
    if args.entropy_thr is not None: overrides["entropy_threshold"]    = args.entropy_thr
    if args.peak_thr    is not None: overrides["peak_mean_threshold"]  = args.peak_thr
    if args.topk_thr    is not None: overrides["topk_ratio_threshold"] = args.topk_thr
    if args.cross_thr   is not None: overrides["cross_layer_threshold"]= args.cross_thr
    if args.mask_thr    is not None: overrides["mask_confidence_drop"] = args.mask_thr
    if args.mode        is not None: overrides["detection_mode"]       = args.mode

    results = run_evaluation(clean_dir, patched_dir, config=overrides, verbose=args.verbose)

    if results:
        print(f"\nQuick summary:")
        print(f"  TPR (detection)  : {results['tpr']*100:.1f}%")
        print(f"  FPR (false pos)  : {results['fpr']*100:.1f}%")
        print(f"  Accuracy         : {results['accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
