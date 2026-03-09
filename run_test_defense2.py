"""Run Defense 2 — Median Filter.

Usage:
    python run_test_defense2.py                                        # fooled patches
    python run_test_defense2.py --patched data/not_fooled              # not-fooled patches
    python run_test_defense2.py --verbose
    python run_test_defense2.py --kl-thr 0.3 --kernel 3
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "modules" / "5_defense_median"))
from test_defense import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Test Median Filter defense")
    parser.add_argument("--clean",   type=str, default=None,
                        help="Clean images folder (default: data/cifar10/clean)")
    parser.add_argument("--patched", type=str, default=None,
                        help="Patched images folder (default: data/fooled)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--kl-thr",  type=float, default=1.0)
    parser.add_argument("--kernel",  type=int,   default=3)
    parser.add_argument("--require-class-change", action="store_true", default=True)
    args = parser.parse_args()

    clean_dir   = Path(args.clean)   if args.clean   else ROOT / "data" / "cifar10" / "clean"
    patched_dir = Path(args.patched) if args.patched else ROOT / "data" / "fooled"

    results = run_evaluation(
        clean_dir, patched_dir,
        kernel_size=args.kernel,
        kl_threshold=args.kl_thr,
        require_class_change=args.require_class_change,
        verbose=args.verbose,
    )

    if results:
        print(f"\nQuick summary:")
        print(f"  TPR (detection)  : {results['tpr']*100:.1f}%")
        print(f"  FPR (false pos)  : {results['fpr']*100:.1f}%")
        print(f"  Accuracy         : {results['accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
