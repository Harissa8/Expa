"""Module 3 — Save only successfully attacked patched images to data/fooled/.

Clean images are NOT copied — the single copy in data/cifar10/clean/ is used
directly by the defense scripts (matched by filename).

Output:
    data/fooled/    <- patched images that fooled the model

Usage (from adversarial_testing/ root):
    python modules/3_attack_validation/save_fooled.py
"""

import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from test_attacks import run_validation

ROOT = Path(__file__).resolve().parent.parent.parent


def save_fooled(clean_dir: Path, patched_dir: Path,
                out_dir: Path, target_class: int,
                require_original_correct: bool = True) -> None:
    results = run_validation(clean_dir, patched_dir, target_class,
                             require_original_correct)
    if not results:
        print("No results to save.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for r in results:
        if not r["fooled"]:
            continue
        shutil.copy2(patched_dir / r["filename"], out_dir / r["filename"])
        saved += 1

    print(f"\nSaved {saved}/{len(results)} fooled patched images to {out_dir}")
    print(f"  Clean images stay in: {clean_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean",   type=str, default=None)
    parser.add_argument("--patched", type=str, default=None)
    parser.add_argument("--out",     type=str, default=None)
    parser.add_argument("--target",  type=int, default=9)
    parser.add_argument("--no-require-correct", action="store_true")
    args = parser.parse_args()

    clean_dir   = Path(args.clean)   if args.clean   else ROOT / "data" / "cifar10" / "clean"
    patched_dir = Path(args.patched) if args.patched else ROOT / "data" / "cifar10" / "patched"
    out_dir     = Path(args.out)     if args.out     else ROOT / "data" / "fooled"

    save_fooled(clean_dir, patched_dir, out_dir, args.target,
                require_original_correct=not args.no_require_correct)


if __name__ == "__main__":
    main()
