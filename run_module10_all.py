"""run_module10_all.py - Run Module 10 on all patch/dataset combinations."""

import subprocess, sys, csv, re, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RUNS = [
    ("imagenette / banana",       "data3/imagenette/clean",       "data3/imagenette/patched_28/fooled",               954, 28),
    ("imagenette / bee",          "data3/imagenette/clean",       "data3/imagenette/patched_28_bee/fooled",           309, 28),
    ("imagenette / lemon",        "data3/imagenette/clean",       "data3/imagenette/patched_28_lemon/fooled",         951, 28),
    ("imagenette_train / banana", "data3/imagenette_train/clean", "data3/imagenette_train/patched_28/fooled",         954, 28),
    ("imagenette_train / bee",    "data3/imagenette_train/clean", "data3/imagenette_train/patched_28_bee/fooled",     309, 28),
    ("imagenette_train / lemon",  "data3/imagenette_train/clean", "data3/imagenette_train/patched_28_lemon/fooled",   951, 28),
]

TOPK       = 0.05
MAX_IMAGES = 0
MODULE10   = ROOT / "modules" / "10_defense_inpainting" / "test_defense_inpainting.py"

PATTERNS = {
    "n_fooled":    r"Fooled images\s*:\s*(\d+)",
    "tpr":         r"TPR \(attacks confirmed\)\s*:.*->\s*([\d.]+)%",
    "fpr":         r"FPR \(clean confirmed\)\s*:.*->\s*([\d.]+)%",
    "localized":   r"Patch region found\s*:.*->\s*([\d.]+)%",
    "recovered":   r"Target class removed\s*:.*->\s*([\d.]+)%",
    "class_match": r"Original class restored\s*:.*->\s*([\d.]+)%",
    "end_to_end":  r"End-to-end recovery\s*:.*->\s*([\d.]+)%",
}


def parse(output):
    return {k: (m.group(1) if (m := re.search(p, output)) else "N/A")
            for k, p in PATTERNS.items()}


def run_one(label, clean, fooled, target, patch_size, skip_fpr=False):
    print(f"\n  Running: {label} ...", flush=True)
    if skip_fpr:
        print(f"  (FPR check skipped - same clean dir as previous run)", flush=True)
    cmd = [sys.executable, str(MODULE10),
           "--clean", clean, "--fooled", fooled,
           "--patch-size", str(patch_size),
           "--topk", str(TOPK),
           "--max-images", str(MAX_IMAGES),
           "--target-class", str(target)]
    if skip_fpr:
        cmd.append("--no-fpr")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    m = parse(proc.stdout + proc.stderr)
    m["label"] = label
    m["elapsed"] = f"{elapsed:.0f}s"
    print(f"  Done in {elapsed:.0f}s", flush=True)
    return m


def main():
    print("=" * 60)
    print("  Module 10 - All datasets & patches")
    print(f"  TopK=5%  |  unlimited images  |  patch=28px")
    print("=" * 60)

    results = []
    seen_clean = set()
    for label, clean, fooled, target, ps in RUNS:
        skip_fpr = clean in seen_clean
        seen_clean.add(clean)
        results.append(run_one(label, clean, fooled, target, ps, skip_fpr=skip_fpr))

    # ── Per-run results ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS PER DATASET")
    print("=" * 60)
    for m in results:
        print(f"\n  [{m['label']}]")
        print(f"    Images processed : {m['n_fooled']}")
        print(f"    Stage 1+2 - Detection (TPR) : {m['tpr']}%")
        print(f"    Stage 1+2 - False Alarm (FPR): {m['fpr']}%")
        print(f"    Stage 3   - Localized        : {m['localized']}%")
        print(f"    Stage 4   - Target removed   : {m['recovered']}%")
        print(f"    Stage 4   - Class restored   : {m['class_match']}%")
        print(f"    End-to-end recovery          : {m['end_to_end']}%")

    # ── Overall average ────────────────────────────────────────
    def avg(key):
        vals = [float(m[key]) for m in results if m[key] != "N/A"]
        return f"{sum(vals)/len(vals):.1f}" if vals else "N/A"

    print("\n" + "=" * 60)
    print("  OVERALL AVERAGE (all 6 runs)")
    print("=" * 60)
    print(f"    Stage 1+2 - Detection (TPR) : {avg('tpr')}%")
    print(f"    Stage 1+2 - False Alarm (FPR): {avg('fpr')}%")
    print(f"    Stage 3   - Localized        : {avg('localized')}%")
    print(f"    Stage 4   - Target removed   : {avg('recovered')}%")
    print(f"    Stage 4   - Class restored   : {avg('class_match')}%")
    print(f"    End-to-end recovery          : {avg('end_to_end')}%")
    print("=" * 60)

    # ── Save CSV ───────────────────────────────────────────────
    csv_path = ROOT / "results" / "module10_all_results.csv"
    (ROOT / "results").mkdir(exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(PATTERNS.keys()) + ["label", "elapsed"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Saved -> {csv_path}\n")


if __name__ == "__main__":
    main()
