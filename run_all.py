"""run_all.py — Run both defenses, collect per-image metrics, generate graphs.

3 questions answered:
  1. FOOLED patches    → does the defense detect the attack?      (should be HIGH)
  2. NOT FOOLED patches → does the defense still spot the patch?  (informational)
  3. CLEAN images       → does the defense leave them alone?      (should be HIGH)

Outputs (in results/):
  defense_results.csv     — summary per run (appended each time)
  per_image_YYYYMMDD.csv  — every image's metrics for graphing
  plots/                  — auto-generated PNG graphs

Usage:
    python run_all.py
    python run_all.py --kl-thr 0.5 --verbose
    python run_all.py --no-plots
"""

import argparse
import importlib.util
import sys
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent


# ── Load defense modules ───────────────────────────────────────────────────────

def load_defense(module_path: Path, name: str):
    spec   = importlib.util.spec_from_file_location(name, module_path)
    mod    = importlib.util.module_from_spec(spec)
    folder = str(module_path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec.loader.exec_module(mod)
    return mod


# ── Plotting ───────────────────────────────────────────────────────────────────

def make_plots(all_records: list, plot_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[INFO] matplotlib not installed — skipping plots. Run: pip install matplotlib")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    # Split records by defense and label
    med_clean   = [r for r in all_records if r["defense"] == "Median Filter" and r["true_label"] == "CLEAN"]
    med_fooled  = [r for r in all_records if r["defense"] == "Median Filter" and r["true_label"] == "PATCH" and r["dataset"] == "FOOLED"]
    med_nf      = [r for r in all_records if r["defense"] == "Median Filter" and r["true_label"] == "PATCH" and r["dataset"] == "NOT FOOLED"]

    expa_clean  = [r for r in all_records if r["defense"] == "EXPA" and r["true_label"] == "CLEAN"]
    expa_fooled = [r for r in all_records if r["defense"] == "EXPA" and r["true_label"] == "PATCH" and r["dataset"] == "FOOLED"]
    expa_nf     = [r for r in all_records if r["defense"] == "EXPA" and r["true_label"] == "PATCH" and r["dataset"] == "NOT FOOLED"]

    # ── Plot 1: KL divergence distribution (Median Filter) ────────────────────
    if med_clean or med_fooled or med_nf:
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(0, 5, 40)
        if med_clean:
            ax.hist([r["kl_divergence"] for r in med_clean],
                    bins=bins, alpha=0.6, label="Clean", color="steelblue")
        if med_fooled:
            ax.hist([r["kl_divergence"] for r in med_fooled],
                    bins=bins, alpha=0.6, label="Fooled patches", color="tomato")
        if med_nf:
            ax.hist([r["kl_divergence"] for r in med_nf],
                    bins=bins, alpha=0.4, label="Not-fooled patches", color="orange")
        ax.set_xlabel("KL Divergence")
        ax.set_ylabel("Count")
        ax.set_title("Median Filter — KL Divergence Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = plot_dir / "median_kl_distribution.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")

    # ── Plot 2: EXPA metrics distributions ────────────────────────────────────
    for metric, xlabel in [("entropy",    "Spatial Entropy"),
                            ("peak_mean", "Peak / Mean Ratio"),
                            ("topk_energy", "Top-k Energy Ratio"),
                            ("cross_layer_corr", "Cross-Layer Correlation")]:
        groups = [(expa_clean, "Clean", "steelblue"),
                  (expa_fooled, "Fooled patches", "tomato"),
                  (expa_nf,    "Not-fooled patches", "orange")]
        if not any(g for g, _, _ in groups):
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for group, label, color in groups:
            vals = [r[metric] for r in group if metric in r]
            if vals:
                ax.hist(vals, bins=30, alpha=0.6, label=label, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"EXPA — {xlabel} Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = plot_dir / f"expa_{metric}_distribution.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")

    # ── Plot 3: Summary bar chart (detection rates) ────────────────────────────
    summary = {}
    for r in all_records:
        key = (r["defense"], r["dataset"])
        summary.setdefault(key, {"correct": 0, "total": 0})
        summary[key]["total"]   += 1
        summary[key]["correct"] += int(r["correct"])

    labels  = [f"{d}\n{ds}" for (d, ds) in summary]
    rates   = [v["correct"] / max(1, v["total"]) * 100 for v in summary.values()]
    colors  = ["steelblue" if "FOOLED" in k[1] else "orange" for k in summary]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Correct predictions (%)")
    ax.set_title("Defense Accuracy by Dataset")
    ax.set_ylim(0, 110)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    p = plot_dir / "summary_accuracy.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kl-thr",   type=float, default=1.0)
    parser.add_argument("--kernel",   type=int,   default=3)
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip graph generation")
    parser.add_argument("--out",      type=str, default="results/defense_results.csv")
    args = parser.parse_args()

    median = load_defense(ROOT / "modules/5_defense_median/test_defense.py", "median_defense")
    expa   = load_defense(ROOT / "modules/4_defense_expa/test_defense.py",   "expa_defense")

    clean_dir       = ROOT / "data" / "cifar10" / "clean"
    fooled_dir      = ROOT / "data" / "fooled"
    not_fooled_dir  = ROOT / "data" / "not_fooled"

    summary_rows = []
    all_records  = []   # per-image data for graphs

    for defense_name, defense_mod, extra_kwargs in [
        ("Median Filter", median, {"kernel_size": args.kernel,
                                   "kl_threshold": args.kl_thr,
                                   "require_class_change": True}),
        ("EXPA",          expa,   {}),
    ]:
        for ds_name, patched_dir in [("FOOLED",     fooled_dir),
                                      ("NOT FOOLED", not_fooled_dir)]:
            if not patched_dir.exists():
                print(f"\n[SKIP] {ds_name} not found — run save scripts first.")
                continue

            print(f"\n{'='*60}")
            print(f"  {defense_name}  |  {ds_name}")
            print(f"{'='*60}")

            r = defense_mod.run_evaluation(
                clean_dir, patched_dir,
                verbose=args.verbose,
                collect_records=True,
                **extra_kwargs
            )
            if not r:
                continue

            # Tag each per-image record with defense + dataset
            for rec in r.get("records", []):
                rec["defense"] = defense_name
                rec["dataset"] = ds_name
                all_records.append(rec)

            clean_ok  = r["TN"] / max(1, r["TN"] + r["FP"]) * 100
            patch_det = r["TP"] / max(1, r["TP"] + r["FN"]) * 100

            summary_rows.append({
                "Defense":               defense_name,
                "Dataset":               ds_name,
                "Clean images":          r["TN"] + r["FP"],
                "Clean correctly (%)":   round(clean_ok,  1),
                "Patched images":        r["TP"] + r["FN"],
                "Patches detected (%)":  round(patch_det, 1),
                "FPR (%)":               round(r["fpr"] * 100, 1),
                "Used in accuracy":      "YES" if ds_name == "FOOLED" else "informational",
                "KL Threshold":          args.kl_thr,
                "Kernel":                args.kernel,
                "Timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

    # ── Final summary ──────────────────────────────────────────────────────────
    by_defense = {}
    for r in summary_rows:
        by_defense.setdefault(r["Defense"], {})[r["Dataset"]] = r

    print(f"\n\n{'='*72}")
    print(f"  FINAL SUMMARY  (kl-thr={args.kl_thr}  kernel={args.kernel})")
    print(f"{'='*72}")

    for defense_name, ds_map in by_defense.items():
        fooled     = ds_map.get("FOOLED")
        not_fooled = ds_map.get("NOT FOOLED")

        if fooled:
            n_clean  = fooled["Clean images"]
            n_patch  = fooled["Patched images"]
            tp       = round(fooled["Patches detected (%)"] * n_patch / 100)
            tn       = round(fooled["Clean correctly (%)"]  * n_clean / 100)
            accuracy = (tp + tn) / max(1, n_patch + n_clean) * 100
        else:
            accuracy = None

        print(f"\n  {defense_name}")
        print(f"  {'─'*60}")
        if fooled:
            print(f"  Clean images  ({fooled['Clean images']:>4}) → correctly CLEAN   : {fooled['Clean correctly (%)']:>5.1f}%")
            print(f"  Fooled patches ({fooled['Patched images']:>3}) → detected as ATTACK: {fooled['Patches detected (%)']:>5.1f}%")
            print(f"  Accuracy (clean + fooled)              : {accuracy:>5.1f}%")
        if not_fooled:
            print(f"  Not-fooled patches ({not_fooled['Patched images']:>3}) → detected : {not_fooled['Patches detected (%)']:>5.1f}%  [informational]")

    print(f"\n{'='*72}\n")

    # ── Save summary CSV ───────────────────────────────────────────────────────
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    out_path = ROOT / args.out
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary saved to:    {out_path}")

    # ── Save per-image CSV ─────────────────────────────────────────────────────
    if all_records:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        per_image_path = results_dir / f"per_image_{ts}.csv"
        with open(per_image_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
            writer.writeheader()
            writer.writerows(all_records)
        print(f"Per-image data saved: {per_image_path}")

    # ── Generate plots ─────────────────────────────────────────────────────────
    if not args.no_plots and all_records:
        print(f"\nGenerating plots ...")
        make_plots(all_records, results_dir / "plots")


if __name__ == "__main__":
    main()
