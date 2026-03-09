"""run_all_alexnet.py — Run all defenses on AlexNet / STL-10 / 224×224 data.

Differences vs run_all.py (CIFAR-10):
  Model:   AlexNet (ImageNet pretrained)   vs ResNet-20 (CIFAR-10)
  Images:  224×224 STL-10                 vs 32×32 CIFAR-10
  Kernel:  5×5 median filter              vs 3×3
  EXPA layers: features.3 (192×27×27)     vs layer1 (16×32×32)
               features.10 (256×13×13)    vs layer3 (64×8×8)
  Target:  class 954 (banana/ImageNet)    vs class 9 (truck/CIFAR-10)

Modules run:
  7  — EXPA detection
  8  — Median Filter detection
  9  — Cascade (Median → EXPA)
  10 — Full end-to-end: Cascade + Localization + Inpainting (AlexNet re-classify)
  11 — Full end-to-end: Cascade + Localization + Inpainting (ResNet-50 re-classify)

Usage:
    python run_all_alexnet.py                                          # STL-10, 28px
    python run_all_alexnet.py --patch-size 48                          # STL-10, 48px
    python run_all_alexnet.py --data-root data3/imagenette             # ImageNette, 28px
    python run_all_alexnet.py --data-root data3/imagenette --patch-size 48
    python run_all_alexnet.py --verbose
    python run_all_alexnet.py --kl-thr 0.5
    python run_all_alexnet.py --max-images 0                          # all images (no limit)
"""

import argparse
import importlib.util
import sys
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent


def load_defense(module_path, name):
    spec   = importlib.util.spec_from_file_location(name, module_path)
    mod    = importlib.util.module_from_spec(spec)
    folder = str(module_path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec.loader.exec_module(mod)
    return mod


def make_plots(all_records, plot_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[INFO] matplotlib not installed — skipping plots.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    med_clean  = [r for r in all_records if r["defense"] == "Median Filter" and r["true_label"] == "CLEAN"]
    med_fooled = [r for r in all_records if r["defense"] == "Median Filter" and r["true_label"] == "PATCH" and r["dataset"] == "FOOLED"]
    med_nf     = [r for r in all_records if r["defense"] == "Median Filter" and r["true_label"] == "PATCH" and r["dataset"] == "NOT FOOLED"]
    expa_clean  = [r for r in all_records if r["defense"] == "EXPA" and r["true_label"] == "CLEAN"]
    expa_fooled = [r for r in all_records if r["defense"] == "EXPA" and r["true_label"] == "PATCH" and r["dataset"] == "FOOLED"]
    expa_nf     = [r for r in all_records if r["defense"] == "EXPA" and r["true_label"] == "PATCH" and r["dataset"] == "NOT FOOLED"]

    bins = [i * 0.2 for i in range(26)]

    # KL distribution
    if med_clean or med_fooled:
        fig, ax = plt.subplots(figsize=(8, 4))
        if med_clean:  ax.hist([r["kl_divergence"] for r in med_clean],  bins=30, alpha=0.6, label="Clean",          color="steelblue")
        if med_fooled: ax.hist([r["kl_divergence"] for r in med_fooled], bins=30, alpha=0.6, label="Fooled patches", color="tomato")
        if med_nf:     ax.hist([r["kl_divergence"] for r in med_nf],     bins=30, alpha=0.4, label="Not-fooled",     color="orange")
        ax.set_xlabel("KL Divergence"); ax.set_ylabel("Count")
        ax.set_title("AlexNet — Median Filter KL Distribution")
        ax.legend(); ax.grid(True, alpha=0.3)
        p = plot_dir / "alexnet_median_kl.png"
        fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {p}")

    # EXPA entropy
    for metric, label in [("entropy", "Spatial Entropy"), ("peak_mean", "Peak/Mean"),
                           ("topk_energy", "Top-k Energy"), ("cross_layer_corr", "Cross-Layer Corr")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for group, lbl, color in [(expa_clean, "Clean", "steelblue"),
                                   (expa_fooled, "Fooled", "tomato"),
                                   (expa_nf, "Not-fooled", "orange")]:
            vals = [r[metric] for r in group if metric in r]
            if vals: ax.hist(vals, bins=30, alpha=0.6, label=lbl, color=color)
        ax.set_xlabel(label); ax.set_ylabel("Count")
        ax.set_title(f"AlexNet EXPA — {label}")
        ax.legend(); ax.grid(True, alpha=0.3)
        p = plot_dir / f"alexnet_expa_{metric}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {p}")

    # Summary bar
    summary = {}
    for r in all_records:
        key = (r["defense"], r["dataset"])
        summary.setdefault(key, {"correct": 0, "total": 0})
        summary[key]["total"]   += 1
        summary[key]["correct"] += int(r["correct"])
    labels = [f"{d}\n{ds}" for (d, ds) in summary]
    rates  = [v["correct"] / max(1, v["total"]) * 100 for v in summary.values()]
    colors = ["steelblue" if "FOOLED" in k[1] else "orange" for k in summary]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, rates, color=colors, edgecolor="white")
    ax.set_ylabel("Correct (%)"); ax.set_title("AlexNet Defense Accuracy")
    ax.set_ylim(0, 110)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    p = plot_dir / "alexnet_summary_accuracy.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kl-thr",     type=float, default=1.0)
    parser.add_argument("--kernel",     type=int,   default=5)
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--no-plots",   action="store_true")
    parser.add_argument("--patch-size", type=int,   default=28,
                        help="Patch size (default: 28)")
    parser.add_argument("--data-root",  type=str,   default="data2/stl10",
                        help="Data root folder (default: data2/stl10). E.g. data3/imagenette")
    parser.add_argument("--out",        type=str,   default=None)
    parser.add_argument("--max-images", type=int,   default=0,
                        help="Max images per split (default: 0=unlimited)")
    parser.add_argument("--topk",       type=float, default=0.05,
                        help="EigenCAM top-k fraction for localization in mod10/11 (default: 0.05)")
    args = parser.parse_args()

    ps        = args.patch_size
    data_root = ROOT / args.data_root
    out_csv   = args.out or f"results/defense_results_{data_root.parent.name}_{data_root.name}_{ps}.csv"

    median   = load_defense(ROOT / "modules/8_defense_median_alexnet/test_defense_alexnet.py", "median_alexnet")
    expa     = load_defense(ROOT / "modules/7_defense_expa_alexnet/test_defense_alexnet.py",   "expa_alexnet")
    cascade  = load_defense(ROOT / "modules/9_defense_cascade/test_defense_cascade.py",        "cascade_alexnet")
    mod10    = load_defense(ROOT / "modules/10_defense_inpainting/test_defense_inpainting.py", "mod10_alexnet")

    clean_dir      = data_root / "clean"
    fooled_dir     = data_root / f"patched_{ps}" / "fooled"
    not_fooled_dir = data_root / f"patched_{ps}" / "not_fooled"

    summary_rows = []
    all_records  = []

    for defense_name, defense_mod, extra_kwargs in [
        ("Median Filter", median,  {"kernel_size": args.kernel, "kl_threshold": args.kl_thr}),
        ("EXPA",          expa,    {}),
        ("Cascade",       cascade, {"kernel_size": args.kernel, "kl_threshold": args.kl_thr}),
    ]:
        for ds_name, patched_dir in [("FOOLED",     fooled_dir),
                                      ("NOT FOOLED", not_fooled_dir)]:
            if not patched_dir.exists():
                print(f"\n[SKIP] {ds_name} not found — run save_splits_alexnet.py first.")
                continue

            print(f"\n{'='*60}")
            print(f"  {defense_name}  |  {ds_name}")
            print(f"{'='*60}")

            r = defense_mod.run_evaluation(
                clean_dir, patched_dir,
                verbose=args.verbose,
                collect_records=True,
                max_images=args.max_images or None,
                **extra_kwargs
            )
            if not r:
                continue

            for rec in r.get("records", []):
                rec["defense"] = defense_name
                rec["dataset"] = ds_name
                all_records.append(rec)

            clean_ok  = r["TN"] / max(1, r["TN"] + r["FP"]) * 100
            patch_det = r["TP"] / max(1, r["TP"] + r["FN"]) * 100

            summary_rows.append({
                "Defense":               defense_name,
                "Dataset":               ds_name,
                "Model":                 "AlexNet",
                "Image size":            "224×224",
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

    # ── Module 10 — Full end-to-end inpainting defense ────────────────────────
    inpainting_rows = []
    if fooled_dir.exists():
        print(f"\n{'='*60}")
        print(f"  Module 10  |  Full end-to-end (Cascade + Localize + Inpaint)")
        print(f"{'='*60}")

        r10 = mod10.run_evaluation(
            clean_dir, fooled_dir,
            kernel_size=args.kernel,
            kl_threshold=args.kl_thr,
            topk_percent=args.topk,
            verbose=args.verbose,
            max_images=args.max_images or None,
        )
        if r10:
            inpainting_rows.append({
                "Defense":            "Module 10 (Inpainting)",
                "Dataset":            "FOOLED",
                "Model":              "AlexNet",
                "Image size":         "224×224",
                "Fooled images":      r10["n_total"],
                "TPR (%)":            round(r10["detection_rate"],   1),
                "FPR (%)":            round(r10["fpr"] * 100,        1),
                "Localization (%)":   round(r10["localize_rate"],    1),
                "Target removed (%)": round(r10["recovery_rate"],    1),
                "Class restored (%)": round(r10["class_match_rate"], 1),
                "End-to-end (%)":     round(r10["end_to_end"],       1),
                "topk":               args.topk,
                "KL Threshold":       args.kl_thr,
                "Kernel":             args.kernel,
                "Timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

            print(f"\n\n{'='*72}")
            print(f"  MODULE 10 SUMMARY — patch={ps}px  topk={args.topk}")
            print(f"{'='*72}")
            print(f"  Fooled images         : {r10['n_total']}")
            print(f"  TPR (cascade)         : {r10['detection_rate']:>5.1f}%")
            print(f"  FPR                   : {r10['fpr']*100:>5.1f}%")
            print(f"  Localization          : {r10['localize_rate']:>5.1f}%")
            print(f"  Target removed        : {r10['recovery_rate']:>5.1f}%")
            print(f"  Class restored        : {r10['class_match_rate']:>5.1f}%")
            print(f"  End-to-end recovery   : {r10['end_to_end']:>5.1f}%")
            print(f"{'='*72}\n")

            results_dir = ROOT / "results"
            results_dir.mkdir(exist_ok=True)
            inp_csv   = results_dir / f"inpainting_results_{data_root.parent.name}_{data_root.name}_{ps}.csv"
            write_hdr = not inp_csv.exists()
            with open(inp_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=inpainting_rows[0].keys())
                if write_hdr:
                    writer.writeheader()
                writer.writerows(inpainting_rows)
            print(f"Module 10 results saved to: {inp_csv}")
    else:
        print(f"\n[SKIP] Module 10 — fooled dir not found.")

    # ── Final summary ──────────────────────────────────────────────────────────
    by_defense = {}
    for r in summary_rows:
        by_defense.setdefault(r["Defense"], {})[r["Dataset"]] = r

    print(f"\n\n{'='*72}")
    print(f"  FINAL SUMMARY — AlexNet 224×224  patch={ps}px  (kl-thr={args.kl_thr}  kernel={args.kernel})")
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

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if summary_rows:
        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        out_path    = ROOT / out_csv
        write_header = not out_path.exists()
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary saved to: {out_path}")

    if all_records:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        per_img = ROOT / "results" / f"per_image_alexnet_{ts}.csv"
        with open(per_img, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
            writer.writeheader()
            writer.writerows(all_records)
        print(f"Per-image data:   {per_img}")

    if not args.no_plots and all_records:
        print("\nGenerating plots ...")
        make_plots(all_records, ROOT / "results" / "plots")


if __name__ == "__main__":
    main()
