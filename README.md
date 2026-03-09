# EXPA — Explainability-Based Defense against Adversarial Patch Attacks

A modular pipeline for generating adversarial patches and evaluating a 4-stage defense system (EXPA) using EigenCAM-based localization and median inpainting.

Tested on **ImageNette** and **STL-10** with **AlexNet**, and on **CIFAR-10** with **ResNet-20**.

---

## Results Summary

| Dataset | Model | TPR | FPR | Localization |
|---|---|---|---|---|
| ImageNette Split-1 | AlexNet | 98.5% | 19.6% | 100% |
| ImageNette Split-2 | AlexNet | 98.7% | 19.2% | 100% |
| STL-10 | AlexNet | 99.0% | 6.9% | 100% |

**Average across all runs: TPR = 98.7%, FPR = 17.6%**

---

## How EXPA Works

The EXPA defense is a 4-stage cascade applied to every incoming image:

```
Stage 1 — Detection      5×5 median filter + KL divergence threshold (τ=1.0)
Stage 2 — Confirmation   EigenCAM (XAI) anomaly metrics: entropy, peak/mean, top-k energy, cross-layer correlation
Stage 3 — Localization   Top-5% heatmap threshold → bounding box + 8px padding
Stage 4 — Recovery       Median inpainting of localized region only
```

- **Stage 1** catches images where median filtering significantly shifts the model's confidence distribution.
- **Stage 2** reduces false positives by verifying that the heatmap shows a concentrated anomalous region (not just natural texture).
- **Stage 3 & 4** precisely remove the patch and restore the correct prediction.

---

## Project Structure

```
adversarial_testing/
├── modules/
│   ├── 1_data_acquisition/        Download CIFAR-10, save as PNG
│   ├── 2_patch_generation/        PGD adversarial patch generation (CIFAR-10 / ResNet-20)
│   ├── 3_attack_validation/       Validate and split fooled/not-fooled pairs
│   ├── 4_defense_expa/            EXPA defense (ResNet-20)
│   ├── 5_defense_median/          Median filter defense (ResNet-20)
│   ├── 6_alexnet_data/            Download ImageNette + STL-10, generate patches (AlexNet)
│   ├── 7_defense_expa_alexnet/    EXPA defense (AlexNet)
│   ├── 8_defense_median_alexnet/  Median filter defense (AlexNet)
│   ├── 9_defense_cascade/         Full 4-stage EXPA cascade (AlexNet)
│   └── 10_defense_inpainting/     Inpainting recovery module
│
├── report/
│   ├── patch_evolution.py         Visualize PGD patch every 20 iterations
│   ├── median_effect.py           Clean | Patched | After median filter (3 examples)
│   └── heatmap_viz.py             Grad-CAM 4-row figure: clean/patched/localized/recovered
│
├── run_all.py                     Run both defenses on CIFAR-10 + export CSV
├── run_all_alexnet.py             Run full EXPA cascade on ImageNette + STL-10
├── run_test_defense1.py           EXPA defense only (ResNet-20)
├── run_test_defense2.py           Median filter defense only (ResNet-20)
├── run_module10_all.py            Run inpainting module on all datasets
├── generate_figures.py            Generate all result figures
├── config.yaml                    Global configuration
└── data/  data2/  data3/          (gitignored — not included in repo)
```

---

## Quick Start

### CIFAR-10 / ResNet-20

```bash
# 1. Download 200 CIFAR-10 images
python modules/1_data_acquisition/download_cifar.py --n 200

# 2. Generate adversarial patch (PGD, target=truck, 28x28 patch, 100 iterations)
python modules/2_patch_generation/generate_patches.py --size 28 --iters 100

# 3. Split into fooled / not-fooled pairs
python modules/3_attack_validation/save_fooled.py
python modules/3_attack_validation/save_not_fooled.py

# 4. Run both defenses and compare
python run_all.py
```

### ImageNette + STL-10 / AlexNet

```bash
# 1. Download datasets
python modules/6_alexnet_data/download_imagenette.py
python modules/6_alexnet_data/download_stl10.py

# 2. Generate patches and save splits
python modules/6_alexnet_data/generate_patches_alexnet.py
python modules/6_alexnet_data/save_splits_alexnet.py

# 3. Run full EXPA cascade
python run_all_alexnet.py
```

---

## Patch Generation — PGD

Patches are generated using **Projected Gradient Descent (PGD)**:

```
delta^(t+1) = clip( delta^(t) + alpha * sign( grad_delta Loss ) , 0, 1 )
```

- `alpha = 0.05` (step size)
- `100 iterations`
- `28×28 patch` placed at a random position on a 224×224 image
- Target class: **banana** (ImageNet class 954) for AlexNet experiments

Each pixel of the patch is updated to maximize the cross-entropy loss toward the target class.

---

## Visualization Scripts

Located in `report/` — run with your PyTorch conda environment:

```bash
# PGD patch evolution (snapshots every 20 iterations)
python report/patch_evolution.py

# Median filter effect on 3 example images
python report/median_effect.py

# Grad-CAM heatmap: clean / patched / localized / recovered
python report/heatmap_viz.py
```

---

## Requirements

Each module has its own `requirements.txt`. Core dependencies:

- `torch >= 1.12`
- `torchvision`
- `Pillow`
- `numpy`
- `matplotlib`
- `pyyaml`

Install per module:
```bash
pip install -r modules/4_defense_expa/requirements.txt
```

---

## Data

Datasets are **not included** in this repository (gitignored).

| Folder | Dataset | Source |
|---|---|---|
| `data/` | CIFAR-10 (32×32) | Auto-downloaded via torchvision |
| `data2/` | STL-10 (96×96) | Auto-downloaded via torchvision |
| `data3/` | ImageNette-320 | [fastai/imagenette](https://github.com/fastai/imagenette) |

---

## Report

This code accompanies the report:
**"EXPA: Explainability-Based Detection and Recovery of Adversarial Patches"**
ENSTA  — Systems Application, 2025
