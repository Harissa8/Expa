"""
median_effect.py
----------------
Shows the effect of 5x5 median filter on adversarial patched images.

3-panel figure per example:
  Clean image  |  Patched image (fooled)  |  After median filter

Saves: report/median_effect.png

Run:
    python median_effect.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from PIL import ImageFilter

# Try to load AlexNet for predictions (optional)
try:
    import torch
    import torch.nn.functional as F
    import torchvision.models as models
    import torchvision.transforms as transforms

    print("Loading AlexNet for predictions ...")
    try:
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    except TypeError:
        model = models.alexnet(pretrained=True)
    model.eval()

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    to_t = transforms.ToTensor()

    # ImageNet class names (just the ones we need)
    CLASS_NAMES = {
        0:   "tench",
        309: "bee",
        951: "lemon",
        954: "banana",
    }

    def predict(img_np):
        """img_np: H x W x 3 uint8 -> (class_id, class_name, confidence%)"""
        t = to_t(Image.fromarray(img_np)).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(norm(t)), dim=1)[0]
        idx  = probs.argmax().item()
        conf = probs[idx].item() * 100
        name = CLASS_NAMES.get(idx, f"class {idx}")
        return idx, name, conf

    HAS_TORCH = True
    print("AlexNet loaded.\n")

except ImportError:
    HAS_TORCH = False
    print("torch not found — showing images without predictions.\n")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.join(os.path.dirname(__file__), "..")
CLEAN   = os.path.join(BASE, "data3/imagenette/clean")
FOOLED  = os.path.join(BASE, "data3/imagenette/patched_28/fooled")
OUT     = os.path.join(os.path.dirname(__file__), "median_effect.png")

# Pick a few visually diverse examples
EXAMPLES = [
    "img_00001_tench.png",
    "img_00397_english_springer.png",
    "img_01151_chain_saw.png",
]

PATCH_SIZE = 28
IMG_SIZE   = 224
r0 = (IMG_SIZE - PATCH_SIZE) // 2
r1 = r0 + PATCH_SIZE
c0, c1 = r0, r1

# ── Build figure ──────────────────────────────────────────────────────────────
n_rows = len(EXAMPLES)
fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.8 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]

col_titles = ["Clean image", "Patched image\n(adversarial patch applied)", "After 5×5 median filter\n(Stage 1 output)"]
for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=12, fontweight="bold", pad=8)

for row, fname in enumerate(EXAMPLES):
    clean_path  = os.path.join(CLEAN,  fname)
    fooled_path = os.path.join(FOOLED, fname)

    if not os.path.exists(clean_path):
        print(f"Missing clean:  {clean_path}")
        continue
    if not os.path.exists(fooled_path):
        print(f"Missing fooled: {fooled_path}")
        continue

    # Load images as numpy arrays
    clean_np  = np.array(Image.open(clean_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    fooled_np = np.array(Image.open(fooled_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

    # Apply 5x5 median filter using PIL (matches paper implementation)
    median_np = np.array(
        Image.fromarray(fooled_np).filter(ImageFilter.MedianFilter(size=5))
    )

    # Get predictions if torch available
    if HAS_TORCH:
        _, clean_name,  clean_conf  = predict(clean_np)
        _, fooled_name, fooled_conf = predict(fooled_np)
        _, median_name, median_conf = predict(median_np)
        print(f"{fname}")
        print(f"  clean  → {clean_name}  ({clean_conf:.1f}%)")
        print(f"  fooled → {fooled_name} ({fooled_conf:.1f}%)")
        print(f"  median → {median_name} ({median_conf:.1f}%)\n")

    # ── Col 0: Clean ──────────────────────────────────────────────────────────
    ax = axes[row, 0]
    ax.imshow(clean_np)
    if HAS_TORCH:
        ax.set_xlabel(f'Predicted: "{clean_name}"\n({clean_conf:.1f}% confidence)',
                      fontsize=10, color="green", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # ── Col 1: Patched ────────────────────────────────────────────────────────
    ax = axes[row, 1]
    ax.imshow(fooled_np)
    rect = mpatches.FancyBboxPatch(
        (c0, r0), PATCH_SIZE, PATCH_SIZE,
        boxstyle="square,pad=0",
        linewidth=2, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect)
    if HAS_TORCH:
        ax.set_xlabel(f'Predicted: "{fooled_name}"\n({fooled_conf:.1f}% confidence)',
                      fontsize=10, color="red", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # ── Col 2: After median filter ────────────────────────────────────────────
    ax = axes[row, 2]
    ax.imshow(median_np)
    # Highlight where patch was (now blurred)
    rect2 = mpatches.FancyBboxPatch(
        (c0, r0), PATCH_SIZE, PATCH_SIZE,
        boxstyle="square,pad=0",
        linewidth=2, edgecolor="orange", facecolor="none", linestyle="--"
    )
    ax.add_patch(rect2)
    if HAS_TORCH:
        color = "green" if median_name != "banana" else "red"
        ax.set_xlabel(f'Predicted: "{median_name}"\n({median_conf:.1f}% confidence)',
                      fontsize=10, color=color, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

# Row labels (true class)
labels = ["Tench (fish)", "English Springer (dog)", "Chain Saw"]
for row in range(n_rows):
    axes[row, 0].set_ylabel(labels[row], fontsize=11, labelpad=8, rotation=90, va="center")

fig.suptitle(
    "Stage 1: Effect of 5×5 Median Filter on Adversarial Patches\n"
    "Red box = patch location  |  Orange dashed = patch region after filtering",
    fontsize=13, fontweight="bold", y=1.01
)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
plt.show()
