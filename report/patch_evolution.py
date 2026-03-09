"""
patch_evolution.py
------------------
Visualises how the 28x28 banana adversarial patch changes during PGD
optimisation. Snapshots are captured at iterations 0, 20, 40, 60, 80, 100.

Top row  : raw patch pixels (what gets printed / applied)
Bottom row: patch pasted on the source image (what the model sees)

Run from the report/ folder:
    python patch_evolution.py

Output:  report/patch_evolution.png
"""

import os, glob
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_CLASS = 954            # ImageNet class 954 = banana
PATCH_SIZE   = 28             # pixels (matches paper)
ALPHA        = 0.05           # PGD step size (matches paper)
N_ITERS      = 100            # total iterations (matches paper)
SNAP_EVERY   = 20             # capture snapshot every N iterations
IMG_SIZE     = 224            # AlexNet input size
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLEAN_DIR = os.path.join(os.path.dirname(__file__), "../data3/imagenette/clean")
OUT_FILE  = os.path.join(os.path.dirname(__file__), "patch_evolution.png")

# ── Load AlexNet ──────────────────────────────────────────────────────────────
print("Loading AlexNet (ImageNet pretrained) ...")
try:
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
except TypeError:                          # older torchvision
    model = models.alexnet(pretrained=True)

model = model.to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad_(False)

# ImageNet normalisation expected by AlexNet
norm = transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ── Load one clean image ───────────────────────────────────────────────────────
patterns = ["*.png", "*.jpg", "*.JPEG", "*.jpeg"]
paths = []
for pat in patterns:
    paths += glob.glob(os.path.join(CLEAN_DIR, pat))
paths.sort()

if paths:
    img_path = paths[0]
    x = to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    print(f"Source image : {os.path.basename(img_path)}")
else:
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    print("Source image : random noise (data3/imagenette/clean not found)")

# ── Patch location — centre of image ─────────────────────────────────────────
r0 = (IMG_SIZE - PATCH_SIZE) // 2
r1 = r0 + PATCH_SIZE
c0 = (IMG_SIZE - PATCH_SIZE) // 2
c1 = c0 + PATCH_SIZE

target = torch.tensor([TARGET_CLASS], device=DEVICE)

# ── PGD optimisation ──────────────────────────────────────────────────────────
# Initialise patch with random uniform noise in [0, 1]  (matches paper)
patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE, device=DEVICE)

snap_iters = [0] + list(range(SNAP_EVERY, N_ITERS + 1, SNAP_EVERY))
snaps_patch = {}   # iter -> (H, W, 3) numpy patch
snaps_image = {}   # iter -> (H, W, 3) numpy full image with patch applied
confs       = {}   # iter -> banana confidence (%)

def to_numpy(t):
    """Tensor [3,H,W] or [1,3,H,W] -> numpy (H,W,3) clipped to [0,1]."""
    t = t.squeeze().detach().cpu()
    return t.permute(1, 2, 0).numpy().clip(0, 1)

def apply_patch(base_image, p):
    """Return a copy of base_image with patch p pasted at centre."""
    img = base_image.clone()
    img[:, :, r0:r1, c0:c1] = p
    return img

# Snapshot at iter 0 (random init)
snaps_patch[0] = to_numpy(patch)
snaps_image[0] = to_numpy(apply_patch(x, patch))
# Compute initial confidence
with torch.no_grad():
    logits0 = model(norm(apply_patch(x, patch)))
    confs[0] = F.softmax(logits0, dim=1)[0, TARGET_CLASS].item() * 100

print(f"\nPGD  α={ALPHA}  iters={N_ITERS}  target=banana({TARGET_CLASS})")
print(f"Initial banana confidence : {confs[0]:.1f}%")

for it in range(1, N_ITERS + 1):
    p = patch.clone().requires_grad_(True)
    x_adv = apply_patch(x, p)

    logits = model(norm(x_adv))
    loss   = F.cross_entropy(logits, target)
    loss.backward()

    with torch.no_grad():
        patch = (patch + ALPHA * p.grad.sign()).clamp(0.0, 1.0)

    if it in snap_iters:
        conf = F.softmax(logits, dim=1)[0, TARGET_CLASS].item() * 100
        snaps_patch[it] = to_numpy(patch)
        snaps_image[it] = to_numpy(apply_patch(x, patch))
        confs[it]       = conf
        print(f"  iter {it:3d}   banana conf = {conf:.1f}%")

# ── Figure ────────────────────────────────────────────────────────────────────
iters  = sorted(snaps_patch.keys())
n_cols = len(iters)

fig, axes = plt.subplots(2, n_cols, figsize=(2.8 * n_cols, 6.0))
fig.suptitle(
    f"Banana Adversarial Patch — PGD Evolution\n"
    f"α = {ALPHA}   |   patch = {PATCH_SIZE}×{PATCH_SIZE} px   |   "
    f"target class 954 (banana)   |   AlexNet",
    fontsize=13, fontweight="bold"
)

for col, it in enumerate(iters):
    # ── Top row: raw patch ────────────────────────────────────────────────
    ax_top = axes[0, col]
    ax_top.imshow(snaps_patch[it])
    ax_top.set_title(f"Iter {it}\n{confs[it]:.1f}% conf", fontsize=10)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for sp in ax_top.spines.values():
        sp.set_edgecolor("#333")
        sp.set_linewidth(1.2)

    # ── Bottom row: patch applied to image ────────────────────────────────
    ax_bot = axes[1, col]
    ax_bot.imshow(snaps_image[it])
    # Draw a red rectangle around the patch location
    rect = mpatches.FancyBboxPatch(
        (c0, r0), PATCH_SIZE, PATCH_SIZE,
        boxstyle="square,pad=0",
        linewidth=1.5, edgecolor="red", facecolor="none"
    )
    ax_bot.add_patch(rect)
    ax_bot.set_xticks([])
    ax_bot.set_yticks([])
    for sp in ax_bot.spines.values():
        sp.set_edgecolor("#333")
        sp.set_linewidth(1.0)

# Row labels
axes[0, 0].set_ylabel("Raw patch", fontsize=11, labelpad=6)
axes[1, 0].set_ylabel("On image\n(red box = patch)", fontsize=11, labelpad=6)

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"\nSaved  →  {OUT_FILE}")
plt.show()
