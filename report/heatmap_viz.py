"""
heatmap_viz.py
--------------
Shows EigenCAM heatmap on clean vs patched image side by side.

Layout (2 rows x 2 cols):
  Clean image  |  Clean EigenCAM heatmap overlay
  Patched image|  Patched EigenCAM heatmap overlay  (attention concentrated on patch)

Saves: report/heatmap_viz.png

Run:
    python heatmap_viz.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# ── Config ────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")
OUT  = os.path.join(os.path.dirname(__file__), "heatmap_viz.png")

# Try multiple candidates — saves one figure per image so you can pick the best
CANDIDATES = [
    ("img_01151_chain_saw.png",       "chain_saw"),
    ("img_00397_english_springer.png","english_springer"),
    ("img_00785_cassette_player.png", "cassette_player"),
    ("img_00001_tench.png",           "tench"),
    ("img_00082_tench.png",           "tench2"),
]

IMG_SIZE   = 224
PATCH_SIZE = 28
r0 = c0 = (IMG_SIZE - PATCH_SIZE) // 2
r1 = c1 = r0 + PATCH_SIZE

CLASS_NAMES = {
    0:   "tench",       217: "English springer", 482: "cassette player",
    491: "chain saw",   497: "church",           566: "French horn",
    569: "garbage truck", 571: "gas pump",       574: "golf ball",
    701: "parachute",   309: "bee",              951: "lemon",
    954: "banana",
}

# ── Load AlexNet ──────────────────────────────────────────────────────────────
print("Loading AlexNet ...")
try:
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
except TypeError:
    model = models.alexnet(pretrained=True)
model.eval()

norm  = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
to_t  = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ── Grad-CAM (class-discriminative) ──────────────────────────────────────────
_acts = {}

def _fwd_hook(m, inp, out):
    _acts["feat"] = out          # keep in graph — DO NOT detach

model.features[10].register_forward_hook(_fwd_hook)

def gradcam(img_np, target_class=None):
    """
    img_np       : H x W x 3 uint8
    target_class : force a class (None = predicted class)
    """
    t = to_t(Image.fromarray(img_np)).unsqueeze(0)

    model.zero_grad()
    logits = model(norm(t))

    # retain_grad so we can read gradients on this non-leaf tensor
    _acts["feat"].retain_grad()

    probs = F.softmax(logits, dim=1)[0]
    idx   = probs.argmax().item()
    conf  = probs[idx].item() * 100
    name  = CLASS_NAMES.get(idx, f"class {idx}")

    cls = target_class if target_class is not None else idx
    model.zero_grad()
    logits[0, cls].backward()

    grads = _acts["feat"].grad.squeeze(0)        # [C, H, W]
    acts  = _acts["feat"].detach().squeeze(0)    # [C, H, W]

    # Channel weights = global average pooled gradients
    weights = grads.mean(dim=[1, 2])             # [C]
    cam = (weights[:, None, None] * acts).sum(dim=0)  # [H, W]
    cam = F.relu(cam)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear", align_corners=False
    ).squeeze().detach().numpy()

    return cam_up, name, conf

def overlay(img_np, heatmap, alpha=0.50):
    """Blend heatmap (jet colormap) onto image."""
    heat_rgb = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return (img_np * (1 - alpha) + heat_rgb * alpha).clip(0, 255).astype(np.uint8)

def stage4_inpaint(fooled_np, heatmap, pad=8):
    """
    Stage 3+4: localize patch from heatmap, inpaint with median filter.
    Returns: inpainted image, bounding box (y1,x1,y2,x2)
    """
    from PIL import ImageFilter
    # Top-5% threshold → binary mask
    threshold = np.percentile(heatmap, 95)
    mask = (heatmap >= threshold)

    # Bounding box of mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    # Add padding
    y1 = max(0,   y1 - pad)
    x1 = max(0,   x1 - pad)
    y2 = min(IMG_SIZE, y2 + pad)
    x2 = min(IMG_SIZE, x2 + pad)

    # Apply 5x5 median filter to whole image, paste only patch region
    filtered = np.array(Image.fromarray(fooled_np).filter(ImageFilter.MedianFilter(size=5)))
    inpainted = fooled_np.copy()
    inpainted[y1:y2, x1:x2] = filtered[y1:y2, x1:x2]

    return inpainted, (y1, x1, y2, x2)

# ── Loop over candidates ──────────────────────────────────────────────────────
for fname, label in CANDIDATES:
    clean_path  = os.path.join(BASE, "data3/imagenette/clean",             fname)
    fooled_path = os.path.join(BASE, "data3/imagenette/patched_28/fooled", fname)

    if not os.path.exists(clean_path) or not os.path.exists(fooled_path):
        print(f"Skipping {fname} (file not found)")
        continue

    clean_np  = np.array(Image.open(clean_path ).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    fooled_np = np.array(Image.open(fooled_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

    clean_heat,  clean_name,  clean_conf  = gradcam(clean_np,  target_class=None)
    fooled_heat, fooled_name, fooled_conf = gradcam(fooled_np, target_class=954)

    # Stage 4: localize + inpaint
    recov_np, bbox = stage4_inpaint(fooled_np, fooled_heat)
    recov_heat, recov_name, recov_conf = gradcam(recov_np, target_class=None)
    by1, bx1, by2, bx2 = bbox

    print(f"{fname}")
    print(f"  clean  → {clean_name}  ({clean_conf:.1f}%)")
    print(f"  fooled → {fooled_name} ({fooled_conf:.1f}%)")
    print(f"  recov  → {recov_name}  ({recov_conf:.1f}%)  bbox=({by1},{bx1},{by2},{bx2})\n")

    clean_overlay  = overlay(clean_np,  clean_heat)
    fooled_overlay = overlay(fooled_np, fooled_heat)
    recov_overlay  = overlay(recov_np,  recov_heat)

    # Stage 3 localization view: patched image with bbox drawn (via matplotlib)

    recov_color = "green" if recov_name != "banana" else "red"

    # ── Figure: 4 rows × 2 cols ───────────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(10, 17))
    fig.suptitle(
        f"EXPA Full Pipeline  [{fname}]\n"
        "Left: image   |   Right: Grad-CAM attention (red = high attention)",
        fontsize=12, fontweight="bold"
    )

    panels = [
        # (ax, image, title, color, show_patch_box, show_bbox)
        (axes[0,0], clean_np,      f'Step 1 — Clean input\nPredicted: "{clean_name}" ({clean_conf:.1f}%)',      "green",     False, False),
        (axes[0,1], clean_overlay, "Grad-CAM — clean\n(attention distributed on object)",                       "dimgray",   False, False),
        (axes[1,0], fooled_np,     f'Step 2 — After attack\nPredicted: "{fooled_name}" ({fooled_conf:.1f}%)',   "red",       True,  False),
        (axes[1,1], fooled_overlay,"Grad-CAM — patched\n(attention hijacked by patch)",                         "dimgray",   True,  False),
        (axes[2,0], fooled_np,     f'Step 3 — Patch localized\nDetected region (yellow box)',                   "darkorange",False, True),
        (axes[2,1], fooled_overlay,"Grad-CAM — localization\n(top-5% threshold → bounding box)",                "dimgray",   False, True),
        (axes[3,0], recov_np,      f'Step 4 — Recovered image\nPredicted: "{recov_name}" ({recov_conf:.1f}%)', recov_color, False, False),
        (axes[3,1], recov_overlay, "Grad-CAM — recovered\n(attention back to natural object)",                  "dimgray",   False, False),
    ]

    for ax, img, ttl, color, show_patch, show_bbox in panels:
        ax.imshow(img)
        if show_patch:
            ax.add_patch(mpatches.FancyBboxPatch(
                (c0, r0), PATCH_SIZE, PATCH_SIZE,
                boxstyle="square,pad=0",
                linewidth=2, edgecolor="white", facecolor="none"
            ))
        if show_bbox:
            ax.add_patch(mpatches.FancyBboxPatch(
                (bx1, by1), bx2-bx1, by2-by1,
                boxstyle="square,pad=0",
                linewidth=2.5, edgecolor="yellow", facecolor="none",
                linestyle="--"
            ))
        ax.set_title(ttl, fontsize=10, color=color, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    # Row labels
    for row, lbl in enumerate(["Step 1\nClean", "Step 2\nAttacked", "Step 3\nLocalized", "Step 4\nRecovered"]):
        axes[row, 0].set_ylabel(lbl, fontsize=10, fontweight="bold", labelpad=6)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Attention intensity")

    out_path = OUT.replace(".png", f"_{label}.png")
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()

print("\nAll done. Check report/ folder for heatmap_viz_*.png")
