import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(13, 16))
ax.set_xlim(1.5, 12.5)
ax.set_ylim(-2.5, 15.5)
ax.axis('off')


def add_box(ax, x, y, width, height, text, color='lightblue', fontsize=12):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.15",
                         edgecolor='#333333',
                         facecolor=color,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text,
            ha='center', va='center',
            fontsize=fontsize, weight='bold', wrap=True,
            multialignment='center')


def add_arrow(ax, x1, y1, x2, y2, label='', label_offset=(0.3, 0)):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->',
                            mutation_scale=22,
                            linewidth=2,
                            color='#222222')
    ax.add_patch(arrow)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', fontsize=8, style='italic', color='#444444')


# ── Column centres ────────────────────────────────────────────────
CX   = 4.0   # main pipeline centre-x
BW   = 3.5   # main box width
BX   = CX - BW / 2  # left edge of main boxes

DX   = 8.0   # decision box left edge
DW   = 1.8   # decision box width

PX   = 10.1  # CLEAN/pass box left edge
PW   = 1.7   # pass box width

# ── Vertical positions (bottom of each box) ───────────────────────
Y_INPUT  = 13.5
Y_S1     = 11.0
Y_S2     =  8.0
Y_S3     =  5.0
Y_S4     =  2.2
Y_RECLF  =  0.0
Y_RESULT = -2.2

BH_INPUT = 1.0
BH_STAGE = 1.8
BH_RECLF = 1.5
BH_RESLT = 1.8

# Decision boxes sit at mid-height of corresponding stage box
def dec_y(stage_y, stage_h, dec_h=0.9):
    return stage_y + (stage_h - dec_h) / 2

# ── INPUT ──────────────────────────────────────────────────────────
add_box(ax, BX, Y_INPUT, BW, BH_INPUT, 'INPUT IMAGE\n224×224 RGB', '#AED6F1')

# ── STAGE 1 ────────────────────────────────────────────────────────
add_box(ax, BX, Y_S1, BW, BH_STAGE,
        'STAGE 1\nMedian Filter\nKL Divergence', '#A9DFBF')
add_arrow(ax, CX, Y_INPUT, CX, Y_S1 + BH_STAGE)  # Input → S1

D1Y = dec_y(Y_S1, BH_STAGE)
add_box(ax, DX, D1Y, DW, 0.9, 'KL > 1.0?', '#F9E79F')
add_box(ax, PX, D1Y, PW, 0.9, 'CLEAN\n(pass)', '#ABEBC6')
add_arrow(ax, BX + BW, D1Y + 0.45, DX, D1Y + 0.45)           # S1 → decision
add_arrow(ax, DX + DW, D1Y + 0.65, PX, D1Y + 0.65, 'No', label_offset=(0, 0.15))

# ── STAGE 2 ────────────────────────────────────────────────────────
add_box(ax, BX, Y_S2, BW, BH_STAGE,
        'STAGE 2\nEigenCAM\n4 Metrics + Vote', '#A9DFBF')
add_arrow(ax, CX, Y_S1, CX, Y_S2 + BH_STAGE, 'suspicious', label_offset=(0.6, 0))

D2Y = dec_y(Y_S2, BH_STAGE)
add_box(ax, DX, D2Y, DW, 0.9, '≥2 metrics?\n+mask drop?', '#F9E79F')
add_box(ax, PX, D2Y, PW, 0.9, 'CLEAN\n(pass)', '#ABEBC6')
add_arrow(ax, BX + BW, D2Y + 0.45, DX, D2Y + 0.45)
add_arrow(ax, DX + DW, D2Y + 0.65, PX, D2Y + 0.65, 'No', label_offset=(0, 0.15))

# ── STAGE 3 ────────────────────────────────────────────────────────
add_box(ax, BX, Y_S3, BW, BH_STAGE,
        'STAGE 3\nTop-5%\nPatch Localization', '#A9DFBF')
add_arrow(ax, CX, Y_S2, CX, Y_S3 + BH_STAGE, 'confirmed', label_offset=(0.6, 0))

# ── STAGE 4 ────────────────────────────────────────────────────────
add_box(ax, BX, Y_S4, BW, BH_STAGE,
        'STAGE 4\nMedian Filter\nInpainting', '#A9DFBF')
add_arrow(ax, CX, Y_S3, CX, Y_S4 + BH_STAGE)

# ── RE-CLASSIFY ────────────────────────────────────────────────────
add_box(ax, BX, Y_RECLF, BW, BH_RECLF,
        'RE-CLASSIFY\nRecovered Image', '#AED6F1')
add_arrow(ax, CX, Y_S4, CX, Y_RECLF + BH_RECLF)

# ── RESULT ─────────────────────────────────────────────────────────
add_box(ax, BX, Y_RESULT, BW, BH_RESLT,
        'RESULT\nTarget Removed\nClass Restored', '#F1948A')
add_arrow(ax, CX, Y_RECLF, CX, Y_RESULT + BH_RESLT)

plt.title('')
plt.tight_layout()
plt.savefig('figures/pipeline_overview.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/pipeline_overview.png', bbox_inches='tight', dpi=300)
plt.close()

print("Figure saved → figures/pipeline_overview.pdf / .png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Defense trade-offs scatter plot
# ══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 6))

methods = {
    'Adversarial Training':      (5,    85,   'red'),
    'Median Filter Alone':       (34.5, 99.5, 'orange'),
    'SentiNet':                  (5,    92,   'blue'),
    'Our Cascade (ImageNette)':  (19.6, 98.7, 'green'),
    'Our Cascade (STL-10)':      (6.9,  99.5, 'darkgreen'),
}

for method, (fpr, tpr, color) in methods.items():
    ax.scatter(fpr, tpr, s=200, c=color, alpha=0.7, edgecolors='black', linewidth=2)
    offset_x = 2 if fpr < 30 else -8
    ax.annotate(method, (fpr, tpr),
                xytext=(offset_x, 2),
                textcoords='offset points',
                fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

# Practical deployment zone
ax.axhspan(90, 100, xmin=0, xmax=0.25, alpha=0.2, color='green',
           label='Practical Deployment Zone')

# Ideal point
ax.scatter(0, 100, s=300, c='gold', marker='*', edgecolors='black', linewidth=2, zorder=10)
ax.annotate('Ideal\n(0% FP, 100% TP)', (0, 100),
            xytext=(5, -8), textcoords='offset points',
            fontsize=9, style='italic')

ax.set_xlabel('False Positive Rate (%)', fontsize=12, weight='bold')
ax.set_ylabel('True Positive Rate (%)', fontsize=12, weight='bold')
ax.set_title('Adversarial Patch Defense Trade-offs', fontsize=14, weight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-2, 50)
ax.set_ylim(80, 102)

textstr = 'Our Cascade:\nImageNette: TPR 98.7%, FPR 19.6%\nSTL-10:     TPR 99.5%, FPR  6.9%'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
ax.text(0.72, 0.05, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=props, weight='bold')

plt.tight_layout()
plt.savefig('figures/defense_tradeoffs.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/defense_tradeoffs.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure saved → figures/defense_tradeoffs.pdf / .png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Patch attack example: clean | patched (1x2 side-by-side)
# Same size/style as Figures 1 and 2
# ══════════════════════════════════════════════════════════════════

CLEAN_DIR  = "data3/imagenette/clean"
FOOLED_DIR = "data3/imagenette/patched_28/fooled"


def _first_image(folder):
    """Return the first .png/.jpg found in folder, or None."""
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPEG"):
        hits = sorted(glob.glob(os.path.join(folder, ext)))
        if hits:
            return hits[0]
    return None


clean_path  = _first_image(CLEAN_DIR)
fooled_path = _first_image(FOOLED_DIR)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

panels = [
    (clean_path,  "Clean Image",                                   False),
    (fooled_path, "Adversarial Patch Applied\n(predicted: banana)", True),
]

for ax, (path, title, red_border) in zip(axes, panels):
    if path and os.path.exists(path):
        img = mpimg.imread(path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "Image not found", ha='center', va='center',
                fontsize=9, color='red', transform=ax.transAxes)
        ax.set_facecolor('#f0f0f0')
    ax.axis('off')
    ax.set_title(title, fontsize=11, weight='bold', pad=8)
    if red_border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('red')
            spine.set_linewidth(3)

fig.suptitle('Adversarial Patch Attack: Before and After',
             fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('figures/patch_attack_examples.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/patch_attack_examples.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Figure saved → figures/patch_attack_examples.png / .pdf")
