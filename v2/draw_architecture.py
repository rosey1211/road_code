"""
Generates road_code/v2/architecture.png — a diagram of the RoadCNN architecture.
Run: python draw_architecture.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_W, FIG_H = 22, 14
FS = 1.45   # global font-size scale factor
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG      = "#1a1a2e"
C_INPUT   = "#16213e"
C_PREP    = "#0f3460"
C_BACK    = "#533483"
C_CLS     = "#e94560"
C_ROW     = "#0a7a5a"
C_HEAD    = "#1a7a4a"
C_OUT     = "#c77d00"
C_ARROW   = "#aaaacc"
C_TEXT    = "#f0f0f0"
C_ANN     = "#ffdd88"
C_NOTE    = "#aaddff"

def box(x, y, w, h, color, alpha=0.85, radius=0.25):
    r = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.05,rounding_size={radius}",
                       facecolor=color, edgecolor="#ccccdd",
                       linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(r)
    return r

def label(x, y, text, size=9, color=C_TEXT, bold=False, ha="center", va="center"):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=size * FS, color=color, ha=ha, va=va,
            fontweight=weight, zorder=5)

def arrow(x0, y0, x1, y1, color=C_ARROW, lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=4)

def divider(x0, x1, y, color="#445566", lw=0.8):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, ls="--", zorder=2)


# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
ax.text(11, 13.5, "RoadCNN Architecture", fontsize=18, color=C_TEXT,
        ha="center", va="center", fontweight="bold", zorder=5)
ax.text(11, 13.1, "160×120 input  ·  40 buckets/row  ·  3 scan rows  ·  dual-path output",
        fontsize=10, color=C_NOTE, ha="center", va="center", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN LAYOUT (left → right):
#   [Input] → [Preprocess] → [Backbone] → [split] → [Classifier] | [Row Heads]
# ─────────────────────────────────────────────────────────────────────────────

# ── INPUT ─────────────────────────────────────────────────────────────────────
box(0.3, 5.5, 2.2, 3.2, C_INPUT)
label(1.4, 7.6, "INPUT", 10, bold=True)
label(1.4, 7.1, "RGB Image", 9, C_NOTE)
label(1.4, 6.7, "any resolution", 8, C_ANN)
label(1.4, 6.3, "[B, 3, H, W]", 8)
label(1.4, 5.85, "tensor float32", 8, C_ANN)

arrow(2.5, 7.1, 3.0, 7.1)

# ── PREPROCESS ────────────────────────────────────────────────────────────────
box(3.0, 5.2, 2.6, 3.8, C_PREP)
label(4.3, 8.65, "PREPROCESS", 10, bold=True)
label(4.3, 8.2,  "Resize (bilinear)", 8.5, C_NOTE)
label(4.3, 7.85, "→ 160 × 120", 8, C_ANN)
label(4.3, 7.4,  "Crop sky & hood", 8.5, C_NOTE)
label(4.3, 7.05, "crop_top / crop_bottom", 8, C_ANN)
label(4.3, 6.6,  "Normalize", 8.5, C_NOTE)
label(4.3, 6.25, "ImageNet mean/std", 8, C_ANN)
label(4.3, 5.7,  "[B, 3, H_crop, 160]", 8)

arrow(5.6, 7.1, 6.1, 7.1)

# ── BACKBONE ──────────────────────────────────────────────────────────────────
box(6.1, 3.2, 3.4, 8.1, C_BACK, alpha=0.8)
label(7.8, 11.0, "BACKBONE", 11, bold=True)

stages = [
    ("Stage 1", "Conv3×3-BN-ReLU ×2", "3 → 64",   "+ MaxPool 2×2", "→ [B,  64, 60, 80]"),
    ("Stage 2", "Conv3×3-BN-ReLU ×2", "64 → 128",  "+ MaxPool 2×2", "→ [B, 128, 30, 40]"),
    ("Stage 3", "Conv3×3-BN-ReLU ×2", "128 → 256", "+ MaxPool 2×2", "→ [B, 256, 15, 20]"),
    ("Stage 4", "Conv3×3-BN-ReLU ×2", "256 → 512", "no pool",       "→ [B, 512, 15, 20]"),
]
y_st = 10.1
for s, ops, ch, pool, out in stages:
    box(6.3, y_st - 1.4, 3.0, 1.25, "#6a45a0", alpha=0.7, radius=0.18)
    label(7.8, y_st - 0.45, s, 8.5, C_ANN, bold=True)
    label(7.8, y_st - 0.82, ops, 8)
    label(7.8, y_st - 1.12, f"{ch}   {pool}", 7.5, C_NOTE)
    if s != "Stage 1":
        ax.plot([7.8, 7.8], [y_st + 0.02, y_st - 0.05 + 1.4], color=C_ARROW, lw=1.2,
                zorder=3)
        ax.annotate("", xy=(7.8, y_st - 0.02), xytext=(7.8, y_st + 0.01),
                    arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.2), zorder=4)
    y_st -= 1.55

label(7.8, 3.55, "Feature Map: [B, 512, 15, 20]", 9, C_ANN, bold=True)

# ── SPLIT ARROW ───────────────────────────────────────────────────────────────
# from backbone bottom → two paths
arrow(9.5, 7.1, 10.0, 7.1)
# split line
ax.plot([10.0, 10.0], [3.8, 10.5], color=C_ARROW, lw=1.5, zorder=3)
ax.plot([10.0, 10.3], [10.5, 10.5], color=C_ARROW, lw=1.5, zorder=3)  # up to classifier
ax.annotate("", xy=(10.3, 10.5), xytext=(9.95, 10.5),
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.5), zorder=4)
ax.plot([10.0, 10.3], [3.8, 3.8], color=C_ARROW, lw=1.5, zorder=3)   # down to row heads
ax.annotate("", xy=(10.3, 3.8), xytext=(9.95, 3.8),
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.5), zorder=4)

# ── CLASSIFIER HEAD ───────────────────────────────────────────────────────────
box(10.3, 9.2, 3.8, 2.8, C_CLS, alpha=0.85)
label(12.2, 11.65, "CLASSIFIER HEAD", 10, bold=True)
label(12.2, 11.2,  "Global Avg Pool → [B, 512]", 8.5, C_NOTE)
label(12.2, 10.8,  "Linear(512→256) + ReLU", 8.5)
label(12.2, 10.45, "Dropout(p=0.4)", 8.5)
label(12.2, 10.05, "Linear(256→2)  →  cls_logits", 8.5)
label(12.2, 9.65,  "softmax → P(road present)", 8, C_ANN)

arrow(14.1, 10.6, 14.7, 10.6)

box(14.7, 10.0, 2.6, 1.3, C_OUT, alpha=0.9)
label(16.0, 10.75, "road_present", 9, bold=True)
label(16.0, 10.3,  "road_confidence", 8.5)

# ── ROW HEADS ─────────────────────────────────────────────────────────────────
box(10.3, 0.6, 3.8, 6.2, C_ROW, alpha=0.8)
label(12.2, 6.45, "ROW HEADS  (×3, independent)", 10, bold=True)

label(12.2, 6.0, "Per-row strip extraction:", 9, C_NOTE)
label(12.2, 5.65, "slice 3 rows at scan-line fraction", 8.5)
label(12.2, 5.3,  "→ [B, 512, 3, 20]", 8, C_ANN)

label(12.2, 4.8,  "Non-uniform bucket pooling:", 9, C_NOTE)
label(12.2, 4.45, "40 columns, edge 5× wider than centre", 8.5)
label(12.2, 4.1,  "10% margin each side excluded", 8.5)
label(12.2, 3.75, "→ [B, 512, 40]", 8, C_ANN)

label(12.2, 3.25, "Conv1d(512→64, k=3) + ReLU", 9, C_NOTE)
label(12.2, 2.90, "Conv1d(64→1,  k=1)", 9, C_NOTE)
label(12.2, 2.5,  "→ [B, 40]  bucket logits", 8, C_ANN)

label(12.2, 2.0,  "sigmoid → per-bucket probability", 8.5)
label(12.2, 1.6,  "cluster peak detection + temporal", 8.5)
label(12.2, 1.2,  "bias → cx_frac (normalised column)", 8, C_ANN)

arrow(14.1, 3.7, 14.7, 3.7)

# ── ROW OUTPUT BOXES ──────────────────────────────────────────────────────────
row_ys = [5.1, 3.6, 2.1]
row_labels = ["r0  FAR   (highest crop)", "r1  MID", "r2  NEAR  (lowest crop)"]
for ry, rl in zip(row_ys, row_labels):
    box(14.7, ry - 0.55, 2.6, 1.05, C_OUT, alpha=0.9)
    label(16.0, ry + 0.05, rl, 8.5, C_TEXT, bold=True)
    label(16.0, ry - 0.3, "cx_frac,  confidence", 8)

# arrows to row boxes
for ry in row_ys:
    arrow(14.1, ry - 0.0, 14.7, ry - 0.0)

# ── COMBINED OUTPUT ───────────────────────────────────────────────────────────
box(17.6, 1.5, 4.0, 9.5, C_OUT, alpha=0.75)
label(19.6, 10.65, "PUBLISHED OUTPUT", 10, bold=True)
label(19.6, 10.2,  "CenterlineResult msg", 9, C_ANN)

items = [
    ("road_present", "bool"),
    ("road_confidence", "float32"),
    ("peak_confidence", "float32  (mean)"),
    ("row_confidences", "float32[3]"),
    ("points", "Point32[]  x,y,z"),
    ("", ""),
    ("visual topic:", "sensor_msgs/Image"),
    ("  guide lines", "dashed rows"),
    ("  debug bars", "sigmoid response"),
    ("  peak circles", "per-row peaks"),
    ("  Bezier curve", "centerline"),
    ("  fork branches", "if 2 r1 peaks"),
]
yi = 9.65
for k, v in items:
    if k == "":
        ax.plot([17.8, 21.4], [yi - 0.1, yi - 0.1], color="#557799", lw=0.6, ls="--", zorder=3)
        yi -= 0.3
        continue
    label(17.9, yi, k, 8, C_ANN, ha="left")
    label(21.4, yi, v, 7.5, C_NOTE, ha="right")
    yi -= 0.57

# arrows classifier → combined
arrow(17.3, 10.6, 17.6, 10.6)
# arrows row heads → combined
for ry in row_ys:
    arrow(17.3, ry - 0.0, 17.6, ry - 0.0)

# ── ANNOTATIONS ───────────────────────────────────────────────────────────────
# Note about scan rows
ax.text(0.3, 2.5,
        "Scan rows r0/r1/r2:\n"
        "fractions of full image\n"
        "height stored in checkpoint\n"
        "(overrideable in infer_config)",
        fontsize=7.5 * FS, color=C_NOTE, va="top", zorder=5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a2040", edgecolor="#335577", alpha=0.85))

ax.text(0.3, 4.8,
        "Non-uniform buckets:\n"
        "centre buckets narrower\n"
        "(5× higher res at road)\n"
        "edge buckets wider\n"
        "(sky / grass less critical)",
        fontsize=7.5 * FS, color=C_NOTE, va="top", zorder=5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a2040", edgecolor="#335577", alpha=0.85))

ax.text(0.3, 8.0,
        "Dual-path design:\n"
        "Classifier = global (GAP)\n"
        "Row heads = spatial\n"
        "Both share backbone\n"
        "features [B, 512, 15, 20]",
        fontsize=7.5 * FS, color=C_NOTE, va="top", zorder=5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a2040", edgecolor="#335577", alpha=0.85))

# ── SECTION LABELS ────────────────────────────────────────────────────────────
for tx, ty, txt in [
    (12.2, 8.85, "Global path: road/no-road decision"),
    (12.2, 0.35, "Spatial path: per-row centerline position"),
]:
    ax.text(tx, ty, txt, fontsize=7.5 * FS, color="#778899", ha="center", va="center",
            style="italic", zorder=5)

plt.tight_layout(pad=0.3)
out = "/home/rosey1211/code/road_code/v2/architecture.png"
plt.savefig(out, dpi=210, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {out}")
