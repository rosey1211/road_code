"""
Generates four supplementary diagrams for RoadCNN_Architecture.md.
Run: python draw_diagrams.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DPI = 180
BG  = "#1a1a2e"
C_TEXT  = "#f0f0f0"
C_NOTE  = "#aaddff"
C_ANN   = "#ffdd88"
C_GREEN = "#2ecc71"
C_RED   = "#e74c3c"
C_BLUE  = "#3498db"
C_PURP  = "#9b59b6"
C_ORANGE= "#e67e22"
C_TEAL  = "#1abc9c"

def savefig(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 — Bucket spacing: uniform vs non-uniform
# ═════════════════════════════════════════════════════════════════════════════
def diagram_buckets():
    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Bucket Spacing: Uniform vs. Non-Uniform  (N=20 shown for clarity)",
                 color=C_TEXT, fontsize=14, fontweight="bold", y=0.98)

    N = 20
    margin = 0.10
    R = 5.0

    # ── compute non-uniform edges ─────────────────────────────────────────────
    def edges_nonlinear(N, margin, R):
        active = 1.0 - 2 * margin
        c0 = 3.0 / (R + 2.0)
        c2 = 12.0 * (R - 1.0) / (R + 2.0)
        out = []
        for i in range(N + 1):
            t = i / N
            p = c0 * t + c2 * ((t - 0.5)**3 + 0.125) / 3.0
            out.append(margin + p * active)
        return out

    uniform_edges    = [margin + i / N * (1 - 2*margin) for i in range(N + 1)]
    nonuniform_edges = edges_nonlinear(N, margin, R)

    for ax, edges, title, colour in zip(
            axes,
            [uniform_edges, nonuniform_edges],
            ["Uniform spacing  (BUCKET_NONLINEARITY = 1.0)",
             "Non-uniform spacing  (BUCKET_NONLINEARITY = 5.0)  —  centre buckets 5× narrower"],
            [C_BLUE, C_GREEN]):

        ax.set_facecolor("#0d1117")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.3, 1.3)
        ax.axis("off")
        ax.set_title(title, color=C_TEXT, fontsize=11, pad=6)

        bar_y, bar_h = 0.35, 0.45

        # draw buckets
        for i in range(N):
            x0, x1 = edges[i], edges[i+1]
            mid = (x0 + x1) / 2
            # alternate shade
            shade = 0.55 if i % 2 == 0 else 0.38
            fc = tuple(c * shade for c in matplotlib.colors.to_rgb(colour))
            rect = mpatches.FancyBboxPatch((x0 + 0.002, bar_y), x1 - x0 - 0.004, bar_h,
                                           boxstyle="round,pad=0.002",
                                           facecolor=fc, edgecolor=colour,
                                           linewidth=0.8, zorder=3)
            ax.add_patch(rect)
            if i % 4 == 0 or i == N - 1:
                ax.text(mid, bar_y - 0.12, str(i), color=C_NOTE, fontsize=7,
                        ha="center", va="top", zorder=4)

        # margin shading
        for x0, x1 in [(0, margin), (1 - margin, 1)]:
            rect = mpatches.Rectangle((x0, bar_y - 0.05), x1 - x0, bar_h + 0.1,
                                       facecolor="#330000", edgecolor=C_RED,
                                       linewidth=0.8, alpha=0.5, zorder=2)
            ax.add_patch(rect)
        ax.text(margin / 2, bar_y + bar_h + 0.08, "margin\n(excluded)",
                color=C_RED, fontsize=7.5, ha="center", va="bottom")
        ax.text(1 - margin / 2, bar_y + bar_h + 0.08, "margin\n(excluded)",
                color=C_RED, fontsize=7.5, ha="center", va="bottom")

        # width annotations for centre and edge bucket
        ctr = N // 2
        for bi, label_txt, col in [(0, "edge bucket\n(wide)", C_ORANGE),
                                    (ctr, "centre bucket\n(narrow)", C_TEAL)]:
            ex0, ex1 = edges[bi], edges[bi + 1]
            ey = bar_y + bar_h + 0.28
            ax.annotate("", xy=(ex1, ey), xytext=(ex0, ey),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=1.4))
            ax.text((ex0 + ex1) / 2, ey + 0.10, label_txt,
                    color=col, fontsize=8, ha="center", va="bottom")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "/home/rosey1211/code/road_code/v2/diag_buckets.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 — Strip extraction from the feature map
# ═════════════════════════════════════════════════════════════════════════════
def diagram_strip():
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Strip Extraction from the Backbone Feature Map  [B, 512, 15, 20]",
                 color=C_TEXT, fontsize=14, fontweight="bold")

    ROWS, COLS = 15, 20
    cell_w = 0.30
    cell_h = 0.38
    ox, oy = 0.9, 0.8

    row_fracs   = [0.22,  0.50,  0.72]
    row_colours = [C_RED, C_GREEN, C_ORANGE]
    row_labels  = ["r0 — FAR row   (frac = 0.22)",
                   "r1 — MID row   (frac = 0.50)",
                   "r2 — NEAR row  (frac = 0.72)"]
    strip_desc  = ["strip → [B, 512, 3, 20]"] * 3

    # feature map grid
    for r in range(ROWS):
        for c in range(COLS):
            x = ox + c * cell_w
            y = oy + (ROWS - 1 - r) * cell_h
            highlight = None
            for frac, col in zip(row_fracs, row_colours):
                target_row = int(frac * ROWS)
                if abs(r - target_row) <= 1:
                    highlight = col
                    break
            if highlight:
                fc, alpha = highlight, 0.60
            else:
                fc, alpha = "#2a3a5a", 0.75
            rect = mpatches.Rectangle((x, y), cell_w - 0.025, cell_h - 0.025,
                                       facecolor=fc, edgecolor="#334455",
                                       linewidth=0.4, alpha=alpha, zorder=3)
            ax.add_patch(rect)

    fm_cx  = ox + COLS * cell_w / 2
    fm_top = oy + ROWS * cell_h
    ax.text(fm_cx, fm_top + 0.18, f"{COLS} columns  (W = 20)",
            color=C_NOTE, fontsize=10, ha="center", va="bottom")
    ax.text(ox - 0.25, oy + ROWS * cell_h / 2, f"{ROWS} rows\n(H = 15)",
            color=C_NOTE, fontsize=10, ha="right", va="center", rotation=90)

    # row fraction axis
    ax.text(ox - 0.65, fm_top,   "0.0", color="#778899", fontsize=8.5, va="center")
    ax.text(ox - 0.65, oy,       "1.0", color="#778899", fontsize=8.5, va="center")
    ax.annotate("", xy=(ox - 0.52, oy), xytext=(ox - 0.52, fm_top),
                arrowprops=dict(arrowstyle="->", color="#445566", lw=1.0))
    ax.text(ox - 0.90, oy + ROWS * cell_h / 2, "row\nfraction",
            color="#778899", fontsize=8, ha="center", va="center", rotation=90)

    # strip labels and output boxes
    x_right = ox + COLS * cell_w + 0.20
    label_x = x_right + 0.80
    box_w, box_h = 4.6, 1.10

    for frac, col, lbl, desc in zip(row_fracs, row_colours, row_labels, strip_desc):
        target_row = int(frac * ROWS)
        cy = oy + (ROWS - 1 - target_row) * cell_h + cell_h / 2
        y0 = oy + (ROWS - 1 - (target_row + 1)) * cell_h
        y1 = oy + (ROWS - 1 - (target_row - 1)) * cell_h + cell_h

        # bracket
        for yy in [y0, y1]:
            ax.plot([x_right, x_right + 0.15], [yy, yy], color=col, lw=1.5, zorder=5)
        ax.plot([x_right + 0.15]*2, [y0, y1], color=col, lw=1.5, zorder=5)

        # arrow
        ax.annotate("", xy=(label_x - 0.05, cy), xytext=(x_right + 0.18, cy),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.6))

        # output box
        rect = mpatches.FancyBboxPatch((label_x, cy - box_h/2), box_w, box_h,
                                        boxstyle="round,pad=0.07",
                                        facecolor="#0d1a2a", edgecolor=col,
                                        linewidth=1.8, zorder=4)
        ax.add_patch(rect)
        ax.text(label_x + box_w/2, cy + 0.22, lbl,
                color=col, fontsize=10.5, ha="center", va="center", fontweight="bold", zorder=6)
        ax.text(label_x + box_w/2, cy - 0.22, desc,
                color=C_NOTE, fontsize=9.5, ha="center", va="center", zorder=6)

    plt.tight_layout()
    savefig(fig, "/home/rosey1211/code/road_code/v2/diag_strip.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 — Conv1d kernel: 3 buckets → 64 outputs
# ═════════════════════════════════════════════════════════════════════════════
def diagram_conv1d():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_title("Conv1d(512→64, kernel=3):  Local Bucket Context at Position i",
                 color=C_TEXT, fontsize=13, fontweight="bold")

    # ── three input buckets ───────────────────────────────────────────────────
    bucket_cols = [C_BLUE, C_GREEN, C_PURP]
    bucket_xs   = [1.2, 4.5, 7.8]
    bucket_lbls = ["Bucket  i−1\n(left neighbour)", "Bucket  i\n(target)", "Bucket  i+1\n(right neighbour)"]

    feat_labels = ["road colour", "edge detect", "texture", "shadow", "sky detector",
                   "gradient mag", "H-edge", "V-edge", "saturation", "brightness",
                   "…", "…  (512 total)"]

    for bx, col, lbl in zip(bucket_xs, bucket_cols, bucket_lbls):
        # bucket box
        rect = mpatches.FancyBboxPatch((bx, 2.8), 1.9, 3.6,
                                        boxstyle="round,pad=0.08",
                                        facecolor="#0d1a2a", edgecolor=col,
                                        linewidth=1.8, zorder=3)
        ax.add_patch(rect)
        ax.text(bx + 0.95, 6.6, lbl, color=col, fontsize=9.5,
                ha="center", va="bottom", fontweight="bold")

        # feature rows inside box
        for fi, fl in enumerate(feat_labels):
            fy = 6.2 - fi * 0.28
            if fy < 2.9:
                break
            alpha = 0.5 if fl.startswith("…") else 0.85
            ax.text(bx + 0.95, fy, fl, color=C_NOTE, fontsize=7,
                    ha="center", va="center", alpha=alpha)

        ax.text(bx + 0.95, 2.65, "512 features", color=col,
                fontsize=8, ha="center", va="top", style="italic")

    # ── weight matrix box ─────────────────────────────────────────────────────
    mx, my = 4.5, 0.6
    mw, mh = 4.0, 1.6
    rect = mpatches.FancyBboxPatch((mx, my), mw, mh,
                                    boxstyle="round,pad=0.08",
                                    facecolor="#1a0a2e", edgecolor=C_ANN,
                                    linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    ax.text(mx + mw/2, my + mh - 0.25, "Weight Matrix", color=C_ANN,
            fontsize=11, ha="center", va="center", fontweight="bold")
    ax.text(mx + mw/2, my + 0.65, "1,536 inputs  ×  64 outputs", color=C_TEXT,
            fontsize=10, ha="center", va="center")
    ax.text(mx + mw/2, my + 0.28, "(512 × 3 buckets = 1,536)", color="#778899",
            fontsize=8.5, ha="center", va="center")

    # ── arrows from bucket boxes to weight matrix ─────────────────────────────
    for bx, col in zip(bucket_xs, bucket_cols):
        ax.annotate("", xy=(mx + mw/2, my + mh),
                    xytext=(bx + 0.95, 2.8),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4,
                                    connectionstyle="arc3,rad=0.0"))

    # ── output: 64 nodes ──────────────────────────────────────────────────────
    out_x_start = 10.5
    out_y_base  = 0.55
    out_spacing = 0.28
    n_show = 14
    for i in range(n_show):
        cy = out_y_base + i * out_spacing
        shade = C_TEAL if i < n_show - 2 else "#445566"
        circ = plt.Circle((out_x_start, cy), 0.10, color=shade, zorder=4)
        ax.add_patch(circ)
    ax.text(out_x_start, out_y_base + (n_show - 1) * out_spacing + 0.35,
            "64 outputs\n(per bucket i)", color=C_TEAL, fontsize=10,
            ha="center", va="bottom", fontweight="bold")
    ax.text(out_x_start, out_y_base + (n_show//2) * out_spacing + 0.05,
            "…", color=C_NOTE, fontsize=14, ha="center", va="center")

    ax.annotate("", xy=(out_x_start - 0.15, out_y_base + (n_show//2) * out_spacing),
                xytext=(mx + mw + 0.1, my + mh / 2),
                arrowprops=dict(arrowstyle="->", color=C_ANN, lw=1.5))

    ax.text(6.5, 0.22, "Same 1,536×64 weights applied at every bucket position  (weight sharing)",
            color="#778899", fontsize=8.5, ha="center", va="bottom", style="italic")

    plt.tight_layout()
    savefig(fig, "/home/rosey1211/code/road_code/v2/diag_conv1d.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 4 — Sigmoid output + cluster detection
# ═════════════════════════════════════════════════════════════════════════════
def diagram_clusters():
    rng = np.random.default_rng(42)
    N = 40

    # synthetic sigmoid output: one main cluster ~bucket 18, secondary ~bucket 30
    probs = np.zeros(N)
    for centre, amp, width in [(18, 0.88, 3.5), (30, 0.62, 2.8)]:
        for b in range(N):
            probs[b] += amp * np.exp(-0.5 * ((b - centre) / width) ** 2)
    probs += rng.uniform(0.0, 0.06, N)
    probs = np.clip(probs, 0, 1)

    peak_conf  = probs.max()
    thresh_frac = 0.45
    threshold  = max(0.05, peak_conf * thresh_frac)

    # find clusters
    clusters = []
    in_cluster = False
    for i, p in enumerate(probs):
        if p >= threshold:
            if not in_cluster:
                start = i
                in_cluster = True
        else:
            if in_cluster:
                clusters.append((start, i - 1))
                in_cluster = False
    if in_cluster:
        clusters.append((start, N - 1))

    # weighted centroids
    centroids = []
    for lo, hi in clusters:
        idxs = np.arange(lo, hi + 1)
        w    = probs[idxs]
        cx   = (idxs * w).sum() / w.sum()
        centroids.append((cx, probs[int(round(cx))]))

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"hspace": 0.55})
    fig.patch.set_facecolor(BG)
    fig.suptitle("Sigmoid Output, Cluster Detection, and Peak Extraction",
                 color=C_TEXT, fontsize=14, fontweight="bold", y=0.98)

    cluster_colours = [C_GREEN, C_ORANGE]

    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor("#0d1117")
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(-0.08, 1.22)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#334455")
        ax.tick_params(colors=C_NOTE, labelsize=8)
        ax.set_xlabel("Bucket index  (0 = left edge, 39 = right edge)", color=C_NOTE, fontsize=9)
        ax.set_ylabel("Sigmoid probability", color=C_NOTE, fontsize=9)

        # raw bars
        for i, p in enumerate(probs):
            col = "#2a3a5a"
            for ci, (lo, hi) in enumerate(clusters):
                if lo <= i <= hi:
                    col = cluster_colours[ci % len(cluster_colours)]
                    break
            alpha = 0.85 if col != "#2a3a5a" else 0.5
            ax.bar(i, p, color=col, width=0.75, alpha=alpha, zorder=3)

        # threshold line
        ax.axhline(threshold, color=C_RED, lw=1.4, ls="--", zorder=4)
        ax.text(N - 0.3, threshold + 0.03,
                f"threshold = {threshold:.2f}\n(peak × {thresh_frac})",
                color=C_RED, fontsize=8, ha="right", va="bottom")

        if ax_idx == 0:
            ax.set_title("Raw sigmoid output — bars coloured by cluster membership",
                         color=C_TEXT, fontsize=10, pad=6)
        else:
            ax.set_title("Cluster peaks — weighted centroid marks the detected position",
                         color=C_TEXT, fontsize=10, pad=6)

            # cluster brackets + centroid markers
            for ci, ((lo, hi), (cx, cy_approx), col) in \
                    enumerate(zip(clusters, centroids, cluster_colours)):
                # bracket under bars
                bracket_y = -0.055
                ax.plot([lo - 0.4, hi + 0.4], [bracket_y, bracket_y], color=col, lw=2.0)
                ax.plot([lo - 0.4, lo - 0.4], [bracket_y, bracket_y + 0.025], color=col, lw=2.0)
                ax.plot([hi + 0.4, hi + 0.4], [bracket_y, bracket_y + 0.025], color=col, lw=2.0)
                ax.text((lo + hi) / 2, bracket_y - 0.025,
                        f"Cluster {ci+1}  (buckets {lo}–{hi})",
                        color=col, fontsize=8.5, ha="center", va="top", fontweight="bold")

                # centroid vertical line
                ax.axvline(cx, color=col, lw=2.0, ls="-", alpha=0.9, zorder=5)

                # diamond marker at peak height
                peak_h = probs[int(round(cx))]
                ax.plot(cx, peak_h + 0.06, marker="v", color=col,
                        markersize=10, zorder=6)

                label_y = 1.12
                cx_frac = cx / (N - 1)
                ax.text(cx, label_y,
                        f"cx_frac = {cx_frac:.3f}\n(conf = {peak_h:.2f})",
                        color=col, fontsize=8.5, ha="center", va="bottom",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a1a2a",
                                  edgecolor=col, alpha=0.9))

            # primary / secondary labels
            ax.text(centroids[0][0], 0.75, "PRIMARY\npeak", color=C_GREEN,
                    fontsize=8, ha="center", va="bottom", style="italic")
            ax.text(centroids[1][0], 0.50, "SECONDARY\npeak", color=C_ORANGE,
                    fontsize=8, ha="center", va="bottom", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "/home/rosey1211/code/road_code/v2/diag_clusters.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 5 — Bucket pooling: how 20 feature map columns → 40 bucket vectors
# ═════════════════════════════════════════════════════════════════════════════
def diagram_bucket_pooling():
    N       = 40
    W       = 20      # feature map columns
    margin  = 0.10
    R       = 5.0

    def compute_edges(N, margin, R):
        active = 1.0 - 2 * margin
        c0 = 3.0 / (R + 2.0)
        c2 = 12.0 * (R - 1.0) / (R + 2.0)
        out = []
        for i in range(N + 1):
            t = i / N
            p = c0 * t + c2 * ((t - 0.5)**3 + 0.125) / 3.0
            out.append(margin + p * active)
        return np.array(out)

    edges = compute_edges(N, margin, R)

    # for each bucket, compute which feature map columns contribute
    # x0_px, x1_px = pixel span in feature map (W=20)
    bucket_spans = []
    for b in range(N):
        x0 = max(0,   int(edges[b]   * W))
        x1 = min(W,   int(edges[b+1] * W))
        x1 = max(x1, x0 + 1)   # at least 1 column always
        bucket_spans.append((x0, x1))

    # assign a display colour to each feature-map column so pooling is obvious
    col_colours = plt.cm.tab20(np.linspace(0, 1, W))

    fig, axes = plt.subplots(3, 1, figsize=(15, 9),
                             gridspec_kw={"height_ratios": [1.4, 0.9, 1.4],
                                          "hspace": 0.55})
    fig.patch.set_facecolor(BG)
    fig.suptitle("Bucket Pooling: Feature Map Columns → Bucket Feature Vectors",
                 color=C_TEXT, fontsize=14, fontweight="bold", y=0.98)

    # ── Panel 1: feature map strip columns ───────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor("#0d1117")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.set_title("Step 1 — Feature Map Strip  [B, 512, 3, 20]:  20 columns",
                  color=C_TEXT, fontsize=11)

    col_w = 1.0 / W
    for c in range(W):
        x0c = c * col_w
        fc  = col_colours[c]
        rect = mpatches.Rectangle((x0c + 0.002, 0.15), col_w - 0.004, 0.65,
                                   facecolor=fc, edgecolor="#ffffff",
                                   linewidth=0.8, alpha=0.85, zorder=3)
        ax0.add_patch(rect)
        ax0.text(x0c + col_w / 2, 0.04, str(c), color=C_NOTE,
                 fontsize=7.5, ha="center", va="bottom")

    ax0.text(0.5, 0.95, "Each column holds 512 feature values  (one per channel)",
             color=C_ANN, fontsize=9, ha="center", va="top")
    ax0.text(-0.01, 0.48, "512\nchannels", color=C_NOTE, fontsize=8,
             ha="right", va="center")

    # margin indicators
    for x, lbl in [(margin, "← margin"), (1 - margin, "margin →")]:
        ax0.axvline(x, color=C_RED, lw=1.5, ls="--", alpha=0.8)
    ax0.text(margin / 2, 0.88, "excluded", color=C_RED, fontsize=8, ha="center")
    ax0.text(1 - margin / 2, 0.88, "excluded", color=C_RED, fontsize=8, ha="center")

    # ── Panel 2: mapping diagram (columns → buckets) ──────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor("#0d1117")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title("Step 2 — Column-to-Bucket Mapping  (mean-pool each column span)",
                  color=C_TEXT, fontsize=11)

    # draw connector lines from source columns to destination buckets
    # show a representative sample to avoid clutter
    highlight_buckets = list(range(0, 5)) + list(range(17, 23)) + list(range(35, 40))
    for b in highlight_buckets:
        x0b = edges[b]
        x1b = edges[b + 1]
        bx_mid = (x0b + x1b) / 2
        x0c, x1c = bucket_spans[b]
        # source column range in fraction space
        src_lo = x0c / W
        src_hi = x1c / W
        src_mid = (src_lo + src_hi) / 2
        col = col_colours[min(x0c, W - 1)]
        ax1.annotate("", xy=(bx_mid, 0.05), xytext=(src_mid, 0.95),
                     arrowprops=dict(arrowstyle="->", color=col,
                                     lw=1.0, alpha=0.7))

    ax1.text(0.25, 0.50, "Edge buckets span\nmultiple columns\n(wide → lower res)",
             color=C_ORANGE, fontsize=9, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a0d00",
                       edgecolor=C_ORANGE, alpha=0.85))
    ax1.text(0.75, 0.50, "Centre buckets\nshare one column\n(narrow → higher res)",
             color=C_TEAL, fontsize=9, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#001a1a",
                       edgecolor=C_TEAL, alpha=0.85))

    ax1.text(0.5, 0.98, "20 feature map columns", color="#778899",
             fontsize=8, ha="center", va="top", style="italic")
    ax1.text(0.5, 0.02, "40 buckets", color="#778899",
             fontsize=8, ha="center", va="bottom", style="italic")

    # ── Panel 3: resulting 40 bucket vectors ─────────────────────────────────
    ax2 = axes[2]
    ax2.set_facecolor("#0d1117")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("Step 3 — Result: 40 Bucket Feature Vectors  [B, 512, 40]",
                  color=C_TEXT, fontsize=11)

    for b in range(N):
        x0b = edges[b]
        x1b = edges[b + 1]
        # colour based on dominant source column
        src_col_idx = min(bucket_spans[b][0], W - 1)
        fc = col_colours[src_col_idx]
        rect = mpatches.Rectangle((x0b + 0.001, 0.15), x1b - x0b - 0.002, 0.65,
                                   facecolor=fc, edgecolor="#ffffff",
                                   linewidth=0.6, alpha=0.85, zorder=3)
        ax2.add_patch(rect)
        if b % 8 == 0 or b == N - 1:
            ax2.text((x0b + x1b) / 2, 0.04, str(b), color=C_NOTE,
                     fontsize=7.5, ha="center", va="bottom")

    # annotate centre vs edge resolution
    ctr_lo = edges[N // 2 - 1]
    ctr_hi = edges[N // 2 + 1]
    ax2.annotate("", xy=(ctr_hi, 0.88), xytext=(ctr_lo, 0.88),
                 arrowprops=dict(arrowstyle="<->", color=C_TEAL, lw=1.5))
    ax2.text((ctr_lo + ctr_hi) / 2, 0.96, "narrow\n(high res)",
             color=C_TEAL, fontsize=8, ha="center", va="top")

    edge_lo = edges[0]
    edge_hi = edges[2]
    ax2.annotate("", xy=(edge_hi, 0.88), xytext=(edge_lo, 0.88),
                 arrowprops=dict(arrowstyle="<->", color=C_ORANGE, lw=1.5))
    ax2.text((edge_lo + edge_hi) / 2, 0.96, "wide\n(low res)",
             color=C_ORANGE, fontsize=8, ha="center", va="top")

    ax2.text(0.5, 0.06,
             "Each bucket vector = mean of feature values over its column span  →  passed to Conv1d",
             color=C_ANN, fontsize=9, ha="center", va="bottom")

    ax2.text(-0.01, 0.48, "512\nchannels", color=C_NOTE, fontsize=8,
             ha="right", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "/home/rosey1211/code/road_code/v2/diag_bucket_pooling.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    diagram_buckets()
    diagram_strip()
    diagram_conv1d()
    diagram_clusters()
    diagram_bucket_pooling()
    print("All diagrams saved.")
