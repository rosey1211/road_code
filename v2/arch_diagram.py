# ─── road_code/arch_diagram.py ───────────────────────────────────────────────
"""
Generate a PNG architecture diagram for RoadCNN using Pillow only.
No matplotlib, no graphviz, no display required.

Usage
-----
python arch_diagram.py              # saves arch.png next to this file
python arch_diagram.py --out /tmp/arch.png
"""

import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

# ─── Config (must match model.py / config.py) ─────────────────────────────────
N_BUCKETS   = 80
IMG_W       = 320
IMG_H       = 240
ROW_FRACS   = [0.70, 0.58, 0.48]
ROW_YS      = [int(f * IMG_H) for f in ROW_FRACS]

# ─── Visual style ─────────────────────────────────────────────────────────────
BG          = (18,  24,  38)
C_INPUT     = ( 28,  68, 140)   # blue   — input
C_RESIZE    = ( 28,  68, 140)
C_CONV      = (  6,  95,  70)   # green  — conv blocks
C_POOL      = (  4,  55,  38)   # dark green — pool
C_GAP       = ( 72,  28, 145)   # purple — gap
C_NECK      = (130,  80,  14)   # amber  — neck
C_CLS       = (130,  28,  28)   # red    — classifier head
C_ROW       = (120,  55,  12)   # orange — row heads
C_ARROW     = (120, 130, 150)
C_TEXT      = (240, 240, 240)
C_DIM       = (160, 200, 180)   # tensor-shape colour


def _font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _font_bold(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ─── Drawing primitives ───────────────────────────────────────────────────────
def draw_box(draw, x, y, w, h, colour, radius=10):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=colour)


def draw_arrow(draw, x1, y1, x2, y2, colour=C_ARROW):
    draw.line([(x1, y1), (x2, y2)], fill=colour, width=2)
    # arrowhead
    ah = 8
    draw.polygon([(x2, y2), (x2 - 5, y2 - ah), (x2 + 5, y2 - ah)], fill=colour)


def text_block(draw, cx, cy, lines, font_bold, font_reg, line_h=20):
    """Draw multiline text centred at (cx, cy). First line uses bold font."""
    total_h = len(lines) * line_h
    y = cy - total_h // 2
    for i, (text, colour) in enumerate(lines):
        font = font_bold if i == 0 else font_reg
        bbox = draw.textbbox((0, 0), text, font=font)
        tw   = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, y), text, font=font, fill=colour)
        y += line_h


# ─── Layer spec helpers ───────────────────────────────────────────────────────
def spine_layer(label, dim_str, colour, tall=False):
    return dict(kind="spine", label=label, dim=dim_str,
                colour=colour, tall=tall)

def head_layer(label, dim_str, colour):
    return dict(kind="head", label=label, dim=dim_str, colour=colour)


# ─── Build layer list ─────────────────────────────────────────────────────────
SPINE = [
    spine_layer("Input",
                f"[B, 3, H, W]  —  any resolution",
                C_INPUT),
    spine_layer("Bilinear Resize  →  320 × 240",
                f"[B, 3, {IMG_H}, {IMG_W}]",
                C_RESIZE),
    spine_layer("Conv Block 1      3 → 64\nConv3×3 → BN → ReLU → Conv3×3 → BN → ReLU",
                f"[B, 64, {IMG_H}, {IMG_W}]",
                C_CONV, tall=True),
    spine_layer(f"MaxPool 2×2",
                f"[B, 64, {IMG_H//2}, {IMG_W//2}]",
                C_POOL),
    spine_layer("Conv Block 2      64 → 128\nConv3×3 → BN → ReLU → Conv3×3 → BN → ReLU",
                f"[B, 128, {IMG_H//2}, {IMG_W//2}]",
                C_CONV, tall=True),
    spine_layer("MaxPool 2×2",
                f"[B, 128, {IMG_H//4}, {IMG_W//4}]",
                C_POOL),
    spine_layer("Conv Block 3      128 → 256\nConv3×3 → BN → ReLU → Conv3×3 → BN → ReLU",
                f"[B, 256, {IMG_H//4}, {IMG_W//4}]",
                C_CONV, tall=True),
    spine_layer("MaxPool 2×2",
                f"[B, 256, {IMG_H//8}, {IMG_W//8}]",
                C_POOL),
    spine_layer("Conv Block 4      256 → 512\nConv3×3 → BN → ReLU → Conv3×3 → BN → ReLU",
                f"[B, 512, {IMG_H//8}, {IMG_W//8}]",
                C_CONV, tall=True),
    spine_layer("MaxPool 2×2",
                f"[B, 512, {IMG_H//16}, {IMG_W//16}]",
                C_POOL),
    spine_layer("Global Average Pool",
                "[B, 512, 1, 1]",
                C_GAP),
    spine_layer("Neck\nFlatten → Linear(512, 256) → ReLU → Dropout(0.4)",
                "[B, 256]",
                C_NECK, tall=True),
]

HEADS = [
    head_layer("Classifier Head\nLinear(256, 2)",
               "cls_logits  [B, 2]\n(CE loss)",
               C_CLS),
    head_layer(f"Row 0 Head\nLinear(256, {N_BUCKETS})\ny ≈ {ROW_FRACS[0]:.0%}  ({ROW_YS[0]} px)",
               f"row0_logits  [B, {N_BUCKETS}]\n(BCE loss)",
               C_ROW),
    head_layer(f"Row 1 Head\nLinear(256, {N_BUCKETS})\ny ≈ {ROW_FRACS[1]:.0%}  ({ROW_YS[1]} px)",
               f"row1_logits  [B, {N_BUCKETS}]\n(BCE loss)",
               C_ROW),
    head_layer(f"Row 2 Head\nLinear(256, {N_BUCKETS})\ny ≈ {ROW_FRACS[2]:.0%}  ({ROW_YS[2]} px)",
               f"row2_logits  [B, {N_BUCKETS}]\n(BCE loss)",
               C_ROW),
]


# ─── Layout constants ─────────────────────────────────────────────────────────
CANVAS_W    = 1140
BOX_W       = 560
BOX_H_STD   = 56
BOX_H_TALL  = 80
GAP_Y       = 18       # vertical gap between boxes (space for arrow)
PAD_X       = (CANVAS_W - BOX_W) // 2

HEAD_W      = 240
HEAD_H      = 90
HEAD_GAP_X  = 16
HEADS_TOTAL = len(HEADS) * HEAD_W + (len(HEADS) - 1) * HEAD_GAP_X
HEAD_START_X = (CANVAS_W - HEADS_TOTAL) // 2

FONT_MAIN   = 15
FONT_DIM    = 13
FONT_TITLE  = 18
LINE_H      = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "arch.png"))
    args = ap.parse_args()

    fb = _font_bold(FONT_MAIN)
    fr = _font(FONT_MAIN)
    fd = _font(FONT_DIM)
    ft = _font_bold(FONT_TITLE)

    # ── Pre-pass: compute total height ───────────────────────────────────────
    total_h = 60   # top margin + title
    for layer in SPINE:
        total_h += (BOX_H_TALL if layer["tall"] else BOX_H_STD) + GAP_Y
    total_h += HEAD_H + GAP_Y + 60   # heads + bottom margin

    img  = Image.new("RGB", (CANVAS_W, total_h), BG)
    draw = ImageDraw.Draw(img)

    # ── Title ─────────────────────────────────────────────────────────────────
    title = "RoadCNN Architecture"
    tbbox = draw.textbbox((0, 0), title, font=ft)
    draw.text(((CANVAS_W - (tbbox[2] - tbbox[0])) // 2, 18),
              title, font=ft, fill=C_TEXT)

    # ── Spine ─────────────────────────────────────────────────────────────────
    y = 60
    cx = CANVAS_W // 2
    prev_bot = None

    for layer in SPINE:
        bh = BOX_H_TALL if layer["tall"] else BOX_H_STD
        bx = PAD_X

        # arrow from previous box
        if prev_bot is not None:
            draw_arrow(draw, cx, prev_bot + 2, cx, y - 2)

        draw_box(draw, bx, y, BOX_W, bh, layer["colour"])

        # label lines
        raw_label = layer["label"]
        label_lines = raw_label.split("\n")
        dim_str    = layer["dim"]

        # centre text vertically in box
        n_lines = len(label_lines) + 1   # +1 for dim line
        block_h = n_lines * LINE_H
        ty = y + (bh - block_h) // 2

        for i, line in enumerate(label_lines):
            font  = fb if i == 0 else fr
            col   = C_TEXT
            tbbox = draw.textbbox((0, 0), line, font=font)
            tw    = tbbox[2] - tbbox[0]
            draw.text((cx - tw // 2, ty), line, font=font, fill=col)
            ty += LINE_H

        # dim string
        dbbox = draw.textbbox((0, 0), dim_str, font=fd)
        dw    = dbbox[2] - dbbox[0]
        draw.text((cx - dw // 2, ty), dim_str, font=fd, fill=C_DIM)

        prev_bot = y + bh
        y = prev_bot + GAP_Y

    # ── Fan-out arrows to heads ───────────────────────────────────────────────
    fan_y   = y
    head_cy = [HEAD_START_X + i * (HEAD_W + HEAD_GAP_X) + HEAD_W // 2
               for i in range(len(HEADS))]

    # horizontal bus line
    bus_y = fan_y + GAP_Y // 2
    draw.line([(head_cy[0], prev_bot + 2),
               (head_cy[0], bus_y)], fill=C_ARROW, width=2)
    draw.line([(head_cy[0], bus_y),
               (head_cy[-1], bus_y)], fill=C_ARROW, width=2)
    for hcx in head_cy:
        draw_arrow(draw, hcx, bus_y, hcx, fan_y + GAP_Y)

    # vertical drop from neck to bus
    draw.line([(cx, prev_bot + 2), (cx, bus_y)], fill=C_ARROW, width=2)
    draw.line([(cx, bus_y), (head_cy[0], bus_y)], fill=C_ARROW, width=2)

    # ── Head boxes ────────────────────────────────────────────────────────────
    hy = fan_y + GAP_Y
    for i, (head, hcx) in enumerate(zip(HEADS, head_cy)):
        hx = hcx - HEAD_W // 2
        draw_box(draw, hx, hy, HEAD_W, HEAD_H, head["colour"])

        lines   = head["label"].split("\n")
        dim_lines = head["dim"].split("\n")
        n_lines = len(lines) + len(dim_lines)
        block_h = n_lines * LINE_H
        ty = hy + (HEAD_H - block_h) // 2

        for j, line in enumerate(lines):
            font  = fb if j == 0 else fr
            tbbox = draw.textbbox((0, 0), line, font=font)
            tw    = tbbox[2] - tbbox[0]
            draw.text((hcx - tw // 2, ty), line, font=font, fill=C_TEXT)
            ty += LINE_H
        for line in dim_lines:
            dbbox = draw.textbbox((0, 0), line, font=fd)
            dw    = dbbox[2] - dbbox[0]
            draw.text((hcx - dw // 2, ty), line, font=fd, fill=C_DIM)
            ty += LINE_H

    img.save(args.out, dpi=(150, 150))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
