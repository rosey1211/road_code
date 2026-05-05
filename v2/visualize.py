# ─── road_code/visualize.py ───────────────────────────────────────────────────
"""
Centerline visualization utilities — usable during training and inference.

Public API
----------
overlay_predictions(image, outputs, model, threshold, gt_masks)
    Draw predicted (and optionally ground-truth) centerline points on an image.
    Returns an annotated uint8 numpy array (HWC, RGB).

show_training_batch(images, outputs, model, gt_masks, n_show, title)
    Show a matplotlib grid of training/validation images with overlays.
    Call at the end of an epoch or whenever you want a visual check.

Coordinate mapping
------------------
All positions are computed from *fractions* of the display image dimensions,
so the overlay is correct at any resolution — whether showing the raw camera
frame, a dataset thumbnail, or the model's 320×240 internal size.
    row_y  = row_fraction   × display_height
    col_x  = bucket_center_fraction × display_width
             where bucket_center_fraction = (bucket_idx + 0.5) / n_buckets
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless-safe; switch to "TkAgg" / "Qt5Agg" if interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


# ─── Colour palette (RGB tuples) ──────────────────────────────────────────────
_PRED_COLOUR = (0,   220,  0)    # green  — predicted buckets
_GT_COLOUR   = (255, 200,  0)    # yellow — ground-truth buckets
_ROW_COLOUR  = (100, 180, 255)   # blue   — row guide lines
_TICK_COLOUR = (0,    60, 160)   # dark blue — bucket boundary hash marks
_CROP_COLOUR = (180,  80, 220)   # violet — crop boundary lines
_TEXT_BG     = (0,   0,   0)     # black  — label background


# ─── Internal helpers ─────────────────────────────────────────────────────────
def _to_numpy_rgb(image) -> np.ndarray:
    """Accept PIL Image, uint8 numpy (HWC), or float32 numpy (HWC) → uint8 HWC."""
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    if img.ndim == 2:                       # grayscale → RGB
        img = np.stack([img] * 3, axis=-1)
    return img.copy()


def _denorm_tensor(tensor: torch.Tensor, norm_mean, norm_std) -> np.ndarray:
    """Denormalize a CHW float tensor back to HWC uint8 numpy."""
    t = tensor.cpu().float().clone()
    for c in range(3):
        t[c] = t[c] * norm_std[c] + norm_mean[c]
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def _draw_circle(canvas: np.ndarray, cx: int, cy: int, radius: int,
                 colour, thickness: int = -1):
    """Draw a filled or outlined circle without OpenCV."""
    h, w = canvas.shape[:2]
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    for y in range(y0, y1):
        for x in range(x0, x1):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if thickness == -1:             # filled
                if dist <= radius:
                    canvas[y, x] = colour
            else:                           # outlined
                if abs(dist - radius) <= thickness:
                    canvas[y, x] = colour


def _draw_hline(canvas: np.ndarray, y: int, colour, dash: int = 8):
    """Draw a dashed horizontal line."""
    h, w = canvas.shape[:2]
    if 0 <= y < h:
        for x in range(0, w):
            if (x // dash) % 2 == 0:
                canvas[y, x] = colour


# ─── Core overlay function ────────────────────────────────────────────────────
def overlay_predictions(image,
                        outputs,
                        model,
                        threshold: float = 0.5,
                        gt_masks=None,
                        norm_mean=None,
                        norm_std=None) -> np.ndarray:
    """
    Draw centerline predictions (and optional ground truth) onto an image.

    Parameters
    ----------
    image       : PIL Image | uint8/float32 numpy HWC | CHW float32 tensor
                  If a normalised tensor, supply norm_mean and norm_std.
    outputs     : tuple (cls_logits, r0_logits, r1_logits, r2_logits) from model.forward()
                  Each is a 1-D or [1,N] tensor.
    model       : RoadCNN instance — supplies row_fractions, n_buckets, norm_mean/std.
    threshold   : float — sigmoid threshold above which a bucket is considered active.
    gt_masks    : FloatTensor [3, N_BUCKETS] or None — ground-truth binary masks.
    norm_mean   : override model.norm_mean (use if passing image without a model instance).
    norm_std    : override model.norm_std.

    Returns
    -------
    canvas : uint8 numpy array (HWC, RGB) with overlay drawn.
    """
    # ── Resolve normalisation constants ──────────────────────────────────────
    mean = norm_mean if norm_mean is not None else (model.norm_mean if model else None)
    std  = norm_std  if norm_std  is not None else (model.norm_std  if model else None)

    # ── Convert image to uint8 numpy HWC ─────────────────────────────────────
    if isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] == 3:     # CHW tensor
            if mean is not None and std is not None:
                canvas = _denorm_tensor(image, mean, std)
            else:
                canvas = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected tensor shape: {image.shape}")
    else:
        canvas = _to_numpy_rgb(image)

    display_h, display_w = canvas.shape[:2]

    # ── Decode outputs ────────────────────────────────────────────────────────
    cls_logits, r0_logits, r1_logits, r2_logits = [
        o.detach().cpu().squeeze(0) for o in outputs
    ]
    row_logits_list = [r0_logits, r1_logits, r2_logits]

    road_present = cls_logits.softmax(0)[1].item() > 0.5
    cls_conf     = cls_logits.softmax(0)[1].item()

    n_buckets     = model.n_buckets
    row_fractions = model.row_fractions

    bar_max_h  = max(4, display_h // 8)   # max bar height in pixels
    tick_h     = max(2, display_h // 40)
    tick_thick = max(1, display_w // 160)  # 1px at 160w, 2px at 320w, etc.

    # Per-row edge tables from model
    def _row_edges(r):
        edges = model.bucket_edges_all[r]
        bx0 = [max(0,           int(edges[b].item()     * display_w)) for b in range(n_buckets)]
        bx1 = [min(display_w-1, max(int(edges[b+1].item() * display_w), bx0[b]+1))
               for b in range(n_buckets)]
        exs = [int(edges[b].item() * display_w) for b in range(n_buckets + 1)]
        return bx0, bx1, exs

    # ── Draw row guide lines + bucket boundary hash marks ────────────────────
    for ri, frac in enumerate(row_fractions):
        row_y = int(frac * display_h)
        _draw_hline(canvas, row_y, _ROW_COLOUR, dash=10)
        _, _, exs = _row_edges(ri)
        for ex in exs:
            if 0 <= ex < display_w:
                y0t = max(0, row_y - tick_h)
                y1t = min(display_h, row_y + tick_h + 1)
                canvas[y0t:y1t, max(0, ex - tick_thick // 2):min(display_w, ex + (tick_thick + 1) // 2 + 1)] = _TICK_COLOUR

    # ── Draw ground-truth bars (yellow, translucent vertical bar) ────────────
    _GT_ALPHA = 0.35   # 0 = invisible, 1 = opaque
    if gt_masks is not None:
        gt        = gt_masks.cpu().float()
        gt_colour = np.array(_GT_COLOUR, dtype=np.float32)
        for r, frac in enumerate(row_fractions):
            row_y = int(frac * display_h)
            y0 = max(0, row_y - bar_max_h)
            y1 = row_y
            bx0, bx1, _ = _row_edges(r)
            for b in range(n_buckets):
                if gt[r, b].item() > 0.9:
                    region = canvas[y0:y1, bx0[b]:bx1[b]+1].astype(np.float32)
                    canvas[y0:y1, bx0[b]:bx1[b]+1] = np.clip(
                        region * (1 - _GT_ALPHA) + gt_colour * _GT_ALPHA, 0, 255
                    ).astype(np.uint8)

    # ── Draw predicted probability bars (green, translucent, height ∝ prob) ──
    _PRED_ALPHA  = 0.45
    _pred_colour = np.array(_PRED_COLOUR, dtype=np.float32)
    if road_present:
        for r, (frac, logits) in enumerate(zip(row_fractions, row_logits_list)):
            row_y = int(frac * display_h)
            probs = torch.sigmoid(logits)
            bx0, bx1, _ = _row_edges(r)
            for b in range(n_buckets):
                p = probs[b].item()
                if p < 0.01:
                    continue
                bar_h = max(1, int(p * bar_max_h))
                y1 = row_y
                y0 = max(0, row_y - bar_h)
                region = canvas[y0:y1, bx0[b]:bx1[b]+1].astype(np.float32)
                canvas[y0:y1, bx0[b]:bx1[b]+1] = np.clip(
                    region * (1 - _PRED_ALPHA) + _pred_colour * _PRED_ALPHA, 0, 255
                ).astype(np.uint8)

    # ── Crop boundary lines (violet) ─────────────────────────────────────────
    crop_top    = getattr(model, "crop_top",    0.0) if model else 0.0
    crop_bottom = getattr(model, "crop_bottom", 0.0) if model else 0.0
    if crop_top > 0:
        _draw_hline(canvas, int(crop_top * display_h), _CROP_COLOUR, dash=6)
    if crop_bottom > 0:
        _draw_hline(canvas, int((1.0 - crop_bottom) * display_h), _CROP_COLOUR, dash=6)

    # ── Road/no-road label ────────────────────────────────────────────────────
    label      = f"ROAD {cls_conf*100:.0f}%" if road_present else f"NO ROAD {(1-cls_conf)*100:.0f}%"
    label_col  = _PRED_COLOUR if road_present else (220, 50, 50)
    # Simple text via matplotlib annotation is added in show_training_batch;
    # here we tint the top-left corner as a quick indicator.
    tint_h = max(4, display_h // 40)
    progress = int(cls_conf * display_w) if road_present else int((1 - cls_conf) * display_w)
    canvas[:tint_h, :progress] = label_col

    return canvas, label


# ─── Training batch visualizer ────────────────────────────────────────────────
def show_training_batch(images,
                        outputs,
                        model,
                        gt_masks=None,
                        n_show: int = 8,
                        title: str = "Predictions",
                        threshold: float = 0.5,
                        save_path: str = None):
    """
    Display / save a grid of images with centerline overlays.

    Typically called at the end of a validation epoch.

    Parameters
    ----------
    images    : FloatTensor [B, 3, H, W] — normalised batch from the dataloader
    outputs   : tuple (cls_logits, r0, r1, r2) each [B, *]
    model     : RoadCNN instance
    gt_masks  : FloatTensor [B, 3, N_BUCKETS] or None
    n_show    : how many images to include in the grid (max B)
    title     : figure title
    threshold : sigmoid threshold for predicted bucket activation
    save_path : if given, saves the figure to this path instead of / in addition to showing

    Returns
    -------
    fig : matplotlib Figure
    """
    n = min(n_show, images.size(0))
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3 + 0.5))
    fig.suptitle(title, fontsize=12)
    axes = np.array(axes).reshape(-1)   # flatten regardless of shape

    # Unpack per-sample outputs
    cls_batch = outputs[0]
    row_batches = outputs[1:]

    for i in range(len(axes)):
        ax = axes[i]
        if i >= n:
            ax.axis("off")
            continue

        sample_outputs = (
            cls_batch[i:i+1],
            row_batches[0][i:i+1],
            row_batches[1][i:i+1],
            row_batches[2][i:i+1],
        )
        sample_gt = gt_masks[i] if gt_masks is not None else None

        canvas, label = overlay_predictions(
            images[i], sample_outputs, model,
            threshold=threshold, gt_masks=sample_gt
        )

        ax.imshow(canvas)
        ax.set_title(label, fontsize=8,
                     color="green" if label.startswith("ROAD") else "red")
        ax.axis("off")

    # Legend
    pred_patch = mpatches.Patch(color=(0, 220/255, 0),       label="Prediction")
    gt_patch   = mpatches.Patch(color=(1, 200/255, 0),       label="Ground truth")
    row_patch  = mpatches.Patch(color=(100/255, 180/255, 1), label="Row guide")
    crop_patch = mpatches.Patch(color=(180/255, 80/255, 220/255), label="Crop boundary")
    fig.legend(handles=[pred_patch, gt_patch, row_patch, crop_patch],
               loc="lower center", ncol=4, fontsize=8, framealpha=0.7)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ─── Training metrics curves ──────────────────────────────────────────────────
def save_metrics_plot(history: dict, save_path: str, best_epoch: int = None):
    """
    Save a 3-panel loss / accuracy curve that is updated after every epoch.

    Parameters
    ----------
    history    : dict with lists keyed by
                 'tr_loss', 'va_loss',
                 'tr_cls',  'va_cls',
                 'tr_row',  'va_row',
                 'tr_acc',  'va_acc'
    save_path  : file to write (PNG)
    best_epoch : 1-based epoch index to mark with a star (optional)
    """
    epochs = list(range(1, len(history["tr_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle("Training progress", fontsize=11)

    _TRAIN = "#4C9BE8"   # blue
    _VAL   = "#E8854C"   # orange

    # ── Panel 1: total loss ───────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["tr_loss"], color=_TRAIN, label="train")
    ax.plot(epochs, history["va_loss"], color=_VAL,   label="val", linestyle="--")
    ax.set_title("Total loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Panel 2: road classification accuracy ─────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["tr_acc"]], color=_TRAIN, label="train")
    ax.plot(epochs, [a * 100 for a in history["va_acc"]], color=_VAL,   label="val", linestyle="--")
    ax.set_title("Road classification accuracy (%)")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 101)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Panel 3: row (bucket) loss ────────────────────────────────────────────
    ax = axes[2]
    ax.plot(epochs, history["tr_row"], color=_TRAIN, label="train")
    ax.plot(epochs, history["va_row"], color=_VAL,   label="val", linestyle="--")
    ax.set_title("Row (bucket) loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Best-epoch marker on all panels ──────────────────────────────────────
    if best_epoch is not None and 1 <= best_epoch <= len(epochs):
        for ax in axes:
            ax.axvline(best_epoch, color="gold", linewidth=1.2,
                       linestyle=":", label=f"best (ep {best_epoch})")
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Single-image inference overlay (no matplotlib) ──────────────────────────
def overlay_inference(frame_bgr_np, outputs, model, threshold=0.5):
    """
    Lightweight overlay for real-time inference with OpenCV frames.

    Parameters
    ----------
    frame_bgr_np : uint8 numpy HWC in BGR (as returned by cv2.VideoCapture)
    outputs      : tuple (cls_logits, r0, r1, r2) each [1, *]
    model        : RoadCNN instance
    threshold    : sigmoid threshold

    Returns
    -------
    annotated frame in BGR uint8 — ready for cv2.imshow
    """
    # BGR → RGB for our drawing helper, then back
    rgb = frame_bgr_np[:, :, ::-1].copy()
    canvas, label = overlay_predictions(rgb, outputs, model, threshold=threshold)
    return canvas[:, :, ::-1]   # RGB → BGR


# ─── Quick standalone test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import config

    config.N_BUCKETS = 20
    config.init()

    from model import RoadCNN

    model = RoadCNN(
        image_width   = config.IMAGE_WIDTH,
        image_height  = config.IMAGE_HEIGHT,
        n_buckets     = config.N_BUCKETS,
        row_fractions = config.ROW_FRACTIONS,
        norm_mean     = config.NORM_MEAN,
        norm_std      = config.NORM_STD,
    )

    # Fake outputs: road present, buckets 8 and 12 active on all rows
    B = 4
    cls_logits = torch.tensor([[3.0, -3.0]] * B)          # road present
    bucket_logits = torch.zeros(B, config.N_BUCKETS)
    bucket_logits[:, 8]  = 4.0
    bucket_logits[:, 12] = 4.0                             # simulate fork

    outputs = (cls_logits, bucket_logits, bucket_logits, bucket_logits)

    # Fake ground truth
    gt = torch.zeros(B, config.N_ROWS, config.N_BUCKETS)
    gt[:, :, 9] = 1.0

    # Fake images (random noise stand-in)
    images = torch.rand(B, 3, 240, 320)

    fig = show_training_batch(images, outputs, model, gt_masks=gt,
                              n_show=B, title="Visualize test",
                              save_path="visualize_test.png")
    plt.show()
    print("Done. Check visualize_test.png")
