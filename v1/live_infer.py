# ─── road_code/live_infer.py ──────────────────────────────────────────────────
"""
Inference viewer — processes each frame in a dataset and overlays:
  • Dashed row guide lines (blue)
  • Small yellow translucent vertical bars at GT bucket positions
  • Filled green circle at the peak predicted bucket per row
  • Green lines connecting peak_r0→peak_r1 and peak_r1→peak_r2

All paths and display settings are read from infer_config.py.
CLI flags override infer_config values when provided.

Usage
-----
python live_infer.py
python live_infer.py --checkpoint /path/to/model.pth
python live_infer.py --dataset    /path/to/data/split
python live_infer.py --delay 0        # wait for keypress between images
python live_infer.py --delay 2.0      # auto-advance every 2 s
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import infer_config as cfg
from dataset import RoadDataset
from model   import RoadCNN


WINDOW = "live_infer  |  q = quit  |  any key = advance"

# ── Colours (RGB uint8) ───────────────────────────────────────────────────────
_PRED   = (50,  160, 255)   # predicted peak circles / lines — bright blue
_GUIDE  = (120, 120, 220)   # row guide dashes
_YELLOW = (255, 220, 0)     # GT bars


# ── Image helpers ─────────────────────────────────────────────────────────────
def _denorm(tensor, mean, std):
    """CHW float32 tensor → uint8 HWC numpy (RGB)."""
    m = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    s = np.array(std,  dtype=np.float32).reshape(3, 1, 1)
    img = tensor.cpu().numpy() * s + m
    return (img.clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


def _dashed_hline(canvas, y, colour=_GUIDE, dash=10):
    """Draw a dashed horizontal line at pixel row y (RGB canvas)."""
    h, w = canvas.shape[:2]
    if not (0 <= y < h):
        return
    for x in range(0, w, dash * 2):
        canvas[y, x:x + dash] = colour


# ── Main overlay builder ───────────────────────────────────────────────────────
def _build_frame(image_tensor, outputs, gt_masks, model, threshold=cfg.ROAD_THRESHOLD):
    """
    Returns an RGB uint8 canvas with all overlays applied.

    Parameters
    ----------
    image_tensor : CHW float tensor (single image, already on CPU)
    outputs      : tuple (cls_logits, r0, r1, r2) — single-item batch tensors
    gt_masks     : [N_ROWS, N_BUCKETS] float tensor
    model        : RoadCNN instance (provides geometry params)
    threshold    : minimum sigmoid prob to treat a bucket as active (unused in
                   peak selection — the argmax is always drawn)
    """
    canvas = _denorm(image_tensor, model.norm_mean, model.norm_std).copy()
    h, w   = canvas.shape[:2]
    n      = model.n_buckets

    # Unpack outputs (single-item batches → squeeze batch dim)
    cls_logits, r0_l, r1_l, r2_l = [o.detach().cpu().squeeze(0) for o in outputs]
    road_prob    = cls_logits.softmax(0)[1].item()
    road_present = road_prob > 0.5
    row_logits   = [r0_l, r1_l, r2_l]

    # ── GT yellow translucent bars ────────────────────────────────────────────
    # Find the actual GT peak(s) per row: Gaussian centers have value == 1.0,
    # so threshold at > 0.9 to locate them (handles forked roads too).
    bar_half_h = max(6, h // 14)   # tall enough to be visible, but thin (2 px wide)
    overlay    = canvas.copy()
    for r, frac in enumerate(model.row_fractions):
        row_y    = int(frac * h)
        y0       = max(0, row_y - bar_half_h * 2)
        y1       = row_y   # bottom edge sits on the guide line
        # peaks are buckets where the soft-target value is at its maximum (≈1.0)
        gt_peaks = [b for b in range(n) if gt_masks[r, b].item() > 0.9]
        for b in gt_peaks:
            cx = int((b + 0.5) / n * w)
            # 10-pixel-wide vertical bar centred on the bucket centre
            cv2.rectangle(overlay, (cx - 5, y0), (cx + 5, y1),
                          _YELLOW, cv2.FILLED)
    canvas = cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0)

    # ── Row guide dashed lines ────────────────────────────────────────────────
    for frac in model.row_fractions:
        _dashed_hline(canvas, int(frac * h))

    # ── Prediction peaks, confidence, and connecting lines ───────────────────
    peaks       = []   # (cx, row_y)
    row_confs   = []   # peak sigmoid value per row  [0.0, 1.0]
    for r, (frac, logits) in enumerate(zip(model.row_fractions, row_logits)):
        row_y    = int(frac * h)
        probs    = torch.sigmoid(logits)
        peak_b   = probs.argmax().item()
        peak_conf = probs[peak_b].item()          # confidence at the peak bucket
        cx       = int((peak_b + 0.5) / n * w)
        peaks.append((cx, row_y))
        row_confs.append(peak_conf)

    overall_conf = sum(row_confs) / len(row_confs)   # mean across rows

    # Override: treat as no-road if peak confidence is too low
    if overall_conf < cfg.MIN_PEAK_CONF:
        road_present = False

    if road_present:
        # Lines first (drawn under the circles)
        cv2.line(canvas, peaks[0], peaks[1], _PRED, 4, cv2.LINE_AA)
        cv2.line(canvas, peaks[1], peaks[2], _PRED, 4, cv2.LINE_AA)
        # Filled circles at each peak — centre is exactly on the guide-line row
        for (cx, cy), conf in zip(peaks, row_confs):
            cv2.circle(canvas, (cx, cy), 11, _PRED, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 11, (10, 40, 80), 2, cv2.LINE_AA)
            # Small white centre dot so the exact peak location is visible
            cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
            # Confidence label at the circle — 3× scale, dark shadow for legibility
            conf_txt = f"{conf*100:.0f}%"
            tx = min(cx + 15, w - 80)
            cv2.putText(canvas, conf_txt, (tx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 0, 0),   5, cv2.LINE_AA)
            cv2.putText(canvas, conf_txt, (tx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.35, _PRED,        3, cv2.LINE_AA)


    # ── No-road red perimeter ─────────────────────────────────────────────────
    if not road_present:
        b = 20   # border width in pixels
        canvas[:b,  :] = (220, 30, 30)   # top
        canvas[-b:, :] = (220, 30, 30)   # bottom
        canvas[:,  :b] = (220, 30, 30)   # left
        canvas[:, -b:] = (220, 30, 30)   # right

    # ── Status label bar ──────────────────────────────────────────────────────
    bar_col  = (30, 140, 30) if road_present else (160, 40, 40)
    if road_present:
        label = (f"ROAD {road_prob*100:.0f}%"
                 f"   peak conf: {overall_conf*100:.0f}%"
                 f"  [{row_confs[0]*100:.0f}% / {row_confs[1]*100:.0f}% / {row_confs[2]*100:.0f}%]")
    else:
        label = f"NO ROAD {(1-road_prob)*100:.0f}%"
    label_h  = 84
    label_bar = np.full((label_h, w, 3), bar_col, dtype=np.uint8)
    cv2.putText(label_bar, label, (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2, cv2.LINE_AA)
    canvas = np.concatenate([label_bar, canvas], axis=0)

    return canvas, road_prob, road_present, overall_conf, row_confs


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Road inference viewer")
    p.add_argument("--checkpoint", default=cfg.CHECKPOINT,
                   help=f"Path to model checkpoint  (default: {cfg.CHECKPOINT})")
    p.add_argument("--dataset", default=cfg.DATASET_DIR,
                   help=f"Directory containing labels.csv + images  "
                        f"(default: {cfg.DATASET_DIR})")
    p.add_argument("--delay", type=float, default=cfg.DELAY,
                   help=f"Seconds between frames; 0 = wait for keypress  "
                        f"(default: {cfg.DELAY})")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if cfg.FORCE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}\n"
                 "Edit CHECKPOINT in infer_config.py or pass --checkpoint.")

    model, _, epoch = RoadCNN.from_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"Checkpoint : {args.checkpoint}  (epoch {epoch})")
    print(f"Device     : {device}")

    csv_path = os.path.join(args.dataset, cfg.LABELS_CSV_NAME)
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: labels CSV not found: {csv_path}\n"
                 "Edit DATASET_DIR in infer_config.py or pass --dataset.")

    ds = RoadDataset(csv_path=csv_path, images_dir=args.dataset, augment=False)

    # Filter out augmented/enhanced images by filename suffix
    if cfg.EXCLUDE_SUFFIXES:
        def _is_original(filename):
            stem = os.path.splitext(filename)[0]
            return not any(stem.endswith(s) for s in cfg.EXCLUDE_SUFFIXES)
        orig_indices = [i for i, fn in enumerate(ds.df["filename"])
                        if _is_original(fn)]
        n_skipped = len(ds) - len(orig_indices)
        ds = torch.utils.data.Subset(ds, orig_indices)
        if n_skipped:
            print(f"Filtered   : skipped {n_skipped} enhanced images "
                  f"(suffixes: {cfg.EXCLUDE_SUFFIXES})")

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS)
    total  = len(ds)
    print(f"Dataset    : {args.dataset}  ({total} images)")
    if args.delay == 0:
        print("Mode       : any keypress to advance  (q to quit)\n")
    else:
        print(f"Mode       : auto-advance every {args.delay:.1f}s  (q to quit)\n")

    wait_ms  = max(1, int(args.delay * 1000)) if args.delay > 0 else 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    with torch.no_grad():
        for i, (image, road_present, gt_masks) in enumerate(loader, 1):
            image   = image.to(device)
            outputs = model(image)

            img_t = image.squeeze(0).cpu()
            gt_t  = gt_masks.squeeze(0).cpu()
            outs  = tuple(o.cpu() for o in outputs)

            frame, road_prob, pred_road, overall_conf, row_confs = \
                _build_frame(img_t, outs, gt_t, model)

            rp_gt = road_present.item()
            conf_str = (f"  peak conf: {overall_conf*100:.0f}%"
                        f"  [{row_confs[0]*100:.0f}%/{row_confs[1]*100:.0f}%/{row_confs[2]*100:.0f}%]"
                        if pred_road else "")
            print(f"[{i:>4}/{total}]  gt={'road' if rp_gt else 'no-road':<8}  "
                  f"pred={'road' if pred_road else 'no-road'}  ({road_prob*100:.0f}%){conf_str}")

            cv2.imshow(WINDOW, frame[:, :, ::-1])   # RGB → BGR for cv2

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
