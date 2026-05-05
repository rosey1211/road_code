# ─── road_code/live_test.py ───────────────────────────────────────────────────
"""
Real-time per-image test viewer (OpenCV display, no matplotlib).

For each image in the test set, shows a side-by-side window:
  left  — ground truth bucket annotations
  right — model prediction overlay

Usage
-----
python live_test.py
python live_test.py --checkpoint road_best.pth
python live_test.py --delay 0      # wait for any keypress between images
python live_test.py --delay 2.0    # 2-second auto-advance (default: 1.0 s)
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import config
from dataset import RoadDataset
from model   import RoadCNN


# ─── Colours (RGB uint8) ─────────────────────────────────────────────────────
_GT_COLOUR   = (255, 220,   0)   # yellow — ground-truth points
_PRED_COLOUR = ( 50, 220,  50)   # green  — predicted points
_ROW_COLOUR  = (120, 120, 220)   # blue   — row guide lines

WINDOW = "Live Test  |  q = quit"


# ─── Drawing helpers ─────────────────────────────────────────────────────────
def _denorm(tensor, mean, std):
    """CHW float32 tensor → uint8 HWC numpy (RGB)."""
    m = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    s = np.array(std,  dtype=np.float32).reshape(3, 1, 1)
    img = tensor.cpu().numpy() * s + m
    return (img.clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


def _circle(canvas, cx, cy, radius=7, colour=_GT_COLOUR):
    h, w = canvas.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                y, x = cy + dy, cx + dx
                if 0 <= y < h and 0 <= x < w:
                    canvas[y, x] = colour


def _hline_dashed(canvas, y, colour=_ROW_COLOUR, dash=10):
    h, w = canvas.shape[:2]
    if 0 <= y < h:
        for x in range(w):
            if (x // dash) % 2 == 0:
                canvas[y, x] = colour


def _gt_canvas(image_tensor, gt_masks, model):
    """Denormalised image with GT row guides and bucket circles."""
    canvas = _denorm(image_tensor, model.norm_mean, model.norm_std).copy()
    h, w   = canvas.shape[:2]
    for r, frac in enumerate(model.row_fractions):
        row_y = int(frac * h)
        _hline_dashed(canvas, row_y)
        for b in gt_masks[r].nonzero(as_tuple=True)[0].tolist():
            cx = int((b + 0.5) / model.n_buckets * w)
            _circle(canvas, cx, row_y, colour=_GT_COLOUR)
    return canvas


def _pred_canvas(image_tensor, outputs, model, threshold=0.5):
    """Denormalised image with predicted row guides and bucket circles."""
    canvas = _denorm(image_tensor, model.norm_mean, model.norm_std).copy()
    h, w   = canvas.shape[:2]

    cls_logits, r0_logits, r1_logits, r2_logits = [
        o.detach().cpu().squeeze(0) for o in outputs
    ]
    road_prob    = cls_logits.softmax(0)[1].item()
    road_present = road_prob > 0.5
    label        = f"ROAD {road_prob*100:.0f}%" if road_present else f"NO ROAD {(1-road_prob)*100:.0f}%"

    for r, (frac, logits) in enumerate(zip(model.row_fractions,
                                           [r0_logits, r1_logits, r2_logits])):
        row_y = int(frac * h)
        _hline_dashed(canvas, row_y)
        if road_present:
            probs = torch.sigmoid(logits)
            for b in (probs >= threshold).nonzero(as_tuple=True)[0].tolist():
                cx = int((b + 0.5) / model.n_buckets * w)
                _circle(canvas, cx, row_y, colour=_PRED_COLOUR)

    return canvas, label, road_present


def _label_bar(width, text, is_road):
    """Coloured header bar with white label text (RGB)."""
    colour = (30, 160, 30) if is_road else (180, 40, 40)
    bar    = np.full((28, width, 3), colour, dtype=np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def _add_label(canvas_rgb, text, is_road):
    return np.concatenate([_label_bar(canvas_rgb.shape[1], text, is_road),
                           canvas_rgb], axis=0)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Live per-image test viewer")
    p.add_argument("--checkpoint", default=config.CHECKPOINT,
                   help=f"Checkpoint path (default: {config.CHECKPOINT})")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds between images; 0 = wait for keypress (default: 1.0)")
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}\n"
                 f"Run train.py first.")

    model, _, epoch = RoadCNN.from_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"Loaded checkpoint  (epoch {epoch})  |  device: {device}")

    if not os.path.exists(config.TEST_LABELS_CSV):
        sys.exit(f"ERROR: test labels not found: {config.TEST_LABELS_CSV}\n"
                 f"Place test data under {config.TEST_DIR}/")

    test_ds = RoadDataset(csv_path=config.TEST_LABELS_CSV,
                          images_dir=config.TEST_DIR,
                          augment=False)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    total   = len(test_ds)
    print(f"Test set : {total} images")
    if args.delay == 0:
        print("Mode     : any keypress to advance  (q to quit)\n")
    else:
        print(f"Mode     : auto-advance every {args.delay:.1f}s  (q to quit)\n")

    wait_ms = max(1, int(args.delay * 1000)) if args.delay > 0 else 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    with torch.no_grad():
        for i, (image, road_present, gt_masks) in enumerate(loader, 1):
            image   = image.to(device)
            outputs = model(image)

            img_t  = image.squeeze(0).cpu()
            gt_t   = gt_masks.squeeze(0).cpu()
            outs_1 = tuple(o.cpu() for o in outputs)

            gt_img                          = _gt_canvas(img_t, gt_t, model)
            pred_img, pred_label, pred_road = _pred_canvas(img_t, outs_1, model)

            rp    = road_present.item()
            gt_gt = f"GT: {'ROAD' if rp else 'NO ROAD'}  [{i}/{total}]"

            gt_panel   = _add_label(gt_img,   gt_gt,              rp == 1)
            pred_panel = _add_label(pred_img, f"PRED: {pred_label}", pred_road)

            combined = np.concatenate([gt_panel, pred_panel], axis=1)
            cv2.imshow(WINDOW, combined[:, :, ::-1])   # RGB → BGR

            print(f"[{i:>4}/{total}]  gt={'road' if rp else 'no-road':<7}  pred={pred_label}")

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
