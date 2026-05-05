# ─── road_code/test.py ────────────────────────────────────────────────────────
"""
Evaluate a trained checkpoint against the held-out test set.

Usage
-----
python test.py
python test.py --checkpoint road_best.pth
python test.py --checkpoint road_best.pth --batch-size 16
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import config
from dataset   import RoadDataset
from model     import RoadCNN, road_loss
from visualize import show_training_batch, save_metrics_plot

import matplotlib.pyplot as plt


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate road centerline model on test set")
    p.add_argument("--checkpoint",  default=config.CHECKPOINT,
                   help=f"Checkpoint path (default: {config.CHECKPOINT})")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--n-vis",       type=int, default=8,
                   help="Number of test images to include in the visual panel")
    return p.parse_args()


# ─── Evaluation loop ──────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    total_loss = cls_loss_sum = row_loss_sum = 0.0
    correct_cls = n_samples = 0

    with torch.no_grad():
        for images, road_present, bucket_masks in loader:
            images       = images.to(device)
            road_present = road_present.to(device)
            bucket_masks = bucket_masks.to(device)

            outputs = model(images)
            loss, cls_l, row_l = road_loss(outputs, road_present, bucket_masks)

            bs = images.size(0)
            total_loss    += loss.item() * bs
            cls_loss_sum  += cls_l.item() * bs
            row_loss_sum  += row_l.item() * bs
            correct_cls   += (outputs[0].argmax(1) == road_present).sum().item()
            n_samples     += bs

    n = n_samples
    return total_loss/n, cls_loss_sum/n, row_loss_sum/n, correct_cls/n


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}\n"
                 f"Run train.py first.")

    print(f"Loading checkpoint : {args.checkpoint}")
    model, cfg, epoch = RoadCNN.from_checkpoint(args.checkpoint, device=device)
    model.describe()
    print(f"  Trained to epoch : {epoch}\n")

    # ── Test dataset ──────────────────────────────────────────────────────────
    if not os.path.exists(config.TEST_LABELS_CSV):
        sys.exit(f"ERROR: test labels not found: {config.TEST_LABELS_CSV}\n"
                 f"Place your test data under {config.TEST_DIR}/")

    test_ds = RoadDataset(csv_path=config.TEST_LABELS_CSV,
                          images_dir=config.TEST_DIR,
                          augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=False)

    print(f"{'='*65}")
    print(f"  Device     : {device}")
    print(f"  Test set   : {len(test_ds)} samples  ({len(test_loader)} batches)")
    print(f"{'='*65}\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    t0 = time.time()
    te_loss, te_cls, te_row, te_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    print(f"  Test   →  loss: {te_loss:.4f}  cls: {te_cls:.4f}"
          f"  row: {te_row:.4f}  acc: {te_acc*100:.1f}%")
    print(f"  Time   →  {elapsed:.1f}s\n")

    # ── Visual panel ──────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.checkpoint)
    vis_images, vis_rp, vis_masks = next(iter(test_loader))
    vis_images = vis_images.to(device)
    with torch.no_grad():
        vis_outputs = model(vis_images)

    vis_path = os.path.join(out_dir, "test_predictions.png")
    show_training_batch(
        vis_images.cpu(), tuple(o.cpu() for o in vis_outputs),
        model, gt_masks=vis_masks,
        n_show=args.n_vis,
        title=f"Test predictions  —  loss: {te_loss:.4f}  acc: {te_acc*100:.1f}%",
        save_path=vis_path,
    )
    plt.close("all")
    print(f"  Visual   → {vis_path}")

    # ── Single-point metrics chart ────────────────────────────────────────────
    metrics_path = os.path.join(out_dir, "test_metrics.png")
    save_metrics_plot(
        history={
            "tr_loss": [te_loss], "va_loss": [te_loss],
            "tr_cls":  [te_cls],  "va_cls":  [te_cls],
            "tr_row":  [te_row],  "va_row":  [te_row],
            "tr_acc":  [te_acc],  "va_acc":  [te_acc],
        },
        save_path=metrics_path,
    )
    print(f"  Metrics  → {metrics_path}")

    print(f"\n{'='*65}")
    print(f"  Done.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
