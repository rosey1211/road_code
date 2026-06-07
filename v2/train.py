# ─── road_code/train.py ───────────────────────────────────────────────────────
import sys
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import os
import config
import sys
sys.path.insert(0, os.path.dirname(__file__))
from dataset   import RoadDataset, show_dataset_samples
from model     import RoadCNN, road_loss
from visualize import show_training_batch, save_metrics_plot

config.init()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Progress bar ─────────────────────────────────────────────────────────────
def progress_bar(current, total, bar_len=28, **stats):
    filled   = int(bar_len * current / total)
    bar      = "█" * filled + "░" * (bar_len - filled)
    stat_str = "  ".join(f"{k}: {v}" for k, v in stats.items())
    sys.stdout.write(f"\r  [{bar}] {current}/{total}  {stat_str}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")

# ─── Data ─────────────────────────────────────────────────────────────────────
full_ds   = RoadDataset(augment=False)
val_size  = max(1, int(len(full_ds) * config.VAL_SPLIT))
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

# Augmentation only on train split
train_ds.dataset.augment = False   # dataset-level flag not used after split —
                                   # augment via a separate dataset instance:
train_aug_ds = RoadDataset(augment=True)
# Use indices from the split
train_aug_ds = torch.utils.data.Subset(
    RoadDataset(augment=True), train_ds.indices)

_PIN = torch.cuda.is_available()
train_loader = DataLoader(train_aug_ds, batch_size=config.BATCH_SIZE,
                          shuffle=True,  num_workers=8, pin_memory=_PIN)
val_loader   = DataLoader(val_ds,       batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=8, pin_memory=_PIN)

# ─── Model, optimiser, scheduler ──────────────────────────────────────────────
model = RoadCNN(
    image_width   = config.IMAGE_WIDTH,
    image_height  = config.IMAGE_HEIGHT,
    n_buckets     = config.N_BUCKETS,
    row_fractions = config.ROW_FRACTIONS,
    norm_mean     = config.NORM_MEAN,
    norm_std      = config.NORM_STD,
    crop_top      = config.CROP_TOP,
    crop_bottom   = config.CROP_BOTTOM,
    row_bucket_margins   = getattr(config, "ROW_BUCKET_MARGINS",  [0.0, 0.0, 0.0]),
    bucket_nonlinearity  = getattr(config, "BUCKET_NONLINEARITY", 1.0),
).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

n_params = sum(p.numel() for p in model.parameters())
print(f"\n{'='*65}")
print(f"  Device     : {DEVICE}")
print(f"  Parameters : {n_params:,}")
print(f"  Train / Val: {train_size} / {val_size} samples")
print(f"  Epochs     : {config.EPOCHS}  |  Batch size: {config.BATCH_SIZE}")
print(f"{'='*65}\n")

# ─── Pre-training data sample panel ───────────────────────────────────────────
import matplotlib.pyplot as plt
_sample_path = os.path.join(os.path.dirname(config.CHECKPOINT), "pretrain_samples.png")
show_dataset_samples(n=5, save_path=_sample_path)
plt.close("all")

# ─── Mid-epoch visual snapshot ────────────────────────────────────────────────
def _random_val_batch(n=8):
    """Sample n random images from the full val dataset."""
    indices = torch.randperm(len(val_loader.dataset))[:n].tolist()
    samples = [val_loader.dataset[i] for i in indices]
    images  = torch.stack([s[0] for s in samples])
    masks   = torch.stack([s[2] for s in samples])
    return images, masks


def write_vis_snapshot(epoch, batch_idx, label=""):
    """Write a prediction overlay PNG without interrupting the training loop."""
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        vis_images, vis_masks = _random_val_batch(n=8)
        vis_images = vis_images.to(DEVICE)          # ← must be on same device as model
        vis_outputs = model(vis_images)
    save_path = os.path.join(os.path.dirname(config.CHECKPOINT),
                             f"vis_epoch_{epoch:03d}_b{batch_idx:04d}.png")
    show_training_batch(
        vis_images.cpu(), tuple(o.cpu() for o in vis_outputs),
        model, gt_masks=vis_masks,
        n_show=8,
        title=f"Epoch {epoch}/{config.EPOCHS} — batch {batch_idx}{label}",
        save_path=save_path,
    )
    plt.close("all")
    torch.cuda.empty_cache()
    model.train()
    return save_path


# ─── Train / eval loops ───────────────────────────────────────────────────────
def run_epoch(loader, train=True, epoch=0):
    model.train() if train else model.eval()
    total_loss = cls_loss_sum = row_loss_sum = 0.0
    correct_cls = n_samples = 0
    n_batches = len(loader)
    loop_start = time.time()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (images, road_present, bucket_masks) in enumerate(loader, 1):
            images       = images.to(DEVICE)
            road_present = road_present.to(DEVICE)
            bucket_masks = bucket_masks.to(DEVICE)

            outputs = model(images)

            loss, cls_l, row_l = road_loss(outputs, road_present, bucket_masks,
                                           config.LAMBDA_CLS, config.LAMBDA_ROW,
                                           config.ROW_WEIGHTS)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            total_loss    += loss.item() * bs
            cls_loss_sum  += cls_l.item() * bs
            row_loss_sum  += row_l.item() * bs
            correct_cls   += (outputs[0].argmax(1) == road_present).sum().item()
            n_samples     += bs

            elapsed   = time.time() - loop_start
            sps       = n_samples / elapsed if elapsed > 0 else 0.0
            eta_s     = (n_batches - i) * (elapsed / i)
            progress_bar(i, n_batches,
                         loss=f"{total_loss/n_samples:.4f}",
                         cls=f"{cls_loss_sum/n_samples:.4f}",
                         row=f"{row_loss_sum/n_samples:.4f}",
                         acc=f"{correct_cls/n_samples*100:.1f}%",
                         img_s=f"{sps:.0f}",
                         eta=f"{eta_s:.0f}s")

            if train and config.VIS_INTERVAL > 0 and i % config.VIS_INTERVAL == 0:
                snap = write_vis_snapshot(epoch, i)
                sys.stdout.write(f"\n  vis → {snap}\n")
                sys.stdout.flush()

    n = n_samples
    return total_loss/n, cls_loss_sum/n, row_loss_sum/n, correct_cls/n

# ─── Training loop ────────────────────────────────────────────────────────────
best_val_loss    = float("inf")
best_epoch       = None
no_improve_count = 0
total_start      = time.time()
metrics_path  = os.path.join(os.path.dirname(config.CHECKPOINT), "metrics.png")
history = {k: [] for k in
           ["tr_loss", "va_loss", "tr_cls", "va_cls",
            "tr_row",  "va_row",  "tr_acc", "va_acc"]}

for epoch in range(1, config.EPOCHS + 1):
    epoch_start = time.time()
    current_lr  = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch:02d}/{config.EPOCHS}  (lr={current_lr:.6f})")
    print(f"  Training:")
    tr_loss, tr_cls, tr_row, tr_acc = run_epoch(train_loader, train=True,  epoch=epoch)

    print(f"  Validating:")
    va_loss, va_cls, va_row, va_acc = run_epoch(val_loader,   train=False, epoch=epoch)
    scheduler.step()

    epoch_time = time.time() - epoch_start
    is_best    = va_loss < best_val_loss
    if is_best:
        best_val_loss    = va_loss
        best_epoch       = epoch
        no_improve_count = 0
        torch.save({
            # Weights
            "model_state":   model.state_dict(),
            "epoch":         epoch,
            # ── All params needed to reconstruct the model at inference ──
            "image_width":   model.image_width,
            "image_height":  model.image_height,
            "n_buckets":     model.n_buckets,
            "row_fractions": model.row_fractions,
            "row_indices":   model.row_indices,
            "bucket_width":  model.bucket_width,
            "norm_mean":     model.norm_mean,
            "norm_std":      model.norm_std,
            "crop_top":             model.crop_top,
            "crop_bottom":          model.crop_bottom,
            "row_bucket_margins":   model.row_bucket_margins,
            "bucket_nonlinearity":  model.bucket_nonlinearity,
        }, config.CHECKPOINT)

    # ── Trend indicator (val loss vs previous epoch) ──────────────────────────
    if len(history["va_loss"]) == 0:
        trend = ""
    elif va_loss < history["va_loss"][-1] - 1e-5:
        trend = "  ↓ improving"
    elif va_loss > history["va_loss"][-1] + 1e-5:
        trend = "  ↑ overfitting?" if (va_loss > tr_loss * 1.1) else "  ↑ rising"
    else:
        trend = "  → plateau"

    if not is_best:
        no_improve_count += 1

    marker = "  ★ best" if is_best else ""
    early_stop_note = ""
    if config.EARLY_STOP_PATIENCE > 0 and not is_best:
        early_stop_note = f"  (no improvement {no_improve_count}/{config.EARLY_STOP_PATIENCE})"
    print(f"  Train  →  loss: {tr_loss:.4f}  cls: {tr_cls:.4f}  row: {tr_row:.4f}  acc: {tr_acc*100:.1f}%")
    print(f"  Val    →  loss: {va_loss:.4f}  cls: {va_cls:.4f}  row: {va_row:.4f}  acc: {va_acc*100:.1f}%{marker}{trend}{early_stop_note}")
    print(f"  Time   →  {epoch_time:.1f}s  |  elapsed: {time.time()-total_start:.0f}s")

    # ── Append to history and save metrics plot ───────────────────────────────
    for k, v in [("tr_loss", tr_loss), ("va_loss", va_loss),
                 ("tr_cls",  tr_cls),  ("va_cls",  va_cls),
                 ("tr_row",  tr_row),  ("va_row",  va_row),
                 ("tr_acc",  tr_acc),  ("va_acc",  va_acc)]:
        history[k].append(v)
    save_metrics_plot(history, metrics_path, best_epoch=best_epoch)
    print(f"  Metrics → {metrics_path}")

    # ── Visual check: overlay predictions on random val images ───────────────
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        vis_images, vis_masks = _random_val_batch(n=8)
        vis_images = vis_images.to(DEVICE)          # ← must be on same device as model
        vis_outputs = model(vis_images)
    save_path = os.path.join(os.path.dirname(config.CHECKPOINT),
                             f"vis_epoch_{epoch:03d}.png")
    show_training_batch(
        vis_images.cpu(), tuple(o.cpu() for o in vis_outputs),
        model, gt_masks=vis_masks,
        n_show=8, title=f"Epoch {epoch}/{config.EPOCHS} — val predictions",
        save_path=save_path,
    )
    plt.close("all")
    torch.cuda.empty_cache()
    print()

    # ── Early stopping ────────────────────────────────────────────────────────
    if config.EARLY_STOP_PATIENCE > 0 and no_improve_count >= config.EARLY_STOP_PATIENCE:
        print(f"  Early stopping: val loss has not improved for "
              f"{config.EARLY_STOP_PATIENCE} consecutive epochs.")
        print(f"  Best checkpoint is epoch {best_epoch}  "
              f"(val loss {best_val_loss:.4f})  →  {config.CHECKPOINT}")
        print()
        break

total_time = time.time() - total_start
print(f"{'='*65}")
print(f"  Training complete in {total_time/60:.1f} min")
print(f"  Best val loss : {best_val_loss:.4f}  (saved {config.CHECKPOINT})")
print(f"{'='*65}")
