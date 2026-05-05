# ─── road_code/dataset.py ─────────────────────────────────────────────────────
import math
import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import config


class RoadDataset(Dataset):
    """
    Loads road centerline labels from labels.csv and images from data/images/.

    labels.csv schema
    -----------------
    filename     : image filename (relative to data/images/)
    road_present : 1 = road visible, 0 = no road
    loc_r0       : semicolon-separated scaled locations [0.0, 1.0] at row 0
    loc_r1       : semicolon-separated scaled locations [0.0, 1.0] at row 1
    loc_r2       : semicolon-separated scaled locations [0.0, 1.0] at row 2

    Row ordering (image y-axis, 0 = top):
        r0  →  ROW_FRACTIONS[0] * IMAGE_HEIGHT  (e.g. 0.75 * 240 = row 180)  furthest ahead
        r1  →  ROW_FRACTIONS[1] * IMAGE_HEIGHT  (e.g. 0.85 * 240 = row 204)
        r2  →  ROW_FRACTIONS[2] * IMAGE_HEIGHT  (e.g. 0.95 * 240 = row 228)  nearest to camera

    Scaled locations represent the horizontal position as a proportion of the
    image width (0.0 = left edge, 1.0 = right edge).  At load time each value
    is converted to a bucket index:
        bucket_idx = min(int(scaled_loc * N_BUCKETS), N_BUCKETS - 1)

    Storing scaled locations (not bucket indices) makes labels independent of
    N_BUCKETS — you can change the bucket resolution without re-annotating.

    Example rows
    ------------
    img_0001.png, 1, 0.516,         0.516,         0.531
    img_0002.png, 1, 0.406;0.609,   0.422;0.625,   0.422;0.625   <- forked road
    img_0003.png, 0, ,              ,                              <- no road

    Returns (per __getitem__)
    -------------------------
    image        : FloatTensor [3, H, W]  — normalised, NOT resized (model does that)
    road_present : LongTensor  scalar     — 0 or 1
    bucket_masks : FloatTensor [N_ROWS, N_BUCKETS]  — binary 0/1; all-zero when no road
    """

    def __init__(self, csv_path=None, images_dir=None, augment=False):
        config._check()

        csv_path   = csv_path   or config.TRAIN_LABELS_CSV
        images_dir = images_dir or config.TRAIN_DIR

        self.images_dir = images_dir
        df = pd.read_csv(csv_path, dtype=str, comment='#').fillna("")

        # Keep only rows whose image file is actually present on disk
        on_disk = set(os.listdir(images_dir))
        before  = len(df)
        df      = df[df["filename"].isin(on_disk)].reset_index(drop=True)
        skipped = before - len(df)
        if skipped:
            print(f"  [dataset] skipped {skipped} rows from labels.csv "
                  f"— image not found in {images_dir}")
        print(f"  [dataset] loaded {len(df)} samples from {images_dir}")
        self.df = df

        # Basic pixel-value transform — no resize (model handles that)
        t = [transforms.ToTensor(),
             transforms.Normalize(config.NORM_MEAN, config.NORM_STD)]
        if augment:
            t = [transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                        saturation=0.2, hue=0.05),
                 transforms.RandomHorizontalFlip()] + t
        self.transform = transforms.Compose(t)

        self._loc_cols = ["loc_r0", "loc_r1", "loc_r2"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Image ────────────────────────────────────────────────────────────
        img_path = os.path.join(self.images_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)   # [3, H, W] — original resolution

        # ── Road-present label ───────────────────────────────────────────────
        road_present = torch.tensor(int(row["road_present"]), dtype=torch.long)

        # ── Bucket masks  [N_ROWS, N_BUCKETS] ────────────────────────────────
        bucket_masks = torch.zeros(config.N_ROWS, config.N_BUCKETS, dtype=torch.float32)

        if road_present.item() == 1:
            sigma = config.GAUSSIAN_SIGMA
            for r, col in enumerate(self._loc_cols):
                field = str(row[col]).strip()
                if field:
                    for val in field.split(";"):
                        scaled_loc = float(val.strip())
                        assert 0.0 <= scaled_loc <= 1.0, \
                            f"Scaled location {scaled_loc} out of range [0, 1] " \
                            f"in sample {idx}, column {col}"
                        b = fraction_to_bucket(scaled_loc, r)
                        # Gaussian soft target centred on b — take max to
                        # handle forked roads (two peaks don't interfere)
                        for i in range(config.N_BUCKETS):
                            v = math.exp(-0.5 * ((i - b) / sigma) ** 2)
                            if v > bucket_masks[r, i].item():
                                bucket_masks[r, i] = v

        return image, road_present, bucket_masks


# ─── Conversion helpers ───────────────────────────────────────────────────────

def fraction_to_bucket(frac: float, row: int = 0) -> int:
    """
    Convert a [0.0, 1.0] full-image width fraction to a bucket index.

    Uses the per-row BUCKET_EDGES table which encodes the lateral margin
    and nonlinear spacing for that row.  Fractions outside the active region
    are clamped to the nearest edge bucket.
    """
    config._check()
    edges = config.BUCKET_EDGES[row]
    N     = config.N_BUCKETS
    # Binary search: find the bucket whose interval contains frac
    lo, hi = 0, N - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if frac < edges[mid + 1]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def col_to_fraction(col_px: int) -> float:
    """Convert a raw pixel column to a [0.0, 1.0] width fraction."""
    return col_px / config.IMAGE_WIDTH


def col_to_bucket(col_px: int) -> int:
    """Convert a raw pixel column to a bucket index (via fraction)."""
    return fraction_to_bucket(col_to_fraction(col_px))


def bucket_center_fraction(bucket_idx: int, row: int = 0) -> float:
    """
    Return the full-image fractional centre of bucket bucket_idx for the given row.
    Uses the per-row BUCKET_EDGES table so the position correctly
    reflects the nonlinear spacing and per-row margin.
    """
    config._check()
    edges = config.BUCKET_EDGES[row]
    return (edges[bucket_idx] + edges[bucket_idx + 1]) / 2.0


def bucket_center_px(bucket_idx: int) -> float:
    """Return the pixel column at the centre of a bucket."""
    return bucket_center_fraction(bucket_idx) * config.IMAGE_WIDTH


# ─── Dataset sample visualizer ───────────────────────────────────────────────
def show_dataset_samples(n: int = 5, save_path: str = None):
    """
    Display the first n dataset images in a panel.

    Each image has a dashed guide line at every ROW_FRACTION position.
    A bar graph is overlaid at each row: buckets with a label value of 1
    show a solid coloured bar; inactive buckets show no bar.

    Parameters
    ----------
    n         : number of samples to show (capped at dataset length)
    save_path : if given, save the figure to this path

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    config._check()
    ds = RoadDataset(augment=False)
    n  = min(n, len(ds))

    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    axes = np.array(axes).reshape(-1)

    mean = np.array(config.NORM_MEAN)
    std  = np.array(config.NORM_STD)

    # One colour per row (r0 = near / bottom, r1 = mid, r2 = far / top)
    row_colours = ["#00FF88", "#FFCC00", "#FF6688"]

    for i in range(n):
        image, road_present, bucket_masks = ds[i]
        ax = axes[i]

        # Denormalize CHW float tensor → HWC float [0, 1]
        img = image.permute(1, 2, 0).numpy()
        img = (img * std + mean).clip(0.0, 1.0)
        h, w = img.shape[:2]

        ax.imshow(img)
        ax.set_title(f"{'Road' if road_present.item() else 'No road'}",
                     fontsize=9,
                     color="green" if road_present.item() else "red")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)   # match imshow convention (y=0 at top)
        ax.axis("off")

        bucket_w  = w / config.N_BUCKETS
        bar_max_h = h * 0.07   # max bar height = 7% of image height

        for r, frac in enumerate(config.ROW_FRACTIONS):
            row_y = int(frac * h)
            colour = row_colours[r % len(row_colours)]

            # Dashed guide line across the full width
            ax.axhline(row_y, color=colour, linewidth=0.8,
                       linestyle="--", alpha=0.55)

            # Bar graph: one bar per bucket, height = label value (0 or 1)
            for b in range(config.N_BUCKETS):
                val = bucket_masks[r, b].item()
                if val > 0.5:
                    rect = patches.Rectangle(
                        (b * bucket_w, row_y - bar_max_h),
                        bucket_w, bar_max_h,
                        linewidth=0, facecolor=colour, alpha=0.85,
                    )
                    ax.add_patch(rect)

        # Crop boundary lines (violet)
        if config.CROP_TOP and config.CROP_TOP > 0:
            ax.axhline(int(config.CROP_TOP * h), color="#B450DC",
                       linewidth=1.2, linestyle="-", alpha=0.85)
        if config.CROP_BOTTOM and config.CROP_BOTTOM > 0:
            ax.axhline(int((1.0 - config.CROP_BOTTOM) * h), color="#B450DC",
                       linewidth=1.2, linestyle="-", alpha=0.85)

    # Legend
    legend_handles = [
        patches.Patch(color=row_colours[r],
                      label=f"r{r}  y≈{config.ROW_FRACTIONS[r]*100:.0f}%")
        for r in range(config.N_ROWS)
    ]
    legend_handles.append(patches.Patch(color="#B450DC", label="crop boundary"))
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=config.N_ROWS + 1, fontsize=8, framealpha=0.7)
    fig.suptitle("Dataset samples — bar graph = active training buckets",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ─── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    config.init()

    if not os.path.exists(config.TRAIN_LABELS_CSV):
        print(f"labels.csv not found at {config.TRAIN_LABELS_CSV}")
        print("Create data/images/ and data/labels.csv first.")
        sys.exit(0)

    ds = RoadDataset(augment=False)
    print(f"\nDataset size: {len(ds)} samples")

    img, rp, masks = ds[0]
    print(f"image shape       : {list(img.shape)}")
    print(f"road_present      : {rp.item()}")
    print(f"bucket_masks shape: {list(masks.shape)}")
    print(f"active buckets    : {masks.nonzero(as_tuple=False).tolist()}")

    # Round-trip test
    test_col  = 165
    frac      = col_to_fraction(test_col)
    b         = fraction_to_bucket(frac)
    center_px = bucket_center_px(b)
    print(f"\nRound-trip: col {test_col} → fraction {frac:.4f} → bucket {b} → center {center_px:.1f} px")

    # Fork test (two fractions on same row)
    fracs = [0.406, 0.609]
    buckets = [fraction_to_bucket(f) for f in fracs]
    print(f"Fork test: fractions {fracs} → buckets {buckets}")

    # ── Visual sample panel ───────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = show_dataset_samples(n=5, save_path="dataset_samples.png")
    plt.show()
