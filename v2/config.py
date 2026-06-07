# ─── road_code/config.py ──────────────────────────────────────────────────────
# All setup parameters for the road centerline model.
# Edit this file before building your dataset or starting training.

# ── Dataset selection ─────────────────────────────────────────────────────────
# Name of the dataset directory inside data/.
#DATASET_NAME = "paintball_diverse_training_data"
DATASET_NAME = "west_virginia_lr"

# ── Model input size (training resolution) ────────────────────────────────────
# Any input image will be automatically resized to these dimensions inside the
# model's forward() pass, so inference works with any source resolution.
IMAGE_WIDTH  = 160
IMAGE_HEIGHT = 120

# ── Lateral bucketing ─────────────────────────────────────────────────────────
# The image width is divided into N_BUCKETS equal columns.
# Each bucket is an independent binary output unit — multiple buckets can fire
# simultaneously on the same row (supports forked / split roads).
#
# N_BUCKETS must evenly divide IMAGE_WIDTH.
# Sensible choices for IMAGE_WIDTH=320: 8, 10, 16, 20, 32, 40, 64, 80
N_BUCKETS = 40

# Per-row lateral margin: fraction of image width excluded on each side.
# Each entry corresponds to [r0, r1, r2] (furthest → nearest row).
# 0.10 means buckets span from 10% to 90% of the image width for that row.
# 0.0  means buckets cover the full image width (no margin).
ROW_BUCKET_MARGINS = [0.1, 0.1, 0.1]

# Ratio of the widest (edge) bucket to the narrowest (center) bucket in pixels.
# 1.0 = uniform / linear spacing.
# 5.0 means edge buckets are 5× wider than center buckets.
# Higher values concentrate more buckets (higher resolution) near the centre.
BUCKET_NONLINEARITY = 5.0

# Derived — populated by init(). Do not edit.
BUCKET_EDGES = None   # list of N_ROWS lists, each N_BUCKETS+1 fractions

# Derived — do not edit
def _check():
    assert N_BUCKETS is not None, \
        "N_BUCKETS is not set. Edit config.py and choose a value that divides IMAGE_WIDTH."
    assert IMAGE_WIDTH % N_BUCKETS == 0, \
        f"N_BUCKETS={N_BUCKETS} does not evenly divide IMAGE_WIDTH={IMAGE_WIDTH}."

BUCKET_WIDTH = None  # set at import time after N_BUCKETS is filled in

# ── Row fractions and crop params (loaded from dataset.info at init time) ─────
# These are populated by init() — do not set manually.
ROW_FRACTIONS = None   # [loc_r0, loc_r1, loc_r2] as fractions of image height
ROW_INDICES   = None   # corresponding pixel rows in IMAGE_HEIGHT space
N_ROWS        = 3
CROP_TOP      = None   # fraction of image height to remove from top
CROP_BOTTOM   = None   # fraction of image height to remove from bottom

# ── Loss weights ──────────────────────────────────────────────────────────────
LAMBDA_CLS  = 1.0
LAMBDA_ROW  = 1.0
# Per-row weights: r0 is furthest (hardest), r2 is nearest (easiest)
ROW_WEIGHTS    = [1.0, 1.0, 1.0]
# Gaussian soft target sigma in bucket units (spread around GT bucket)
GAUSSIAN_SIGMA = 1.0

# ── Training hyperparameters ──────────────────────────────────────────────────
#BATCH_SIZE    = 1
BATCH_SIZE    = 16
EPOCHS        = 200
LR            = 0.002
WEIGHT_DECAY  = 1e-4   # L2 regularisation — helps prevent overfitting
VAL_SPLIT     = 0.15   # fraction of data used for validation
VIS_INTERVAL  = 0      # write a mid-epoch snapshot every N batches (0 = off)
# Stop training if val loss does not improve for this many consecutive epochs.
# Set to 0 to disable early stopping.
EARLY_STOP_PATIENCE = 15

# ── Paths ─────────────────────────────────────────────────────────────────────
import os
DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
DATASET_DIR     = os.path.join(DATA_DIR, DATASET_NAME)
TRAIN_DIR       = os.path.join(DATASET_DIR, "train")
TEST_DIR        = os.path.join(DATASET_DIR, "test")
TRAIN_LABELS_CSV = os.path.join(TRAIN_DIR, "labels.csv")
TEST_LABELS_CSV  = os.path.join(TEST_DIR,  "labels.csv")
TRAIN_DATASET_INFO = os.path.join(TRAIN_DIR, "dataset.info")
TEST_DATASET_INFO  = os.path.join(TEST_DIR,  "dataset.info")
# Name of the model checkpoint (no extension).
# All inference scripts (live_infer.py, infer.py, test.py) use this as their
# default checkpoint.  Change it here and every script picks up the new model.
#MODEL_NAME      = "paintball_road_best"
MODEL_NAME      = "west_virginia_road_best"
CHECKPOINT      = os.path.join(os.path.dirname(__file__), MODEL_NAME + ".pth")

# ── ImageNet normalisation constants ─────────────────────────────────────────
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]


# ── dataset.info parser ───────────────────────────────────────────────────────
def _parse_dataset_info(path):
    """
    Parse a dataset.info file into a dict of {key: float}.
    Lines starting with # are comments. Format: key = value
    """
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            result[key.strip()] = float(val.strip())
    return result


# ── Runtime init ─────────────────────────────────────────────────────────────
def _compute_bucket_edges(N, margin, R):
    """
    Compute N+1 bucket boundary fractions with nonlinear spacing.

    The derivative of the warp function W(t) is a quadratic:
        W'(t) = c0 + c2*(t-0.5)^2
    chosen so that W'(0.5) = c0 (minimum, centre) and W'(0) = c0+c2/4
    (maximum, edge), with their ratio equal to R.

    R = 1  →  uniform spacing (linear).
    R > 1  →  centre buckets are narrower, edge buckets are wider.
    """
    active = 1.0 - 2.0 * margin
    if R <= 1.0:
        return [margin + i / N * active for i in range(N + 1)]
    c0 = 3.0 / (R + 2.0)
    c2 = 12.0 * (R - 1.0) / (R + 2.0)
    edges = []
    for i in range(N + 1):
        t = i / N
        p = c0 * t + c2 * ((t - 0.5) ** 3 + 0.125) / 3.0
        edges.append(margin + p * active)
    return edges


def init():
    """
    Call once at startup.  Validates config, reads dataset.info, and
    populates all derived values (BUCKET_WIDTH, ROW_FRACTIONS, crop params,
    BUCKET_EDGES).
    """
    global BUCKET_WIDTH, ROW_FRACTIONS, ROW_INDICES, CROP_TOP, CROP_BOTTOM
    global BUCKET_EDGES

    _check()
    BUCKET_WIDTH  = IMAGE_WIDTH // N_BUCKETS
    BUCKET_EDGES  = [
        _compute_bucket_edges(N_BUCKETS, m, BUCKET_NONLINEARITY)
        for m in ROW_BUCKET_MARGINS
    ]

    # ── Read geometry from dataset.info ──────────────────────────────────────
    info_path = TRAIN_DATASET_INFO
    if not os.path.exists(info_path):
        info_path = TEST_DATASET_INFO
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"dataset.info not found in {TRAIN_DIR} or {TEST_DIR}.\n"
            "Expected keys: loc_r0, loc_r1, loc_r2, crop_top, crop_bottom")

    info = _parse_dataset_info(info_path)

    missing = [k for k in ("loc_r0", "loc_r1", "loc_r2", "crop_top", "crop_bottom")
               if k not in info]
    if missing:
        raise KeyError(f"dataset.info is missing keys: {missing}")

    ROW_FRACTIONS = [info["loc_r0"], info["loc_r1"], info["loc_r2"]]
    ROW_INDICES   = [int(f * IMAGE_HEIGHT) for f in ROW_FRACTIONS]
    CROP_TOP      = info["crop_top"]
    CROP_BOTTOM   = info["crop_bottom"]

    print(f"Config ready:")
    print(f"  Dataset      : {DATASET_NAME}")
    print(f"  Input size   : {IMAGE_WIDTH} × {IMAGE_HEIGHT}")
    print(f"  N_BUCKETS    : {N_BUCKETS}  (avg bucket width = {BUCKET_WIDTH} px)")
    print(f"  Bucket nonlinearity: {BUCKET_NONLINEARITY}")
    for ri, (m, edges) in enumerate(zip(ROW_BUCKET_MARGINS, BUCKET_EDGES)):
        _ctr_w  = (edges[N_BUCKETS//2+1] - edges[N_BUCKETS//2]) * IMAGE_WIDTH
        _edge_w = (edges[1] - edges[0]) * IMAGE_WIDTH
        print(f"  r{ri} margin={m:.0%}  centre≈{_ctr_w:.1f}px  edge≈{_edge_w:.1f}px")
    print(f"  Row fractions: {ROW_FRACTIONS}  →  rows {ROW_INDICES}")
    print(f"  Crop top     : {CROP_TOP:.2%}  ({int(CROP_TOP * IMAGE_HEIGHT)} px)")
    print(f"  Crop bottom  : {CROP_BOTTOM:.2%}  ({int(CROP_BOTTOM * IMAGE_HEIGHT)} px)")
    print(f"  Output nodes : 2 (classifier) + {N_ROWS} × {N_BUCKETS} (buckets)"
          f" = {2 + N_ROWS * N_BUCKETS}")
