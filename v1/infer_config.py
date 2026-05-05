# ─── road_code/infer_config.py ────────────────────────────────────────────────
# Configuration for live_infer.py.
# Edit this file before running inference — no other files need to change.

import os

# ── Trained model ─────────────────────────────────────────────────────────────
# Path to the checkpoint produced by train.py.
# All model geometry (image size, n_buckets, row fractions, crop, normalisation)
# is restored automatically from the checkpoint — no need to set them here.
CHECKPOINT = os.path.join(os.path.dirname(__file__), "road_best.pth")

# ── Dataset ───────────────────────────────────────────────────────────────────
# Directory that contains both labels.csv and the image files.
# This can be any train/ or test/ split, or an entirely different dataset.
DATASET_DIR = os.path.join(os.path.dirname(__file__),
                           "..", "data", "diu_gravel_road0", "test")
#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "diu_gravel_road3", "test")
#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "diu_data1", "test")

# Name of the labels CSV inside DATASET_DIR.
LABELS_CSV_NAME = "labels.csv"

# Derived — do not edit
LABELS_CSV = os.path.join(DATASET_DIR, LABELS_CSV_NAME)

# ── Display ───────────────────────────────────────────────────────────────────
# Seconds between frames.  Set to 0 to wait for a keypress between each image.
DELAY = 0.1

# DataLoader worker processes for image prefetching.
# 0 = load in the main thread (safe but slow).
# 2-4 = decode next images in background while current frame is displayed.
NUM_WORKERS = 4

# ── Inference ─────────────────────────────────────────────────────────────────
# Minimum road-present classifier probability to treat a frame as "road".
# Predictions (circles + connecting lines) are only drawn when this threshold
# is exceeded.
ROAD_THRESHOLD = 0.5

# Minimum mean peak confidence across all three rows to accept a road prediction.
# If the combined confidence falls below this, the frame is treated as No Road
# regardless of what the classifier says.
MIN_PEAK_CONF = 0.45

# Run on CPU even when a GPU is available.  Useful on machines where the GPU
# is being used for training while you want to review results in parallel.
FORCE_CPU = False

# ── Image filtering ───────────────────────────────────────────────────────────
# Skip augmented / enhanced images whose filename stem ends with any of these
# suffixes (matched before the file extension).
# Examples that would be excluded: img_001_c.png, img_001_sl.jpg, img_001_f.png
# Set to an empty list [] to disable filtering and use all images.
EXCLUDE_SUFFIXES = ["_c", "_sl", "_sw", "_f"]
