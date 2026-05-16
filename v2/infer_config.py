# ─── road_code/infer_config.py ────────────────────────────────────────────────
# Configuration for live_infer.py.
# Edit this file before running inference — no other files need to change.

import os
import config as _train_cfg

# ── Trained model ─────────────────────────────────────────────────────────────
# Defaults to MODEL_NAME set in config.py — change it there to switch models
# across all scripts at once.  Override here only to use a different model for
# live inference specifically.
CHECKPOINT = _train_cfg.CHECKPOINT

# ── Dataset ───────────────────────────────────────────────────────────────────
# Directory that contains both labels.csv and the image files.
# This can be any train/ or test/ split, or an entirely different dataset.
#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "narrow_trail_through_trees_lr", "test")
#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "diu_gravel_road0_lr", "test")
#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "diu_gravel_road3", "test")
#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "diu_data1_lr", "test")

#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "diu_gravel_road0", "test")

#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "narrow_trail_through_trees_lr", "test")

#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "sand_trail_w_puddles_lr", "test")

#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "thinly_vegetated_terrain_w_small_trees_lr", "test")
#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "thinly_vegetated_terrain_w_small_trees_lr", "test")

DATASET_DIR = os.path.join(os.path.dirname(__file__),
                           "..", "data", "fhl_concatenated_dataset", "test")

#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "trail_w_tree_overhangs", "test")

#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "narrow_trail_thru_trees", "test")

#DATASET_DIR = os.path.join(os.path.dirname(__file__),
#                           "..", "data", "trail_w_vehicle_in_front", "test")

# Name of the labels CSV inside DATASET_DIR.
LABELS_CSV_NAME = "labels.csv"

# Derived — do not edit
LABELS_CSV = os.path.join(DATASET_DIR, LABELS_CSV_NAME)

# ── Display ───────────────────────────────────────────────────────────────────
# Seconds between frames.  Set to 0 to wait for a keypress between each image.
DELAY = 0.001

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
MIN_PEAK_CONF = 0.2

# Minimum sigmoid confidence a *secondary* peak must reach (outside the primary
# plateau) to suppress the road prediction as ambiguous/forked.
# Set high — this is only meant to catch extreme split-road cases.
FORK_SUPPRESS_THRESH = 0.9

# A bucket must reach at least this fraction of the row's peak confidence to be
# counted as part of a cluster.  Raise to ignore faint side lobes; lower to
# catch weaker secondary clusters.  0.45 means 45% of peak.
CLUSTER_THRESH_FRAC = 0.45

# ── Temporal peak disambiguation ──────────────────────────────────────────────
# When multiple clusters exist on a row, the previous frame's primary peak
# position biases cluster selection toward the most spatially consistent choice.
#
# How much to boost a cluster's effective confidence based on proximity to the
# previous frame's peak.  0.0 = disabled.  0.4 = 40% boost at zero distance.
TEMPORAL_BIAS_WEIGHT = 0.4
#
# Distance (as fraction of image width) over which the proximity bonus decays
# to zero.  0.25 = bias applies within the nearest 25% of image width.
TEMPORAL_PROXIMITY_RANGE = 0.25

# Per-frame decay applied to road confidence when the current frame says no-road.
# 0.0 = instant dropout.  0.6 = fades over ~3 frames.  0.85 = very sticky.
ROAD_MOMENTUM_DECAY = 0.6

# Minimum momentum required to keep showing road using the previous frame's peaks.
ROAD_MOMENTUM_THRESH = 0.25

# When a strong peak exists on the farthest row (r0) near a secondary peak
# candidate on the middle row (r1), lower the acceptance threshold to this
# fraction of the primary peak confidence on that row.
# Must be <= FORK_SUPPRESS_THRESH.  0.55 = secondary needs only 55% of primary.
FORK_HINT_THRESH = 0.55

# A far-row peak counts as "nearby" a middle-row candidate if their x positions
# differ by less than this fraction of the image width.
FORK_HINT_PROXIMITY = 0.20

# Maximum lateral swing of a curve beyond its own endpoints' x-range,
# expressed as a fraction of the total vertical span of the branch
# (|p_near.y - p_mid.y| + |p_mid.y - p_far.y|).
# 0.5 = curve may swing at most half the vertical span sideways.
CURVE_SWING_FACTOR = 0.5

# Minimum confidence the far-row (r0) peak must have to draw the Bezier curve
# segment between the middle scan line and the far scan line.
# If r0's peak confidence is below this, only the near→mid segment is drawn.
# 0.0 = always draw  (original behaviour).  0.3 = suppress weak far peaks.
FAR_ROW_CURVE_THRESH = 0.2

# Override the model's scan-line positions (row_fractions).
# Set to a list of 3 ascending fractions, e.g. [0.35, 0.60, 0.80], to use
# custom row heights instead of the values stored in the checkpoint.
# Leave as None to use whatever the checkpoint recorded.
#ROW_FRACTIONS = [0.48,0.58,0.65]
ROW_FRACTIONS = None

# Debug mode: overlay sigmoid response bars on each scan line (translucent green).
# Shows raw model output for all buckets regardless of road_present state.
DEBUG_MODE = True

# Run on CPU even when a GPU is available.  Useful on machines where the GPU
# is being used for training while you want to review results in parallel.
FORCE_CPU = False

# ── Image filtering ───────────────────────────────────────────────────────────
# Skip augmented / enhanced images whose filename stem ends with any of these
# suffixes (matched before the file extension).
# Examples that would be excluded: img_001_c.png, img_001_sl.jpg, img_001_f.png
# Set to an empty list [] to disable filtering and use all images.
EXCLUDE_SUFFIXES = ["_c", "_sl", "_sw", "_f"]
  
