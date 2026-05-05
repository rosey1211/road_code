# Road Centerline CNN

Estimates the centerline of a road in a scene using a deep CNN.
The centerline is sampled at 3 predetermined row positions and encoded
as lateral bucket positions — supports forked/split roads.

---

## Project Structure

```
road_code/
├── config.py       — all hyperparameters and paths
├── dataset.py      — dataset loader (reads labels.csv)
├── model.py        — RoadCNN definition and loss function
├── train.py        — training loop
├── infer.py        — single image or live camera inference
├── visualize.py    — overlay helpers used during training and inference
└── data/
    ├── images/     — place all training images here
    └── labels.csv  — annotation file
```

---

## Setup

### Step 1 — Set N_BUCKETS in config.py

Open `config.py` and set `N_BUCKETS` to a value that **evenly divides 320**:

```python
N_BUCKETS = 20   # 320 / 20 = 16 px per bucket
```

Sensible choices: `8, 10, 16, 20, 32, 40, 64, 80`

More buckets = finer lateral resolution but more output nodes to train.

### Step 2 — Add images

Place all road images (PNG or JPEG, any resolution) into:

```
road_code/data/images/
```

### Step 3 — Create labels.csv

The file `road_code/data/labels.csv` must have this header:

```
filename,road_present,loc_r0,loc_r1,loc_r2
```

| Column | Description |
|---|---|
| `filename` | Image filename (e.g. `img_001.png`) |
| `road_present` | `1` = road visible, `0` = no road |
| `loc_r0` | Fractional position(s) at row 0 — furthest ahead (75% down the image) |
| `loc_r1` | Fractional position(s) at row 1 — middle (85% down) |
| `loc_r2` | Fractional position(s) at row 2 — nearest to camera (95% down) |

**Fractional positions** are `0.0` (left edge) to `1.0` (right edge).
Use semicolons for forks. Leave `loc_*` columns empty when no road.
Lines starting with `#` are treated as comments and ignored.

**Example rows:**

```csv
filename,road_present,loc_r0,loc_r1,loc_r2
# Straight road
img_001.png,1,0.516,0.516,0.531
# Forked road
img_002.png,1,0.406;0.609,0.422;0.625,0.422;0.625
# No road visible
img_003.png,0,,,
```

**Row positions at default config (240 px height):**

| Label | Row fraction | Pixel row | Description |
|---|---|---|---|
| `r0` | 0.75 | 180 | Furthest ahead |
| `r1` | 0.85 | 204 | Middle |
| `r2` | 0.95 | 228 | Nearest to camera |

---

## Verify Dataset Loads

```bash
cd road_code
python3 dataset.py
```

Expected output:
```
Config ready:
  Input size   : 320 × 240
  N_BUCKETS    : 20  (bucket width = 16 px)
  ...
Dataset size: N samples
image shape       : [3, H, W]
road_present      : 1
bucket_masks shape: [3, 20]
active buckets    : [[0, 10], [1, 10], [2, 11]]
```

---

## Train

In `train.py`, set `N_BUCKETS` to match `config.py` by uncommenting this line near the top:

```python
config.N_BUCKETS = 20   # must match config.py
```

Then run:

```bash
cd road_code
python3 train.py
```

**What happens during training:**
- Per-batch progress bar showing loss, cls loss, row loss, and accuracy
- Per-epoch train/val summary; `★ best` marks a new best checkpoint
- `vis_epoch_NNN.png` saved after each epoch — val images with prediction overlays
  - Green dots = predicted centerline buckets
  - Yellow dots = ground truth
  - Blue dashed lines = row guides
- Best checkpoint saved to `road_best.pth`

**Tuning the loss weights (in config.py):**

| Issue | Fix |
|---|---|
| Model doesn't detect roads | Increase `LAMBDA_CLS` |
| Detection OK but wrong column position | Increase `LAMBDA_ROW` |
| Overfitting | Reduce `EPOCHS` or increase `VAL_SPLIT` |
| Too coarse lateral resolution | Increase `N_BUCKETS` (no re-annotation needed) |

---

## Inference

```bash
# Single image
python3 infer.py --image path/to/image.jpg

# Save annotated output
python3 infer.py --image path/to/image.jpg --save output.png

# Live USB camera (default index 0)
python3 infer.py --camera

# Different camera index
python3 infer.py --camera --camera-index 1

# Custom checkpoint or threshold
python3 infer.py --image road.jpg --checkpoint road_best.pth --threshold 0.5
```

**Camera controls:** `Q` / `ESC` — quit &nbsp;&nbsp; `S` — save current frame as `capture.png`
