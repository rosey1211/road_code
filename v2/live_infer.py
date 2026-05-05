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
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import infer_config as cfg
import config as _cfg
from dataset import RoadDataset
from model   import RoadCNN


WINDOW = "live_infer  |  q = quit  |  any key = advance"

# ── Colours (RGB uint8) ───────────────────────────────────────────────────────
_PRED   = (50,  160, 255)   # predicted peak circles / lines — bright blue
_GUIDE  = (0,   210,  80)   # row guide dashes — bright green
_YELLOW = (255, 220, 0)     # GT bars


# ── Image helpers ─────────────────────────────────────────────────────────────
def _denorm(tensor, mean, std):
    """CHW float32 tensor → uint8 HWC numpy (RGB)."""
    m = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    s = np.array(std,  dtype=np.float32).reshape(3, 1, 1)
    img = tensor.cpu().numpy() * s + m
    return (img.clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


def _dashed_hline(canvas, y, colour=_GUIDE, dash=10, thickness=4):
    """Draw a dashed horizontal line at pixel row y (RGB canvas)."""
    h, w = canvas.shape[:2]
    if not (0 <= y < h):
        return
    for x in range(0, w, dash * 2):
        y0 = max(0, y - thickness // 2)
        y1 = min(h, y0 + thickness)
        canvas[y0:y1, x:x + dash] = colour


# ── Main overlay builder ───────────────────────────────────────────────────────
def _build_frame(image_tensor, outputs, gt_masks, model, state):
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

    # Upscale small images to a minimum display height so overlays stay readable.
    # Bucket edge fractions and row fractions are resolution-independent, so
    # all overlay geometry recalculates correctly on the upscaled canvas.
    _MIN_DISPLAY_H = 480
    if h < _MIN_DISPLAY_H:
        _up = _MIN_DISPLAY_H / h
        canvas = cv2.resize(canvas, (int(w * _up), _MIN_DISPLAY_H),
                            interpolation=cv2.INTER_LINEAR)
        h, w = canvas.shape[:2]

    # Scale graphic sizes relative to the 480-px reference height
    _s       = h / 480.0
    _lw      = max(1, round(2  * _s))   # line / curve width
    _cr      = max(2, round(6  * _s))   # peak circle radius
    _dot_r   = max(1, round(2  * _s))   # white centre dot
    _fs      = max(0.5, 0.9 * _s)       # confidence label font scale
    _fs_lbl  = max(0.3, 0.7  * _s)      # status-bar font scale
    _lbl_h   = max(16, round(36 * _s))  # status bar height
    _lbl_y   = max(11, round(26 * _s))  # text baseline inside status bar
    _border  = max(3,  round(8  * _s))  # no-road red border width
    _dash    = max(4,  round(8  * _s))  # guide line dash length
    _txt_off = max(2,  round(6  * _s))  # label y-offset above circle

    def _row_edge_info(ri):
        """Return (bx0, bx1) lists for row ri using the per-row edge table."""
        edges = model.bucket_edges_all[ri]
        _bx0 = [max(0,   int(edges[b].item()     * w))     for b in range(n)]
        _bx1 = [min(w-1, max(int(edges[b+1].item() * w), _bx0[b] + 1))
                for b in range(n)]
        _exs = [int(edges[b].item() * w) for b in range(n + 1)]
        return _bx0, _bx1, _exs

    def _bucket_cx(b, ri):
        edges = model.bucket_edges_all[ri]
        return int(((edges[b].item() + edges[b + 1].item()) / 2.0) * w)

    # Unpack outputs (single-item batches → squeeze batch dim)
    cls_logits, r0_l, r1_l, r2_l = [o.detach().cpu().squeeze(0) for o in outputs]

    road_prob    = cls_logits.softmax(0)[1].item()
    road_present = road_prob > 0.5
    row_logits   = [r0_l, r1_l, r2_l]

    # ── GT yellow translucent bars ────────────────────────────────────────────
    bar_half_h = max(2, h // 14)
    overlay    = canvas.copy()
    for r, frac in enumerate(model.row_fractions):
        bx0, bx1, _ = _row_edge_info(r)
        row_y    = int(frac * h)
        y0       = max(0, row_y - bar_half_h * 2)
        y1       = row_y
        gt_peaks = [b for b in range(n) if gt_masks[r, b].item() > 0.9]
        for b in gt_peaks:
            cv2.rectangle(overlay, (bx0[b], y0), (bx1[b], y1),
                          _YELLOW, cv2.FILLED)
    canvas = cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0)

    # ── Row guide dashed lines + bucket hash marks ────────────────────────────
    for ri, frac in enumerate(model.row_fractions):
        row_y = int(frac * h)
        _dashed_hline(canvas, row_y, dash=_dash, thickness=_lw)

    # ── Debug: sigmoid response bars on each scan line ────────────────────────
    if cfg.DEBUG_MODE:
        _debug_alpha  = 0.45
        _debug_colour = np.array((0, 220, 80), dtype=np.float32)  # bright green
        _bar_max_h    = max(2, h // 8)
        _debug_overlay = canvas.copy()
        for ri, (frac, logits) in enumerate(zip(model.row_fractions, row_logits)):
            row_y  = int(frac * h)
            probs  = torch.sigmoid(logits)
            bx0_d, bx1_d, _ = _row_edge_info(ri)
            for b in range(n):
                p = probs[b].item()
                if p < 0.01:
                    continue
                bar_h = max(1, int(p * _bar_max_h))
                y0d = max(0, row_y - bar_h)
                y1d = row_y
                region = _debug_overlay[y0d:y1d, bx0_d[b]:bx1_d[b]+1].astype(np.float32)
                _debug_overlay[y0d:y1d, bx0_d[b]:bx1_d[b]+1] = np.clip(
                    region * (1 - _debug_alpha) + _debug_colour * _debug_alpha, 0, 255
                ).astype(np.uint8)
        canvas = cv2.addWeighted(_debug_overlay, 1.0, canvas, 0.0, 0)

    # ── Prediction peaks, confidence, and connecting lines ───────────────────
    def _find_row_peaks(ri, logits, hint_peaks=None, prev_cx=None):
        """
        Find 1-2 peaks by cluster detection.
        Buckets above cluster_thresh form contiguous clusters; each cluster
        produces exactly one peak (its weighted centroid / argmax confidence).
        When multiple clusters exist, prev_cx biases selection toward the cluster
        nearest the previous frame's primary peak (resolves ambiguity).
        A secondary cluster is accepted only if it meets the fork threshold
        (lowered when a hint peak from the adjacent far row is nearby).
        Returns (list of peak dicts sorted by cx, primary_conf, n_clusters).
        Each dict: {cx, bl, br, conf, row_y}
        """
        row_y  = int(model.row_fractions[ri] * h)
        probs  = torch.sigmoid(logits)
        p_vals = [probs[b].item() for b in range(n)]
        peak_conf = max(p_vals)

        cluster_thresh = max(0.05, peak_conf * cfg.CLUSTER_THRESH_FRAC)

        # Find contiguous runs of active buckets
        clusters = []
        b_start = None
        for b in range(n):
            if p_vals[b] >= cluster_thresh:
                if b_start is None:
                    b_start = b
            else:
                if b_start is not None:
                    clusters.append((b_start, b - 1))
                    b_start = None
        if b_start is not None:
            clusters.append((b_start, n - 1))

        if not clusters:
            peak_b = int(probs.argmax().item())
            cx = _bucket_cx(peak_b, ri)
            return [{'cx': cx, 'bl': peak_b, 'br': peak_b,
                     'conf': peak_conf, 'row_y': row_y}], peak_conf, 0

        def _cluster_info(b_l, b_r):
            conf = max(p_vals[b] for b in range(b_l, b_r + 1))
            tw   = sum(p_vals[b] for b in range(b_l, b_r + 1))
            cx   = int(round(sum(_bucket_cx(b, ri) * p_vals[b]
                                 for b in range(b_l, b_r + 1)) / tw))
            return {'cx': cx, 'bl': b_l, 'br': b_r, 'conf': conf, 'row_y': row_y}

        cluster_peaks = [_cluster_info(b_l, b_r) for b_l, b_r in clusters]

        # Bias cluster ranking toward previous frame's peak position to resolve
        # ambiguity when two clusters have similar confidence.
        if prev_cx is not None:
            prox_range_px = w * cfg.TEMPORAL_PROXIMITY_RANGE
            for cp in cluster_peaks:
                dist = abs(cp['cx'] - prev_cx)
                proximity = max(0.0, 1.0 - dist / prox_range_px)
                cp['_eff_conf'] = cp['conf'] * (1.0 + cfg.TEMPORAL_BIAS_WEIGHT * proximity)
        else:
            for cp in cluster_peaks:
                cp['_eff_conf'] = cp['conf']

        cluster_peaks.sort(key=lambda x: -x['_eff_conf'])

        primary      = cluster_peaks[0]
        primary_conf = primary['conf']
        result       = [primary]

        if len(cluster_peaks) > 1:
            secondary  = cluster_peaks[1]
            sec_thresh = primary_conf * cfg.FORK_SUPPRESS_THRESH
            if hint_peaks:
                prox_px = w * cfg.FORK_HINT_PROXIMITY
                for hp in hint_peaks:
                    if abs(hp['cx'] - secondary['cx']) < prox_px:
                        sec_thresh = min(sec_thresh,
                                         primary_conf * cfg.FORK_HINT_THRESH)
                        break
            if secondary['conf'] >= sec_thresh:
                result.append(secondary)

        result.sort(key=lambda x: x['cx'])
        # Count only clusters strong enough to be genuinely multimodal
        _SIGNIFICANT_FRAC = 0.5
        n_significant = sum(1 for cp in cluster_peaks
                            if cp['conf'] >= primary_conf * _SIGNIFICANT_FRAC)
        return result, primary_conf, n_significant

    def _draw_branch(p_near, p_mid, p_far, x_lo=0, x_hi=None):
        """Straight line near→mid, then cubic Bezier mid→far (if far exists).
        x_lo/x_hi constrain the curve laterally to prevent fork cross-over."""
        if x_hi is None:
            x_hi = w - 1

        # Tighten x bounds by CURVE_SWING_FACTOR × vertical span of branch
        pts_x = [p_near[0], p_mid[0]]
        vert_span = abs(p_mid[1] - p_near[1])
        if p_far is not None:
            pts_x.append(p_far[0])
            vert_span += abs(p_far[1] - p_mid[1])
        swing = int(cfg.CURVE_SWING_FACTOR * vert_span)
        x_lo = max(x_lo, min(pts_x) - swing)
        x_hi = min(x_hi, max(pts_x) + swing)

        cv2.line(canvas, p_near, p_mid, _PRED, _lw, cv2.LINE_AA)
        if p_far is None:
            return
        _p2 = np.array(p_near, dtype=np.float64)
        _p1 = np.array(p_mid,  dtype=np.float64)
        _p0 = np.array(p_far,  dtype=np.float64)
        _seg = _p1 - _p2
        _seg_len = np.linalg.norm(_seg)
        _tdir_s = _seg / _seg_len if _seg_len > 0 else np.array([0.0, -1.0])
        _chord = _p0 - _p1
        _chord_len = np.linalg.norm(_chord)
        _chord_hat = _chord / _chord_len if _chord_len > 0 else _tdir_s
        _tdir_e = 2.0 * float(np.dot(_tdir_s, _chord_hat)) * _chord_hat - _tdir_s
        _alpha = _chord_len / 3.0
        _t = np.linspace(0, 1, max(20, int(_chord_len)))
        _bez = (np.outer((1-_t)**3, _p1) + np.outer(3*(1-_t)**2*_t, (_p1 + _alpha*_tdir_s)) +
                np.outer(3*(1-_t)*_t**2, (_p0 - _alpha*_tdir_e)) + np.outer(_t**3, _p0))
        _bp = _bez.astype(np.int32)
        _bp[:, 0] = _bp[:, 0].clip(x_lo, x_hi)
        _bp[:, 1] = _bp[:, 1].clip(0, h - 1)
        cv2.polylines(canvas, [_bp.reshape(-1, 1, 2)], False, _PRED, _lw, cv2.LINE_AA)

    # Find peaks farthest→nearest so r0 peaks can hint at r1 secondary peaks.
    # Previous frame's primary cx biases cluster selection to resolve ambiguity.
    prev_cx = state.get('prev_cx', [None, None, None])
    peaks_r0, conf_r0, ncl_r0 = _find_row_peaks(0, row_logits[0],
                                                  prev_cx=prev_cx[0])
    peaks_r1, conf_r1, ncl_r1 = _find_row_peaks(1, row_logits[1],
                                                  hint_peaks=peaks_r0,
                                                  prev_cx=prev_cx[1])
    peaks_r2, conf_r2, ncl_r2 = _find_row_peaks(2, row_logits[2],
                                                  prev_cx=prev_cx[2])
    row_confs = [conf_r0, conf_r1, conf_r2]
    overall_conf = sum(row_confs) / 3.0

    if overall_conf < cfg.MIN_PEAK_CONF:
        road_present = False
    if max(ncl_r0, ncl_r1, ncl_r2) >= 3:
        road_present = False

    # Road momentum: decay when no road detected; restore from previous peaks
    # if momentum is still above threshold (roads don't just disappear)
    if road_present:
        state['road_momentum'] = 1.0
    else:
        state['road_momentum'] = state.get('road_momentum', 0.0) * cfg.ROAD_MOMENTUM_DECAY
        if state['road_momentum'] >= cfg.ROAD_MOMENTUM_THRESH:
            prev = state.get('prev_peaks')
            if prev is not None:
                peaks_r0, peaks_r1, peaks_r2 = prev
                road_present = True

    if road_present:
        # Determine branches: fork if >1 peak on middle or far row
        n_branches = min(max(len(peaks_r1), len(peaks_r0)), 2)

        # Match each r0 peak to the spatially nearest r1 branch.
        # 1 r0 peak  → assign to the closest r1 branch; other branch gets None.
        # 2 r0 peaks → try both pairings, keep the one with min total x-distance.
        r0_for_branch = [None] * n_branches
        if n_branches == 2:
            r1_cx = [peaks_r1[bi]['cx'] if bi < len(peaks_r1) else peaks_r1[-1]['cx']
                     for bi in range(2)]
            if len(peaks_r0) == 1:
                a0 = peaks_r0[0]['cx']
                best = 0 if abs(a0 - r1_cx[0]) <= abs(a0 - r1_cx[1]) else 1
                r0_for_branch[best] = peaks_r0[0]
            elif len(peaks_r0) >= 2:
                a0, a1 = peaks_r0[0]['cx'], peaks_r0[1]['cx']
                if (abs(r1_cx[0]-a1) + abs(r1_cx[1]-a0) <
                        abs(r1_cx[0]-a0) + abs(r1_cx[1]-a1)):
                    r0_for_branch = [peaks_r0[1], peaks_r0[0]]
                else:
                    r0_for_branch = [peaks_r0[0], peaks_r0[1]]
        elif n_branches == 1 and len(peaks_r0) >= 1:
            r0_for_branch = [peaks_r0[0]]

        p_near = (peaks_r2[0]['cx'], peaks_r2[0]['row_y'])
        branches = []
        for bi in range(n_branches):
            r1_pk = peaks_r1[bi] if bi < len(peaks_r1) else peaks_r1[-1]
            r0_pk = r0_for_branch[bi]
            p_mid = (r1_pk['cx'], r1_pk['row_y'])
            p_far = (r0_pk['cx'], r0_pk['row_y']) if r0_pk else None
            branches.append((p_near, p_mid, p_far))

        # Translucent bucket rectangles for all peaks
        bucket_overlay = canvas.copy()
        rect_half_h = max(2, h // 30)
        for ri, pks in [(0, peaks_r0), (1, peaks_r1), (2, peaks_r2)]:
            bx0_r, bx1_r, _ = _row_edge_info(ri)
            for pk in pks:
                pb = (pk['bl'] + pk['br']) // 2
                cy = pk['row_y']
                cv2.rectangle(bucket_overlay,
                              (bx0_r[pb], cy - rect_half_h),
                              (bx1_r[pb], cy + rect_half_h),
                              _PRED, cv2.FILLED)
        canvas = cv2.addWeighted(bucket_overlay, 0.35, canvas, 0.65, 0)

        # For a fork, compute the x-separator between branches and constrain
        # each curve to stay on its own side (prevents cross-over chaos).
        if n_branches == 2:
            sep_x = (branches[0][1][0] + branches[1][1][0]) // 2
            branch_x_bounds = [(0, sep_x), (sep_x, w - 1)]
        else:
            branch_x_bounds = [(0, w - 1)]

        # Draw one curve per branch
        for bi, (p_near, p_mid, p_far) in enumerate(branches):
            x_lo, x_hi = branch_x_bounds[bi]
            _draw_branch(p_near, p_mid, p_far, x_lo=x_lo, x_hi=x_hi)

        # Store peaks for next frame's disambiguation and momentum fallback
        state['prev_cx'] = [
            peaks_r0[0]['cx'] if peaks_r0 else None,
            peaks_r1[0]['cx'] if peaks_r1 else None,
            peaks_r2[0]['cx'] if peaks_r2 else None,
        ]
        state['prev_peaks'] = [peaks_r0, peaks_r1, peaks_r2]

        # Filled circles for all peaks
        all_pks = peaks_r0 + peaks_r1 + peaks_r2
        _txt_margin = max(10, round(80 * _s))
        _txt_x_off  = max(2,  round(15 * _s))
        for pk in all_pks:
            cx, cy, conf = pk['cx'], pk['row_y'], pk['conf']
            cv2.circle(canvas, (cx, cy), _cr,    _PRED,          cv2.FILLED, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), _cr,    (10, 40, 80),   max(1, round(2 * _s)), cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), _dot_r, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
            conf_txt = f"{conf*100:.0f}%"
            tx = min(cx + _txt_x_off, w - _txt_margin)
            cv2.putText(canvas, conf_txt, (tx, cy - _txt_off),
                        cv2.FONT_HERSHEY_PLAIN, _fs, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, conf_txt, (tx, cy - _txt_off),
                        cv2.FONT_HERSHEY_PLAIN, _fs, _PRED,     1, cv2.LINE_AA)


    # ── No-road red perimeter ─────────────────────────────────────────────────
    if not road_present:
        canvas[:_border,  :] = (220, 30, 30)
        canvas[-_border:, :] = (220, 30, 30)
        canvas[:,  :_border] = (220, 30, 30)
        canvas[:, -_border:] = (220, 30, 30)

    # ── Status label bar ──────────────────────────────────────────────────────
    bar_col  = (30, 140, 30) if road_present else (160, 40, 40)
    if road_present:
        label = (f"ROAD {road_prob*100:.0f}%"
                 f"   peak conf: {overall_conf*100:.0f}%"
                 f"  [{row_confs[0]*100:.0f}% / {row_confs[1]*100:.0f}% / {row_confs[2]*100:.0f}%]")
    else:
        label = f"NO ROAD {(1-road_prob)*100:.0f}%"
    label_bar = np.full((_lbl_h, w, 3), bar_col, dtype=np.uint8)
    cv2.putText(label_bar, label, (max(2, round(8 * _s)), _lbl_y),
                cv2.FONT_HERSHEY_SIMPLEX, _fs_lbl, (255, 255, 255), max(1, round(2 * _s)), cv2.LINE_AA)
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

    # Suppress the C++-level NNPACK warning that bypasses Python's warning system.
    # Redirect fd 2 (stderr) to /dev/null only during model forward calls.
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)

    class _SuppressStderr:
        def __enter__(self):
            self._saved = os.dup(2); os.dup2(_devnull_fd, 2)
        def __exit__(self, *_):
            os.dup2(self._saved, 2); os.close(self._saved)

    _suppress = _SuppressStderr()

    model, _, epoch = RoadCNN.from_checkpoint(args.checkpoint, device=device)
    model.eval()
    _cfg.BUCKET_EDGES = model.bucket_edges_all.tolist()
    print(f"Checkpoint : {args.checkpoint}  (epoch {epoch})")
    print(f"Device     : {device}")

    dataset_dir = args.dataset
    csv_path    = os.path.join(dataset_dir, cfg.LABELS_CSV_NAME)
    if not os.path.exists(csv_path):
        parent_dir      = os.path.dirname(dataset_dir)
        parent_csv_path = os.path.join(parent_dir, cfg.LABELS_CSV_NAME)
        if os.path.exists(parent_csv_path):
            print(f"No labels.csv in {dataset_dir}, falling back to {parent_dir}")
            dataset_dir = parent_dir
            csv_path    = parent_csv_path
        else:
            sys.exit(f"ERROR: labels CSV not found in {dataset_dir} or {parent_dir}\n"
                     "Edit DATASET_DIR in infer_config.py or pass --dataset.")

    ds = RoadDataset(csv_path=csv_path, images_dir=dataset_dir, augment=False)

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

    total  = len(ds)
    print(f"Dataset    : {dataset_dir}  ({total} images)")
    if args.delay == 0:
        print("Mode       : any keypress to advance  (q to quit, f = +10, b = -10)\n")
    else:
        print(f"Mode       : auto-advance every {args.delay:.3f}s  (q to quit, f = +10, b = -10)\n")

    wait_ms  = max(1, int(args.delay * 1000)) if args.delay > 0 else 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    idx          = 0
    paused       = False
    state        = {'prev_cx': [None, None, None], 'prev_peaks': None, 'road_momentum': 0.0}
    _rate_count  = 0
    _rate_start  = time.time()
    with torch.no_grad():
        while 0 <= idx < total:
            if not paused:
                image, road_present, gt_masks = ds[idx]
                image = image.unsqueeze(0).to(device)
                with _suppress:
                    outputs = model(image)

                img_t = image.squeeze(0).cpu()
                gt_t  = gt_masks.cpu()
                outs  = tuple(o.cpu() for o in outputs)

                frame, road_prob, pred_road, overall_conf, row_confs = \
                    _build_frame(img_t, outs, gt_t, model, state)


                _rate_count += 1
                if _rate_count % 20 == 0:
                    _elapsed = time.time() - _rate_start
                    print(f"  --- avg cycle rate: {20 / _elapsed:.1f} fps ---")
                    _rate_start = time.time()

                cv2.imshow(WINDOW, frame[:, :, ::-1])   # RGB → BGR for cv2

            key = cv2.waitKey(1 if paused else wait_ms) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("f"):
                idx = min(idx + 50, total - 1)
                paused = False
                state.update({'prev_cx': [None, None, None], 'prev_peaks': None, 'road_momentum': 0.0})
            elif key == ord("b"):
                idx = max(idx - 50, 0)
                paused = False
                state.update({'prev_cx': [None, None, None], 'prev_peaks': None, 'road_momentum': 0.0})
            elif not paused:
                idx += 1

    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
