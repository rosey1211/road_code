# ─── road_code/model.py ───────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building block ───────────────────────────────────────────────────────────
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def row_head(in_ch, hidden=64):
    """
    1-D convolutional head that operates per bucket position.
    kernel_size=3 in the first layer lets adjacent buckets inform each other,
    which is important since the road is spatially continuous.
    Input  : [B, in_ch, N_BUCKETS]
    Output : [B, N_BUCKETS]  (logits)
    """
    return nn.Sequential(
        nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(hidden, 1, kernel_size=1),
    )


# ─── RoadCNN ──────────────────────────────────────────────────────────────────
class RoadCNN(nn.Module):
    """
    Deep CNN for road centerline estimation.

    Architecture
    ------------
    Backbone  : 4 × (Conv3×3-BN-ReLU)×2 + MaxPool2d  →  [B, 512, H', W']
    Classifier: Global Average Pool → Linear(512,256) → ReLU → Dropout → Linear(256,2)
    Row heads : per-row spatial strip pooled to [B, 512, N_BUCKETS]
                → Conv1d(512,64,1) → ReLU → Conv1d(64,1,1) → [B, N_BUCKETS]

    The row heads preserve full horizontal spatial information — each bucket
    position receives features from the correct vertical location in the image
    rather than a globally-pooled summary, which is critical for curved roads.

    Input  : RGB image tensor [B, 3, H, W]  — any resolution; model rescales internally
    Output : tuple of 4 tensors
               cls_logits    [B, 2]           road-present classifier (CE loss)
               row0_logits   [B, N_BUCKETS]   bucket logits for row 0  (BCE loss)
               row1_logits   [B, N_BUCKETS]   bucket logits for row 1  (BCE loss)
               row2_logits   [B, N_BUCKETS]   bucket logits for row 2  (BCE loss)
    """

    def __init__(self, image_width, image_height, n_buckets, row_fractions,
                 norm_mean, norm_std, crop_top=0.0, crop_bottom=0.0,
                 row_bucket_margins=None, bucket_nonlinearity=1.0):
        """
        Parameters
        ----------
        image_width          : int   — training/inference resolution width
        image_height         : int   — training/inference resolution height
        n_buckets            : int   — lateral buckets per row (must divide image_width)
        row_fractions        : list  — 3 floats in full-image coordinates, e.g. [0.22, 0.35, 0.55]
        norm_mean            : list  — per-channel normalisation mean (3 values)
        norm_std             : list  — per-channel normalisation std  (3 values)
        crop_top             : float — fraction of image height removed from the top
        crop_bottom          : float — fraction of image height removed from the bottom
        row_bucket_margins   : list  — 3 floats, lateral margin per row [r0, r1, r2]
                                       (fraction of image width excluded on each side)
        bucket_nonlinearity  : float — ratio of edge/centre bucket width in pixels
                                       (1.0 = uniform, 5.0 = edges 5× wider than centre)
        """
        if row_bucket_margins is None:
            row_bucket_margins = [0.0, 0.0, 0.0]
        assert image_width % n_buckets == 0, \
            f"n_buckets={n_buckets} must evenly divide image_width={image_width}"
        assert len(row_fractions) == 3, "row_fractions must have exactly 3 entries"
        assert len(row_bucket_margins) == 3, "row_bucket_margins must have exactly 3 entries"
        assert 0.0 <= crop_top < 1.0,    f"crop_top={crop_top} must be in [0, 1)"
        assert 0.0 <= crop_bottom < 1.0,  f"crop_bottom={crop_bottom} must be in [0, 1)"
        assert crop_top + crop_bottom < 1.0, "crop_top + crop_bottom must be < 1"
        assert all(0.0 <= m < 0.5 for m in row_bucket_margins), \
            f"all row_bucket_margins must be in [0, 0.5), got {row_bucket_margins}"
        assert bucket_nonlinearity >= 1.0, \
            f"bucket_nonlinearity={bucket_nonlinearity} must be >= 1.0"

        super().__init__()

        # ── Store all config as instance attributes ───────────────────────────
        self.image_width          = image_width
        self.image_height         = image_height
        self.n_buckets            = n_buckets
        self.bucket_width         = image_width // n_buckets
        self.row_bucket_margins   = list(row_bucket_margins)
        self.bucket_nonlinearity  = float(bucket_nonlinearity)

        # Per-row bucket edge fractions: [N_ROWS, N_BUCKETS+1].
        # Registered as a buffer so it is saved in the checkpoint and moved
        # to the correct device automatically.
        edges_list = [
            self._compute_edges(n_buckets, m, bucket_nonlinearity)
            for m in row_bucket_margins
        ]
        self.register_buffer("bucket_edges_all",
                             torch.tensor(edges_list, dtype=torch.float32))
        self.row_fractions = list(row_fractions)
        self.row_indices   = [int(f * image_height) for f in row_fractions]
        self.norm_mean     = list(norm_mean)
        self.norm_std      = list(norm_std)
        self.crop_top      = float(crop_top)
        self.crop_bottom   = float(crop_bottom)
        self.crop_top_px   = int(crop_top * image_height)
        self.crop_bot_px   = int((1.0 - crop_bottom) * image_height)

        # Row fractions remapped into the cropped image's coordinate space.
        # Used at forward time to locate the correct row in the feature map.
        crop_span = 1.0 - crop_top - crop_bottom
        self.row_fracs_in_crop = [
            (f - crop_top) / crop_span for f in row_fractions
        ]

        # ── Backbone ─────────────────────────────────────────────────────────
        # 3 maxpools (not 4) so the feature map is 10 rows tall instead of 5.
        # This gives well-separated strip extraction positions for the 3 scan
        # lines and halves the upsampling needed in adaptive_avg_pool.
        self.backbone = nn.Sequential(
            conv_block(3,   64),  nn.MaxPool2d(2, 2),
            conv_block(64, 128),  nn.MaxPool2d(2, 2),
            conv_block(128, 256), nn.MaxPool2d(2, 2),
            conv_block(256, 512),
        )

        # ── Classifier head (global) ──────────────────────────────────────────
        # Uses GAP — appropriate since road-presence is a global property.
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.head_cls = nn.Linear(256, 2)

        # ── Row heads (spatial) ───────────────────────────────────────────────
        # Each head operates on a row-specific strip of the feature map,
        # preserving horizontal spatial structure.  Conv1d acts per-bucket.
        self.head_row0 = row_head(512)
        self.head_row1 = row_head(512)
        self.head_row2 = row_head(512)

    # ── Bucket edge computation ───────────────────────────────────────────────
    @staticmethod
    def _compute_edges(N, margin, R):
        """
        Return a list of N+1 pixel fractions for bucket boundaries.
        Uses a quadratic-derivative warp so centre buckets are narrower
        and edge buckets are wider by a factor of R.
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

    # ── Row feature extractor ─────────────────────────────────────────────────
    def _row_features(self, feat_map, row_frac_in_crop, row_idx):
        """
        Extract a horizontal strip from the backbone feature map at the
        specified fractional row position, pool it vertically to 1 row,
        and pool horizontally into n_buckets using this row's edge table.

        Returns [B, 512, N_BUCKETS].
        """
        B, C, H, W = feat_map.shape
        fy = max(0, min(H - 1, int(row_frac_in_crop * H)))
        y0 = max(0, fy - 1)
        y1 = min(H, fy + 2)
        strip  = feat_map[:, :, y0:y1, :]          # [B, C, k, W]
        edges  = self.bucket_edges_all[row_idx]     # [N_BUCKETS+1]
        cols = []
        for b in range(self.n_buckets):
            x0 = max(0, int(edges[b].item()     * W))
            x1 = min(W, int(edges[b + 1].item() * W))
            x1 = max(x1, x0 + 1)
            col = strip[:, :, :, x0:x1].mean(dim=-1)   # [B, C, k]
            cols.append(col)
        pooled = torch.stack(cols, dim=-1)          # [B, C, k, N]
        return pooled.mean(dim=2)                   # [B, C, N]

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x):
        # Resize to training resolution — works for any input resolution
        x = F.interpolate(x, size=(self.image_height, self.image_width),
                          mode='bilinear', align_corners=False)
        # Crop top and bottom bands
        if self.crop_top_px > 0 or self.crop_bot_px < self.image_height:
            x = x[:, :, self.crop_top_px:self.crop_bot_px, :]

        feat = self.backbone(x)   # [B, 512, H', W']

        # Classifier — global
        cls = self.head_cls(self.neck(self.gap(feat)))

        # Row heads — spatially aware
        r0 = self.head_row0(self._row_features(feat, self.row_fracs_in_crop[0], 0)).squeeze(1)
        r1 = self.head_row1(self._row_features(feat, self.row_fracs_in_crop[1], 1)).squeeze(1)
        r2 = self.head_row2(self._row_features(feat, self.row_fracs_in_crop[2], 2)).squeeze(1)

        return cls, r0, r1, r2

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def get_model_config(self):
        return {
            "image_width":   self.image_width,
            "image_height":  self.image_height,
            "n_buckets":     self.n_buckets,
            "row_fractions": self.row_fractions,
            "norm_mean":     self.norm_mean,
            "norm_std":      self.norm_std,
            "crop_top":              self.crop_top,
            "crop_bottom":           self.crop_bottom,
            "row_bucket_margins":    self.row_bucket_margins,
            "bucket_nonlinearity":   self.bucket_nonlinearity,
        }

    @classmethod
    def from_checkpoint(cls, path, device=None):
        device = device or torch.device("cpu")
        ckpt   = torch.load(path, map_location=device)

        required = {"image_width", "image_height", "n_buckets",
                    "row_fractions", "norm_mean", "norm_std",
                    "crop_top", "crop_bottom", "model_state"}
        missing = required - set(ckpt.keys())
        if missing:
            raise KeyError(f"Checkpoint is missing keys: {missing}. "
                           "Re-train with the current train.py to get a complete checkpoint.")

        # Backward compat: old checkpoints had a single bucket_margin scalar
        _margins = ckpt.get("row_bucket_margins",
                            [ckpt.get("bucket_margin", 0.0)] * 3)
        model = cls(
            image_width   = ckpt["image_width"],
            image_height  = ckpt["image_height"],
            n_buckets     = ckpt["n_buckets"],
            row_fractions = ckpt["row_fractions"],
            norm_mean     = ckpt["norm_mean"],
            norm_std      = ckpt["norm_std"],
            crop_top             = ckpt.get("crop_top",            0.0),
            crop_bottom          = ckpt.get("crop_bottom",         0.0),
            row_bucket_margins   = _margins,
            bucket_nonlinearity  = ckpt.get("bucket_nonlinearity", 1.0),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        return model, ckpt.get("model_config", model.get_model_config()), ckpt.get("epoch", -1)

    def describe(self):
        n_params = sum(p.numel() for p in self.parameters())
        crop_h   = self.crop_bot_px - self.crop_top_px
        print(f"RoadCNN  (spatial row heads)")
        print(f"  Input size        : {self.image_width} × {self.image_height}")
        print(f"  Crop top/bottom   : {self.crop_top:.2%} / {self.crop_bottom:.2%}"
              f"  →  active height {crop_h} px")
        print(f"  N_BUCKETS         : {self.n_buckets}  (bucket width = {self.bucket_width} px)")
        print(f"  Bucket nonlin     : {self.bucket_nonlinearity}")
        for ri, m in enumerate(self.row_bucket_margins):
            edges  = self.bucket_edges_all[ri]
            ctr_w  = (edges[self.n_buckets//2+1] - edges[self.n_buckets//2]).item() * self.image_width
            edge_w = (edges[1] - edges[0]).item() * self.image_width
            print(f"  r{ri} margin={m:.0%}  centre≈{ctr_w:.1f}px  edge≈{edge_w:.1f}px")
        print(f"  Row fractions     : {self.row_fractions}  (full-image)")
        print(f"  Row fracs in crop : {[f'{f:.3f}' for f in self.row_fracs_in_crop]}")
        print(f"  Row indices       : {self.row_indices}")
        print(f"  Output nodes      : 2 + 3×{self.n_buckets} = {2 + 3 * self.n_buckets}")
        print(f"  Parameters        : {n_params:,}")


# ─── Loss function ────────────────────────────────────────────────────────────
def road_loss(outputs, road_present, bucket_masks,
              lambda_cls=1.0, lambda_row=1.0, row_weights=None):
    """
    Combined classification + per-row bucket BCE loss.

    outputs      : (cls_logits, r0, r1, r2)
    road_present : LongTensor [B]
    bucket_masks : FloatTensor [B, 3, N_BUCKETS]
    lambda_cls   : weight for classifier loss
    lambda_row   : weight for bucket losses
    row_weights  : list of 3 floats weighting r0/r1/r2 (default equal)
    """
    if row_weights is None:
        row_weights = [1.0, 1.0, 1.0]

    cls_logits, r0, r1, r2 = outputs

    cls_l = F.cross_entropy(cls_logits, road_present)

    road_mask = road_present.bool()
    if road_mask.any():
        w0, w1, w2 = row_weights
        # BCE with Gaussian soft targets — better than MSE for sparse distributions.
        # BCE naturally equilibrates: sigmoid→1 where target=1, sigmoid→0 where
        # target=0, and intermediate values where target is in (0,1).
        # MSE + sigmoid has vanishing gradients once outputs are pushed near 0.
        r0_l = F.binary_cross_entropy_with_logits(r0[road_mask], bucket_masks[road_mask, 0])
        r1_l = F.binary_cross_entropy_with_logits(r1[road_mask], bucket_masks[road_mask, 1])
        r2_l = F.binary_cross_entropy_with_logits(r2[road_mask], bucket_masks[road_mask, 2])
        w_sum = w0 + w1 + w2
        row_l = (w0 * r0_l + w1 * r1_l + w2 * r2_l) / w_sum
    else:
        row_l = torch.tensor(0.0, device=cls_logits.device)

    total = lambda_cls * cls_l + lambda_row * row_l
    return total, cls_l, row_l


# ─── Quick model summary ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import config
    config.init()

    model = RoadCNN(
        image_width   = config.IMAGE_WIDTH,
        image_height  = config.IMAGE_HEIGHT,
        n_buckets     = config.N_BUCKETS,
        row_fractions = config.ROW_FRACTIONS,
        norm_mean     = config.NORM_MEAN,
        norm_std      = config.NORM_STD,
        crop_top      = config.CROP_TOP,
        crop_bottom   = config.CROP_BOTTOM,
    )
    model.describe()

    dummy = torch.randn(2, 3, 480, 640)
    cls_l, r0, r1, r2 = model(dummy)
    print(f"\ncls_logits : {list(cls_l.shape)}")
    print(f"row logits : {list(r0.shape)}")
