# ─── road_code/infer.py ───────────────────────────────────────────────────────
"""
Road centerline inference — single image or live USB camera.

Usage
-----
# Single image
python infer.py --image path/to/image.jpg

# Live camera (default index 0)
python infer.py --camera

# Live camera on a different index
python infer.py --camera --camera-index 1

# Override checkpoint path
python infer.py --image road.jpg --checkpoint road_best.pth

Controls (camera mode)
----------------------
  Q / ESC  — quit
  S        — save current annotated frame as capture.png
"""

import argparse
import sys
import os
import time

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config
from model     import RoadCNN
from visualize import overlay_predictions, overlay_inference


# ─── Load model from checkpoint ───────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        sys.exit(f"ERROR: checkpoint not found: {checkpoint_path}\n"
                 f"Run train.py first.")

    print(f"Loading checkpoint: {checkpoint_path}")
    model, cfg, epoch = RoadCNN.from_checkpoint(checkpoint_path, device=device)
    model.describe()
    print(f"  Trained to epoch: {epoch}\n")
    return model, cfg


# ─── Build preprocessing transform from checkpoint norms ─────────────────────
def make_transform(norm_mean, norm_std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])


# ─── Single image inference ───────────────────────────────────────────────────
def run_image(args, device):
    model, cfg = load_model(args.checkpoint, device)
    transform  = make_transform(model.norm_mean, model.norm_std)

    img_path = args.image
    if not os.path.exists(img_path):
        sys.exit(f"ERROR: image not found: {img_path}")

    pil_img = Image.open(img_path).convert("RGB")
    tensor  = transform(pil_img).unsqueeze(0).to(device)   # [1, 3, H, W]

    with torch.no_grad():
        outputs = model(tensor)

    # Decode road-present classification
    cls_probs   = outputs[0].squeeze(0).softmax(0)
    road_present = cls_probs[0].item() > 0.5
    cls_conf     = cls_probs[0].item()

    print(f"Classification : {'ROAD' if road_present else 'NO ROAD'}  "
          f"({cls_conf*100:.1f}% road confidence)")

    # Decode active buckets per row
    if road_present:
        for r, (logits, row_idx) in enumerate(
                zip(outputs[1:], model.row_indices)):
            probs  = torch.sigmoid(logits.squeeze(0))
            active = (probs >= args.threshold).nonzero(as_tuple=True)[0].tolist()
            centers = [f"{(b + 0.5) / model.n_buckets * model.image_width:.1f}px "
                       f"(bucket {b})" for b in active]
            print(f"  Row {r} (y={row_idx}): {centers if centers else 'no active bucket'}")

    # Draw overlay and display / save
    canvas, label = overlay_predictions(
        pil_img, tuple(o.squeeze(0) for o in outputs),
        model, threshold=args.threshold
    )

    try:
        import cv2
        bgr = canvas[:, :, ::-1]
        cv2.imshow("Road Inference", bgr)
        print("\nPress any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except ImportError:
        # Fall back to matplotlib if OpenCV not available
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.imshow(canvas)
        plt.title(label)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if args.save:
        out_path = args.save
        Image.fromarray(canvas).save(out_path)
        print(f"Saved: {out_path}")


# ─── Live camera inference ────────────────────────────────────────────────────
def run_camera(args, device):
    try:
        import cv2
    except ImportError:
        sys.exit("ERROR: opencv-python is required for camera mode.\n"
                 "Install with: pip install opencv-python")

    model, cfg = load_model(args.checkpoint, device)
    transform  = make_transform(model.norm_mean, model.norm_std)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open camera index {args.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Camera {args.camera_index} opened. Press Q/ESC to quit, S to save frame.\n")

    WIN = "Road Centerline Inference"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 640, 480)

    fps  = 0.0
    frame_bgr = None

    while True:
        t0 = time.time()

        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue

        # BGR → RGB PIL → tensor
        frame_rgb = frame_bgr[:, :, ::-1]
        pil_frame = Image.fromarray(frame_rgb.astype(np.uint8))
        tensor    = transform(pil_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)

        # Draw overlay (returns BGR)
        annotated = overlay_inference(frame_bgr, outputs, model,
                                      threshold=args.threshold)

        # FPS counter
        fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-6)
        cv2.putText(annotated, f"FPS {fps:.0f}",
                    (8, annotated.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(WIN, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('s'), ord('S')) and frame_bgr is not None:
            cv2.imwrite("capture.png", annotated)
            print("Saved: capture.png")

    cap.release()
    cv2.destroyAllWindows()


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Road centerline inference")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image",  type=str, help="Path to a single image file")
    mode.add_argument("--camera", action="store_true", help="Live USB camera mode")

    p.add_argument("--checkpoint",    default=config.CHECKPOINT,
                   help=f"Checkpoint file (default: {config.CHECKPOINT})")
    p.add_argument("--threshold",     type=float, default=0.5,
                   help="Sigmoid threshold for bucket activation (default: 0.5)")
    p.add_argument("--camera-index",  type=int, default=0,
                   help="Camera device index (default: 0)")
    p.add_argument("--save",          type=str, default=None,
                   help="Save annotated image to this path (image mode only)")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cpu")   # CPU-only; add --cuda flag if needed

    if args.camera:
        run_camera(args, device)
    else:
        run_image(args, device)
