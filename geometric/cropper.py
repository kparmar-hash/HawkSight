"""
YOLO Robot Crop Extractor
--------------------------
Uses a trained YOLOv8 detector to find robots in frames
and saves padded, resized crops ready for keypoint labeling.

Requirements:
    pip install ultralytics opencv-python tqdm

Usage:
    python extract_crops.py --weights runs/detect/frc2026_robots/weights/best.pt
                            --frames  data/frames
                            --out     data/crops
"""

import argparse
import logging
from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_CONF    = 0.45    # detection confidence threshold
DEFAULT_PAD     = 0.12    # padding around each crop as fraction of box size
DEFAULT_MIN_PX  = 40      # minimum crop width/height in pixels — skip tiny robots
DEFAULT_RESIZE  = 640     # resize crops to this square size for uniform labeling UX
DEFAULT_QUALITY = 95      # JPEG save quality


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def extract_crops(
    weights:    str,
    frames_dir: str,
    out_dir:    str,
    conf:       float = DEFAULT_CONF,
    pad:        float = DEFAULT_PAD,
    min_px:     int   = DEFAULT_MIN_PX,
    resize:     int   = DEFAULT_RESIZE,
    device:     str   = "",
) -> dict:
    """
    Run YOLO detector on every frame in frames_dir, crop each detection,
    and save to out_dir.

    Returns a summary dict.
    """
    model  = YOLO(weights)
    frames = sorted(Path(frames_dir).rglob("*.jpg"))

    if not frames:
        log.error(f"No .jpg frames found in {frames_dir}")
        return {}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_saved  = 0
    total_skipped = 0
    per_frame_counts = []

    log.info(f"Loaded model: {weights}")
    log.info(f"Processing {len(frames)} frames  →  crops to {out_dir}")

    for img_path in tqdm(frames, desc="Extracting crops"):
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning(f"Could not read {img_path.name} — skipping")
            continue

        img_h, img_w = img.shape[:2]

        results = model(
            img_path,
            conf=conf,
            verbose=False,
            device=device,
        )[0]

        frame_saved = 0

        for det_idx, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box.tolist())

            # Compute padding
            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = int(box_w * pad)
            pad_y = int(box_h * pad)

            # Clamp to image bounds
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(img_w, x2 + pad_x)
            cy2 = min(img_h, y2 + pad_y)

            crop_w = cx2 - cx1
            crop_h = cy2 - cy1

            # Skip crops that are too small to be useful
            if crop_w < min_px or crop_h < min_px:
                total_skipped += 1
                continue

            crop = img[cy1:cy2, cx1:cx2]

            # Resize to uniform size for consistent labeling UX in CVAT
            if resize:
                crop = cv2.resize(crop, (resize, resize), interpolation=cv2.INTER_LINEAR)

            # Filename encodes source frame + detection index for traceability
            stem     = img_path.stem
            out_path = out / f"{stem}_det{det_idx:02d}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_QUALITY])

            total_saved += 1
            frame_saved += 1

        per_frame_counts.append(frame_saved)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    avg_per_frame = sum(per_frame_counts) / max(len(per_frame_counts), 1)
    zero_detect   = sum(1 for c in per_frame_counts if c == 0)

    log.info("")
    log.info("=" * 50)
    log.info(f"Frames processed : {len(frames)}")
    log.info(f"Crops saved      : {total_saved}")
    log.info(f"Crops skipped    : {total_skipped}  (too small, < {min_px}px)")
    log.info(f"Avg detections   : {avg_per_frame:.1f} per frame")
    log.info(f"Zero-detect frames: {zero_detect}  (missed detections — review these)")
    log.info(f"Output dir       : {out_dir}")
    log.info("=" * 50)
    log.info("Next step: upload data/crops/ to CVAT for keypoint labeling")

    return {
        "frames_processed": len(frames),
        "crops_saved":       total_saved,
        "crops_skipped":     total_skipped,
        "zero_detect_frames": zero_detect,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract robot crops using trained YOLO detector")
    p.add_argument("--weights",  required=True,
                   help="Path to best.pt from YOLO training")
    p.add_argument("--frames",   default="data/frames",
                   help="Directory of extracted video frames (default: data/frames)")
    p.add_argument("--out",      default="data/crops",
                   help="Output directory for crops (default: data/crops)")
    p.add_argument("--conf",     type=float, default=DEFAULT_CONF,
                   help=f"Detection confidence threshold (default: {DEFAULT_CONF})")
    p.add_argument("--pad",      type=float, default=DEFAULT_PAD,
                   help=f"Padding around crop as fraction of box size (default: {DEFAULT_PAD})")
    p.add_argument("--min-px",   type=int,   default=DEFAULT_MIN_PX,
                   help=f"Minimum crop dimension in pixels (default: {DEFAULT_MIN_PX})")
    p.add_argument("--resize",   type=int,   default=DEFAULT_RESIZE,
                   help=f"Resize crops to this square size, 0 to disable (default: {DEFAULT_RESIZE})")
    p.add_argument("--device",   type=str,   default="",
                   help="Device: 0 for GPU, 'cpu', 'mps' for Apple Silicon (default: auto)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_crops(
        weights    = args.weights,
        frames_dir = args.frames,
        out_dir    = args.out,
        conf       = args.conf,
        pad        = args.pad,
        min_px     = args.min_px,
        resize     = args.resize,
        device     = args.device,
    )