"""
FRC Robot Detector — Video Inference
=====================================
Runs a trained YOLOv8 model over every frame of an MP4, draws bounding boxes
and confidence scores, and writes an annotated output video.

Usage:
    pip install ultralytics opencv-python tqdm
    python detect_robots_video.py --input match.mp4 --weights best.pt

Optional flags:
    --output       out.mp4        Where to save the annotated video
    --conf         0.4            Confidence threshold (0–1)
    --iou          0.45           NMS IoU threshold
    --imgsz        640            Inference image size (should match training)
    --device       0              GPU id, or 'cpu'
    --show                        Display live preview while processing
    --no-labels                   Hide class name / confidence text on boxes
    --trace        detections.csv Also write a per-frame CSV of every detection
"""

import argparse
import csv
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────

# One distinct colour per class index (BGR).  Cycles if > 10 classes.
PALETTE = [
    (0, 200, 255),   # yellow-ish
    (255, 80,  80),  # blue
    (80,  255, 80),  # green
    (255, 80,  200), # pink
    (80,  200, 255), # orange
    (200, 80,  255), # purple
    (80,  255, 200), # teal
    (255, 200, 80),  # sky blue
    (200, 255, 80),  # lime
    (80,  80,  255), # red
]

def _color(class_id: int) -> tuple:
    return PALETTE[class_id % len(PALETTE)]


def draw_detections(frame, boxes, show_labels: bool = True) -> int:
    """
    Draw bounding boxes on *frame* in-place.
    boxes — ultralytics Boxes object from result.boxes
    Returns the number of detections drawn.
    """
    if boxes is None or len(boxes) == 0:
        return 0

    xyxy   = boxes.xyxy.cpu().numpy()    # (N, 4) absolute pixel coords
    confs  = boxes.conf.cpu().numpy()    # (N,)
    cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)
    names  = boxes.data  # we'll get names from the model later via result.names

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        conf   = confs[i]
        cls_id = cls_ids[i]
        color  = _color(cls_id)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            label = f"robot {conf:.2f}"   # overwritten below if names available
            # Black background pill for text readability
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return len(xyxy)


def draw_detections_with_names(frame, result, show_labels: bool = True) -> int:
    """Wrapper that also resolves class names from result.names dict."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0

    xyxy    = boxes.xyxy.cpu().numpy()
    confs   = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    names   = result.names  # {int: str}

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        conf   = float(confs[i])
        cls_id = int(cls_ids[i])
        color  = _color(cls_id)
        name   = names.get(cls_id, f"cls{cls_id}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            label = f"{name} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return len(xyxy)


# ─────────────────────────────────────────────────────────────
# Overlay: frame counter + FPS
# ─────────────────────────────────────────────────────────────

def draw_hud(frame, frame_idx: int, total_frames: int, fps_processing: float, n_detections: int):
    h, w = frame.shape[:2]
    lines = [
        f"Frame {frame_idx}/{total_frames}",
        f"Detections: {n_detections}",
        f"Processing: {fps_processing:.1f} fps",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


# ─────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────

def process_video(
    input_path: Path,
    output_path: Path,
    weights_path: Path,
    conf_thresh: float = 0.4,
    iou_thresh: float  = 0.45,
    imgsz: int         = 640,
    device: str        = "0",
    show: bool         = False,
    show_labels: bool  = True,
    trace_path: Path   = None,
):
    # ── Load model ──────────────────────────────────────────
    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    # ── Open video ──────────────────────────────────────────
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video  : {width}×{height}  {src_fps:.2f} fps  {total_frames} frames")

    # ── Output writer ────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Cannot open output writer: {output_path}")

    # ── Optional CSV trace ───────────────────────────────────
    csv_file = csv_writer = None
    if trace_path:
        csv_file = open(trace_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"])

    # ── Per-frame stats ──────────────────────────────────────
    total_detections = 0
    frame_idx        = 0
    t_start          = time.perf_counter()

    with tqdm(total=total_frames, unit="frame", desc="Detecting") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # ── Run inference ────────────────────────────────
            results = model.predict(
                source=frame,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            result = results[0]   # single image → single result

            # ── Draw boxes ───────────────────────────────────
            n = draw_detections_with_names(frame, result, show_labels)
            total_detections += n

            # ── HUD overlay ──────────────────────────────────
            elapsed = time.perf_counter() - t_start
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            draw_hud(frame, frame_idx, total_frames, proc_fps, n)

            # ── Write frame ──────────────────────────────────
            writer.write(frame)

            # ── CSV trace ────────────────────────────────────
            if csv_writer and result.boxes is not None and len(result.boxes):
                xyxy    = result.boxes.xyxy.cpu().numpy()
                confs   = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    cid  = int(cls_ids[i])
                    name = result.names.get(cid, f"cls{cid}")
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    csv_writer.writerow([frame_idx, cid, name, f"{confs[i]:.4f}", x1, y1, x2, y2])

            # ── Live preview (optional) ───────────────────────
            if show:
                cv2.imshow("FRC Robot Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped early by user.")
                    break

            pbar.update(1)

    # ── Cleanup ──────────────────────────────────────────────
    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(f"\n── Summary ───────────────────────────────────────")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Total detections : {total_detections}")
    print(f"  Avg per frame    : {total_detections / max(frame_idx, 1):.2f}")
    print(f"  Wall time        : {elapsed:.1f}s  ({frame_idx/elapsed:.1f} fps)")
    print(f"  Output video     : {output_path}")
    if trace_path:
        print(f"  Detection CSV    : {trace_path}")
    print(f"──────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FRC Robot Detector — Video Inference")
    parser.add_argument("--input",     required=True,         help="Path to input .mp4")
    parser.add_argument("--weights",   required=True,         help="Path to best.pt")
    parser.add_argument("--output",    default="out.mp4",     help="Output annotated video")
    parser.add_argument("--conf",      type=float, default=0.4)
    parser.add_argument("--iou",       type=float, default=0.45)
    parser.add_argument("--imgsz",     type=int,   default=640)
    parser.add_argument("--device",    type=str,   default="0",
                        help="GPU id (e.g. 0) or 'cpu'")
    parser.add_argument("--show",      action="store_true",
                        help="Show live preview window while processing")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide class/confidence text on bounding boxes")
    parser.add_argument("--trace",     default=None,
                        help="Optional: path to write per-frame detections CSV")
    args = parser.parse_args()

    process_video(
        input_path   = Path(args.input),
        output_path  = Path(args.output),
        weights_path = Path(args.weights),
        conf_thresh  = args.conf,
        iou_thresh   = args.iou,
        imgsz        = args.imgsz,
        device       = args.device,
        show         = args.show,
        show_labels  = not args.no_labels,
        trace_path   = Path(args.trace) if args.trace else None,
    )


if __name__ == "__main__":
    main()