"""
FRC Robot Detector — Persistent IDs via Dead-Reckoning
=======================================================
Tracks robots across frames with stable IDs. When a robot disappears,
its track keeps extrapolating forward using last known velocity. When a
detection reappears near the predicted ghost position, it re-associates
with the original ID and colour — no new ID assigned.

Usage:
    pip install ultralytics opencv-python tqdm
    python detect_robots_persistent.py --input match.mp4 --weights best.pt

Key flags:
    --trail        8      frames of centroid history to average velocity over
    --predict      40     arrow length in pixels
    --max-lost     60     frames a ghost track lives before being truly dropped
    --ghost-dist   120    max pixel distance from predicted pos to re-associate
"""

import argparse
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
from ultralytics.utils.plotting import Annotator, colors


# ─────────────────────────────────────────────────────────────
# Palette — fixed per track ID so colour never changes
# ─────────────────────────────────────────────────────────────

PALETTE = [
    (0,   200, 255),
    (255,  80,  80),
    ( 80, 255,  80),
    (255,  80, 200),
    ( 80, 200, 255),
    (200,  80, 255),
    ( 80, 255, 200),
    (255, 200,  80),
    (200, 255,  80),
    ( 80,  80, 255),
    (255, 165,   0),
    (  0, 128, 255),
]

def _color(track_id: int) -> tuple:
    return PALETTE[track_id % len(PALETTE)]


# ─────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────

def iou(a, b) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (areaA + areaB - inter)


# ─────────────────────────────────────────────────────────────
# Track
# ─────────────────────────────────────────────────────────────

class Track:
    _next_id = 0

    def __init__(self, box: list, class_id: int, class_name: str, trail_len: int):
        self.id         = Track._next_id
        Track._next_id += 1
        self.class_id   = class_id
        self.class_name = class_name
        self.box        = box                    # last REAL box [x1,y1,x2,y2]
        self.trail_len  = trail_len
        self.centroids  = [self._centroid(box)]  # confirmed centroid history
        self.lost       = 0                      # frames since last real match
        # Predicted (ghost) state — updated every frame regardless of match
        self.pred_box   = list(box)
        self.pred_cx, self.pred_cy = self._centroid(box)

    # ── helpers ─────────────────────────────────────────────

    @staticmethod
    def _centroid(box) -> tuple:
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    def velocity(self) -> tuple:
        if len(self.centroids) < 2:
            return (0.0, 0.0)
        dx = self.centroids[-1][0] - self.centroids[0][0]
        dy = self.centroids[-1][1] - self.centroids[0][1]
        n  = len(self.centroids) - 1
        return (dx / n, dy / n)

    # ── called when a real detection matches this track ─────

    def confirm(self, box: list):
        self.box  = box
        self.lost = 0
        c = self._centroid(box)
        self.centroids.append(c)
        if len(self.centroids) > self.trail_len:
            self.centroids.pop(0)
        self.pred_box = list(box)
        self.pred_cx, self.pred_cy = c

    # ── called every frame for lost tracks (dead-reckoning) ─

    def extrapolate(self):
        """Advance the predicted position by one frame using current velocity."""
        self.lost += 1
        vx, vy = self.velocity()
        w = self.pred_box[2] - self.pred_box[0]
        h = self.pred_box[3] - self.pred_box[1]
        self.pred_cx = int(self.pred_cx + vx)
        self.pred_cy = int(self.pred_cy + vy)
        # Keep predicted box centred on predicted centroid
        self.pred_box = [
            self.pred_cx - w // 2,
            self.pred_cy - h // 2,
            self.pred_cx + w // 2,
            self.pred_cy + h // 2,
        ]

    @property
    def is_ghost(self) -> bool:
        return self.lost > 0


# ─────────────────────────────────────────────────────────────
# Persistent Tracker
# ─────────────────────────────────────────────────────────────

class PersistentTracker:
    """
    Two-pass matching each frame:
      Pass 1 — match detections to ACTIVE tracks by IoU (same as before).
      Pass 2 — match remaining detections to GHOST tracks by distance from
               their dead-reckoned predicted centroid.

    A ghost that gets re-matched recovers its original ID and colour.
    Ghosts only die after `max_lost` frames with no match.
    """

    def __init__(
        self,
        iou_threshold:  float = 0.25,
        ghost_dist_px:  int   = 120,
        max_lost:       int   = 60,
        trail_len:      int   = 8,
    ):
        self.iou_threshold = iou_threshold
        self.ghost_dist_px = ghost_dist_px
        self.max_lost      = max_lost
        self.trail_len     = trail_len
        self.tracks: list[Track] = []

    def _centroid(self, box) -> tuple:
        return ((box[0]+box[2])//2, (box[1]+box[3])//2)

    def update(self, detections: list[dict]) -> list[Track]:
        unmatched_dets = list(range(len(detections)))

        # ── Pass 1: active tracks → IoU matching ────────────
        active_tracks = [t for t in self.tracks if not t.is_ghost]
        matched_tracks = set()

        for t in active_tracks:
            best_score, best_di = self.iou_threshold, None
            for di in unmatched_dets:
                score = iou(detections[di]["box"], t.box)
                if score > best_score:
                    best_score, best_di = score, di
            if best_di is not None:
                t.confirm(detections[best_di]["box"])
                matched_tracks.add(t.id)
                unmatched_dets.remove(best_di)

        # Extrapolate active tracks that didn't get a match
        for t in active_tracks:
            if t.id not in matched_tracks:
                t.extrapolate()

        # ── Pass 2: ghost tracks → distance matching ────────
        ghost_tracks = [t for t in self.tracks if t.is_ghost]
        matched_ghosts = set()

        for t in ghost_tracks:
            best_dist, best_di = self.ghost_dist_px, None
            for di in unmatched_dets:
                cx, cy = self._centroid(detections[di]["box"])
                dist   = ((cx - t.pred_cx)**2 + (cy - t.pred_cy)**2) ** 0.5
                if dist < best_dist:
                    best_dist, best_di = dist, di
            if best_di is not None:
                t.confirm(detections[best_di]["box"])
                matched_ghosts.add(t.id)
                unmatched_dets.remove(best_di)

        # Extrapolate ghosts that still didn't get a match
        for t in ghost_tracks:
            if t.id not in matched_ghosts:
                t.extrapolate()

        # ── Pass 3: remaining detections → new tracks ───────
        for di in unmatched_dets:
            det = detections[di]
            self.tracks.append(Track(det["box"], det["class_id"], det["class_name"], self.trail_len))

        # ── Reap truly dead ghosts ───────────────────────────
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]

        return self.tracks


# ─────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────

def _pad_box(box: list, pad: int, frame_w: int, frame_h: int) -> list:
    """Expand box by `pad` px on every side, clamped to frame bounds."""
    return [
        max(0,       box[0] - pad),
        max(0,       box[1] - pad),
        min(frame_w, box[2] + pad),
        min(frame_h, box[3] + pad),
    ]


def _dashed_rect(frame, box, color, dash=14, gap=7, thickness=2):
    x1, y1, x2, y2 = box
    pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
    for i in range(len(pts)-1):
        p1, p2 = pts[i], pts[i+1]
        length = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5
        if length == 0:
            continue
        ux, uy = (p2[0]-p1[0])/length, (p2[1]-p1[1])/length
        d, draw = 0.0, True
        while d < length:
            seg = min(dash if draw else gap, length - d)
            if draw:
                sx = int(p1[0] + ux * d);       sy = int(p1[1] + uy * d)
                ex = int(p1[0] + ux * (d+seg)); ey = int(p1[1] + uy * (d+seg))
                cv2.line(frame, (sx,sy), (ex,ey), color, thickness, cv2.LINE_AA)
            d += seg
            draw = not draw


def draw_track(frame, track: Track, predict_px: int, show_labels: bool,
               pad: int = 18):
    """
    Draw one track onto `frame`.
      pad        — pixels to expand the detected box outward on each side
      predict_px — length of the motion arrow in pixels
    """
    fh, fw = frame.shape[:2]
    color  = _color(track.id)

    if track.is_ghost:
        # ── Ghost: dashed predicted box ─────────────────────
        pbox = _pad_box(track.pred_box, pad, fw, fh)
        _dashed_rect(frame, pbox, color, thickness=2)

        if show_labels:
            label = f"#{track.id} lost:{track.lost}"
            cv2.putText(frame, label, (pbox[0]+2, pbox[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    else:
        # ── Active: Annotator box with thick border + filled label ──
        pbox = _pad_box(track.box, pad, fw, fh)
        label = f"#{track.id} {track.class_name}" if show_labels else ""

        ann = Annotator(frame, line_width=4, font_size=16)
        ann.box_label(pbox, label=label, color=color, txt_color=(0, 0, 0))

        # Extra outer glow ring for visibility
        x1, y1, x2, y2 = pbox
        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (255,255,255), 1, cv2.LINE_AA)

    # ── Motion vector (active + ghost) ──────────────────────
    if len(track.centroids) >= 2:
        vx, vy = track.velocity()
        speed  = (vx**2 + vy**2) ** 0.5
        if speed > 0.5:
            cx, cy = track.pred_cx, track.pred_cy
            ex = int(cx + vx/speed * predict_px)
            ey = int(cy + vy/speed * predict_px)

            # Semi-transparent ghost box at predicted destination
            bw = (track.pred_box[2]-track.pred_box[0]) // 2
            bh = (track.pred_box[3]-track.pred_box[1]) // 2
            overlay = frame.copy()
            cv2.rectangle(overlay, (ex-bw, ey-bh), (ex+bw, ey+bh), color, 2)
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

            # White outline on arrow for contrast against any background
            thickness = 3 if not track.is_ghost else 2
            cv2.arrowedLine(frame, (cx,cy), (ex,ey), (255,255,255),
                            thickness+2, tipLength=0.30, line_type=cv2.LINE_AA)
            cv2.arrowedLine(frame, (cx,cy), (ex,ey), color,
                            thickness, tipLength=0.30, line_type=cv2.LINE_AA)

            # Filled dot at origin
            cv2.circle(frame, (cx,cy), 6, (255,255,255), -1)
            cv2.circle(frame, (cx,cy), 5, color, -1)


def draw_hud(frame, frame_idx, total_frames, proc_fps, active, ghost):
    lines = [
        f"Frame {frame_idx}/{total_frames}",
        f"Active: {active}  Ghost: {ghost}",
        f"{proc_fps:.1f} fps",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        y += 22


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────

def process_video(
    input_path:   Path,
    output_path:  Path,
    weights_path: Path,
    conf_thresh:  float = 0.4,
    iou_thresh:   float = 0.45,
    imgsz:        int   = 640,
    device:       str   = "0",
    show:         bool  = False,
    show_labels:  bool  = True,
    trail_len:    int   = 8,
    predict_px:   int   = 40,
    max_lost:     int   = 60,
    ghost_dist:   int   = 120,
    pad:          int   = 18,
):
    print(f"Loading model: {weights_path}")
    model   = YOLO(str(weights_path))
    tracker = PersistentTracker(
        iou_threshold=0.25,
        ghost_dist_px=ghost_dist,
        max_lost=max_lost,
        trail_len=trail_len,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}×{h}  {src_fps:.2f} fps  {total_frames} frames")

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h)
    )

    frame_idx = 0
    t_start   = time.perf_counter()

    with tqdm(total=total_frames, unit="frame", desc="Tracking") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = model.predict(
                source=frame, conf=conf_thresh, iou=iou_thresh,
                imgsz=imgsz, device=device, verbose=False,
            )
            result = results[0]

            detections = []
            if result.boxes is not None and len(result.boxes):
                xyxy    = result.boxes.xyxy.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    cid = int(cls_ids[i])
                    detections.append({
                        "box":        list(map(int, xyxy[i])),
                        "class_id":   cid,
                        "class_name": result.names.get(cid, f"cls{cid}"),
                    })

            tracks = tracker.update(detections)

            for track in tracks:
                draw_track(frame, track, predict_px=predict_px,
                           show_labels=show_labels, pad=pad)

            elapsed  = time.perf_counter() - t_start
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            active   = sum(1 for t in tracks if not t.is_ghost)
            ghost    = sum(1 for t in tracks if t.is_ghost)
            draw_hud(frame, frame_idx, total_frames, proc_fps, active, ghost)

            writer.write(frame)

            if show:
                cv2.imshow("FRC Persistent Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            pbar.update(1)

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(f"\nDone — {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"Output: {output_path}\n")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--weights",    required=True)
    p.add_argument("--output",     default="out_persistent.mp4")
    p.add_argument("--conf",       type=float, default=0.4)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--imgsz",      type=int,   default=640)
    p.add_argument("--device",     type=str,   default="0")
    p.add_argument("--show",       action="store_true")
    p.add_argument("--no-labels",  action="store_true")
    p.add_argument("--trail",      type=int,   default=8)
    p.add_argument("--predict",    type=int,   default=40)
    p.add_argument("--max-lost",   type=int,   default=60,
                   help="Frames before a ghost track is permanently dropped")
    p.add_argument("--ghost-dist", type=int,   default=120,
                   help="Max pixel distance from predicted position to re-associate")
    p.add_argument("--pad",        type=int,   default=18,
                   help="Pixels to expand each bounding box outward on all sides")
    args = p.parse_args()

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
        trail_len    = args.trail,
        predict_px   = args.predict,
        max_lost     = args.max_lost,
        ghost_dist   = args.ghost_dist,
        pad          = args.pad,
    )

if __name__ == "__main__":
    main()