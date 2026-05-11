"""
FRC Robot Detector — Persistent IDs + Robot Registry
=====================================================
Tracks up to max_robots (default 6) robots with stable IDs across the whole
video. Once the roster is full, no new IDs are ever created — reappearing
robots are forced back onto their original ID. A log panel shows every robot
ever detected with colour, status, and frame count.

Usage:
    pip install ultralytics opencv-python tqdm
    python detect_robots_persistent.py --input match.mp4 --weights best.pt

Key flags:
    --max-robots   6      hard cap on unique IDs (set to number of robots in match)
    --trail        8      frames of history to average velocity over
    --predict      40     motion arrow length in pixels
    --max-lost     150    frames a ghost survives (150 = 5s at 30fps)
    --ghost-dist   120    distance threshold for normal ghost re-association (px)
    --pad          18     pixels to expand each bounding box outward
"""

import argparse
import time
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


# ─────────────────────────────────────────────────────────────
# Palette — indexed by track ID, never changes
# ─────────────────────────────────────────────────────────────

PALETTE = [
    (  0, 200, 255),   # 0 — yellow
    (255,  80,  80),   # 1 — blue
    ( 80, 255,  80),   # 2 — green
    (255,  80, 200),   # 3 — pink
    ( 80, 200, 255),   # 4 — orange
    (200,  80, 255),   # 5 — purple
    ( 80, 255, 200),   # 6 — teal
    (255, 200,  80),   # 7 — sky
    (200, 255,  80),   # 8 — lime
    ( 80,  80, 255),   # 9 — red
]

def _color(track_id: int) -> tuple:
    return PALETTE[track_id % len(PALETTE)]


# ─────────────────────────────────────────────────────────────
# IoU
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
        self.box        = box
        self.trail_len  = trail_len
        self.centroids  = [self._centroid(box)]
        self.lost       = 0
        self.pred_box   = list(box)
        self.pred_cx, self.pred_cy = self._centroid(box)

    @staticmethod
    def _centroid(box) -> tuple:
        return ((box[0]+box[2])//2, (box[1]+box[3])//2)

    def velocity(self) -> tuple:
        if len(self.centroids) < 2:
            return (0.0, 0.0)
        dx = self.centroids[-1][0] - self.centroids[0][0]
        dy = self.centroids[-1][1] - self.centroids[0][1]
        return (dx / (len(self.centroids)-1), dy / (len(self.centroids)-1))

    def confirm(self, box: list):
        self.box  = box
        self.lost = 0
        c = self._centroid(box)
        self.centroids.append(c)
        if len(self.centroids) > self.trail_len:
            self.centroids.pop(0)
        self.pred_box = list(box)
        self.pred_cx, self.pred_cy = c

    def extrapolate(self):
        self.lost += 1
        vx, vy = self.velocity()
        w = self.pred_box[2] - self.pred_box[0]
        h = self.pred_box[3] - self.pred_box[1]
        self.pred_cx = int(self.pred_cx + vx)
        self.pred_cy = int(self.pred_cy + vy)
        self.pred_box = [
            self.pred_cx - w//2, self.pred_cy - h//2,
            self.pred_cx + w//2, self.pred_cy + h//2,
        ]

    @property
    def is_ghost(self) -> bool:
        return self.lost > 0


# ─────────────────────────────────────────────────────────────
# Persistent Tracker — hard roster cap
# ─────────────────────────────────────────────────────────────

class PersistentTracker:
    """
    Three-pass matching per frame.

    Pass 1 — active tracks matched by IoU.
    Pass 2 — ghost tracks matched by distance from dead-reckoned centroid.
    Pass 3 — if roster < max_tracks: spawn new track.
              if roster full: force onto nearest ghost (no distance limit).
              This guarantees IDs never exceed max_tracks after the roster
              fills, so robots reappearing from the trench or behind a
              structure always recover their original ID.
    """

    def __init__(self, iou_threshold=0.25, ghost_dist_px=120,
                 max_lost=150, trail_len=8, max_tracks=6):
        self.iou_threshold = iou_threshold
        self.ghost_dist_px = ghost_dist_px
        self.max_lost      = max_lost
        self.trail_len     = trail_len
        self.max_tracks    = max_tracks
        self.tracks: list[Track] = []

    def _centroid(self, box):
        return ((box[0]+box[2])//2, (box[1]+box[3])//2)

    def _dist(self, box, tx, ty) -> float:
        cx, cy = self._centroid(box)
        return ((cx-tx)**2 + (cy-ty)**2) ** 0.5

    def update(self, detections: list[dict]) -> list[Track]:
        unmatched = list(range(len(detections)))

        # Pass 1 — IoU match to active tracks
        active = [t for t in self.tracks if not t.is_ghost]
        matched = set()
        for t in active:
            best_s, best_i = self.iou_threshold, None
            for i in unmatched:
                s = iou(detections[i]["box"], t.box)
                if s > best_s:
                    best_s, best_i = s, i
            if best_i is not None:
                t.confirm(detections[best_i]["box"])
                matched.add(t.id)
                unmatched.remove(best_i)
        for t in active:
            if t.id not in matched:
                t.extrapolate()

        # Pass 2 — distance match to ghost tracks (with threshold)
        ghosts = [t for t in self.tracks if t.is_ghost]
        matched_g = set()
        for t in ghosts:
            best_d, best_i = self.ghost_dist_px, None
            for i in unmatched:
                d = self._dist(detections[i]["box"], t.pred_cx, t.pred_cy)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                t.confirm(detections[best_i]["box"])
                matched_g.add(t.id)
                unmatched.remove(best_i)
        for t in ghosts:
            if t.id not in matched_g:
                t.extrapolate()

        # Pass 3 — remaining detections
        roster_full = len(self.tracks) >= self.max_tracks
        for i in unmatched:
            det = detections[i]
            if not roster_full:
                self.tracks.append(
                    Track(det["box"], det["class_id"], det["class_name"], self.trail_len)
                )
                roster_full = len(self.tracks) >= self.max_tracks
            else:
                # Roster full — force onto nearest ghost, no threshold.
                # A detection that cleared passes 1+2 must be a robot whose
                # ghost drifted far (e.g. stopped behind a structure).
                all_ghosts = [t for t in self.tracks if t.is_ghost]
                if all_ghosts:
                    nearest = min(all_ghosts,
                                  key=lambda t: self._dist(det["box"], t.pred_cx, t.pred_cy))
                    nearest.confirm(det["box"])

        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        return self.tracks


# ─────────────────────────────────────────────────────────────
# Robot Registry
# ─────────────────────────────────────────────────────────────

class RobotEntry:
    def __init__(self, track_id: int, class_name: str):
        self.track_id    = track_id
        self.class_name  = class_name
        self.frames_seen = 0
        self.frames_lost = 0
        self.status      = "ACTIVE"   # ACTIVE | GHOST | LOST
        self.path        = []         # List of (x, y) centroids


class RobotRegistry:
    """Permanent ordered log of every robot ever detected this video."""

    def __init__(self):
        self.entries: dict[int, RobotEntry] = {}
        self.video_width = 0
        self.video_height = 0

    def update(self, tracks: list):
        live_ids = {t.id for t in tracks}
        for t in tracks:
            if t.id not in self.entries:
                self.entries[t.id] = RobotEntry(t.id, t.class_name)
            e = self.entries[t.id]
            if not t.is_ghost:
                e.frames_seen += 1
                e.frames_lost  = 0
                e.status       = "ACTIVE"
                # Record the centroid for path annotation
                cx, cy = t.pred_cx, t.pred_cy
                e.path.append((int(cx), int(cy)))
            else:
                e.frames_lost  = t.lost
                e.status       = "GHOST"
        for tid, e in self.entries.items():
            if tid not in live_ids:
                e.status = "LOST"

    def ordered(self) -> list:
        return sorted(self.entries.values(), key=lambda e: e.track_id)

    def save_json(self, output_path: Path):
        """Save the registry data to a JSON file."""
        data = {
            "video_info": {
                "width": self.video_width,
                "height": self.video_height
            },
            "robots": [
                {
                    "track_id": e.track_id,
                    "class_name": e.class_name,
                    "frames_seen": e.frames_seen,
                    "path": e.path
                }
                for e in self.ordered()
            ]
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Registry saved to {output_path}")


# ─────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────

def _pad_box(box, pad, fw, fh):
    return [max(0,box[0]-pad), max(0,box[1]-pad),
            min(fw,box[2]+pad), min(fh,box[3]+pad)]


def _dashed_rect(frame, box, color, dash=14, gap=7, thickness=2):
    x1, y1, x2, y2 = box
    pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
    for i in range(len(pts)-1):
        p1, p2 = pts[i], pts[i+1]
        length = ((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5
        if length == 0:
            continue
        ux, uy = (p2[0]-p1[0])/length, (p2[1]-p1[1])/length
        d, draw = 0.0, True
        while d < length:
            seg = min(dash if draw else gap, length-d)
            if draw:
                sx,sy = int(p1[0]+ux*d), int(p1[1]+uy*d)
                ex,ey = int(p1[0]+ux*(d+seg)), int(p1[1]+uy*(d+seg))
                cv2.line(frame,(sx,sy),(ex,ey),color,thickness,cv2.LINE_AA)
            d += seg
            draw = not draw


def draw_track(frame, track: Track, predict_px: int, show_labels: bool, pad: int = 18):
    fh, fw = frame.shape[:2]
    color  = _color(track.id)

    if track.is_ghost:
        pbox = _pad_box(track.pred_box, pad, fw, fh)
        _dashed_rect(frame, pbox, color, thickness=2)
        if show_labels:
            cv2.putText(frame, f"#{track.id} lost:{track.lost}f",
                        (pbox[0]+2, pbox[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    else:
        pbox  = _pad_box(track.box, pad, fw, fh)
        label = f"#{track.id} {track.class_name}" if show_labels else ""
        ann   = Annotator(frame, line_width=4, font_size=16)
        ann.box_label(pbox, label=label, color=color, txt_color=(0,0,0))
        x1,y1,x2,y2 = pbox
        cv2.rectangle(frame,(x1-2,y1-2),(x2+2,y2+2),(255,255,255),1,cv2.LINE_AA)

    # Motion vector
    if len(track.centroids) >= 2:
        vx, vy = track.velocity()
        speed  = (vx**2+vy**2)**0.5
        if speed > 0.5:
            cx, cy = track.pred_cx, track.pred_cy
            ex = int(cx + vx/speed * predict_px)
            ey = int(cy + vy/speed * predict_px)
            bw = (track.pred_box[2]-track.pred_box[0])//2
            bh = (track.pred_box[3]-track.pred_box[1])//2
            overlay = frame.copy()
            cv2.rectangle(overlay,(ex-bw,ey-bh),(ex+bw,ey+bh),color,2)
            cv2.addWeighted(overlay,0.28,frame,0.72,0,frame)
            thick = 3 if not track.is_ghost else 2
            cv2.arrowedLine(frame,(cx,cy),(ex,ey),(255,255,255),
                            thick+2,tipLength=0.30,line_type=cv2.LINE_AA)
            cv2.arrowedLine(frame,(cx,cy),(ex,ey),color,
                            thick,tipLength=0.30,line_type=cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),6,(255,255,255),-1)
            cv2.circle(frame,(cx,cy),5,color,-1)


def draw_hud(frame, frame_idx, total_frames, proc_fps, active, ghost):
    lines = [
        f"Frame {frame_idx}/{total_frames}",
        f"Active:{active}  Ghost:{ghost}",
        f"{proc_fps:.1f} fps",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame,line,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(frame,line,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
        y += 22


def draw_log_panel(frame, registry: RobotRegistry):
    """Bottom-right panel showing every robot ever detected."""
    entries = registry.ordered()
    if not entries:
        return

    fh, fw   = frame.shape[:2]
    font     = cv2.FONT_HERSHEY_SIMPLEX
    row_h    = 36
    swatch_w = 16
    margin   = 10
    fs       = 0.50

    sample      = "#0  robot  ACTIVE  seen:9999f"
    (tw, th), _ = cv2.getTextSize(sample, font, fs, 1)
    panel_w     = swatch_w + margin*3 + tw + 8
    panel_h     = len(entries)*row_h + margin*2 + 22

    px1 = fw - panel_w - 10
    py1 = fh - panel_h - 10
    px2 = fw - 10
    py2 = fh - 10

    overlay = frame.copy()
    cv2.rectangle(overlay,(px1,py1),(px2,py2),(15,15,15),-1)
    cv2.addWeighted(overlay,0.75,frame,0.25,0,frame)
    cv2.rectangle(frame,(px1,py1),(px2,py2),(70,70,70),1)

    cv2.putText(frame,"ROBOT REGISTRY",(px1+margin,py1+margin+13),
                font,0.48,(180,180,180),1,cv2.LINE_AA)

    for i, e in enumerate(entries):
        ry    = py1 + margin + 22 + i*row_h
        color = _color(e.track_id)

        cv2.rectangle(frame,
                      (px1+margin, ry+5),
                      (px1+margin+swatch_w, ry+row_h-7),
                      color, -1)

        status_color = (80,255,80)  if e.status == "ACTIVE" \
                  else (80,200,255) if e.status == "GHOST" \
                  else (100,100,100)

        cv2.putText(frame, f"#{e.track_id}  {e.class_name}",
                    (px1+margin+swatch_w+6, ry+th+3),
                    font, fs, (230,230,230), 1, cv2.LINE_AA)

        sub = (f"{e.status} {e.frames_lost}f  seen:{e.frames_seen}f"
               if e.status == "GHOST"
               else f"{e.status}  seen:{e.frames_seen}f")
        cv2.putText(frame, sub,
                    (px1+margin+swatch_w+6, ry+th+18),
                    font, 0.38, status_color, 1, cv2.LINE_AA)


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
    max_lost:     int   = 150,
    ghost_dist:   int   = 120,
    pad:          int   = 18,
    max_robots:   int   = 6,
):
    print(f"Loading model: {weights_path}")
    model    = YOLO(str(weights_path))
    tracker  = PersistentTracker(
        iou_threshold=0.25,
        ghost_dist_px=ghost_dist,
        max_lost=max_lost,
        trail_len=trail_len,
        max_tracks=max_robots,
    )
    registry = RobotRegistry()

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    registry.video_width = w
    registry.video_height = h
    print(f"Video: {w}x{h}  {src_fps:.2f} fps  {total_frames} frames")

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
            registry.update(tracks)

            for track in tracks:
                draw_track(frame, track, predict_px=predict_px,
                           show_labels=show_labels, pad=pad)

            elapsed  = time.perf_counter() - t_start
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            active   = sum(1 for t in tracks if not t.is_ghost)
            ghost    = sum(1 for t in tracks if t.is_ghost)
            draw_hud(frame, frame_idx, total_frames, proc_fps, active, ghost)
            draw_log_panel(frame, registry)

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
    print(f"\nDone -- {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"\n-- Final Robot Registry --")
    for e in registry.ordered():
        print(f"  #{e.track_id}  {e.class_name:<12}  seen:{e.frames_seen:>5}f  {e.status}")
    print(f"Output: {output_path}\n")

    return registry


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True)
    p.add_argument("--weights",     required=True)
    p.add_argument("--output",      default="out_persistent.mp4")
    p.add_argument("--conf",        type=float, default=0.4)
    p.add_argument("--iou",         type=float, default=0.45)
    p.add_argument("--imgsz",       type=int,   default=640)
    p.add_argument("--device",      type=str,   default="0")
    p.add_argument("--show",        action="store_true")
    p.add_argument("--no-labels",   action="store_true")
    p.add_argument("--trail",       type=int,   default=8)
    p.add_argument("--predict",     type=int,   default=40)
    p.add_argument("--max-lost",    type=int,   default=150,
                   help="Frames a ghost survives (150=5s@30fps)")
    p.add_argument("--ghost-dist",  type=int,   default=120,
                   help="Distance threshold (px) for normal ghost re-association")
    p.add_argument("--pad",         type=int,   default=18,
                   help="Pixels to expand each bounding box outward on all sides")
    p.add_argument("--max-robots",  type=int,   default=6,
                   help="Hard cap on unique IDs — match the number of robots on field")
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
        max_robots   = args.max_robots,
    )

if __name__ == "__main__":
    main()