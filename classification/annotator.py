"""
FRC Robot Detector — ByteTrack + Alliance Colour + Path Recording
=================================================================
Tracks up to 6 robots using Ultralytics' native ByteTrack, classifies each
robot as red or blue alliance from bumper colour, records the full path of
every robot, and exports paths to JSON for downstream analysis.

Usage:
    pip install ultralytics opencv-python tqdm scipy
    python detect_robots_persistent.py --input match.mp4 --weights best.pt

Key flags:
    --max-robots   6           hard cap — only first N stable tracks are registered
    --min-frames   8           frames a track must be seen before entering registry
    --teams        "0:254,1:1678,2:2601,3:4,4:5,5:6"   map track IDs to team numbers
    --predict      40          motion arrow length in pixels
    --pad          18          pixels to expand bounding box outward on all sides
    --trail        30          how many recent path points to draw on screen
    --export-json  paths.json  save registry + paths to JSON after processing
"""

import argparse
import json
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
from ultralytics.utils.plotting import Annotator


# ─────────────────────────────────────────────────────────────
# Palette — indexed by registry slot (0–5), never changes
# ─────────────────────────────────────────────────────────────

PALETTE = [
    (  0, 200, 255),   # slot 0
    ( 80, 255,  80),   # slot 1
    (200,  80, 255),   # slot 2
    ( 80, 255, 200),   # slot 3
    (255, 200,  80),   # slot 4
    (200, 255,  80),   # slot 5
]

# Alliance override colours — used for box border when alliance is known
ALLIANCE_COLOR = {
    "red":  (60,  60,  255),
    "blue": (255, 120,  40),
}

def _slot_color(slot: int) -> tuple:
    return PALETTE[slot % len(PALETTE)]


# ─────────────────────────────────────────────────────────────
# Bumper / alliance colour detection
# ─────────────────────────────────────────────────────────────

def detect_alliance(frame, box) -> str:
    """
    Crops the bottom 30% of the bounding box (where bumpers live) and
    checks whether the dominant hue is red or blue in HSV.
    Returns 'red', 'blue', or 'unknown'.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(fw, x2)
    y1, y2 = max(0, y1), min(fh, y2)

    bh = y2 - y1
    # Take only the bottom 30% — bumpers are at the base of the robot
    crop_y1 = max(y1, y2 - int(bh * 0.30))
    crop = frame[crop_y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"

    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h    = hsv[:, :, 0]
    s    = hsv[:, :, 1]

    # Only consider sufficiently saturated pixels
    mask = s > 60

    red_mask  = mask & ((h < 15) | (h > 160))
    blue_mask = mask & ((h > 95) & (h < 135))

    red_px  = int(red_mask.sum())
    blue_px = int(blue_mask.sum())

    if red_px == 0 and blue_px == 0:
        return "unknown"
    if red_px > blue_px * 1.5:
        return "red"
    if blue_px > red_px * 1.5:
        return "blue"
    return "unknown"


# ─────────────────────────────────────────────────────────────
# Robot Registry
# ─────────────────────────────────────────────────────────────

class RobotEntry:
    def __init__(self, track_id: int, slot: int, class_name: str):
        self.track_id    = track_id
        self.slot        = slot          # 0–5, permanent display slot
        self.class_name  = class_name
        self.frames_seen = 0
        self.frames_lost = 0
        self.status      = "ACTIVE"      # ACTIVE | LOST
        self.alliance    = "unknown"     # red | blue | unknown
        self.path        = []            # list of (cx, cy) confirmed centroids
        self.last_box    = None


class RobotRegistry:
    """
    Stable registry of up to max_robots robots.

    ByteTrack handles identity — the registry's job is:
      - Only admit a track once it has been seen for min_frames (stability gate)
      - Cap at max_robots total entries, ever
      - Record the full path and alliance colour per robot
      - Export to JSON
    """

    def __init__(self, max_robots: int = 6, min_frames: int = 8):
        self.max_robots  = max_robots
        self.min_frames  = min_frames
        self.entries: dict[int, RobotEntry] = {}   # track_id → entry
        self._pending: dict[int, int] = {}         # track_id → frames seen while pending
        self._next_slot  = 0
        self.video_width  = 0
        self.video_height = 0

    # ── called once per frame ────────────────────────────────

    def update(self, detections: list[dict], frame):
        """
        detections: list of {track_id, box, class_name, conf}
        frame:      current BGR frame (for alliance detection)
        """
        live_ids = {d["track_id"] for d in detections}

        for det in detections:
            tid  = det["track_id"]
            box  = det["box"]
            cx   = (box[0] + box[2]) // 2
            cy   = (box[1] + box[3]) // 2

            if tid in self.entries:
                # Known robot — update
                e = self.entries[tid]
                e.frames_seen += 1
                e.frames_lost  = 0
                e.status       = "ACTIVE"
                e.last_box     = box
                e.path.append((cx, cy))
                # Re-sample alliance every 10 frames (bumpers can be occluded)
                if e.frames_seen % 10 == 0 or e.alliance == "unknown":
                    detected = detect_alliance(frame, box)
                    if detected != "unknown":
                        e.alliance = detected

            else:
                # Pending — count stability frames
                self._pending[tid] = self._pending.get(tid, 0) + 1
                if (self._pending[tid] >= self.min_frames
                        and self._next_slot < self.max_robots):
                    # Graduate into registry
                    slot = self._next_slot
                    self._next_slot += 1
                    e = RobotEntry(tid, slot, det["class_name"])
                    e.frames_seen = self._pending.pop(tid)
                    e.last_box    = box
                    e.alliance    = detect_alliance(frame, box)
                    e.path.append((cx, cy))
                    self.entries[tid] = e

        # Mark entries not seen this frame as LOST
        for tid, e in self.entries.items():
            if tid not in live_ids:
                e.frames_lost += 1
                e.status = "LOST"

        # Prune pending tracks that vanished before graduating
        stale = [tid for tid in self._pending if tid not in live_ids]
        for tid in stale:
            del self._pending[tid]

    def ordered(self) -> list[RobotEntry]:
        return sorted(self.entries.values(), key=lambda e: e.slot)

    # ── JSON export ──────────────────────────────────────────

    def save_json(self, path: Path, team_map: dict):
        data = {
            "video_info": {
                "width":  self.video_width,
                "height": self.video_height,
            },
            "robots": [
                {
                    "track_id":   e.track_id,
                    "slot":       e.slot,
                    "team":       team_map.get(e.slot, team_map.get(e.track_id, None)),
                    "alliance":   e.alliance,
                    "class_name": e.class_name,
                    "frames_seen": e.frames_seen,
                    "path":       e.path,
                }
                for e in self.ordered()
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Paths saved → {path}")


# ─────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────

def _pad_box(box, pad, fw, fh):
    return [
        max(0, box[0]-pad), max(0, box[1]-pad),
        min(fw, box[2]+pad), min(fh, box[3]+pad),
    ]


def _velocity_from_path(path: list, n: int = 8) -> tuple:
    """Average velocity over the last n path points."""
    if len(path) < 2:
        return (0.0, 0.0)
    recent = path[-min(n, len(path)):]
    dx = recent[-1][0] - recent[0][0]
    dy = recent[-1][1] - recent[0][1]
    frames = len(recent) - 1
    return (dx / frames, dy / frames)


def draw_robot(frame, entry: RobotEntry, team_map: dict,
               predict_px: int, show_labels: bool,
               trail_len: int, pad: int):
    """Draw box, path trail, and motion vector for one robot."""
    fh, fw = frame.shape[:2]
    color  = _slot_color(entry.slot)

    # ── Bounding box via Annotator ───────────────────────────
    if entry.last_box is not None:
        pbox = _pad_box(entry.last_box, pad, fw, fh)

        team  = team_map.get(entry.slot, team_map.get(entry.track_id))
        if show_labels:
            if team:
                label = f"Team {team}"
            else:
                label = f"#{entry.slot} {entry.class_name}"
            if entry.alliance != "unknown":
                label += f" ({'R' if entry.alliance == 'red' else 'B'})"
        else:
            label = ""

        ann = Annotator(frame, line_width=4, font_size=16)
        ann.box_label(pbox, label=label, color=color, txt_color=(0, 0, 0))

        # Alliance-coloured outer ring
        alliance_col = ALLIANCE_COLOR.get(entry.alliance, (200, 200, 200))
        x1,y1,x2,y2 = pbox
        cv2.rectangle(frame, (x1-3,y1-3), (x2+3,y2+3), alliance_col, 2, cv2.LINE_AA)

    # ── Path trail ───────────────────────────────────────────
    if len(entry.path) >= 2:
        recent = entry.path[-trail_len:]
        for i in range(1, len(recent)):
            # Fade older segments
            alpha = i / len(recent)
            seg_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, recent[i-1], recent[i], seg_color, 2, cv2.LINE_AA)

        # Dot at current position
        cv2.circle(frame, recent[-1], 5, (255,255,255), -1)
        cv2.circle(frame, recent[-1], 4, color, -1)

    # ── Motion vector ────────────────────────────────────────
    if len(entry.path) >= 2:
        vx, vy = _velocity_from_path(entry.path)
        speed  = (vx**2 + vy**2) ** 0.5
        if speed > 0.5:
            cx, cy = entry.path[-1]
            ex = int(cx + vx/speed * predict_px)
            ey = int(cy + vy/speed * predict_px)
            cv2.arrowedLine(frame, (cx,cy), (ex,ey), (255,255,255),
                            5, tipLength=0.30, line_type=cv2.LINE_AA)
            cv2.arrowedLine(frame, (cx,cy), (ex,ey), color,
                            3, tipLength=0.30, line_type=cv2.LINE_AA)


def draw_hud(frame, frame_idx, total_frames, proc_fps, n_active):
    lines = [
        f"Frame {frame_idx}/{total_frames}",
        f"Active: {n_active}",
        f"{proc_fps:.1f} fps",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 1, cv2.LINE_AA)
        y += 22


def draw_log_panel(frame, registry: RobotRegistry, team_map: dict):
    """Bottom-right panel — one row per registered robot."""
    entries = registry.ordered()
    if not entries:
        return

    fh, fw   = frame.shape[:2]
    font     = cv2.FONT_HERSHEY_SIMPLEX
    row_h    = 38
    swatch_w = 16
    margin   = 10
    fs       = 0.50

    sample      = "Team 9999  RED  seen:99999f"
    (tw, th), _ = cv2.getTextSize(sample, font, fs, 1)
    panel_w     = swatch_w + margin*3 + tw + 8
    panel_h     = len(entries)*row_h + margin*2 + 22

    px1 = fw - panel_w - 10
    py1 = fh - panel_h - 10
    px2 = fw - 10
    py2 = fh - 10

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1,py1), (px2,py2), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (px1,py1), (px2,py2), (70,70,70), 1)

    # Header
    cv2.putText(frame, "ROBOT REGISTRY", (px1+margin, py1+margin+13),
                font, 0.48, (180,180,180), 1, cv2.LINE_AA)

    for e in entries:
        ry    = py1 + margin + 22 + e.slot * row_h
        color = _slot_color(e.slot)

        # Alliance-tinted swatch border
        alliance_col = ALLIANCE_COLOR.get(e.alliance, (80, 80, 80))
        cv2.rectangle(frame,
                      (px1+margin-2, ry+3),
                      (px1+margin+swatch_w+2, ry+row_h-5),
                      alliance_col, 2)
        cv2.rectangle(frame,
                      (px1+margin, ry+5),
                      (px1+margin+swatch_w, ry+row_h-7),
                      color, -1)

        # Team number or slot ID
        team = team_map.get(e.slot, team_map.get(e.track_id))
        top_label = f"Team {team}" if team else f"Slot #{e.slot}"
        cv2.putText(frame, top_label,
                    (px1+margin+swatch_w+6, ry+th+3),
                    font, fs, (230,230,230), 1, cv2.LINE_AA)

        # Alliance + status + frames
        alliance_str = e.alliance.upper() if e.alliance != "unknown" else "?"
        status_color = (80,255,80) if e.status == "ACTIVE" else (100,100,100)
        sub = f"{alliance_str}  {e.status}  seen:{e.frames_seen}f"
        cv2.putText(frame, sub,
                    (px1+margin+swatch_w+6, ry+th+18),
                    font, 0.38, status_color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# Team map parser
# ─────────────────────────────────────────────────────────────

def parse_team_map(teams_str: str) -> dict:
    """
    Parse "0:254,1:1678,2:2601" into {0: 254, 1: 1678, 2: 2601}.
    Keys are registry slots (0-5). Values are FRC team numbers.
    """
    if not teams_str:
        return {}
    result = {}
    for part in teams_str.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            result[int(k.strip())] = int(v.strip())
        except ValueError:
            pass
    return result


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
    predict_px:   int   = 40,
    pad:          int   = 18,
    trail_len:    int   = 30,
    max_robots:   int   = 6,
    min_frames:   int   = 8,
    team_map:     dict  = None,
    export_json:  Path  = None,
):
    team_map = team_map or {}

    print(f"Loading model: {weights_path}")
    model    = YOLO(str(weights_path))
    registry = RobotRegistry(max_robots=max_robots, min_frames=min_frames)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    registry.video_width  = w
    registry.video_height = h
    print(f"Video: {w}x{h}  {src_fps:.2f} fps  {total_frames} frames")
    if team_map:
        print(f"Teams: {team_map}")

    CROP_TOP    = int(h * 0.10)   # cut scoreboard
    CROP_BOTTOM = int(h * 0.79)   # cut ground-level strip — was 0.88, too low
    crop_h = CROP_BOTTOM - CROP_TOP
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, crop_h)
    )

    frame_idx = 0
    t_start   = time.perf_counter()

    with tqdm(total=total_frames, unit="frame", desc="Tracking") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Crop to overhead field view only
            frame = frame[CROP_TOP:CROP_BOTTOM, :]

            # ── ByteTrack inference ──────────────────────────
            # persist=True is critical — tells Ultralytics to keep
            # the ByteTrack state alive between calls on this model instance.
            results = model.track(
                source=frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            result = results[0]

            # ── Extract detections with ByteTrack IDs ────────
            detections = []
            if (result.boxes is not None
                    and result.boxes.id is not None
                    and len(result.boxes)):
                xyxy     = result.boxes.xyxy.cpu().numpy()
                cls_ids  = result.boxes.cls.cpu().numpy().astype(int)
                confs    = result.boxes.conf.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    cid = int(cls_ids[i])
                    detections.append({
                        "track_id":   int(track_ids[i]),
                        "box":        list(map(int, xyxy[i])),
                        "class_id":   cid,
                        "class_name": result.names.get(cid, f"cls{cid}"),
                        "conf":       float(confs[i]),
                    })

            # ── Update registry ──────────────────────────────
            registry.update(detections, frame)

            # ── Draw all registered robots ───────────────────
            for entry in registry.ordered():
                if entry.status == "ACTIVE":
                    draw_robot(frame, entry, team_map,
                               predict_px=predict_px,
                               show_labels=show_labels,
                               trail_len=trail_len,
                               pad=pad)

            elapsed  = time.perf_counter() - t_start
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            n_active = sum(1 for e in registry.ordered() if e.status == "ACTIVE")
            draw_hud(frame, frame_idx, total_frames, proc_fps, n_active)
            draw_log_panel(frame, registry, team_map)

            writer.write(frame)

            if show:
                cv2.imshow("FRC Robot Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            pbar.update(1)

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print(f"\nDone -- {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"\n-- Final Robot Registry --")
    for e in registry.ordered():
        team = team_map.get(e.slot, team_map.get(e.track_id, "?"))
        print(f"  Slot {e.slot}  Team {team:<6}  {e.alliance:<7}"
              f"  seen:{e.frames_seen:>5}f  path pts:{len(e.path)}")
    print(f"\nOutput video: {output_path}")

    if export_json:
        registry.save_json(export_json, team_map)

    return registry


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="FRC Robot Tracker — ByteTrack + Alliance Colour + Path Recording"
    )
    p.add_argument("--input",        required=True,
                   help="Path to input .mp4")
    p.add_argument("--weights",      required=True,
                   help="Path to best.pt")
    p.add_argument("--output",       default="out_tracked.mp4",
                   help="Output annotated video")
    p.add_argument("--conf",         type=float, default=0.4)
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--imgsz",        type=int,   default=640)
    p.add_argument("--device",       type=str,   default="0",
                   help="GPU id or 'cpu'")
    p.add_argument("--show",         action="store_true",
                   help="Live preview window")
    p.add_argument("--no-labels",    action="store_true",
                   help="Hide labels on bounding boxes")
    p.add_argument("--predict",      type=int,   default=40,
                   help="Motion arrow length in pixels")
    p.add_argument("--pad",          type=int,   default=18,
                   help="Pixels to expand each bounding box outward")
    p.add_argument("--trail",        type=int,   default=30,
                   help="Number of recent path points drawn as a trail on screen")
    p.add_argument("--max-robots",   type=int,   default=6,
                   help="Max robots to track (default 6 for a full match)")
    p.add_argument("--min-frames",   type=int,   default=8,
                   help="Frames a track must be stable before entering registry")
    p.add_argument("--teams",        type=str,   default="",
                   help='Map registry slots to team numbers: "0:254,1:1678,2:2601"')
    p.add_argument("--export-json",  type=str,   default="",
                   help="Path to save robot paths as JSON (e.g. paths.json)")
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
        predict_px   = args.predict,
        pad          = args.pad,
        trail_len    = args.trail,
        max_robots   = args.max_robots,
        min_frames   = args.min_frames,
        team_map     = parse_team_map(args.teams),
        export_json  = Path(args.export_json) if args.export_json else None,
    )

if __name__ == "__main__":
    main()