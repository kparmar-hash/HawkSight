#TODO So the robot registry gets to 14 robots in total, however the robots keep swapping when they pass each other,
# Is this just a training issue or could we help this with something in the code? Shorten robot registry to 6


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

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
from ultralytics.utils.plotting import Annotator

# pyrefly: ignore [missing-import]
from scipy.optimize import linear_sum_assignment


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


# -------------------------------------------------------------
# Histogram
# -------------------------------------------------------------

def color_histogram(frame, box, bins=16) -> np.ndarray:
    """Small HSV histogram of the cropped box region."""
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(fw, box[2]), min(fh, box[3])
    if x2 <= x1 or y2 <= y1:
        return np.zeros(bins * 2, dtype=np.float32)
    crop = frame[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist   = np.concatenate([h_hist, s_hist])
    norm   = hist.sum()
    return hist / norm if norm > 0 else hist

def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0 = identical, 1 = completely different."""
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))

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


def dedupe_multiview_detections(
    detections: list[dict],
    frame_width: int,
    frame_height: int,
    view_count: int,
    match_threshold: float,
) -> list[dict]:
    """
    Merge duplicate detections of the same robot across stitched camera views.

    Assumes the full frame is split into `view_count` equal-width panes showing
    the same field. Detections from different panes that land at nearly the same
    normalized in-pane position are treated as the same physical robot.
    """
    if view_count <= 1 or len(detections) <= 1:
        return detections

    pane_width = frame_width / view_count
    if pane_width <= 0 or frame_height <= 0:
        return detections

    enriched = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        view_idx = int(cx / pane_width)
        view_idx = max(0, min(view_count - 1, view_idx))
        local_x = (cx - (view_idx * pane_width)) / pane_width
        local_y = cy / frame_height
        enriched.append({
            **det,
            "view_idx": view_idx,
            "norm_pos": (local_x, local_y),
        })

    groups: list[dict] = []
    for det in sorted(enriched, key=lambda d: d.get("confidence", 0.0), reverse=True):
        placed = False
        for group in groups:
            if det["class_id"] != group["class_id"]:
                continue
            if det["view_idx"] in group["views"]:
                continue

            gx, gy = group["norm_pos"]
            dx = det["norm_pos"][0] - gx
            dy = det["norm_pos"][1] - gy
            if (dx * dx + dy * dy) ** 0.5 > match_threshold:
                continue

            group["members"].append(det)
            group["views"].add(det["view_idx"])
            count = len(group["members"])
            group["norm_pos"] = (
                ((gx * (count - 1)) + det["norm_pos"][0]) / count,
                ((gy * (count - 1)) + det["norm_pos"][1]) / count,
            )
            if det.get("confidence", 0.0) > group["best"].get("confidence", 0.0):
                group["best"] = det
            placed = True
            break

        if not placed:
            groups.append({
                "class_id": det["class_id"],
                "members": [det],
                "views": {det["view_idx"]},
                "norm_pos": det["norm_pos"],
                "best": det,
            })

    merged = []
    for group in groups:
        best = dict(group["best"])
        best["duplicate_views"] = sorted(group["views"])
        best["duplicate_count"] = len(group["members"])
        best.pop("view_idx", None)
        best.pop("norm_pos", None)
        merged.append(best)
    return merged


def filter_layout_detections(
    detections: list[dict],
    frame_width: int,
    frame_height: int,
    layout_mode: str,
    main_view_bottom_ratio: float,
) -> list[dict]:
    """
    Mark which detections may create brand-new robot IDs for known layouts.

    `top-main-bottom-sides` matches the stream layout with one large full-field
    view on top and two smaller inserts along the bottom. In that layout, the
    top pane is the authoritative source for spawning registry IDs, but the
    auxiliary views may still refresh already-known robots.
    """
    tagged = [dict(det, can_spawn=True) for det in detections]

    if layout_mode == "full-frame":
        return tagged

    if layout_mode == "top-main-bottom-sides":
        cutoff_y = int(frame_height * main_view_bottom_ratio)
        for det in tagged:
            center_y = (det["box"][1] + det["box"][3]) / 2.0
            det["can_spawn"] = center_y <= cutoff_y
        return tagged

    return tagged


# ─────────────────────────────────────────────────────────────
# Track
# ─────────────────────────────────────────────────────────────

class Track:
    _next_id = 0

    def __init__(self, box, class_id, class_name, trail_len, frame=None):
        self.id          = Track._next_id
        Track._next_id  += 1
        self.class_id    = class_id
        self.class_name  = class_name
        self.box         = box
        self.trail_len   = trail_len
        self.centroids   = [self._centroid(box)]
        self.lost        = 0
        self.pred_box    = list(box)
        self.pred_cx, self.pred_cy = self._centroid(box)
        # Appearance: rolling average of last N histograms
        self.appearance: np.ndarray | None = (
            color_histogram(frame, box) if frame is not None else None
        )


    def update_appearance(self, frame, box, alpha=0.3):
        """Exponential moving average so appearance adapts slowly."""
        new_hist = color_histogram(frame, box)
        if self.appearance is None:
            self.appearance = new_hist
        else:
            self.appearance = (1 - alpha) * self.appearance + alpha * new_hist

    

    @staticmethod
    def _centroid(box) -> tuple:
        return ((box[0]+box[2])//2, (box[1]+box[3])//2)

    def velocity(self) -> tuple:
        if len(self.centroids) < 2:
            return (0.0, 0.0)
        dx = self.centroids[-1][0] - self.centroids[0][0]
        dy = self.centroids[-1][1] - self.centroids[0][1]
        return (dx / (len(self.centroids)-1), dy / (len(self.centroids)-1))

    def confirm(self, box, frame=None):
        self.box  = box
        self.lost = 0
        c = self._centroid(box)
        self.centroids.append(c)
        if len(self.centroids) > self.trail_len:
            self.centroids.pop(0)
        self.pred_box = list(box)
        self.pred_cx, self.pred_cy = c
        if frame is not None:
            self.update_appearance(frame, box)

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
    Pass 3 — if roster < max_tracks: spawn new track from spawn-eligible views.
              if roster full: force onto nearest ghost (no distance limit).
              This guarantees IDs never exceed max_tracks after the roster
              fills, so robots reappearing from the trench or behind a
              structure always recover their original ID.
    """

    def __init__(self, iou_threshold=0.25, ghost_dist_px=120,
                 max_lost=150, trail_len=8, max_tracks=6,
                 appearance_weight=0.35):
        self.iou_threshold    = iou_threshold
        self.ghost_dist_px    = ghost_dist_px
        self.max_lost         = max_lost
        self.trail_len        = trail_len
        self.max_tracks       = max_tracks
        self.appearance_weight = appearance_weight  # 0 = pure geometry, 1 = pure appearance
        self.tracks: list[Track] = []

    def _centroid(self, box):
        return ((box[0]+box[2])//2, (box[1]+box[3])//2)

    def _dist(self, box, tx, ty):
        cx, cy = self._centroid(box)
        return ((cx-tx)**2 + (cy-ty)**2) ** 0.5

    def _nearby_tracks(self, tracks: list[Track], distance_px: float = 120.0) -> set[int]:
        crowded_ids: set[int] = set()
        for i, track_a in enumerate(tracks):
            for track_b in tracks[i + 1:]:
                dist = self._dist(track_a.box, track_b.pred_cx, track_b.pred_cy)
                if dist <= distance_px:
                    crowded_ids.add(track_a.id)
                    crowded_ids.add(track_b.id)
        return crowded_ids

    def update(self, detections: list[dict], frame=None) -> list[Track]:
        main_detections = [det for det in detections if det.get("can_spawn", True)]
        aux_detections = [det for det in detections if not det.get("can_spawn", True)]
        unmatched_main = list(range(len(main_detections)))

        # The main view is authoritative for matching and spawning IDs.
        active = [t for t in self.tracks if not t.is_ghost]
        matched_track_ids = set()
        if active and unmatched_main:
            crowded_ids = self._nearby_tracks(active)
            cost = np.ones((len(active), len(main_detections)))
            for ti, t in enumerate(active):
                for di in unmatched_main:
                    iou_score = iou(main_detections[di]["box"], t.box)
                    if iou_score < self.iou_threshold:
                        continue

                    geo_cost = 1.0 - iou_score
                    if t.id in crowded_ids and t.appearance is not None:
                        det_hist = color_histogram(frame, main_detections[di]["box"]) if frame is not None else None
                        app_cost = hist_distance(t.appearance, det_hist) if det_hist is not None else 0.5
                        cost[ti, di] = (1 - self.appearance_weight) * geo_cost + self.appearance_weight * app_cost
                    else:
                        cost[ti, di] = geo_cost

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_det_ids = set()
            for ti, di in zip(row_ind, col_ind):
                if cost[ti, di] < 1.0:
                    active[ti].confirm(main_detections[di]["box"], frame)
                    matched_track_ids.add(active[ti].id)
                    matched_det_ids.add(di)

            unmatched_main = [i for i in unmatched_main if i not in matched_det_ids]

        for t in active:
            if t.id not in matched_track_ids:
                t.extrapolate()

        ghosts = [t for t in self.tracks if t.is_ghost]
        matched_ghost_ids = set()
        for t in ghosts:
            best_d, best_i = self.ghost_dist_px, None
            for i in unmatched_main:
                d = self._dist(main_detections[i]["box"], t.pred_cx, t.pred_cy)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                t.confirm(main_detections[best_i]["box"], frame)
                matched_ghost_ids.add(t.id)
                unmatched_main.remove(best_i)
        for t in ghosts:
            if t.id not in matched_ghost_ids:
                t.extrapolate()

        roster_full = len(self.tracks) >= self.max_tracks
        spawned_or_forced_ids = set()
        for i in unmatched_main:
            det = main_detections[i]
            if not roster_full:
                new_track = Track(det["box"], det["class_id"], det["class_name"],
                                  self.trail_len, frame)
                self.tracks.append(new_track)
                spawned_or_forced_ids.add(new_track.id)
                roster_full = len(self.tracks) >= self.max_tracks
            else:
                all_ghosts = [t for t in self.tracks if t.is_ghost]
                if all_ghosts:
                    nearest = min(
                        all_ghosts,
                        key=lambda t: self._dist(det["box"], t.pred_cx, t.pred_cy),
                    )
                    nearest.confirm(det["box"], frame)
                    spawned_or_forced_ids.add(nearest.id)

        # Auxiliary views can refine existing tracks, but never spawn IDs.
        unmatched_aux = list(range(len(aux_detections)))
        main_claimed_ids = matched_track_ids | matched_ghost_ids | spawned_or_forced_ids

        aux_active = [t for t in self.tracks if not t.is_ghost and t.id not in main_claimed_ids]
        matched_aux_track_ids = set()
        if aux_active and unmatched_aux:
            cost = np.ones((len(aux_active), len(aux_detections)))
            for ti, t in enumerate(aux_active):
                for di in unmatched_aux:
                    iou_score = iou(aux_detections[di]["box"], t.box)
                    if iou_score >= self.iou_threshold:
                        cost[ti, di] = 1.0 - iou_score

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_aux_det_ids = set()
            for ti, di in zip(row_ind, col_ind):
                if cost[ti, di] < 1.0:
                    aux_active[ti].confirm(aux_detections[di]["box"], frame)
                    matched_aux_track_ids.add(aux_active[ti].id)
                    matched_aux_det_ids.add(di)
            unmatched_aux = [i for i in unmatched_aux if i not in matched_aux_det_ids]

        aux_ghosts = [
            t for t in self.tracks
            if t.is_ghost and t.id not in main_claimed_ids and t.id not in matched_aux_track_ids
        ]
        for t in aux_ghosts:
            best_d, best_i = self.ghost_dist_px, None
            for i in unmatched_aux:
                d = self._dist(aux_detections[i]["box"], t.pred_cx, t.pred_cy)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                t.confirm(aux_detections[best_i]["box"], frame)
                unmatched_aux.remove(best_i)

        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        return self.tracks

        unmatched = list(range(len(detections)))

        # Pass 1 — Hungarian IoU match to active tracks
        active = [t for t in self.tracks if not t.is_ghost]
        if active and unmatched:
            crowded_ids = self._nearby_tracks(active)

            # Build combined cost matrix (rows=tracks, cols=detections)
            cost = np.ones((len(active), len(detections)))
            for ti, t in enumerate(active):
                for di in unmatched:
                    iou_score = iou(detections[di]["box"], t.box)
                    if iou_score < self.iou_threshold:
                        continue  # leave at 1.0 = infeasible

                    geo_cost = 1.0 - iou_score

                    # Blend in appearance cost when tracks are crowded
                    if t.id in crowded_ids and t.appearance is not None:
                        det_hist = color_histogram(frame, detections[di]["box"]) if frame is not None else None
                        app_cost = hist_distance(t.appearance, det_hist) if det_hist is not None else 0.5
                        cost[ti, di] = (1 - self.appearance_weight) * geo_cost + self.appearance_weight * app_cost
                    else:
                        cost[ti, di] = geo_cost

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_track_ids = set()
            matched_det_ids   = set()
            for ti, di in zip(row_ind, col_ind):
                if cost[ti, di] < 1.0:  # feasible assignment
                    active[ti].confirm(detections[di]["box"], frame)
                    matched_track_ids.add(active[ti].id)
                    matched_det_ids.add(di)

            unmatched = [i for i in unmatched if i not in matched_det_ids]
            for t in active:
                if t.id not in matched_track_ids:
                    t.extrapolate()

        elif active:
            for t in active:
                t.extrapolate()

        # Pass 2 — distance match to ghost tracks (unchanged)
        ghosts = [t for t in self.tracks if t.is_ghost]
        matched_g = set()
        for t in ghosts:
            best_d, best_i = self.ghost_dist_px, None
            for i in unmatched:
                d = self._dist(detections[i]["box"], t.pred_cx, t.pred_cy)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                t.confirm(detections[best_i]["box"], frame)
                matched_g.add(t.id)
                unmatched.remove(best_i)
        for t in ghosts:
            if t.id not in matched_g:
                t.extrapolate()

        # Pass 3 — spawn or force onto nearest ghost (unchanged logic)
        roster_full = len(self.tracks) >= self.max_tracks
        for i in unmatched:
            det = detections[i]
            if not roster_full and det.get("can_spawn", True):
                self.tracks.append(
                    Track(det["box"], det["class_id"], det["class_name"],
                          self.trail_len, frame)
                )
                roster_full = len(self.tracks) >= self.max_tracks
            else:
                all_ghosts = [t for t in self.tracks if t.is_ghost]
                if all_ghosts:
                    nearest = min(all_ghosts,
                                  key=lambda t: self._dist(det["box"], t.pred_cx, t.pred_cy))
                    nearest.confirm(det["box"], frame)

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

    def __init__(self, max_robots=6, min_frames_to_register=5):
        self.entries: dict[int, RobotEntry] = {}
        self.pending: dict[int, RobotEntry] = {}  # seen but not yet confirmed
        self.max_robots = max_robots
        self.min_frames = min_frames_to_register
        self.video_width = 0
        self.video_height = 0

    def update(self, tracks: list):
        live_ids = {t.id for t in tracks}

        for t in tracks:
            target = self.entries if t.id in self.entries else self.pending
            if t.id not in target:
                target[t.id] = RobotEntry(t.id, t.class_name)

            e = target[t.id]
            if not t.is_ghost:
                e.frames_seen += 1
                e.frames_lost  = 0
                e.status       = "ACTIVE"
                e.path.append((int(t.pred_cx), int(t.pred_cy)))
            else:
                e.frames_lost = t.lost
                e.status      = "GHOST"

            # Graduate from pending to registry once stable and under cap
            if t.id in self.pending and e.frames_seen >= self.min_frames:
                if len(self.entries) < self.max_robots:
                    self.entries[t.id] = self.pending.pop(t.id)

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
    layout_mode:  str   = "full-frame",
    main_view_bottom_ratio: float = 1.0,
    view_count:   int   = 1,
    cross_view_match: float = 0.08,
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
                        "confidence": float(result.boxes.conf[i]),
                    })

            detections = filter_layout_detections(
                detections=detections,
                frame_width=w,
                frame_height=h,
                layout_mode=layout_mode,
                main_view_bottom_ratio=main_view_bottom_ratio,
            )

            detections = dedupe_multiview_detections(
                detections=detections,
                frame_width=w,
                frame_height=h,
                view_count=view_count,
                match_threshold=cross_view_match,
            )

            tracks = tracker.update(detections, frame=frame)
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
    p.add_argument("--layout-mode", type=str, default="full-frame",
                   choices=["full-frame", "top-main-bottom-sides"],
                   help="Broadcast layout used to decide which pane owns robot IDs")
    p.add_argument("--main-view-bottom-ratio", type=float, default=1.0,
                   help="Bottom Y ratio of the main top view when using top-main-bottom-sides")
    p.add_argument("--view-count",  type=int,   default=1,
                   help="Number of equal-width stitched views in each frame")
    p.add_argument("--cross-view-match", type=float, default=0.08,
                   help="Normalized distance threshold for merging the same robot across views")
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
        layout_mode  = args.layout_mode,
        main_view_bottom_ratio = args.main_view_bottom_ratio,
        view_count   = args.view_count,
        cross_view_match = args.cross_view_match,
    )

if __name__ == "__main__":
    main()
