"""
field_mapper.py — Homography + Zone classification for FRC match video
=======================================================================

Workflow
--------
1. Call `calibrate_interactively(field_img_path, video_frame)` once to pick
   point correspondences, then save the result to a JSON file.
2. Load that JSON to build a FieldMapper for every subsequent run.
3. For each detected bounding-box center, call `mapper.classify(px, py)` to
   get "Red" | "Neutral" | "Blue" | "Unknown".
4. Call `mapper.render_minimap(field_img, robot_points)` to get an overlay.

Zone definition
---------------
Zones are defined as lists of (x, y) polygons IN FIELD-IMAGE PIXEL SPACE.
Edit ZONE_POLYGONS at the bottom of this file to match your field image.
Run `python field_mapper.py --define-zones field.png` for an interactive tool.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

ZONE_COLORS_BGR = {
    "Red":     (60,  60,  220),
    "Neutral": (60,  180, 60),
    "Blue":    (220, 80,  60),
    "Unknown": (120, 120, 120),
}

ROBOT_DOT_RADIUS   = 10
MINIMAP_LONG_EDGE  = 320   # pixels; minimap is scaled to fit this


@dataclass
class FieldMapper:
    """
    Wraps the homography matrix and zone polygons.
    All zone polygon coordinates are in *field-image pixel space*.
    """
    H: np.ndarray                          # 3×3 homography (video → field)
    zones: dict[str, np.ndarray]           # name → (N,2) int32 polygon pts
    field_img_shape: tuple[int, int]       # (h, w) of the reference field image

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "FieldMapper":
        data = json.loads(Path(path).read_text())
        H    = np.array(data["H"], dtype=np.float64)
        zones = {
            name: np.array(pts, dtype=np.int32)
            for name, pts in data["zones"].items()
        }
        shape = tuple(data["field_img_shape"])
        return cls(H=H, zones=zones, field_img_shape=shape)

    def save_json(self, path: str | Path):
        data = {
            "H":               self.H.tolist(),
            "zones":           {n: p.tolist() for n, p in self.zones.items()},
            "field_img_shape": list(self.field_img_shape),
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"[FieldMapper] Saved calibration → {path}")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def video_to_field(self, px: float, py: float) -> tuple[float, float]:
        """Map a single video-frame pixel to field-image pixel."""
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def classify(self, px: float, py: float) -> str:
        """Return zone name for a video-frame pixel coordinate."""
        fx, fy = self.video_to_field(px, py)
        fpt = np.array([[fx, fy]], dtype=np.float32)
        for name, poly in self.zones.items():
            if cv2.pointPolygonTest(poly.astype(np.float32), (fx, fy), False) >= 0:
                return name
        return "Unknown"

    def render_minimap(
        self,
        field_img: np.ndarray,
        robot_positions: list[dict],   # [{px, py, label, zone}, ...]
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Returns a minimap image (copy of field_img) with:
          - semi-transparent zone fills
          - a dot per robot, coloured by zone
          - small text label
        Scale to MINIMAP_LONG_EDGE for embedding.
        """
        minimap = field_img.copy()
        h, w = minimap.shape[:2]

        # ── Draw zone fills ────────────────────────────────────────────
        overlay = minimap.copy()
        for name, poly in self.zones.items():
            color = ZONE_COLORS_BGR.get(name, (128, 128, 128))
            cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, alpha, minimap, 1 - alpha, 0, minimap)

        # ── Draw zone borders ──────────────────────────────────────────
        for name, poly in self.zones.items():
            color = ZONE_COLORS_BGR.get(name, (128, 128, 128))
            cv2.polylines(minimap, [poly], isClosed=True, color=color, thickness=2)

        # ── Draw robots ────────────────────────────────────────────────
        for robot in robot_positions:
            fx, fy = self.video_to_field(robot["px"], robot["py"])
            fx, fy = int(fx), int(fy)
            zone   = robot.get("zone", "Unknown")
            color  = ZONE_COLORS_BGR.get(zone, (200, 200, 200))
            label  = robot.get("label", "")

            cv2.circle(minimap, (fx, fy), ROBOT_DOT_RADIUS, (0, 0, 0), -1)         # shadow
            cv2.circle(minimap, (fx, fy), ROBOT_DOT_RADIUS - 1, color,  -1)
            cv2.circle(minimap, (fx, fy), ROBOT_DOT_RADIUS,     (255, 255, 255), 1)  # border

            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.putText(minimap, label,
                            (fx - tw // 2, fy + ROBOT_DOT_RADIUS + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(minimap, label,
                            (fx - tw // 2, fy + ROBOT_DOT_RADIUS + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # ── Scale down ─────────────────────────────────────────────────
        scale = MINIMAP_LONG_EDGE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(minimap, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration: interactive point picker
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_interactively(
    field_img_path: str | Path,
    video_source: str | Path | np.ndarray,   # path or already-read frame
    n_points: int = 6,
    frame_index: int = 3,                    # 0-based; default = 4th frame
) -> FieldMapper:
    """
    Opens two OpenCV windows side-by-side.
    Click the same real-world landmark on BOTH images to build correspondences.
    At least 4 point pairs are required; 6–8 improves accuracy.

    Keyboard controls
    -----------------
    Click  → place / confirm a point
    z      → undo last point pair
    c      → confirm and compute homography (requires ≥4 pairs)
    q      → quit without saving
    """
    field_img = cv2.imread(str(field_img_path))
    if field_img is None:
        raise FileNotFoundError(f"Cannot read field image: {field_img_path}")

    if isinstance(video_source, (str, Path)):
        cap = cv2.VideoCapture(str(video_source))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame = None
        for i in range(frame_index + 1):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise IOError(
                    f"Cannot read frame {frame_index} from: {video_source} "
                    f"(video has {total} frames)"
                )
        cap.release()
    else:
        frame = video_source.copy()

    # Resize for screen comfort
    MAX_DIM = 700
    def _fit(img):
        h, w = img.shape[:2]
        s = min(MAX_DIM / h, MAX_DIM / w, 1.0)
        return cv2.resize(img, (int(w * s), int(h * s))), s

    field_disp, fs = _fit(field_img)
    frame_disp, vs = _fit(frame)

    video_pts:  list[tuple[float, float]] = []
    field_pts:  list[tuple[float, float]] = []
    _state      = {"next": "video"}   # alternates video → field → video …

    COLORS = [(0, 255, 255), (0, 165, 255), (0, 255, 0),
              (255, 0, 255), (255, 200, 0), (200, 0, 255)]

    def _draw_state():
        fd = field_disp.copy()
        vd = frame_disp.copy()
        for i, (fp, vp) in enumerate(zip(field_pts, video_pts)):
            c = COLORS[i % len(COLORS)]
            cv2.circle(fd, (int(fp[0]), int(fp[1])), 6, c, -1)
            cv2.putText(fd, str(i + 1), (int(fp[0]) + 8, int(fp[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            cv2.circle(vd, (int(vp[0]), int(vp[1])), 6, c, -1)
            cv2.putText(vd, str(i + 1), (int(vp[0]) + 8, int(vp[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        msg_v = f"Click VIDEO  point #{len(video_pts)+1}" \
                if _state["next"] == "video" else f"Waiting…"
        msg_f = f"Click FIELD  point #{len(field_pts)+1}" \
                if _state["next"] == "field" else f"Waiting…"
        cv2.putText(vd, msg_v, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
        cv2.putText(fd, msg_f, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
        cv2.imshow("VIDEO frame (click landmarks)", vd)
        cv2.imshow("FIELD top-down (click same landmarks)", fd)

    def _on_video(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and _state["next"] == "video":
            video_pts.append((x / vs, y / vs))
            _state["next"] = "field"
            _draw_state()

    def _on_field(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and _state["next"] == "field":
            field_pts.append((x / fs, y / fs))
            _state["next"] = "video"
            _draw_state()

    cv2.namedWindow("VIDEO frame (click landmarks)")
    cv2.namedWindow("FIELD top-down (click same landmarks)")
    cv2.setMouseCallback("VIDEO frame (click landmarks)", _on_video)
    cv2.setMouseCallback("FIELD top-down (click same landmarks)", _on_field)
    _draw_state()

    H = None
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("z") and video_pts:
            video_pts.pop()
            if len(field_pts) > len(video_pts):
                field_pts.pop()
            _state["next"] = "video"
            _draw_state()
        elif key == ord("c"):
            if len(video_pts) >= 4 and len(video_pts) == len(field_pts):
                src = np.array(video_pts, dtype=np.float32)
                dst = np.array(field_pts,  dtype=np.float32)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                inliers = int(mask.sum())
                print(f"[calibrate] Homography computed — {inliers}/{len(src)} inliers")
                break
            else:
                print("[calibrate] Need ≥4 matched pairs. Keep clicking.")
        elif key == ord("q"):
            print("[calibrate] Aborted.")
            break

    cv2.destroyAllWindows()

    if H is None:
        raise RuntimeError("Calibration aborted or insufficient points.")

    fh, fw = field_img.shape[:2]
    # Return mapper with empty zones — fill in define_zones_interactively next
    return FieldMapper(H=H, zones={}, field_img_shape=(fh, fw))


# ─────────────────────────────────────────────────────────────────────────────
# Zone definition: interactive polygon drawer
# ─────────────────────────────────────────────────────────────────────────────

_ZONE_NAMES_DEFAULT = ["Red", "Neutral", "Blue"]

def define_zones_interactively(
    field_img_path: str | Path,
    zone_names: list[str] = _ZONE_NAMES_DEFAULT,
) -> dict[str, np.ndarray]:
    """
    For each zone name, lets you click polygon vertices on the field image.

    Keyboard controls
    -----------------
    Click  → add vertex
    z      → undo last vertex
    Enter  → close polygon and move to next zone
    q      → quit early
    """
    field_img = cv2.imread(str(field_img_path))
    if field_img is None:
        raise FileNotFoundError(str(field_img_path))

    MAX_DIM = 900
    h, w = field_img.shape[:2]
    scale = min(MAX_DIM / h, MAX_DIM / w, 1.0)
    disp_size = (int(w * scale), int(h * scale))

    zones: dict[str, np.ndarray] = {}
    win = "Define zones (click vertices; Enter=done, z=undo, q=quit)"
    cv2.namedWindow(win)

    for zone_name in zone_names:
        pts: list[tuple[int, int]] = []
        color = ZONE_COLORS_BGR.get(zone_name, (180, 180, 180))

        def _on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((int(x / scale), int(y / scale)))

        cv2.setMouseCallback(win, _on_click)

        while True:
            disp = cv2.resize(field_img, disp_size)
            # Draw existing zones
            for zn, zp in zones.items():
                zc = ZONE_COLORS_BGR.get(zn, (100, 100, 100))
                scaled = (zp * scale).astype(np.int32)
                cv2.fillPoly(disp, [scaled], zc)
                cv2.polylines(disp, [scaled], True, zc, 2)

            # Draw current polygon
            if pts:
                scaled_pts = [(int(p[0] * scale), int(p[1] * scale)) for p in pts]
                for p in scaled_pts:
                    cv2.circle(disp, p, 5, color, -1)
                if len(scaled_pts) > 1:
                    cv2.polylines(disp, [np.array(scaled_pts, np.int32)],
                                  False, color, 2)

            cv2.putText(disp,
                        f"Zone: {zone_name}  ({len(pts)} pts)  Enter=done  z=undo",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(disp,
                        f"Zone: {zone_name}  ({len(pts)} pts)  Enter=done  z=undo",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.imshow(win, disp)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("z") and pts:
                pts.pop()
            elif key in (13, 10):   # Enter
                if len(pts) >= 3:
                    zones[zone_name] = np.array(pts, dtype=np.int32)
                    print(f"[define_zones] '{zone_name}' saved with {len(pts)} vertices")
                    break
                else:
                    print(f"[define_zones] Need ≥3 points for '{zone_name}'")
            elif key == ord("q"):
                print("[define_zones] Early exit.")
                cv2.destroyAllWindows()
                return zones

    cv2.destroyAllWindows()
    return zones


# ─────────────────────────────────────────────────────────────────────────────
# Minimap compositor — call once per frame from detect_robots_video.py
# ─────────────────────────────────────────────────────────────────────────────

def embed_minimap(
    frame: np.ndarray,
    minimap: np.ndarray,
    corner: str = "bottom-right",
    margin: int = 12,
    border: int = 2,
) -> np.ndarray:
    """Paste minimap onto frame in-place. Returns frame."""
    fh, fw = frame.shape[:2]
    mh, mw = minimap.shape[:2]

    # Add a dark border
    bordered = cv2.copyMakeBorder(
        minimap, border, border, border, border,
        cv2.BORDER_CONSTANT, value=(30, 30, 30)
    )
    bh, bw = bordered.shape[:2]

    if corner == "bottom-right":
        y1, x1 = fh - bh - margin, fw - bw - margin
    elif corner == "bottom-left":
        y1, x1 = fh - bh - margin, margin
    elif corner == "top-right":
        y1, x1 = margin, fw - bw - margin
    else:   # top-left
        y1, x1 = margin, margin

    y2, x2 = y1 + bh, x1 + bw
    frame[y1:y2, x1:x2] = bordered
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point — run this file directly to calibrate + define zones
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate homography + define zone polygons")
    parser.add_argument("--field",   required=True, help="Top-down field image (.png/.jpg)")
    parser.add_argument("--video",   default=None,  help="Match video (used for calibration frame)")
    parser.add_argument("--frame-index", type=int, default=3,
                        help="0-based frame index to use for calibration (default: 3 = 4th frame)")
    parser.add_argument("--out",     default="field_calibration.json", help="Output calibration JSON")
    parser.add_argument("--define-zones-only", action="store_true",
                        help="Skip homography; only (re)define zone polygons on an existing JSON")
    args = parser.parse_args()

    if args.define_zones_only:
        calib = FieldMapper.from_json(args.out)
        new_zones = define_zones_interactively(args.field)
        calib.zones.update(new_zones)
        calib.save_json(args.out)
        print("Zones updated.")
    else:
        if not args.video:
            sys.exit("--video is required for homography calibration")
        mapper = calibrate_interactively(args.field, args.video, frame_index=args.frame_index)
        zones  = define_zones_interactively(args.field)
        mapper.zones = zones
        mapper.save_json(args.out)
        print(f"\nDone! Load calibration with FieldMapper.from_json('{args.out}')")