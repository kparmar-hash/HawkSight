"""
HawkScan — Live Video Detector
Grabs the first frame from a video/stream, calibrates against it,
then runs robot detection on every subsequent frame.

Usage:
    python identify.py --topdown configs/field.png --video dataset/raw/tests/tests_0009.png
    python run.py --topdown field.png --video 0           (webcam index)
    python run.py --topdown field.png --video match.mp4 --calib saved.json   (skip calibration)
    python run.py --topdown field.png --video match.mp4 --recalibrate        (force redo calib)
    python run.py --topdown field.png --video match.mp4 --save output.mp4
"""

import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = "runs/detect/robot_detector/weights/best.pt"
CONF         = 0.25
CALIB_FILE   = "hawkscan_calib.json"
PANEL_HEIGHT = 700
MIN_POINTS   = 4

ZONE_BEHAVIOR = {
    "red_alliance_zone":  "SHOOTING",
    "blue_alliance_zone": "SHOOTING",
    "neutral_zone":       "FERRYING",
}

DEFAULT_ZONE_FRACS = {
    "red_alliance_zone":  (0.00, 0.00, 0.25, 1.00),
    "neutral_zone":       (0.25, 0.00, 0.75, 1.00),
    "blue_alliance_zone": (0.75, 0.00, 1.00, 1.00),
}

# ── Calibration UI ────────────────────────────────────────────────────────────

class CalibState:
    def __init__(self):
        self.pairs      = []   # [(td_pt, cam_pt), ...]
        self.left_click = None
        self.side       = "left"

    def undo(self):
        if self.pairs:
            self.pairs.pop()
            self.left_click = None
            self.side = "left"

calib_state = CalibState()
td_panel_w  = 0
td_scale    = 1.0
cam_scale   = 1.0

def mouse_cb(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    on_left = x < td_panel_w

    if calib_state.side == "left" and on_left:
        calib_state.left_click = (x / td_scale, y / td_scale)
        calib_state.side = "right"
        print(f"  [{len(calib_state.pairs)+1}] Top-down clicked — now click the same point on the camera frame")

    elif calib_state.side == "right" and not on_left:
        if calib_state.left_click is None:
            return
        cam_x = (x - td_panel_w) / cam_scale
        cam_y = y / cam_scale
        calib_state.pairs.append((calib_state.left_click, (cam_x, cam_y)))
        print(f"  Pair {len(calib_state.pairs)} set.{' Press ENTER to calibrate.' if len(calib_state.pairs) >= MIN_POINTS else ''}")
        calib_state.left_click = None
        calib_state.side = "left"

def scale_to_height(img, h):
    oh, ow = img.shape[:2]
    s = h / oh
    return cv2.resize(img, (int(ow * s), h)), s

def draw_calib_panel(img, pts, pending, label, scale):
    vis = img.copy()
    for i, pt in enumerate(pts):
        sx, sy = int(pt[0] * scale), int(pt[1] * scale)
        cv2.circle(vis, (sx, sy), 7, (0, 255, 80), -1)
        cv2.putText(vis, str(i+1), (sx+9, sy-9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
    if pending:
        sx, sy = int(pending[0] * scale), int(pending[1] * scale)
        cv2.circle(vis, (sx, sy), 7, (0, 200, 255), 2)
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return vis

def run_calibration(td_img, cam_frame, calib_out_path):
    global td_panel_w, td_scale, cam_scale, calib_state
    calib_state = CalibState()

    td_disp,  td_scale  = scale_to_height(td_img,    PANEL_HEIGHT)
    cam_disp, cam_scale = scale_to_height(cam_frame, PANEL_HEIGHT)
    td_panel_w = td_disp.shape[1]

    # Build zones from fracs
    h, w = td_img.shape[:2]
    zones = {
        name: (xf1*w, yf1*h, xf2*w, yf2*h)
        for name, (xf1, yf1, xf2, yf2) in DEFAULT_ZONE_FRACS.items()
    }

    print("\nHawkScan Calibration")
    print("────────────────────────────────────────────────────")
    print("LEFT panel = top-down field map")
    print("RIGHT panel = first frame of your video")
    print(f"Click {MIN_POINTS}+ matching landmark pairs (field corners work best).")
    print("Click LEFT first, then the same point on the RIGHT.")
    print("R = undo  |  ENTER = save & start detection  |  Q = quit\n")

    cv2.namedWindow("HawkScan Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("HawkScan Calibration", mouse_cb)

    while True:
        n     = len(calib_state.pairs)
        ready = n >= MIN_POINTS

        left_pts  = [p[0] for p in calib_state.pairs]
        right_pts = [p[1] for p in calib_state.pairs]

        left_label  = f"TOP-DOWN  ({n} pairs){'  ENTER=start' if ready else '  — click here first'}"
        right_label = f"CAMERA FRAME{'  — click matching point' if calib_state.side=='right' else ''}"

        td_vis  = draw_calib_panel(td_disp.copy(), left_pts, calib_state.left_click, left_label, td_scale)
        cam_vis = draw_calib_panel(cam_disp.copy(), right_pts, None, right_label, cam_scale)
        frame   = np.hstack([td_vis, cam_vis])
        cv2.line(frame, (td_panel_w, 0), (td_panel_w, PANEL_HEIGHT), (80, 80, 80), 2)

        cv2.imshow("HawkScan Calibration", frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            calib_state.undo()
            print("  Undid last pair.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None
        elif key == 13 and ready:   # ENTER
            td_pts  = np.float32([p[0] for p in calib_state.pairs])
            cam_pts = np.float32([p[1] for p in calib_state.pairs])
            H, mask = cv2.findHomography(cam_pts, td_pts, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else n
            print(f"\n  Homography computed — {inliers}/{n} inliers")

            calib = {
                "topdown_size": [td_img.shape[1], td_img.shape[0]],
                "homography":   H.tolist(),
                "zones":        {k: list(v) for k, v in zones.items()},
            }
            with open(calib_out_path, "w") as f:
                json.dump(calib, f, indent=2)
            print(f"  Saved → {calib_out_path}\n")
            cv2.destroyAllWindows()
            return calib

# ── Detection ─────────────────────────────────────────────────────────────────

def load_calib(path):
    with open(path) as f:
        calib = json.load(f)
    calib["homography"] = np.float32(calib["homography"])
    return calib

def camera_to_topdown(px, py, H):
    pt  = np.float32([[[px, py]]])
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0][0][0]), float(out[0][0][1])

def find_zone(tx, ty, zones):
    for name, bounds in zones.items():
        x1, y1, x2, y2 = bounds
        if x1 <= tx <= x2 and y1 <= ty <= y2:
            return name
    return "unknown"

def process_frame(frame, model, H, zones):
    results = model(frame, conf=CONF, verbose=False)
    r       = results[0]
    ann     = Annotator(frame.copy())

    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        cx, cy   = (x1 + x2) / 2, (y1 + y2) / 2
        tx, ty   = camera_to_topdown(cx, cy, H)
        zone     = find_zone(tx, ty, zones)
        behavior = ZONE_BEHAVIOR.get(zone, "UNKNOWN")
        label    = f"{zone.replace('_',' ').upper()} | {behavior}"
        ann.box_label([x1, y1, x2, y2], label, color=colors(i, True))

    return ann.result()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topdown",     required=True, help="Top-down field map image")
    parser.add_argument("--video",       required=True, help="Video file path or camera index (0, 1, …)")
    parser.add_argument("--calib",       default=CALIB_FILE)
    parser.add_argument("--recalibrate", action="store_true", help="Force redo calibration")
    parser.add_argument("--save",        default=None, metavar="OUTPUT.mp4")
    args = parser.parse_args()

    td_img = cv2.imread(args.topdown)
    if td_img is None:
        print(f"Cannot open top-down image: {args.topdown}")
        return

    # Open video source — accept image files, video files, or camera index
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    is_image   = Path(args.video).suffix.lower() in IMAGE_EXTS

    if is_image:
        first_frame = cv2.imread(args.video)
        if first_frame is None:
            print(f"Cannot open image: {args.video}")
            return
        cap = None
    else:
        src = int(args.video) if args.video.isdigit() else args.video
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"Cannot open video: {args.video}")
            return
        ret, first_frame = cap.read()
        if not ret:
            print("Could not read first frame from video.")
            cap.release()
            return

    # Strip alpha channel if present (e.g. RGBA PNG)
    if first_frame.shape[2] == 4:
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGRA2BGR)

    # ── Calibration ───────────────────────────────────────────────────────────
    calib_path = Path(args.calib)
    if calib_path.exists() and not args.recalibrate:
        print(f"Loading calibration from {calib_path}")
        calib = load_calib(calib_path)
    else:
        print("Starting calibration against first video frame…")
        calib = run_calibration(td_img, first_frame, calib_path)
        if calib is None:
            print("Calibration cancelled.")
            cap.release()
            return
        calib["homography"] = np.float32(calib["homography"])

    H     = calib["homography"]
    zones = calib["zones"]

    # ── Detection loop ────────────────────────────────────────────────────────
    model = YOLO(MODEL_PATH)

    writer = None
    if args.save and not is_image:
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps    = cap.get(cv2.CAP_PROP_FPS) if cap else 30
        writer = cv2.VideoWriter(args.save, fourcc, fps or 30, (w, h))

    print("Running detection — press Q to quit.\n")

    # Always process the first frame
    out = process_frame(first_frame, model, H, zones)
    cv2.imshow("HawkScan", out)
    if writer:
        writer.write(out)

    if is_image:
        # Still image — just show result and wait for keypress
        cv2.waitKey(0)
    else:
        # Video / stream loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out = process_frame(frame, model, H, zones)
            cv2.imshow("HawkScan", out)
            if writer:
                writer.write(out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()

    if writer:
        writer.release()
        print(f"Saved → {args.save}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()