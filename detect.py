"""
HawkScan — Robot Field Position Detector
Requires hawkscan_calib.json produced by calibrate.py.

Usage:
    python detect.py path/to/image.jpg
    python detect.py path/to/image.jpg --save
    python detect.py path/to/image.jpg --debug field.png   ← shows robot positions on top-down map
    python detect.py path/to/image.jpg --calib my_calib.json
"""

import json
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

MODEL_PATH = "runs/detect/robot_detector/weights/best.pt"
CONF       = 0.25

ZONE_BEHAVIOR = {
    "red_alliance_zone":  "SHOOTING",
    "blue_alliance_zone": "SHOOTING",
    "neutral_zone":       "FERRYING",
}

ZONE_COLORS = {
    "red_alliance_zone":  (60,  60,  220),
    "neutral_zone":       (60,  180, 220),
    "blue_alliance_zone": (220, 60,  60),
}

def load_calib(path):
    with open(path) as f:
        calib = json.load(f)
    calib["homography"] = np.float32(calib["homography"])
    return calib

def camera_to_topdown(px, py, H):
    pt  = np.float32([[[px, py]]])
    out = cv2.perspectiveTransform(pt, H)
    tx  = float(out[0][0][0])
    ty  = float(out[0][0][1])
    # Temporarily flip x to test if zones are mirrored
    td_w = 600   # your field.png width
    return td_w - tx, ty

def find_zone(tx, ty, zones):
    for name, bounds in zones.items():
        x1, y1, x2, y2 = bounds
        if x1 <= tx <= x2 and y1 <= ty <= y2:
            return name
    return "unknown"

def draw_debug(topdown_path, zones, robot_td_positions, calib):
    """
    Overlay zones and mapped robot positions onto the top-down field image.
    Shows whether the homography is landing robots in the right place.
    """
    td = cv2.imread(topdown_path)
    if td is None:
        print(f"Could not open top-down image: {topdown_path}")
        return

    td_w, td_h = calib["topdown_size"]
    overlay = td.copy()

    # Draw zones
    for name, (x1, y1, x2, y2) in zones.items():
        color = ZONE_COLORS.get(name, (200, 200, 200))
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        cv2.rectangle(td,      (int(x1), int(y1)), (int(x2), int(y2)), color,  2)
        cv2.putText(td, name.replace("_", " "), (int(x1)+4, int(y1)+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    td = cv2.addWeighted(overlay, 0.2, td, 0.8, 0)

    # Draw mapped robot positions
    for i, (tx, ty, conf) in enumerate(robot_td_positions):
        ix, iy = int(tx), int(ty)
        color  = colors(i, True)
        bgr    = (int(color[2]*255), int(color[1]*255), int(color[0]*255)) if isinstance(color[0], float) else color
        in_bounds = 0 <= ix < td_w and 0 <= iy < td_h
        marker_color = bgr if in_bounds else (0, 0, 255)
        # Clamp for drawing even if out of bounds
        draw_x = max(4, min(td.shape[1]-4, ix))
        draw_y = max(4, min(td.shape[0]-4, iy))
        cv2.drawMarker(td, (draw_x, draw_y), marker_color, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(td, f"R{i+1} ({ix},{iy})", (draw_x+6, draw_y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, marker_color, 1)
        if not in_bounds:
            print(f"  ⚠ Robot {i+1} mapped OUTSIDE top-down image ({ix},{iy}) — recalibrate")

    print(f"\nTop-down image bounds: 0–{td_w}px wide, 0–{td_h}px tall")
    print("Green crosses = in bounds, Red = out of bounds")
    print("If robots are red or far off, run calibrate.py again with more accurate point pairs.\n")

    cv2.imshow("HawkScan Debug — Top-Down", td)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run(image_path, calib_path, save, debug_topdown):
    calib = load_calib(calib_path)
    H     = calib["homography"]
    zones = calib["zones"]

    model   = YOLO(MODEL_PATH)
    results = model(image_path, conf=CONF, verbose=False)
    r       = results[0]

    print(f"Detections: {len(r.boxes)}")

    robots = [
        (*box.xyxy[0].tolist(), box.conf.item())
        for box in r.boxes
    ]

    if not robots:
        print("No robots detected.")
        return

    img = r.orig_img.copy()
    ann = Annotator(img)
    robot_td_positions = []

    for i, (x1, y1, x2, y2, conf) in enumerate(robots):
        cx, cy   = (x1 + x2) / 2, (y1 + y2) / 2
        tx, ty   = camera_to_topdown(cx, cy, H)
        robot_td_positions.append((tx, ty, conf))

        zone     = find_zone(tx, ty, zones)
        behavior = ZONE_BEHAVIOR.get(zone, "UNKNOWN")
        label    = f"Robot {i+1} | {zone.replace('_',' ').upper()} | {behavior}"

        print(f"{label}  (conf {conf:.0%},  topdown pos {tx:.0f}px, {ty:.0f}px)")
        ann.box_label([x1, y1, x2, y2], label, color=colors(i, True))

    result_img = ann.result()

    if debug_topdown:
        draw_debug(debug_topdown, zones, robot_td_positions, calib)

    if save:
        out = "hawkscan_result.jpg"
        cv2.imwrite(out, result_img)
        print(f"Saved → {out}")
    else:
        cv2.imshow("HawkScan", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--save",  action="store_true")
    parser.add_argument("--calib", default="hawkscan_calib.json")
    parser.add_argument("--debug", metavar="TOPDOWN_IMAGE",
                        help="Show robot positions mapped onto this top-down field image")
    args = parser.parse_args()
    run(args.image, args.calib, args.save, args.debug)