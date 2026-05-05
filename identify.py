"""
HawkScan — Interactive Robot Identifier
Click on a robot in the image — YOLO runs only on that region and draws one box.

Usage:
    python identify.py                  (opens file picker)
    python identify.py path/to/img.jpg  (opens directly)
"""

import sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = "runs/detect/robot_detector/weights/best.pt"
CONF         = 0.25
CROP_PADDING = 200    # px (original scale) added around the click to form the crop window
BOX_COLOR    = "#FF3B3B"
BOX_WIDTH    = 3
MAX_DISPLAY  = 900    # max window dimension; image scales down to fit

# ── App ───────────────────────────────────────────────────────────────────────

class HawkScanApp:
    def __init__(self, root: tk.Tk, image_path: str):
        self.root = root
        self.root.title("HawkScan — Click a robot")
        self.root.configure(bg="#1a1a1a")

        self.model    = YOLO(MODEL_PATH)
        self.original = Image.open(image_path).convert("RGB")
        self.orig_w, self.orig_h = self.original.size
        self.scale    = self._fit_scale(self.original.size, MAX_DISPLAY)
        dw = int(self.orig_w * self.scale)
        dh = int(self.orig_h * self.scale)
        self.display_base = self.original.resize((dw, dh), Image.LANCZOS)
        self.active_box   = None   # (x1, y1, x2, y2, conf) in original coords

        self._build_ui(dw, dh)
        self._refresh_canvas()
        self._set_status("Click on a robot to identify it.")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self, w: int, h: int):
        self.status_var = tk.StringVar()
        tk.Label(
            self.root, textvariable=self.status_var,
            bg="#1a1a1a", fg="#ffffff",
            font=("Helvetica", 12), pady=8
        ).pack(fill=tk.X)

        self.canvas = tk.Canvas(
            self.root, width=w, height=h,
            bg="#1a1a1a", highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(padx=16, pady=(0, 8))
        self.canvas.bind("<Button-1>", self._on_click)

        btn_frame = tk.Frame(self.root, bg="#1a1a1a")
        btn_frame.pack(pady=(0, 12))

        for label, cmd in [("Clear", self._clear), ("Open New Image", self._open_new)]:
            tk.Button(
                btn_frame, text=label, command=cmd,
                bg="#333333", fg="#ffffff",
                font=("Helvetica", 11), relief="flat", padx=12, pady=4
            ).pack(side=tk.LEFT, padx=6)

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_at(self, click_ox: float, click_oy: float):
        """
        Crop a window around (click_ox, click_oy) in original pixel coords,
        run YOLO on just that patch, then return the best box remapped back
        to original coords — or None if nothing found.
        """
        pad = CROP_PADDING

        # Clamp crop to image bounds
        cx1 = max(0,           int(click_ox - pad))
        cy1 = max(0,           int(click_oy - pad))
        cx2 = min(self.orig_w, int(click_ox + pad))
        cy2 = min(self.orig_h, int(click_oy + pad))

        crop    = self.original.crop((cx1, cy1, cx2, cy2))
        results = self.model(crop, conf=CONF, verbose=False)

        if not results[0].boxes:
            return None

        # Pick the detection whose centre is closest to the click (in crop space)
        click_in_crop_x = click_ox - cx1
        click_in_crop_y = click_oy - cy1
        best, best_dist = None, float("inf")

        for box in results[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            cx = (bx1 + bx2) / 2
            cy = (by1 + by2) / 2
            dist = (cx - click_in_crop_x) ** 2 + (cy - click_in_crop_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best = (bx1, by1, bx2, by2, box.conf.item())

        if best is None:
            return None

        # Remap from crop coords → original image coords
        bx1, by1, bx2, by2, conf = best
        return (bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1, conf)

    # ── Canvas ────────────────────────────────────────────────────────────────

    def _refresh_canvas(self):
        img = self.display_base.copy()

        if self.active_box:
            x1, y1, x2, y2, conf = self.active_box
            sx1 = int(x1 * self.scale)
            sy1 = int(y1 * self.scale)
            sx2 = int(x2 * self.scale)
            sy2 = int(y2 * self.scale)

            draw = ImageDraw.Draw(img)
            for t in range(BOX_WIDTH):
                draw.rectangle([sx1-t, sy1-t, sx2+t, sy2+t], outline=BOX_COLOR)
            label = f"FRC Robot  {conf:.0%}"
            draw.rectangle([sx1, sy1 - 22, sx1 + len(label)*8 + 8, sy1], fill=BOX_COLOR)
            draw.text((sx1 + 4, sy1 - 20), label, fill="#ffffff")

        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

    # ── Interaction ───────────────────────────────────────────────────────────

    def _on_click(self, event):
        ox = event.x / self.scale
        oy = event.y / self.scale

        self._set_status("Detecting…")
        self.root.update_idletasks()

        box = self._detect_at(ox, oy)

        if box:
            self.active_box = box
            self._refresh_canvas()
            self._set_status(f"Robot locked  ·  confidence {box[4]:.0%}")
        else:
            self._set_status("No robot found near that click — try clicking closer to the robot.")

    def _clear(self):
        self.active_box = None
        self._refresh_canvas()
        self._set_status("Click on a robot to identify it.")

    def _open_new(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")]
        )
        if path:
            self.root.destroy()
            launch(path)

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    @staticmethod
    def _fit_scale(size: tuple, max_dim: int) -> float:
        return min(max_dim / size[0], max_dim / size[1], 1.0)

# ── Entry point ───────────────────────────────────────────────────────────────

def launch(path: str = None):
    if path is None:
        tmp = tk.Tk()
        tmp.withdraw()
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")]
        )
        tmp.destroy()
        if not path:
            return

    root = tk.Tk()
    root.resizable(False, False)
    HawkScanApp(root, path)
    root.mainloop()

