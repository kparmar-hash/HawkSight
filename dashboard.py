"""
HawkSight Dashboard
====================
A simple tkinter dashboard that displays the FRC field image alongside
six clickable robot name / ID placeholders.

Usage:
    python dashboard.py
"""
import os
import sys

# ── Fix Tcl/Tk lookup for Python 3.13+ on Windows ───────────
# The installer puts tcl under <prefix>/tcl/ but tkinter searches
# <prefix>/lib/.  Setting these env vars before import resolves it.
_python_dir = os.path.dirname(sys.executable)
# Walk up from a venv's Scripts/ dir to find the base Python install
_base_dir = getattr(sys, "_base_executable", sys.executable)
_base_dir = os.path.dirname(_base_dir)
_tcl_dir = os.path.join(_base_dir, "tcl")
if os.path.isdir(_tcl_dir):
    for child in os.listdir(_tcl_dir):
        full = os.path.join(_tcl_dir, child)
        if child.startswith("tcl") and os.path.isdir(full):
            os.environ.setdefault("TCL_LIBRARY", full)
        elif child.startswith("tk") and os.path.isdir(full):
            os.environ.setdefault("TK_LIBRARY", full)

import tkinter as tk
from tkinter import messagebox
from pathlib import Path
# pyrefly: ignore [missing-import]
from PIL import Image, ImageTk


# ─────────────────────────────────────────────────────────────
# Colour palette  (dark theme)
# ─────────────────────────────────────────────────────────────

BG_DARK    = "#1a1a2e"
BG_PANEL   = "#16213e"
BG_CARD    = "#0f3460"
ACCENT     = "#e94560"
TEXT_LIGHT = "#eaeaea"
TEXT_DIM   = "#8a8a9a"
HOVER_BG   = "#1a4a7a"

# Per-robot accent colours (alternating red / blue alliance feel)
ROBOT_COLORS = [
    "#e94560",  # Red 1
    "#e94560",  # Red 2
    "#e94560",  # Red 3
    "#4e9af5",  # Blue 1
    "#4e9af5",  # Blue 2
    "#4e9af5",  # Blue 3
]


# ─────────────────────────────────────────────────────────────
# Placeholder robot data
# ─────────────────────────────────────────────────────────────

ROBOTS = [
    {"id": "R1"},
    {"id": "R2"},
    {"id": "R3"},
    {"id": "B1"},
    {"id": "B2"},
    {"id": "B3"},
]

ACTIVE_BG = "#1a4a7a"    # selected card background
INACTIVE_BG = BG_CARD    # default card background


# ─────────────────────────────────────────────────────────────
# Dashboard Application
# ─────────────────────────────────────────────────────────────

class HawkSightDashboard:
    """Main dashboard window."""

    FIELD_IMAGE_PATH = Path(__file__).resolve().parent / "configs" / "field.png"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("HawkSight — Field Dashboard")
        self.root.configure(bg=BG_DARK)
        self.root.minsize(960, 600)

        # Try to start maximised (cross-platform safe)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.attributes("-zoomed", True)

        self._selected_robot: str | None = None
        self._robot_frames: dict[str, tk.Frame] = {}

        self._build_ui()

    # ── UI construction ──────────────────────────────────────

    def _build_ui(self):
        # Top title bar
        title_bar = tk.Frame(self.root, bg=ACCENT, height=48)
        title_bar.pack(fill=tk.X, side=tk.TOP)
        title_bar.pack_propagate(False)

        tk.Label(
            title_bar,
            text="🦅  HAWKSIGHT  DASHBOARD",
            font=("Segoe UI", 16, "bold"),
            bg=ACCENT,
            fg="white",
        ).pack(side=tk.LEFT, padx=16)

        tk.Label(
            title_bar,
            text="Field Overview",
            font=("Segoe UI", 11),
            bg=ACCENT,
            fg="#cccccc",
        ).pack(side=tk.RIGHT, padx=16)

        # Main content area  (sidebar + field)
        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # ── Sidebar ──────────────────────────────────────────
        sidebar = tk.Frame(content, bg=BG_PANEL, width=260)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        # Selected-robot display at the top of the sidebar
        self._selected_label = tk.Label(
            sidebar,
            text="No robot selected",
            font=("Segoe UI", 11, "bold"),
            bg=BG_PANEL,
            fg=TEXT_LIGHT,
            anchor="w",
        )
        self._selected_label.pack(fill=tk.X, padx=18, pady=(16, 4))

        tk.Frame(sidebar, bg=TEXT_DIM, height=1).pack(fill=tk.X, padx=18, pady=(0, 8))

        tk.Label(
            sidebar,
            text="ROBOTS",
            font=("Segoe UI", 12, "bold"),
            bg=BG_PANEL,
            fg=TEXT_DIM,
            anchor="w",
        ).pack(fill=tk.X, padx=18, pady=(10, 8))

        for idx, robot in enumerate(ROBOTS):
            self._make_robot_card(sidebar, idx, robot)

        # Spacer
        tk.Frame(sidebar, bg=BG_PANEL).pack(fill=tk.BOTH, expand=True)

        # Footer inside sidebar
        tk.Label(
            sidebar,
            text="Click a robot to select",
            font=("Segoe UI", 9, "italic"),
            bg=BG_PANEL,
            fg=TEXT_DIM,
        ).pack(side=tk.BOTTOM, pady=12)

        # ── Field image area ─────────────────────────────────
        field_frame = tk.Frame(content, bg=BG_DARK)
        field_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        field_frame.rowconfigure(0, weight=1)
        field_frame.columnconfigure(0, weight=1)

        self._field_label = tk.Label(field_frame, bg=BG_DARK, bd=0)
        self._field_label.grid(row=0, column=0, sticky="nsew")

        # Load & display the field image
        self._field_photo = None  # prevent GC
        self._load_field_image()

        # Re-scale on resize
        self._field_label.bind("<Configure>", self._on_field_resize)

    # ── Robot card widget ────────────────────────────────────

    def _make_robot_card(self, parent: tk.Frame, idx: int, robot: dict):
        accent = ROBOT_COLORS[idx]
        rid = robot["id"]

        card = tk.Frame(parent, bg=INACTIVE_BG, cursor="hand2", highlightthickness=0)
        card.pack(fill=tk.X, padx=14, pady=4)

        # Coloured left stripe
        stripe = tk.Frame(card, bg=accent, width=5)
        stripe.pack(side=tk.LEFT, fill=tk.Y)

        info = tk.Frame(card, bg=INACTIVE_BG)
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=8)

        id_label = tk.Label(
            info,
            text=rid,
            font=("Consolas", 12, "bold"),
            bg=INACTIVE_BG,
            fg=accent,
            anchor="w",
        )
        id_label.pack(anchor="w")

        # Store reference
        self._robot_frames[rid] = card

        # ── Helpers to set card background ──────────────────
        def _set_card_bg(bg: str):
            """Apply bg to card and its non-stripe children."""
            card.configure(bg=bg)
            info.configure(bg=bg)
            id_label.configure(bg=bg)

        def _current_base_bg() -> str:
            """Return the base bg for this card (active or inactive)."""
            return ACTIVE_BG if self._selected_robot == rid else INACTIVE_BG

        # Hover effects  (event arg required by tkinter bind)
        def on_hover(_event):
            if self._selected_robot != rid:
                _set_card_bg(HOVER_BG)

        def on_leave(_event):
            _set_card_bg(_current_base_bg())

        # Click → select
        def on_click(_event):
            self._select_robot(rid)

        for widget in (card, stripe, info, id_label):
            widget.bind("<Enter>", on_hover)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", on_click)
            
    # ── Robot selection ──────────────────────────────────────

    def _reset_card(self, robot_id: str):
        """Reset a card back to its inactive appearance."""
        card = self._robot_frames[robot_id]
        for w in card.winfo_children():
            if isinstance(w, tk.Frame) and w.cget("background") not in ROBOT_COLORS:
                for c in w.winfo_children():
                    # pyrefly: ignore [unexpected-keyword]
                    c.configure(background=INACTIVE_BG)
                w.configure(background=INACTIVE_BG)
        card.configure(background=INACTIVE_BG)

    def _select_robot(self, robot_id: str):
        # Deselect if clicking the already-selected robot
        if self._selected_robot == robot_id:
            self._reset_card(robot_id)
            self._selected_robot = None
            self._selected_label.configure(text="No robot selected")
            print(f"[HawkSight] Deselected robot: {robot_id}")
            return

        # Reset previous selection back to inactive
        if self._selected_robot and self._selected_robot in self._robot_frames:
            self._reset_card(self._selected_robot)

        self._selected_robot = robot_id

        # Highlight new selection
        card = self._robot_frames[robot_id]
        for w in card.winfo_children():
            if isinstance(w, tk.Frame) and w.cget("background") not in ROBOT_COLORS:
                for c in w.winfo_children():
                    # pyrefly: ignore [unexpected-keyword]
                    c.configure(background=ACTIVE_BG)
                w.configure(background=ACTIVE_BG)
        card.configure(background=ACTIVE_BG)

        # Update the label at the top of the sidebar
        self._selected_label.configure(text=f"▶  {robot_id}")

        print(f"[HawkSight] Selected robot: {robot_id}")

    # ── Field image loading / resizing ───────────────────────

    def _load_field_image(self):
        if not self.FIELD_IMAGE_PATH.exists():
            self._field_label.configure(
                text="⚠  field.png not found\n" + str(self.FIELD_IMAGE_PATH),
                fg=ACCENT,
                font=("Segoe UI", 12),
            )
            return

        self._field_pil = Image.open(self.FIELD_IMAGE_PATH)
        self._render_field()

    def _render_field(self):
        if not hasattr(self, "_field_pil"):
            return

        # Fit image into available space while preserving aspect ratio
        lbl_w = self._field_label.winfo_width()  or 800
        lbl_h = self._field_label.winfo_height() or 600

        img_w, img_h = self._field_pil.size
        scale = min(lbl_w / img_w, lbl_h / img_h, 1.0)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = self._field_pil.resize((new_w, new_h), Image.LANCZOS)
        self._field_photo = ImageTk.PhotoImage(resized)
        self._field_label.configure(image=self._field_photo)

    def _on_field_resize(self, _event):
        self._render_field()


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    HawkSightDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
