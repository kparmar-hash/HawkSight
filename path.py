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

    