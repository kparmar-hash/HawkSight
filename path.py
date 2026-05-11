#TODO: Read the output mp4 file that the tracker has an annotate a path for each robot labelled in robot registry
# The path should be saved as a JSON map and then loaded and displayed by the dashboard when called upon, this file
# will remain the main file that is called, then it will call dashboard.py to display the specific paths.

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics.utils.plotting import Annotator, colors

import json

frames = 0


#TODO: Read and split the MP4 file
def framify_mp4(input_path: str, pad: int = 18, show: bool = False):
    global frames
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {w}x{h}  {src_fps:.2f} fps  {total_frames} frames")

    frames = total_frames
    print(f"Frames total exported: {frames}")


#TODO: Recieve robot_registry and map each robot to an array/list of its path coordinates (x, y)
# The array/list should be indexed by frame number
def initialize_robot_paths(robot_registry, total_frames):
    """
    Maps each robot from the registry to a list of (x, y) coordinates.
    The list is indexed by frame number (0 to total_frames - 1).
    """
    path_map = {}
    
    # Handle if robot_registry is the object from main.py or just a list of IDs
    ids = robot_registry.entries.keys() if hasattr(robot_registry, 'entries') else robot_registry
    
    for robot_id in ids:
        # Initialize an empty path (None) for every frame in the video
        # We use str(robot_id) because JSON keys must be strings for compatibility
        path_map[str(robot_id)] = [None] * total_frames
        
    return path_map