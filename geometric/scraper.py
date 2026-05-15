"""
TBA 2026 FRC Match Video Scraper + Frame Extractor
---------------------------------------------------
Downloads match videos from The Blue Alliance for 2026 events
and extracts frames for YOLO bounding box labeling.

Requirements:
    pip install requests tqdm opencv-python
    brew install yt-dlp ffmpeg   (mac)
    sudo apt install ffmpeg      (linux) + pip install yt-dlp

Usage:
    # Set your TBA key (free at thebluealliance.com/account)
    export TBA_KEY="your_key_here"

    # Download from 3 random regionals and extract frames
    python tba_2026_scraper.py --events 3 --matches-per-event 5

    # Target a specific event
    python tba_2026_scraper.py --event-key 2026gacmp

    # Just extract frames from already-downloaded videos
    python tba_2026_scraper.py --extract-only --videos-dir data/videos
"""

import os
import sys
import json
import random
import logging
import argparse
import subprocess
from pathlib import Path

import cv2
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TBA_KEY   = "A1zE9cucdkqjWns5B1DAXijYnbWHhqQ7fS05PVQVjhbKP1Mx4bO4pRGhUcitZZgK"
TBA_BASE  = "https://www.thebluealliance.com/api/v3"

# FRC match is ~2:30. Skip the first and last N seconds
# to avoid pre-match setup and post-match celebration frames
# where robots are stationary / piled together.
SKIP_START_SEC = 8
SKIP_END_SEC   = 10

# Extract 2 frames per second during the match window
EXTRACT_FPS = 5.0

# Official event types only (no offseason/preseason)
# 0=Regional, 1=District, 2=District Champ, 3=Champ Division, 4=Champ Finals
OFFICIAL_EVENT_TYPES = {0, 1, 2, 3, 4}

OUTPUT_DIR = Path("data")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TBA API helpers
# ---------------------------------------------------------------------------

def _tba_get(path: str) -> list | dict:
    if not TBA_KEY:
        log.error("TBA_KEY not set. Get a free key at thebluealliance.com/account")
        sys.exit(1)
    resp = requests.get(
        f"{TBA_BASE}{path}",
        headers={"X-TBA-Auth-Key": TBA_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def get_2026_events(event_types: set = OFFICIAL_EVENT_TYPES) -> list[dict]:
    """Return all official 2026 FRC events, sorted by start date."""
    log.info("Fetching 2026 event list from TBA...")
    events = _tba_get("/events/2026")
    filtered = [e for e in events if e.get("event_type") in event_types]
    filtered.sort(key=lambda e: e.get("start_date", ""))
    log.info(f"Found {len(filtered)} official 2026 events")
    return filtered


def get_event_matches_with_video(event_key: str) -> list[dict]:
    """Return matches for an event that have a YouTube video attached."""
    matches = _tba_get(f"/event/{event_key}/matches")
    with_video = [
        m for m in matches
        if any(v["type"] == "youtube" for v in m.get("videos", []))
    ]
    log.info(f"  {event_key}: {len(with_video)}/{len(matches)} matches have video")
    return with_video


def youtube_url(match: dict) -> str | None:
    for v in match.get("videos", []):
        if v["type"] == "youtube":
            return f"https://www.youtube.com/watch?v={v['key']}"
    return None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_match_video(match: dict, out_dir: Path) -> Path | None:
    """Download a single match video. Returns path to .mp4, or None on failure."""
    url = youtube_url(match)
    if not url:
        return None

    match_key = match["key"]
    out_path   = out_dir / f"{match_key}.mp4"

    if out_path.exists():
        log.info(f"  Already downloaded: {match_key}")
        return out_path

    log.info(f"  Downloading: {match_key}  ({url})")
    try:
        subprocess.run(
            [
                "yt-dlp",
                "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/mp4",
                "--merge-output-format", "mp4",
                "--no-playlist",
                "--quiet",
                "--progress",
                "-o", str(out_path),
                url,
            ],
            check=True,
        )
        return out_path
    except subprocess.CalledProcessError as e:
        log.warning(f"  yt-dlp failed for {match_key}: {e}")
        return None


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total / fps


def extract_frames_ffmpeg(
    video_path: Path,
    out_dir: Path,
    start_sec: float,
    end_sec: float,
    fps: float = EXTRACT_FPS,
    prefix: str = "frame",
) -> int:
    """
    Extract frames between start_sec and end_sec using ffmpeg.
    Returns number of frames saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = end_sec - start_sec
    if duration <= 0:
        log.warning(f"  Skipping {video_path.name}: no usable time window")
        return 0

    out_pattern = str(out_dir / f"{prefix}_%07d.jpg")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-ss", str(start_sec),
                "-i", str(video_path),
                "-t", str(duration),
                "-vf", f"fps={fps}",
                "-q:v", "2",          # JPEG quality (1-31, lower = better)
                "-vsync", "vfr",
                out_pattern,
                "-hide_banner",
                "-loglevel", "error",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        log.warning(f"  ffmpeg failed for {video_path.name}: {e}")
        return 0

    saved = len(list(out_dir.glob(f"{prefix}_*.jpg")))
    return saved


def filter_blurry_frames(frames_dir: Path, blur_threshold: float = 80.0) -> tuple[int, int]:
    """
    Remove blurry frames using Laplacian variance.
    Returns (kept, removed).
    """
    frames = list(frames_dir.glob("*.jpg"))
    removed = 0
    for f in tqdm(frames, desc="  Filtering blurry", leave=False):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None or cv2.Laplacian(img, cv2.CV_64F).var() < blur_threshold:
            f.unlink()
            removed += 1
    return len(frames) - removed, removed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_event(event: dict, max_matches: int, frames_dir: Path, videos_dir: Path) -> int:
    """Download matches for one event and extract frames. Returns frame count."""
    event_key  = event["key"]
    event_name = event.get("short_name") or event.get("name", event_key)
    log.info(f"\n{'='*60}")
    log.info(f"Event: {event_name}  ({event_key})")
    log.info(f"{'='*60}")

    matches = get_event_matches_with_video(event_key)
    if not matches:
        log.warning("  No matches with video — skipping")
        return 0

    # Prefer qualification matches for variety; fall back to all
    qual = [m for m in matches if m.get("comp_level") == "qm"]
    pool = qual if len(qual) >= max_matches else matches

    # Sample evenly across the event (early + mid + late qual matches)
    if len(pool) > max_matches:
        indices = [int(i * len(pool) / max_matches) for i in range(max_matches)]
        pool = [pool[i] for i in indices]

    total_frames = 0
    event_frames_dir = frames_dir / event_key
    event_videos_dir = videos_dir / event_key
    event_videos_dir.mkdir(parents=True, exist_ok=True)

    for match in pool:
        match_key = match["key"]
        log.info(f"\n  Match: {match_key}")

        video_path = download_match_video(match, event_videos_dir)
        if not video_path:
            continue

        duration = get_video_duration(video_path)
        log.info(f"  Duration: {duration:.1f}s")

        start = SKIP_START_SEC
        end   = max(start + 10, duration - SKIP_END_SEC)

        match_frames_dir = event_frames_dir / match_key
        n = extract_frames_ffmpeg(
            video_path,
            match_frames_dir,
            start_sec=start,
            end_sec=end,
            fps=EXTRACT_FPS,
            prefix=match_key.replace("_", "-"),
        )
        log.info(f"  Extracted: {n} raw frames")

        kept, removed = filter_blurry_frames(match_frames_dir)
        log.info(f"  After blur filter: {kept} kept, {removed} removed")
        total_frames += kept

    return total_frames


def extract_only(videos_dir: Path, frames_dir: Path) -> int:
    """Extract frames from all .mp4 files already in videos_dir."""
    videos = list(videos_dir.rglob("*.mp4"))
    log.info(f"Found {len(videos)} videos in {videos_dir}")
    total = 0
    for vp in videos:
        duration = get_video_duration(vp)
        out_dir  = frames_dir / vp.stem
        n = extract_frames_ffmpeg(
            vp, out_dir,
            start_sec=SKIP_START_SEC,
            end_sec=max(SKIP_START_SEC + 10, duration - SKIP_END_SEC),
            prefix=vp.stem,
        )
        kept, removed = filter_blurry_frames(out_dir)
        log.info(f"  {vp.name}: {kept} frames kept ({removed} blurry removed)")
        total += kept
    return total


def count_all_frames(frames_dir: Path) -> int:
    return len(list(frames_dir.rglob("*.jpg")))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TBA 2026 frame scraper")
    p.add_argument("--event-key",         type=str,  default=None,
                   help="Target a specific event (e.g. 2026gacmp)")
    p.add_argument("--events",            type=int,  default=3,
                   help="Number of events to sample (default: 3)")
    p.add_argument("--matches-per-event", type=int,  default=5,
                   help="Matches to pull per event (default: 5)")
    p.add_argument("--target-frames",     type=int,  default=500,
                   help="Stop after reaching this many total frames (default: 500)")
    p.add_argument("--extract-only",      action="store_true",
                   help="Skip download, just extract frames from existing videos")
    p.add_argument("--videos-dir",        type=str,  default="data/videos",
                   help="Where to save/read .mp4 files")
    p.add_argument("--frames-dir",        type=str,  default="data/frames",
                   help="Where to save extracted frames")
    p.add_argument("--event-types",       type=int,  nargs="+",
                   default=list(OFFICIAL_EVENT_TYPES),
                   help="TBA event type codes (0=Regional 1=District 2=DistrictChamp 3=ChampDiv 4=ChampFinals)")
    p.add_argument("--seed",              type=int,  default=42,
                   help="Random seed for event/match sampling")
    return p.parse_args()


def main():
    args       = parse_args()
    frames_dir = Path(args.frames_dir)
    videos_dir = Path(args.videos_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # ---- extract-only mode ------------------------------------------------
    if args.extract_only:
        total = extract_only(videos_dir, frames_dir)
        log.info(f"\nDone. Total frames: {total}")
        return

    # ---- check existing frames --------------------------------------------
    existing = count_all_frames(frames_dir)
    log.info(f"Existing frames: {existing}  |  Target: {args.target_frames}")
    if existing >= args.target_frames:
        log.info("Already at target. Nothing to do.")
        return

    # ---- single event mode ------------------------------------------------
    if args.event_key:
        fake_event = {"key": args.event_key, "name": args.event_key,
                      "event_type": 0, "start_date": ""}
        n = process_event(fake_event, args.matches_per_event, frames_dir, videos_dir)
        log.info(f"\nNew frames: {n}  |  Total: {existing + n}")
        return

    # ---- multi-event sampling mode ----------------------------------------
    events = get_2026_events(event_types=set(args.event_types))
    if not events:
        log.error("No events returned. Check your TBA key and event type filters.")
        return

    # Sample across different event types for lighting/venue diversity
    # Group by type, then pick evenly across groups
    by_type: dict[int, list] = {}
    for e in events:
        by_type.setdefault(e["event_type"], []).append(e)

    sampled = []
    while len(sampled) < args.events and any(by_type.values()):
        for t in sorted(by_type.keys()):
            if by_type[t] and len(sampled) < args.events:
                sampled.append(by_type[t].pop(
                    random.randint(0, len(by_type[t]) - 1)
                ))

    log.info(f"\nSelected {len(sampled)} events:")
    for e in sampled:
        log.info(f"  [{e['event_type']}] {e.get('name')}  ({e['key']})")

    grand_total = existing
    for event in sampled:
        if grand_total >= args.target_frames:
            log.info(f"\nReached target of {args.target_frames} frames — stopping early.")
            break
        n = process_event(event, args.matches_per_event, frames_dir, videos_dir)
        grand_total += n
        log.info(f"\nRunning total: {grand_total} / {args.target_frames}")

    log.info(f"\n{'='*60}")
    log.info(f"Done.  Total frames in {frames_dir}: {count_all_frames(frames_dir)}")
    log.info(f"Next step: upload data/frames/ to Roboflow for bounding box labeling")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()