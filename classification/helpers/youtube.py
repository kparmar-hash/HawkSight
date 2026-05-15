"""
yt_frame_extractor.py
─────────────────────
Downloads one or more YouTube videos and extracts one frame per minute,
saving them as numbered JPEGs inside a dedicated sub-folder per video.

Usage
-----
Single video:
    python yt_frame_extractor.py "https://youtu.be/VIDEO_ID" --out ./frames

Multiple videos (pass URL list file):
    python yt_frame_extractor.py --url-file urls.txt --out ./frames

Options:
    --out       Root output directory  (default: ./frames)
    --interval  Seconds between frames (default: 60  → 1 frame/min)
    --quality   JPEG quality 1-95     (default: 90)
    --keep-video  Don't delete the downloaded .mp4 after extraction
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path




# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    """Turn a video title into a safe folder name."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:80] or "video"


def get_video_title(url: str) -> str:
    """Ask yt-dlp for the video title without downloading anything."""
    result = subprocess.run(
        ["yt-dlp", "--get-title", "--no-playlist", url],
        capture_output=True, text=True
    )
    title = result.stdout.strip().splitlines()[0] if result.returncode == 0 else "video"
    return sanitize(title)


def download_video(url: str, dest: Path) -> Path:
    """Download best-quality mp4 to dest/source.mp4 and return the path."""
    video_path = dest / "source.mp4"
    if video_path.exists():
        print(f"  [cache] {video_path} already exists, skipping download.")
        return video_path

    print(f"  Downloading: {url}")
    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", str(video_path),
            "--no-playlist",
            url,
        ],
        check=True,
    )
    return video_path


def get_duration(video_path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    interval_seconds: int = 60,
    jpeg_quality: int = 90,
) -> int:
    """
    Extract one frame every `interval_seconds` from video_path using ffmpeg.
    Saves frame_000001.jpg, frame_000002.jpg, … inside frames_dir.
    Works with any codec (H.264, H.265, VP9, AV1, etc.).

    Returns the number of frames saved.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    duration = get_duration(video_path)
    expected = max(1, int(duration / interval_seconds))
    print(f"  Duration: {duration/60:.1f} min | "
          f"Interval: {interval_seconds}s | ~{expected} frames expected")

    # ffmpeg: -vf fps=1/N pulls exactly 1 frame every N seconds
    # -q:v maps to JPEG quality (2=best … 31=worst; invert from 1-95 scale)
    q = max(2, min(31, int(2 + (95 - jpeg_quality) * 29 / 94)))

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps=1/{interval_seconds}",
        "-q:v", str(q),
        "-vsync", "vfr",
        str(frames_dir / "frame_%06d.jpg"),
    ]

    print(f"  Running ffmpeg …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])   # last 2 KB of stderr for diagnosis
        raise RuntimeError("ffmpeg extraction failed — see output above.")

    saved = len(list(frames_dir.glob("frame_*.jpg")))
    return saved


def process_url(
    url: str,
    root_out: Path,
    interval: int,
    quality: int,
    keep_video: bool,
) -> None:
    title = get_video_title(url)
    video_dir = root_out / title
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Title  : {title}")
    print(f"  Output : {video_dir}")

    video_path = download_video(url, video_dir)
    frames_dir = video_dir / "frames"

    saved = extract_frames(video_path, frames_dir, interval, quality)
    print(f"  ✓ Saved {saved} frames → {frames_dir}")

    if not keep_video:
        video_path.unlink(missing_ok=True)
        print(f"  ✓ Removed {video_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract 1 frame/minute from YouTube videos."
    )
    parser.add_argument("urls", nargs="*", help="One or more YouTube URLs")
    parser.add_argument("--url-file", help="Text file with one URL per line")
    parser.add_argument("--out", default="./frames", help="Root output directory")
    parser.add_argument("--interval", type=int, default=20,
                        help="Seconds between captured frames (default: 60)")
    parser.add_argument("--quality", type=int, default=90,
                        help="JPEG quality 1-95 (default: 90)")
    parser.add_argument("--keep-video", action="store_true",
                        help="Keep downloaded .mp4 files")
    args = parser.parse_args()

    urls = list(args.urls)
    if args.url_file:
        with open(args.url_file) as f:
            urls += [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not urls:
        parser.print_help()
        sys.exit(1)

    root_out = Path(args.out)
    root_out.mkdir(parents=True, exist_ok=True)

    print(f"Output root : {root_out.resolve()}")
    print(f"Interval    : every {args.interval}s")
    print(f"URLs        : {len(urls)}")

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        try:
            process_url(url, root_out, args.interval, args.quality, args.keep_video)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Download failed: {e}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n{'─'*60}")
    print("Done.")


if __name__ == "__main__":
    main()