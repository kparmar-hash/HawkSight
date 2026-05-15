"""
YouTube Video Downloader
========================
Downloads a YouTube video as MP4 to tests/videos/.

Usage:
    pip install yt-dlp
    python download_video.py <youtube_url>

Example:
    python download_video.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
"""

import argparse
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("tests/videos")


def download(url: str, out_dir: Path = OUTPUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(out_dir / "%(title)s.%(ext)s"),
        url,
    ]

    print(f"Downloading: {url}")
    print(f"Saving to  : {out_dir.resolve()}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Download failed. Make sure yt-dlp is installed: pip install yt-dlp")
        sys.exit(1)

    print(f"\nDone. File saved to {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Download a YouTube video to tests/videos/")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--out", default=str(OUTPUT_DIR), help="Output directory (default: tests/videos)")
    args = parser.parse_args()

    download(args.url, Path(args.out))


if __name__ == "__main__":
    main()