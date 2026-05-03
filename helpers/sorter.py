"""
HawkScan — Robot Image Renamer
Renames all images in a directory to robot_XXXX sequentially.
"""

from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DIRECTORY   = Path("dataset/raw/tests")
START_AT    = 4       # change this to continue from a specific number
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".webp"}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    files = sorted(f for f in DIRECTORY.iterdir() if f.suffix.lower() in IMAGE_EXTS)

    if not files:
        print(f"No images found in {DIRECTORY.resolve()}")
        return

    print(f"Found {len(files)} images. Renaming from robot_{START_AT:04d}…\n")

    for i, file in enumerate(files, start=START_AT):
        new_name = DIRECTORY / f"tests_{i:04d}{file.suffix.lower()}"
        if file == new_name:
            print(f"  [skip] {file.name} already correct")
            continue
        file.rename(new_name)
        print(f"  {file.name} → {new_name.name}")

    print(f"\n── Done. Renamed {len(files)} files.")
