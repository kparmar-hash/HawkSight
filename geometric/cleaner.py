from PIL import Image
from pathlib import Path
import hashlib

def phash(img_path: Path, size: int = 8) -> str:
    img = Image.open(img_path).convert("L").resize((size, size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p > avg else "0" for p in pixels)

def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))

def dedup_frames(frames_dir: str, threshold: int = 8):
    frames = list(Path(frames_dir).rglob("*.jpg"))
    print(f"Scanning {len(frames)} frames...")

    hashes = {f: phash(f) for f in frames}
    removed = set()

    for i, (f1, h1) in enumerate(hashes.items()):
        if f1 in removed:
            continue
        for f2, h2 in list(hashes.items())[i+1:]:
            if f2 not in removed and hamming(h1, h2) <= threshold:
                f2.unlink()
                removed.add(f2)

    print(f"Removed {len(removed)} duplicates")
    print(f"Remaining: {len(frames) - len(removed)} frames")

dedup_frames("dataset/raw/2026ohmv")