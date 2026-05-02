import os
import time
import requests
from pathlib import Path
from duckduckgo_search import DDGS

# ── Config ────────────────────────────────────────────────────────────────────
QUERY = "FRC 2026 robot"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw" / "robots_static"
MAX_IMAGES = 20

# ── Main ──────────────────────────────────────────────────────────────────────
def get_next_index(directory: Path) -> int:
    """Finds the highest robot_XXXX index in the directory."""
    existing_files = list(directory.glob("robot_*.*"))
    max_idx = 0
    for f in existing_files:
        try:
            num = int(f.stem.split('_')[1])
            if num > max_idx:
                max_idx = num
        except (IndexError, ValueError):
            pass
    return max_idx + 1

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    start_idx = get_next_index(OUTPUT_DIR)
    print(f"Starting to save from index: robot_{start_idx:04d}")
    print(f"Searching DuckDuckGo Images for: '{QUERY}'\n")
    
    with DDGS() as ddgs:
        results = ddgs.images(
            keywords=QUERY,
            max_results=MAX_IMAGES,
            safesearch="off"
        )
        
        idx = start_idx
        for res in results:
            url = res.get('image')
            if not url: 
                continue
                
            try:
                print(f"Downloading {url[:80]}...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Determine extension from content-type or URL
                content_type = response.headers.get('content-type', '')
                if 'image/png' in content_type:
                    ext = '.png'
                elif 'image/jpeg' in content_type:
                    ext = '.jpg'
                elif 'image/webp' in content_type:
                    ext = '.webp'
                else:
                    ext = '.jpg' # Fallback
                
                filename = OUTPUT_DIR / f"robot_{idx:04d}{ext}"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"  → Saved to {filename.name}\n")
                idx += 1
                time.sleep(1) # Respectful delay
                
            except Exception as e:
                print(f"  → [Error] Failed to download: {e}\n")
