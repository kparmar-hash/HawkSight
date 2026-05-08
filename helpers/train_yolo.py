"""
YOLOv8 Training Pipeline
========================
1. Takes a Roboflow dataset that only has a `train` split
2. Re-splits into train / val / test (75 / 15 / 10 by default)
3. Writes a clean data.yaml
4. Trains YOLOv8, then evaluates on the test set

Usage:
    pip install ultralytics scikit-learn pyyaml
    python train_yolov8.py --dataset ./your_dataset --model yolov8n.pt --epochs 50
"""

import argparse
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


# ─────────────────────────────────────────────
# 1. Split dataset
# ─────────────────────────────────────────────

def split_dataset(
    dataset_root: Path,
    output_root: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Reads images + labels from dataset_root/train/ and writes
    a fresh train/val/test split under output_root/.

    Returns a dict with split sizes for reporting.
    """
    src_images = dataset_root / "train" / "images"
    src_labels = dataset_root / "train" / "labels"

    # Collect all images that have a matching label file
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = sorted([
        p for p in src_images.iterdir()
        if p.suffix.lower() in image_exts
        and (src_labels / p.with_suffix(".txt").name).exists()
    ])

    if not all_images:
        raise FileNotFoundError(
            f"No paired image/label files found under {src_images}.\n"
            "Make sure labels/ contains .txt files with the same stem as the images."
        )

    print(f"Found {len(all_images)} paired image/label files.")

    # Two-pass split: first carve out test, then split remainder into train/val
    train_val, test_imgs = train_test_split(all_images, test_size=test_ratio, random_state=seed)
    adjusted_val = val_ratio / (1.0 - test_ratio)
    train_imgs, val_imgs = train_test_split(train_val, test_size=adjusted_val, random_state=seed)

    splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    for split_name, imgs in splits.items():
        img_out = output_root / "images" / split_name
        lbl_out = output_root / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            lbl_path = src_labels / img_path.with_suffix(".txt").name
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    sizes = {k: len(v) for k, v in splits.items()}
    print(f"Split → train: {sizes['train']}  val: {sizes['val']}  test: {sizes['test']}")
    return sizes


# ─────────────────────────────────────────────
# 2. Write data.yaml
# ─────────────────────────────────────────────

def write_yaml(dataset_root: Path, output_root: Path) -> Path:
    """
    Reads class names from the original data.yaml (Roboflow export)
    and writes a fresh one pointing at the new split folders.
    """
    src_yaml = dataset_root / "data.yaml"
    if not src_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {src_yaml}")

    with open(src_yaml) as f:
        original = yaml.safe_load(f)

    # Roboflow uses 'names' as either a list or dict — normalise to list
    names = original.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    new_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(names),
        "names": names,
    }

    out_path = output_root / "dataset.yaml"
    with open(out_path, "w") as f:
        yaml.dump(new_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote dataset.yaml → {out_path}")
    print(f"Classes ({len(names)}): {names}")
    return out_path


# ─────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────

def train(
    yaml_path: Path,
    model_weights: str = "hawk_robot_model.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/train",
    name: str = "exp",
):
    model = YOLO(model_weights)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=10,          # early-stop if val/mAP50 doesn't improve for 10 epochs
        save=True,
        plots=True,
        verbose=True,
    )
    return results


# ─────────────────────────────────────────────
# 4. Evaluate on test set
# ─────────────────────────────────────────────

def evaluate(yaml_path: Path, weights_path: Path, imgsz: int = 640):
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(yaml_path),
        split="test",          # explicitly use the test split
        imgsz=imgsz,
        plots=True,
        verbose=True,
    )
    print("\n── Test-set metrics ──────────────────────────")
    print(f"  mAP@0.50      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.50:0.95 : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")
    print("─────────────────────────────────────────────\n")
    return metrics


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 split + train + evaluate pipeline")
    parser.add_argument("--dataset",  type=str, required=True,
                        help="Path to your Roboflow dataset root (contains train/ and data.yaml)")
    parser.add_argument("--output",   type=str, default="dataset_split",
                        help="Where to write the re-split dataset (default: dataset_split/)")
    parser.add_argument("--model",    type=str, default="yolov8n.pt",
                        help="YOLOv8 weights to start from (n/s/m/l/x, or a .pt path)")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--val-ratio",  type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    output_root  = Path(args.output)

    # Step 1 — split
    split_dataset(
        dataset_root, output_root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # Step 2 — yaml
    yaml_path = write_yaml(dataset_root, output_root)

    # Step 3 — train
    train(
        yaml_path=yaml_path,
        model_weights=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    # Step 4 — evaluate on test set using best weights
    best_weights = Path("runs/train/exp/weights/best.pt")
    if not best_weights.exists():
        # fallback: look for the most recent training run
        candidates = sorted(Path("runs/train").glob("*/weights/best.pt"))
        if candidates:
            best_weights = candidates[-1]
        else:
            print("Could not find best.pt — skipping test evaluation.")
            return

    evaluate(yaml_path, best_weights, imgsz=args.imgsz)


if __name__ == "__main__":
    main()