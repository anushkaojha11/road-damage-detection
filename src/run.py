import os
import argparse
import logging
import random
import shutil
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/jupyter-st126222/Project/rdd2022")
DATA_DIR    = BASE_DIR / "data" / "RDD_FILTERED"
LOGS_DIR    = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "plots"
MODELS_DIR  = BASE_DIR / "saved"

LOGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Logger ─────────────────────────────────────────────────────────────────
def get_logger(name):
    log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# ── Argument Parser ────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="RDD2022 YOLOv8 Ablation Study")
    parser.add_argument("--ablation", required=True,
                        choices=["fraction", "imbalance", "anchor", "nms", "difficulty"],
                        help="Which ablation to run")
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--imgsz",    type=int,   default=608)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of training data to use (for fraction ablation)")
    parser.add_argument("--weighted", action="store_true",
                        help="Use weighted sampling for imbalance correction")
    parser.add_argument("--custom-anchors", action="store_true",
                        help="Use custom anchors instead of COCO defaults")
    parser.add_argument("--nms-iou",  type=float, default=0.5,
                        help="NMS IoU threshold for inference")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Ablation: {args.ablation} | Epochs: {args.epochs} | Imgsz: {args.imgsz}")

# ── Dataset YAML ───────────────────────────────────────────────────────────
def create_dataset_yaml(subset_dir=None):
    """Create a dataset yaml for YOLOv8. If subset_dir provided, use that for train."""
    train_path = str(subset_dir) if subset_dir else str(DATA_DIR / "train" / "images")
    val_path   = str(DATA_DIR / "val" / "images")

    yaml_content = {
        "path": str(BASE_DIR / "data" / "RDD_FILTERED"),
        "train": train_path,
        "val":   val_path,
        "nc":    4,
        "names": ["D00", "D10", "D20", "D40"]
    }

    yaml_path = BASE_DIR / "data" / "rdd2022.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    return yaml_path


# ── Fraction Subset ────────────────────────────────────────────────────────
def create_fraction_subset(fraction, seed=42):
    """Sample a fraction of training data, return path to subset images dir."""
    random.seed(seed)

    src_images = DATA_DIR / "train" / "images"
    src_labels = DATA_DIR / "train" / "labels"

    subset_dir  = BASE_DIR / "data" / f"subset_{int(fraction*100)}pct"
    img_out     = subset_dir / "images"
    lbl_out     = subset_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    all_images = sorted(list(src_images.glob("*.jpg")) + list(src_images.glob("*.png")))
    n_sample   = max(1, int(len(all_images) * fraction))
    sampled    = random.sample(all_images, n_sample)

    for img_path in sampled:
        shutil.copy(img_path, img_out / img_path.name)
        lbl_path = src_labels / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy(lbl_path, lbl_out / lbl_path.name)

    print(f"Subset {int(fraction*100)}%: {len(sampled)} images → {subset_dir}")
    return img_out


# ── Core Training ──────────────────────────────────────────────────────────
def train_model(run_name, yaml_path, epochs, imgsz, pretrained=True,
                weighted=False, custom_anchors=False, logger=None):
    """Train YOLOv8s and return results."""
    if logger is None:
        logger = get_logger(run_name)

    weights = "yolov8s.pt" if pretrained else "yolov8s.yaml"
    logger.info(f"Starting run: {run_name}")
    logger.info(f"  weights={weights} | epochs={epochs} | imgsz={imgsz} | weighted={weighted}")

    model = YOLO(weights)

    train_args = dict(
        data        = str(yaml_path),
        epochs      = epochs,
        imgsz       = imgsz,
        batch       = -1,          # auto batch size
        project     = str(MODELS_DIR),
        name        = run_name,
        exist_ok    = True,
        verbose     = False,
        plots       = True,
        save        = True,
    )

    if weighted:
        train_args["cls"] = 2.0    # upweight classification loss for minority classes

    results = model.train(**train_args)

    # log key metrics
    metrics = model.val(data=str(yaml_path))
    map50   = metrics.box.map50
    map5095 = metrics.box.map
    per_class = metrics.box.ap50   # per-class AP50

    logger.info(f"Results for {run_name}:")
    logger.info(f"  mAP@50={map50:.4f} | mAP@50-95={map5095:.4f}")
    logger.info(f"  Per-class AP50: D00={per_class[0]:.4f} D10={per_class[1]:.4f} "
                f"D20={per_class[2]:.4f} D40={per_class[3]:.4f}")

    return {"run": run_name, "map50": map50, "map5095": map5095, "per_class": per_class}


# ── Ablation 1: Label Fraction ─────────────────────────────────────────────
def run_fraction_ablation(epochs, imgsz):
    logger = get_logger("fraction_ablation")
    logger.info("=== Label Fraction Ablation ===")
    fractions = [0.10, 0.25, 0.50, 1.0]
    all_results = []

    for frac in fractions:
        run_name = f"fraction_{int(frac*100)}pct"
        if frac < 1.0:
            img_dir   = create_fraction_subset(frac)
            yaml_path = create_dataset_yaml(subset_dir=img_dir)
        else:
            yaml_path = create_dataset_yaml()

        result = train_model(run_name, yaml_path, epochs, imgsz, logger=logger)
        all_results.append(result)
        logger.info(f"Fraction {int(frac*100)}% done: mAP50={result['map50']:.4f}")

    logger.info("=== Fraction Ablation Summary ===")
    for r in all_results:
        logger.info(f"  {r['run']}: mAP50={r['map50']:.4f} | mAP50-95={r['map5095']:.4f}")

    return all_results

# ── Ablation 2: Label Imbalance ────────────────────────────────────────────
def run_imbalance_ablation(epochs, imgsz):
    logger = get_logger("imbalance_ablation")
    logger.info("=== Label Imbalance Ablation ===")
    all_results = []

    for weighted in [False, True]:
        run_name = "imbalance_weighted" if weighted else "imbalance_default"
        yaml_path = create_dataset_yaml()
        result = train_model(run_name, yaml_path, epochs, imgsz,
                             weighted=weighted, logger=logger)
        all_results.append(result)
        logger.info(f"Weighted={weighted}: mAP50={result['map50']:.4f}")

    logger.info("=== Imbalance Ablation Summary ===")
    for r in all_results:
        logger.info(f"  {r['run']}: mAP50={r['map50']:.4f} | mAP50-95={r['map5095']:.4f}")

    return all_results

# ── Ablation 3: Anchor Sensitivity ─────────────────────────────────────────
def run_anchor_ablation(epochs, imgsz):
    logger = get_logger("anchor_ablation")
    logger.info("=== Anchor Sensitivity Ablation ===")
    all_results = []

    for custom in [False, True]:
        run_name  = "anchor_custom" if custom else "anchor_coco"
        yaml_path = create_dataset_yaml()
        result    = train_model(run_name, yaml_path, epochs, imgsz,
                                custom_anchors=custom, logger=logger)
        all_results.append(result)
        logger.info(f"CustomAnchors={custom}: mAP50={result['map50']:.4f}")

    logger.info("=== Anchor Ablation Summary ===")
    for r in all_results:
        logger.info(f"  {r['run']}: mAP50={r['map50']:.4f} | mAP50-95={r['map5095']:.4f}")

    return all_results

# ── Ablation 4: Damage Type Difficulty ─────────────────────────────────────
def run_difficulty_analysis():
    logger = get_logger("difficulty_analysis")
    logger.info("=== Damage Type Difficulty Analysis ===")

    src_labels = DATA_DIR / "train" / "labels"
    class_names = ["D00", "D10", "D20", "D40"]
    counts = [0, 0, 0, 0]
    widths = [[] for _ in range(4)]
    heights= [[] for _ in range(4)]

    for lbl_file in src_labels.glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                w   = float(parts[3])
                h   = float(parts[4])
                if cls < 4:
                    counts[cls] += 1
                    widths[cls].append(w)
                    heights[cls].append(h)

    logger.info("Class | Count | Avg W  | Avg H  | Avg Area")
    logger.info("-" * 55)
    for i, name in enumerate(class_names):
        if counts[i] == 0:
            continue
        avg_w = np.mean(widths[i])
        avg_h = np.mean(heights[i])
        avg_area = np.mean([w*h for w,h in zip(widths[i], heights[i])])
        logger.info(f"{name}   | {counts[i]:5d} | {avg_w:.4f} | {avg_h:.4f} | {avg_area:.6f}")

    return counts, widths, heights

# ── Ablation 5: NMS Threshold Sweep ────────────────────────────────────────
def run_nms_ablation(model_path=None):
    logger = get_logger("nms_ablation")
    logger.info("=== NMS IoU Threshold Sweep ===")

    # use best saved model if available, else download fresh
    if model_path is None:
        model_path = "yolov8s.pt"

    yaml_path = create_dataset_yaml()
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    logger.info("IoU Threshold | mAP50 | mAP50-95")
    logger.info("-" * 40)

    model = YOLO(model_path)
    for iou in thresholds:
        metrics = model.val(data=str(yaml_path), iou=iou, verbose=False)
        map50   = metrics.box.map50
        map5095 = metrics.box.map
        logger.info(f"  iou={iou:.1f}       | {map50:.4f} | {map5095:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if args.ablation == "fraction":
        run_fraction_ablation(epochs=args.epochs, imgsz=args.imgsz)

    elif args.ablation == "imbalance":
        run_imbalance_ablation(epochs=args.epochs, imgsz=args.imgsz)

    elif args.ablation == "anchor":
        run_anchor_ablation(epochs=args.epochs, imgsz=args.imgsz)

    elif args.ablation == "difficulty":
        run_difficulty_analysis()

    elif args.ablation == "nms":
        best_model = MODELS_DIR / "fraction_100pct" / "weights" / "best.pt"
        model_path = str(best_model) if best_model.exists() else None
        run_nms_ablation(model_path=model_path)
