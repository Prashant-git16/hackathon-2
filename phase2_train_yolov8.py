"""
ZeroDefect AI — Phase 2: YOLOv8 Training
Run this on Google Colab (GPU runtime) for best speed.
Training time: ~20-40 min on T4 GPU for 50 epochs.

Colab setup cells to run first:
    !pip install ultralytics -q
    !pip install albumentations -q
    from google.colab import drive
    drive.mount('/content/drive')

Then upload yolo_dataset/ to your Drive and set DATA_YAML below.
"""



from ultralytics import YOLO
from pathlib import Path
import yaml, json, os, shutil
from datetime import datetime

# ── CONFIG — edit these paths ─────────────────────────────────────────────────
DATA_YAML   = "./yolo_dataset/data.yaml"   # path to your data.yaml
MODEL_BASE  = "yolov8n.pt"                 # start with nano for speed; use yolov8s.pt for accuracy
EPOCHS      = 50                           # 50 is enough; bump to 100 if time allows
IMG_SIZE    = 512
BATCH_SIZE  = 16                           # reduce to 8 if OOM on Colab
PROJECT_DIR = "./runs/zerodefect"
RUN_NAME    = f"cast_v1_{datetime.now().strftime('%H%M')}"
DEVICE      = 0                            # 0 = first GPU; 'cpu' for local testing

# ── VERIFY CONFIG ─────────────────────────────────────────────────────────────
def check_config():
    yaml_path = Path(DATA_YAML)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {yaml_path}\n"
            "Run phase1_prepare_dataset.py first."
        )
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    print(f"[OK] data.yaml loaded")
    print(f"     Classes : {cfg['names']}")
    print(f"     nc      : {cfg['nc']}")
    print(f"     path    : {cfg['path']}")

    # Count images
    base = Path(cfg["path"])
    for split in ["train", "val", "test"]:
        imgs = list((base / "images" / split).glob("*"))
        print(f"     {split:5s}   : {len(imgs)} images")


# ── TRAINING ─────────────────────────────────────────────────────────────────
def train():
    print(f"\n{'='*60}")
    print("  ZeroDefect AI — Phase 2: YOLOv8 Training")
    print(f"{'='*60}")
    check_config()

    # Load pretrained YOLOv8 (downloads automatically on first run)
    model = YOLO(MODEL_BASE)
    print(f"\n[OK] Loaded base model: {MODEL_BASE}")

    # ── Training arguments ────────────────────────────────────────────────
    results = model.train(
        data        = DATA_YAML,
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        device      = DEVICE,
        project     = PROJECT_DIR,
        name        = RUN_NAME,

        # Optimiser
        optimizer   = "AdamW",
        lr0         = 0.001,
        lrscheduler = "cosine",
        warmup_epochs = 3,

        # Augmentation (on top of Albumentations pipeline)
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 10,
        translate   = 0.1,
        scale       = 0.5,
        fliplr      = 0.5,
        flipud      = 0.2,
        mosaic      = 0.5,
        mixup       = 0.1,

        # Saving
        save        = True,
        save_period = 10,   # checkpoint every 10 epochs
        patience    = 20,   # early stopping

        # Logging
        plots       = True,
        verbose     = True,
    )

    # ── Validation ────────────────────────────────────────────────────────
    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    print(f"\n[OK] Training complete!")
    print(f"     Best weights → {best_weights}")

    # Run validation on test set
    model_best = YOLO(str(best_weights))
    val_results = model_best.val(
        data   = DATA_YAML,
        split  = "test",
        imgsz  = IMG_SIZE,
        device = DEVICE,
    )

    # Extract key metrics
    metrics = {
        "mAP50"    : float(val_results.box.map50),
        "mAP50_95" : float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall"   : float(val_results.box.mr),
        "weights"  : str(best_weights.resolve()),
        "run_name" : RUN_NAME,
    }

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"  mAP@50       : {metrics['mAP50']:.4f}  (target > 0.90)")
    print(f"  mAP@50-95    : {metrics['mAP50_95']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"{'='*60}\n")

    # Save metrics for later phases
    metrics_path = Path(PROJECT_DIR) / RUN_NAME / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved → {metrics_path}")

    if metrics["mAP50"] >= 0.90:
        print("[  ] mAP target MET — ready for Phase 3!")
    else:
        print(f"[!] mAP = {metrics['mAP50']:.3f} < 0.90")
        print("    Try: more epochs, yolov8s.pt base model, or more data augmentation.")

    print(f"\n  COPY THIS PATH FOR PHASE 4:")
    print(f"  WEIGHTS = '{best_weights.resolve()}'")
    return metrics


# ── EXPORT (optional - for faster CPU inference) ──────────────────────────────
def export_onnx(weights_path: str):
    """Export to ONNX for faster CPU inference at the hackathon."""
    model = YOLO(weights_path)
    model.export(format="onnx", imgsz=IMG_SIZE, simplify=True)
    print(f"[OK] ONNX model exported alongside {weights_path}")


if __name__ == "__main__":
    train()
    # Uncomment to export after training:
    # export_onnx(f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
