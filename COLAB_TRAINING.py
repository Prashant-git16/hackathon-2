# ZeroDefect AI — Google Colab Training Notebook
# Copy each cell block into a Colab cell and run in order.
# Runtime: GPU (T4 or better) — Runtime > Change runtime type > GPU

# ═══════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ═══════════════════════════════════════════════════════════
"""
!pip install ultralytics albumentations -q
print("Done!")
"""

# ═══════════════════════════════════════════════════════════
# CELL 2 — Mount Drive & upload dataset
# ═══════════════════════════════════════════════════════════
"""
from google.colab import drive, files
drive.mount('/content/drive')

# Option A: Upload yolo_dataset.zip directly
# uploaded = files.upload()   # then unzip below

# Option B: Use dataset already in Drive
# !cp -r '/content/drive/MyDrive/yolo_dataset' /content/yolo_dataset

# If you uploaded a zip:
# !unzip yolo_dataset.zip -d /content/
"""

# ═══════════════════════════════════════════════════════════
# CELL 3 — Verify dataset
# ═══════════════════════════════════════════════════════════
"""
from pathlib import Path
import yaml

data_yaml = Path('/content/yolo_dataset/data.yaml')
assert data_yaml.exists(), "data.yaml not found!"

with open(data_yaml) as f:
    cfg = yaml.safe_load(f)

print("Classes:", cfg['names'])
for split in ['train','val','test']:
    imgs = list(Path(f"/content/yolo_dataset/images/{split}").glob("*"))
    print(f"  {split}: {len(imgs)} images")
"""

# ═══════════════════════════════════════════════════════════
# CELL 4 — Train YOLOv8
# ═══════════════════════════════════════════════════════════
"""
from ultralytics import YOLO

model = YOLO('yolov8s.pt')   # s = small, good balance of speed/accuracy

results = model.train(
    data      = '/content/yolo_dataset/data.yaml',
    epochs    = 30,
    imgsz     = 512,
    batch     = 16,
    device    = 0,            # GPU
    optimizer = 'AdamW',
    lr0       = 0.001,
    warmup_epochs = 3,
    patience  = 20,
    hsv_h     = 0.015,
    hsv_s     = 0.7,
    hsv_v     = 0.4,
    degrees   = 10,
    scale     = 0.5,
    fliplr    = 0.5,
    mosaic    = 0.5,
    save      = True,
    plots     = True,
    project   = '/content/runs',
    name      = 'zerodefect_v1',
)
print("Training complete!")
"""

# ═══════════════════════════════════════════════════════════
# CELL 5 — Validate on test set
# ═══════════════════════════════════════════════════════════
"""
best_weights = '/content/runs/zerodefect_v1/weights/best.pt'
model_best = YOLO(best_weights)

val = model_best.val(
    data  = '/content/yolo_dataset/data.yaml',
    split = 'test',
    imgsz = 512,
)

print(f"mAP@50:    {val.box.map50:.4f}  (target > 0.90)")
print(f"mAP@50-95: {val.box.map:.4f}")
print(f"Precision: {val.box.mp:.4f}")
print(f"Recall:    {val.box.mr:.4f}")
"""

# ═══════════════════════════════════════════════════════════
# CELL 6 — Save weights to Drive
# ═══════════════════════════════════════════════════════════
"""
import shutil

# Save to Drive
shutil.copy(
    '/content/runs/zerodefect_v1/weights/best.pt',
    '/content/drive/MyDrive/zerodefect_best.pt'
)
print("Weights saved to Google Drive!")

# Also download directly
from google.colab import files
files.download('/content/runs/zerodefect_v1/weights/best.pt')
"""

# ═══════════════════════════════════════════════════════════
# CELL 7 — Quick test inference
# ═══════════════════════════════════════════════════════════
"""
import cv2
from pathlib import Path
import random

model = YOLO('/content/runs/zerodefect_v1/weights/best.pt')

# Pick a random test image
test_imgs = list(Path('/content/yolo_dataset/images/test').glob('*.jpg'))
test_img = random.choice(test_imgs)

results = model(str(test_img), conf=0.45)
results[0].show()
print(f"Image: {test_img.name}")
print(f"Detections: {len(results[0].boxes)}")
"""
