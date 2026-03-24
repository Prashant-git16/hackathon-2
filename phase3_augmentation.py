"""
ZeroDefect AI — Phase 3: Advanced Augmentation Pipeline
Two parts:
  A) Albumentations pipeline (used during training via callbacks)
  B) Synthetic defect generator — creates fake defect images from OK images
     using OpenCV texture overlays (addresses class imbalance & rare defects)

Run standalone to generate synthetic defects:
    python phase3_augmentation.py --ok_dir ./yolo_dataset/images/train \
                                   --out_dir ./yolo_dataset/synthetic \
                                   --n 200
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import argparse, random, shutil
from tqdm import tqdm

# ── ALBUMENTATIONS PIPELINE ───────────────────────────────────────────────────

def get_train_transforms(img_size: int = 512):
    """
    Heavy augmentation pipeline for training.
    Covers: lighting, noise, blur, geometric — all common in real factory conditions.
    """
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size,
                            scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),

        # ── Lighting & color (simulate factory lighting changes) ──
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.8),

        # ── Noise (sensor noise, compression artifacts) ──
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.5),

        # ── Blur (motion blur from conveyor, out-of-focus) ──
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.Defocus(radius=(1, 3), p=1.0),
        ], p=0.3),

        # ── Geometric ──
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),

        # ── Occlusion / cutout (simulate partial obstruction) ──
        A.CoarseDropout(max_holes=4, max_height=40, max_width=40,
                        min_holes=1, min_height=10, min_width=10,
                        fill_value=128, p=0.3),

        # ── Image quality degradation ──
        A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


def get_val_transforms(img_size: int = 512):
    """Minimal transforms for validation — only resize + normalize."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


# ── SYNTHETIC DEFECT GENERATOR ────────────────────────────────────────────────

class SyntheticDefectGenerator:
    """
    Generates synthetic defective casting images from OK images.
    Applies realistic surface defects using OpenCV texture operations.
    Defect types: scratch, crack, pit/void, discoloration, edge chip
    """

    DEFECT_TYPES = ["scratch", "crack", "pit", "discoloration", "edge_chip", "blob"]

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    # ── Individual defect generators ─────────────────────────────────────

    def _add_scratch(self, img: np.ndarray) -> np.ndarray:
        """Long thin linear scratch across surface."""
        h, w = img.shape[:2]
        img = img.copy()
        num_scratches = random.randint(1, 4)
        for _ in range(num_scratches):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            angle = random.uniform(-30, 30)
            length = random.randint(w // 4, w // 2)
            x2 = int(x1 + length * np.cos(np.radians(angle)))
            y2 = int(y1 + length * np.sin(np.radians(angle)))
            thickness = random.randint(1, 3)
            color_shift = random.choice([-60, -40, 60, 80])  # darker or lighter
            color = (
                max(0, min(255, int(img[y1 % h, x1 % w, 0]) + color_shift)),
                max(0, min(255, int(img[y1 % h, x1 % w, 1]) + color_shift)),
                max(0, min(255, int(img[y1 % h, x1 % w, 2]) + color_shift)),
            )
            cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        return img

    def _add_crack(self, img: np.ndarray) -> np.ndarray:
        """Irregular crack pattern (polyline with noise)."""
        h, w = img.shape[:2]
        img = img.copy()
        x = random.randint(w // 4, 3 * w // 4)
        y = random.randint(h // 4, 3 * h // 4)
        points = [(x, y)]
        length = random.randint(50, 150)
        angle = random.uniform(0, 360)
        for _ in range(length // 5):
            angle += random.gauss(0, 20)
            dx = int(5 * np.cos(np.radians(angle)))
            dy = int(5 * np.sin(np.radians(angle)))
            nx, ny = points[-1][0] + dx, points[-1][1] + dy
            nx = max(0, min(w - 1, nx))
            ny = max(0, min(h - 1, ny))
            points.append((nx, ny))
        pts = np.array(points, np.int32)
        cv2.polylines(img, [pts], False, (20, 20, 20), random.randint(1, 3), cv2.LINE_AA)
        return img

    def _add_pit(self, img: np.ndarray) -> np.ndarray:
        """Small dark circular pits/voids on surface."""
        h, w = img.shape[:2]
        img = img.copy()
        num_pits = random.randint(2, 8)
        for _ in range(num_pits):
            cx = random.randint(20, w - 20)
            cy = random.randint(20, h - 20)
            r  = random.randint(3, 15)
            # Dark center with slight gradient
            cv2.circle(img, (cx, cy), r, (15, 15, 15), -1)
            cv2.circle(img, (cx, cy), r + 2, (60, 60, 60), 1)
        return img

    def _add_discoloration(self, img: np.ndarray) -> np.ndarray:
        """Blotchy discoloration patches (rust, oxide)."""
        h, w = img.shape[:2]
        img = img.copy().astype(np.float32)
        num_patches = random.randint(1, 3)
        for _ in range(num_patches):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            rx = random.randint(20, 80)
            ry = random.randint(20, 80)
            # Random tint: rust-brown, white oxide, grey
            tint = random.choice([
                np.array([0.0, -20.0, -40.0]),   # rust (reduce green/blue)
                np.array([30.0, 30.0, 30.0]),     # bright spot
                np.array([-30.0, -30.0, -30.0]),  # dark patch
            ])
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (cx, cy), (rx, ry), random.uniform(0, 360), 0, 360, 1.0, -1)
            # Feather the mask
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            for c in range(3):
                img[:, :, c] += tint[c] * mask
        return np.clip(img, 0, 255).astype(np.uint8)

    def _add_edge_chip(self, img: np.ndarray) -> np.ndarray:
        """Missing material at edge (chip)."""
        h, w = img.shape[:2]
        img = img.copy()
        edge = random.choice(["top", "bottom", "left", "right"])
        chip_w = random.randint(20, 80)
        chip_h = random.randint(10, 40)
        if edge == "top":
            x = random.randint(0, w - chip_w)
            pts = np.array([[x, 0], [x + chip_w, 0],
                             [x + chip_w // 2, chip_h]], np.int32)
        elif edge == "bottom":
            x = random.randint(0, w - chip_w)
            pts = np.array([[x, h], [x + chip_w, h],
                             [x + chip_w // 2, h - chip_h]], np.int32)
        elif edge == "left":
            y = random.randint(0, h - chip_h)
            pts = np.array([[0, y], [0, y + chip_h],
                             [chip_w, y + chip_h // 2]], np.int32)
        else:
            y = random.randint(0, h - chip_h)
            pts = np.array([[w, y], [w, y + chip_h],
                             [w - chip_w, y + chip_h // 2]], np.int32)
        # Fill with background color (sample from near that edge)
        bg = img[min(5, h - 1), min(5, w - 1)].tolist()
        cv2.fillPoly(img, [pts], bg)
        return img

    def _add_blob(self, img: np.ndarray) -> np.ndarray:
        """Foreign material inclusion (dark or bright blob)."""
        h, w = img.shape[:2]
        img = img.copy()
        cx = random.randint(30, w - 30)
        cy = random.randint(30, h - 30)
        axes = (random.randint(5, 25), random.randint(5, 25))
        angle = random.randint(0, 180)
        color_val = random.choice([10, 30, 200, 230])  # dark or bright
        color = (color_val, color_val, color_val)
        cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color, -1)
        return img

    # ── Main generate function ────────────────────────────────────────────

    def generate(self, img: np.ndarray,
                 defect_types: list = None,
                 num_defects: int = None) -> np.ndarray:
        """Apply 1-3 random defect types to an image."""
        if defect_types is None:
            defect_types = random.sample(
                self.DEFECT_TYPES,
                k=num_defects or random.randint(1, 3)
            )
        fn_map = {
            "scratch"       : self._add_scratch,
            "crack"         : self._add_crack,
            "pit"           : self._add_pit,
            "discoloration" : self._add_discoloration,
            "edge_chip"     : self._add_edge_chip,
            "blob"          : self._add_blob,
        }
        result = img.copy()
        for dtype in defect_types:
            result = fn_map[dtype](result)
        return result

    def generate_batch(self,
                       ok_dir: Path,
                       out_dir: Path,
                       n: int = 200,
                       also_copy_labels: bool = True):
        """
        Generate n synthetic defect images from OK images in ok_dir.
        Saves images + YOLO labels to out_dir.
        """
        exts = {".jpg", ".jpeg", ".png"}
        ok_images = [f for f in ok_dir.rglob("*") if f.suffix.lower() in exts
                     and "def_" not in f.name]

        if not ok_images:
            print(f"[WARN] No OK images found in {ok_dir}")
            return

        out_img_dir = out_dir / "images"
        out_lbl_dir = out_dir / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Generating {n} synthetic defects from {len(ok_images)} OK images...")
        generated = 0

        for i in tqdm(range(n)):
            src = random.choice(ok_images)
            img = cv2.imread(str(src))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            defect_img = self.generate(img)
            defect_img_bgr = cv2.cvtColor(defect_img, cv2.COLOR_RGB2BGR)

            out_name = f"synth_{i:04d}_{src.stem}.jpg"
            cv2.imwrite(str(out_img_dir / out_name), defect_img_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            # YOLO label — class 0 (defect), whole-image bbox
            lbl_path = out_lbl_dir / f"synth_{i:04d}_{src.stem}.txt"
            lbl_path.write_text("0 0.500000 0.500000 1.000000 1.000000\n")
            generated += 1

        print(f"[OK] Generated {generated} synthetic defect images → {out_dir}")
        return generated


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic defect images")
    parser.add_argument("--ok_dir",  type=str, required=True,
                        help="Directory of OK casting images")
    parser.add_argument("--out_dir", type=str, default="./yolo_dataset/synthetic",
                        help="Output directory for synthetic defects")
    parser.add_argument("--n",       type=int, default=200,
                        help="Number of synthetic images to generate")
    args = parser.parse_args()

    gen = SyntheticDefectGenerator()
    gen.generate_batch(
        ok_dir  = Path(args.ok_dir),
        out_dir = Path(args.out_dir),
        n       = args.n,
    )
    print("\n[DONE] Copy synthetic images into yolo_dataset/images/train/")
    print("       Copy synthetic labels into yolo_dataset/labels/train/")
    print("       Then re-run Phase 2 training with augmented dataset.")


if __name__ == "__main__":
    main()
