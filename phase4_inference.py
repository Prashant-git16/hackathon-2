"""
ZeroDefect AI — Phase 4: Inference Engine
==========================================
Pipeline:
  Input → Casting Validator → (If Valid) → YOLO Defect Detection → Output

Decisions:
  ✓ ACCEPT  — valid casting, no defects found
  ✗ DEFECT  — valid casting, defect(s) detected by YOLO
  ⚠ INVALID — not a casting product (rejected by validator)
"""

import cv2
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from ultralytics import YOLO
from casting_validator import CastingValidator, ValidationResult, get_validator

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONF_THRESHOLD    = 0.45   # Lowered slightly to catch more real defects
IOU_THRESHOLD     = 0.45
CASTING_THRESHOLD = 0.42


@dataclass
class FrameResult:
    decision:          str            # "ACCEPT", "DEFECT", "INVALID"
    status_message:    str
    inference_ms:      float
    detections:        List[Dict]
    validation_result: Optional[ValidationResult]  # Full object (not a dict)


# ── INFERENCE ENGINE ──────────────────────────────────────────────────────────

class DefectInferenceEngine:
    def __init__(self, weights_path: str, device: str = "cpu",
                 casting_threshold: float = CASTING_THRESHOLD):
        print(f"[ZeroDefect] Loading YOLO model from {weights_path} on {device}…")
        self.model = YOLO(weights_path)
        self.model.to(device)
        self.validator = get_validator(threshold=casting_threshold)
        # Inspect class names from the model
        self._class_names: Dict[int, str] = {}

    def _get_class_names(self) -> Dict[int, str]:
        """Read class names from the loaded YOLO model."""
        if not self._class_names and hasattr(self.model, "names"):
            self._class_names = self.model.names  # {0: 'defect', 1: 'ok'} etc.
        return self._class_names

    # ------------------------------------------------------------------
    def predict_frame(self, img_bgr: np.ndarray,
                      conf: float = CONF_THRESHOLD) -> FrameResult:
        t0 = time.time()

        # ── STEP 1: Gatekeeper — is this a casting? ───────────────────
        val_res = self.validator.validate(img_bgr)

        if not val_res.is_valid:
            inference_ms = (time.time() - t0) * 1000
            return FrameResult(
                decision="INVALID",
                status_message=val_res.reason,
                inference_ms=inference_ms,
                detections=[],
                validation_result=val_res,
            )

        # ── STEP 2: YOLO defect detection ─────────────────────────────
        results      = self.model.predict(img_bgr, conf=conf,
                                          iou=IOU_THRESHOLD, verbose=False)
        inference_ms = (time.time() - t0) * 1000

        class_names = self._get_class_names()

        detections: List[Dict] = []
        has_defect = False

        for r in results:
            for box in r.boxes:
                cls_id     = int(box.cls[0])
                conf_score = float(box.conf[0])
                xyxy       = box.xyxy[0].cpu().numpy().tolist()

                # Resolve class name from model; fall back to id-based heuristic
                raw_name = class_names.get(cls_id, str(cls_id)).lower()

                # Treat anything that is not explicitly "ok" / "good" as a defect
                is_defect = raw_name not in ("ok", "good", "pass", "no_defect",
                                             "no defect", "non_defect")

                cls_label = "defect" if is_defect else "ok"

                detections.append({
                    "class":      cls_label,
                    "raw_class":  raw_name,
                    "confidence": conf_score,
                    "bbox":       xyxy,
                })

                if is_defect:
                    has_defect = True

        decision       = "DEFECT" if has_defect else "ACCEPT"
        status_message = ("⚠ Defect(s) Detected!" if has_defect
                          else "✓ No Defect – Product Accepted")

        return FrameResult(
            decision=decision,
            status_message=status_message,
            inference_ms=inference_ms,
            detections=detections,
            validation_result=val_res,
        )

    # ------------------------------------------------------------------
    def annotate_frame(self, img_bgr: np.ndarray,
                       result: FrameResult) -> np.ndarray:
        annotated = img_bgr.copy()
        h, w      = annotated.shape[:2]
        banner_h  = max(55, h // 12)

        # ── INVALID ───────────────────────────────────────────────────
        if result.decision == "INVALID":
            cv2.rectangle(annotated, (0, 0), (w, banner_h), (0, 140, 255), -1)
            cv2.putText(annotated,
                        "⚠ INVALID — Not a Casting Product",
                        (12, banner_h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
            # Smaller sub-text with reason
            short_reason = result.status_message[:80]
            cv2.putText(annotated, short_reason,
                        (12, banner_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 100), 1,
                        cv2.LINE_AA)
            return annotated

        # ── ACCEPT / DEFECT banner ─────────────────────────────────────
        if result.decision == "ACCEPT":
            banner_color = (20, 180, 50)      # green
            banner_text  = "✓ ACCEPT — No Defect Found"
        else:
            banner_color = (0, 30, 210)       # red
            banner_text  = "✗ DEFECT — Defective Product"

        cv2.rectangle(annotated, (0, 0), (w, banner_h), banner_color, -1)
        cv2.putText(annotated, banner_text,
                    (12, banner_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)

        # Show inference latency in top-right corner
        lat_text = f"{result.inference_ms:.0f} ms"
        (tw, th), _ = cv2.getTextSize(lat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(annotated, lat_text,
                    (w - tw - 10, banner_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1,
                    cv2.LINE_AA)

        # ── BOUNDING BOXES ────────────────────────────────────────────
        for det in result.detections:
            x1, y1, x2, y2 = map(int, det["bbox"])

            # Clamp coordinates to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue  # skip degenerate boxes

            is_defect = det["class"] == "defect"
            box_color = (0, 30, 210) if is_defect else (20, 180, 50)   # red / green
            label     = f"{det['class'].upper()} {det['confidence']:.0%}"

            # Draw thick box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Label background pill
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            label_y_top = max(y1 - lh - 8, banner_h + 4)
            cv2.rectangle(annotated,
                          (x1, label_y_top),
                          (x1 + lw + 4, label_y_top + lh + 6),
                          box_color, -1)
            cv2.putText(annotated, label,
                        (x1 + 2, label_y_top + lh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)

        return annotated


# ── STANDALONE RUN MODES (not used by Streamlit dashboard) ────────────────────

def run_webcam(engine: DefectInferenceEngine, conf: float = CONF_THRESHOLD):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Live feed started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result    = engine.predict_frame(frame, conf=conf)
        annotated = engine.annotate_frame(frame, result)

        fps_text = f"Latency: {result.inference_ms:.1f} ms | {result.decision}"
        cv2.putText(annotated, fps_text,
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("ZeroDefect AI - Live Inspection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroDefect AI Inference")
    parser.add_argument("--weights", type=str, default="best.pt")
    parser.add_argument("--source",  type=str, required=True,
                        help="'webcam' or path to image file")
    parser.add_argument("--device",  type=str, default="cpu")
    parser.add_argument("--conf",    type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    engine = DefectInferenceEngine(args.weights, device=args.device)

    if args.source.lower() == "webcam":
        run_webcam(engine, conf=args.conf)
    else:
        src = Path(args.source)
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")

        img    = cv2.imread(str(src))
        result = engine.predict_frame(img, conf=args.conf)
        annotated = engine.annotate_frame(img, result)

        out_path = src.parent / f"result_{src.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"\nSaved → {out_path}")
        print(f"Decision : {result.decision}")
        print(f"Message  : {result.status_message}")
        print(f"Defects  : {len(result.detections)}")