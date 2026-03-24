"""
casting_validator.py — Improved Casting Validator
===================================================
Uses multi-feature analysis to distinguish iron casting products
from unrelated objects (phones, faces, cups, paper, etc.)

Checks applied (in order):
  1. Image is not empty / too small
  2. Object occupies meaningful area of frame
  3. Aspect ratio is plausible for a casting
  4. Colour profile — castings are dark/grey metallic, not vivid
  5. Texture gradient — castings have fine-grain texture edges
  6. Brightness range — metal castings are mostly mid-dark tones
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ValidationResult:
    is_valid: bool
    score: float
    reason: str
    confidence: float = 0.0          # alias kept for UI
    scores: Dict[str, float] = field(default_factory=dict)


class CastingValidator:
    """
    Heuristic-based validator for iron/metal casting images.
    All thresholds are tunable via the sidebar sliders.
    """

    def __init__(self, threshold: float = 0.42):
        # This attribute is read by the dashboard sidebar slider
        self.casting_threshold = threshold

    # ------------------------------------------------------------------
    def validate(self, img_bgr: np.ndarray) -> ValidationResult:
        if img_bgr is None or img_bgr.size == 0:
            return ValidationResult(False, 0.0, "Empty image frame.", 0.0, {})

        h, w = img_bgr.shape[:2]
        total_pixels = h * w

        scores: Dict[str, float] = {}

        # ── STEP 1: Find largest object contour ───────────────────────
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges   = cv2.Canny(blurred, 30, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return ValidationResult(False, 0.0,
                                    "Invalid: No clear object detected.", 0.0,
                                    {"contour": 0.0})

        largest  = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        area_ratio   = (bw * bh) / total_pixels
        aspect_ratio = float(bw) / bh if bh > 0 else 0.0

        scores["area_ratio"]   = round(area_ratio, 3)
        scores["aspect_ratio"] = round(aspect_ratio, 3)

        # ── STEP 2: Object must fill at least 8 % of the frame ────────
        if area_ratio < 0.08:
            return ValidationResult(False, 0.15,
                                    "Invalid: Object too small or too far away.",
                                    0.15, scores)

        # ── STEP 3: Aspect ratio filter (wider than 4:1 is very unlikely) ─
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            return ValidationResult(False, 0.15,
                                    "Invalid: Shape does not match a casting.",
                                    0.15, scores)

        # ── Work on ROI (the detected object bounding box) ────────────
        roi = img_bgr[y: y + bh, x: x + bw]
        if roi.size == 0:
            return ValidationResult(False, 0.0,
                                    "Invalid: Object boundary error.", 0.0, scores)

        # ── STEP 4: Colour / Saturation check ─────────────────────────
        # Iron castings are desaturated grey/dark.
        # Vivid colours → phone screen / cups / paper → INVALID
        hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        scores["mean_saturation"] = round(mean_sat, 1)

        if mean_sat > 80:          # very colourful → not a casting
            return ValidationResult(False, 0.1,
                                    f"Invalid: Object is too colourful (sat={mean_sat:.0f}).",
                                    0.1, scores)

        # ── STEP 5: Brightness / Mean grey check ──────────────────────
        # Castings are typically 30–200 brightness range (dark metal/sand).
        # A white paper sheet or bright phone screen will have very high brightness.
        # A human face will also be mid-bright but will fail the texture check.
        mean_brightness = float(np.mean(gray[y: y + bh, x: x + bw]))
        scores["mean_brightness"] = round(mean_brightness, 1)

        if mean_brightness > 210:
            return ValidationResult(False, 0.1,
                                    "Invalid: Object too bright (likely paper/screen).",
                                    0.1, scores)

        # ── STEP 6: Edge density (texture) check ──────────────────────
        # Metal castings have fine-grain surface texture → many edges.
        # Plain paper, skin, or clear plastic → very few edges inside the box.
        roi_gray  = gray[y: y + bh, x: x + bw]
        roi_edges = cv2.Canny(roi_gray, 40, 120)
        edge_density = float(np.sum(roi_edges > 0)) / (bw * bh)
        scores["edge_density"] = round(edge_density, 4)

        if edge_density < 0.04:    # very smooth → not a casting
            return ValidationResult(False, 0.2,
                                    "Invalid: Surface too smooth (not a metal casting).",
                                    0.2, scores)

        # ── STEP 7: Composite casting score ───────────────────────────
        # Combine individual sub-scores into a single composite
        # Higher is more "casting-like"
        sat_score   = max(0.0, 1.0 - mean_sat / 80)          # 0-1, lower sat = better
        bright_score = max(0.0, 1.0 - mean_brightness / 210) # 0-1, darker = better
        edge_score   = min(1.0, edge_density / 0.15)          # 0-1, more edges = better

        composite = (sat_score * 0.35 + bright_score * 0.25 + edge_score * 0.40)
        scores["composite"] = round(composite, 3)

        if composite < self.casting_threshold:
            reason = (f"Invalid: Casting score {composite:.2f} below threshold "
                      f"{self.casting_threshold:.2f}.")
            return ValidationResult(False, composite, reason, composite, scores)

        # ── PASSED ALL CHECKS ─────────────────────────────────────────
        return ValidationResult(True, composite,
                                "Valid Casting – Checking for defects…",
                                composite, scores)


def get_validator(threshold: float = 0.42) -> CastingValidator:
    return CastingValidator(threshold)