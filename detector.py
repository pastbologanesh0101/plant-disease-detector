"""Classical computer-vision pipeline for heuristic plant-leaf health analysis.

IMPORTANT SCOPE NOTE: This module does NOT use any trained machine-learning
model (no CNN, no scikit-learn classifier, nothing "trained" on labeled
disease data). It performs:

  1. Foreground/background segmentation (Otsu thresholding on saturation)
     to isolate the leaf from its background.
  2. Color-space segmentation (HSV thresholding) to separate "healthy green"
     pixels from "unhealthy" (brown/yellow/near-black spot) pixels.
  3. Connected-component / blob analysis to count and size distinct
     unhealthy spots.
  4. A hand-written heuristic (if/threshold based, not learned) that maps
     those measurements to a healthy / mild-stress / diseased label and a
     0-100 severity score.

See README.md for the full rationale and the honest limitations of this
approach relative to a real, dataset-trained plant-disease classifier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np


class ImageLoadError(Exception):
    """Raised when an image file cannot be found, opened, or decoded."""


# ---------------------------------------------------------------------------
# HSV threshold ranges (OpenCV HSV convention: H in [0, 179], S/V in [0, 255])
# ---------------------------------------------------------------------------
# Tuned by inspection against the synthetic leaf images this project
# generates (see synthetic.py). They are deliberately broad, illustrative
# ranges -- not fit to any real dataset.

# "Healthy" foliage: a broad green hue band with moderate-to-high saturation
# and value (i.e. a vivid, bright green rather than a dull/dark one).
HEALTHY_GREEN_LOWER = np.array([28, 40, 40], dtype=np.uint8)
HEALTHY_GREEN_UPPER = np.array([95, 255, 255], dtype=np.uint8)

# "Unhealthy" yellow/brown blotches (chlorosis, blight, leaf-spot disease):
# warmer, lower hue band than healthy green.
UNHEALTHY_YELLOW_BROWN_LOWER = np.array([5, 40, 30], dtype=np.uint8)
UNHEALTHY_YELLOW_BROWN_UPPER = np.array([27, 255, 230], dtype=np.uint8)

# Near-black / necrotic tissue: very low brightness regardless of hue.
NECROTIC_VALUE_MAX = 55

# Ignore tiny specks (anti-aliasing noise, JPEG artifacts) below this area.
MIN_SPOT_AREA_PX = 15

# ---------------------------------------------------------------------------
# Heuristic classification thresholds (illustrative, hand-picked -- see
# README.md "Chosen thresholds and rationale").
# ---------------------------------------------------------------------------
UNHEALTHY_PCT_MILD = 5.0       # >= 5% unhealthy-colored leaf area -> mild stress
UNHEALTHY_PCT_DISEASED = 15.0  # >= 15% unhealthy-colored leaf area -> diseased
SPOT_COUNT_MILD = 2            # >= 2 distinct spots pushes a clean-% leaf to mild
SPOT_COUNT_DISEASED = 6        # >= 6 distinct spots pushes toward diseased


@dataclass
class Blob:
    area: int
    centroid: tuple
    bbox: tuple  # (x, y, w, h)


@dataclass
class AnalysisResult:
    leaf_area_px: int
    frame_area_px: int
    leaf_area_pct: float          # leaf foreground as % of whole frame
    unhealthy_px: int
    unhealthy_pct: float          # unhealthy pixels as % of LEAF area
    spot_count: int
    spots: List[Blob] = field(default_factory=list)
    severity_score: float = 0.0
    classification: str = "healthy"


def _load_bgr(image_path: str) -> np.ndarray:
    """Load an image file into a BGR numpy array, raising ImageLoadError
    with a clear message on any failure (missing file, unreadable/corrupt
    data, unsupported format, etc.)."""
    if not isinstance(image_path, (str, os.PathLike)):
        raise ImageLoadError(f"Invalid image path type: {type(image_path)!r}")

    if not os.path.exists(image_path):
        raise ImageLoadError(f"Image file not found: {image_path}")

    if os.path.isdir(image_path):
        raise ImageLoadError(f"Expected an image file but got a directory: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageLoadError(
            f"Could not decode image file (missing, corrupt, or unsupported "
            f"format): {image_path}"
        )
    return img


def segment_leaf_foreground(bgr: np.ndarray) -> np.ndarray:
    """Segment the leaf (foreground) from a roughly solid-color background.

    Assumption: the background is significantly less saturated / more
    uniform than the leaf itself -- true for a leaf photographed on a
    plain sheet of paper or table, and for the synthetic test images this
    project generates. We threshold the HSV saturation channel with Otsu's
    method (automatic, data-driven threshold selection -- still classical
    CV, not machine learning) and then clean up the mask morphologically.

    Returns a boolean mask, True where a leaf pixel is detected.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask > 0


def segment_unhealthy_pixels(bgr: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask of pixels, within leaf_mask, whose color falls
    in the "unhealthy" (yellow/brown/necrotic) HSV ranges."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    yellow_brown = cv2.inRange(hsv, UNHEALTHY_YELLOW_BROWN_LOWER, UNHEALTHY_YELLOW_BROWN_UPPER)
    necrotic = ((hsv[:, :, 2] < NECROTIC_VALUE_MAX).astype(np.uint8)) * 255

    unhealthy = cv2.bitwise_or(yellow_brown, necrotic)
    unhealthy_bool = unhealthy > 0
    unhealthy_bool &= leaf_mask

    return unhealthy_bool


def find_unhealthy_blobs(unhealthy_mask: np.ndarray, min_area: int = MIN_SPOT_AREA_PX) -> List[Blob]:
    """Use connected-component analysis to find distinct unhealthy spots."""
    mask_u8 = unhealthy_mask.astype(np.uint8) * 255
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    blobs: List[Blob] = []
    for i in range(1, num_labels):  # label 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]
        blobs.append(Blob(area=area, centroid=(float(cx), float(cy)), bbox=(x, y, w, h)))

    return blobs


def compute_severity_score(unhealthy_pct: float, spot_count: int) -> float:
    """A simple heuristic severity score in [0, 100].

    Weighted mostly by the fraction of unhealthy-colored leaf tissue, with
    a smaller additive contribution from how fragmented the damage is
    (more, smaller spots read as more advanced/spreading disease than one
    big blotch of the same area). Capped at 100.
    """
    score = unhealthy_pct * 2.5 + min(spot_count, 20) * 1.5
    return float(min(100.0, score))


def classify(unhealthy_pct: float, spot_count: int) -> str:
    """Heuristic (rule-based, not learned) health classification.

    Thresholds (see README.md for rationale):
      - healthy:      unhealthy_pct < 5%  AND spot_count < 2
      - mild-stress:  unhealthy_pct < 15% (or a few small spots present)
      - diseased:     unhealthy_pct >= 15% OR spot_count >= 6
    """
    if unhealthy_pct >= UNHEALTHY_PCT_DISEASED or spot_count >= SPOT_COUNT_DISEASED:
        return "diseased"
    if unhealthy_pct >= UNHEALTHY_PCT_MILD or spot_count >= SPOT_COUNT_MILD:
        return "mild-stress"
    return "healthy"


def analyze_bgr_image(bgr: np.ndarray) -> AnalysisResult:
    """Run the full classical-CV pipeline on an already-loaded BGR image
    (numpy array), returning an AnalysisResult. Useful for tests / synthetic
    images that never touch disk."""
    frame_h, frame_w = bgr.shape[:2]
    frame_area_px = frame_h * frame_w

    leaf_mask = segment_leaf_foreground(bgr)
    leaf_area_px = int(np.count_nonzero(leaf_mask))
    leaf_area_pct = 100.0 * leaf_area_px / frame_area_px if frame_area_px else 0.0

    if leaf_area_px == 0:
        return AnalysisResult(
            leaf_area_px=0,
            frame_area_px=frame_area_px,
            leaf_area_pct=0.0,
            unhealthy_px=0,
            unhealthy_pct=0.0,
            spot_count=0,
            spots=[],
            severity_score=0.0,
            classification="healthy",
        )

    unhealthy_mask = segment_unhealthy_pixels(bgr, leaf_mask)
    unhealthy_px = int(np.count_nonzero(unhealthy_mask))
    unhealthy_pct = 100.0 * unhealthy_px / leaf_area_px

    blobs = find_unhealthy_blobs(unhealthy_mask)
    spot_count = len(blobs)

    severity = compute_severity_score(unhealthy_pct, spot_count)
    label = classify(unhealthy_pct, spot_count)

    return AnalysisResult(
        leaf_area_px=leaf_area_px,
        frame_area_px=frame_area_px,
        leaf_area_pct=leaf_area_pct,
        unhealthy_px=unhealthy_px,
        unhealthy_pct=unhealthy_pct,
        spot_count=spot_count,
        spots=blobs,
        severity_score=severity,
        classification=label,
    )


def analyze_image(image_path: str) -> AnalysisResult:
    """Load an image from disk and run the full analysis pipeline on it.

    Raises ImageLoadError with a clear message if the file is missing,
    unreadable, or not a decodable image.
    """
    bgr = _load_bgr(image_path)
    return analyze_bgr_image(bgr)
