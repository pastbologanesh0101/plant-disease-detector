"""Programmatic generation of synthetic "leaf" test/demo images.

We do NOT download or scrape any real plant photographs. Instead we draw
simple synthetic leaves with PIL/numpy: a green ellipse on a plain
background, optionally with brown/yellow blotches ("disease spots") drawn
on top at a controllable count/size, so the classical-CV pipeline in
detector.py has fully known, reproducible ground truth to be tested against.

These images are intentionally simple/cartoonish -- they are for
demonstrating and unit-testing the image-processing pipeline, not for
representing photorealistic foliage.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

# Reasonable default colors (RGB)
DEFAULT_BACKGROUND_RGB = (240, 240, 235)   # plain light gray/off-white backdrop
DEFAULT_LEAF_RGB = (34, 139, 34)           # forest green
DEFAULT_BLOTCH_RGB = (139, 90, 43)         # brown
DEFAULT_YELLOW_BLOTCH_RGB = (196, 168, 40)  # dull yellow


def generate_leaf_image(
    width: int = 400,
    height: int = 400,
    background_rgb=DEFAULT_BACKGROUND_RGB,
    leaf_rgb=DEFAULT_LEAF_RGB,
    num_blotches: int = 0,
    blotch_radius_range=(8, 14),
    blotch_rgb=DEFAULT_BLOTCH_RGB,
    margin: int = 40,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate a synthetic leaf image as a BGR numpy array (OpenCV
    convention) so it can be fed directly into detector.analyze_bgr_image.

    A green ellipse represents the leaf; `num_blotches` brown/yellow
    circular blotches are stamped onto it at random positions within the
    leaf ellipse to simulate disease spots.
    """
    rng = random.Random(seed)

    img = Image.new("RGB", (width, height), background_rgb)
    draw = ImageDraw.Draw(img)

    x0, y0, x1, y1 = margin, margin, width - margin, height - margin
    draw.ellipse([x0, y0, x1, y1], fill=leaf_rgb)

    cx, cy = width / 2.0, height / 2.0
    rx, ry = (width - 2 * margin) / 2.0, (height - 2 * margin) / 2.0

    placed = 0
    attempts = 0
    max_attempts = max(50, num_blotches * 50)
    while placed < num_blotches and attempts < max_attempts:
        attempts += 1
        angle = rng.uniform(0, 2 * math.pi)
        # keep blotch centers within 80% of the ellipse radius so they stay
        # comfortably inside the leaf, not spilling onto the background
        r_frac = rng.uniform(0, 0.75)
        bx = cx + r_frac * rx * math.cos(angle)
        by = cy + r_frac * ry * math.sin(angle)
        radius = rng.uniform(*blotch_radius_range)
        draw.ellipse([bx - radius, by - radius, bx + radius, by + radius], fill=blotch_rgb)
        placed += 1

    rgb_array = np.array(img, dtype=np.uint8)
    bgr_array = rgb_array[:, :, ::-1].copy()  # RGB -> BGR for OpenCV
    return bgr_array


def generate_healthy_leaf(width: int = 400, height: int = 400, seed: int | None = 42) -> np.ndarray:
    """A clean green leaf with zero disease blotches."""
    return generate_leaf_image(width=width, height=height, num_blotches=0, seed=seed)


def generate_diseased_leaf(
    width: int = 400,
    height: int = 400,
    num_blotches: int = 10,
    blotch_radius_range=(18, 26),
    seed: int | None = 7,
) -> np.ndarray:
    """A leaf with large brown blotches covering a substantial area."""
    return generate_leaf_image(
        width=width,
        height=height,
        num_blotches=num_blotches,
        blotch_radius_range=blotch_radius_range,
        seed=seed,
    )


def generate_leaf_with_fixed_blotches(
    positions,
    width: int = 400,
    height: int = 400,
    background_rgb=DEFAULT_BACKGROUND_RGB,
    leaf_rgb=DEFAULT_LEAF_RGB,
    blotch_radius: float = 12.0,
    blotch_rgb=DEFAULT_BLOTCH_RGB,
    margin: int = 40,
) -> np.ndarray:
    """Draw a leaf with blotches at exact, caller-specified (x, y) centers.

    Useful for tests that need to assert an *exact* known spot count: the
    caller picks well-separated positions so the blotches never merge into
    a single connected component.
    """
    img = Image.new("RGB", (width, height), background_rgb)
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = margin, margin, width - margin, height - margin
    draw.ellipse([x0, y0, x1, y1], fill=leaf_rgb)

    for (bx, by) in positions:
        draw.ellipse(
            [bx - blotch_radius, by - blotch_radius, bx + blotch_radius, by + blotch_radius],
            fill=blotch_rgb,
        )

    rgb_array = np.array(img, dtype=np.uint8)
    return rgb_array[:, :, ::-1].copy()


def save_image_bgr(bgr: np.ndarray, path: str) -> None:
    """Save a BGR numpy array to disk as an image file (via PIL, converting
    back to RGB first)."""
    rgb = bgr[:, :, ::-1]
    Image.fromarray(rgb).save(path)
