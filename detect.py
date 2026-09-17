#!/usr/bin/env python3
"""CLI entry point for the AI Plant Disease Detector.

Usage:
    python detect.py path/to/leaf_image.jpg

This runs a CLASSICAL computer-vision heuristic pipeline (HSV color
segmentation + connected-component blob analysis + rule-based thresholds).
It is NOT a trained machine-learning / CNN classifier -- see README.md for
why, and for what a real deep-learning version of this tool would need.
"""

from __future__ import annotations

import argparse
import sys

from detector import ImageLoadError, analyze_image


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Classical-CV heuristic plant leaf health analyzer "
            "(HSV segmentation + blob detection, no trained ML model)."
        )
    )
    parser.add_argument("image_path", help="Path to a leaf image file (jpg/png/etc.)")
    args = parser.parse_args(argv)

    try:
        result = analyze_image(args.image_path)
    except ImageLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Image:                  {args.image_path}")
    print(f"Leaf area coverage:     {result.leaf_area_pct:.1f}% of frame")
    print(f"Unhealthy-colored area: {result.unhealthy_pct:.1f}% of leaf area")
    print(f"Detected spots/blotches:{result.spot_count:>5}")
    print(f"Severity score:         {result.severity_score:.1f} / 100")
    print(f"Assessment:             {result.classification.upper()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
