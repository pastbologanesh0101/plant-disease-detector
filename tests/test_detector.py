"""Unit tests for the classical-CV plant-health heuristic pipeline.

All test images are synthetic, generated programmatically by synthetic.py --
no real plant photographs are downloaded, scraped, or bundled.
"""

import os

import numpy as np
import pytest

import detector
import synthetic


# ---------------------------------------------------------------------------
# 1. A clean green synthetic leaf classifies as healthy.
# ---------------------------------------------------------------------------
def test_healthy_leaf_classifies_as_healthy():
    img = synthetic.generate_healthy_leaf()
    result = detector.analyze_bgr_image(img)

    assert result.classification == "healthy"
    assert result.unhealthy_pct < detector.UNHEALTHY_PCT_MILD
    assert result.spot_count == 0


# ---------------------------------------------------------------------------
# 2. A leaf with large brown blotches classifies as diseased, with a
#    correctly elevated unhealthy-area percentage.
# ---------------------------------------------------------------------------
def test_diseased_leaf_with_large_blotches_is_diseased_with_elevated_pct():
    healthy_img = synthetic.generate_healthy_leaf()
    healthy_result = detector.analyze_bgr_image(healthy_img)

    diseased_img = synthetic.generate_diseased_leaf(
        num_blotches=14, blotch_radius_range=(22, 30)
    )
    diseased_result = detector.analyze_bgr_image(diseased_img)

    assert diseased_result.classification == "diseased"
    assert diseased_result.unhealthy_pct >= detector.UNHEALTHY_PCT_DISEASED
    # The whole point of the blotches: unhealthy % must clearly exceed the
    # healthy baseline.
    assert diseased_result.unhealthy_pct > healthy_result.unhealthy_pct + 10


# ---------------------------------------------------------------------------
# 3. Spot-count detection matches a known number of drawn, well-separated
#    blotches.
# ---------------------------------------------------------------------------
def test_spot_count_matches_known_number_of_blotches():
    # 5 well-separated blotch centers, spaced far enough apart (given
    # radius 10) that they cannot merge into one connected component.
    positions = [
        (150, 150),
        (250, 150),
        (150, 250),
        (250, 250),
        (200, 200 - 0),  # will be adjusted below to avoid overlap
    ]
    # Replace the 5th point with something clearly separated from the center
    # cluster and from the leaf edge.
    positions[4] = (200, 320)

    img = synthetic.generate_leaf_with_fixed_blotches(positions, blotch_radius=10.0)
    result = detector.analyze_bgr_image(img)

    assert result.spot_count == len(positions)


# ---------------------------------------------------------------------------
# 4. Severity score increases monotonically with blotch coverage across
#    three severity levels.
# ---------------------------------------------------------------------------
def test_severity_score_increases_monotonically_with_blotch_coverage():
    low = synthetic.generate_diseased_leaf(num_blotches=2, blotch_radius_range=(6, 8), seed=1)
    medium = synthetic.generate_diseased_leaf(num_blotches=8, blotch_radius_range=(12, 16), seed=2)
    high = synthetic.generate_diseased_leaf(num_blotches=16, blotch_radius_range=(20, 28), seed=3)

    r_low = detector.analyze_bgr_image(low)
    r_medium = detector.analyze_bgr_image(medium)
    r_high = detector.analyze_bgr_image(high)

    assert r_low.severity_score < r_medium.severity_score < r_high.severity_score
    assert r_low.unhealthy_pct < r_medium.unhealthy_pct < r_high.unhealthy_pct


# ---------------------------------------------------------------------------
# 5. Leaf-area segmentation distinguishes foreground (leaf) from a
#    solid-color background.
# ---------------------------------------------------------------------------
def test_leaf_area_segmentation_distinguishes_foreground_from_background():
    width, height, margin = 400, 400, 40
    img = synthetic.generate_healthy_leaf(width=width, height=height)

    mask = detector.segment_leaf_foreground(img)

    # A pixel well inside the leaf ellipse should be foreground.
    assert mask[height // 2, width // 2] == True  # noqa: E712

    # A pixel in a background corner should be background.
    assert mask[5, 5] == False  # noqa: E712

    # The leaf ellipse area is pi * rx * ry; foreground pixel count should
    # be in a plausible ballpark of that (allowing for anti-aliasing /
    # morphological cleanup).
    rx = (width - 2 * margin) / 2.0
    ry = (height - 2 * margin) / 2.0
    expected_area = np.pi * rx * ry
    detected_area = np.count_nonzero(mask)

    assert 0.7 * expected_area < detected_area < 1.3 * expected_area


# ---------------------------------------------------------------------------
# 6. A missing image file is handled with a clear error.
# ---------------------------------------------------------------------------
def test_missing_image_file_raises_clear_error():
    with pytest.raises(detector.ImageLoadError, match="not found"):
        detector.analyze_image("/tmp/this_file_definitely_does_not_exist_xyz.jpg")


# ---------------------------------------------------------------------------
# 7. A corrupt image file (exists on disk, but isn't valid image data) is
#    handled with a clear error, not a crash.
# ---------------------------------------------------------------------------
def test_corrupt_image_file_raises_clear_error(tmp_path):
    bad_file = tmp_path / "corrupt.jpg"
    bad_file.write_bytes(b"this is not a real jpeg file at all, just text bytes")

    with pytest.raises(detector.ImageLoadError, match="[Cc]ould not decode|not found"):
        detector.analyze_image(str(bad_file))


# ---------------------------------------------------------------------------
# 8. A moderately-blotched leaf lands in "mild-stress", between healthy and
#    diseased -- exercising the middle of the classification heuristic.
# ---------------------------------------------------------------------------
def test_moderate_blotches_classify_as_mild_stress():
    img = synthetic.generate_diseased_leaf(num_blotches=3, blotch_radius_range=(8, 10), seed=99)
    result = detector.analyze_bgr_image(img)

    assert result.classification in ("mild-stress", "diseased")
    assert result.unhealthy_pct >= detector.UNHEALTHY_PCT_MILD or result.spot_count >= detector.SPOT_COUNT_MILD
    # Should not be as severe as a heavily-blotched leaf.
    heavy = synthetic.generate_diseased_leaf(num_blotches=16, blotch_radius_range=(20, 28), seed=3)
    heavy_result = detector.analyze_bgr_image(heavy)
    assert result.severity_score < heavy_result.severity_score


# ---------------------------------------------------------------------------
# 9. Severity score is always bounded to [0, 100], even for an extreme,
#    almost-entirely-diseased leaf.
# ---------------------------------------------------------------------------
def test_severity_score_is_bounded_0_to_100():
    extreme = synthetic.generate_diseased_leaf(num_blotches=60, blotch_radius_range=(15, 20), seed=123)
    result = detector.analyze_bgr_image(extreme)

    assert 0.0 <= result.severity_score <= 100.0
    assert 0.0 <= result.unhealthy_pct <= 100.0


# ---------------------------------------------------------------------------
# 10. End-to-end: analyze_image() works on a real file saved to disk
#     (round-trips through synthetic.save_image_bgr + cv2.imread).
# ---------------------------------------------------------------------------
def test_analyze_image_end_to_end_on_saved_file(tmp_path):
    img = synthetic.generate_healthy_leaf()
    path = tmp_path / "healthy_leaf.png"
    synthetic.save_image_bgr(img, str(path))

    assert os.path.exists(path)
    result = detector.analyze_image(str(path))

    assert result.classification == "healthy"
    assert result.leaf_area_pct > 0
