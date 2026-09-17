# AI Plant Disease Detector

A small, self-contained Python tool that analyzes a photo of a plant leaf
and produces a **heuristic** health assessment (healthy / mild-stress /
diseased) using **classical computer vision** — HSV color-space
segmentation, Otsu thresholding, and connected-component blob analysis.

## Important: this is classical CV, not a trained ML classifier

**This project deliberately does not use a trained deep-learning model
(no CNN, no PyTorch/TensorFlow, no `.h5`/`.pt` weights).**

Real plant-disease classification (e.g. distinguishing early blight from
late blight from bacterial spot) requires a large, labeled image dataset
and training a convolutional neural network on it — that's simply not
feasible as a from-scratch, self-contained portfolio project with no
external data.

Instead, this project demonstrates the **classical image-processing
pipeline** that a plant-health tool would typically use *before* or
*instead of* a learned classifier — for segmentation and feature
extraction — and then applies a **hand-written, threshold-based heuristic**
(not fit to any data) to turn those measurements into a label. Think of it
as "the plumbing a real system would need," honestly presented as a
heuristic, not as an accuracy claim about diagnosing real plant diseases.

## Pipeline

For each input image, `detector.py` runs:

1. **Foreground/background segmentation** (`segment_leaf_foreground`)
   Converts the image to HSV and applies Otsu's method (an automatic,
   data-driven threshold — still classical CV, not ML) to the saturation
   channel. This assumes the leaf is photographed against a relatively
   uniform, low-saturation background (a sheet of paper, a table, etc.),
   which is a common setup for leaf photos and is how the synthetic test
   images here are constructed. The mask is cleaned up with morphological
   opening/closing to remove speckle noise.

2. **Color-space segmentation** (`segment_unhealthy_pixels`)
   Within the leaf mask, pixels are classified in HSV space:
   - **Healthy green**: hue ≈ 28–95° (OpenCV's 0–179 hue scale), moderate
     to high saturation/value — a vivid, bright green.
   - **Unhealthy yellow/brown**: hue ≈ 5–27°, simulating chlorosis
     (yellowing) and blight/leaf-spot browning.
   - **Necrotic/near-black**: value < 55, regardless of hue, simulating
     dead tissue.

3. **Blob / spot detection** (`find_unhealthy_blobs`)
   `cv2.connectedComponentsWithStats` finds distinct connected regions of
   "unhealthy" pixels, filtering out anything under 15px² as noise. Each
   blob's area, centroid, and bounding box are recorded — this gives a
   spot count and spot sizes, not just a raw unhealthy-pixel percentage.

4. **Heuristic classification** (`classify`, `compute_severity_score`)
   Rule-based thresholds on `unhealthy_pct` (unhealthy pixels as % of leaf
   area, not of the whole frame) and `spot_count`:

   | Condition                                                    | Label        |
   |---------------------------------------------------------------|--------------|
   | `unhealthy_pct >= 15%` or `spot_count >= 6`                    | diseased     |
   | `unhealthy_pct >= 5%` or `spot_count >= 2`                     | mild-stress  |
   | otherwise                                                      | healthy      |

   **Rationale (approximate/illustrative, not derived from real data):**
   a handful of small spots (< 5% of leaf area, 0–1 blobs) is treated as
   within normal variation for a "healthy" leaf. Once unhealthy tissue
   crosses roughly 1/20th of the leaf, or there are multiple distinct
   spots, that's flagged as early stress. Beyond ~15% coverage, or six-plus
   separate spots (suggesting spreading infection rather than one bruise),
   the leaf is flagged as diseased.

   The **severity score** (0–100) is `min(100, unhealthy_pct * 2.5 +
   min(spot_count, 20) * 1.5)` — weighted mostly by how much of the leaf
   is affected, with a smaller bonus for fragmentation (many small spots
   read as more "actively spreading" than one big blotch of equal area).

## Honest limitations

- HSV thresholds were tuned by eye against this project's own synthetic
  images, not calibrated against real photographs, varying lighting,
  camera white-balance, or leaf species/cultivar color variation.
- The foreground/background split assumes a plain, low-saturation
  backdrop; it will not reliably segment a leaf photographed in situ
  against other green foliage or cluttered backgrounds.
- "Disease" here really means "leaf regions colored brown/yellow/black" —
  it cannot distinguish disease from other causes of discoloration
  (sunburn, nutrient deficiency, natural senescence, soil/mud, shadows).
- All thresholds are illustrative starting points, documented above, not
  values validated against ground-truth diagnoses.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python detect.py path/to/leaf_image.jpg
```

### Example output

Generate a synthetic test image and run the detector on it:

```bash
python3 -c "
import synthetic
img = synthetic.generate_healthy_leaf()
synthetic.save_image_bgr(img, 'healthy_leaf.png')
"
python detect.py healthy_leaf.png
```

```
Image:                  healthy_leaf.png
Leaf area coverage:     50.5% of frame
Unhealthy-colored area: 0.0% of leaf area
Detected spots/blotches:    0
Severity score:         0.0 / 100
Assessment:             HEALTHY
```

And on a synthetic leaf generated with 14 large brown blotches
(`synthetic.generate_diseased_leaf(num_blotches=14, blotch_radius_range=(22, 30))`):

```
Image:                  diseased_leaf.png
Leaf area coverage:     50.5% of frame
Unhealthy-colored area: 20.9% of leaf area
Detected spots/blotches:    4
Severity score:         58.3 / 100
Assessment:             DISEASED
```

(Spot count is 4, not 14, because at that size/density several blotches
overlap and merge into fewer, larger connected components — a real
consequence of the connected-component approach worth knowing about.)

## Synthetic test images

`synthetic.py` generates leaf images programmatically with PIL/numpy: a
green ellipse ("leaf") on a plain background, with optional brown/yellow
circular blotches ("disease spots") stamped on at controllable count,
size, and position. **No real plant photographs are downloaded, scraped,
or bundled** — every image used in tests and demos is generated at
runtime with known, reproducible ground truth.

## Running the tests

```bash
pip install -r requirements.txt pytest
pytest tests/ -v
```

10 unit tests cover: healthy classification, diseased classification with
elevated unhealthy-percentage, exact spot-count detection, monotonically
increasing severity score across severity levels, foreground/background
leaf-area segmentation, missing-file and corrupt-file error handling, and
an end-to-end run against a file saved to disk.

## Project layout

```
plant-disease-detector/
├── detect.py          # CLI entry point
├── detector.py         # Classical-CV pipeline + heuristic classification
├── synthetic.py        # Synthetic leaf/blotch image generation (PIL/numpy)
├── tests/
│   ├── conftest.py
│   └── test_detector.py
├── requirements.txt
├── LICENSE
└── .github/workflows/tests.yml
```

## License

MIT — see [LICENSE](LICENSE).
