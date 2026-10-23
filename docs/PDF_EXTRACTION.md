# BOOX PDF Vector Ink Extraction & Layout Geometry

Technical mechanics for extracting and segmenting handwriting from Onyx BOOX tablet PDF exports without computer vision noise.

---

## 1. Vector Ink Layer Decoupling

Standard document OCR often fails on e-ink exports because the paper template (dot grid, lined paper, margins) interferes with character segmentation. 

Reverse engineering the PDF structure of BOOX tablet exports revealed a crucial architectural property:

* **Background Template**: Stored as a background raster image (`/XObject /FXX1...`).
* **Handwritten Strokes**: Stored as native vector PDF `/Ink` annotations under the page's `/Annots` catalog.

By rendering *only* the vector `/InkList` coordinate paths onto a blank white canvas, the dot grid and paper texture vanish completely. No OpenCV morphological opening, thresholding, or dot-filtering heuristics are required.

---

## 2. Geometric Line Clustering

Because every stroke annotation includes its exact bounding box `Rect = [x1, y1, x2, y2]`, line segmentation is solved geometrically:

1. **Vertical Stem-Binding**: Minor horizontal crossbars atop tall vertical stems (e.g. pronoun capital "I" or "T"s) are bound to the underlying stem before line grouping.
2. **Proximity Clustering**: Strokes with vertical centers within 42px (~half standard line pitch) are merged into candidate line groups.
3. **Horizontal Overlap Merge**: Adjacent clusters overlapping horizontally within 56px vertically merge into a single line.
4. **Asymmetric Render Padding**: Line image strips are rendered with asymmetric padding (`top=24px`, `bottom=28px`, `horizontal=18px`) to ensure tall ascenders and deep descenders (`y`, `g`, `p`) are not clipped.

---

## 3. Landscape Two-Column Layout Splitting

Older BOOX volumes (e.g., from 2022) use a **landscape two-column format** ($2480 \times 1860\text{px}$).

`extract_lines.py` detects landscape orientation (`width > height`) and splits strokes across the central 50px gutter at $X=1240$:
* **Left Column**: Strokes where $X < 1240$ are grouped top-to-bottom first.
* **Right Column**: Strokes where $X \ge 1240$ are grouped top-to-bottom second.

This preserves natural reading order across multi-column pages without column interweaving.
