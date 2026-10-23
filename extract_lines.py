"""
Extract handwritten lines from BOOX tablet PDF exports.

BOOX PDFs export:
1. Dot-grid / lined template as a raster background image (/XObject).
2. All handwritten pen strokes as PDF /Ink annotations (/Annots).

This script extracts pure vector ink paths, clusters strokes geometrically
using baseline-anchored asymmetric envelopes (preventing quote/tall-letter fracturing),
and renders clean line image crops for HTR model training (e.g. TrOCR).
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import pypdf


@dataclass
class Stroke:
    index: int
    rect: Tuple[float, float, float, float]  # (left, top, right, bottom) in top-down coordinates
    paths: List[List[Tuple[float, float]]]   # list of (x, y) point sequences in top-down coordinates
    width: float
    color: Tuple[int, int, int] = (0, 0, 0)
    is_rule: bool = False
    is_underline: bool = False

    @property
    def left(self) -> float:
        return self.rect[0]

    @property
    def top(self) -> float:
        return self.rect[1]

    @property
    def right(self) -> float:
        return self.rect[2]

    @property
    def bottom(self) -> float:
        return self.rect[3]

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def width_span(self) -> float:
        return self.right - self.left

    @property
    def y_center(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def x_center(self) -> float:
        return (self.left + self.right) / 2.0


@dataclass
class LineGroup:
    line_number: int
    page_number: int
    baseline_y: float
    strokes: List[Stroke] = field(default_factory=list)

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        if not self.strokes:
            return (0.0, 0.0, 0.0, 0.0)
        left = min(s.left for s in self.strokes)
        top = min(s.top for s in self.strokes)
        right = max(s.right for s in self.strokes)
        bottom = max(s.bottom for s in self.strokes)
        return (left, top, right, bottom)

    @property
    def y_center(self) -> float:
        if not self.strokes:
            return self.baseline_y
        return np.mean([s.y_center for s in self.strokes])


def extract_page_strokes(page: pypdf.PageObject) -> Tuple[float, float, List[Stroke]]:
    """Extract all /Ink annotations from a PDF page in top-down coordinates."""
    mediabox = page.mediabox
    page_width = float(mediabox.width)
    page_height = float(mediabox.height)

    annots_ref = page.get("/Annots")
    if not annots_ref:
        return page_width, page_height, []

    annots_obj = annots_ref.get_object() if hasattr(annots_ref, "get_object") else annots_ref
    strokes: List[Stroke] = []

    for idx, annot_item in enumerate(annots_obj):
        obj = annot_item.get_object() if hasattr(annot_item, "get_object") else annot_item
        if obj.get("/Subtype") != "/Ink":
            continue

        raw_rect = [float(x) for x in obj.get("/Rect", [0, 0, 0, 0])]
        left = raw_rect[0]
        right = raw_rect[2]
        top = page_height - raw_rect[3]
        bottom = page_height - raw_rect[1]

        # Extract stroke width
        bs = obj.get("/BS")
        if bs:
            bs = bs.get_object() if hasattr(bs, "get_object") else bs
            stroke_width = float(bs.get("/W", 4.0))
        else:
            stroke_width = 4.0

        # Extract ink paths
        inklist = obj.get("/InkList")
        if not inklist:
            continue
        inklist = inklist.get_object() if hasattr(inklist, "get_object") else inklist

        paths: List[List[Tuple[float, float]]] = []
        for path_obj in inklist:
            p = path_obj.get_object() if hasattr(path_obj, "get_object") else path_obj
            coords = [float(c) for c in p]
            pt_list = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    px = coords[i]
                    py = page_height - coords[i + 1]  # Flip to top-down
                    pt_list.append((px, py))
            if pt_list:
                paths.append(pt_list)

        # Detect full-width dividing rules (e.g. underline across whole page)
        st_w = right - left
        st_h = bottom - top
        is_rule = (st_w > 800) and (st_h < 45)
        is_underline = (st_w > 60) and (st_h < 25) and (st_w / max(1.0, st_h) > 4.5)

        strokes.append(
            Stroke(
                index=idx,
                rect=(left, top, right, bottom),
                paths=paths,
                width=stroke_width,
                is_rule=is_rule,
                is_underline=is_underline,
            )
        )

    return page_width, page_height, strokes


def cluster_strokes_into_lines(
    strokes: List[Stroke],
    page_number: int,
    page_width: Optional[float] = None,
    page_height: Optional[float] = None,
) -> List[LineGroup]:
    """
    Cluster strokes into lines using unified stroke proximity with line-merge tuning.
    
    If page_width > page_height (landscape orientation), automatically detects
    a two-column layout separated by a central gutter, clustering left column
    then right column to maintain natural reading order.
    """
    text_strokes = [s for s in strokes if not s.is_rule]
    if not text_strokes:
        return []

    # Check for landscape two-column layout
    if page_width is not None and page_height is not None and page_width > page_height:
        gutter = page_width / 2.0
        left_strokes = [s for s in text_strokes if s.x_center < gutter]
        right_strokes = [s for s in text_strokes if s.x_center >= gutter]

        # Only use two-column mode if both columns have significant stroke content (> 15% each)
        if len(left_strokes) > 0.15 * len(text_strokes) and len(right_strokes) > 0.15 * len(text_strokes):
            left_lines = _cluster_single_column(left_strokes, page_number)
            right_lines = _cluster_single_column(right_strokes, page_number)
            all_lines = left_lines + right_lines
            for idx, l in enumerate(all_lines):
                l.line_number = idx + 1
            return all_lines

    return _cluster_single_column(text_strokes, page_number)


def _cluster_single_column(
    text_strokes: List[Stroke],
    page_number: int,
) -> List[LineGroup]:
    """Cluster a single column of strokes into top-to-bottom lines."""
    if not text_strokes:
        return []

    # Primary strokes: height >= 18px (letters, vertical stems, major strokes)
    primary = [s for s in text_strokes if s.height >= 18.0]
    minor = [s for s in text_strokes if s.height < 18.0]
    if len(primary) < 5:
        primary = text_strokes
        minor = []

    primary.sort(key=lambda s: s.y_center)

    raw_clusters = []
    curr = []
    for s in primary:
        if not curr:
            curr.append(s)
        else:
            line_y = np.median([x.y_center for x in curr])
            # Tolerance: 42px handles tall capitals and quotes within the same line
            if abs(s.y_center - line_y) <= 42.0:
                curr.append(s)
            else:
                raw_clusters.append(curr)
                curr = [s]
    if curr:
        raw_clusters.append(curr)

    # Post-merge pass: if two clusters overlap horizontally and have vertical distance < 56px,
    # or if an orphan cluster has <= 3 strokes and distance < 65px, merge them!
    merged = []
    for c in raw_clusters:
        if not merged:
            merged.append(c)
        else:
            prev = merged[-1]
            prev_y = np.median([x.y_center for x in prev])
            curr_y = np.median([x.y_center for x in c])
            dist = curr_y - prev_y

            prev_l = min(x.left for x in prev)
            prev_r = max(x.right for x in prev)
            curr_l = min(x.left for x in c)
            curr_r = max(x.right for x in c)
            horiz_overlap = not (curr_r < prev_l or curr_l > prev_r)

            if horiz_overlap and (dist < 56.0 or len(c) <= 3 or len(prev) <= 3):
                prev.extend(c)
            else:
                merged.append(c)

    # Assign minor strokes (dots on i, commas, small accents, serifs)
    for m in minor:
        # Stem-binding check: If this mark sits directly atop a vertical stem in a cluster
        # (e.g. top crossbar of tall capital 'I' or 'T'), bind it directly to that stem's cluster.
        stem_cluster = None
        for c in merged:
            for s in c:
                if s.height >= 30.0 and abs(s.x_center - m.x_center) <= 22.0:
                    if -5.0 <= (s.top - m.bottom) <= 25.0:
                        stem_cluster = c
                        break
            if stem_cluster is not None:
                break

        if stem_cluster is not None:
            stem_cluster.append(m)
            continue

        best_c = None
        min_d = float("inf")
        for c in merged:
            c_y = np.median([x.y_center for x in c])
            c_l = min(x.left for x in c)
            c_r = max(x.right for x in c)
            dist = abs(m.y_center - c_y)
            # Prioritize line that contains the mark horizontally
            in_x = (c_l - 25.0 <= m.x_center <= c_r + 25.0)
            score = dist if in_x else dist + 50.0
            if score < min_d:
                min_d = score
                best_c = c
        if best_c is not None:
            best_c.append(m)

    # Build line groups sorted top-to-bottom
    lines: List[LineGroup] = []
    for idx, c in enumerate(merged):
        c.sort(key=lambda s: s.left)
        b_y = float(np.median([s.bottom for s in c]))
        lines.append(LineGroup(line_number=idx + 1, page_number=page_number, baseline_y=b_y, strokes=c))

    lines.sort(key=lambda l: l.bounding_box[1])
    for idx, l in enumerate(lines):
        l.line_number = idx + 1

    return lines


def render_line_strip(
    line: LineGroup,
    padding: int = 18,
    padding_top: int = 22,
    padding_bottom: int = 26,
    min_width: int = 100,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    ink_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Render a LineGroup to a clean PIL Image with generous vertical padding."""
    left, top, right, bottom = line.bounding_box

    crop_left = max(0, int(np.floor(left)) - padding)
    crop_top = max(0, int(np.floor(top)) - padding_top)
    crop_right = int(np.ceil(right)) + padding
    crop_bottom = int(np.ceil(bottom)) + padding_bottom

    width = max(min_width, crop_right - crop_left)
    height = max(32, crop_bottom - crop_top)

    img = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(img)

    for stroke in line.strokes:
        s_width = max(2, int(round(stroke.width)))
        for path in stroke.paths:
            if len(path) < 2:
                px, py = path[0]
                lx = px - crop_left
                ly = py - crop_top
                r = s_width / 2.0
                draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=ink_color)
            else:
                adjusted_pts = [(px - crop_left, py - crop_top) for (px, py) in path]
                draw.line(adjusted_pts, fill=ink_color, width=s_width, joint="curve")

    return img


def load_existing_annotations(catalog_path: Path) -> List[Dict]:
    """Load previously saved ground truth annotations if available."""
    if not catalog_path.exists():
        return []
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [item for item in data if item.get("ground_truth", "").strip()]
    except Exception as e:
        print(f"Notice: Could not load existing annotations ({e})")
        return []


def match_annotation_to_line(new_bbox: List[float], new_page: int, old_annotations: List[Dict]) -> str:
    """
    Match a newly extracted line to an old annotation using vertical overlap and page number.
    Returns the ground_truth string if matched, else empty string.
    """
    n_top, n_bottom = new_bbox[1], new_bbox[3]
    best_match = ""
    best_iou = 0.0

    for old in old_annotations:
        if old.get("page") != new_page:
            continue
        text = old.get("ground_truth", "").strip()
        if not text:
            continue
        o_bbox = old.get("bounding_box", [0, 0, 0, 0])
        o_top, o_bottom = o_bbox[1], o_bbox[3]

        # 1D vertical intersection over union
        inter_top = max(n_top, o_top)
        inter_bottom = min(n_bottom, o_bottom)
        inter_h = max(0.0, inter_bottom - inter_top)

        union_top = min(n_top, o_top)
        union_bottom = max(n_bottom, o_bottom)
        union_h = max(1.0, union_bottom - union_top)

        iou = inter_h / union_h
        if iou > best_iou and iou >= 0.40:
            best_iou = iou
            best_match = text

    return best_match


def process_pdf(
    pdf_path: str,
    output_dir: str,
    pages: Optional[List[int]] = None,
    padding: int = 15,
) -> List[Dict]:
    """Process a BOOX PDF and save line crops to output_dir, preserving prior annotations."""
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(pdf_path).stem
    metadata_file = out_path / f"{base_name}_lines.json"
    csv_file = out_path / "dataset.csv"

    # Check for existing annotations to preserve
    existing_annotations = load_existing_annotations(metadata_file)
    if existing_annotations:
        print(f"Found {len(existing_annotations)} existing annotations to migrate.")

    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    target_pages = pages if pages else list(range(1, total_pages + 1))

    catalog = []
    migrated_count = 0

    for page_num in target_pages:
        if page_num < 1 or page_num > total_pages:
            print(f"Skipping invalid page {page_num} (total pages: {total_pages})")
            continue

        page = reader.pages[page_num - 1]
        width, height, strokes = extract_page_strokes(page)
        lines = cluster_strokes_into_lines(strokes, page_number=page_num, page_width=width, page_height=height)
        print(f"Page {page_num}: Extracted {len(strokes)} strokes -> {len(lines)} lines")

        for line in lines:
            line_img = render_line_strip(line, padding=padding, padding_top=24, padding_bottom=28)
            img_filename = f"{base_name}_p{page_num:02d}_line{line.line_number:02d}.png"
            img_save_path = images_dir / img_filename
            line_img.save(img_save_path, "PNG")

            bbox = [round(x, 1) for x in line.bounding_box]

            # Attempt automatic label migration
            gt_text = match_annotation_to_line(bbox, page_num, existing_annotations)
            if gt_text:
                migrated_count += 1

            # Use clean relative path under output directory (e.g. batch_aug25/images/...)
            rel_img_path = f"{out_path.name}/images/{img_filename}"

            catalog.append({
                "source_pdf": str(pdf_path),
                "page": page_num,
                "line": line.line_number,
                "image_filename": img_filename,
                "image_path": rel_img_path,
                "width": line_img.width,
                "height": line_img.height,
                "stroke_count": len(line.strokes),
                "baseline_y": round(line.baseline_y, 1),
                "bounding_box": bbox,
                "ground_truth": gt_text,
            })

    # Save catalog metadata JSON
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    # Sync dataset.csv
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "ground_truth", "page", "line", "stroke_count"])
        for item in catalog:
            writer.writerow([
                item["image_filename"],
                item.get("ground_truth", ""),
                item.get("page", 1),
                item.get("line", 1),
                item.get("stroke_count", 0),
            ])

    print(f"\nProcessing complete: {len(catalog)} lines saved to {images_dir}")
    print(f"Catalog saved to {metadata_file}")
    print(f"Dataset CSV saved to {csv_file}")
    if existing_annotations:
        print(f"Successfully migrated {migrated_count} of {len(existing_annotations)} annotations!")

    return catalog


def main():
    parser = argparse.ArgumentParser(description="Extract clean line images from BOOX PDF journals.")
    parser.add_argument("pdf_path", help="Path to BOOX journal PDF")
    parser.add_argument("--output-dir", default="extracted_lines", help="Directory to save line images and catalog")
    parser.add_argument("--pages", nargs="+", type=int, help="Specific 1-based page numbers to extract (default: all)")
    parser.add_argument("--padding", type=int, default=15, help="Pixel padding around line crops (default: 15)")

    args = parser.parse_args()
    process_pdf(args.pdf_path, args.output_dir, pages=args.pages, padding=args.padding)


if __name__ == "__main__":
    main()
