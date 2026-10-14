"""
Catalog page segmentation module.
Splits a rendered catalog page into individual design bounding boxes:
- Applies morphological closing to bridge apertures and cutouts into solid blobs
- Filters out text lines, page margins, extreme aspect ratios, and tiny noise
- Isolates single designs or multi-design catalog grids
- Orders bounding boxes in natural row-major reading order
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import config

logger = logging.getLogger(__name__)


def sort_boxes_row_major(
    boxes: List[Tuple[int, int, int, int]],
    y_threshold_fraction: float = 0.40,
) -> List[Tuple[int, int, int, int]]:
    """
    Sorts 2D bounding boxes (x, y, w, h) in natural row-major reading order:
    Top-to-bottom, then left-to-right within each row.
    """
    if not boxes:
        return []
    if len(boxes) <= 1:
        return boxes

    # Median height for vertical row clustering threshold
    median_h = float(np.median([h for _, _, _, h in boxes]))
    y_threshold = max(25.0, median_h * y_threshold_fraction)

    # Calculate (box, center_x, center_y)
    with_centers = [
        (box, float(box[0] + box[2] / 2.0), float(box[1] + box[3] / 2.0))
        for box in boxes
    ]
    with_centers.sort(key=lambda item: item[2])  # sort by y-center

    rows: List[List[Tuple[Tuple[int, int, int, int], float, float]]] = []
    current_row: List[Tuple[Tuple[int, int, int, int], float, float]] = []
    current_row_y = 0.0

    for item in with_centers:
        _, _, yc = item
        if not current_row:
            current_row.append(item)
            current_row_y = yc
        else:
            if abs(yc - current_row_y) <= y_threshold:
                current_row.append(item)
                current_row_y = sum(x[2] for x in current_row) / len(current_row)
            else:
                rows.append(current_row)
                current_row = [item]
                current_row_y = yc

    if current_row:
        rows.append(current_row)

    sorted_boxes: List[Tuple[int, int, int, int]] = []
    for row in rows:
        row.sort(key=lambda item: item[1])  # sort left-to-right by x-center
        for item in row:
            sorted_boxes.append(item[0])

    return sorted_boxes


def segment_designs(
    page_bgr: np.ndarray,
    page_mask: Optional[np.ndarray] = None,
    min_area: int = config.MIN_DESIGN_AREA,
    max_aspect_ratio: float = config.MAX_ASPECT_RATIO,
) -> List[Tuple[int, int, int, int]]:
    """
    Splits a rendered catalog page into individual designs.

    Args:
        page_bgr: Full page BGR image at standard DPI.
        page_mask: Binary stencil mask (uint8, 255=metal, 0=voids). If None,
                   computed using normalize_stencil(page_bgr, crop=False).
        min_area: Minimum design area in pixels (w * h).
        max_aspect_ratio: Maximum allowable width/height or height/width aspect ratio.

    Returns:
        List of bounding boxes (x, y, w, h) in page coordinates.
    """
    if page_bgr is None or page_bgr.size == 0:
        return []

    page_h, page_w = page_bgr.shape[:2]
    page_total_area = float(page_h * page_w)

    if page_mask is None:
        from preprocess import normalize_stencil  # type: ignore
        page_mask, _ = normalize_stencil(page_bgr, crop=False)

    # Step 1: Morphological closing to bridge cut-outs into solid blobs
    # Use 7x7 or 9x9 kernel with 1 iteration to close internal cutouts without merging adjacent tiles
    k_size = max(7, min(11, int(min(page_h, page_w) * 0.01)))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(page_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 2: Find contours of solid blobs
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes: List[Tuple[int, int, int, int]] = []
    min_page_fraction_area = page_total_area * 0.004
    effective_min_area = max(float(min_area), min_page_fraction_area)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box_area = float(w * h)

        if box_area < effective_min_area:
            continue

        # Skip entire page border frame if it encloses almost the whole page (>92%)
        if box_area > (page_total_area * 0.92):
            continue

        # Skip headers / footers (thin banners at top or bottom)
        if (y < int(0.06 * page_h) or y + h > int(0.95 * page_h)) and h < int(0.08 * page_h):
            continue

        # Check aspect ratio
        aspect = float(w) / float(max(1, h))
        if aspect > max_aspect_ratio or aspect < (1.0 / max_aspect_ratio):
            continue

        candidate_boxes.append((int(x), int(y), int(w), int(h)))

    # Step 3: Remove subsumed / nested boxes
    filtered_boxes: List[Tuple[int, int, int, int]] = []
    for i, b1 in enumerate(candidate_boxes):
        x1, y1, w1, h1 = b1
        is_subsumed = False
        for j, b2 in enumerate(candidate_boxes):
            if i == j:
                continue
            x2, y2, w2, h2 = b2
            if x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1):
                is_subsumed = True
                break
        if not is_subsumed:
            filtered_boxes.append(b1)

    # Step 4: Handle single-design pages or fallback
    if len(filtered_boxes) == 0:
        # Fallback: Find bounding box of all white content in page_mask
        white_pixels = np.argwhere(page_mask == 255)
        if len(white_pixels) > 50:
            ymin, xmin = white_pixels.min(axis=0)
            ymax, xmax = white_pixels.max(axis=0)
            w = int(xmax - xmin + 1)
            h = int(ymax - ymin + 1)
            if w * h >= effective_min_area:
                return [(int(xmin), int(ymin), w, h)]
        # Default full page
        return [(0, 0, page_w, page_h)]

    # If only one candidate design exists, return it
    if len(filtered_boxes) == 1:
        return filtered_boxes

    # Step 5: Sort multiple designs in row-major order
    return sort_boxes_row_major(filtered_boxes)
