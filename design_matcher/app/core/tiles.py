"""
Tile detection, segmentation, and sub-region extraction from catalog pages.
Detects bright silver rectangles on dark backgrounds, sorts them row-major,
extracts stencil masks and overlapping sub-regions for indexing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from app.core.preprocess import (
    convert_tile_to_stencil_mask,
    normalize_stencil,
    remove_pink_watermark,
)

logger = logging.getLogger(__name__)

# Check pytesseract availability
try:
    import pytesseract  # type: ignore

    _HAS_PYTESSERACT = True
except ImportError:
    _HAS_PYTESSERACT = False


@dataclass
class TileCrop:
    """Represents a cropped tile or sub-region ready for embedding."""
    tile_index: int  # 0-indexed order on the page
    crop_type: str  # 'full', 'top_half', 'bottom_half', 'left_half', 'right_half', 'center'
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in original page coordinates
    crop_bgr: np.ndarray  # Cropped BGR image
    stencil_mask: np.ndarray  # Binary stencil mask (uint8, 0/255)
    design_number: Optional[str] = None


def extract_design_number_ocr(
    page_bgr: np.ndarray,
    tile_bbox: Tuple[int, int, int, int],
) -> Optional[str]:
    """
    Attempts to read the design number printed beneath a tile box using pytesseract.
    Returns None if pytesseract is unavailable or no text is detected.
    """
    if not _HAS_PYTESSERACT:
        return None

    x, y, w, h = tile_bbox
    page_h, page_w = page_bgr.shape[:2]

    # Region directly below the tile
    label_y1 = y + h
    label_h = min(int(h * 0.22), page_h - label_y1)
    if label_h <= 5:
        return None

    # Slightly expand horizontally to catch label centering
    pad_x = int(w * 0.1)
    label_x1 = max(0, x - pad_x)
    label_x2 = min(page_w, x + w + pad_x)

    label_crop = page_bgr[label_y1 : label_y1 + label_h, label_x1:label_x2]
    if label_crop.size == 0:
        return None

    try:
        gray = cv2.cvtColor(label_crop, cv2.COLOR_BGR2GRAY)
        # Often text is white/light on dark navy or dark on light; Otsu handles both
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if dark text on light background
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        text = pytesseract.image_to_string(
            binary,
            config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-._#",
        ).strip()

        # Sanitize text
        cleaned = "".join(c for c in text if c.isalnum() or c in "-_#.").strip()
        return cleaned if cleaned else None
    except Exception as exc:
        logger.debug("Tesseract OCR skipped: %s", exc)
        return None


def sort_boxes_row_major(
    boxes: List[Tuple[int, int, int, int]],
    y_threshold_fraction: float = 0.45,
) -> List[Tuple[int, int, int, int]]:
    """
    Sorts 2D bounding boxes in natural row-major reading order (top-to-bottom, left-to-right).
    Robust against slight row misalignments and slight height variations.

    Args:
        boxes: List of (x, y, w, h) boxes.
        y_threshold_fraction: Fraction of median height used to group into the same row.

    Returns:
        Sorted list of bounding boxes.
    """
    if not boxes:
        return []
    if len(boxes) == 1:
        return boxes

    # Calculate median height to determine row banding threshold
    heights = [h for _, _, _, h in boxes]
    median_h = float(np.median(heights))
    y_threshold = max(20.0, median_h * y_threshold_fraction)

    # Sort initially by Y center
    boxes_with_centers = [
        (box, box[0] + box[2] / 2.0, box[1] + box[3] / 2.0) for box in boxes
    ]
    boxes_with_centers.sort(key=lambda item: item[2])  # sort by y_center

    # Group into rows
    rows: List[List[Tuple[Tuple[int, int, int, int], float, float]]] = []
    current_row: List[Tuple[Tuple[int, int, int, int], float, float]] = []
    current_row_y = 0.0

    for item in boxes_with_centers:
        _, _, yc = item
        if not current_row:
            current_row.append(item)
            current_row_y = yc
        else:
            if abs(yc - current_row_y) <= y_threshold:
                current_row.append(item)
                # Update running average Y of current row
                current_row_y = sum(x[2] for x in current_row) / len(current_row)
            else:
                rows.append(current_row)
                current_row = [item]
                current_row_y = yc

    if current_row:
        rows.append(current_row)

    # Within each row, sort left to right by x center
    sorted_boxes: List[Tuple[int, int, int, int]] = []
    for row in rows:
        row.sort(key=lambda item: item[1])  # sort by xc
        for item in row:
            sorted_boxes.append(item[0])

    return sorted_boxes


def split_wide_box_seams(crop_bgr: np.ndarray, min_aspect: float = 1.15) -> List[Tuple[int, int]]:
    """
    Recursively splits wide candidate bounding boxes that contain vertical seam valleys.
    Returns list of (x_rel, w_rel).
    """
    h, w = crop_bgr.shape[:2]
    if w < 60 or h < 60 or (w / h) < min_aspect:
        return [(0, w)]

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr.copy()
    col_means = np.mean(gray, axis=0)

    # Smooth column means
    k = max(3, int(w * 0.015))
    if k % 2 == 0:
        k += 1
    smooth = np.convolve(col_means, np.ones(k) / k, mode="same")

    median_val = float(np.median(smooth))
    # Valleys must be in the central interior (at least 18% away from edges)
    margin = int(w * 0.18)
    if margin >= w - margin:
        return [(0, w)]

    interior = smooth[margin : w - margin]
    min_idx = int(np.argmin(interior)) + margin
    min_val = float(smooth[min_idx])

    # Seam detection criteria:
    # 1. Dark valley (intensity < 115 and drop of at least 30% below median panel brightness)
    if min_val < 115 and min_val < (median_val * 0.70):
        # Find continuous valley segment around min_idx
        thresh = min_val + 25
        left_bound = min_idx
        while left_bound > margin and smooth[left_bound] < thresh:
            left_bound -= 1
        right_bound = min_idx
        while right_bound < w - margin and smooth[right_bound] < thresh:
            right_bound += 1

        split_x = (left_bound + right_bound) // 2
        if split_x <= margin or split_x >= w - margin:
            return [(0, w)]

        # Recursively check left and right pieces
        left_crop = crop_bgr[:, :split_x]
        right_crop = crop_bgr[:, split_x:]

        left_splits = split_wide_box_seams(left_crop, min_aspect=min_aspect)
        right_splits = split_wide_box_seams(right_crop, min_aspect=min_aspect)

        results: List[Tuple[int, int]] = []
        for x_rel, w_sub in left_splits:
            results.append((x_rel, w_sub))
        for x_rel, w_sub in right_splits:
            results.append((split_x + x_rel, w_sub))
        return results

    return [(0, w)]


def detect_tile_boxes(
    page_bgr: np.ndarray,
    min_area_fraction: float = 0.005,
    max_area_fraction: float = 0.65,
) -> List[Tuple[int, int, int, int]]:
    """
    Detects design tile bounding boxes on a catalog page.

    Args:
        page_bgr: Catalog page BGR image (~200 DPI).
        min_area_fraction: Minimum bounding box area as fraction of page area.
        max_area_fraction: Maximum bounding box area as fraction of page area (ignores whole page boxes).

    Returns:
        List of bounding boxes (x, y, w, h) sorted in row-major order.
    """
    if page_bgr is None or page_bgr.size == 0:
        return []

    page_h, page_w = page_bgr.shape[:2]
    page_area = float(page_h * page_w)

    # Step 1: Remove pink watermark
    clean_page = remove_pink_watermark(page_bgr)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(clean_page, cv2.COLOR_BGR2GRAY)

    # Step 3: Bright silver panel thresholding
    # Navy background is dark (< 80), silver panels are bright (> 120)
    # Using Gaussian blur + Otsu thresholding separates the panels cleanly
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Morphological closing to bridge cut-out apertures inside panels
    # Use a moderate kernel size (7x7) with 1 iteration to bridge internal cutouts
    # while preserving distinct gaps between adjacent panels and text labels.
    kernel_size = 7
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Step 5: Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = page_area * min_area_fraction
    max_area = page_area * max_area_fraction

    candidate_boxes: List[Tuple[int, int, int, int]] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box_area = float(w * h)

        # Ignore tiny noise or massive whole-page boxes
        if box_area < min_area or box_area > max_area:
            continue

        # Ignore thin headers / banners at very top of page
        if y < 0.05 * page_h and h < 0.08 * page_h:
            continue

        # Aspect ratio check (tiles are generally rectangular panels)
        aspect = float(w) / float(h)
        if aspect < 0.12 or aspect > 8.0:
            continue

        # Contour fill ratio (rectangularity)
        fill_ratio = cv2.contourArea(cnt) / box_area
        if fill_ratio < 0.40:
            continue

        candidate_boxes.append((x, y, w, h))

    # Remove overlapping or nested boxes if any (keep the outer one)
    filtered_boxes: List[Tuple[int, int, int, int]] = []
    for i, b1 in enumerate(candidate_boxes):
        x1, y1, w1, h1 = b1
        is_subsumed = False
        for j, b2 in enumerate(candidate_boxes):
            if i == j:
                continue
            x2, y2, w2, h2 = b2
            # Check if b1 is inside b2
            if x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1):
                is_subsumed = True
                break
        if not is_subsumed:
            filtered_boxes.append(b1)

    # Step 6: Split merged wide boxes if they contain vertical seam valleys
    seam_split_boxes: List[Tuple[int, int, int, int]] = []
    for x, y, w, h in filtered_boxes:
        crop = clean_page[y : y + h, x : x + w]
        splits = split_wide_box_seams(crop, min_aspect=1.15)
        for x_sub, w_sub in splits:
            seam_split_boxes.append((x + x_sub, y, w_sub, h))

    # Step 7: Sort in row-major order
    sorted_boxes = sort_boxes_row_major(seam_split_boxes)
    logger.debug("Detected %d tile boxes on page of size %dx%d", len(sorted_boxes), page_w, page_h)
    return sorted_boxes


def extract_tile_crops(
    page_bgr: np.ndarray,
    tile_boxes: List[Tuple[int, int, int, int]],
    include_subregions: bool = True,
    run_ocr: bool = True,
    page_mask: Optional[np.ndarray] = None,
) -> List[TileCrop]:
    """
    Extracts full tile crops and overlapping sub-regions for each detected box on a page.
    Uses normalize_stencil applied across the rendered catalog PDF page for uniform polarity.

    Args:
        page_bgr: Cleaned page image (BGR).
        tile_boxes: Detected tile bounding boxes in row-major order.
        include_subregions: If True, also generates top/bottom/left/right/center crops.
        run_ocr: If True and pytesseract available, attempts to read design number.
        page_mask: Optional pre-normalized stencil mask for the entire page.

    Returns:
        List of TileCrop instances ready for stencil mask processing and embedding.
    """
    results: List[TileCrop] = []
    page_h, page_w = page_bgr.shape[:2]

    if page_mask is None and page_bgr is not None and page_bgr.size > 0:
        page_mask, _ = normalize_stencil(page_bgr, crop=False)

    for tile_idx, (x, y, w, h) in enumerate(tile_boxes):
        # Ensure coordinates are safely within page boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(page_w, x + w)
        y2 = min(page_h, y + h)

        tile_bgr = page_bgr[y1:y2, x1:x2].copy()
        if tile_bgr.size == 0:
            continue

        # Optional OCR for design number
        design_no = extract_design_number_ocr(page_bgr, (x, y, w, h)) if run_ocr else None

        # Full tile mask from normalized page mask
        full_mask = page_mask[y1:y2, x1:x2].copy() if page_mask is not None else convert_tile_to_stencil_mask(tile_bgr)
        results.append(
            TileCrop(
                tile_index=tile_idx,
                crop_type="full",
                bbox=(x, y, w, h),
                crop_bgr=tile_bgr,
                stencil_mask=full_mask,
                design_number=design_no,
            )
        )

        if not include_subregions:
            continue

        th, tw = tile_bgr.shape[:2]

        # Generate overlapping sub-regions for partial matching
        # Top half (60% height)
        top_h = int(th * 0.60)
        top_crop = tile_bgr[:top_h, :]
        top_mask = full_mask[:top_h, :].copy() if full_mask is not None else convert_tile_to_stencil_mask(top_crop)
        results.append(
            TileCrop(
                tile_index=tile_idx,
                crop_type="top_half",
                bbox=(x, y, w, top_h),
                crop_bgr=top_crop,
                stencil_mask=top_mask,
                design_number=design_no,
            )
        )

        # Bottom half (60% height)
        bot_y = int(th * 0.40)
        bot_crop = tile_bgr[bot_y:, :]
        bot_mask = full_mask[bot_y:, :].copy() if full_mask is not None else convert_tile_to_stencil_mask(bot_crop)
        results.append(
            TileCrop(
                tile_index=tile_idx,
                crop_type="bottom_half",
                bbox=(x, y + bot_y, w, th - bot_y),
                crop_bgr=bot_crop,
                stencil_mask=bot_mask,
                design_number=design_no,
            )
        )

        # Left half (60% width)
        left_w = int(tw * 0.60)
        left_crop = tile_bgr[:, :left_w]
        left_mask = full_mask[:, :left_w].copy() if full_mask is not None else convert_tile_to_stencil_mask(left_crop)
        results.append(
            TileCrop(
                tile_index=tile_idx,
                crop_type="left_half",
                bbox=(x, y, left_w, h),
                crop_bgr=left_crop,
                stencil_mask=left_mask,
                design_number=design_no,
            )
        )

        # Right half (60% width)
        right_x = int(tw * 0.40)
        right_crop = tile_bgr[:, right_x:]
        right_mask = full_mask[:, right_x:].copy() if full_mask is not None else convert_tile_to_stencil_mask(right_crop)
        results.append(
            TileCrop(
                tile_index=tile_idx,
                crop_type="right_half",
                bbox=(x + right_x, y, tw - right_x, h),
                crop_bgr=right_crop,
                stencil_mask=right_mask,
                design_number=design_no,
            )
        )

        # Center sub-region (60% width, 60% height)
        cy1, cy2 = int(th * 0.20), int(th * 0.80)
        cx1, cx2 = int(tw * 0.20), int(tw * 0.80)
        center_crop = tile_bgr[cy1:cy2, cx1:cx2]
        center_mask = full_mask[cy1:cy2, cx1:cx2].copy() if full_mask is not None else convert_tile_to_stencil_mask(center_crop)
        results.append(
            TileCrop(
                tile_index=tile_idx,
                crop_type="center",
                bbox=(x + cx1, y + cy1, cx2 - cx1, cy2 - cy1),
                crop_bgr=center_crop,
                stencil_mask=center_mask,
                design_number=design_no,
            )
        )

    return results
