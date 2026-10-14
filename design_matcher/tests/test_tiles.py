"""
Unit tests for tile detection, row-major sorting, and sub-region extraction.
Uses synthetic catalog pages with dark navy background, silver panels, pink watermark, and header.
"""

import cv2
import numpy as np
import pytest

from app.core.tiles import (
    detect_tile_boxes,
    extract_tile_crops,
    sort_boxes_row_major,
)


def create_synthetic_catalog_page(
    width: int = 1200,
    height: int = 1600,
    rows: int = 3,
    cols: int = 2,
    add_watermark: bool = True,
    add_header: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Creates a synthetic catalog page representing real client PDF pages:
    - Dark navy background
    - Header text banner
    - Silver panels with internal cutout patterns
    - Pink diagonal watermark
    """
    # Dark navy background (BGR: ~ (45, 25, 15))
    page = np.full((height, width, 3), (45, 25, 15), dtype=np.uint8)

    # Optional header at top
    if add_header:
        cv2.putText(
            page,
            "ARCHITECTURAL LASER CUT PANELS - CATALOG 2026",
            (80, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.line(page, (50, 70), (width - 50, 70), (100, 100, 100), 2)

    # Compute panel positions
    margin_top = 120
    margin_side = 80
    margin_bottom = 60
    usable_w = width - 2 * margin_side
    usable_h = height - margin_top - margin_bottom

    gap_x = 60
    gap_y = 50

    panel_w = (usable_w - (cols - 1) * gap_x) // cols
    panel_h = (usable_h - (rows - 1) * gap_y) // rows

    expected_boxes: list[tuple[int, int, int, int]] = []

    for r in range(rows):
        for c in range(cols):
            x = margin_side + c * (panel_w + gap_x)
            y = margin_top + r * (panel_h + gap_y)

            # Draw silver panel body (BGR: (215, 215, 215))
            page[y : y + panel_h, x : x + panel_w] = (215, 215, 215)

            # Draw internal cut-outs (dark navy back-fill)
            # Center circular cutout
            cx = x + panel_w // 2
            cy = y + panel_h // 2
            cr = min(panel_w, panel_h) // 4
            cv2.circle(page, (cx, cy), cr, (45, 25, 15), -1)

            # Decorative cut slots
            cv2.rectangle(
                page,
                (x + 25, y + 25),
                (x + panel_w - 25, y + 45),
                (45, 25, 15),
                -1,
            )
            cv2.rectangle(
                page,
                (x + 25, y + panel_h - 45),
                (x + panel_w - 25, y + panel_h - 25),
                (45, 25, 15),
                -1,
            )

            # Draw design number beneath panel
            cv2.putText(
                page,
                f"DES-{r*cols + c + 101}",
                (x + 20, min(height - 10, y + panel_h + 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

            expected_boxes.append((x, y, panel_w, panel_h))

    # Add diagonal pink watermark across the page
    if add_watermark:
        cv2.line(page, (0, height // 4), (width, height * 3 // 4), (180, 50, 240), 8)
        cv2.putText(
            page,
            "CONFIDENTIAL WATERMARK",
            (200, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (180, 50, 240),
            4,
            cv2.LINE_AA,
        )

    return page, expected_boxes


def test_detect_tile_boxes_synthetic():
    """Tests detection of silver tiles in the presence of watermarks and headers."""
    rows, cols = 3, 2
    page, expected_boxes = create_synthetic_catalog_page(
        width=1000,
        height=1400,
        rows=rows,
        cols=cols,
        add_watermark=True,
        add_header=True,
    )

    detected_boxes = detect_tile_boxes(page)

    # Should detect all 6 panels
    assert len(detected_boxes) == rows * cols

    # Verify row-major ordering and approximate coordinates
    for i, (det, exp) in enumerate(zip(detected_boxes, expected_boxes)):
        dx, dy, dw, dh = det
        ex, ey, ew, eh = exp

        # Tolerance within 10 pixels
        assert abs(dx - ex) < 15, f"Tile {i} x mismatch: {dx} vs {exp[0]}"
        assert abs(dy - ey) < 15, f"Tile {i} y mismatch: {dy} vs {exp[1]}"
        assert abs(dw - ew) < 15, f"Tile {i} width mismatch: {dw} vs {exp[2]}"
        assert abs(dh - eh) < 15, f"Tile {i} height mismatch: {dh} vs {exp[3]}"


def test_sort_boxes_row_major():
    """Tests row-major sorting of jittered 2D boxes."""
    # Jittered boxes in a 2x2 grid
    b_top_left = (50, 102, 100, 100)
    b_top_right = (200, 98, 100, 100)
    b_bot_left = (52, 250, 100, 100)
    b_bot_right = (205, 248, 100, 100)

    # Shuffled order
    shuffled = [b_bot_right, b_top_right, b_bot_left, b_top_left]
    sorted_res = sort_boxes_row_major(shuffled)

    assert sorted_res[0] == b_top_left
    assert sorted_res[1] == b_top_right
    assert sorted_res[2] == b_bot_left
    assert sorted_res[3] == b_bot_right


def test_extract_tile_crops_and_subregions():
    """Tests extraction of full tile and sub-region crops."""
    page, expected_boxes = create_synthetic_catalog_page(rows=1, cols=1)
    tile_boxes = [expected_boxes[0]]

    crops = extract_tile_crops(page, tile_boxes, include_subregions=True, run_ocr=False)

    crop_types = [c.crop_type for c in crops]
    assert "full" in crop_types
    assert "top_half" in crop_types
    assert "bottom_half" in crop_types
    assert "left_half" in crop_types
    assert "right_half" in crop_types
    assert "center" in crop_types

    # Full crop mask verification
    full_crop = next(c for c in crops if c.crop_type == "full")
    assert full_crop.stencil_mask.ndim == 2
    assert set(np.unique(full_crop.stencil_mask)).issubset({0, 255})


def test_split_wide_box_seams():
    """Tests splitting of merged adjacent tiles with a vertical seam valley."""
    from app.core.tiles import split_wide_box_seams

    # Create two silver panels side by side with a dark seam in the middle
    h = 200
    w_left, w_right = 140, 140
    gap = 10
    total_w = w_left + gap + w_right

    crop_bgr = np.full((h, total_w, 3), 40, dtype=np.uint8)  # dark background
    crop_bgr[:, :w_left] = (210, 210, 210)  # left panel
    crop_bgr[:, w_left + gap :] = (210, 210, 210)  # right panel

    splits = split_wide_box_seams(crop_bgr, min_aspect=1.15)
    assert len(splits) == 2
    s1, s2 = splits
    # First panel starts at 0, second starts around w_left
    assert s1[0] == 0
    assert abs(s1[1] - w_left) <= gap
    assert s2[0] >= w_left
