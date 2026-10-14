"""
Unit tests for image_crop module:
- order_points
- crop_rect
- warp_four_points (sharpness verification via variance of Laplacian)
- coordinate mapping (view <-> image)
"""

import cv2
import numpy as np
import pytest

from app.core.image_crop import (
    crop_rect,
    image_to_view_coords,
    order_points,
    sharpen,
    view_to_image_coords,
    warp_four_points,
)


def test_order_points():
    """Tests ordering 4 points into [TL, TR, BR, BL] regardless of input order."""
    # Input points in shuffled order: BL, TR, TL, BR
    pts = np.array([[20, 280], [380, 25], [15, 20], [390, 290]], dtype=np.float32)
    ordered = order_points(pts)

    assert ordered.shape == (4, 2)
    # TL: smallest sum -> [15, 20]
    assert np.allclose(ordered[0], [15, 20])
    # TR: smallest diff (y - x) -> [380, 25] (25 - 380 = -355)
    assert np.allclose(ordered[1], [380, 25])
    # BR: largest sum -> [390, 290]
    assert np.allclose(ordered[2], [390, 290])
    # BL: largest diff (y - x) -> [20, 280] (280 - 20 = 260)
    assert np.allclose(ordered[3], [20, 280])


def test_crop_rect_zero_loss():
    """Tests that crop_rect performs direct numpy slicing without resampling."""
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape((100, 100, 3))
    rect = [20.0, 30.0, 40.0, 50.0]
    cropped = crop_rect(img, rect)

    assert cropped.shape == (50, 40, 3)
    # Must be exact identity slice
    expected = img[30:80, 20:60]
    assert np.array_equal(cropped, expected)


def test_warp_skewed_checkerboard_sharpness():
    """
    Warps a synthetic skewed checkerboard and asserts:
    1. The output is rectangular
    2. Edge sharpness (variance of Laplacian) is not lower than the source
    """
    # 1. Create a regular checkerboard
    base_grid = np.zeros((400, 400), dtype=np.uint8)
    for r in range(0, 400, 40):
        for c in range(0, 400, 40):
            if ((r // 40) + (c // 40)) % 2 == 0:
                base_grid[r : r + 40, c : c + 40] = 255

    # 2. Skew the checkerboard with a perspective transformation (simulating a photo taken at an angle)
    src_corners = np.array([[50, 60], [350, 40], [380, 360], [30, 340]], dtype=np.float32)
    dst_corners = np.array([[80, 100], [320, 70], [360, 330], [50, 370]], dtype=np.float32)
    m_skew = cv2.getPerspectiveTransform(src_corners, dst_corners)
    skewed_source = cv2.warpPerspective(base_grid, m_skew, (400, 400), flags=cv2.INTER_LINEAR)
    skewed_bgr = cv2.cvtColor(skewed_source, cv2.COLOR_GRAY2BGR)

    # 3. Rectify using warp_four_points on dst_corners
    rectified = warp_four_points(skewed_bgr, dst_corners, sharpen_output=True, upscale_small=False)

    # Output must be 2D rectangular image
    assert rectified.ndim == 3
    assert rectified.shape[0] > 0
    assert rectified.shape[1] > 0
    assert rectified.dtype == np.uint8

    # Edge sharpness via variance of Laplacian
    def laplacian_var(im: np.ndarray) -> float:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    src_sharpness = laplacian_var(skewed_bgr)
    rect_sharpness = laplacian_var(rectified)

    # Assert output edge sharpness is not lower than source
    assert rect_sharpness >= src_sharpness, (
        f"Expected rectified sharpness ({rect_sharpness:.1f}) to be >= source ({src_sharpness:.1f})"
    )


def test_coordinate_mapping_roundtrip():
    """Tests view <-> image coordinate transformation accuracy."""
    img_w, img_h = 1920, 1080
    scale = 0.5
    off_x, off_y = 50.0, 30.0

    orig_ix, orig_iy = 640.0, 480.0
    vx, vy = image_to_view_coords(orig_ix, orig_iy, scale, off_x, off_y)
    assert vx == 50.0 + (640.0 * 0.5)
    assert vy == 30.0 + (480.0 * 0.5)

    res_ix, res_iy = view_to_image_coords(vx, vy, scale, off_x, off_y, img_w, img_h)
    assert abs(res_ix - orig_ix) < 1e-4
    assert abs(res_iy - orig_iy) < 1e-4
