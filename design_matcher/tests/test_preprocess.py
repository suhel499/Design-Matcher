"""
Unit tests for image preprocessing, watermark removal, stencil mask conversion, and letterboxing.
"""

import cv2
import numpy as np
import pytest

from app.core.preprocess import (
    convert_tile_to_stencil_mask,
    letterbox_image,
    prepare_mask_for_model,
    remove_pink_watermark,
)


def test_remove_pink_watermark_synthetic():
    """Tests that pink/magenta watermark pixels are detected and inpainted."""
    # Create a neutral gray image simulating a silver panel
    img = np.full((120, 120, 3), 200, dtype=np.uint8)

    # Draw a bold diagonal pink stripe (BGR: High Blue & Red, Low Green -> Magenta/Pink)
    # Pink in BGR: e.g. (180, 50, 240)
    cv2.line(img, (10, 10), (110, 110), (180, 50, 240), thickness=6)

    # Check that pink pixels exist initially
    hsv_initial = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pink_mask_initial = cv2.inRange(hsv_initial, np.array([150, 40, 50]), np.array([180, 255, 255]))
    assert cv2.countNonZero(pink_mask_initial) > 100

    # Inpaint watermark
    cleaned = remove_pink_watermark(img)

    # Verify that the pink watermark pixels are reduced by over 90%
    hsv_cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
    pink_mask_after = cv2.inRange(hsv_cleaned, np.array([150, 40, 50]), np.array([180, 255, 255]))
    assert cv2.countNonZero(pink_mask_after) < 15


def test_convert_tile_to_stencil_mask():
    """Tests conversion of a silver tile with dark cutouts into a binary stencil mask."""
    # Silver panel body: light gray (210, 210, 210)
    tile_bgr = np.full((150, 100, 3), 210, dtype=np.uint8)

    # Dark cutouts inside the panel: dark navy (20, 30, 50)
    # Cutout 1: circle in center
    cv2.circle(tile_bgr, (50, 50), 20, (50, 30, 20), -1)
    # Cutout 2: rectangular slot
    cv2.rectangle(tile_bgr, (25, 90), (75, 120), (50, 30, 20), -1)

    mask = convert_tile_to_stencil_mask(tile_bgr)

    # Verify mask properties
    assert mask.shape == (150, 100)
    assert mask.dtype == np.uint8
    # Unique values must only be 0 and 255
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255})

    # Center circle cutout should be 0 (black cutout)
    assert mask[50, 50] == 0
    # Outer frame border should be 255 (white solid metal)
    assert mask[5, 5] == 255
    assert mask[140, 5] == 255


def test_letterbox_image():
    """Tests that letterbox preserves aspect ratio and outputs fixed 224x224."""
    # Tall image: 200 high x 100 wide
    tall = np.full((200, 100), 255, dtype=np.uint8)
    lb_tall = letterbox_image(tall, target_size=(224, 224), pad_value=0)

    assert lb_tall.shape == (224, 224)
    # Scaled height should be 224, scaled width should be round(100 * 224/200) = 112
    # Left and right padding should be 0
    assert lb_tall[112, 10] == 0  # padding area
    assert lb_tall[112, 112] == 255  # centered content

    # Wide image: 100 high x 300 wide
    wide = np.full((100, 300), 255, dtype=np.uint8)
    lb_wide = letterbox_image(wide, target_size=(224, 224), pad_value=0)

    assert lb_wide.shape == (224, 224)
    # Top and bottom padding should be 0
    assert lb_wide[10, 112] == 0  # top padding
    assert lb_wide[112, 112] == 255  # center content


def test_prepare_mask_for_model():
    """Tests preparing a 224x224 mask into a normalized float32 CHW tensor."""
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[50:150, 50:150] = 255

    tensor = prepare_mask_for_model(mask)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    # ImageNet normalized values can be negative and positive
    assert np.min(tensor) < 0.0
    assert np.max(tensor) > 0.0


def test_enhance_photo_contrast_and_sharpness():
    """Tests that contrast and sharpness are boosted on blurry or low-contrast input."""
    from app.core.preprocess import enhance_photo_contrast_and_sharpness

    # Generate a soft blurred pattern
    base = np.full((120, 120, 3), 128, dtype=np.uint8)
    cv2.circle(base, (60, 60), 30, (80, 80, 80), -1)
    blurred = cv2.GaussianBlur(base, (7, 7), 2.5)

    enhanced = enhance_photo_contrast_and_sharpness(blurred, sharpness_boost=1.5, clahe_clip=2.5)

    assert enhanced.shape == blurred.shape
    assert enhanced.dtype == np.uint8
    # Edge gradient magnitude should be strictly sharper in enhanced image
    grad_blurred = cv2.Sobel(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0)
    grad_enhanced = cv2.Sobel(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0)
    assert np.max(np.abs(grad_enhanced)) > np.max(np.abs(grad_blurred))


def test_detect_stencil_polarity():
    """Tests structural perimeter polarity detection for raw metal vs dark installed gates."""
    from app.core.preprocess import detect_stencil_polarity

    # Case 1: Standard raw metal sheet (solid metal perimeter = 255/white, cutouts = 0)
    raw_sheet_mask = np.ones((200, 150), dtype=np.uint8) * 255
    raw_sheet_mask[30:170, 25:125] = 0  # internal aperture
    assert detect_stencil_polarity(raw_sheet_mask) is True

    # Case 2: Dark painted gate against bright sky (metal frame was labeled 0, daylight cutouts labeled 255)
    dark_gate_mask = cv2.bitwise_not(raw_sheet_mask)
    assert detect_stencil_polarity(dark_gate_mask) is False


def test_detect_sheet_quadrilateral_synthetic():
    """Tests multi-strategy 4-corner boundary detection on a sheet with background."""
    from app.core.preprocess import detect_sheet_quadrilateral, four_point_perspective_transform

    # Cardboard background (warm brownish-yellow: B=80, G=140, R=190)
    bg = np.zeros((400, 350, 3), dtype=np.uint8)
    bg[:, :] = (80, 140, 190)

    # Metal sheet in center (neutral silver gray: B=180, G=185, R=180)
    sheet_pts = np.array([[60, 40], [290, 50], [270, 360], [70, 350]], dtype=np.int32)
    cv2.fillPoly(bg, [sheet_pts], (180, 185, 180))

    # Add internal cutouts inside the sheet
    cv2.rectangle(bg, (110, 100), (230, 280), (80, 140, 190), -1)

    quad = detect_sheet_quadrilateral(bg)
    assert quad is not None
    assert quad.shape == (4, 2)

    # Warp perspective using detected quad
    warped = four_point_perspective_transform(bg, quad)
    assert warped.shape[0] > 100
    assert warped.shape[1] > 100


def test_convert_photo_to_stencil_mask_modes():
    """Tests convert_photo_to_stencil_mask with auto, raw_sheet, installed_product, and blurry modes."""
    from app.core.preprocess import convert_photo_to_stencil_mask

    # Simulated installed black gate: dark metal frame (30, 30, 30), bright daylight holes (240, 240, 240)
    gate_img = np.full((180, 140, 3), 30, dtype=np.uint8)
    cv2.rectangle(gate_img, (30, 30), (110, 150), (240, 240, 240), -1)

    # Installed product mode should automatically ensure metal frame is normalized to 255 (white)
    mask_gate = convert_photo_to_stencil_mask(gate_img, auto_perspective=False, mode="installed_product")
    assert mask_gate.shape == (180, 140)
    # The outer perimeter border must be 255 (white metal)
    assert mask_gate[10, 10] == 255
    # The interior hole must be 0 (black cutout)
    assert mask_gate[90, 70] == 0

    # Blurry mode test
    blurry_img = cv2.GaussianBlur(gate_img, (7, 7), 2.0)
    mask_blurry = convert_photo_to_stencil_mask(blurry_img, auto_perspective=False, mode="blurry")
    assert mask_blurry.shape == (180, 140)
    assert set(np.unique(mask_blurry)).issubset({0, 255})


def test_normalize_stencil_synthetic_light_and_dark_backdrops():
    """
    Tests normalize_stencil with synthetic images:
    1. Light panel on dark backdrop
    2. Dark panel on light backdrop
    Asserts that in both cases, the output mask always has metal as white (255)
    and cutouts as black (0).
    """
    from app.core.preprocess import normalize_stencil

    # 1. Light panel (intensity ~210) on dark backdrop (intensity ~20)
    img_light_on_dark = np.full((300, 300, 3), 20, dtype=np.uint8)  # dark backdrop
    img_light_on_dark[50:250, 50:250] = (210, 210, 210)  # light metal panel
    cv2.circle(img_light_on_dark, (150, 150), 30, (20, 20, 20), -1)  # dark cutout hole

    mask1, info1 = normalize_stencil(img_light_on_dark, crop=True)

    assert mask1.dtype == np.uint8
    assert set(np.unique(mask1)).issubset({0, 255})
    assert mask1.shape == (200, 200)  # cropped to metal bounding box
    assert info1["polarity_inverted"] is False
    # Center cutout must be 0 (black void)
    assert mask1[100, 100] == 0
    # Solid panel body must be 255 (white metal)
    assert mask1[10, 10] == 255
    # Metal must be the predominant class
    assert (mask1 == 255).mean() > 0.80

    # 2. Dark panel (intensity ~40) on light backdrop (intensity ~240)
    img_dark_on_light = np.full((300, 300, 3), 240, dtype=np.uint8)  # light backdrop
    img_dark_on_light[50:250, 50:250] = (40, 40, 40)  # dark metal panel
    cv2.circle(img_dark_on_light, (150, 150), 30, (240, 240, 240), -1)  # light cutout showing backdrop

    mask2, info2 = normalize_stencil(img_dark_on_light, crop=True)

    assert mask2.dtype == np.uint8
    assert set(np.unique(mask2)).issubset({0, 255})
    assert mask2.shape == (200, 200)  # cropped to metal bounding box
    assert info2["polarity_inverted"] is True
    # Center cutout must be 0 (black void)
    assert mask2[100, 100] == 0
    # Solid panel body must be 255 (white metal)
    assert mask2[10, 10] == 255
    # Metal must be the predominant class
    assert (mask2 == 255).mean() > 0.80


