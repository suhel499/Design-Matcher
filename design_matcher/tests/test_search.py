"""
Tests for search engine, multi-rotation matching, dual-polarity handling, and shape re-ranking.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core.rerank import compute_chamfer_similarity, compute_mask_iou
from app.core.search import rotate_image


def test_rotate_image():
    """Tests 90, 180, and 270 degree image rotations."""
    img = np.zeros((40, 20), dtype=np.uint8)
    img[0, 0] = 255  # top-left corner

    r90 = rotate_image(img, 90)
    assert r90.shape == (20, 40)
    assert r90[0, 39] == 255  # top-right corner

    r180 = rotate_image(img, 180)
    assert r180.shape == (40, 20)
    assert r180[39, 19] == 255  # bottom-right corner

    r270 = rotate_image(img, 270)
    assert r270.shape == (20, 40)
    assert r270[19, 0] == 255  # bottom-left corner


def test_mask_iou_exact_and_shifted():
    """Tests IoU computation between identical, disjoint, and partial masks."""
    m1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(m1, (20, 20), (80, 80), 255, -1)

    # Identical mask -> IoU == 1.0
    iou_identical = compute_mask_iou(m1, m1)
    assert abs(iou_identical - 1.0) < 1e-4

    # Disjoint mask -> IoU == 0.0
    m2 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(m2, (0, 0), (15, 15), 255, -1)
    iou_disjoint = compute_mask_iou(m1, m2)
    assert abs(iou_disjoint - 0.0) < 1e-4

    # Shifted mask -> IoU > 0 and < 1
    m3 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(m3, (30, 20), (90, 80), 255, -1)
    iou_shifted = compute_mask_iou(m1, m3)
    assert 0.5 < iou_shifted < 0.9


def test_chamfer_similarity():
    """Tests Chamfer edge similarity metric."""
    m1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(m1, (50, 50), 30, 255, -1)

    # Identical mask -> high similarity
    sim_identical = compute_chamfer_similarity(m1, m1)
    assert sim_identical > 0.95

    # Slightly dilated mask -> high similarity
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m2 = cv2.dilate(m1, k)
    sim_dilated = compute_chamfer_similarity(m1, m2)
    assert sim_dilated > 0.85

    # Completely different shape (square in corner)
    m3 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(m3, (5, 5), (25, 25), 255, -1)
    sim_diff = compute_chamfer_similarity(m1, m3)
    assert sim_diff < sim_identical

    # Mismatched dimensions (e.g. 100x100 vs 150x80) should not raise IndexError
    m4 = np.zeros((150, 80), dtype=np.uint8)
    cv2.circle(m4, (40, 75), 25, 255, -1)
    sim_mismatch = compute_chamfer_similarity(m1, m4)
    assert 0.0 <= sim_mismatch <= 1.0


def test_cad_wireframe_normalization():
    """Tests that CAD wireframe screenshots (cream background, thin strokes) normalize to white metal & black cuts."""
    from app.core.preprocess import is_cad_wireframe, normalize_stencil

    # Synthetic CAD screenshot: cream background (241, 238, 230) with thin maroon lines (70, 40, 140)
    img = np.full((300, 300, 3), (230, 238, 241), dtype=np.uint8)  # BGR
    # Draw geometric cutouts (circle and rectangle)
    cv2.circle(img, (150, 150), 60, (70, 40, 140), 2)
    cv2.rectangle(img, (50, 50), (250, 250), (70, 40, 140), 2)

    is_cad, bg_median = is_cad_wireframe(img)
    assert is_cad is True
    assert bg_median > 200

    mask, info = normalize_stencil(img, crop=False)
    assert info.get("is_cad") is True
    # White metal should be majority, not black void
    white_ratio = np.mean(mask == 255)
    assert white_ratio > 0.70
    # Center and lines should be cutouts (0)
    assert np.any(mask == 0)
