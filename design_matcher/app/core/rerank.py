"""
Re-ranking engine for Design Matcher.
Refines vector search candidate matches using geometrical shape similarity
(Chamfer distance, mask IoU after alignment, and keypoint matching).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between two binary masks.

    Args:
        mask1: First binary mask (uint8, 0/255).
        mask2: Second binary mask (uint8, 0/255).

    Returns:
        float: IoU score between 0.0 and 1.0.
    """
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)

    b1 = mask1 > 127
    b2 = mask2 > 127

    intersection = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_chamfer_similarity(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Computes Chamfer edge distance similarity between two binary masks.
    Returns normalized similarity score in range [0.0, 1.0].
    """
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Extract edges
    e1 = cv2.Canny(mask1, 50, 150)
    e2 = cv2.Canny(mask2, 50, 150)

    pts1 = np.column_stack(np.where(e1 > 0))
    pts2 = np.column_stack(np.where(e2 > 0))

    if len(pts1) == 0 or len(pts2) == 0:
        return 0.0

    # Distance transforms for fast Chamfer computation
    dt1 = cv2.distanceTransform(cv2.bitwise_not(e1), cv2.DIST_L2, 3)
    dt2 = cv2.distanceTransform(cv2.bitwise_not(e2), cv2.DIST_L2, 3)

    d1 = np.mean([dt2[pt[0], pt[1]] for pt in pts1])
    d2 = np.mean([dt1[pt[0], pt[1]] for pt in pts2])
    avg_dist = (d1 + d2) / 2.0

    # Exponential similarity decay
    sim = float(np.exp(-avg_dist / 15.0))
    return max(0.0, min(1.0, sim))
