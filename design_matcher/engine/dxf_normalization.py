"""
dxf_normalization.py - Geometry Normalization and Multi-Modal Feature Extraction.

Normalizes vector geometry (from DXF) and raster stencil shapes (from PNG) into
consistent, coordinate- and scale-invariant representations:
- Centers coordinate origin to geometric centroid.
- Normalizes drawing scale to unit bounds while preserving true aspect ratio.
- Uniform arc-length resampling across all polylines so vertex density and segmentation
  do not affect matching.
- Extracts robust shape descriptors for Stage 1 candidate retrieval:
  * Aspect ratio & fill ratio
  * Radial-distance histograms from centroid (rotation-invariant)
  * Pairwise-distance distribution descriptors (rotation- and mirror-invariant)
  * Contour and hole topology statistics
  * Invariant Hu shape moments
  * Compact 128-dimensional L2-normalized descriptor vector
- Produces canonical point clouds and canonical 256x256 stencil masks for Stage 2 geometric verification.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.distance import pdist

from .dxf_parser import ExtractedComponent, ExtractedPolyline

logger = logging.getLogger(__name__)

CANONICAL_POINTS_COUNT = 600
DESCRIPTOR_DIM = 128


@dataclass
class NormalizedGeometry:
    """Standardized representation of a 2D design ready for multi-stage matching."""
    component_name: str
    aspect_ratio: float
    width: float
    height: float
    centroid: Tuple[float, float]
    scale: float                               # Diagonal or max-side normalization scale
    point_cloud: np.ndarray                    # (K, 2) float32 coordinates centered at (0, 0), scaled [-1, 1]
    polylines: List[np.ndarray]                # List of (N, 2) float32 normalized polylines
    closed_flags: List[bool]
    radial_hist: np.ndarray                    # (32,) float32 radial distance histogram
    pairwise_hist: np.ndarray                  # (32,) float32 pairwise distance histogram
    hu_moments: np.ndarray                     # (7,) float32 log-transformed Hu moments
    feature_vector: np.ndarray                 # (128,) float32 L2-normalized retrieval descriptor
    contour_count: int
    hole_count: int
    canonical_mask: np.ndarray                 # (256, 256) uint8 binary raster mask


def _resample_polyline_uniform(pts: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Resamples a polyline uniformly along its cumulative arc length.
    Ensures designs with different vertex segmentations produce identical point distributions.
    """
    if len(pts) <= 1:
        return pts
    if len(pts) == 2 and num_samples == 2:
        return pts

    # Compute cumulative distance along vertices
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    total_len = seg_lens.sum()

    if total_len < 1e-7:
        return np.repeat(pts[:1], num_samples, axis=0)

    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    target_dists = np.linspace(0.0, total_len, num_samples)

    x_interp = np.interp(target_dists, cum_dist, pts[:, 0])
    y_interp = np.interp(target_dists, cum_dist, pts[:, 1])

    return np.stack([x_interp, y_interp], axis=1).astype(np.float32)


def _compute_radial_histogram(pts: np.ndarray, num_bins: int = 32) -> np.ndarray:
    """Computes a normalized radial distance histogram from the centroid."""
    if len(pts) == 0:
        return np.zeros(num_bins, dtype=np.float32)

    radii = np.hypot(pts[:, 0], pts[:, 1])
    max_r = radii.max()
    if max_r < 1e-6:
        hist = np.zeros(num_bins, dtype=np.float32)
        hist[0] = 1.0
        return hist

    norm_r = radii / max_r
    hist, _ = np.histogram(norm_r, bins=num_bins, range=(0.0, 1.0), density=True)
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    return (hist / norm) if norm > 1e-7 else hist


def _compute_pairwise_histogram(pts: np.ndarray, num_bins: int = 32, max_sample: int = 250) -> np.ndarray:
    """Computes a normalized histogram of pairwise distances between sampled points."""
    if len(pts) < 2:
        return np.zeros(num_bins, dtype=np.float32)

    # Subsample if necessary to keep pairwise computation instantaneous
    if len(pts) > max_sample:
        indices = np.linspace(0, len(pts) - 1, max_sample, dtype=int)
        sample_pts = pts[indices]
    else:
        sample_pts = pts

    dists = pdist(sample_pts)
    if len(dists) == 0 or dists.max() < 1e-6:
        hist = np.zeros(num_bins, dtype=np.float32)
        hist[0] = 1.0
        return hist

    norm_d = dists / dists.max()
    hist, _ = np.histogram(norm_d, bins=num_bins, range=(0.0, 1.0), density=True)
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    return (hist / norm) if norm > 1e-7 else hist


def _compute_hu_moments_from_mask(mask: np.ndarray) -> np.ndarray:
    """Computes 7 log-scale Hu moments from a binary mask."""
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    # Log scale transform: -sign(h) * log10(abs(h))
    log_hu = np.zeros(7, dtype=np.float32)
    for i in range(7):
        val = hu[i]
        if abs(val) > 1e-20:
            log_hu[i] = -math.copysign(1.0, val) * math.log10(abs(val))
    norm = np.linalg.norm(log_hu)
    return (log_hu / norm) if norm > 1e-7 else log_hu


def _render_canonical_mask_from_polylines(
    norm_polys: List[np.ndarray],
    closed_flags: List[bool],
    target_size: int = 256,
) -> np.ndarray:
    """
    Renders normalized polylines (coordinates in [-1, 1]) onto a standard 256x256 canvas.
    Standard convention: 255 = solid metal, 0 = cutouts/slits.
    """
    canvas = np.full((target_size, target_size), 255, dtype=np.uint8)
    margin = 12
    avail = target_size - 2 * margin

    # Find bounding box in normalized space
    all_pts = np.concatenate([p for p in norm_polys if len(p) > 0], axis=0) if norm_polys else np.zeros((1, 2))
    min_x, min_y = all_pts[:, 0].min(), all_pts[:, 1].min()
    max_x, max_y = all_pts[:, 0].max(), all_pts[:, 1].max()

    span_x = max(1e-4, max_x - min_x)
    span_y = max(1e-4, max_y - min_y)
    scale = avail / max(span_x, span_y)

    # Center in canvas
    ox = margin + (avail - span_x * scale) / 2.0
    oy = margin + (avail - span_y * scale) / 2.0

    # Draw lines
    line_canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    for pts, is_closed in zip(norm_polys, closed_flags):
        if len(pts) < 2:
            continue
        pix_x = np.round(ox + (pts[:, 0] - min_x) * scale).astype(np.int32)
        pix_y = np.round(oy + (max_y - pts[:, 1]) * scale).astype(np.int32)  # Flip Y
        pix_pts = np.stack([pix_x, pix_y], axis=1)
        cv2.polylines(line_canvas, [pix_pts], isClosed=bool(is_closed), color=255, thickness=2, lineType=cv2.LINE_AA)

    # Cutout regions: dilate lines and subtract from metal canvas
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(line_canvas, k, iterations=1)
    canvas[dilated > 0] = 0
    return canvas


def normalize_dxf_component(component: ExtractedComponent) -> NormalizedGeometry:
    """
    Normalizes a DXF ExtractedComponent into a NormalizedGeometry representation.
    """
    polylines = component.polylines
    if not polylines:
        dummy_pts = np.zeros((CANONICAL_POINTS_COUNT, 2), dtype=np.float32)
        dummy_desc = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
        dummy_mask = np.full((256, 256), 255, dtype=np.uint8)
        return NormalizedGeometry(
            component_name=component.name,
            aspect_ratio=1.0,
            width=0.0,
            height=0.0,
            centroid=(0.0, 0.0),
            scale=1.0,
            point_cloud=dummy_pts,
            polylines=[],
            closed_flags=[],
            radial_hist=np.zeros(32, dtype=np.float32),
            pairwise_hist=np.zeros(32, dtype=np.float32),
            hu_moments=np.zeros(7, dtype=np.float32),
            feature_vector=dummy_desc,
            contour_count=0,
            hole_count=0,
            canonical_mask=dummy_mask,
        )

    # 1. Collect all points to compute true centroid and scale
    all_raw_pts = np.concatenate([p.points for p in polylines if len(p.points) > 0], axis=0)
    bx0, by0, bx1, by1 = component.bbox
    cw = max(1e-4, bx1 - bx0)
    ch = max(1e-4, by1 - by0)
    aspect_ratio = float(cw / ch)

    cx = float(np.mean(all_raw_pts[:, 0]))
    cy = float(np.mean(all_raw_pts[:, 1]))
    max_dimension = max(cw, ch)
    scale = float(max_dimension / 2.0) if max_dimension > 1e-6 else 1.0

    # 2. Translate and scale all polylines to [-1, 1] frame
    norm_polys: List[np.ndarray] = []
    closed_flags: List[bool] = []
    total_arc_length = 0.0
    poly_lengths: List[float] = []

    for pl in polylines:
        pts = pl.points
        if len(pts) < 2:
            continue
        norm_pts = np.empty_like(pts, dtype=np.float32)
        norm_pts[:, 0] = (pts[:, 0] - cx) / scale
        norm_pts[:, 1] = (pts[:, 1] - cy) / scale
        norm_polys.append(norm_pts)
        closed_flags.append(pl.is_closed)

        diffs = np.diff(norm_pts, axis=0)
        plen = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
        poly_lengths.append(plen)
        total_arc_length += plen

    if total_arc_length < 1e-6:
        total_arc_length = 1.0

    # 3. Resample polylines proportionally into fixed CANONICAL_POINTS_COUNT point cloud
    resampled_segments: List[np.ndarray] = []
    allocated_points = 0

    for i, norm_pts in enumerate(norm_polys):
        ratio = poly_lengths[i] / total_arc_length
        num_pts = max(2, int(round(ratio * CANONICAL_POINTS_COUNT)))
        allocated_points += num_pts
        seg_resampled = _resample_polyline_uniform(norm_pts, num_pts)
        resampled_segments.append(seg_resampled)

    point_cloud = np.concatenate(resampled_segments, axis=0) if resampled_segments else np.zeros((CANONICAL_POINTS_COUNT, 2), dtype=np.float32)
    # Ensure exact shape (CANONICAL_POINTS_COUNT, 2)
    if len(point_cloud) > CANONICAL_POINTS_COUNT:
        point_cloud = point_cloud[:CANONICAL_POINTS_COUNT]
    elif len(point_cloud) < CANONICAL_POINTS_COUNT:
        pad = np.repeat(point_cloud[-1:], CANONICAL_POINTS_COUNT - len(point_cloud), axis=0)
        point_cloud = np.concatenate([point_cloud, pad], axis=0)

    # 4. Compute descriptors
    radial_hist = _compute_radial_histogram(point_cloud, num_bins=32)
    pairwise_hist = _compute_pairwise_histogram(point_cloud, num_bins=32)

    # Render canonical mask fast directly from normalized polylines
    canonical_mask = _render_canonical_mask_from_polylines(norm_polys, closed_flags, target_size=256)

    hu_moments = _compute_hu_moments_from_mask(canonical_mask)

    # Downsampled 8x8 spatial grid from canonical mask (64 elements)
    grid_8x8 = cv2.resize(canonical_mask, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0
    grid_norm = np.linalg.norm(grid_8x8)
    if grid_norm > 1e-7:
        grid_8x8 = grid_8x8 / grid_norm

    # Statistical features (128 - 32 - 32 - 7 - 48 = 9 features)
    contour_count = len(norm_polys)
    closed_count = sum(1 for c in closed_flags if c)
    hole_count = max(0, closed_count - 1)
    fill_ratio = float((canonical_mask == 255).mean())
    log_ar = float(math.log(max(1e-2, min(100.0, aspect_ratio))))
    stats = np.array([
        log_ar,
        fill_ratio,
        min(1.0, contour_count / 50.0),
        min(1.0, closed_count / 30.0),
        min(1.0, hole_count / 30.0),
        min(1.0, total_arc_length / 20.0),
        float(point_cloud[:, 0].std()),
        float(point_cloud[:, 1].std()),
        float(np.hypot(point_cloud[:, 0], point_cloud[:, 1]).std()),
    ], dtype=np.float32)

    # Assemble 128-d descriptor vector:
    # 32 (radial) + 32 (pairwise) + 7 (hu) + 9 (stats) + 48 (grid first 48) = 128
    fv = np.concatenate([radial_hist, pairwise_hist, hu_moments, stats, grid_8x8[:48]]).astype(np.float32)
    fv_norm = np.linalg.norm(fv)
    feature_vector = (fv / fv_norm) if fv_norm > 1e-7 else fv

    return NormalizedGeometry(
        component_name=component.name,
        aspect_ratio=aspect_ratio,
        width=cw,
        height=ch,
        centroid=(cx, cy),
        scale=scale,
        point_cloud=point_cloud,
        polylines=norm_polys,
        closed_flags=closed_flags,
        radial_hist=radial_hist,
        pairwise_hist=pairwise_hist,
        hu_moments=hu_moments,
        feature_vector=feature_vector,
        contour_count=contour_count,
        hole_count=hole_count,
        canonical_mask=canonical_mask,
    )


def normalize_raster_stencil(
    stencil_mask: np.ndarray,
    label: str = "Query Stencil",
) -> NormalizedGeometry:
    """
    Normalizes a binary PNG/raster stencil mask (255=metal, 0=void) into the
    exact same NormalizedGeometry representation used for DXF designs.
    Ensures 100% geometric descriptor compatibility between PNG queries and DXF designs.
    """
    if stencil_mask is None or stencil_mask.size == 0:
        stencil_mask = np.full((256, 256), 255, dtype=np.uint8)

    if stencil_mask.ndim == 3:
        stencil_mask = cv2.cvtColor(stencil_mask, cv2.COLOR_BGR2GRAY)

    _, bin_mask = cv2.threshold(stencil_mask, 127, 255, cv2.THRESH_BINARY)
    h, w = bin_mask.shape[:2]

    # Find external contours and hole contours
    contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # If no contours found, fallback
    if not contours:
        contours = [np.array([[[10, 10]], [[w - 10, 10]], [[w - 10, h - 10]], [[10, h - 10]]], dtype=np.int32)]

    # Compute bounding box of non-zero pixels
    coords = cv2.findNonZero(bin_mask)
    if coords is not None:
        bx, by, bw, bh = cv2.boundingRect(coords)
    else:
        bx, by, bw, bh = 0, 0, w, h

    cw = max(1e-4, float(bw))
    ch = max(1e-4, float(bh))
    aspect_ratio = float(cw / ch)
    cx = float(bx + bw / 2.0)
    cy = float(by + bh / 2.0)
    max_dim = max(cw, ch)
    scale = float(max_dim / 2.0)

    # Convert contours to normalized polylines
    norm_polys: List[np.ndarray] = []
    closed_flags: List[bool] = []
    poly_lengths: List[float] = []
    total_arc_length = 0.0

    for c in contours:
        if len(c) < 2:
            continue
        pts = c.squeeze(axis=1).astype(np.float32)
        norm_pts = np.empty_like(pts)
        norm_pts[:, 0] = (pts[:, 0] - cx) / scale
        norm_pts[:, 1] = (cy - pts[:, 1]) / scale  # Invert image Y to match CAD coordinate space

        diffs = np.diff(norm_pts, axis=0)
        plen = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
        if plen < 1e-4:
            continue

        norm_polys.append(norm_pts)
        closed_flags.append(True)
        poly_lengths.append(plen)
        total_arc_length += plen

    if not norm_polys or total_arc_length < 1e-6:
        total_arc_length = 1.0

    # Resample into fixed CANONICAL_POINTS_COUNT point cloud
    resampled_segments: List[np.ndarray] = []
    for i, norm_pts in enumerate(norm_polys):
        ratio = poly_lengths[i] / total_arc_length
        num_pts = max(2, int(round(ratio * CANONICAL_POINTS_COUNT)))
        resampled_segments.append(_resample_polyline_uniform(norm_pts, num_pts))

    point_cloud = np.concatenate(resampled_segments, axis=0) if resampled_segments else np.zeros((CANONICAL_POINTS_COUNT, 2), dtype=np.float32)
    if len(point_cloud) > CANONICAL_POINTS_COUNT:
        point_cloud = point_cloud[:CANONICAL_POINTS_COUNT]
    elif len(point_cloud) < CANONICAL_POINTS_COUNT:
        pad = np.repeat(point_cloud[-1:], CANONICAL_POINTS_COUNT - len(point_cloud), axis=0)
        point_cloud = np.concatenate([point_cloud, pad], axis=0)

    # Descriptors
    radial_hist = _compute_radial_histogram(point_cloud, num_bins=32)
    pairwise_hist = _compute_pairwise_histogram(point_cloud, num_bins=32)

    # Resize crop to 256x256 canonical mask
    crop = bin_mask[by : by + bh, bx : bx + bw]
    max_c = max(bw, bh)
    scale_c = 232.0 / float(max(1, max_c))
    nw = max(1, min(256, int(round(bw * scale_c))))
    nh = max(1, min(256, int(round(bh * scale_c))))
    resized_c = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    canonical_mask = np.full((256, 256), 255, dtype=np.uint8)
    ox = (256 - nw) // 2
    oy = (256 - nh) // 2
    canonical_mask[oy : oy + nh, ox : ox + nw] = resized_c

    hu_moments = _compute_hu_moments_from_mask(canonical_mask)

    grid_8x8 = cv2.resize(canonical_mask, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0
    grid_norm = np.linalg.norm(grid_8x8)
    if grid_norm > 1e-7:
        grid_8x8 = grid_8x8 / grid_norm

    contour_count = len(norm_polys)
    closed_count = sum(1 for c in closed_flags if c)
    hole_count = max(0, closed_count - 1)
    fill_ratio = float((canonical_mask == 255).mean())
    log_ar = float(math.log(max(1e-2, min(100.0, aspect_ratio))))

    stats = np.array([
        log_ar,
        fill_ratio,
        min(1.0, contour_count / 50.0),
        min(1.0, closed_count / 30.0),
        min(1.0, hole_count / 30.0),
        min(1.0, total_arc_length / 20.0),
        float(point_cloud[:, 0].std()),
        float(point_cloud[:, 1].std()),
        float(np.hypot(point_cloud[:, 0], point_cloud[:, 1]).std()),
    ], dtype=np.float32)

    fv = np.concatenate([radial_hist, pairwise_hist, hu_moments, stats, grid_8x8[:48]]).astype(np.float32)
    fv_norm = np.linalg.norm(fv)
    feature_vector = (fv / fv_norm) if fv_norm > 1e-7 else fv

    return NormalizedGeometry(
        component_name=label,
        aspect_ratio=aspect_ratio,
        width=cw,
        height=ch,
        centroid=(cx, cy),
        scale=scale,
        point_cloud=point_cloud,
        polylines=norm_polys,
        closed_flags=closed_flags,
        radial_hist=radial_hist,
        pairwise_hist=pairwise_hist,
        hu_moments=hu_moments,
        feature_vector=feature_vector,
        contour_count=contour_count,
        hole_count=hole_count,
        canonical_mask=canonical_mask,
    )
