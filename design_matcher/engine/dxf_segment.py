"""
DXF Drawing Segmentation Module.
Splits a multi-design DXF CAD drawing or sheet into individual laser cut designs.
Detects:
1. Single top-level panel border (one design).
2. Multiple designs via large closed panel boundary rectangles.
3. Spatial proximity clustering (buffered bbox union).
Associates nearest text labels (e.g., 'PD 010') to each design.
Orders designs left-to-right, top-to-bottom.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from .config import EngineConfig, config as default_config
from .dxf_reader import DxfDrawingData, DxfLabel, read_dxf
from .linework_to_mask import linework_to_mask

logger = logging.getLogger(__name__)


@dataclass
class DxfDesign:
    """Represents a single segmented design from a DXF drawing."""

    bbox: Tuple[float, float, float, float]  # (x, y, w, h) in drawing units
    drawing_bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    entity_count: int
    label: str
    mask: np.ndarray
    mask_info: Dict[str, Any]
    polylines: List[Tuple[np.ndarray, bool]]


def _is_approx_rectangle(poly: Polygon, tol: float = 0.05) -> bool:
    """Checks whether a polygon is approximately a rectangular panel."""
    if not poly.is_valid or poly.area <= 0:
        return False
    min_rect = poly.minimum_rotated_rectangle
    if min_rect.area <= 0:
        return False
    fill = poly.area / min_rect.area
    return fill >= (1.0 - tol)


def _find_nearest_label(
    design_box: Tuple[float, float, float, float],
    labels: List[DxfLabel],
    default_label: str,
) -> str:
    """
    Finds the nearest TEXT/MTEXT label within 1.5 design-heights below or above
    the design bounding box, horizontally overlapping it.
    """
    min_x, min_y, max_x, max_y = design_box
    w = max_x - min_x
    h = max_y - min_y

    h_margin = 0.2 * w
    valid_candidates: List[Tuple[float, str]] = []

    for lbl in labels:
        lx = lbl.x
        ly = lbl.y
        txt = lbl.text.strip()
        if not txt:
            continue

        # Horizontal overlap check
        if not (min_x - h_margin <= lx <= max_x + h_margin):
            continue

        # Vertical distance check (DXF Y points UP)
        # Below: ly <= min_y and (min_y - ly) <= 1.5 * h
        # Above: ly >= max_y and (ly - max_y) <= 1.5 * h
        if ly <= min_y and (min_y - ly) <= 1.5 * h:
            v_dist = min_y - ly
            valid_candidates.append((v_dist, txt))
        elif ly >= max_y and (ly - max_y) <= 1.5 * h:
            v_dist = ly - max_y
            valid_candidates.append((v_dist, txt))

    if not valid_candidates:
        return default_label

    # Sort by minimum vertical distance
    valid_candidates.sort(key=lambda item: item[0])
    return valid_candidates[0][1]


def segment_dxf_drawing(
    dxf_input: Union[str, Path, DxfDrawingData],
    cfg: Optional[EngineConfig] = None,
) -> List[DxfDesign]:
    """
    Segments a DXF drawing into individual designs:
    - If 1 top-level border, returns 1 design.
    - If multiple closed panel rectangles, returns each panel.
    - Otherwise clusters geometry by spatial proximity (buffer + union).
    - Ignores title block frames and groups < 0.5% area.
    - Assigns nearest text labels and generates stencil masks.
    - Orders left-to-right, top-to-bottom.
    """
    c = cfg or default_config

    if isinstance(dxf_input, DxfDrawingData):
        dxf_data = dxf_input
        file_stem = "dxf_drawing"
    else:
        path_obj = Path(dxf_input).resolve()
        file_stem = path_obj.stem
        dxf_data = read_dxf(path_obj, cfg=c)

    polylines = dxf_data.polylines
    labels = dxf_data.labels
    total_bbox = dxf_data.bbox

    if not polylines:
        return []

    min_x, min_y, max_x, max_y = total_bbox
    dw = max(1e-4, max_x - min_x)
    dh = max(1e-4, max_y - min_y)
    drawing_long_side = max(dw, dh)
    drawing_area = dw * dh

    # Step 1: Detect candidate closed panel border rectangles that contain other geometry
    closed_polys: List[Tuple[Polygon, Tuple[float, float, float, float], int]] = []

    for orig_idx, (pts, is_closed) in enumerate(polylines):
        if not is_closed or len(pts) < 4:
            continue
        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and poly.area >= 0.005 * drawing_area:
                if _is_approx_rectangle(poly):
                    bx0, by0, bx1, by1 = poly.bounds
                    closed_polys.append((poly, (float(bx0), float(by0), float(bx1), float(by1)), orig_idx))
        except Exception:
            continue

    # Filter candidate rectangles: a design panel must contain other geometry strictly inside it
    candidate_rects: List[Tuple[Polygon, Tuple[float, float, float, float]]] = []
    for poly, b, orig_idx in closed_polys:
        # Count other polylines strictly inside this rectangle
        inside_entities = 0
        for other_idx, (other_pts, _) in enumerate(polylines):
            if other_idx == orig_idx or len(other_pts) < 2:
                continue
            c_pt = Point(float(np.mean(other_pts[:, 0])), float(np.mean(other_pts[:, 1])))
            if poly.contains(c_pt) and poly.exterior.distance(c_pt) > 1e-4:
                inside_entities += 1
                if inside_entities >= 1:
                    break
        if inside_entities >= 1:
            candidate_rects.append((poly, b))

    # Filter out title-block frame if present (a giant frame containing >= 2 distinct panels)
    design_boxes: List[Tuple[float, float, float, float]] = []

    if candidate_rects:
        candidate_rects.sort(key=lambda item: item[0].area, reverse=True)
        largest_poly, largest_b = candidate_rects[0]

        # Check if the largest rectangle is a title-block frame enclosing >= 2 distinct design panels
        inner_panels = [
            (p, b) for p, b in candidate_rects[1:]
            if largest_poly.contains(p.centroid)
        ]
        if len(inner_panels) >= 2 and largest_poly.area >= 0.80 * drawing_area:
            # Drop title-block frame, keep inner design panels
            candidate_rects = inner_panels

        # Exclude any rectangle that is strictly contained inside another design rectangle
        top_level_rects: List[Tuple[Polygon, Tuple[float, float, float, float]]] = []
        for p, b in candidate_rects:
            is_inside_another = False
            for other_p, _ in candidate_rects:
                if other_p is not p and other_p.area > p.area and other_p.contains(p.centroid):
                    is_inside_another = True
                    break
            if not is_inside_another:
                top_level_rects.append((p, b))

        # Deduplicate overlapping rectangles (IoU > 0.7)
        chosen_rects: List[Tuple[float, float, float, float]] = []
        for poly, b in top_level_rects:
            b_box = box(*b)
            is_dup = False
            for cb in chosen_rects:
                cb_box = box(*cb)
                inter = b_box.intersection(cb_box).area
                union_a = b_box.union(cb_box).area
                if union_a > 0 and (inter / union_a) > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                chosen_rects.append(b)

        if chosen_rects:
            design_boxes = chosen_rects

    # Step 2: If not multiple distinct rectangles, check single top-level border vs spatial clustering
    if not design_boxes:
        # Check single top-level border
        if candidate_rects and candidate_rects[0][0].area >= 0.70 * drawing_area:
            design_boxes = [candidate_rects[0][1]]
        else:
            # Step 2b: Spatial clustering (buffer 2% of drawing long side + union)
            buf_dist = 0.02 * drawing_long_side
            poly_boxes: List[Polygon] = []
            for pts, _ in polylines:
                px0 = float(pts[:, 0].min())
                py0 = float(pts[:, 1].min())
                px1 = float(pts[:, 0].max())
                py1 = float(pts[:, 1].max())
                poly_boxes.append(box(px0, py0, px1, py1).buffer(buf_dist))

            if poly_boxes:
                u = unary_union(poly_boxes)
                clusters: List[Polygon] = []
                if isinstance(u, Polygon):
                    clusters = [u]
                elif hasattr(u, "geoms"):
                    clusters = list(u.geoms)

                # Filter clusters: ignore < 0.5% area
                valid_clusters = []
                for cl in clusters:
                    if cl.area >= 0.005 * drawing_area:
                        valid_clusters.append(cl)

                # If 1 giant cluster encloses multiple sub-groups, or if single cluster
                if len(valid_clusters) <= 1:
                    design_boxes = [(min_x, min_y, max_x, max_y)]
                else:
                    design_boxes = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in [c.bounds for c in valid_clusters]]

    if not design_boxes:
        design_boxes = [(min_x, min_y, max_x, max_y)]

    # Step 3: Sort designs left-to-right, top-to-bottom
    # In DXF coordinates (Y points UP):
    # Higher Y is higher on page (top). We bucket by row height (0.4 * median_h).
    median_h = float(np.median([b[3] - b[1] for b in design_boxes]))
    row_tol = max(1e-3, 0.4 * median_h)

    def sort_key(b: Tuple[float, float, float, float]):
        bx0, by0, bx1, by1 = b
        # Bucket Y descending: higher Y = smaller bucket index
        row_idx = -int(round(by1 / row_tol))
        return (row_idx, bx0)

    design_boxes.sort(key=sort_key)

    # Step 4: Extract geometry for each design box, assign labels, render stencil mask
    results: List[DxfDesign] = []

    for idx, d_bounds in enumerate(design_boxes):
        bx0, by0, bx1, by1 = d_bounds
        bw = max(1e-4, bx1 - bx0)
        bh = max(1e-4, by1 - by0)
        pad = 0.05 * max(bw, bh)

        # Collect polylines within or intersecting this design bbox
        d_polylines: List[Tuple[np.ndarray, bool]] = []
        for pts, is_cl in polylines:
            px0 = pts[:, 0].min()
            py0 = pts[:, 1].min()
            px1 = pts[:, 0].max()
            py1 = pts[:, 1].max()

            # Check overlap with design bbox
            if not (px1 < bx0 - pad or px0 > bx1 + pad or py1 < by0 - pad or py0 > by1 + pad):
                d_polylines.append((pts, is_cl))

        if not d_polylines:
            continue

        # Find nearest text label
        default_lbl = f"{file_stem}#{idx + 1}"
        label_text = _find_nearest_label(d_bounds, labels, default_label=default_lbl)

        # Render stencil mask for this design
        mask, info = linework_to_mask(d_polylines, bbox=d_bounds, cfg=c)

        results.append(
            DxfDesign(
                bbox=(bx0, by0, bw, bh),
                drawing_bounds=d_bounds,
                entity_count=len(d_polylines),
                label=label_text,
                mask=mask,
                mask_info=info,
                polylines=d_polylines,
            )
        )

    return results
