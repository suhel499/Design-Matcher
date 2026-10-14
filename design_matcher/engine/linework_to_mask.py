"""
Linework to Stencil Mask Converter.
Converts 2D CAD/DXF vector linework or outline-only PDF renderings into standardized
binary stencil masks:
- White (255) = solid metal panel material
- Black (0) = cut-outs, apertures, slits, voids

Primary vector pipeline uses Shapely 2.x for endpoint snapping, noding, polygonization,
and even-odd (nesting depth) XOR rasterization.
Fallback raster pipeline uses morphological closing, exterior flood fill, and connected
component adjacency depth.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from shapely import STRtree, set_precision
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, polygonize_full, unary_union

from .config import EngineConfig, config as default_config

logger = logging.getLogger(__name__)


def _snap_endpoints_strtree(
    lines: List[LineString],
    tol: float,
) -> List[LineString]:
    """
    Snaps line endpoints that are within distance tol using Shapely STRtree and union-find.
    Runs in O(N log N) without large distance matrix allocations.
    """
    if not lines or tol <= 0:
        return lines

    coords_list = [list(l.coords) for l in lines]
    pt_objs: List[Point] = []
    for c in coords_list:
        pt_objs.append(Point(c[0]))
        pt_objs.append(Point(c[-1]))

    n_pts = len(pt_objs)
    if n_pts == 0:
        return lines

    tree = STRtree(pt_objs)
    pairs = tree.query(pt_objs, predicate="dwithin", distance=tol)

    parent = list(range(n_pts))

    def find(i: int) -> int:
        curr = i
        while parent[curr] != curr:
            parent[curr] = parent[parent[curr]]
            curr = parent[curr]
        return curr

    src_idx = pairs[0]
    tgt_idx = pairs[1]
    for i, j in zip(src_idx, tgt_idx):
        if i != j:
            ri = find(int(i))
            rj = find(int(j))
            if ri != rj:
                parent[ri] = rj

    coords_arr = np.array([[p.x, p.y] for p in pt_objs], dtype=np.float64)
    clusters: Dict[int, List[np.ndarray]] = {}
    for i in range(n_pts):
        r = find(i)
        clusters.setdefault(r, []).append(coords_arr[i])

    centroids = {r: np.mean(v, axis=0) for r, v in clusters.items()}

    snapped_lines: List[LineString] = []
    idx = 0
    for c in coords_list:
        r_start = find(idx)
        r_end = find(idx + 1)
        idx += 2
        c[0] = (float(centroids[r_start][0]), float(centroids[r_start][1]))
        c[-1] = (float(centroids[r_end][0]), float(centroids[r_end][1]))
        if len(c) >= 2:
            try:
                snapped_lines.append(LineString(c))
            except Exception:
                pass

    return snapped_lines


def _raster_fallback(
    polylines: List[Tuple[np.ndarray, bool]],
    bbox: Tuple[float, float, float, float],
    target_size: int,
    cfg: EngineConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fallback raster pipeline:
    1. Draw linework as 2px white lines on black canvas.
    2. Morphological close (kernel ~0.2% of long side).
    3. Flood fill exterior from corner.
    4. Label enclosed regions by connected components.
    5. Compute nesting depth by flood-fill adjacency.
    6. Set odd depth = 255, even depth = 0.
    """
    min_x, min_y, max_x, max_y = bbox
    w = max(1e-4, max_x - min_x)
    h = max(1e-4, max_y - min_y)
    max_side = max(w, h)

    margin = max(4, int(round(target_size * 0.01)))
    avail = float(target_size - 2 * margin)
    scale = avail / float(max_side)

    canvas_w = int(round(w * scale)) + 2 * margin
    canvas_h = int(round(h * scale)) + 2 * margin
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # 1. Draw linework (flip Y: max_y - y)
    for pts, is_closed in polylines:
        if len(pts) < 2:
            continue
        pix_pts = np.empty((len(pts), 2), dtype=np.int32)
        pix_pts[:, 0] = np.round(margin + (pts[:, 0] - min_x) * scale).astype(np.int32)
        pix_pts[:, 1] = np.round(margin + (max_y - pts[:, 1]) * scale).astype(np.int32)
        cv2.polylines(canvas, [pix_pts], isClosed=bool(is_closed), color=255, thickness=2, lineType=cv2.LINE_8)

    # 2. Morphological close
    k_size = max(3, int(round(target_size * 0.002)))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)

    # 3. Flood fill exterior from corner (0, 0)
    flooded = closed.copy()
    cv2.floodFill(flooded, None, (0, 0), 128)

    # 4. Enclosed regions are pixels that remain 0
    enclosed = (flooded == 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(enclosed, connectivity=8)

    # 5. Adjacency graph across boundary lines
    # Dilation kernel slightly larger than line width to bridge between adjacent regions
    bridge_k = max(7, k_size + 4)
    if bridge_k % 2 == 0:
        bridge_k += 1
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_k, bridge_k))

    adj: Dict[int, set] = {i: set() for i in range(num_labels)}
    for i in range(1, num_labels):
        # Bounding box of component i
        x0 = max(0, stats[i, cv2.CC_STAT_LEFT] - bridge_k)
        y0 = max(0, stats[i, cv2.CC_STAT_TOP] - bridge_k)
        x1 = min(canvas_w, stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] + bridge_k)
        y1 = min(canvas_h, stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] + bridge_k)

        roi_labels = labels[y0:y1, x0:x1]
        roi_flooded = flooded[y0:y1, x0:x1]

        mask_i = (roi_labels == i).astype(np.uint8)
        dilated_i = cv2.dilate(mask_i, bridge_kernel)

        if np.any((dilated_i > 0) & (roi_flooded == 128)):
            adj[0].add(i)
            adj[i].add(0)

        touching = np.unique(roi_labels[dilated_i > 0])
        for j in touching:
            if j > 0 and j != i:
                adj[i].add(int(j))
                adj[int(j)].add(i)

    # BFS from 0 (exterior)
    depth: Dict[int, int] = {0: 0}
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)

    # 6. Set odd depth = 255 (metal), even depth = 0 (void / background)
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for lbl, d in depth.items():
        if lbl > 0 and d % 2 == 1:
            mask[labels == lbl] = 255

    # Any unreached component defaults to metal (depth 1)
    for lbl in range(1, num_labels):
        if lbl not in depth:
            mask[labels == lbl] = 255

    info = {
        "used_fallback": True,
        "n_faces": max(0, num_labels - 1),
        "has_border": 1 in depth and depth[1] == 1,
        "gap_tolerance": float(max_side * cfg.DXF_GAP_TOLERANCE_RATIO),
        "warnings": ["Vector pipeline fell back to raster adjacency."],
    }
    return mask, info


def linework_to_mask(
    polylines: Union[List[Tuple[np.ndarray, bool]], List[np.ndarray]],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    cfg: Optional[EngineConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Converts 2D CAD linework into a standardized binary stencil mask:
    - White (255) = solid metal plate
    - Black (0) = cut-out apertures, holes, voids, slits

    Args:
        polylines: List of Nx2 point arrays (or (pts, is_closed) tuples).
        bbox: Optional (min_x, min_y, max_x, max_y) bounding box in drawing units.
        cfg: Optional EngineConfig.

    Returns:
        (mask, info):
            mask: uint8 2D array (longer side ~MASK_RENDER_SIZE, margin 1%).
            info: dict containing (used_fallback, n_faces, has_border, gap_tolerance, warnings).
    """
    c = cfg or default_config
    target_size = int(c.MASK_RENDER_SIZE)

    # Standardize input format
    std_polylines: List[Tuple[np.ndarray, bool]] = []
    for item in polylines:
        if isinstance(item, tuple) and len(item) == 2:
            pts, is_cl = item
            pts = np.asarray(pts, dtype=np.float64)
            if len(pts) >= 2:
                std_polylines.append((pts, bool(is_cl)))
        else:
            pts = np.asarray(item, dtype=np.float64)
            if len(pts) >= 2:
                # Check if endpoints coincide
                is_cl = bool(len(pts) >= 3 and np.allclose(pts[0], pts[-1], atol=1e-5))
                std_polylines.append((pts, is_cl))

    if not std_polylines:
        blank = np.zeros((target_size, target_size), dtype=np.uint8)
        return blank, {
            "used_fallback": False,
            "n_faces": 0,
            "has_border": False,
            "gap_tolerance": 0.0,
            "warnings": ["No valid polylines supplied."],
        }

    # Compute bounding box if not provided
    if bbox is None:
        all_pts = np.vstack([p[0] for p in std_polylines])
        min_x, min_y = float(all_pts[:, 0].min()), float(all_pts[:, 1].min())
        max_x, max_y = float(all_pts[:, 0].max()), float(all_pts[:, 1].max())
    else:
        min_x, min_y, max_x, max_y = [float(v) for v in bbox]

    w = max(1e-4, max_x - min_x)
    h = max(1e-4, max_y - min_y)
    diagonal = float(np.hypot(w, h))
    max_side = max(w, h)
    total_bbox_area = w * h

    # Configurable tolerance: 0.1% of design diagonal
    gap_tolerance = max(1e-4, diagonal * float(c.DXF_GAP_TOLERANCE_RATIO))

    # Convert polylines to Shapely LineStrings
    shapely_lines: List[LineString] = []
    for pts, _ in std_polylines:
        try:
            if len(pts) >= 2:
                shapely_lines.append(LineString(pts))
        except Exception:
            continue

    # Attempt Primary Vector Pipeline
    try:
        # 1. Endpoint snapping
        snapped_lines = _snap_endpoints_strtree(shapely_lines, tol=gap_tolerance)

        # Snap to small grid and node all linework
        grid_lines = [set_precision(l, grid_size=max(1e-5, gap_tolerance * 0.05)) for l in snapped_lines]
        noded = unary_union(grid_lines)

        # Polygonize
        poly_res, dangles, cuts, _ = polygonize_full(noded)
        faces: List[Polygon] = list(poly_res.geoms)

        # If vector pipeline yields fewer than 3 faces, fall back to raster pipeline
        if len(faces) < 3:
            logger.debug("Vector pipeline produced %d faces (<3). Running fallback raster pipeline.", len(faces))
            return _raster_fallback(std_polylines, (min_x, min_y, max_x, max_y), target_size, c)

        # Collect all boundary rings (exterior and interiors of all faces)
        rings: List[Polygon] = []
        for face in faces:
            if face.exterior and face.exterior.is_valid:
                rings.append(Polygon(face.exterior))
            for inter in face.interiors:
                if inter and inter.is_valid:
                    rings.append(Polygon(inter))

        # Deduplicate rings by geometry / area
        unique_rings: List[Polygon] = []
        seen_hashes = set()
        for r in rings:
            # Round coordinates to deduplicate identical cycle rings
            if r.area <= 1e-6:
                continue
            r_box = tuple(np.round(r.bounds, decimals=3))
            r_area = round(r.area, 3)
            key = (r_box, r_area)
            if key not in seen_hashes:
                seen_hashes.add(key)
                unique_rings.append(r)

        if not unique_rings:
            return _raster_fallback(std_polylines, (min_x, min_y, max_x, max_y), target_size, c)

        # Sort rings from largest to smallest for even-odd XOR
        unique_rings.sort(key=lambda item: item.area, reverse=True)

        # Determine canvas scaling and coordinates
        margin = max(4, int(round(target_size * 0.01)))
        avail = float(target_size - 2 * margin)
        scale = avail / float(max_side)

        canvas_w = int(round(w * scale)) + 2 * margin
        canvas_h = int(round(h * scale)) + 2 * margin
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        # Check for clear panel border (ring containing at least 80% of total bbox area)
        largest_ring = unique_rings[0]
        has_border = bool(largest_ring.area >= 0.80 * total_bbox_area)

        rings_to_xor: List[Polygon] = []
        if has_border:
            # Largest ring forms panel border -> will turn canvas 0 -> 255
            rings_to_xor = unique_rings
        else:
            # NO BORDER case: treat bbox of all geometry padded by 2% as panel
            pad_x = 0.02 * w
            pad_y = 0.02 * h
            p_min_x = max(min_x - pad_x, min_x - 0.05 * max_side)
            p_max_x = min(max_x + pad_x, max_x + 0.05 * max_side)
            p_min_y = max(min_y - pad_y, min_y - 0.05 * max_side)
            p_max_y = min(max_y + pad_y, max_y + 0.05 * max_side)

            px0 = int(round(margin + (p_min_x - min_x) * scale))
            py0 = int(round(margin + (max_y - p_max_y) * scale))
            px1 = int(round(margin + (p_max_x - min_x) * scale))
            py1 = int(round(margin + (max_y - p_min_y) * scale))

            px0 = max(0, min(canvas_w - 1, px0))
            px1 = max(0, min(canvas_w, px1))
            py0 = max(0, min(canvas_h - 1, py0))
            py1 = max(0, min(canvas_h, py1))

            canvas[py0:py1, px0:px1] = 255
            rings_to_xor = unique_rings

        # Rasterize rings into canvas with even-odd XOR
        for r in rings_to_xor:
            coords = np.array(r.exterior.coords, dtype=np.float64)
            if len(coords) < 3:
                continue

            # Convert to image coordinates (flip Y: max_y - y)
            img_x = np.round(margin + (coords[:, 0] - min_x) * scale).astype(np.int32)
            img_y = np.round(margin + (max_y - coords[:, 1]) * scale).astype(np.int32)

            pts_2d = np.column_stack([img_x, img_y])

            # ROI optimization: fill within bounding box of polygon
            rx0 = max(0, int(np.min(img_x)))
            ry0 = max(0, int(np.min(img_y)))
            rx1 = min(canvas_w, int(np.max(img_x)) + 1)
            ry1 = min(canvas_h, int(np.max(img_y)) + 1)

            rw = rx1 - rx0
            rh = ry1 - ry0
            if rw <= 0 or rh <= 0:
                continue

            roi_mask = np.zeros((rh, rw), dtype=np.uint8)
            offset_pts = pts_2d - np.array([rx0, ry0], dtype=np.int32)
            cv2.fillPoly(roi_mask, [offset_pts], 255, lineType=cv2.LINE_8)

            canvas[ry0:ry1, rx0:rx1] ^= roi_mask

        # 4. OPEN lines: draw strokes that do not form closed faces as black cuts
        stroke_thickness = max(2, int(round(target_size * float(c.DXF_OPEN_STROKE_RATIO))))
        open_geoms = list(getattr(dangles, "geoms", [])) + list(getattr(cuts, "geoms", []))
        for open_l in open_geoms:
            try:
                coords = np.array(open_l.coords, dtype=np.float64)
                if len(coords) < 2:
                    continue
                img_x = np.round(margin + (coords[:, 0] - min_x) * scale).astype(np.int32)
                img_y = np.round(margin + (max_y - coords[:, 1]) * scale).astype(np.int32)
                line_pts = np.column_stack([img_x, img_y])
                cv2.polylines(canvas, [line_pts], isClosed=False, color=0, thickness=stroke_thickness, lineType=cv2.LINE_8)
            except Exception:
                continue

        # Re-binarize at 127
        mask = (canvas > 127).astype(np.uint8) * 255

        info = {
            "used_fallback": False,
            "n_faces": len(faces),
            "has_border": has_border,
            "gap_tolerance": gap_tolerance,
            "warnings": [],
        }
        return mask, info

    except Exception as exc:
        logger.warning("Vector pipeline exception: %s. Using raster fallback.", exc, exc_info=True)
        mask, info = _raster_fallback(std_polylines, (min_x, min_y, max_x, max_y), target_size, c)
        info["warnings"].append(f"Vector pipeline exception: {exc}")
        return mask, info


def is_outline_only(page_mask: np.ndarray) -> bool:
    """
    Checks if a rendered page mask represents an outline-only/wireframe drawing:
    - Very thin strokes (>90% background pixels)
    - Foreground disappears under slight morphological erosion
    - Large connected outer frame or border exists
    """
    if page_mask is None or page_mask.size == 0:
        return False

    if page_mask.ndim == 3:
        page_mask = cv2.cvtColor(page_mask, cv2.COLOR_BGR2GRAY)

    h, w = page_mask.shape[:2]
    total_px = float(h * w)

    # In normalize_stencil: white (255) = solid metal, black (0) = cutouts
    # If the page was wireframe outlines on background:
    # background pixels could be either >90% white or >90% black
    ratio_white = float(np.mean(page_mask == 255))
    ratio_black = float(np.mean(page_mask == 0))

    if ratio_white < 0.90 and ratio_black < 0.90:
        return False

    # The minority class is the thin stroke line
    stroke_val = 0 if ratio_white >= 0.90 else 255
    stroke_mask = (page_mask == stroke_val).astype(np.uint8)

    # Check thinness: erode with 3x3 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(stroke_mask, kernel)

    orig_strokes = int(stroke_mask.sum())
    eroded_strokes = int(eroded.sum())

    if orig_strokes == 0:
        return False

    loss_ratio = float(orig_strokes - eroded_strokes) / float(orig_strokes)
    # If >75% of stroke pixels disappear after a 3x3 erosion, strokes are very thin wireframes
    if loss_ratio < 0.75:
        return False

    # Check for large connected frame: stroke mask contains a component spanning >50% of page dimension
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stroke_mask, connectivity=8)
    has_large_frame = False
    for i in range(1, n_labels):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if cw >= 0.5 * w or ch >= 0.5 * h:
            has_large_frame = True
            break

    return has_large_frame


def outline_to_mask(
    gray: np.ndarray,
    cfg: Optional[EngineConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Converts a grayscale outline-only page image into a solid stencil mask using the
    raster parity fill pipeline.
    """
    c = cfg or default_config
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    # Threshold strokes
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Linework as 2px white on black
    k_size = max(3, int(round(max(h, w) * 0.002)))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Flood fill exterior from corner
    flooded = closed.copy()
    cv2.floodFill(flooded, None, (0, 0), 128)

    enclosed = (flooded == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed, connectivity=8)

    bridge_k = max(7, k_size + 4)
    if bridge_k % 2 == 0:
        bridge_k += 1
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_k, bridge_k))

    adj: Dict[int, set] = {i: set() for i in range(num_labels)}
    for i in range(1, num_labels):
        x0 = max(0, stats[i, cv2.CC_STAT_LEFT] - bridge_k)
        y0 = max(0, stats[i, cv2.CC_STAT_TOP] - bridge_k)
        x1 = min(w, stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] + bridge_k)
        y1 = min(h, stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] + bridge_k)

        roi_labels = labels[y0:y1, x0:x1]
        roi_flooded = flooded[y0:y1, x0:x1]

        mask_i = (roi_labels == i).astype(np.uint8)
        dilated_i = cv2.dilate(mask_i, bridge_kernel)

        if np.any((dilated_i > 0) & (roi_flooded == 128)):
            adj[0].add(i)
            adj[i].add(0)

        touching = np.unique(roi_labels[dilated_i > 0])
        for j in touching:
            if j > 0 and j != i:
                adj[i].add(int(j))
                adj[int(j)].add(i)

    depth: Dict[int, int] = {0: 0}
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)

    out_mask = np.zeros((h, w), dtype=np.uint8)
    for lbl, d in depth.items():
        if lbl > 0 and d % 2 == 1:
            out_mask[labels == lbl] = 255

    for lbl in range(1, num_labels):
        if lbl not in depth:
            out_mask[labels == lbl] = 255

    return out_mask, {"used_fallback": True, "n_faces": num_labels - 1}
