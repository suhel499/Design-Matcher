"""
DXF (AutoCAD Drawing Exchange Format) 2D rendering and stencil extraction module.
Uses ezdxf (MIT license - free for commercial use).
Converts CNC / laser cut vector paths into standardized binary stencil masks:
- White (255) = solid metal panel material
- Black (0) = laser cut-out loops, holes, voids, and slits
Handles lines, polylines, circles, arcs, ellipses, splines, hatches, and block references.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import ezdxf
import ezdxf.path as ep
import numpy as np

logger = logging.getLogger(__name__)


class DxfRenderError(Exception):
    """Raised when a DXF file cannot be read or rendered."""
    pass


def _extract_paths_recursive(entities) -> List[ep.Path]:
    """
    Recursively extracts 2D paths from DXF entities, exploding INSERT
    block references using virtual_entities() to preserve coordinate transforms.
    """
    paths: List[ep.Path] = []
    for e in entities:
        dxftype = e.dxftype()
        if dxftype in ("TEXT", "MTEXT", "DIMENSION", "POINT"):
            # Skip annotations, text labels, and dimension markings
            continue
        elif dxftype == "INSERT":
            try:
                sub_entities = list(e.virtual_entities())
                paths.extend(_extract_paths_recursive(sub_entities))
            except Exception as exc:
                logger.debug("Failed to expand virtual entities for INSERT: %s", exc)
        else:
            try:
                p = ep.make_path(e)
                paths.append(p)
            except Exception as exc:
                logger.debug("Could not convert entity %s to path: %s", dxftype, exc)
    return paths


def get_dxf_metadata(dxf_path: Union[str, Path]) -> Dict[str, Any]:
    """Reads basic DXF file dimensions and entity statistics."""
    p = Path(dxf_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")

    try:
        doc = ezdxf.readfile(str(p))
        msp = doc.modelspace()
        paths = _extract_paths_recursive(msp)
        if not paths:
            return {"width": 0.0, "height": 0.0, "entity_count": 0, "path_count": 0}

        bbox = ep.bbox(paths)
        w = float(bbox.extmax.x - bbox.extmin.x)
        h = float(bbox.extmax.y - bbox.extmin.y)
        return {
            "width": max(0.0, w),
            "height": max(0.0, h),
            "entity_count": len(msp),
            "path_count": len(paths),
            "units": doc.header.get("$INSUNITS", 0),
        }
    except Exception as exc:
        raise DxfRenderError(f"Failed to read DXF metadata {dxf_path}: {exc}") from exc


def render_dxf_to_stencil(
    dxf_path: Union[str, Path],
    target_size: int = 800,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Renders a DXF file into a standardized binary stencil mask:
    - White (255) = solid metal plate
    - Black (0) = cutouts, apertures, slits

    Returns:
        (stencil_mask, info_dict):
            stencil_mask: 2D uint8 numpy array
            info_dict: metadata including real dimensions and aspect ratio
    """
    p = Path(dxf_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"DXF file does not exist: {dxf_path}")

    try:
        doc = ezdxf.readfile(str(p))
    except Exception as exc:
        raise DxfRenderError(f"Could not open or parse DXF {dxf_path}: {exc}") from exc

    try:
        msp = doc.modelspace()
        paths = _extract_paths_recursive(msp)
    except Exception as exc:
        raise DxfRenderError(f"Error reading modelspace entities in {dxf_path}: {exc}") from exc

    if not paths:
        # Fallback for empty DXF
        blank = np.full((target_size, target_size), 255, dtype=np.uint8)
        return blank, {"width": 0.0, "height": 0.0, "aspect_ratio": 1.0, "empty": True}

    try:
        bbox = ep.bbox(paths)
        min_x, min_y = float(bbox.extmin.x), float(bbox.extmin.y)
        max_x, max_y = float(bbox.extmax.x), float(bbox.extmax.y)
        w_dxf = max_x - min_x
        h_dxf = max_y - min_y
    except Exception as exc:
        raise DxfRenderError(f"Failed to compute bounding box for {dxf_path}: {exc}") from exc

    if w_dxf <= 1e-4 or h_dxf <= 1e-4:
        blank = np.full((target_size, target_size), 255, dtype=np.uint8)
        return blank, {"width": w_dxf, "height": h_dxf, "aspect_ratio": 1.0, "empty": True}

    aspect_ratio = float(w_dxf) / float(max(1e-4, h_dxf))
    scale = float(target_size) / float(max(w_dxf, h_dxf))

    img_w = max(16, int(round(w_dxf * scale)))
    img_h = max(16, int(round(h_dxf * scale)))

    closed_polys: List[Tuple[float, np.ndarray]] = []
    open_paths: List[np.ndarray] = []

    # Flatten curves/arcs to linear line segments
    flatten_dist = max(0.4, 1.0 / max(scale, 1e-4))
    for path in paths:
        try:
            verts = list(path.flattening(distance=flatten_dist))
            if len(verts) < 2:
                continue
            pts = np.array([
                (int(round((v.x - min_x) * scale)), int(round((max_y - v.y) * scale)))
                for v in verts
            ], dtype=np.int32)

            if path.is_closed and len(verts) >= 3:
                area = abs(cv2.contourArea(pts))
                closed_polys.append((area, pts))
            else:
                open_paths.append(pts)
        except Exception:
            continue

    closed_polys.sort(key=lambda item: item[0], reverse=True)
    total_area = float(img_w * img_h)

    # Render with Even-Odd parity rule so nested apertures, metal webbing,
    # and multi-tier islands are accurately preserved without blacking out.
    stencil = np.zeros((img_h, img_w), dtype=np.uint8)

    # Accumulate closed polygons with bitwise XOR
    for _, pts in closed_polys:
        poly_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)
        stencil = cv2.bitwise_xor(stencil, poly_mask)

    # Determine polarity convention:
    # Outer sheet / solid panel = 255 (white)
    # Laser cut-outs / slits = 0 (black)
    border = np.concatenate([stencil[0, :], stencil[-1, :], stencil[:, 0], stencil[:, -1]])
    border_white_ratio = float(np.mean(border == 255))
    if border_white_ratio < 0.25 and len(closed_polys) > 0:
        largest_area = closed_polys[0][0] if closed_polys else 0
        if largest_area < 0.35 * total_area:
            stencil = cv2.bitwise_not(stencil)

    # Draw open cutting stroke paths (slits, kerfs) with laser line thickness
    stroke_w = max(2, int(round(scale * 1.5)))
    for pts in open_paths:
        cv2.polylines(stencil, [pts], isClosed=False, color=0, thickness=stroke_w)

    info = {
        "width": w_dxf,
        "height": h_dxf,
        "aspect_ratio": aspect_ratio,
        "empty": False,
        "closed_loops": len(closed_polys),
        "open_paths": len(open_paths),
    }
    return stencil, info


def detect_dxf_panels(
    stencil: np.ndarray,
    min_area_ratio: float = 0.08,
) -> List[Tuple[List[int], np.ndarray]]:
    """
    Detects individual design panels within a multi-panel DXF sheet.
    Returns list of (bbox, panel_stencil_mask) where bbox is [x, y, w, h].
    If no distinct subpanels are found, returns empty list.
    """
    if stencil is None or stencil.size == 0:
        return []

    h, w = stencil.shape[:2]
    total_area = float(h * w)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(stencil, connectivity=8)
    panels: List[Tuple[List[int], np.ndarray]] = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_ratio * total_area:
            px = int(stats[i, cv2.CC_STAT_LEFT])
            py = int(stats[i, cv2.CC_STAT_TOP])
            pw = int(stats[i, cv2.CC_STAT_WIDTH])
            ph = int(stats[i, cv2.CC_STAT_HEIGHT])
            if pw >= 40 and ph >= 40 and (pw * ph < 0.95 * total_area):
                crop = stencil[py : py + ph, px : px + pw]
                panels.append(([px, py, pw, ph], crop))

    panels.sort(key=lambda p: (p[0][1], p[0][0]))
    return panels


def render_dxf_preview(
    dxf_path: Union[str, Path],
    target_size: int = 800,
    stencil: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Renders a realistic brushed-metal BGR preview image of the DXF design:
    - Silver metal plate with subtle bevel edges
    - Dark navy cut-out voids
    """
    if stencil is None:
        stencil, _ = render_dxf_to_stencil(dxf_path, target_size=target_size)
    h, w = stencil.shape[:2]

    # Background void: Dark navy (25, 20, 15)
    bgr = np.full((h, w, 3), (25, 20, 15), dtype=np.uint8)

    # Solid metal material: Brushed aluminum/silver (218, 218, 222)
    bgr[stencil == 255] = (218, 218, 222)

    # Laser edge accent line (charcoal grey)
    edges = cv2.Canny(stencil, 100, 200)
    bgr[edges > 0] = (70, 70, 75)

    return bgr


def generate_dxf_thumbnail(
    stencil: np.ndarray,
    thumb_size: int = 120,
) -> bytes:
    """Generates PNG-compressed thumbnail bytes from a stencil mask."""
    h, w = stencil.shape[:2]
    scale = float(thumb_size) / float(max(h, w))
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))

    # Create thumbnail BGR
    thumb_bgr = np.full((th, tw, 3), (25, 20, 15), dtype=np.uint8)
    resized_stencil = cv2.resize(stencil, (tw, th), interpolation=cv2.INTER_AREA)
    thumb_bgr[resized_stencil > 127] = (218, 218, 222)

    _, png = cv2.imencode(".png", thumb_bgr)
    return png.tobytes()
