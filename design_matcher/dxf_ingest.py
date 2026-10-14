"""
DXF Ingestion and Vector-to-Raster Stencil Mask Pipeline for Design Matcher.

Provides:
- load_dxf(path): Robust DXF reader handling R12-2018+, corrupt recovery, and clear DWG warnings.
- extract_designs(doc, config=None) -> list[Design]: Automatic multi-design panel detection,
  recursive block explosion (virtual entities), layer filtering, and spatial text label pairing.
- render_design(design, px_height=1024) -> np.ndarray: Clean non-stretching rasterization of design
  geometry with smooth curve flattening and closed cut stroke thickness.
- convert_line_art_to_stencil_mask(rendered_gray, ...) -> Tuple[np.ndarray, np.ndarray]:
  Converts linework to standardized 0/255 stencil masks (both standard and inverted polarities)
  with gap closing, exterior flood fill, and outer border elimination.
- process_dxf_file(path, px_height=1024) -> list[Design]: Complete end-to-end file pipeline.
- Debug CLI: python dxf_ingest.py file.dxf --out debug/
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import ezdxf
import ezdxf.bbox as ed_bbox
import ezdxf.document
import ezdxf.path as ep
import ezdxf.recover
import numpy as np

# Ensure matplotlib uses Agg backend without GUI windows if ever imported
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib  # pyright: ignore[reportMissingImports]
    matplotlib.use("Agg")
except ImportError:
    pass

# Setup module logger with rotating file handler
logger = logging.getLogger("dxf_ingest")
logger.setLevel(logging.INFO)

_log_file = Path("dxf_ingest.log")
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    try:
        rfh = RotatingFileHandler(str(_log_file), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        rfh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(rfh)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Default Configuration Constants
# ---------------------------------------------------------------------------

DEFAULT_IGNORE_LAYERS = [
    "dim",
    "dimension",
    "text",
    "texts",
    "note",
    "notes",
    "construction",
    "defpoints",
    "hatch-fill",
    "hatch_fill",
    "hatch",
    "frame-only",
    "frame_only",
    "title",
    "border_sheet",
    "grid",
    "axis",
    "viewport",
]

DEFAULT_IGNORE_ENTITY_TYPES = {
    "TEXT",
    "MTEXT",
    "DIMENSION",
    "POINT",
    "LEADER",
    "MULTILEADER",
    "HATCH",
}

SUPPORTED_GEOM_TYPES = {
    "LINE",
    "LWPOLYLINE",
    "POLYLINE",
    "ARC",
    "CIRCLE",
    "ELLIPSE",
    "SPLINE",
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DxfTextLabel:
    """Represents a text or mtext entity extracted from CAD modelspace."""
    text: str
    x: float
    y: float
    height: float


@dataclass
class Design:
    """
    Represents an individual laser-cut panel design extracted from a DXF drawing.
    """
    design_id: str
    design_code: Optional[str]
    bbox: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y) in drawing units
    panel_rect: Optional[Tuple[float, float, float, float]] = None
    entities: List[Any] = field(default_factory=list)
    polylines: List[Tuple[np.ndarray, bool]] = field(default_factory=list)
    index: int = 1
    total_designs: int = 1
    source_path: str = ""
    entity_count: int = 0
    drawing_units: int = 0
    rendered_image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    mask_inverted: Optional[np.ndarray] = None

    @property
    def x(self) -> float:
        return self.bbox[0]

    @property
    def y(self) -> float:
        return self.bbox[1]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def dxf_bbox(self) -> Tuple[float, float, float, float]:
        """Returns (x, y, w, h) in drawing coordinates."""
        return (self.x, self.y, self.width, self.height)


# ---------------------------------------------------------------------------
# 1. DXF Loading with Fallbacks
# ---------------------------------------------------------------------------

def load_dxf(path: Union[str, Path]) -> ezdxf.document.Drawing:
    """
    Loads a DXF CAD drawing from disk:
    - Verifies file exists.
    - Explicitly intercepts .dwg files and raises a clear explanatory message.
    - Uses ezdxf.readfile(), falling back to ezdxf.recover.readfile() for slightly
      corrupt or non-standard AutoCAD files (R12 through 2018+).
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"CAD file not found: {p}")

    if p.suffix.lower() == ".dwg":
        raise ValueError(
            f"DWG format is not supported for '{p.name}'. Please open the drawing in "
            "AutoCAD or DWG TrueView and export/save as DXF format."
        )

    try:
        # Standard fast read
        doc = ezdxf.readfile(str(p))  # pyright: ignore[reportPrivateImportUsage]
        return doc
    except Exception as exc_read:
        logger.warning("ezdxf.readfile failed on %s (%s). Attempting ezdxf.recover...", p.name, exc_read)
        try:
            doc, auditor = ezdxf.recover.readfile(str(p))
            if auditor.has_errors:
                logger.info(
                    "ezdxf.recover repaired %d errors in %s",
                    len(auditor.errors),
                    p.name,
                )
            return doc
        except Exception as exc_recover:
            raise RuntimeError(
                f"Failed to open or recover DXF file '{p.name}': {exc_recover}"
            ) from exc_recover


# ---------------------------------------------------------------------------
# 2. Virtual Entities & Layer Filtering
# ---------------------------------------------------------------------------

def _is_layer_ignored(
    layer_name: str,
    doc: ezdxf.document.Drawing,
    ignore_patterns: Sequence[str],
) -> bool:
    """Checks if a layer is frozen, turned off, or matches non-cut naming patterns."""
    lname_clean = layer_name.lower().strip()
    try:
        layer_obj = doc.layers.get(layer_name)
        if layer_obj is not None:
            if layer_obj.is_off() or layer_obj.is_frozen():
                return True
    except Exception:
        pass

    for pattern in ignore_patterns:
        if pattern.lower() in lname_clean:
            return True
    return False


def _clean_text_content(raw_text: str) -> str:
    """Strips MTEXT formatting codes and trailing whitespace."""
    if not raw_text:
        return ""
    # Remove AutoCAD MTEXT formatting like \P, \A1;, \H10;, {\L...}
    cleaned = re.sub(r"\\[A-Za-z0-9]+;?", " ", raw_text)
    cleaned = re.sub(r"[{}\\]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _explode_and_categorize(
    entities: Sequence[Any],
    doc: ezdxf.document.Drawing,
    ignore_layers: Sequence[str],
    max_depth: int = 8,
    current_depth: int = 0,
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[List[Any], List[DxfTextLabel]]:
    """
    Recursively unpacks entities:
    - Resolves INSERT blocks into virtual entities preserving transform.
    - Separates cut geometry entities from text labels.
    - Records skipped and processed entity counts.
    """
    if stats is None:
        stats = {}
    stats.setdefault("total", 0)
    stats.setdefault("skipped_layer", 0)
    stats.setdefault("skipped_type", 0)
    stats.setdefault("geom", 0)
    stats.setdefault("text", 0)

    geom_entities: List[Any] = []
    labels: List[DxfTextLabel] = []

    for ent in entities:
        stats["total"] += 1
        dxftype = ent.dxftype()
        layer_name = getattr(ent.dxf, "layer", "0")

        # Layer filtering check
        layer_ignored = _is_layer_ignored(layer_name, doc, ignore_layers)

        # Allow text entities to be captured even if on a 'text' named layer
        if dxftype in ("TEXT", "MTEXT"):
            stats["text"] += 1
            try:
                if dxftype == "MTEXT":
                    txt = ent.plain_text() if hasattr(ent, "plain_text") else str(getattr(ent.dxf, "text", ""))
                    h = float(getattr(ent.dxf, "char_height", 10.0))
                else:
                    txt = str(getattr(ent.dxf, "text", ""))
                    h = float(getattr(ent.dxf, "height", 10.0))

                txt_clean = _clean_text_content(txt)
                ins = getattr(ent.dxf, "insert", (0.0, 0.0, 0.0))
                if txt_clean:
                    labels.append(DxfTextLabel(text=txt_clean, x=float(ins[0]), y=float(ins[1]), height=h))
            except Exception as exc:
                logger.debug("Failed extracting text from %s: %s", dxftype, exc)
            continue

        if layer_ignored:
            stats["skipped_layer"] += 1
            continue

        if dxftype == "INSERT":
            if current_depth < max_depth:
                try:
                    sub_ents = list(ent.virtual_entities())
                    sub_geom, sub_lbl = _explode_and_categorize(
                        sub_ents,
                        doc=doc,
                        ignore_layers=ignore_layers,
                        max_depth=max_depth,
                        current_depth=current_depth + 1,
                        stats=stats,
                    )
                    geom_entities.extend(sub_geom)
                    labels.extend(sub_lbl)
                except Exception as exc:
                    logger.debug("Error expanding INSERT virtual entities: %s", exc)
            continue

        if dxftype in SUPPORTED_GEOM_TYPES:
            geom_entities.append(ent)
            stats["geom"] += 1
        else:
            stats["skipped_type"] += 1

    return geom_entities, labels


# ---------------------------------------------------------------------------
# 3. Geometry Bounding Box & Panel Clustering
# ---------------------------------------------------------------------------

def _get_entity_bbox(entity: Any) -> Optional[Tuple[float, float, float, float]]:
    """Calculates (min_x, min_y, max_x, max_y) of an individual DXF entity."""
    dxftype = entity.dxftype()
    try:
        if dxftype == "CIRCLE":
            c = entity.dxf.center
            r = float(entity.dxf.radius)
            return (float(c.x - r), float(c.y - r), float(c.x + r), float(c.y + r))
        elif dxftype == "LINE":
            s = entity.dxf.start
            e = entity.dxf.end
            return (
                min(float(s.x), float(e.x)),
                min(float(s.y), float(e.y)),
                max(float(s.x), float(e.x)),
                max(float(s.y), float(e.y)),
            )
        elif dxftype in ("LWPOLYLINE", "POLYLINE"):
            pts = [(float(p[0]), float(p[1])) for p in entity.get_points()]
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return (min(xs), min(ys), max(xs), max(ys))
        p = ep.make_path(entity)
        bb = ep.bbox([p])
        return (float(bb.extmin.x), float(bb.extmin.y), float(bb.extmax.x), float(bb.extmax.y))
    except Exception:
        try:
            bb = ed_bbox.extents([entity])
            return (float(bb.extmin.x), float(bb.extmin.y), float(bb.extmax.x), float(bb.extmax.y))
        except Exception:
            return None


def _is_closed_rectangle(
    entity: Any,
    bbox: Tuple[float, float, float, float],
    tol: float = 0.08,
) -> bool:
    """Checks whether an entity is a closed rectangular panel border."""
    dxftype = entity.dxftype()
    if dxftype not in ("LWPOLYLINE", "POLYLINE"):
        return False

    is_cl = getattr(entity, "is_closed", False)
    if not is_cl:
        return False

    pts = [(float(p[0]), float(p[1])) for p in entity.get_points()]
    if len(pts) not in (4, 5):
        return False

    min_x, min_y, max_x, max_y = bbox
    bw = max_x - min_x
    bh = max_y - min_y
    if bw <= 1e-4 or bh <= 1e-4:
        return False

    # Check that vertices align closely with the 4 bounding box corners
    corners = {(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)}
    corner_tol = tol * max(bw, bh)
    matched = 0
    for cx, cy in corners:
        if any(math.hypot(px - cx, py - cy) <= corner_tol for px, py in pts):
            matched += 1

    return matched >= 4


def _find_nearest_design_label(
    panel_bbox: Tuple[float, float, float, float],
    labels: List[DxfTextLabel],
    max_vertical_ratio: float = 0.35,
) -> Optional[str]:
    """
    Finds the nearest TEXT/MTEXT label located below the panel
    within ~0.3 x panel height and horizontally overlapping the panel.
    """
    min_x, min_y, max_x, max_y = panel_bbox
    pw = max_x - min_x
    ph = max_y - min_y
    center_x = (min_x + max_x) / 2.0

    valid_candidates: List[Tuple[float, str]] = []

    for lbl in labels:
        lx = lbl.x
        ly = lbl.y
        txt = lbl.text.strip()
        if not txt:
            continue

        # Horizontal alignment check (within panel width plus 15% margin)
        if not (min_x - 0.15 * pw <= lx <= max_x + 0.15 * pw):
            continue

        # Vertical check: in DXF, Y points UP. Below the panel means ly <= min_y
        # Allow slight overlap into the bottom edge of the panel (up to 0.05 * ph)
        v_dist = min_y - ly
        if -0.05 * ph <= v_dist <= max_vertical_ratio * ph:
            # Score distance from the bottom center of the panel
            dist = math.hypot(lx - center_x, ly - min_y)
            valid_candidates.append((dist, txt))

    if not valid_candidates:
        return None

    valid_candidates.sort(key=lambda item: item[0])
    return valid_candidates[0][1]


def extract_designs(
    doc: ezdxf.document.Drawing,
    ignore_layers: Optional[Sequence[str]] = None,
) -> List[Design]:
    """
    Extracts individual laser-cut designs from a DXF document modelspace:
    - Recursively explodes INSERT blocks (virtual entities) to preserve transforms.
    - Filters non-cut layers and non-cut entities.
    - Detects panel borders via closed rectangular outlines or entity proximity clustering.
    - If drawing contains only one design, treats the whole drawing as one design.
    - Associates nearest TEXT/MTEXT label below panel as design_code.
    - Orders designs top-to-bottom, left-to-right.
    """
    t0 = time.time()
    layers_to_ignore = list(ignore_layers) if ignore_layers is not None else DEFAULT_IGNORE_LAYERS

    msp = doc.modelspace()
    stats: Dict[str, int] = {}
    geom_entities, labels = _explode_and_categorize(
        list(msp),
        doc=doc,
        ignore_layers=layers_to_ignore,
        stats=stats,
    )

    # Fallback: if all geometry was filtered, retry with minimal layer filtering
    if not geom_entities:
        logger.info("All entities filtered by ignore layers. Retrying with minimal filtering.")
        geom_entities, labels = _explode_and_categorize(
            list(msp),
            doc=doc,
            ignore_layers=["defpoints", "dim"],
            stats=stats,
        )

    if not geom_entities:
        logger.warning("No geometry found in DXF drawing.")
        return []

    # Compute bounding boxes for all geometry entities
    ent_bboxes: List[Tuple[Any, Tuple[float, float, float, float]]] = []
    for ent in geom_entities:
        bb = _get_entity_bbox(ent)
        if bb is not None and (bb[2] - bb[0] > 1e-5 or bb[3] - bb[1] > 1e-5):
            ent_bboxes.append((ent, bb))

    if not ent_bboxes:
        return []

    # Total drawing bounding box
    all_min_x = min(b[1][0] for b in ent_bboxes)
    all_min_y = min(b[1][1] for b in ent_bboxes)
    all_max_x = max(b[1][2] for b in ent_bboxes)
    all_max_y = max(b[1][3] for b in ent_bboxes)
    total_w = all_max_x - all_min_x
    total_h = all_max_y - all_min_y
    total_area = total_w * total_h

    # Step 1: Detect candidate closed panel border rectangles
    candidate_panel_rects: List[Tuple[Any, Tuple[float, float, float, float]]] = []
    for ent, bb in ent_bboxes:
        if _is_closed_rectangle(ent, bb):
            rw = bb[2] - bb[0]
            rh = bb[3] - bb[1]
            if rw * rh >= 0.005 * total_area:
                candidate_panel_rects.append((ent, bb))

    # Filter out title-block sheet frame enclosing >= 2 distinct panel rectangles
    if candidate_panel_rects:
        candidate_panel_rects.sort(key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]), reverse=True)
        largest_ent, largest_b = candidate_panel_rects[0]
        largest_area = (largest_b[2] - largest_b[0]) * (largest_b[3] - largest_b[1])

        # An inner rectangle is only a candidate panel if it contains geometry inside it
        # or has an associated text label, and has substantial area. Otherwise it is a cutout hole.
        def _is_panel_not_cutout(rect_ent, rect_bb):
            rw = rect_bb[2] - rect_bb[0]
            rh = rect_bb[3] - rect_bb[1]
            if rw * rh < 0.03 * total_area:
                return False
            # Check if it has internal entities
            has_children = any(
                (other_ent is not rect_ent)
                and (other_bb[0] >= rect_bb[0] - 1e-2 and other_bb[2] <= rect_bb[2] + 1e-2)
                and (other_bb[1] >= rect_bb[1] - 1e-2 and other_bb[3] <= rect_bb[3] + 1e-2)
                for other_ent, other_bb in ent_bboxes
            )
            has_label = _find_nearest_design_label(rect_bb, labels) is not None
            return has_children or has_label

        inner_rects = [
            r for r in candidate_panel_rects[1:]
            if r[1][0] >= largest_b[0] and r[1][2] <= largest_b[2]
            and r[1][1] >= largest_b[1] and r[1][3] <= largest_b[3]
            and _is_panel_not_cutout(r[0], r[1])
        ]
        if len(inner_rects) >= 2 and largest_area >= 0.80 * total_area:
            candidate_panel_rects = inner_rects

    # Deduplicate / filter nested borders
    detected_borders: List[Tuple[float, float, float, float]] = []
    for _, bb in candidate_panel_rects:
        # Check if inside an already selected border with larger area
        is_inner = False
        b_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        for ob in detected_borders:
            ob_area = (ob[2] - ob[0]) * (ob[3] - ob[1])
            if ob_area > b_area and ob[0] <= bb[0] and ob[2] >= bb[2] and ob[1] <= bb[1] and ob[3] >= bb[3]:
                is_inner = True
                break
        if not is_inner:
            # Check overlap with existing
            is_dup = False
            for ob in detected_borders:
                ix0 = max(bb[0], ob[0])
                iy0 = max(bb[1], ob[1])
                ix1 = min(bb[2], ob[2])
                iy1 = min(bb[3], ob[3])
                if ix1 > ix0 and iy1 > iy0:
                    inter_a = (ix1 - ix0) * (iy1 - iy0)
                    if inter_a / b_area > 0.8:
                        is_dup = True
                        break
            if not is_dup:
                detected_borders.append(bb)

    # Step 2: If no explicit closed rectangle borders, cluster entities via Union-Find on expanded bboxes
    panels_info: List[Dict[str, Any]] = []

    if len(detected_borders) >= 1:
        # Assign entities to detected borders
        for b_idx, p_box in enumerate(detected_borders):
            bx0, by0, bx1, by1 = p_box
            pad_x = 0.05 * (bx1 - bx0)
            pad_y = 0.05 * (by1 - by0)
            p_ents = []
            for ent, bb in ent_bboxes:
                if not (bb[2] < bx0 - pad_x or bb[0] > bx1 + pad_x or bb[3] < by0 - pad_y or bb[1] > by1 + pad_y):
                    p_ents.append(ent)
            if p_ents:
                panels_info.append({
                    "bbox": p_box,
                    "panel_rect": p_box,
                    "entities": p_ents,
                })

    # If still no panels detected or 0 borders found:
    if not panels_info:
        # Check if single panel or multi-panel cluster
        expand_dist = max(0.02 * max(total_w, total_h), 1.0)
        n_ents = len(ent_bboxes)
        parent = list(range(n_ents))

        def find(i: int) -> int:
            curr = i
            while parent[curr] != curr:
                parent[curr] = parent[parent[curr]]
                curr = parent[curr]
            return curr

        for i in range(n_ents):
            bi = ent_bboxes[i][1]
            e_bi = (bi[0] - expand_dist, bi[1] - expand_dist, bi[2] + expand_dist, bi[3] + expand_dist)
            for j in range(i + 1, n_ents):
                bj = ent_bboxes[j][1]
                # Check overlap
                if not (bj[2] < e_bi[0] or bj[0] > e_bi[2] or bj[3] < e_bi[1] or bj[1] > e_bi[3]):
                    ri = find(i)
                    rj = find(j)
                    if ri != rj:
                        parent[ri] = rj

        clusters: Dict[int, List[int]] = {}
        for i in range(n_ents):
            r = find(i)
            clusters.setdefault(r, []).append(i)

        valid_clusters = []
        for r, indices in clusters.items():
            c_min_x = min(ent_bboxes[idx][1][0] for idx in indices)
            c_min_y = min(ent_bboxes[idx][1][1] for idx in indices)
            c_max_x = max(ent_bboxes[idx][1][2] for idx in indices)
            c_max_y = max(ent_bboxes[idx][1][3] for idx in indices)
            c_area = (c_max_x - c_min_x) * (c_max_y - c_min_y)
            if c_area >= 0.005 * total_area or len(indices) >= 5:
                valid_clusters.append((indices, (c_min_x, c_min_y, c_max_x, c_max_y)))

        if len(valid_clusters) <= 1:
            # Single design drawing
            panels_info.append({
                "bbox": (all_min_x, all_min_y, all_max_x, all_max_y),
                "panel_rect": None,
                "entities": [e for e, _ in ent_bboxes],
            })
        else:
            for indices, c_box in valid_clusters:
                panels_info.append({
                    "bbox": c_box,
                    "panel_rect": None,
                    "entities": [ent_bboxes[idx][0] for idx in indices],
                })

    if not panels_info:
        panels_info.append({
            "bbox": (all_min_x, all_min_y, all_max_x, all_max_y),
            "panel_rect": None,
            "entities": [e for e, _ in ent_bboxes],
        })

    # Step 3: Sort designs top-to-bottom, left-to-right (in DXF: higher Y first, then lower X)
    median_h = float(np.median([p["bbox"][3] - p["bbox"][1] for p in panels_info]))
    row_tol = max(1e-3, 0.4 * median_h)

    def sort_key(p: Dict[str, Any]):
        bx0, by0, bx1, by1 = p["bbox"]
        row_bucket = -int(round(by1 / row_tol))
        return (row_bucket, bx0)

    panels_info.sort(key=sort_key)

    insunits = doc.header.get("$INSUNITS", 0)
    total_panels = len(panels_info)
    designs: List[Design] = []

    for idx, p in enumerate(panels_info, start=1):
        panel_box = p["bbox"]
        label_code = _find_nearest_design_label(panel_box, labels)

        d_obj = Design(
            design_id=f"design_{idx}",
            design_code=label_code,
            bbox=panel_box,
            panel_rect=p.get("panel_rect"),
            entities=p["entities"],
            index=idx,
            total_designs=total_panels,
            entity_count=len(p["entities"]),
            drawing_units=int(insunits),
        )
        designs.append(d_obj)

    elapsed = time.time() - t0
    logger.info(
        "Extracted %d designs (%d labels found) from DXF in %.2f s (skipped %d entities).",
        len(designs),
        len([d for d in designs if d.design_code]),
        elapsed,
        stats.get("skipped_layer", 0) + stats.get("skipped_type", 0),
    )
    return designs


# ---------------------------------------------------------------------------
# 4. Rendering Design Linework
# ---------------------------------------------------------------------------

def render_design(
    design: Design,
    px_height: int = 1024,
) -> np.ndarray:
    """
    Renders ONLY that design's geometry cropped to the panel border plus a small margin:
    - Preserves aspect ratio, never stretches.
    - Flattens curves (arcs, circles, ellipses, splines) with fine tolerance.
    - Draws cut lines with stroke thickness ~0.4% - 0.8% of panel width.
    - Returns a clean grayscale uint8 image (white lines on black background).
    """
    min_x, min_y, max_x, max_y = design.bbox
    bw = max(1e-4, max_x - min_x)
    bh = max(1e-4, max_y - min_y)

    margin_ratio = 0.015
    margin_px = max(4, int(round(px_height * margin_ratio)))
    avail_h = px_height - 2 * margin_px
    scale = avail_h / float(bh)

    canvas_w = int(round(bw * scale)) + 2 * margin_px
    canvas_h = px_height
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # Stroke thickness: ~0.6% of panel width (within 0.4% - 0.8%), minimum 2 px
    panel_px_w = int(round(bw * scale))
    stroke_thickness = max(2, int(round(panel_px_w * 0.006)))

    # Chord distance for curve flattening: fine tolerance (sub-pixel precision)
    flatten_dist = max(1e-4, 0.5 / scale)

    for entity in design.entities:
        dxftype = entity.dxftype()
        try:
            if dxftype == "CIRCLE":
                cx = float(entity.dxf.center.x)
                cy = float(entity.dxf.center.y)
                r = float(entity.dxf.radius)
                pix_cx = int(round(margin_px + (cx - min_x) * scale))
                pix_cy = int(round(margin_px + (max_y - cy) * scale))
                pix_r = max(1, int(round(r * scale)))
                cv2.circle(canvas, (pix_cx, pix_cy), pix_r, 255, thickness=stroke_thickness, lineType=cv2.LINE_AA)
                continue

            path = ep.make_path(entity)
            verts = list(path.flattening(distance=flatten_dist))
            if len(verts) < 2:
                continue

            pix_pts = np.empty((len(verts), 2), dtype=np.int32)
            for i, v in enumerate(verts):
                ix = int(round(margin_px + (v.x - min_x) * scale))
                iy = int(round(margin_px + (max_y - v.y) * scale))
                pix_pts[i, 0] = max(0, min(canvas_w - 1, ix))
                pix_pts[i, 1] = max(0, min(canvas_h - 1, iy))

            is_cl = bool(path.is_closed or (len(verts) >= 3 and math.hypot(verts[0].x - verts[-1].x, verts[0].y - verts[-1].y) <= flatten_dist * 2))
            cv2.polylines(canvas, [pix_pts], isClosed=is_cl, color=255, thickness=stroke_thickness, lineType=cv2.LINE_AA)
        except Exception as exc:
            logger.debug("Error rendering entity %s: %s", dxftype, exc)

    return canvas


# ---------------------------------------------------------------------------
# 5. Stencil Mask Conversion (Dual Polarity + Gap Closing + Flood Fill)
# ---------------------------------------------------------------------------

def convert_line_art_to_stencil_mask(
    rendered_gray: np.ndarray,
    has_explicit_border: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts rendered grayscale line art into the standardized stencil-mask format:
    1. Morphology CLOSE seals small gaps and stencil bridges.
    2. Flood-fills exterior and border regions to separate enclosed cut-outs from solid metal:
       - Outer panel / solid metal = White (255)
       - Cut-out apertures / voids = Black (0)
       - Solid islands inside holes = White (255)
    3. Removes the outer border cut line itself so the panel is solid metal to the edges.
    4. Returns (mask_standard, mask_inverted) for dual polarity indexing.
    """
    h, w = rendered_gray.shape[:2]

    # Step 1: Binarize and morphology close to seal small bridge gaps
    _, bin_lines = cv2.threshold(rendered_gray, 50, 255, cv2.THRESH_BINARY)
    k_close = max(5, int(round(max(h, w) * 0.012)))
    if k_close % 2 == 0:
        k_close += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    closed_lines = cv2.morphologyEx(bin_lines, cv2.MORPH_CLOSE, kernel)

    # Step 2: Find outermost contour representing the panel boundary
    ext_contours, _ = cv2.findContours(closed_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if ext_contours:
        panel_cnt = max(ext_contours, key=cv2.contourArea)
    else:
        panel_cnt = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.int32)

    # Step 3: Entire panel interior starts as solid metal (white 255)
    metal_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(metal_mask, [panel_cnt], -1, 255, thickness=-1)

    # Step 4: Empty space (background + cutouts) where lines are not present
    empty_regions = cv2.bitwise_not(closed_lines)
    empty_inside = cv2.bitwise_and(empty_regions, metal_mask)

    # Flood fill exterior background from image corners
    flooded = empty_regions.copy()
    for pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flooded[pt[1], pt[0]] == 255:
            cv2.floodFill(flooded, None, pt, 128)

    # Flood fill the metal body from inner rim of the border
    border_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(border_mask, [panel_cnt], -1, 255, thickness=k_close * 2)
    seed_pts = np.argwhere((empty_inside > 0) & (border_mask > 0))
    if len(seed_pts) > 0:
        step = max(1, len(seed_pts) // 10)
        for pt in seed_pts[::step]:
            sy, sx = pt
            if flooded[sy, sx] == 255:
                cv2.floodFill(flooded, None, (int(sx), int(sy)), 128)

    # Enclosed holes are unflooded empty regions inside the metal panel
    holes = (flooded == 255) & (metal_mask == 255)

    # Final standard mask: panel is metal (255), holes are 0, exterior is 0
    mask_std = metal_mask.copy()
    mask_std[holes] = 0

    # Inverted polarity: inside panel, holes are 255, metal is 0, exterior is 0
    mask_inv = np.zeros((h, w), dtype=np.uint8)
    mask_inv[(metal_mask == 255) & (mask_std == 0)] = 255

    return mask_std, mask_inv


# ---------------------------------------------------------------------------
# 6. Complete Ingestion Orchestrator
# ---------------------------------------------------------------------------

def process_dxf_file(
    dxf_path: Union[str, Path],
    px_height: int = 1024,
    ignore_layers: Optional[Sequence[str]] = None,
) -> List[Design]:
    """
    High-level ingestion function:
    1. Loads DXF CAD file.
    2. Extracts all designs and labels.
    3. Renders each design cleanly.
    4. Generates both standard and inverted stencil masks.
    """
    p = Path(dxf_path).resolve()
    doc = load_dxf(p)
    designs = extract_designs(doc, ignore_layers=ignore_layers)

    for d in designs:
        d.source_path = str(p)
        gray = render_design(d, px_height=px_height)
        m_std, m_inv = convert_line_art_to_stencil_mask(gray, has_explicit_border=d.panel_rect is not None)
        d.rendered_image = gray
        d.mask = m_std
        d.mask_inverted = m_inv

    return designs


# ---------------------------------------------------------------------------
# 7. Debug CLI: python dxf_ingest.py file.dxf --out debug/
# ---------------------------------------------------------------------------

def _run_debug_cli():
    """Command-line interface for testing, visualizing, and debugging DXF splitting."""
    parser = argparse.ArgumentParser(description="DXF Ingestion and Panel Splitting Debug CLI")
    parser.add_argument("dxf_file", help="Path to the input .dxf file")
    parser.add_argument("--out", default="debug", help="Output directory to save debug images (default: debug/)")
    parser.add_argument("--height", type=int, default=1024, help="Render pixel height (default: 1024)")
    args = parser.parse_args()

    in_path = Path(args.dxf_file).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[DXF Ingest CLI] Processing: {in_path.name}")
    t0 = time.time()

    designs = process_dxf_file(in_path, px_height=args.height)
    elapsed = time.time() - t0

    print(f"Extracted {len(designs)} designs in {elapsed:.2f}s:\n")

    for d in designs:
        code_str = d.design_code or "NO_LABEL"
        print(f"  • Design {d.index}/{d.total_designs}: Code='{code_str}', BBox={tuple(round(v, 1) for v in d.bbox)}, Entities={d.entity_count}")

        # Save rendered linework
        if d.rendered_image is not None:
            cv2.imwrite(str(out_dir / f"panel_{d.index}_rendered.png"), d.rendered_image)
        # Save standard stencil mask
        if d.mask is not None:
            cv2.imwrite(str(out_dir / f"panel_{d.index}_mask_normal.png"), d.mask)
        # Save inverted stencil mask
        if d.mask_inverted is not None:
            cv2.imwrite(str(out_dir / f"panel_{d.index}_mask_inverted.png"), d.mask_inverted)

    # Generate full drawing overlay showing detected panels and text labels
    try:
        doc = load_dxf(in_path)
        msp = doc.modelspace()
        geom_ents, labels = _explode_and_categorize(list(msp), doc=doc, ignore_layers=DEFAULT_IGNORE_LAYERS)
        if geom_ents:
            # Render whole drawing at modest scale
            all_paths = []
            for e in geom_ents:
                try:
                    all_paths.append(ep.make_path(e))
                except Exception:
                    pass
            if all_paths:
                bb = ep.bbox(all_paths)
                min_x, min_y = float(bb.extmin.x), float(bb.extmin.y)
                max_x, max_y = float(bb.extmax.x), float(bb.extmax.y)
                dw = max(1e-4, max_x - min_x)
                dh = max(1e-4, max_y - min_y)

                canvas_dim = 1600
                scale = float(canvas_dim - 64) / max(dw, dh)
                ow = int(round(dw * scale)) + 64
                oh = int(round(dh * scale)) + 64

                overlay_bgr = np.full((oh, ow, 3), (25, 20, 15), dtype=np.uint8)

                # Draw linework in grey
                for p_obj in all_paths:
                    for v in p_obj.flattening(distance=max(1e-4, 1.0 / scale)):
                        ix = int(round(32 + (v.x - min_x) * scale))
                        iy = int(round(32 + (max_y - v.y) * scale))
                        if 0 <= ix < ow and 0 <= iy < oh:
                            overlay_bgr[iy, ix] = (180, 180, 185)

                # Draw detected panel bounding boxes and labels in bright cyan / green
                colors = [(0, 255, 0), (255, 200, 0), (0, 255, 255), (100, 150, 255)]
                for d in designs:
                    col = colors[(d.index - 1) % len(colors)]
                    px0 = int(round(32 + (d.bbox[0] - min_x) * scale))
                    py0 = int(round(32 + (max_y - d.bbox[3]) * scale))
                    px1 = int(round(32 + (d.bbox[2] - min_x) * scale))
                    py1 = int(round(32 + (max_y - d.bbox[1]) * scale))

                    cv2.rectangle(overlay_bgr, (px0, py0), (px1, py1), col, 3)

                    lbl_text = f"#{d.index}: {d.design_code or 'No Label'}"
                    cv2.putText(
                        overlay_bgr,
                        lbl_text,
                        (px0 + 4, max(20, py0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        col,
                        2,
                        cv2.LINE_AA,
                    )

                overlay_path = out_dir / "overlay_detected_panels.png"
                cv2.imwrite(str(overlay_path), overlay_bgr)
                cv2.imwrite(str(out_dir / f"{in_path.stem}_overview_overlay.png"), overlay_bgr)
                print(f"\nOverlay with detected bounding boxes saved to: {overlay_path}")
    except Exception as exc:
        logger.warning("Could not generate overlay image: %s", exc)

    print(f"All debug images saved successfully to '{out_dir}'.\n")


if __name__ == "__main__":
    _run_debug_cli()
