"""
dxf_parser.py - Production-Grade DXF Parsing and Geometry Extraction.

Features:
- Handles LINE, LWPOLYLINE, POLYLINE, CIRCLE, ARC, ELLIPSE, SPLINE, INSERT blocks, HATCH boundaries.
- Correctly accumulates block transformations using virtual entity expansion.
- Prevents infinite recursion and cycle loops with visited sets and depth limits.
- Filters non-geometric layers (dimensions, defpoints, text, annotations) with safe fallback.
- Detects multi-design drawings (panel borders and spatial clusters).
- Robust error recovery using ezdxf.recover and graceful handling of empty or corrupt files.
- Preserves design components, entity counts, labels, and drawing units.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import ezdxf
import ezdxf.bbox as ed_bbox
import ezdxf.document
import ezdxf.path as ep
import ezdxf.recover
import numpy as np

logger = logging.getLogger(__name__)

# Common layers to ignore
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
    "frame-only",
    "border_sheet",
    "grid",
    "axis",
    "viewport",
    "vport",
]

DEFAULT_IGNORE_ENTITY_TYPES = {
    "DIMENSION",
    "POINT",
    "LEADER",
    "MULTILEADER",
    "RAY",
    "XLINE",
}

SUPPORTED_GEOM_TYPES = {
    "LINE",
    "LWPOLYLINE",
    "POLYLINE",
    "ARC",
    "CIRCLE",
    "ELLIPSE",
    "SPLINE",
    "HATCH",
}


class DxfParserError(Exception):
    """Raised when DXF parsing encounters an unrecoverable error."""
    pass


@dataclass
class DxfAnnotation:
    """Represents text or label annotation found in DXF."""
    text: str
    x: float
    y: float
    height: float = 10.0


@dataclass
class ExtractedPolyline:
    """Represents a continuous 2D polyline extracted from DXF entities."""
    points: np.ndarray             # (N, 2) float64 array of coordinates
    is_closed: bool = False
    source_layer: str = "0"
    source_type: str = "UNKNOWN"
    entity_handle: Optional[str] = None


@dataclass
class ExtractedComponent:
    """Represents an isolated design or sub-panel within a DXF drawing."""
    component_index: int
    name: str                      # Associated label or default code (e.g. 'PD 010' or 'Component 1')
    bbox: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    polylines: List[ExtractedPolyline] = field(default_factory=list)
    entity_count: int = 0
    is_closed_panel: bool = False
    outer_border: Optional[np.ndarray] = None

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return (self.width / h) if h > 1e-6 else 1.0


@dataclass
class ParsedDxfDrawing:
    """Complete parsed DXF drawing data with extracted geometry components."""
    file_path: str
    drawing_units: int             # AutoCAD $INSUNITS
    total_entity_count: int
    overall_bbox: Tuple[float, float, float, float]
    components: List[ExtractedComponent]
    annotations: List[DxfAnnotation]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _is_layer_ignored(
    layer_name: str,
    doc: ezdxf.document.Drawing,
    ignore_list: List[str],
    strict: bool = True,
) -> bool:
    """Checks if a layer is turned off, frozen, or in the ignore list."""
    lname_lower = layer_name.lower().strip()
    try:
        layer = doc.layers.get(layer_name)
        if layer is not None and (layer.is_off() or layer.is_frozen()):
            return True
    except Exception:
        pass

    if strict:
        for pat in ignore_list:
            if pat.lower() in lname_lower:
                return True
    return False


def _explode_entities_recursive(
    entities,
    doc: ezdxf.document.Drawing,
    ignore_layers: List[str],
    strict_layers: bool,
    max_depth: int = 8,
    current_depth: int = 0,
    visited_blocks: Optional[Set[str]] = None,
    diag: Optional[Dict[str, int]] = None,
) -> Tuple[List[Any], List[DxfAnnotation]]:
    """
    Recursively decomposes entities and block references (INSERT) applying transformations.
    Guards against cycle references and unbounded recursion.
    """
    if visited_blocks is None:
        visited_blocks = set()
    if diag is None:
        diag = {}

    diag.setdefault("total_inspected", 0)
    diag.setdefault("skipped_layer", 0)
    diag.setdefault("skipped_type", 0)
    diag.setdefault("geom_entities", 0)
    diag.setdefault("insert_blocks", 0)
    diag.setdefault("text_labels", 0)

    geom_entities: List[Any] = []
    annotations: List[DxfAnnotation] = []

    for ent in entities:
        diag["total_inspected"] += 1
        dxftype = ent.dxftype()

        # Check layer
        layer_name = str(getattr(ent.dxf, "layer", "0"))
        if _is_layer_ignored(layer_name, doc, ignore_layers, strict=strict_layers):
            if dxftype in ("TEXT", "MTEXT") and "text" in layer_name.lower():
                # Allow text extraction for annotations even on text layers
                pass
            else:
                diag["skipped_layer"] += 1
                continue

        # Extract text annotations
        if dxftype in ("TEXT", "MTEXT"):
            diag["text_labels"] += 1
            try:
                if dxftype == "MTEXT":
                    txt = ent.plain_text().strip()
                    h = float(getattr(ent.dxf, "char_height", 10.0))
                else:
                    txt = str(getattr(ent.dxf, "text", "")).strip()
                    h = float(getattr(ent.dxf, "height", 10.0))
                ins = getattr(ent.dxf, "insert", (0.0, 0.0, 0.0))
                if txt:
                    annotations.append(DxfAnnotation(text=txt, x=float(ins[0]), y=float(ins[1]), height=h))
            except Exception as e:
                logger.debug("Failed extracting text annotation: %s", e)
            continue

        if dxftype in DEFAULT_IGNORE_ENTITY_TYPES:
            diag["skipped_type"] += 1
            continue

        # Explode INSERT blocks
        if dxftype == "INSERT":
            diag["insert_blocks"] += 1
            block_name = str(getattr(ent.dxf, "name", ""))
            block_key = f"{block_name}@{current_depth}"
            if current_depth < max_depth and block_key not in visited_blocks:
                visited_blocks.add(block_key)
                try:
                    sub_ents = list(ent.virtual_entities())
                    sub_geom, sub_ann = _explode_entities_recursive(
                        sub_ents,
                        doc=doc,
                        ignore_layers=ignore_layers,
                        strict_layers=strict_layers,
                        max_depth=max_depth,
                        current_depth=current_depth + 1,
                        visited_blocks=visited_blocks,
                        diag=diag,
                    )
                    geom_entities.extend(sub_geom)
                    annotations.extend(sub_ann)
                except Exception as e:
                    logger.debug("Error expanding virtual entities for block '%s': %s", block_name, e)
                finally:
                    visited_blocks.remove(block_key)
            continue

        # Supported geometry entities
        if dxftype in SUPPORTED_GEOM_TYPES:
            geom_entities.append(ent)
            diag["geom_entities"] += 1
        else:
            diag["skipped_type"] += 1

    return geom_entities, annotations


def _entity_to_paths(entity) -> List[ep.Path]:
    """Converts a supported DXF entity to one or more ezdxf.path.Path objects."""
    dxftype = entity.dxftype()
    try:
        if dxftype == "HATCH":
            return list(ep.from_hatch(entity))
        else:
            p = ep.make_path(entity)
            return [p]
    except Exception as e:
        logger.debug("Could not make path for %s: %s", dxftype, e)
        return []


def _extract_polylines_from_paths(
    paths: List[ep.Path],
    flatten_dist: float,
) -> List[ExtractedPolyline]:
    """Flattens ezdxf paths into ExtractedPolyline objects using adaptive flattening distance."""
    result: List[ExtractedPolyline] = []
    for path in paths:
        try:
            verts = list(path.flattening(distance=flatten_dist))
            if len(verts) < 2:
                continue
            pts = np.array([(float(v.x), float(v.y)) for v in verts], dtype=np.float64)
            # Check if closed
            is_closed = bool(path.is_closed or (len(pts) >= 3 and np.linalg.norm(pts[0] - pts[-1]) < max(1e-4, flatten_dist * 2.0)))
            result.append(ExtractedPolyline(points=pts, is_closed=is_closed))
        except Exception as e:
            logger.debug("Error flattening path: %s", e)
            continue
    return result


def _find_matching_annotation(
    bbox: Tuple[float, float, float, float],
    annotations: List[DxfAnnotation],
    default_name: str,
) -> str:
    """Associates the closest text annotation (e.g. 'PD 010') to a component box."""
    if not annotations:
        return default_name

    min_x, min_y, max_x, max_y = bbox
    w = max(1e-4, max_x - min_x)
    h = max(1e-4, max_y - min_y)
    h_margin = 0.25 * w

    cands: List[Tuple[float, str]] = []
    for ann in annotations:
        txt = ann.text.strip()
        if not txt:
            continue
        # Horizontal range check
        if min_x - h_margin <= ann.x <= max_x + h_margin:
            # Vertical check (below or above within 1.5 * h)
            if ann.y <= min_y and (min_y - ann.y) <= 1.5 * h:
                cands.append((min_y - ann.y, txt))
            elif ann.y >= max_y and (ann.y - max_y) <= 1.5 * h:
                cands.append((ann.y - max_y, txt))

    if cands:
        cands.sort(key=lambda item: item[0])
        return cands[0][1]

    return default_name


def parse_dxf_drawing(
    dxf_path: Union[str, Path],
    ignore_layers: Optional[List[str]] = None,
    flatten_dist_ratio: float = 0.0005,
) -> ParsedDxfDrawing:
    """
    Parses a DXF file and extracts clean vector geometry components and annotations.

    Args:
        dxf_path: Path to the .dxf file.
        ignore_layers: Optional list of layer names/patterns to exclude.
        flatten_dist_ratio: Ratio of diagonal to use for curve chord flattening.

    Returns:
        ParsedDxfDrawing with segmented components and annotations.
    """
    p = Path(dxf_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"DXF file does not exist: {p}")

    if p.suffix.lower() == ".dwg":
        raise DxfParserError("DWG format is not supported: please export or convert to DXF.")

    # 1. Open with recover fallback
    doc: Optional[ezdxf.document.Drawing] = None
    try:
        doc = ezdxf.readfile(str(p))
    except Exception as e_read:
        logger.debug("Standard ezdxf.readfile failed (%s); trying recover.readfile...", e_read)
        try:
            doc, auditor = ezdxf.recover.readfile(str(p))
            if auditor.has_errors:
                logger.debug("Recovered %s with %d errors fixed.", p.name, len(auditor.errors))
        except Exception as e_rec:
            raise DxfParserError(f"Failed to open DXF {p.name}: {e_rec}") from e_rec

    if doc is None:
        raise DxfParserError(f"Could not load valid document from {p.name}")

    drawing_units = int(doc.header.get("$INSUNITS", 0))
    msp = doc.modelspace()
    diag: Dict[str, Any] = {"modelspace_entities": len(msp)}

    active_ignore = list(ignore_layers) if ignore_layers is not None else DEFAULT_IGNORE_LAYERS

    # 2. Explode entities
    geom_entities, annotations = _explode_entities_recursive(
        msp,
        doc=doc,
        ignore_layers=active_ignore,
        strict_layers=True,
        max_depth=8,
        diag=diag,
    )

    # Fallback if strict layers filtered everything
    if not geom_entities:
        logger.info("DXF %s: All geometry filtered by strict layers. Retrying with minimal filter.", p.name)
        geom_entities, annotations = _explode_entities_recursive(
            msp,
            doc=doc,
            ignore_layers=["defpoints", "dim"],
            strict_layers=False,
            max_depth=8,
            diag=diag,
        )

    if not geom_entities:
        return ParsedDxfDrawing(
            file_path=str(p),
            drawing_units=drawing_units,
            total_entity_count=0,
            overall_bbox=(0.0, 0.0, 0.0, 0.0),
            components=[],
            annotations=annotations,
            diagnostics={"empty": True, **diag},
        )

    # 3. Convert entities to ezdxf Paths
    raw_paths: List[ep.Path] = []
    for ent in geom_entities:
        raw_paths.extend(_entity_to_paths(ent))

    if not raw_paths:
        return ParsedDxfDrawing(
            file_path=str(p),
            drawing_units=drawing_units,
            total_entity_count=len(geom_entities),
            overall_bbox=(0.0, 0.0, 0.0, 0.0),
            components=[],
            annotations=annotations,
            diagnostics={"no_paths": True, **diag},
        )

    # 4. Overall bounding box & adaptive flattening
    try:
        bbox_obj = ep.bbox(raw_paths)
        min_x, min_y = float(bbox_obj.extmin.x), float(bbox_obj.extmin.y)
        max_x, max_y = float(bbox_obj.extmax.x), float(bbox_obj.extmax.y)
    except Exception as e:
        raise DxfParserError(f"Failed to compute bounding box for {p.name}: {e}") from e

    w = max(0.0, max_x - min_x)
    h = max(0.0, max_y - min_y)
    diagonal = float(math.hypot(w, h))
    flatten_dist = max(1e-4, diagonal * flatten_dist_ratio)

    polylines = _extract_polylines_from_paths(raw_paths, flatten_dist)
    if not polylines:
        return ParsedDxfDrawing(
            file_path=str(p),
            drawing_units=drawing_units,
            total_entity_count=len(geom_entities),
            overall_bbox=(min_x, min_y, max_x, max_y),
            components=[],
            annotations=annotations,
            diagnostics={"no_polylines": True, **diag},
        )

    # 5. Component / Panel Detection
    # Detect if drawing contains multiple distinct panel borders (e.g. catalog sheets)
    components = _segment_into_components(polylines, (min_x, min_y, max_x, max_y), annotations)

    diag.update({
        "total_polylines": len(polylines),
        "total_components": len(components),
        "diagonal": diagonal,
        "width": w,
        "height": h,
    })

    return ParsedDxfDrawing(
        file_path=str(p),
        drawing_units=drawing_units,
        total_entity_count=len(geom_entities),
        overall_bbox=(min_x, min_y, max_x, max_y),
        components=components,
        annotations=annotations,
        diagnostics=diag,
    )


def _segment_into_components(
    polylines: List[ExtractedPolyline],
    overall_bbox: Tuple[float, float, float, float],
    annotations: List[DxfAnnotation],
) -> List[ExtractedComponent]:
    """
    Segments extracted polylines into individual design components:
    - Identifies outer closed panel borders.
    - If multiple outer closed panel rectangles exist, assigns inner curves to respective panel containers.
    - Otherwise returns the whole drawing as a single component.
    """
    if not polylines:
        return []

    # Calculate bounding box for each polyline
    bbs = []
    closed_flags = []
    areas = []
    for pl in polylines:
        pts = pl.points
        bx0, by0 = pts[:, 0].min(), pts[:, 1].min()
        bx1, by1 = pts[:, 0].max(), pts[:, 1].max()
        pw, ph = max(1e-6, bx1 - bx0), max(1e-6, by1 - by0)
        bbs.append((bx0, by0, bx1, by1))
        closed_flags.append(pl.is_closed)
        areas.append(pw * ph)

    bbs_arr = np.array(bbs)
    areas_arr = np.array(areas)
    closed_arr = np.array(closed_flags, dtype=bool)

    # Find closed outer panels that are not contained inside another closed curve
    n = len(polylines)
    closed_indices = np.where(closed_arr)[0]
    is_contained = np.zeros(n, dtype=bool)

    if len(closed_indices) > 0:
        c_bbs = bbs_arr[closed_indices]
        c_areas = areas_arr[closed_indices]

        # Vectorized containment check across all closed candidates
        cond_x0 = c_bbs[:, 0:1] >= c_bbs[:, 0].T - 1e-2
        cond_y0 = c_bbs[:, 1:2] >= c_bbs[:, 1].T - 1e-2
        cond_x1 = c_bbs[:, 2:3] <= c_bbs[:, 2].T + 1e-2
        cond_y1 = c_bbs[:, 3:4] <= c_bbs[:, 3].T + 1e-2
        cond_area = c_areas[:, None] < c_areas[None, :]

        inside_matrix = cond_x0 & cond_y0 & cond_x1 & cond_y1 & cond_area
        is_contained[closed_indices] = inside_matrix.any(axis=1)

    candidate_borders = np.where(closed_arr & (~is_contained))[0]

    # Filter candidate borders: if multiple, check aspect ratio & area
    total_area = areas_arr.sum()
    if len(candidate_borders) > 1:
        pw = bbs_arr[candidate_borders, 2] - bbs_arr[candidate_borders, 0]
        ph = bbs_arr[candidate_borders, 3] - bbs_arr[candidate_borders, 1]
        c_areas = areas_arr[candidate_borders]
        med_area = np.median(c_areas)
        # Keep candidate borders with at least 25% of median area
        valid_mask = c_areas > 0.25 * med_area
        candidate_borders = candidate_borders[valid_mask]

    # If we have multiple distinct panel borders (>= 2):
    if len(candidate_borders) >= 2:
        # Sort in reading order (top to bottom, left to right)
        cy = (bbs_arr[candidate_borders, 1] + bbs_arr[candidate_borders, 3]) / 2.0
        cx = (bbs_arr[candidate_borders, 0] + bbs_arr[candidate_borders, 2]) / 2.0
        med_h = np.median(bbs_arr[candidate_borders, 3] - bbs_arr[candidate_borders, 1])
        order = np.lexsort((cx, -np.round(cy / max(med_h, 1e-3))))
        candidate_borders = candidate_borders[order]

        components: List[ExtractedComponent] = []
        for idx, border_idx in enumerate(candidate_borders, start=1):
            bbox_i = tuple(bbs_arr[border_idx])
            name = _find_matching_annotation(bbox_i, annotations, f"Panel {idx}")
            comp = ExtractedComponent(
                component_index=idx,
                name=name,
                bbox=bbox_i,
                polylines=[polylines[border_idx]],
                entity_count=1,
                is_closed_panel=True,
                outer_border=polylines[border_idx].points,
            )
            components.append(comp)

        # Assign each other polyline to the component whose bounding box contains its center
        mid_x = (bbs_arr[:, 0] + bbs_arr[:, 2]) / 2.0
        mid_y = (bbs_arr[:, 1] + bbs_arr[:, 3]) / 2.0
        comp_boxes = np.array([comp.bbox for comp in components], dtype=np.float64)

        inside_comp = (
            (mid_x[:, None] >= comp_boxes[:, 0] - 1e-2)
            & (mid_x[:, None] <= comp_boxes[:, 2] + 1e-2)
            & (mid_y[:, None] >= comp_boxes[:, 1] - 1e-2)
            & (mid_y[:, None] <= comp_boxes[:, 3] + 1e-2)
        )
        has_match = inside_comp.any(axis=1)
        assigned_comp = inside_comp.argmax(axis=1)
        cand_border_set = set(candidate_borders)

        for k in range(n):
            if k in cand_border_set or not has_match[k]:
                continue
            c_idx = int(assigned_comp[k])
            components[c_idx].polylines.append(polylines[k])
            components[c_idx].entity_count += 1

        # Filter out empty dummy components
        valid_comps = [c for c in components if len(c.polylines) > 0]
        if valid_comps:
            return valid_comps

    # Otherwise: single component for the entire drawing
    name = _find_matching_annotation(overall_bbox, annotations, "Design 1")
    outer_pts = polylines[0].points if (len(polylines) > 0 and polylines[0].is_closed) else None
    return [
        ExtractedComponent(
            component_index=1,
            name=name,
            bbox=overall_bbox,
            polylines=polylines,
            entity_count=len(polylines),
            is_closed_panel=bool(outer_pts is not None),
            outer_border=outer_pts,
        )
    ]
