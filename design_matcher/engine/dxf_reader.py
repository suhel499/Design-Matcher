"""
DXF reader module for CAD line drawing extraction.
Uses ezdxf (MIT license - commercial safe).
Extracts normalized 2D linework, text labels, and bounding boxes from modelspace and blocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ezdxf
import ezdxf.bbox as ed_bbox
import ezdxf.path as ep
import numpy as np

from .config import EngineConfig, config as default_config

logger = logging.getLogger(__name__)


class DxfReaderError(Exception):
    """Raised when a DXF file cannot be parsed or read."""
    pass


@dataclass
class DxfLabel:
    """Represents a text entity extracted from a DXF drawing."""
    text: str
    x: float
    y: float
    height: float


@dataclass
class DxfDrawingData:
    """Contains all extracted linework, text labels, and geometry bounds from a DXF."""
    polylines: List[Tuple[np.ndarray, bool]]  # List of (Nx2 float64 points, is_closed)
    labels: List[DxfLabel]
    bbox: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _is_layer_ignored(
    layer_name: str,
    doc: ezdxf.document.Drawing,
    ignore_list: List[str],
    strict_ignore: bool = True,
) -> bool:
    """
    Checks if an entity layer is OFF, FROZEN, or matches configurable ignore patterns.
    """
    lname_lower = layer_name.lower().strip()
    try:
        layer = doc.layers.get(layer_name)
        if layer is not None:
            if layer.is_off() or layer.is_frozen():
                return True
    except Exception:
        pass

    if strict_ignore:
        for pattern in ignore_list:
            if pattern.lower() in lname_lower:
                return True

    return False


def _explode_entities_recursive(
    entities,
    doc: ezdxf.document.Drawing,
    ignore_list: List[str],
    strict_ignore: bool,
    max_depth: int,
    current_depth: int = 0,
    diag: Optional[Dict[str, int]] = None,
) -> Tuple[List[Any], List[DxfLabel]]:
    """
    Recursively unpacks entities, exploding INSERT blocks up to max_depth.
    Separates path entities from text labels and records entity diagnostics.
    """
    path_entities: List[Any] = []
    labels: List[DxfLabel] = []

    if diag is None:
        diag = {}
    diag.setdefault("total", 0)
    diag.setdefault("skipped_layer", 0)
    diag.setdefault("skipped_type", 0)
    diag.setdefault("inserts", 0)
    diag.setdefault("paths", 0)
    diag.setdefault("texts", 0)

    for entity in entities:
        diag["total"] += 1
        dxftype = entity.dxftype()

        # Check layer visibility
        layer_name = getattr(entity.dxf, "layer", "0")
        if _is_layer_ignored(layer_name, doc, ignore_list, strict_ignore=strict_ignore):
            # Allow text entities to be inspected for labels even if layer is named like text
            if dxftype in ("TEXT", "MTEXT") and "text" in layer_name.lower():
                pass
            else:
                diag["skipped_layer"] += 1
                continue

        if dxftype in ("TEXT", "MTEXT"):
            diag["texts"] += 1
            try:
                if dxftype == "MTEXT":
                    txt = entity.plain_text().strip()
                    h = float(getattr(entity.dxf, "char_height", 10.0))
                else:
                    txt = str(getattr(entity.dxf, "text", "")).strip()
                    h = float(getattr(entity.dxf, "height", 10.0))

                ins = getattr(entity.dxf, "insert", (0.0, 0.0, 0.0))
                if txt:
                    labels.append(DxfLabel(text=txt, x=float(ins[0]), y=float(ins[1]), height=h))
            except Exception as exc:
                logger.debug("Failed extracting text from %s: %s", dxftype, exc)
            continue

        if dxftype in ("DIMENSION", "POINT", "LEADER", "MULTILEADER"):
            diag["skipped_type"] += 1
            continue

        if dxftype == "INSERT":
            diag["inserts"] += 1
            if current_depth < max_depth:
                try:
                    sub_ents = list(entity.virtual_entities())
                    sub_paths, sub_labels = _explode_entities_recursive(
                        sub_ents,
                        doc=doc,
                        ignore_list=ignore_list,
                        strict_ignore=strict_ignore,
                        max_depth=max_depth,
                        current_depth=current_depth + 1,
                        diag=diag,
                    )
                    path_entities.extend(sub_paths)
                    labels.extend(sub_labels)
                except Exception as exc:
                    logger.debug("Error expanding INSERT virtual entities: %s", exc)
            continue

        # Supported geometry entities
        if dxftype in ("LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE", "LWPOLYLINE", "POLYLINE", "HATCH"):
            path_entities.append(entity)
            diag["paths"] += 1
        else:
            diag["skipped_type"] += 1

    return path_entities, labels


def _entity_to_paths(entity) -> List[ep.Path]:
    """Converts a supported DXF entity to one or more ezdxf.path.Path objects."""
    dxftype = entity.dxftype()
    try:
        if dxftype == "HATCH":
            return list(ep.from_hatch(entity))
        else:
            p = ep.make_path(entity)
            return [p]
    except Exception as exc:
        logger.debug("Could not convert entity %s to Path: %s", dxftype, exc)
        return []


def read_dxf(
    dxf_path: Union[str, Path],
    cfg: Optional[EngineConfig] = None,
) -> DxfDrawingData:
    """
    Reads a DXF file and extracts clean 2D vector linework and text labels.

    Args:
        dxf_path: Path to the .dxf file.
        cfg: Optional EngineConfig instance with tuning parameters.

    Returns:
        DxfDrawingData with polylines, labels, bounding box, and diagnostics.
    """
    c = cfg or default_config
    p = Path(dxf_path).resolve()

    if not p.is_file():
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")

    # Check for unsupported DWG format
    if p.suffix.lower() == ".dwg":
        raise DxfReaderError("DWG not supported: please export to DXF")

    # Step 1: Open with ezdxf.recover to tolerate slightly corrupted or non-standard files
    doc = None
    try:
        doc, auditor = ezdxf.recover.readfile(str(p))
        if auditor.has_errors:
            logger.debug("Recovered DXF %s with %d errors fixed", p.name, len(auditor.errors))
    except Exception as recover_exc:
        logger.debug("ezdxf.recover failed for %s (%s), attempting regular readfile", p.name, recover_exc)
        try:
            doc = ezdxf.readfile(str(p))
        except Exception as read_exc:
            raise DxfReaderError(f"Failed to open or parse DXF {p.name}: {read_exc}") from read_exc

    if doc is None:
        raise DxfReaderError(f"Failed to obtain valid document for {p.name}")

    msp = doc.modelspace()
    diag: Dict[str, Any] = {"total_entities": len(msp)}

    # Step 2: Explode entities with layer filtering
    ignore_layers = list(c.DXF_IGNORE_LAYERS) if c.DXF_IGNORE_LAYERS else []
    path_entities, labels = _explode_entities_recursive(
        msp,
        doc=doc,
        ignore_list=ignore_layers,
        strict_ignore=True,
        max_depth=c.DXF_MAX_BLOCK_DEPTH,
        diag=diag,
    )

    used_fallback_layers = False
    # If strict ignore rules removed ALL geometry, retry with minimal layer filtering
    if not path_entities:
        logger.info("DXF %s: All geometry filtered by ignore rules. Retrying with minimal layer filtering.", p.name)
        diag_fallback: Dict[str, Any] = {"total_entities": len(msp)}
        path_entities, labels = _explode_entities_recursive(
            msp,
            doc=doc,
            ignore_list=["dim", "defpoints"],
            strict_ignore=False,
            max_depth=c.DXF_MAX_BLOCK_DEPTH,
            diag=diag_fallback,
        )
        used_fallback_layers = True
        diag["fallback_filtering"] = diag_fallback

    diag["used_fallback_layers"] = used_fallback_layers

    # Step 3: Convert entities to paths
    raw_paths: List[ep.Path] = []
    for ent in path_entities:
        raw_paths.extend(_entity_to_paths(ent))

    if not raw_paths:
        diag["empty"] = True
        return DxfDrawingData(
            polylines=[],
            labels=labels,
            bbox=(0.0, 0.0, 0.0, 0.0),
            diagnostics=diag,
        )

    # Step 4: Compute drawing bounding box
    try:
        bbox_obj = ep.bbox(raw_paths)
        min_x, min_y = float(bbox_obj.extmin.x), float(bbox_obj.extmin.y)
        max_x, max_y = float(bbox_obj.extmax.x), float(bbox_obj.extmax.y)
    except Exception as exc:
        raise DxfReaderError(f"Failed to compute bounding box for {p.name}: {exc}") from exc

    w = max(0.0, max_x - min_x)
    h = max(0.0, max_y - min_y)
    diagonal = float(np.hypot(w, h))

    # Adaptive curve flattening distance: 0.05% of drawing diagonal (min 1e-4)
    flatten_dist = max(1e-4, diagonal * float(c.DXF_FLATTEN_DIST_RATIO))

    # Step 5: Flatten paths into point arrays
    polylines: List[Tuple[np.ndarray, bool]] = []
    for path in raw_paths:
        try:
            verts = list(path.flattening(distance=flatten_dist))
            if len(verts) < 2:
                continue
            pts = np.array([(float(v.x), float(v.y)) for v in verts], dtype=np.float64)
            is_closed = bool(path.is_closed and len(verts) >= 3)
            polylines.append((pts, is_closed))
        except Exception as exc:
            logger.debug("Error flattening path: %s", exc)
            continue

    diag.update({
        "polyline_count": len(polylines),
        "label_count": len(labels),
        "diagonal": diagonal,
        "width": w,
        "height": h,
    })

    return DxfDrawingData(
        polylines=polylines,
        labels=labels,
        bbox=(min_x, min_y, max_x, max_y),
        diagnostics=diag,
    )
