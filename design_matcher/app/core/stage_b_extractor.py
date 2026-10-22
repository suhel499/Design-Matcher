"""
Stage B: Visual Rendering and Tile Extraction Worker.
Processes drawing geometry and PDF raster pages into standardized 224x224
binary masks and thumbnails.
Completely decoupled from PyTorch/DINOv2 embedding (which executes in the main
process to guarantee safety, avoid model unpickling overhead, and optimize batch throughput).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.pdf_render import render_pdf_page
from app.core.preprocess import letterbox_image
from app.core.tiles import detect_tile_boxes, extract_tile_crops

logger = logging.getLogger(__name__)


def process_pdf_page_visual(
    pdf_path: str,
    page_idx: int,
    dpi: int = 200,
    include_subregions: bool = True,
    run_ocr: bool = True,
) -> Dict[str, Any]:
    """
    Renders a single PDF page, detects design tiles, segments masks, and generates thumbnails.
    Picklable worker function safe for ProcessPoolExecutor.
    """
    p = Path(pdf_path)
    res: Dict[str, Any] = {
        "pdf_path": str(p.resolve()),
        "page_idx": page_idx,
        "tiles": [],
        "error": None,
    }

    try:
        page_bgr = render_pdf_page(p, page_idx, dpi=dpi)
        tile_boxes = detect_tile_boxes(page_bgr)
        crops = extract_tile_crops(
            page_bgr,
            tile_boxes,
            include_subregions=include_subregions,
            run_ocr=run_ocr,
        )

        tiles_data = []
        for crop in crops:
            if crop.stencil_mask is None or crop.stencil_mask.size == 0:
                continue

            # Standard 224x224 DINOv2 input mask
            m_224 = letterbox_image(crop.stencil_mask, target_size=(224, 224), pad_value=0)

            # High-efficiency thumbnail (max 160px dimension)
            th_h, th_w = crop.crop_bgr.shape[:2]
            scale = 160.0 / max(th_h, th_w)
            tw = max(1, int(round(th_w * scale)))
            th = max(1, int(round(th_h * scale)))
            thumb_bgr = cv2.resize(crop.crop_bgr, (tw, th), interpolation=cv2.INTER_AREA)

            tiles_data.append({
                "page_number": page_idx + 1,
                "tile_index": crop.tile_index,
                "bbox": list(crop.bbox),
                "crop_type": crop.crop_type,
                "design_number": crop.design_number,
                "source_type": "pdf",
                "design_code": crop.design_number,
                "mask_224": m_224,
                "thumb_bgr": thumb_bgr,
            })

        res["tiles"] = tiles_data
    except Exception as exc:
        logger.warning("Error rendering page %d of '%s': %s", page_idx + 1, p.name, exc)
        res["error"] = str(exc)

    return res


def process_dxf_visual(dxf_path: str) -> Dict[str, Any]:
    """
    Parses a DXF CAD drawing, explodes panels, renders clean stencil masks and previews.
    Picklable worker function safe for ProcessPoolExecutor.
    """
    p = Path(dxf_path)
    res: Dict[str, Any] = {
        "dxf_path": str(p.resolve()),
        "tiles": [],
        "error": None,
    }

    try:
        from dxf_ingest import process_dxf_file
        designs = process_dxf_file(p, px_height=800)
    except Exception as exc:
        logger.warning("dxf_ingest failed on '%s': %s. Trying fallback renderer.", p.name, exc)
        designs = []

    if not designs:
        try:
            from app.core.dxf_render import render_dxf_to_stencil
            from dxf_ingest import Design
            stencil, _ = render_dxf_to_stencil(p, target_size=800)
            d_fb = Design(
                design_id="design_1",
                design_code=p.stem,
                bbox=(0.0, 0.0, float(stencil.shape[1]), float(stencil.shape[0])),
                mask=stencil,
                mask_inverted=cv2.bitwise_not(stencil),
                index=1,
                total_designs=1,
            )
            designs = [d_fb]
        except Exception as fb_exc:
            logger.error("Failed to render fallback DXF '%s': %s", p.name, fb_exc)
            res["error"] = str(fb_exc)
            return res

    tiles_data = []
    for d in designs:
        variants = [
            (d.mask, 0),
            (d.mask_inverted, 1),
        ]
        for m_arr, pol_var in variants:
            if m_arr is None or m_arr.size == 0:
                continue

            m_224 = letterbox_image(m_arr, target_size=(224, 224), pad_value=0)

            # Preview thumbnail: brushed silver on navy background
            th_h, th_w = m_arr.shape[:2]
            scale = 160.0 / max(th_h, th_w)
            tw = max(1, int(round(th_w * scale)))
            th = max(1, int(round(th_h * scale)))
            p_thumb = np.full((th, tw, 3), (25, 20, 15), dtype=np.uint8)
            r_m = cv2.resize(m_arr, (tw, th), interpolation=cv2.INTER_AREA)
            p_thumb[r_m > 127] = (218, 218, 222)

            d_code = d.design_code or p.stem
            d_num = d.design_code or f"{p.stem}_P{d.index}"
            idx_info = f"design {d.index} of {d.total_designs}"

            tiles_data.append({
                "page_number": d.index,
                "tile_index": d.index if pol_var == 0 else d.index + 1000,
                "bbox": [int(d.x), int(d.y), max(1, int(round(d.width))), max(1, int(round(d.height)))],
                "crop_type": "dxf_panel" if d.total_designs > 1 else "dxf",
                "design_number": d_num,
                "source_type": "dxf",
                "design_code": d_code,
                "dxf_bbox": [float(d.x), float(d.y), float(d.width), float(d.height)],
                "polarity_variant": pol_var,
                "entity_count": d.entity_count,
                "design_index_info": idx_info,
                "mask_224": m_224,
                "thumb_bgr": p_thumb,
            })

    res["tiles"] = tiles_data
    return res
