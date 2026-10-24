"""
cad_image_finder.py - Extraction of Raster Images, Referenced IMAGEDEF Files,
Embedded OLE Objects, and Header Thumbnails from DXF and DWG CAD Drawings.

Features:
1. IMAGE entity & IMAGEDEF path inspection with robust relative path resolution
   against the original CAD folder.
2. Graceful detection and reporting of missing raster image references.
3. Embedded OLE object image extraction (BMP, PNG, JPEG binary payloads in OLE2FRAME).
4. Direct binary DWG header preview extraction (at the image seeker offset) as a fast preview/fallback.
5. Integration into the unified TargetCandidate model for stencil matching and UI display.
"""

from __future__ import annotations

import logging
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import ezdxf
import numpy as np

from app.core.dwg_reader import extract_dwg_embedded_thumbnail, is_dwg_file
from app.core.stencil import (
    CANVAS_MARGIN,
    CANVAS_SIZE,
    create_stencil_from_raster,
)
from engine.dxf_normalization import (
    DESCRIPTOR_DIM,
    _compute_hu_moments_from_mask,
    _compute_pairwise_histogram,
    _compute_radial_histogram,
)

logger = logging.getLogger(__name__)


@dataclass
class CadExtractedImage:
    """Represents a raster or OLE image extracted or referenced inside a CAD drawing."""

    source_file: str                     # Original CAD file path (.dwg or .dxf)
    source_type: str                     # 'raster_ref', 'embedded_ole', 'dwg_header_preview'
    label: str                           # Display name or referenced file name
    referenced_path: Optional[str]       # Stored filename in IMAGEDEF (if applicable)
    resolved_path: Optional[str]         # Resolved existing filesystem path
    is_missing: bool                     # True if referenced raster file does not exist on disk
    image_bgr: Optional[np.ndarray]      # Decoded BGR image (uint8) if available
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # (x, y, w, h) in CAD units
    width_px: int = 0
    height_px: int = 0
    status_message: str = ""


def resolve_cad_image_path(raw_path: str, cad_folder: Path) -> Tuple[Optional[Path], bool]:
    """
    Resolves an AutoCAD image reference path against the CAD drawing's parent folder.

    Checks:
    1. cad_folder / raw_path
    2. cad_folder / basename(raw_path)
    3. cad_folder / 'images' / basename(raw_path)
    4. Absolute path directly if present on local filesystem.

    Returns:
        (resolved_path, is_missing)
    """
    if not raw_path or not raw_path.strip():
        return None, True

    # Normalize backslashes (Windows CAD paths) to standard separators
    cleaned_rel = raw_path.replace("\\", "/").strip()
    base_name = Path(cleaned_rel).name

    candidates: List[Path] = [
        cad_folder / cleaned_rel,
        cad_folder / base_name,
        cad_folder / "images" / base_name,
        cad_folder / "img" / base_name,
        cad_folder / "textures" / base_name,
    ]

    # If raw_path is absolute on the current machine
    try:
        abs_p = Path(raw_path)
        if abs_p.is_absolute():
            candidates.insert(0, abs_p)
    except Exception:
        pass

    for cand in candidates:
        try:
            if cand.is_file() and cand.stat().st_size > 0:
                return cand.resolve(), False
        except (OSError, PermissionError):
            continue

    return None, True


def extract_images_from_dxf(
    dxf_path: Union[str, Path],
    original_file_path: Optional[Union[str, Path]] = None,
) -> List[CadExtractedImage]:
    """
    Extracts all raster image references (IMAGE / IMAGEDEF) and embedded OLE objects
    from a DXF file (whether native DXF or DWG-converted DXF).
    """
    dxf_p = Path(dxf_path).resolve()
    orig_p = Path(original_file_path).resolve() if original_file_path else dxf_p
    cad_folder = orig_p.parent

    extracted: List[CadExtractedImage] = []

    if not dxf_p.is_file():
        return extracted

    try:
        try:
            doc = ezdxf.readfile(str(dxf_p))
        except Exception:
            from ezdxf import recover
            doc, _ = recover.readfile(str(dxf_p))
    except Exception as exc:
        logger.warning("Could not read DXF for image extraction '%s': %s", dxf_p.name, exc)
        return extracted

    # 1. Inspect IMAGE entities in modelspace and paperspace layouts
    all_layouts = list(doc.layouts)
    for layout in all_layouts:
        for entity in layout:
            dxftype = entity.dxftype()
            if dxftype == "IMAGE":
                try:
                    # Find IMAGEDEF
                    idef_handle = getattr(entity.dxf, "image_def_handle", None)
                    idef = None
                    if idef_handle and idef_handle in doc.entitydb:
                        idef = doc.entitydb.get(idef_handle)
                    elif hasattr(entity, "image_def") and entity.image_def:
                        idef = entity.image_def

                    raw_filename = getattr(idef.dxf, "filename", "") if idef else ""
                    resolved_file, is_missing = resolve_cad_image_path(raw_filename, cad_folder)

                    # Extract CAD insertion coordinates & scale
                    insert_pt = getattr(entity.dxf, "insert", (0.0, 0.0, 0.0))
                    u_vec = getattr(entity.dxf, "u_pixel", (1.0, 0.0, 0.0))
                    v_vec = getattr(entity.dxf, "v_pixel", (0.0, 1.0, 0.0))
                    size_pixels = getattr(entity.dxf, "image_size", (100.0, 100.0))

                    # Calculate approximate bounding box in CAD drawing units
                    w_cad = abs(float(u_vec[0] * size_pixels[0])) if len(u_vec) > 0 else 100.0
                    h_cad = abs(float(v_vec[1] * size_pixels[1])) if len(v_vec) > 1 else 100.0
                    cad_bbox = (float(insert_pt[0]), float(insert_pt[1]), max(1.0, w_cad), max(1.0, h_cad))

                    loaded_bgr = None
                    w_px, h_px = int(size_pixels[0]), int(size_pixels[1])

                    if resolved_file and not is_missing:
                        try:
                            loaded_bgr = cv2.imread(str(resolved_file), cv2.IMREAD_COLOR)
                            if loaded_bgr is not None:
                                h_px, w_px = loaded_bgr.shape[:2]
                        except Exception as im_err:
                            logger.debug("Failed decoding image %s: %s", resolved_file, im_err)

                    label_name = Path(raw_filename).name if raw_filename else f"Raster Image #{entity.dxf.handle}"
                    status_msg = "Referenced image loaded" if loaded_bgr is not None else (
                        f"Missing referenced image: '{raw_filename}'" if is_missing else "Decode failed"
                    )

                    extracted.append(CadExtractedImage(
                        source_file=str(orig_p),
                        source_type="raster_ref",
                        label=label_name,
                        referenced_path=raw_filename,
                        resolved_path=str(resolved_file) if resolved_file else None,
                        is_missing=is_missing,
                        image_bgr=loaded_bgr,
                        bbox=cad_bbox,
                        width_px=w_px,
                        height_px=h_px,
                        status_message=status_msg,
                    ))
                except Exception as ent_err:
                    logger.debug("Error processing IMAGE entity in %s: %s", dxf_p.name, ent_err)

            # 2. Inspect OLE2FRAME / OLEFRAME entities
            elif dxftype in ("OLE2FRAME", "OLEFRAME"):
                try:
                    bin_data = getattr(entity, "binary_data", None)
                    if bin_data and isinstance(bin_data, (bytes, bytearray)):
                        img_arr = _extract_image_from_ole_stream(bin_data)
                        if img_arr is not None:
                            h_px, w_px = img_arr.shape[:2]
                            extracted.append(CadExtractedImage(
                                source_file=str(orig_p),
                                source_type="embedded_ole",
                                label=f"Embedded OLE Image #{entity.dxf.handle}",
                                referenced_path=None,
                                resolved_path=None,
                                is_missing=False,
                                image_bgr=img_arr,
                                bbox=(0.0, 0.0, float(w_px), float(h_px)),
                                width_px=w_px,
                                height_px=h_px,
                                status_message="Embedded OLE object decoded",
                            ))
                except Exception as ole_err:
                    logger.debug("Error inspecting OLE frame in %s: %s", dxf_p.name, ole_err)

    return extracted


def _extract_image_from_ole_stream(data: bytes) -> Optional[np.ndarray]:
    """Scans binary OLE stream for embedded BMP, PNG, or JPEG image signatures."""
    if len(data) < 32:
        return None

    # Check for PNG header
    png_pos = data.find(b"\x89PNG\r\n\x1a\n")
    if png_pos != -1:
        try:
            arr = cv2.imdecode(np.frombuffer(data[png_pos:], np.uint8), cv2.IMREAD_COLOR)
            if arr is not None and arr.size > 0:
                return arr
        except Exception:
            pass

    # Check for JPEG header
    jpg_pos = data.find(b"\xFF\xD8\xFF")
    if jpg_pos != -1:
        try:
            arr = cv2.imdecode(np.frombuffer(data[jpg_pos:], np.uint8), cv2.IMREAD_COLOR)
            if arr is not None and arr.size > 0:
                return arr
        except Exception:
            pass

    # Check for BMP header
    bm_pos = 0
    limit = min(len(data), 5 * 1024 * 1024)
    while True:
        bm_pos = data.find(b"BM", bm_pos, limit)
        if bm_pos == -1 or bm_pos + 40 > len(data):
            break
        # Verify DIB header size 40 (BITMAPINFOHEADER)
        if data[bm_pos + 14:bm_pos + 18] == b"\x28\x00\x00\x00":
            try:
                arr = cv2.imdecode(np.frombuffer(data[bm_pos:], np.uint8), cv2.IMREAD_COLOR)
                if arr is not None and arr.shape[0] >= 16 and arr.shape[1] >= 16:
                    return arr
            except Exception:
                pass
        bm_pos += 2

    return None


def extract_cad_images_all(
    cad_file_path: Union[str, Path],
    custom_oda_path: Optional[str] = None,
) -> List[CadExtractedImage]:
    """
    Top-level unified image extractor for both DXF and DWG files.
    - If DWG: converts to cached DXF, extracts IMAGE/OLE, and includes header preview thumbnail.
    - If DXF: extracts IMAGE/OLE directly.
    """
    p = Path(cad_file_path).resolve()
    if not p.is_file():
        return []

    images: List[CadExtractedImage] = []
    ext = p.suffix.lower()

    if ext == ".dxf":
        images.extend(extract_images_from_dxf(p, original_file_path=p))

    elif ext == ".dwg":
        # 1. Include DWG Header preview thumbnail as fallback
        header_thumb = extract_dwg_embedded_thumbnail(p)
        if header_thumb is not None and header_thumb.size > 0:
            h_px, w_px = header_thumb.shape[:2]
            images.append(CadExtractedImage(
                source_file=str(p),
                source_type="dwg_header_preview",
                label=f"DWG Header Preview ({p.name})",
                referenced_path=None,
                resolved_path=None,
                is_missing=False,
                image_bgr=header_thumb,
                bbox=(0.0, 0.0, float(w_px), float(h_px)),
                width_px=w_px,
                height_px=h_px,
                status_message="AutoCAD header preview thumbnail",
            ))

        # 2. Convert DWG to cached DXF and extract referenced IMAGE entities and OLE objects
        from app.core.dwg_converter import convert_dwg_to_dxf_single
        converted_dxf = convert_dwg_to_dxf_single(p, custom_oda_path=custom_oda_path)
        if converted_dxf and converted_dxf.is_file():
            dxf_images = extract_images_from_dxf(converted_dxf, original_file_path=p)
            images.extend(dxf_images)
        else:
            logger.info("Conversion not available for '%s'; relying on DWG header preview.", p.name)

    return images


def build_candidate_from_extracted_image(
    extracted_image: CadExtractedImage,
    candidate_index: int,
) -> Optional[Any]:  # Returns TargetCandidate
    """
    Transforms a CadExtractedImage into a TargetCandidate so it integrates
    seamlessly with the Find Design Stencil Matcher and Results UI.
    """
    from app.core.design_finder import TargetCandidate

    orig_p = Path(extracted_image.source_file)
    fpath_str = str(orig_p.resolve())
    fname = orig_p.name
    ftype = orig_p.suffix.lower().lstrip(".")

    # If the image was missing, generate a distinct missing placeholder candidate
    if extracted_image.is_missing or extracted_image.image_bgr is None:
        placeholder_bgr = np.full((256, 256, 3), (30, 41, 59), dtype=np.uint8)
        # Draw red border and text
        cv2.rectangle(placeholder_bgr, (5, 5), (250, 250), (60, 60, 220), 2)
        cv2.putText(placeholder_bgr, "MISSING REF", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 240), 2)
        png_bytes = cv2.imencode(".png", placeholder_bgr)[1].tobytes()

        cid = f"{fpath_str}:img:{candidate_index}"
        dummy_feat = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
        dummy_hu = np.zeros(7, dtype=np.float32)
        dummy_radial = np.zeros(32, dtype=np.float32)
        dummy_pairwise = np.zeros(32, dtype=np.float32)

        return TargetCandidate(
            id=cid,
            file_path=fpath_str,
            filename=fname,
            file_type=ftype,
            page_or_sheet=1,
            candidate_index=candidate_index,
            label=f"[Missing Image] {extracted_image.label}",
            bbox=extracted_image.bbox,
            aspect_ratio=1.0,
            entity_count=1,
            feature_vector=dummy_feat,
            hu_moments=dummy_hu,
            radial_hist=dummy_radial,
            pairwise_hist=dummy_pairwise,
            polylines=[],
            closed_flags=[],
            bitmap_png=png_bytes,
        )

    # Image is loaded: build normalized stencil descriptors
    bgr = extracted_image.image_bgr
    h, w = bgr.shape[:2]
    aspect_ratio = float(w) / float(max(1, h))

    # Convert to grayscale and binary mask
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # Use stencil processor to extract polylines and descriptors
    st = create_stencil_from_raster(binary, threshold_val=128, smoothing_val=1.5, min_speck_area=15)
    png_bytes = cv2.imencode(".png", bgr)[1].tobytes()

    cid = f"{fpath_str}:img:{candidate_index}"
    display_label = extracted_image.label or f"Image {candidate_index + 1}"

    return TargetCandidate(
        id=cid,
        file_path=fpath_str,
        filename=fname,
        file_type=ftype,
        page_or_sheet=1,
        candidate_index=candidate_index,
        label=display_label,
        bbox=extracted_image.bbox,
        aspect_ratio=aspect_ratio,
        entity_count=len(st.polylines) if st.polylines else 1,
        feature_vector=st.feature_vector,
        hu_moments=st.hu_moments,
        radial_hist=st.radial_hist,
        pairwise_hist=st.pairwise_hist,
        polylines=st.polylines,
        closed_flags=st.closed_flags,
        bitmap_png=png_bytes,
    )
