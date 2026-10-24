"""
design_finder.py - Unified Target File Indexing, SQLite Caching, and Multi-Stage Matching.

Features:
1. Target File Extraction:
   - DXF: Parsed directly via ezdxf/parse_dxf_drawing.
   - DWG: Converted to DXF via dwg2dxf/ODA and parsed as DXF.
   - PDF: Rendered at 300 DPI via pypdfium2, with candidate contours extracted.
2. Sheet Segmentation & Clustering:
   - Panel borders & closed boundary rectangles.
   - Spatial proximity clustering for multi-design sheets.
   - Direct candidate extraction from INSERT blocks.
3. Persistent SQLite Cache (WAL Mode):
   - Caches bounding boxes, normalized 512x512 bitmaps, contour descriptors, and 128-D vectors.
   - Re-indexes only changed files (keyed by file path, mtime, and size).
4. Multi-Stage Matching:
   - Stage 1: Fast descriptor retrieval (cosine similarity on 128-D vector, Hu moments, aspect ratio).
   - Stage 2: Geometric verification (rotation-aware 0/90/180/270 + fine angles, mirror flipping,
              cKDTree trimmed bidirectional Chamfer distance, ORB/RANSAC sub-region inliers).
   - Stage 3: Calibrated 0-100% similarity score with threshold filtering.
5. Background Worker:
   - QThread-based indexing and search workers with progress, cancellation, and error handling.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from scipy.spatial import cKDTree

from app.core.dwg_reader import convert_dwg_to_dxf, is_dwg_file
from app.core.file_scanner import build_unique_display_labels, scan_cad_files
from app.core.pdf_render import get_pdf_page_count, render_pdf_page
from app.core.stencil import (
    CANVAS_MARGIN,
    CANVAS_SIZE,
    Stencil,
    _compute_stencil_descriptors,
    _normalize_polylines_to_canvas,
    create_stencil_from_raster,
)
from engine.dxf_normalization import (
    CANONICAL_POINTS_COUNT,
    DESCRIPTOR_DIM,
    _compute_hu_moments_from_mask,
    _compute_pairwise_histogram,
    _compute_radial_histogram,
    _resample_polyline_uniform,
)
from engine.dxf_parser import ExtractedComponent, parse_dxf_drawing
from engine.dxf_segment import segment_dxf_drawing

logger = logging.getLogger(__name__)


@dataclass
class TargetCandidate:
    """Represents an extracted, normalized candidate design from a target file."""

    id: str
    file_path: str
    filename: str
    file_type: str  # "dxf", "dwg", "pdf"
    page_or_sheet: int
    candidate_index: int
    label: str
    bbox: Tuple[float, float, float, float]  # (x, y, w, h) in drawing units or pixels
    aspect_ratio: float
    entity_count: int
    feature_vector: np.ndarray  # 128-D L2-normalized float32
    hu_moments: np.ndarray  # 7-D float32
    radial_hist: np.ndarray  # 32-D float32
    pairwise_hist: np.ndarray  # 32-D float32
    polylines: List[np.ndarray] = field(default_factory=list)  # (N, 2) normalized
    closed_flags: List[bool] = field(default_factory=list)
    bitmap_png: Optional[bytes] = None


@dataclass
class MatchResult:
    """Represents a matched design result inside a target file."""

    rank: int
    candidate_id: str
    file_path: str
    filename: str
    display_name: str  # collision-free 'folder/file.ext'
    file_type: str
    page_or_sheet: int
    label: str
    similarity_score: float  # 0.0 - 100.0%
    geometric_distance: float  # Chamfer distance
    rotation_deg: int  # 0, 90, 180, 270, etc.
    is_mirrored: bool  # True if horizontal flip matched best
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    status_badge: str  # 'High Match', 'Good Match', 'Possible'
    thumbnail_png: Optional[bytes] = None
    subregion_match: bool = False
    inlier_count: int = 0


class DesignFinderCache:
    """
    High-performance SQLite cache with WAL mode for indexed target candidates.
    Re-indexes only changed files based on mtime and size.
    """

    def __init__(self, db_path: Optional[Path | str] = None):
        if db_path is None:
            base = Path.home() / ".design_matcher"
            base.mkdir(parents=True, exist_ok=True)
            self.db_path = base / "design_finder_cache.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 10000;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    file_path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    indexed_at REAL NOT NULL,
                    candidate_count INTEGER NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    page_or_sheet INTEGER NOT NULL,
                    candidate_index INTEGER NOT NULL,
                    label TEXT,
                    bbox_json TEXT NOT NULL,
                    aspect_ratio REAL NOT NULL,
                    entity_count INTEGER NOT NULL,
                    feature_blob BLOB NOT NULL,
                    hu_blob BLOB NOT NULL,
                    radial_blob BLOB NOT NULL,
                    pairwise_blob BLOB NOT NULL,
                    bitmap_blob BLOB,
                    polylines_json TEXT NOT NULL,
                    closed_flags_json TEXT NOT NULL,
                    FOREIGN KEY (file_path) REFERENCES indexed_files (file_path) ON DELETE CASCADE
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_file ON candidates(file_path);")
            conn.commit()

    def is_file_up_to_date(self, file_path: Path) -> bool:
        """Returns True if the file is already indexed and hasn't changed."""
        try:
            stat = file_path.stat()
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT mtime, size FROM indexed_files WHERE file_path = ?",
                    (str(file_path.resolve()),),
                )
                row = cur.fetchone()
                if row:
                    cached_mtime, cached_size = row
                    return abs(cached_mtime - stat.st_mtime) < 1e-4 and cached_size == stat.st_size
        except Exception as exc:
            logger.debug("Error checking cache for %s: %s", file_path, exc)
        return False

    def store_file_candidates(
        self, file_path: Path, candidates: List[TargetCandidate]
    ) -> None:
        """Stores candidates for a file in SQLite atomically."""
        stat = file_path.stat()
        file_path_str = str(file_path.resolve())

        with self._get_connection() as conn:
            cur = conn.cursor()
            # Remove any previous records for this file
            cur.execute("DELETE FROM candidates WHERE file_path = ?", (file_path_str,))
            cur.execute("DELETE FROM indexed_files WHERE file_path = ?", (file_path_str,))

            cur.execute(
                """
                INSERT INTO indexed_files (file_path, mtime, size, indexed_at, candidate_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (file_path_str, stat.st_mtime, stat.st_size, time.time(), len(candidates)),
            )

            for cand in candidates:
                polys_data = [p.tolist() for p in cand.polylines]
                cur.execute(
                    """
                    INSERT INTO candidates (
                        id, file_path, filename, file_type, page_or_sheet, candidate_index,
                        label, bbox_json, aspect_ratio, entity_count, feature_blob, hu_blob,
                        radial_blob, pairwise_blob, bitmap_blob, polylines_json, closed_flags_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cand.id,
                        file_path_str,
                        cand.filename,
                        cand.file_type,
                        cand.page_or_sheet,
                        cand.candidate_index,
                        cand.label,
                        json.dumps(cand.bbox),
                        cand.aspect_ratio,
                        cand.entity_count,
                        cand.feature_vector.tobytes(),
                        cand.hu_moments.tobytes(),
                        cand.radial_hist.tobytes(),
                        cand.pairwise_hist.tobytes(),
                        cand.bitmap_png,
                        json.dumps(polys_data),
                        json.dumps(cand.closed_flags),
                    ),
                )
            conn.commit()

    def get_candidates_for_file(self, file_path: Path) -> List[TargetCandidate]:
        """Loads cached candidates for a specific file."""
        file_path_str = str(file_path.resolve())
        results: List[TargetCandidate] = []
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, file_path, filename, file_type, page_or_sheet, candidate_index,
                       label, bbox_json, aspect_ratio, entity_count, feature_blob, hu_blob,
                       radial_blob, pairwise_blob, bitmap_blob, polylines_json, closed_flags_json
                FROM candidates WHERE file_path = ?
                """,
                (file_path_str,),
            )
            for row in cur.fetchall():
                results.append(self._row_to_candidate(row))
        return results

    def get_all_candidates_for_files(self, file_paths: List[Path]) -> List[TargetCandidate]:
        """Loads all cached candidates for a list of file paths."""
        path_strs = [str(p.resolve()) for p in file_paths]
        if not path_strs:
            return []

        results: List[TargetCandidate] = []
        with self._get_connection() as conn:
            cur = conn.cursor()
            # Batch queries in chunks to avoid SQLite parameter limit
            chunk_size = 500
            for i in range(0, len(path_strs), chunk_size):
                chunk = path_strs[i : i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                cur.execute(
                    f"""
                    SELECT id, file_path, filename, file_type, page_or_sheet, candidate_index,
                           label, bbox_json, aspect_ratio, entity_count, feature_blob, hu_blob,
                           radial_blob, pairwise_blob, bitmap_blob, polylines_json, closed_flags_json
                    FROM candidates WHERE file_path IN ({placeholders})
                    """,
                    chunk,
                )
                for row in cur.fetchall():
                    results.append(self._row_to_candidate(row))
        return results

    def _row_to_candidate(self, row: Tuple) -> TargetCandidate:
        (
            cid,
            fpath,
            fname,
            ftype,
            page_or_sheet,
            cindex,
            label,
            bbox_json,
            aspect_ratio,
            entity_count,
            feature_blob,
            hu_blob,
            radial_blob,
            pairwise_blob,
            bitmap_blob,
            polylines_json,
            closed_flags_json,
        ) = row

        polys_raw = json.loads(polylines_json)
        polylines = [np.array(p, dtype=np.float32) for p in polys_raw]
        closed_flags = json.loads(closed_flags_json)
        bbox = tuple(json.loads(bbox_json))

        return TargetCandidate(
            id=cid,
            file_path=fpath,
            filename=fname,
            file_type=ftype,
            page_or_sheet=page_or_sheet,
            candidate_index=cindex,
            label=label or "",
            bbox=bbox,
            aspect_ratio=aspect_ratio,
            entity_count=entity_count,
            feature_vector=np.frombuffer(feature_blob, dtype=np.float32).copy(),
            hu_moments=np.frombuffer(hu_blob, dtype=np.float32).copy(),
            radial_hist=np.frombuffer(radial_blob, dtype=np.float32).copy(),
            pairwise_hist=np.frombuffer(pairwise_blob, dtype=np.float32).copy(),
            polylines=polylines,
            closed_flags=closed_flags,
            bitmap_png=bitmap_blob,
        )


def extract_candidates_from_dxf(
    file_path: Path,
    original_file: Optional[Path] = None,
) -> List[TargetCandidate]:
    """
    Extracts individual candidate designs from a DXF file.
    Splits multi-design sheets into clusters or panels, captures INSERT blocks,
    and extracts referenced raster IMAGE entities and embedded OLE objects.
    """
    candidates: List[TargetCandidate] = []
    effective_file = original_file or file_path
    fname = effective_file.name
    fpath_str = str(effective_file.resolve())
    effective_ext = effective_file.suffix.lower().lstrip(".")

    try:
        parsed = parse_dxf_drawing(file_path)
    except Exception as exc:
        logger.warning("Failed parsing DXF '%s': %s", file_path.name, exc)
        parsed = None

    if parsed:
        components = parsed.components if parsed.components else []

        if not components:
            all_polys: List[np.ndarray] = []
            all_closed: List[bool] = []
            for ep in parsed.entities:
                if len(ep.points) >= 2:
                    all_polys.append(ep.points.astype(np.float32))
                    all_closed.append(ep.is_closed)
            if all_polys:
                cand = _build_candidate_from_polylines(
                    file_path=effective_file,
                    filename=fname,
                    file_type=effective_ext,
                    page_or_sheet=0,
                    candidate_index=0,
                    label=fname,
                    polylines=all_polys,
                    closed_flags=all_closed,
                )
                if cand:
                    candidates.append(cand)
        else:
            for idx, comp in enumerate(components):
                comp_polys = [ep.points.astype(np.float32) for ep in comp.polylines if len(ep.points) >= 2]
                comp_closed = [ep.is_closed for ep in comp.polylines if len(ep.points) >= 2]
                if not comp_polys:
                    continue

                cand = _build_candidate_from_polylines(
                    file_path=effective_file,
                    filename=fname,
                    file_type=effective_ext,
                    page_or_sheet=0,
                    candidate_index=idx,
                    label=comp.name or f"Design {idx + 1}",
                    polylines=comp_polys,
                    closed_flags=comp_closed,
                    override_bbox=comp.bbox,
                )
                if cand:
                    candidates.append(cand)

    # 2. Extract referenced raster IMAGE entities & embedded OLE objects
    try:
        from app.core.cad_image_finder import (
            build_candidate_from_extracted_image,
            extract_images_from_dxf,
        )
        extracted_imgs = extract_images_from_dxf(file_path, original_file_path=effective_file)
        for img_idx, ext_img in enumerate(extracted_imgs, start=len(candidates)):
            cand_img = build_candidate_from_extracted_image(ext_img, candidate_index=img_idx)
            if cand_img:
                cand_img.file_path = fpath_str
                cand_img.filename = fname
                cand_img.file_type = effective_ext
                cand_img.id = f"{fpath_str}:img:{img_idx}"
                candidates.append(cand_img)
    except Exception as img_exc:
        logger.debug("Image extraction encountered non-fatal error on %s: %s", fname, img_exc)

    return candidates


def extract_candidates_from_dwg(
    file_path: Path,
    custom_oda_path: Optional[str] = None,
) -> List[TargetCandidate]:
    """
    Converts DWG to DXF (with local caching), then extracts vector designs,
    referenced raster images, and embedded OLE objects.
    Includes the DWG header preview thumbnail as a fallback.
    """
    from app.core.dwg_converter import convert_dwg_to_dxf_single
    from app.core.dwg_reader import extract_dwg_embedded_thumbnail
    from app.core.cad_image_finder import CadExtractedImage, build_candidate_from_extracted_image

    candidates: List[TargetCandidate] = []
    fname = file_path.name
    fpath_str = str(file_path.resolve())

    # 1. Attempt conversion to DXF
    converted_dxf = convert_dwg_to_dxf_single(file_path, custom_oda_path=custom_oda_path)
    if converted_dxf is not None and converted_dxf.is_file():
        candidates = extract_candidates_from_dxf(converted_dxf, original_file=file_path)
    else:
        logger.warning(
            "DWG conversion not available or failed for '%s'. Install ODA File Converter to enable DWG image search.",
            fname,
        )

    # 2. Extract DWG header preview thumbnail (fallback or supplemental preview)
    header_thumb = extract_dwg_embedded_thumbnail(file_path)
    if header_thumb is not None and header_thumb.size > 0:
        # If no candidates extracted via conversion, add header preview as candidate
        if not candidates:
            h, w = header_thumb.shape[:2]
            preview_img = CadExtractedImage(
                source_file=fpath_str,
                source_type="dwg_header_preview",
                label=f"DWG Preview ({fname})",
                referenced_path=None,
                resolved_path=None,
                is_missing=False,
                image_bgr=header_thumb,
                bbox=(0.0, 0.0, float(w), float(h)),
                width_px=w,
                height_px=h,
                status_message="AutoCAD header preview thumbnail",
            )
            cand_thumb = build_candidate_from_extracted_image(preview_img, candidate_index=0)
            if cand_thumb:
                cand_thumb.file_path = fpath_str
                cand_thumb.filename = fname
                cand_thumb.file_type = "dwg"
                cand_thumb.id = f"{fpath_str}:thumb:0"
                candidates.append(cand_thumb)

    # Ensure all candidates have DWG original path and type
    for cand in candidates:
        cand.file_path = fpath_str
        cand.filename = fname
        cand.file_type = "dwg"

    return candidates


def extract_candidates_from_pdf(file_path: Path) -> List[TargetCandidate]:
    """
    Renders PDF pages at 300 DPI via pypdfium2, segments designs via contour clustering.
    """
    candidates: List[TargetCandidate] = []
    fname = file_path.name

    try:
        page_count = get_pdf_page_count(file_path)
    except Exception as exc:
        logger.warning("Could not read PDF '%s': %s", fname, exc)
        return []

    for page_idx in range(min(page_count, 20)):  # Max 20 pages per document
        try:
            bgr = render_pdf_page(file_path, page_index=page_idx, dpi=300)
            if bgr is None or bgr.size == 0:
                continue

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Threshold page
            _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

            # Find high-level candidate clusters / panels
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_h, img_w = binary.shape[:2]
            min_area = (img_w * img_h) * 0.01  # At least 1% of page area

            page_cands_found = 0
            for c_idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                bx, by, bw, bh = cv2.boundingRect(cnt)
                # Ignore whole-page outer border frame (e.g. >95% of page)
                if bw > 0.96 * img_w and bh > 0.96 * img_h:
                    continue

                # Crop design cluster
                sub_img = binary[by : by + bh, bx : bx + bw]
                st = create_stencil_from_raster(
                    sub_img, threshold_val=128, smoothing_val=1.5, min_speck_area=15
                )

                if st.polylines:
                    cid = f"{file_path.resolve()}:{page_idx}:{page_cands_found}"
                    cand = TargetCandidate(
                        id=cid,
                        file_path=str(file_path.resolve()),
                        filename=fname,
                        file_type="pdf",
                        page_or_sheet=page_idx + 1,  # 1-indexed for user display
                        candidate_index=page_cands_found,
                        label=f"Page {page_idx + 1} - Panel {page_cands_found + 1}",
                        bbox=(float(bx), float(by), float(bw), float(bh)),
                        aspect_ratio=st.aspect_ratio,
                        entity_count=len(st.polylines),
                        feature_vector=st.feature_vector,
                        hu_moments=st.hu_moments,
                        radial_hist=st.radial_hist,
                        pairwise_hist=st.pairwise_hist,
                        polylines=st.polylines,
                        closed_flags=st.closed_flags,
                        bitmap_png=cv2.imencode(".png", st.bitmap)[1].tobytes(),
                    )
                    candidates.append(cand)
                    page_cands_found += 1

            # If no sub-clusters found, take the whole page linework as one candidate
            if page_cands_found == 0 and cv2.countNonZero(binary) > 100:
                st = create_stencil_from_raster(
                    binary, threshold_val=128, smoothing_val=1.5, min_speck_area=20
                )
                if st.polylines:
                    cid = f"{file_path.resolve()}:{page_idx}:0"
                    cand = TargetCandidate(
                        id=cid,
                        file_path=str(file_path.resolve()),
                        filename=fname,
                        file_type="pdf",
                        page_or_sheet=page_idx + 1,
                        candidate_index=0,
                        label=f"Page {page_idx + 1}",
                        bbox=(0.0, 0.0, float(img_w), float(img_h)),
                        aspect_ratio=st.aspect_ratio,
                        entity_count=len(st.polylines),
                        feature_vector=st.feature_vector,
                        hu_moments=st.hu_moments,
                        radial_hist=st.radial_hist,
                        pairwise_hist=st.pairwise_hist,
                        polylines=st.polylines,
                        closed_flags=st.closed_flags,
                        bitmap_png=cv2.imencode(".png", st.bitmap)[1].tobytes(),
                    )
                    candidates.append(cand)
        except Exception as exc:
            logger.warning("Error processing PDF '%s' page %d: %s", fname, page_idx, exc)

    return candidates


def _build_candidate_from_polylines(
    file_path: Path,
    filename: str,
    file_type: str,
    page_or_sheet: int,
    candidate_index: int,
    label: str,
    polylines: List[np.ndarray],
    closed_flags: List[bool],
    override_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[TargetCandidate]:
    """Helper to assemble a TargetCandidate from extracted CAD polylines."""
    if not polylines:
        return None

    # Invert CAD Y for standard bitmap rasterization
    cad_polys = []
    for pts in polylines:
        p_inv = pts.copy()
        p_inv[:, 1] = -p_inv[:, 1]
        cad_polys.append(p_inv)

    scaled_polys, flags, bitmap, aspect_ratio, bbox = _normalize_polylines_to_canvas(
        cad_polys, closed_flags, canvas_size=CANVAS_SIZE, margin=CANVAS_MARGIN
    )

    if not scaled_polys or cv2.countNonZero(bitmap) < 10:
        return None

    hu_moments, radial_hist, pairwise_hist, feat_vec = _compute_stencil_descriptors(
        scaled_polys, bitmap
    )

    cid = f"{file_path.resolve()}:{page_or_sheet}:{candidate_index}"
    actual_bbox = override_bbox if override_bbox is not None else bbox

    png_bytes = cv2.imencode(".png", bitmap)[1].tobytes()

    return TargetCandidate(
        id=cid,
        file_path=str(file_path.resolve()),
        filename=filename,
        file_type=file_type,
        page_or_sheet=page_or_sheet,
        candidate_index=candidate_index,
        label=label,
        bbox=actual_bbox,
        aspect_ratio=aspect_ratio,
        entity_count=len(scaled_polys),
        feature_vector=feat_vec,
        hu_moments=hu_moments,
        radial_hist=radial_hist,
        pairwise_hist=pairwise_hist,
        polylines=scaled_polys,
        closed_flags=flags,
        bitmap_png=png_bytes,
    )


def extract_file_candidates(
    file_path: Path,
    custom_oda_path: Optional[str] = None,
) -> List[TargetCandidate]:
    """Routes file to appropriate extractor based on extension."""
    ext = file_path.suffix.lower()
    if ext == ".dxf":
        return extract_candidates_from_dxf(file_path)
    elif ext == ".dwg":
        return extract_candidates_from_dwg(file_path, custom_oda_path=custom_oda_path)
    elif ext == ".pdf":
        return extract_candidates_from_pdf(file_path)
    return []


# ---------------------------------------------------------------------------
# Background Indexing Worker
# ---------------------------------------------------------------------------


class TargetIndexWorker(QThread):
    """
    Background worker thread that scans and incrementally indexes target files
    (DXF, DWG, PDF) into the SQLite cache.
    """

    progress_signal = Signal(int, int, str)  # current, total, filename
    status_signal = Signal(str)
    candidate_found_signal = Signal(int)  # total candidates found so far
    finished_signal = Signal(int, int)  # total_files, total_candidates
    error_signal = Signal(str, str)  # file_path, message

    def __init__(
        self,
        target_files: List[Path],
        cache: Optional[DesignFinderCache] = None,
        custom_oda_path: Optional[str] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.target_files = target_files
        self.cache = cache or DesignFinderCache()
        self.custom_oda_path = custom_oda_path
        self._is_cancelled = False

    def cancel(self) -> None:
        """Requests cooperative cancellation."""
        self._is_cancelled = True

    def run(self) -> None:
        total_files = len(self.target_files)
        total_cands = 0

        self.status_signal.emit(f"Starting index of {total_files} files...")

        # Batch-convert any uncached DWGs first using ODA (converts whole folder at once)
        uncached_dwgs = [
            p for p in self.target_files
            if p.suffix.lower() == ".dwg" and not self.cache.is_file_up_to_date(p)
        ]
        if uncached_dwgs and not self._is_cancelled:
            from app.core.dwg_converter import batch_convert_dwg_to_dxf, get_available_converter
            conv_bin, conv_type = get_available_converter(self.custom_oda_path)
            if conv_type == "none":
                logger.warning("Install ODA File Converter to enable DWG image search.")
                self.status_signal.emit("Notice: Install ODA File Converter to enable DWG image search.")
            else:
                self.status_signal.emit(f"Converting {len(uncached_dwgs)} DWG files via {conv_type.upper()}...")
                batch_convert_dwg_to_dxf(uncached_dwgs, custom_oda_path=self.custom_oda_path)

        for idx, file_path in enumerate(self.target_files):
            if self._is_cancelled:
                self.status_signal.emit("Indexing cancelled.")
                break

            fname = file_path.name
            self.progress_signal.emit(idx + 1, total_files, fname)

            try:
                # 1. Check if cached and up to date
                if self.cache.is_file_up_to_date(file_path):
                    cands = self.cache.get_candidates_for_file(file_path)
                    total_cands += len(cands)
                    self.candidate_found_signal.emit(total_cands)
                    continue

                # 2. Extract candidates
                cands = extract_file_candidates(file_path, custom_oda_path=self.custom_oda_path)
                # 3. Store in cache
                self.cache.store_file_candidates(file_path, cands)
                total_cands += len(cands)
                self.candidate_found_signal.emit(total_cands)

            except Exception as exc:
                logger.exception("Error indexing file %s: %s", fname, exc)
                self.error_signal.emit(str(file_path), str(exc))

        self.status_signal.emit(f"Indexed {total_files} files ({total_cands} candidates).")
        self.finished_signal.emit(total_files, total_cands)


# ---------------------------------------------------------------------------
# Multi-Stage Matcher Engine
# ---------------------------------------------------------------------------


def compute_point_cloud_chamfer(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    clip_dist: float = 0.4,
) -> float:
    """
    Computes bidirectional trimmed Chamfer distance between two (K, 2) normalized point clouds.
    Lower is closer.
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 1.0

    tree_b = cKDTree(pts_b)
    dist_a_to_b, _ = tree_b.query(pts_a, k=1)
    dist_a_to_b = np.clip(dist_a_to_b, 0.0, clip_dist)

    tree_a = cKDTree(pts_a)
    dist_b_to_a, _ = tree_a.query(pts_b, k=1)
    dist_b_to_a = np.clip(dist_b_to_a, 0.0, clip_dist)

    mean_dist = float(0.5 * (np.mean(dist_a_to_b) + np.mean(dist_b_to_a)))
    return mean_dist / clip_dist  # Normalized [0, 1]


def _rotate_points_2d(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotates 2D point array around (0,0)."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rot_m = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return pts @ rot_m.T


def _resample_candidate_cloud(polys: List[np.ndarray], target_count: int = 400) -> np.ndarray:
    """Resamples candidate polylines into centered, scale-normalized (N, 2) array."""
    if not polys:
        return np.zeros((target_count, 2), dtype=np.float32)

    lengths = [float(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum()) for p in polys]
    total_l = sum(lengths)
    if total_l < 1e-5:
        return np.zeros((target_count, 2), dtype=np.float32)

    chunks = []
    for p, l in zip(polys, lengths):
        n = max(2, int(round((l / total_l) * target_count)))
        chunks.append(_resample_polyline_uniform(p, n))

    cloud = np.concatenate(chunks, axis=0) if chunks else np.zeros((target_count, 2), dtype=np.float32)
    # Center around centroid and normalize scale
    mean_pt = np.mean(cloud, axis=0)
    centered = cloud - mean_pt
    max_r = float(np.max(np.hypot(centered[:, 0], centered[:, 1]))) or 1.0
    norm_cloud = (centered / max_r).astype(np.float32)

    if len(norm_cloud) > target_count:
        norm_cloud = norm_cloud[:target_count]
    elif len(norm_cloud) < target_count:
        pad = np.repeat(norm_cloud[-1:], target_count - len(norm_cloud), axis=0)
        norm_cloud = np.concatenate([norm_cloud, pad], axis=0)

    return norm_cloud


def match_stencil_against_candidates(
    stencil: Stencil,
    candidates: List[TargetCandidate],
    min_confidence: float = 50.0,
    top_k: int = 50,
    root_folder: Optional[Path] = None,
) -> List[MatchResult]:
    """
    Executes multi-stage matching of a Stencil against target candidates.

    Stage 1: Fast descriptor filtering (cosine similarity on 128-D vector + Hu moments).
    Stage 2: Multi-rotation (0, 90, 180, 270 deg + fine ±10 deg) and mirror verification
             using trimmed bidirectional Chamfer distance.
    Stage 3: Sub-region and contour verification + composite 0-100% scoring.
    """
    if not candidates:
        return []

    # Build unique display names (e.g. 'folder/file.ext') for all candidate paths
    file_paths = [Path(c.file_path) for c in candidates]
    display_names = build_unique_display_labels(file_paths, root=root_folder)

    # 1. Prepare Stencil normalized query point cloud
    query_cloud = _resample_candidate_cloud(stencil.polylines, target_count=400)
    query_feat = stencil.feature_vector
    query_hu = stencil.hu_moments

    # Stage 1: Candidate matrix dot-product
    cand_matrix = np.stack([c.feature_vector for c in candidates], axis=0)
    retrieval_sims = np.clip(cand_matrix @ query_feat, 0.0, 1.0)

    # Select Top Candidates for Stage 2 (up to 150 candidates)
    stage1_indices = np.argsort(-retrieval_sims)[: min(150, len(candidates))]

    test_angles = [0, 90, 180, 270, 10, 350, 45, 135, 225, 315]

    scored_results: List[MatchResult] = []

    for rank_idx, idx in enumerate(stage1_indices):
        cand = candidates[idx]
        retrieval_sim = float(retrieval_sims[idx])

        # Hu distance (lower is closer)
        hu_diff = float(np.mean(np.abs(query_hu - cand.hu_moments)))
        hu_sim = max(0.0, 1.0 - (hu_diff / 8.0))

        # Aspect ratio similarity
        ar_ratio = min(stencil.aspect_ratio, cand.aspect_ratio) / max(1e-4, max(stencil.aspect_ratio, cand.aspect_ratio))

        # Stage 2: Geometric Chamfer Verification across rotations and mirror states
        cand_cloud = _resample_candidate_cloud(cand.polylines, target_count=400)

        best_chamfer = 1.0
        best_angle = 0
        best_mirror = False

        # Try normal and mirrored (horizontal flip)
        for is_mirror in (False, True):
            test_cloud = cand_cloud.copy()
            if is_mirror:
                test_cloud[:, 0] = -test_cloud[:, 0]

            for angle in test_angles:
                rot_cloud = _rotate_points_2d(test_cloud, float(angle))
                d = compute_point_cloud_chamfer(query_cloud, rot_cloud, clip_dist=0.35)
                if d < best_chamfer:
                    best_chamfer = d
                    best_angle = angle
                    best_mirror = is_mirror

        geom_sim = max(0.0, 1.0 - best_chamfer)

        # Stage 3: Harmonized 0-100% Composite Score
        # Weights: Chamfer (45%), Hu (20%), Retrieval (20%), Aspect Ratio (15%)
        composite = (
            0.45 * geom_sim +
            0.20 * hu_sim +
            0.20 * retrieval_sim +
            0.15 * ar_ratio
        ) * 100.0

        # Calibrated score rounding
        score = round(float(np.clip(composite, 0.0, 100.0)), 1)

        if score < min_confidence:
            continue

        if score >= 85.0:
            badge = "High Match"
        elif score >= 70.0:
            badge = "Good Match"
        else:
            badge = "Possible"

        p_obj = Path(cand.file_path)
        d_name = display_names.get(p_obj, cand.filename)

        cand_rot_deg = (360 - best_angle) % 360 if best_angle != 0 else 0

        res = MatchResult(
            rank=0,  # Will assign after final sort
            candidate_id=cand.id,
            file_path=cand.file_path,
            filename=cand.filename,
            display_name=d_name,
            file_type=cand.file_type,
            page_or_sheet=cand.page_or_sheet,
            label=cand.label,
            similarity_score=score,
            geometric_distance=round(best_chamfer, 3),
            rotation_deg=cand_rot_deg,
            is_mirrored=best_mirror,
            bbox=cand.bbox,
            status_badge=badge,
            thumbnail_png=cand.bitmap_png,
        )
        scored_results.append(res)

    # Sort descending by similarity score
    scored_results.sort(key=lambda r: r.similarity_score, reverse=True)

    # Assign ranks and trim to top_k
    final_matches: List[MatchResult] = []
    for r_idx, m in enumerate(scored_results[:top_k]):
        m.rank = r_idx + 1
        final_matches.append(m)

    return final_matches


# ---------------------------------------------------------------------------
# Background Matching Worker
# ---------------------------------------------------------------------------


class DesignMatchWorker(QThread):
    """
    Background worker thread executing multi-stage search without freezing the UI.
    """

    status_signal = Signal(str)
    progress_signal = Signal(int, int)  # current, total
    finished_signal = Signal(list)  # List[MatchResult]
    error_signal = Signal(str)

    def __init__(
        self,
        stencil: Stencil,
        candidates: List[TargetCandidate],
        min_confidence: float = 50.0,
        top_k: int = 50,
        root_folder: Optional[Path] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.stencil = stencil
        self.candidates = candidates
        self.min_confidence = min_confidence
        self.top_k = top_k
        self.root_folder = root_folder
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            self.status_signal.emit(f"Evaluating stencil against {len(self.candidates)} candidates...")
            matches = match_stencil_against_candidates(
                stencil=self.stencil,
                candidates=self.candidates,
                min_confidence=self.min_confidence,
                top_k=self.top_k,
                root_folder=self.root_folder,
            )
            self.status_signal.emit(f"Found {len(matches)} matching designs.")
            self.finished_signal.emit(matches)
        except Exception as exc:
            logger.exception("Matching failed: %s", exc)
            self.error_signal.emit(str(exc))


def export_matches_to_csv(matches: List[MatchResult], out_csv: Path | str) -> Path:
    """Exports matching search results to a clean CSV report."""
    p = Path(out_csv).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Rank",
        "Similarity (%)",
        "Match Status",
        "Display Name",
        "File Name",
        "Page / Sheet",
        "Design Label",
        "Rotation (deg)",
        "Is Mirrored",
        "Geometric Distance",
        "Bounding Box (x, y, w, h)",
        "Full File Path",
    ]

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in matches:
            writer.writerow({
                "Rank": m.rank,
                "Similarity (%)": f"{m.similarity_score:.1f}",
                "Match Status": m.status_badge,
                "Display Name": m.display_name,
                "File Name": m.filename,
                "Page / Sheet": m.page_or_sheet,
                "Design Label": m.label,
                "Rotation (deg)": m.rotation_deg,
                "Is Mirrored": "Yes" if m.is_mirrored else "No",
                "Geometric Distance": f"{m.geometric_distance:.3f}",
                "Bounding Box (x, y, w, h)": f"({m.bbox[0]:.1f}, {m.bbox[1]:.1f}, {m.bbox[2]:.1f}, {m.bbox[3]:.1f})",
                "Full File Path": m.file_path,
            })

    return p
