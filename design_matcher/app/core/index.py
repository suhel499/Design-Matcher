"""
Vector indexing engine and catalog indexing pipeline.
Coordinates PDF rendering, tile segmentation, DINOv2 embedding,
FAISS vector storage (IndexIDMap2 + IndexFlatIP), and SQLite metadata persistence.
Supports incremental indexing, thumbnail generation, progress reporting, and cancellation.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import faiss  # type: ignore
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

import numpy as np

from app.core.db import CatalogDatabase
from app.core.embed import EMBEDDING_DIM, DINOv2Embedder
from app.core.pdf_render import (
    PdfRenderError,
    calculate_file_hash,
    get_pdf_page_count,
    render_pdf_page,
)
from app.core.preprocess import letterbox_image
from app.core.tiles import detect_tile_boxes, extract_tile_crops

logger = logging.getLogger(__name__)


@dataclass
class IndexProgress:
    """Progress status report emitted during catalog indexing."""
    phase: str  # 'scanning', 'indexing', 'saving', 'complete', 'cancelled'
    current_file_idx: int = 0
    total_files: int = 0
    current_pdf_path: str = ""
    current_page: int = 0
    total_pages_in_file: int = 0
    total_tiles_indexed: int = 0
    message: str = ""
    percent: float = 0.0


def get_default_data_dir() -> Path:
    """Returns local path for storing the FAISS index, SQLite DB, and thumbnails."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path.home() / ".local" / "share"
    data_dir = base / "design_matcher"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class CatalogIndex:
    """
    Manages vector similarity indexing with FAISS and metadata in SQLite.
    Employs IndexIDMap2(IndexFlatIP) to sync vector IDs 1-to-1 with SQLite tile IDs.
    """

    def __init__(self, data_dir: Optional[Path | str] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else get_default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "catalog_metadata.db"
        self.faiss_path = self.data_dir / "catalog_vectors.faiss"
        self.thumbnails_dir = self.data_dir / "thumbnails"
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

        self.db = CatalogDatabase(self.db_path)
        self.embedder = DINOv2Embedder.get_instance()
        self._index: Optional[faiss.IndexIDMap2] = None
        self._init_faiss_index()

    def _init_faiss_index(self) -> None:
        """Loads an existing FAISS index from disk or initializes a new one."""
        if self.faiss_path.is_file():
            try:
                logger.info("Loading existing FAISS index from %s", self.faiss_path)
                loaded = faiss.read_index(str(self.faiss_path))
                # Ensure index is wrapped in IndexIDMap2 if loaded as standard index
                if not isinstance(loaded, faiss.IndexIDMap2):
                    self._index = faiss.IndexIDMap2(loaded)
                else:
                    self._index = loaded
                logger.info("FAISS index loaded with %d vectors.", self._index.ntotal)
                return
            except Exception as exc:
                logger.warning("Failed to load FAISS index from disk (%s). Recreating new index.", exc)

        # Create new IndexIDMap2 wrapping IndexFlatIP (cosine similarity on L2-normalized vectors)
        self._base_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._index = faiss.IndexIDMap2(self._base_index)
        logger.info("Initialized new FAISS IndexIDMap2 with dimension %d.", EMBEDDING_DIM)

    def save_index(self) -> None:
        """Serializes FAISS vector index to disk."""
        if self._index is not None:
            temp_path = self.faiss_path.with_suffix(".tmp")
            faiss.write_index(self._index, str(temp_path))
            if temp_path.exists():
                temp_path.replace(self.faiss_path)
            logger.info("Saved FAISS index with %d vectors to %s", self._index.ntotal, self.faiss_path)

    @property
    def total_vectors(self) -> int:
        """Returns the number of vectors stored in FAISS."""
        return self._index.ntotal if self._index is not None else 0

    def add_vectors(self, vectors: np.ndarray, ids: List[int]) -> None:
        """
        Adds vectors with corresponding integer IDs to FAISS.
        Vectors must be float32, shape (N, 384), and L2-normalized.
        """
        if len(ids) == 0:
            return
        if self._index is None:
            self._init_faiss_index()

        vectors_np = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_np = np.array(ids, dtype=np.int64)
        self._index.add_with_ids(vectors_np, ids_np)

    def remove_vectors(self, ids: List[int]) -> None:
        """Removes vectors by IDs from FAISS index."""
        if not ids or self._index is None or self._index.ntotal == 0:
            return
        ids_np = np.array(ids, dtype=np.int64)
        try:
            self._index.remove_ids(ids_np)
            logger.debug("Removed %d vectors from FAISS index.", len(ids))
        except Exception as exc:
            logger.warning("Error removing IDs from FAISS index: %s", exc)

    def search_vectors(self, query_vectors: np.ndarray, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches FAISS for closest matching vectors.

        Args:
            query_vectors: Shape (Q, 384) float32 L2-normalized queries.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            scores: Shape (Q, top_k) float32 cosine similarity scores.
            indices: Shape (Q, top_k) int64 tile IDs.
        """
        if self._index is None or self._index.ntotal == 0:
            return np.empty((len(query_vectors), 0), dtype=np.float32), np.empty((len(query_vectors), 0), dtype=np.int64)

        k = min(top_k, self._index.ntotal)
        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
        scores, indices = self._index.search(queries, k)
        return scores, indices

    def search_dxf_vectors(self, query_vectors: np.ndarray, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Directly queries all indexed DXF vectors using exact cosine similarity,
        guaranteeing that DXF candidates are NEVER crowded out by PDF pages in FAISS.
        """
        if self._index is None or self._index.ntotal == 0 or len(query_vectors) == 0:
            return np.empty((len(query_vectors), 0), dtype=np.float32), np.empty((len(query_vectors), 0), dtype=np.int64)

        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM tiles WHERE pdf_path LIKE ?;", ("%.dxf",))
            dxf_ids = [r[0] for r in cur.fetchall()]

        if not dxf_ids:
            return np.empty((len(query_vectors), 0), dtype=np.float32), np.empty((len(query_vectors), 0), dtype=np.int64)

        dxf_vecs = []
        valid_ids = []
        for tid in dxf_ids:
            try:
                v = self._index.reconstruct(int(tid))
                dxf_vecs.append(v)
                valid_ids.append(tid)
            except Exception:
                continue

        if not dxf_vecs:
            return np.empty((len(query_vectors), 0), dtype=np.float32), np.empty((len(query_vectors), 0), dtype=np.int64)

        dxf_mat = np.array(dxf_vecs, dtype=np.float32)
        valid_ids_np = np.array(valid_ids, dtype=np.int64)

        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
        all_scores = queries @ dxf_mat.T

        k = min(top_k, len(valid_ids))
        res_scores = np.empty((len(queries), k), dtype=np.float32)
        res_indices = np.empty((len(queries), k), dtype=np.int64)

        for q_idx in range(len(queries)):
            sort_idx = np.argsort(-all_scores[q_idx])[:k]
            res_scores[q_idx] = all_scores[q_idx, sort_idx]
            res_indices[q_idx] = valid_ids_np[sort_idx]

        return res_scores, res_indices

    def index_catalog(
        self,
        root_dir: Path | str,
        dpi: int = 200,
        include_subregions: bool = True,
        run_ocr: bool = True,
        progress_cb: Optional[Callable[[IndexProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        force_reindex: bool = False,
    ) -> int:
        """
        Recursively indexes all PDF catalogs found under root_dir.
        Performs incremental indexing by hashing files and skipping unchanged ones.

        Args:
            root_dir: Root directory to search for PDFs.
            dpi: Rendering DPI for catalog pages (~200 DPI recommended).
            include_subregions: Whether to generate overlapping sub-region crops.
            run_ocr: Whether to run OCR on design numbers under tiles.
            progress_cb: Optional callback for reporting progress.
            cancel_check: Optional callable returning True if indexing should abort.
            force_reindex: If True, re-indexes even unchanged files.

        Returns:
            Total number of newly indexed tiles.
        """
        root_path = Path(root_dir).resolve()
        if not root_path.is_dir():
            raise NotADirectoryError(f"Catalog directory does not exist: {root_dir}")

        def emit_progress(p: IndexProgress) -> None:
            if progress_cb:
                progress_cb(p)

        def is_cancelled() -> bool:
            return cancel_check() if cancel_check else False

        emit_progress(IndexProgress(phase="scanning", message=f"Scanning '{root_path}' for PDF & DXF files..."))

        # 1. Discover all PDF and DXF catalog files
        catalog_paths: List[Path] = []
        valid_exts = {".pdf", ".dxf"}
        for dirpath, _, filenames in os.walk(root_path):
            for fname in filenames:
                if Path(fname).suffix.lower() in valid_exts:
                    catalog_paths.append(Path(dirpath) / fname)

        catalog_paths.sort()
        total_files = len(catalog_paths)
        logger.info("Found %d catalog files in %s", total_files, root_path)

        if total_files == 0:
            emit_progress(IndexProgress(phase="complete", message="No PDF or DXF files found.", percent=100.0))
            return 0

        # 2. Cleanup records for files removed from disk
        existing_path_strings = {str(p.resolve()) for p in catalog_paths}
        del_paths, del_ids = self.db.cleanup_missing_files(existing_path_strings)
        if del_ids:
            self.remove_vectors(del_ids)
            logger.info("Removed %d deleted files and %d obsolete tile vectors.", len(del_paths), len(del_ids))

        # 3. Process each file (PDF or DXF)
        total_tiles_indexed = 0

        for file_idx, cat_p in enumerate(catalog_paths, start=1):
            if is_cancelled():
                emit_progress(IndexProgress(phase="cancelled", message="Indexing cancelled by user."))
                self.save_index()
                return total_tiles_indexed

            file_str = str(cat_p.resolve())
            folder_rel = str(cat_p.parent.relative_to(root_path)) if cat_p.parent != root_path else "."
            is_dxf = cat_p.suffix.lower() == ".dxf"

            emit_progress(
                IndexProgress(
                    phase="indexing",
                    current_file_idx=file_idx,
                    total_files=total_files,
                    current_pdf_path=cat_p.name,
                    percent=float((file_idx - 1) / total_files * 100.0),
                    message=f"Checking {cat_p.name} ({file_idx}/{total_files})...",
                )
            )

            # Check if unchanged
            try:
                current_hash = calculate_file_hash(cat_p)
                file_size = cat_p.stat().st_size
                last_mod = cat_p.stat().st_mtime
            except Exception as exc:
                logger.warning("Could not access file '%s': %s", cat_p, exc)
                continue

            existing_record = self.db.get_indexed_file(file_str)
            if existing_record and not force_reindex:
                if existing_record["file_hash"] == current_hash:
                    logger.debug("Skipping unchanged file: %s", cat_p.name)
                    continue
                else:
                    logger.info("File changed on disk, re-indexing: %s", cat_p.name)
                    old_ids = self.db.delete_file_records(file_str)
                    self.remove_vectors(old_ids)

            if is_dxf:
                # Process DXF CAD file using dxf_ingest pipeline
                from dxf_ingest import process_dxf_file
                from app.core.dxf_render import render_dxf_preview

                try:
                    designs = process_dxf_file(cat_p, px_height=800)
                except Exception as exc:
                    logger.error("dxf_ingest failed on '%s': %s. Using fallback renderer.", cat_p.name, exc)
                    designs = []

                if not designs:
                    # Fallback to single sheet rendering if dxf_ingest produced no designs
                    try:
                        from app.core.dxf_render import render_dxf_to_stencil
                        stencil, _ = render_dxf_to_stencil(cat_p, target_size=800)
                        from dxf_ingest import Design
                        d_fb = Design(
                            design_id="design_1",
                            design_code=cat_p.stem,
                            bbox=(0.0, 0.0, float(stencil.shape[1]), float(stencil.shape[0])),
                            mask=stencil,
                            mask_inverted=cv2.bitwise_not(stencil),
                            index=1,
                            total_designs=1,
                        )
                        designs = [d_fb]
                    except Exception as fb_exc:
                        logger.error("Failed to render fallback DXF '%s': %s", cat_p.name, fb_exc)
                        continue

                page_count = len(designs)
                self.db.record_indexed_file(
                    pdf_path=file_str,
                    folder=folder_rel,
                    file_hash=current_hash,
                    file_size=file_size,
                    last_modified=last_mod,
                    page_count=page_count,
                    tile_count=0,
                )

                tiles_to_insert = []
                masks_to_embed = []

                for d in designs:
                    # Index both normal and inverted polarity variants
                    variants = [
                        (d.mask, 0),
                        (d.mask_inverted, 1),
                    ]
                    for m_arr, pol_var in variants:
                        if m_arr is None or m_arr.size == 0:
                            continue
                        m_224 = letterbox_image(m_arr, target_size=(224, 224), pad_value=0)

                        # Preview thumbnail (brushed silver on navy background)
                        th_h, th_w = m_arr.shape[:2]
                        s_dim = 160.0 / max(th_h, th_w)
                        tw = max(1, int(round(th_w * s_dim)))
                        th = max(1, int(round(th_h * s_dim)))
                        p_thumb = np.full((th, tw, 3), (25, 20, 15), dtype=np.uint8)
                        r_m = cv2.resize(m_arr, (tw, th), interpolation=cv2.INTER_AREA)
                        p_thumb[r_m > 127] = (218, 218, 222)

                        t_suf = f"_v{pol_var}"
                        p_t_key = f"{current_hash[:12]}_dxf_d{d.index}{t_suf}.jpg"
                        p_t_path = self.thumbnails_dir / p_t_key
                        cv2.imwrite(str(p_t_path), p_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

                        p_m_key = f"{current_hash[:12]}_dxf_d{d.index}{t_suf}_mask.png"
                        p_m_path = self.thumbnails_dir / p_m_key
                        cv2.imwrite(str(p_m_path), m_224)

                        d_code = d.design_code or cat_p.stem
                        d_num = d.design_code or f"{cat_p.stem}_P{d.index}"
                        idx_info = f"design {d.index} of {d.total_designs}"

                        tiles_to_insert.append({
                            "pdf_path": file_str,
                            "folder": folder_rel,
                            "page_number": d.index,
                            "tile_index": d.index if pol_var == 0 else d.index + 1000,
                            "bbox": [int(d.x), int(d.y), max(1, int(round(d.width))), max(1, int(round(d.height)))],
                            "crop_type": "dxf_panel" if d.total_designs > 1 else "dxf",
                            "design_number": d_num,
                            "thumbnail_path": str(p_t_path.relative_to(self.data_dir)),
                            "file_hash": current_hash,
                            "source_type": "dxf",
                            "design_code": d_code,
                            "dxf_bbox": [float(d.x), float(d.y), float(d.width), float(d.height)],
                            "polarity_variant": pol_var,
                            "entity_count": d.entity_count,
                            "design_index_info": idx_info,
                        })
                        masks_to_embed.append(m_224)

                embeddings = self.embedder.embed_masks(masks_to_embed, batch_size=len(masks_to_embed))
                inserted_ids = self.db.insert_tiles_batch(tiles_to_insert)
                self.add_vectors(embeddings, inserted_ids)

                file_tiles_count = len(inserted_ids)
                total_tiles_indexed += len(designs)

                self.db.record_indexed_file(
                    pdf_path=file_str,
                    folder=folder_rel,
                    file_hash=current_hash,
                    file_size=file_size,
                    last_modified=last_mod,
                    page_count=page_count,
                    tile_count=file_tiles_count,
                )
                logger.info("Indexed DXF %s: %d designs (%d vector variants).", cat_p.name, len(designs), file_tiles_count)
                continue

            # Index this PDF
            try:
                page_count = get_pdf_page_count(cat_p)
            except PdfRenderError as exc:
                logger.error("Failed to read PDF '%s': %s", cat_p, exc)
                continue

            # Pre-record file entry to satisfy foreign key constraints for tile inserts
            self.db.record_indexed_file(
                pdf_path=file_str,
                folder=folder_rel,
                file_hash=current_hash,
                file_size=file_size,
                last_modified=last_mod,
                page_count=page_count,
                tile_count=0,
            )

            file_tiles_count = 0

            for page_idx in range(page_count):
                if is_cancelled():
                    emit_progress(IndexProgress(phase="cancelled", message="Indexing cancelled by user."))
                    self.save_index()
                    return total_tiles_indexed

                page_no = page_idx + 1
                emit_progress(
                    IndexProgress(
                        phase="indexing",
                        current_file_idx=file_idx,
                        total_files=total_files,
                        current_pdf_path=cat_p.name,
                        current_page=page_no,
                        total_pages_in_file=page_count,
                        total_tiles_indexed=total_tiles_indexed,
                        percent=float(((file_idx - 1) + (page_no / page_count)) / total_files * 100.0),
                        message=f"{cat_p.name} - Rendering page {page_no}/{page_count}...",
                    )
                )

                # Render page at 200 DPI
                try:
                    page_bgr = render_pdf_page(cat_p, page_idx, dpi=dpi)
                except Exception as exc:
                    logger.warning("Error rendering page %d of '%s': %s", page_no, cat_p.name, exc)
                    continue

                # Detect tile boxes
                tile_boxes = detect_tile_boxes(page_bgr)
                if not tile_boxes:
                    logger.debug("No tiles found on page %d of '%s'", page_no, cat_p.name)
                    del page_bgr
                    continue

                # Extract crops and sub-regions
                tile_crops = extract_tile_crops(
                    page_bgr,
                    tile_boxes,
                    include_subregions=include_subregions,
                    run_ocr=run_ocr,
                )

                # Prepare metadata and masks for embedding
                masks_to_embed: List[np.ndarray] = []
                tile_records_to_insert: List[dict] = []
                temp_thumbnails: List[Tuple[np.ndarray, str]] = []

                for crop in tile_crops:
                    # Letterbox mask to 224x224
                    mask_224 = letterbox_image(crop.stencil_mask, target_size=(224, 224), pad_value=0)
                    masks_to_embed.append(mask_224)

                    # Thumbnail generation (saved with placeholder name until DB assigns ID)
                    thumb_img = letterbox_image(crop.crop_bgr, target_size=(160, 160), pad_value=20)
                    thumb_key = f"{current_hash[:12]}_p{page_no}_t{crop.tile_index}_{crop.crop_type}.jpg"
                    thumb_path = self.thumbnails_dir / thumb_key
                    cv2.imwrite(str(thumb_path), thumb_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

                    # Save binary stencil mask directly for rapid, exact IoU / Chamfer reranking
                    mask_key = f"{current_hash[:12]}_p{page_no}_t{crop.tile_index}_{crop.crop_type}_mask.png"
                    mask_path = self.thumbnails_dir / mask_key
                    cv2.imwrite(str(mask_path), mask_224)

                    tile_records_to_insert.append(
                        {
                            "pdf_path": file_str,
                            "folder": folder_rel,
                            "page_number": page_no,
                            "tile_index": crop.tile_index,
                            "bbox": crop.bbox,
                            "crop_type": crop.crop_type,
                            "design_number": crop.design_number,
                            "thumbnail_path": str(thumb_path.relative_to(self.data_dir)),
                            "file_hash": current_hash,
                        }
                    )

                # Batch embed with DINOv2
                if masks_to_embed:
                    embeddings = self.embedder.embed_masks(masks_to_embed, batch_size=16)

                    # Insert records into SQLite
                    inserted_ids = self.db.insert_tiles_batch(tile_records_to_insert)

                    # Add vectors to FAISS with matched IDs
                    self.add_vectors(embeddings, inserted_ids)

                    file_tiles_count += len(inserted_ids)
                    total_tiles_indexed += len(inserted_ids)

                del page_bgr

            # Record file in database
            self.db.record_indexed_file(
                pdf_path=file_str,
                folder=folder_rel,
                file_hash=current_hash,
                file_size=file_size,
                last_modified=last_mod,
                page_count=page_count,
                tile_count=file_tiles_count,
            )

            logger.info("Indexed %s: %d pages, %d tiles.", cat_p.name, page_count, file_tiles_count)

        # 4. Save FAISS index
        emit_progress(IndexProgress(phase="saving", message="Saving FAISS vector index to disk...", percent=99.0))
        self.save_index()

        emit_progress(
            IndexProgress(
                phase="complete",
                total_tiles_indexed=total_tiles_indexed,
                percent=100.0,
                message=f"Indexing completed! Successfully indexed {total_tiles_indexed} tile vectors.",
            )
        )
        return total_tiles_indexed
