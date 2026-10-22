"""
Two-Stage Production Indexing Engine for Design Matcher.
Implements:
1. Stage A: High-speed discovery (os.scandir), lightweight metadata parsing (pypdfium2 / ezdxf),
   persistent manifest storage (catalogue_files in SQLite WAL), and instant full-text search (FTS5).
2. Stage B: Asynchronous visual feature generation (ProcessPoolExecutor), normalized 224x224
   binary masks, high-throughput DINOv2 batch embedding, and FAISS vector alignment.
3. Stage C: Zero-overhead incremental updates (skips unchanged files by (size, modified_ns)),
   graceful crash recovery, and single-writer batched database commits.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from app.core.db import CatalogDatabase
from app.core.profiling import ProfilingSession, measure_stage
from app.core.stage_a_extractor import extract_metadata_batch, extract_single_file_metadata
from app.core.stage_b_extractor import process_dxf_visual, process_pdf_page_visual

logger = logging.getLogger(__name__)


@dataclass
class TwoStageProgress:
    """Detailed progress emitted across Stage A and Stage B."""
    phase: str  # 'scanning', 'stage_a_metadata', 'stage_b_visual', 'saving', 'complete', 'cancelled'
    stage: str  # 'A', 'B', or 'idle'
    current_item: int = 0
    total_items: int = 0
    current_file: str = ""
    total_tiles_indexed: int = 0
    percent: float = 0.0
    message: str = ""


class TwoStageIndexingEngine:
    """
    Coordinates high-throughput parallel catalog ingestion adhering to
    the production indexing optimization specification.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".dxf", ".dwg"}

    def __init__(
        self,
        catalog_index: Any,  # CatalogIndex instance
        max_workers: Optional[int] = None,
    ) -> None:
        self.catalog_index = catalog_index
        self.db: CatalogDatabase = catalog_index.db
        self.data_dir: Path = catalog_index.data_dir
        self.thumbnails_dir: Path = catalog_index.thumbnails_dir
        self.embedder = catalog_index.embedder

        # Calculate optimal worker count (bounded ProcessPoolExecutor)
        cpu_cnt = os.cpu_count() or 2
        self.max_workers = max_workers or min(4, max(1, cpu_cnt - 1))

    @staticmethod
    def scan_directory_scandir(root_path: Path) -> List[Tuple[str, int, int]]:
        """
        Fast recursive directory scanner using os.scandir().
        Returns:
            List of (normalized_path_str, modified_ns, file_size) tuples.
        """
        results: List[Tuple[str, int, int]] = []
        stack = [root_path]

        while stack:
            current_dir = stack.pop()
            try:
                with os.scandir(current_dir) as it:
                    for entry in it:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                ext = Path(entry.name).suffix.lower()
                                if ext in TwoStageIndexingEngine.SUPPORTED_EXTENSIONS or ext == ".dwg":
                                    stat_res = entry.stat()
                                    mod_ns = getattr(stat_res, "st_mtime_ns", int(stat_res.st_mtime * 1e9))
                                    results.append((str(Path(entry.path).resolve()), mod_ns, stat_res.st_size))
                        except (PermissionError, FileNotFoundError):
                            continue
            except (PermissionError, FileNotFoundError):
                continue

        results.sort(key=lambda t: t[0])
        return results

    def run_two_stage_index(
        self,
        root_dir: Path | str,
        dpi: int = 200,
        include_subregions: bool = True,
        run_ocr: bool = True,
        mode: str = "full",  # 'full', 'metadata_only', 'visual_only'
        progress_cb: Optional[Callable[[TwoStageProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes the two-stage indexing workflow.
        Returns a dictionary of execution metrics and summary counts.
        """
        root_path = Path(root_dir).resolve()
        if not root_path.is_dir():
            raise NotADirectoryError(f"Root catalogue directory does not exist: {root_dir}")

        session = ProfilingSession("two_stage_indexing")

        def emit(prog: TwoStageProgress) -> None:
            if progress_cb:
                progress_cb(prog)

        def is_cancelled() -> bool:
            return cancel_check() if cancel_check else False

        # ---------------------------------------------------------------------
        # Step 1: High-Speed Directory Discovery
        # ---------------------------------------------------------------------
        emit(TwoStageProgress(phase="scanning", stage="idle", message="Scanning directory catalogue with os.scandir()..."))
        with session.timer("discovery"):
            scanned_files = self.scan_directory_scandir(root_path)

        total_scanned = len(scanned_files)
        logger.info("Discovery complete: found %d candidate files in %s", total_scanned, root_path)

        if total_scanned == 0:
            emit(TwoStageProgress(phase="complete", stage="idle", percent=100.0, message="No files found."))
            session.finish(0, 0)
            return session.get_summary()

        # ---------------------------------------------------------------------
        # Step 2: Incremental Change Detection & Missing Files Pruning
        # ---------------------------------------------------------------------
        with session.timer("change_detection"):
            scanned_paths_set = {t[0] for t in scanned_files}
            # Clean missing records from SQLite manifest
            del_manifest = self.db.cleanup_missing_catalogue_files(scanned_paths_set)
            del_paths, del_vector_ids = self.db.cleanup_missing_files(scanned_paths_set)
            if del_vector_ids:
                self.catalog_index.remove_vectors(del_vector_ids)
                logger.info("Cleaned %d missing files and %d obsolete vector IDs", len(del_paths), len(del_vector_ids))

            # Retrieve existing manifest records
            existing_manifest = {r["path"]: r for r in self.db.get_all_catalogue_files()}

            files_needing_stage_a: List[Tuple[str, int, int]] = []
            files_needing_stage_b: List[Dict[str, Any]] = []
            unchanged_count = 0

            for fpath, mod_ns, fsize in scanned_files:
                ext = Path(fpath).suffix.lower()
                rec = existing_manifest.get(fpath)

                if rec and not force_reindex:
                    # Check if file has same size and modification timestamp
                    size_match = int(rec.get("file_size", 0)) == fsize
                    time_match = int(rec.get("modified_ns", 0)) == mod_ns

                    if size_match and time_match:
                        # File is unmodified on disk
                        meta_ok = rec.get("metadata_status") == "completed"
                        vis_ok = rec.get("visual_status") == "completed"

                        if meta_ok and vis_ok:
                            unchanged_count += 1
                            continue
                        elif meta_ok and not vis_ok:
                            # Stage A is already done; queue for Stage B
                            files_needing_stage_b.append(rec)
                            continue

                # File is new, changed, or forced reindex
                files_needing_stage_a.append((fpath, mod_ns, fsize))

        logger.info(
            "Change classification: %d unchanged (0ms skip), %d need Stage A, %d need Stage B only",
            unchanged_count, len(files_needing_stage_a), len(files_needing_stage_b),
        )

        # ---------------------------------------------------------------------
        # Step 3: Stage A — Searchable Metadata Extraction
        # ---------------------------------------------------------------------
        if mode in ("full", "metadata_only") and files_needing_stage_a:
            total_stage_a = len(files_needing_stage_a)
            emit(TwoStageProgress(
                phase="stage_a_metadata",
                stage="A",
                current_item=0,
                total_items=total_stage_a,
                message=f"Stage A: Extracting metadata for {total_stage_a} files with {self.max_workers} workers...",
            ))

            batch_size = 32
            file_chunks = [
                files_needing_stage_a[i : i + batch_size]
                for i in range(0, total_stage_a, batch_size)
            ]

            processed_a = 0
            with session.timer("stage_a_execution"):
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_chunk = {
                        executor.submit(extract_metadata_batch, chunk): chunk
                        for chunk in file_chunks
                    }

                    for future in concurrent.futures.as_completed(future_to_chunk):
                        if is_cancelled():
                            emit(TwoStageProgress(phase="cancelled", stage="A", message="Indexing cancelled."))
                            session.finish(processed_a, 0)
                            return session.get_summary()

                        try:
                            batch_results = future.result()
                        except Exception as exc:
                            logger.error("Stage A worker batch error: %s", exc)
                            continue

                        # Batched database write in main process
                        with session.timer("db_write_stage_a"):
                            self.db.upsert_catalogue_files_batch(batch_results)

                        for res_rec in batch_results:
                            processed_a += 1
                            if res_rec.get("metadata_status") == "completed":
                                files_needing_stage_b.append(res_rec)

                        pct = float(processed_a / total_stage_a * 100.0)
                        emit(TwoStageProgress(
                            phase="stage_a_metadata",
                            stage="A",
                            current_item=processed_a,
                            total_items=total_stage_a,
                            current_file=batch_results[-1]["filename"] if batch_results else "",
                            percent=pct,
                            message=f"Stage A: Extracted metadata ({processed_a}/{total_stage_a}) - Search is ready!",
                        ))

            logger.info("Stage A complete: %d records processed and indexed for FTS", processed_a)

        if mode == "metadata_only":
            emit(TwoStageProgress(
                phase="complete",
                stage="idle",
                percent=100.0,
                message=f"Stage A indexing complete. {len(files_needing_stage_a)} files indexed.",
            ))
            session.finish(len(files_needing_stage_a), 0)
            return session.get_summary()

        # ---------------------------------------------------------------------
        # Step 4: Stage B — Visual Similarity, Tile Segmentation & DINOv2
        # ---------------------------------------------------------------------
        total_tiles_indexed = 0
        total_stage_b = len(files_needing_stage_b)

        if mode in ("full", "visual_only") and total_stage_b > 0:
            emit(TwoStageProgress(
                phase="stage_b_visual",
                stage="B",
                current_item=0,
                total_items=total_stage_b,
                message=f"Stage B: Generating visual descriptors for {total_stage_b} files...",
            ))

            for file_idx, file_rec in enumerate(files_needing_stage_b, start=1):
                if is_cancelled():
                    emit(TwoStageProgress(phase="cancelled", stage="B", message="Indexing cancelled by user."))
                    self.catalog_index.save_index()
                    session.finish(file_idx, total_tiles_indexed)
                    return session.get_summary()

                fpath = file_rec["path"]
                fname = file_rec["filename"]
                ext = file_rec.get("extension", Path(fpath).suffix.lower())
                folder_rel = str(Path(fpath).parent.relative_to(root_path)) if Path(fpath).parent != root_path else "."
                content_hash = file_rec.get("content_hash") or ""

                emit(TwoStageProgress(
                    phase="stage_b_visual",
                    stage="B",
                    current_item=file_idx,
                    total_items=total_stage_b,
                    current_file=fname,
                    total_tiles_indexed=total_tiles_indexed,
                    percent=float((file_idx - 1) / total_stage_b * 100.0),
                    message=f"Stage B: Processing visual features for {fname} ({file_idx}/{total_stage_b})...",
                ))

                # Mark as processing
                self.db.update_catalogue_visual_status(fpath, "processing")

                tiles_to_insert = []
                masks_to_embed = []

                if ext in (".dxf", ".dwg"):
                    with session.timer("render_dxf"):
                        dxf_res = process_dxf_visual(fpath)

                    if dxf_res.get("error"):
                        self.db.update_catalogue_visual_status(fpath, "failed", dxf_res["error"])
                        continue

                    # Pre-record to indexed_files for backwards compatibility
                    page_cnt = len(dxf_res["tiles"])
                    self.db.record_indexed_file(
                        pdf_path=fpath,
                        folder=folder_rel,
                        file_hash=content_hash,
                        file_size=file_rec["file_size"],
                        last_modified=float(file_rec["modified_ns"] / 1e9),
                        page_count=page_cnt,
                        tile_count=0,
                    )

                    for t in dxf_res["tiles"]:
                        p_t_key = f"{content_hash[:12]}_dxf_p{t['page_number']}_t{t['tile_index']}.jpg"
                        p_t_path = self.thumbnails_dir / p_t_key
                        cv2.imwrite(str(p_t_path), t["thumb_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 85])

                        tiles_to_insert.append({
                            "pdf_path": fpath,
                            "folder": folder_rel,
                            "page_number": t["page_number"],
                            "tile_index": t["tile_index"],
                            "bbox": t["bbox"],
                            "crop_type": t["crop_type"],
                            "design_number": t["design_number"],
                            "thumbnail_path": str(p_t_path.relative_to(self.data_dir)),
                            "file_hash": content_hash,
                            "source_type": "dxf",
                            "design_code": t.get("design_code", fname),
                            "dxf_bbox": t.get("dxf_bbox", t["bbox"]),
                            "polarity_variant": t.get("polarity_variant", 0),
                            "entity_count": t.get("entity_count", 0),
                            "design_index_info": t.get("design_index_info", ""),
                        })
                        masks_to_embed.append(t["mask_224"])

                elif ext == ".pdf":
                    page_count = int(file_rec.get("page_count", 1))

                    self.db.record_indexed_file(
                        pdf_path=fpath,
                        folder=folder_rel,
                        file_hash=content_hash,
                        file_size=file_rec["file_size"],
                        last_modified=float(file_rec["modified_ns"] / 1e9),
                        page_count=page_count,
                        tile_count=0,
                    )

                    # Parallel page rendering across ProcessPoolExecutor
                    with session.timer("render_pdf"):
                        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                            futures = [
                                executor.submit(
                                    process_pdf_page_visual,
                                    fpath,
                                    p_idx,
                                    dpi,
                                    include_subregions,
                                    run_ocr,
                                )
                                for p_idx in range(page_count)
                            ]

                            for f in concurrent.futures.as_completed(futures):
                                page_res = f.result()
                                if page_res.get("error"):
                                    continue

                                for t in page_res["tiles"]:
                                    p_t_key = f"{content_hash[:12]}_pdf_p{t['page_number']}_t{t['tile_index']}_{t['crop_type']}.jpg"
                                    p_t_path = self.thumbnails_dir / p_t_key
                                    cv2.imwrite(str(p_t_path), t["thumb_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 85])

                                    tiles_to_insert.append({
                                        "pdf_path": fpath,
                                        "folder": folder_rel,
                                        "page_number": t["page_number"],
                                        "tile_index": t["tile_index"],
                                        "bbox": t["bbox"],
                                        "crop_type": t["crop_type"],
                                        "design_number": t["design_number"],
                                        "thumbnail_path": str(p_t_path.relative_to(self.data_dir)),
                                        "file_hash": content_hash,
                                        "source_type": "pdf",
                                        "design_code": t.get("design_code", ""),
                                        "dxf_bbox": None,
                                        "polarity_variant": 0,
                                        "entity_count": 0,
                                        "design_index_info": f"page {t['page_number']}",
                                    })
                                    masks_to_embed.append(t["mask_224"])

                # Embedding & Indexing
                if masks_to_embed:
                    with session.timer("dinov2_embedding"):
                        embeddings = self.embedder.embed_masks(masks_to_embed, batch_size=32)

                    with session.timer("db_write_tiles"):
                        inserted_ids = self.db.insert_tiles_batch(tiles_to_insert)

                    with session.timer("faiss_vector_add"):
                        self.catalog_index.add_vectors(embeddings, inserted_ids)

                    total_tiles_indexed += len(inserted_ids)

                    # Update file records with tile counts
                    self.db.record_indexed_file(
                        pdf_path=fpath,
                        folder=folder_rel,
                        file_hash=content_hash,
                        file_size=file_rec["file_size"],
                        last_modified=float(file_rec["modified_ns"] / 1e9),
                        page_count=int(file_rec.get("page_count", 1)),
                        tile_count=len(inserted_ids),
                    )

                self.db.update_catalogue_visual_status(fpath, "completed")

                # Periodically save FAISS index every 25 files
                if file_idx % 25 == 0:
                    with session.timer("faiss_disk_save"):
                        self.catalog_index.save_index()

        # Final save
        with session.timer("final_index_save"):
            self.catalog_index.save_index()

        session.finish(total_scanned, total_tiles_indexed)
        logger.info("\n%s", session.format_report())

        emit(TwoStageProgress(
            phase="complete",
            stage="idle",
            current_item=total_scanned,
            total_items=total_scanned,
            total_tiles_indexed=total_tiles_indexed,
            percent=100.0,
            message=f"Indexing complete! Processed {total_scanned} files, generated {total_tiles_indexed} visual vectors.",
        ))

        return session.get_summary()
