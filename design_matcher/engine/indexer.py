"""
Catalog indexing engine.
Recursively scans folders for PDFs and CAD DXF files, extracts designs,
computes canonical masks and embeddings, and persists metadata to SQLite and FAISS.
Incremental (skips unchanged, updates changed, deletes removed) and cancellable.
Processes DXFs via multiprocessing pool with main-thread atomic DB writes.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import multiprocessing
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from .config import EngineConfig, config
from .dxf_reader import DxfReaderError, read_dxf
from .dxf_segment import segment_dxf_drawing
from .features import BaseEmbedder, canonical_mask, get_default_embedder
from .index_store import IndexStore
from .linework_to_mask import is_outline_only, outline_to_mask
from .pdf_render import PdfEncryptedError, PdfRenderError, get_pdf_page_count, render_pdf_page
from .segment import segment_designs
from .utils import compute_file_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Summary statistics for an indexing run."""

    scanned_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    deleted_files: int = 0
    indexed_designs: int = 0
    errors: int = 0
    cancelled: bool = False


def _process_page_worker(args: Tuple[str, int, int, Optional[str]]) -> dict:
    """
    Picklable worker function for PDF page rendering and segmentation.
    """
    pdf_path_str, page_no, dpi, embedder_type = args
    try:
        page_bgr = render_pdf_page(pdf_path_str, page_no, dpi=dpi)
        from preprocess import normalize_stencil  # type: ignore

        page_mask, _ = normalize_stencil(page_bgr, crop=False)

        # Detect outline-only / wireframe PDF pages
        if is_outline_only(page_mask):
            page_mask, _ = outline_to_mask(page_bgr)

        bboxes = segment_designs(page_bgr, page_mask=page_mask)
        embedder = get_default_embedder(embedder_type)
        page_designs = []

        for x, y, w, h in bboxes:
            crop_bgr = page_bgr[y : y + h, x : x + w]
            crop_mask = page_mask[y : y + h, x : x + w]
            if crop_bgr.size == 0 or crop_mask.size == 0:
                continue

            can_mask, ar = canonical_mask(crop_mask)
            fill_ratio = float((can_mask == 255).mean())
            mask_hash = hashlib.md5(can_mask.tobytes()).hexdigest()

            # Thumbnail generation (120px max dimension, PNG)
            crop_h, crop_w = crop_bgr.shape[:2]
            scale = 120.0 / float(max(crop_h, crop_w))
            tw = max(1, int(round(crop_w * scale)))
            th = max(1, int(round(crop_h * scale)))
            thumb_bgr = cv2.resize(crop_bgr, (tw, th), interpolation=cv2.INTER_AREA)
            _, thumb_png = cv2.imencode(".png", thumb_bgr)

            # Canonical mask PNG for fast zero-render re-ranking
            _, can_png = cv2.imencode(".png", can_mask)
            vec = embedder.embed(can_mask, ar)

            page_designs.append({
                "page_no": page_no,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "thumb": thumb_png.tobytes(),
                "canonical_mask": can_png.tobytes(),
                "mask_hash": mask_hash,
                "aspect_ratio": ar,
                "fill_ratio": fill_ratio,
                "vector": vec,
                "source_type": "pdf",
            })

        return {"page_no": page_no, "designs": page_designs, "error": None}
    except Exception as exc:
        return {"page_no": page_no, "designs": [], "error": str(exc)}


def _process_dxf_worker(args: Tuple[str, Optional[str]]) -> dict:
    """
    Picklable worker function for parsing, segmenting, and embedding DXF files.
    Uses engine.dxf_segment and linework_to_mask for vector polygonization and nesting depth.
    """
    dxf_path_str, embedder_type = args
    try:
        dxf_data = read_dxf(dxf_path_str)
        if not dxf_data.polylines:
            return {
                "entity_count": 0,
                "designs": [],
                "unique_design_count": 0,
                "error": f"DXF contains no readable geometry or is corrupted: {Path(dxf_path_str).name}",
            }
        segmented_designs = segment_dxf_drawing(dxf_data)
        if not segmented_designs:
            return {
                "entity_count": 0,
                "designs": [],
                "unique_design_count": 0,
                "error": f"DXF drawing contains no design components: {Path(dxf_path_str).name}",
            }
        embedder = get_default_embedder(embedder_type)
        out_designs = []

        total_des = len(segmented_designs)
        for idx, d in enumerate(segmented_designs):
            can_mask, ar = canonical_mask(d.mask)
            fill_ratio = float((can_mask == 255).mean())
            mask_hash = hashlib.md5(can_mask.tobytes()).hexdigest()

            mh, mw = d.mask.shape[:2]
            scale = 120.0 / float(max(mh, mw))
            tw = max(1, int(round(mw * scale)))
            th = max(1, int(round(mh * scale)))
            thumb_bgr = np.full((th, tw, 3), (25, 20, 15), dtype=np.uint8)
            resized_mask = cv2.resize(d.mask, (tw, th), interpolation=cv2.INTER_AREA)
            thumb_bgr[resized_mask > 127] = (218, 218, 222)
            _, thumb_png = cv2.imencode(".png", thumb_bgr)

            _, can_png = cv2.imencode(".png", can_mask)
            vec = embedder.embed(can_mask, ar)

            bx, by, bw, bh = d.bbox
            out_designs.append({
                "source_type": "dxf",
                "label": d.label,
                "design_code": d.label,
                "bbox_x": float(bx),
                "bbox_y": float(by),
                "bbox_w": float(bw),
                "bbox_h": float(bh),
                "dxf_bbox": (float(bx), float(by), float(bw), float(bh)),
                "polarity_variant": 0,
                "entity_count": d.entity_count,
                "design_index_info": f"design {idx + 1} of {total_des}",
                "x": 0,
                "y": 0,
                "w": max(1, int(round(bw))),
                "h": max(1, int(round(bh))),
                "thumb": thumb_png.tobytes(),
                "canonical_mask": can_png.tobytes(),
                "mask_hash": mask_hash,
                "aspect_ratio": ar,
                "fill_ratio": fill_ratio,
                "vector": vec,
            })

        return {
            "entity_count": len(dxf_data.polylines),
            "designs": out_designs,
            "unique_design_count": len(segmented_designs),
            "error": None,
        }
    except Exception as exc:
        return {
            "entity_count": 0,
            "designs": [],
            "unique_design_count": 0,
            "error": str(exc),
        }


def find_pdf_files(root_folders: List[Union[str, Path]]) -> List[Path]:
    """Recursively scans folders for *.pdf and *.dxf files, case-insensitively with Unicode support."""
    catalog_paths: List[Path] = []
    seen: Set[str] = set()
    valid_exts = {".pdf", ".dxf"}

    for folder in root_folders:
        folder_path = Path(folder).resolve()
        if not folder_path.exists():
            continue
        if folder_path.is_file() and folder_path.suffix.lower() in valid_exts:
            p_str = str(folder_path)
            if p_str not in seen:
                seen.add(p_str)
                catalog_paths.append(folder_path)
            continue

        for root, _, files in os.walk(str(folder_path)):
            for file in files:
                if Path(file).suffix.lower() in valid_exts:
                    full_p = Path(root, file).resolve()
                    p_str = str(full_p)
                    if p_str not in seen:
                        seen.add(p_str)
                        catalog_paths.append(full_p)

    return sorted(catalog_paths)


find_catalog_files = find_pdf_files


def find_dwg_files(root_folders: List[Union[str, Path]]) -> List[Path]:
    """Recursively scans folders for *.dwg files to detect unsupported formats."""
    dwg_paths: List[Path] = []
    seen: Set[str] = set()

    for folder in root_folders:
        folder_path = Path(folder).resolve()
        if not folder_path.exists():
            continue
        if folder_path.is_file() and folder_path.suffix.lower() == ".dwg":
            p_str = str(folder_path)
            if p_str not in seen:
                seen.add(p_str)
                dwg_paths.append(folder_path)
            continue

        for root, _, files in os.walk(str(folder_path)):
            for file in files:
                if Path(file).suffix.lower() == ".dwg":
                    full_p = Path(root, file).resolve()
                    p_str = str(full_p)
                    if p_str not in seen:
                        seen.add(p_str)
                        dwg_paths.append(full_p)

    return sorted(dwg_paths)


def index_folders(
    folders: List[Union[str, Path]],
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    store: Optional[IndexStore] = None,
    embedder: Optional[BaseEmbedder] = None,
    cfg: EngineConfig = config,
) -> IndexStats:
    """
    Scans folders and incrementally indexes both PDF catalogs and CAD DXF drawings.

    Args:
        folders: List of root folder paths to index.
        progress_cb: Optional callback(current_file, page, total_pages, message).
        cancel_event: Optional threading.Event to signal cancellation.
        store: Optional IndexStore instance.
        embedder: Optional Embedder instance.
        cfg: Configuration settings.

    Returns:
        IndexStats with summary of indexing results.
    """
    multiprocessing.freeze_support()
    stats = IndexStats()

    if store is None:
        store = IndexStore(db_path=cfg.DB_PATH, index_path=cfg.INDEX_PATH)
    if embedder is None:
        embedder = get_default_embedder(cfg.EMBEDDER_TYPE, cfg)

    # 1. Discover all disk catalog files and check for unsupported DWG files
    disk_files = find_catalog_files(folders)
    stats.scanned_files = len(disk_files)
    disk_path_strs = {str(p) for p in disk_files}

    # Record unsupported DWG files in index_errors
    dwg_files = find_dwg_files(folders)
    for dwg in dwg_files:
        msg = "DWG not supported: please export to DXF"
        logger.warning("Skipping unsupported DWG file: %s", dwg.name)
        store.record_error(str(dwg), msg)
        stats.errors += 1

    folder_resolved_strs = [str(Path(f).resolve()) for f in folders]

    # 2. Reconcile deleted PDFs and DXFs
    indexed_pdfs = store.get_all_indexed_pdf_paths()
    for indexed_path, row in indexed_pdfs.items():
        if cancel_event and cancel_event.is_set():
            stats.cancelled = True
            store.save()
            return stats

        in_scanned = any(indexed_path.startswith(f_root) for f_root in folder_resolved_strs)
        if in_scanned and indexed_path not in disk_path_strs:
            logger.info("Removing deleted PDF from index: %s", indexed_path)
            store.remove_pdf(row["id"])
            stats.deleted_files += 1

    indexed_dxfs = store.get_all_indexed_dxf_paths()
    for indexed_path, row in indexed_dxfs.items():
        if cancel_event and cancel_event.is_set():
            stats.cancelled = True
            store.save()
            return stats

        in_scanned = any(indexed_path.startswith(f_root) for f_root in folder_resolved_strs)
        if in_scanned and indexed_path not in disk_path_strs:
            logger.info("Removing deleted DXF from index: %s", indexed_path)
            store.remove_dxf(row["id"])
            stats.deleted_files += 1

    # Separate discovered files into PDFs and DXFs
    pdf_files = [p for p in disk_files if p.suffix.lower() == ".pdf"]
    dxf_files = [p for p in disk_files if p.suffix.lower() == ".dxf"]

    # 3. Process DXF files incrementally
    for dxf_path in dxf_files:
        if cancel_event and cancel_event.is_set():
            stats.cancelled = True
            store.save()
            return stats

        dxf_path_str = str(dxf_path)
        folder_str = str(dxf_path.parent)
        fname = dxf_path.name

        try:
            size, mtime, sha1 = compute_file_fingerprint(dxf_path)
        except Exception as exc:
            logger.error("Failed to fingerprint DXF %s: %s", dxf_path_str, exc)
            store.record_error(dxf_path_str, f"Fingerprint error: {exc}")
            stats.errors += 1
            continue

        existing_dxf = indexed_dxfs.get(dxf_path_str)
        if existing_dxf:
            if (
                existing_dxf["size"] == size
                and abs(existing_dxf["mtime"] - mtime) < 0.001
                and existing_dxf["sha1"] == sha1
            ):
                stats.skipped_files += 1
                if progress_cb:
                    progress_cb(fname, 0, 1, f"Skipped unchanged DXF {fname}")
                continue
            else:
                logger.info("DXF modified, re-indexing: %s", dxf_path_str)
                store.remove_dxf(existing_dxf["id"])

        if progress_cb:
            progress_cb(fname, 0, 1, f"Parsing DXF {fname}")

        # Process DXF via worker
        dxf_res = None
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_process_dxf_worker, (dxf_path_str, cfg.EMBEDDER_TYPE))
                dxf_res = future.result()
        except Exception as exc:
            logger.debug("ProcessPool failed for DXF (%s), using in-process execution", exc)
            dxf_res = _process_dxf_worker((dxf_path_str, cfg.EMBEDDER_TYPE))

        if dxf_res.get("error"):
            logger.error("Failed to index DXF %s: %s", fname, dxf_res["error"])
            store.record_error(dxf_path_str, dxf_res["error"])
            stats.errors += 1
            continue

        entity_cnt = dxf_res.get("entity_count", 0)
        dxf_id = store.upsert_dxf_record(
            path=dxf_path_str,
            folder=folder_str,
            size=size,
            mtime=mtime,
            sha1=sha1,
            entity_count=entity_cnt,
        )

        dxf_designs = dxf_res.get("designs", [])
        if dxf_designs:
            records = []
            vectors = []
            for d in dxf_designs:
                rec_dict = dict(d)
                rec_dict["dxf_id"] = dxf_id
                rec_dict["pdf_id"] = None
                rec_dict["page_no"] = None
                records.append(rec_dict)
                vectors.append(rec_dict.pop("vector"))

            vec_matrix = np.stack(vectors, axis=0)
            inserted_ids = store.add_designs(records, vec_matrix)
            uniq_cnt = dxf_res.get("unique_design_count") or len([r for r in records if r.get("polarity_variant", 0) == 0]) or len(inserted_ids)
            stats.indexed_designs += uniq_cnt

            # Link paired dual-polarity variants under same design group
            for i in range(0, len(inserted_ids) - 1, 2):
                if (
                    records[i].get("polarity_variant") == 0
                    and records[i + 1].get("polarity_variant") == 1
                ):
                    store.link_same_design_group(inserted_ids[i], inserted_ids[i + 1])

            # Dedup linking: link DXF and PDF designs sharing identical mask_hash or sim > 0.995
            for rec_idx, (ins_id, rec) in enumerate(zip(inserted_ids, records)):
                cur = store.conn.cursor()
                cur.execute(
                    "SELECT id FROM designs WHERE mask_hash = ? AND id != ? LIMIT 1",
                    (rec["mask_hash"], ins_id),
                )
                dup_row = cur.fetchone()
                if dup_row:
                    store.link_same_design_group(ins_id, int(dup_row["id"]))
                elif store.index.ntotal > 1:
                    v = vec_matrix[rec_idx].reshape(1, -1)
                    scores, idxs = store.search_vectors(v, top_k=2)
                    if len(scores) > 0 and len(scores[0]) > 1:
                        for s, match_id in zip(scores[0], idxs[0]):
                            if match_id != ins_id and match_id != -1 and s >= 0.995:
                                store.link_same_design_group(ins_id, int(match_id))
                                break

        stats.indexed_files += 1

    # 4. Process PDF files incrementally
    for pdf_path in pdf_files:
        if cancel_event and cancel_event.is_set():
            stats.cancelled = True
            store.save()
            return stats

        pdf_path_str = str(pdf_path)
        folder_str = str(pdf_path.parent)
        fname = pdf_path.name

        try:
            size, mtime, sha1 = compute_file_fingerprint(pdf_path)
        except Exception as exc:
            logger.error("Failed to fingerprint PDF %s: %s", pdf_path_str, exc)
            store.record_error(pdf_path_str, f"Fingerprint error: {exc}")
            stats.errors += 1
            continue

        existing_rec = indexed_pdfs.get(pdf_path_str)
        if existing_rec:
            if (
                existing_rec["size"] == size
                and abs(existing_rec["mtime"] - mtime) < 0.001
                and existing_rec["sha1"] == sha1
            ):
                stats.skipped_files += 1
                if progress_cb:
                    progress_cb(fname, 0, existing_rec["page_count"], f"Skipped unchanged {fname}")
                continue
            else:
                logger.info("PDF changed, re-indexing: %s", pdf_path_str)
                store.remove_pdf(existing_rec["id"])

        try:
            page_count = get_pdf_page_count(pdf_path)
        except (PdfEncryptedError, PdfRenderError) as exc:
            logger.warning("Skipping invalid/protected PDF %s: %s", fname, exc)
            store.record_error(pdf_path_str, str(exc))
            stats.errors += 1
            continue
        except Exception as exc:
            logger.error("Unexpected error opening PDF %s: %s", fname, exc)
            store.record_error(pdf_path_str, f"Open error: {exc}")
            stats.errors += 1
            continue

        pdf_id = store.upsert_pdf_record(
            path=pdf_path_str,
            folder=folder_str,
            size=size,
            mtime=mtime,
            sha1=sha1,
            page_count=page_count,
        )

        if progress_cb:
            progress_cb(fname, 0, page_count, f"Indexing {fname} (0/{page_count} pages)")

        page_tasks = [
            (pdf_path_str, p, cfg.RENDER_DPI, cfg.EMBEDDER_TYPE)
            for p in range(page_count)
        ]

        page_results = []
        max_workers = min(cfg.MAX_WORKERS, max(1, page_count))

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_process_page_worker, task): task[1]
                    for task in page_tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    if cancel_event and cancel_event.is_set():
                        stats.cancelled = True
                        store.save()
                        return stats
                    p_no = futures[future]
                    try:
                        res = future.result()
                        page_results.append(res)
                        if progress_cb:
                            progress_cb(fname, len(page_results), page_count, f"Processed page {len(page_results)}/{page_count}")
                    except Exception as exc:
                        logger.error("Page %d of %s failed: %s", p_no, fname, exc)
                        store.record_error(pdf_path_str, f"Page {p_no} failed: {exc}")
                        stats.errors += 1
        except Exception as exc:
            logger.warning("ProcessPoolExecutor fallback to sequential (%s)", exc)
            for task in page_tasks:
                if cancel_event and cancel_event.is_set():
                    stats.cancelled = True
                    store.save()
                    return stats
                res = _process_page_worker(task)
                page_results.append(res)
                if progress_cb:
                    progress_cb(fname, len(page_results), page_count, f"Processed page {len(page_results)}/{page_count}")

        page_results.sort(key=lambda item: item["page_no"])
        file_records: List[dict] = []
        file_vectors: List[np.ndarray] = []

        for p_res in page_results:
            if p_res.get("error"):
                logger.warning("Page error in %s: %s", fname, p_res["error"])
                continue
            for d in p_res.get("designs", []):
                file_records.append({
                    "pdf_id": pdf_id,
                    "dxf_id": None,
                    "source_type": "pdf",
                    "page_no": d["page_no"],
                    "label": None,
                    "bbox_x": None,
                    "bbox_y": None,
                    "bbox_w": None,
                    "bbox_h": None,
                    "x": d["x"],
                    "y": d["y"],
                    "w": d["w"],
                    "h": d["h"],
                    "thumb": d["thumb"],
                    "canonical_mask": d["canonical_mask"],
                    "mask_hash": d["mask_hash"],
                    "aspect_ratio": d["aspect_ratio"],
                    "fill_ratio": d["fill_ratio"],
                })
                file_vectors.append(d["vector"])

        if file_records:
            vec_matrix = np.stack(file_vectors, axis=0)
            inserted_ids = store.add_designs(file_records, vec_matrix)
            stats.indexed_designs += len(inserted_ids)

            # Dedup linking against DXF designs sharing identical mask_hash or sim > 0.995
            for rec_idx, (ins_id, rec) in enumerate(zip(inserted_ids, file_records)):
                cur = store.conn.cursor()
                cur.execute(
                    "SELECT id FROM designs WHERE mask_hash = ? AND id != ? LIMIT 1",
                    (rec["mask_hash"], ins_id),
                )
                dup_row = cur.fetchone()
                if dup_row:
                    store.link_same_design_group(ins_id, int(dup_row["id"]))
                elif store.index.ntotal > 1:
                    v = vec_matrix[rec_idx].reshape(1, -1)
                    scores, idxs = store.search_vectors(v, top_k=2)
                    if len(scores) > 0 and len(scores[0]) > 1:
                        for s, match_id in zip(scores[0], idxs[0]):
                            if match_id != ins_id and match_id != -1 and s >= 0.995:
                                store.link_same_design_group(ins_id, int(match_id))
                                break

        stats.indexed_files += 1

    store.save()
    return stats
