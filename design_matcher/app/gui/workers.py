"""
Background QThread workers for Design Matcher GUI.
Ensures non-blocking catalog indexing and responsive search queries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from app.core.index import CatalogIndex, IndexProgress
from app.core.preprocess import normalize_stencil
from app.core.search import DesignSearcher, SearchResult

logger = logging.getLogger(__name__)


class IndexWorker(QThread):
    """Worker thread for running catalog indexing without blocking the UI."""

    progress_signal = Signal(dict)
    finished_signal = Signal(int)
    error_signal = Signal(str)

    def __init__(
        self,
        catalog_index: CatalogIndex,
        root_dir: Path | str,
        dpi: int = 200,
        include_subregions: bool = True,
        run_ocr: bool = True,
        force: bool = False,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.catalog_index = catalog_index
        self.root_dir = root_dir
        self.dpi = dpi
        self.include_subregions = include_subregions
        self.run_ocr = run_ocr
        self.force = force
        self._is_cancelled = False

    def cancel(self) -> None:
        """Requests worker cancellation."""
        self._is_cancelled = True

    def run(self) -> None:
        """Executes indexing in background thread."""
        def progress_cb(p: IndexProgress) -> None:
            self.progress_signal.emit({
                "phase": p.phase,
                "current_file_idx": p.current_file_idx,
                "total_files": p.total_files,
                "current_pdf_path": p.current_pdf_path,
                "current_page": p.current_page,
                "total_pages_in_file": p.total_pages_in_file,
                "total_tiles_indexed": p.total_tiles_indexed,
                "message": p.message,
                "percent": p.percent,
            })

        try:
            new_tiles = self.catalog_index.index_catalog(
                root_dir=self.root_dir,
                dpi=self.dpi,
                include_subregions=self.include_subregions,
                run_ocr=self.run_ocr,
                progress_cb=progress_cb,
                cancel_check=lambda: self._is_cancelled,
                force_reindex=self.force,
            )
            self.finished_signal.emit(new_tiles)
        except Exception as exc:
            logger.exception("IndexWorker error: %s", exc)
            self.error_signal.emit(str(exc))


class SearchWorker(QThread):
    """Worker thread for computing embeddings and searching the vector index."""

    results_signal = Signal(list)
    error_signal = Signal(str)

    def __init__(
        self,
        searcher: DesignSearcher,
        mask: np.ndarray,
        top_k: int = 5,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.searcher = searcher
        self.mask = mask
        self.top_k = top_k

    def run(self) -> None:
        """Executes vector search in background thread."""
        try:
            results = self.searcher.search_mask(self.mask, top_k=self.top_k)
            self.results_signal.emit(results)
        except Exception as exc:
            logger.exception("SearchWorker error: %s", exc)
            self.error_signal.emit(str(exc))


class PreprocessWorker(QThread):
    """Worker thread for normalizing customer photo to a stencil mask without freezing the UI."""

    finished_signal = Signal(np.ndarray, dict)
    error_signal = Signal(str)

    def __init__(
        self,
        bgr_img: np.ndarray,
        crop: bool = True,
        mode: str = "auto",
        enhance: bool = True,
        threshold_offset: int = 0,
        invert: bool = False,
        trim_border_pct: float = 0.0,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.bgr_img = bgr_img
        self.crop = crop
        self.mode = mode
        self.enhance = enhance
        self.threshold_offset = threshold_offset
        self.invert = invert
        self.trim_border_pct = trim_border_pct

    def run(self) -> None:
        """Executes stencil normalization in background thread."""
        try:
            mask, info = normalize_stencil(
                self.bgr_img,
                crop=self.crop,
                mode=self.mode,
                enhance=self.enhance,
                threshold_offset=self.threshold_offset,
                invert=self.invert,
                trim_border_pct=self.trim_border_pct,
            )
            self.finished_signal.emit(mask, info)
        except Exception as exc:
            logger.exception("PreprocessWorker error: %s", exc)
            self.error_signal.emit(str(exc))


class DxfIndexWorker(QThread):
    """Worker thread for indexing DXF CAD library folder with cancellation and progress reporting."""

    progress_signal = Signal(dict)
    finished_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(
        self,
        db: Any,
        folder_path: Path | str,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.folder_path = folder_path
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        def progress_cb(p: Any) -> None:
            self.progress_signal.emit({
                "phase": getattr(p, "phase", ""),
                "processed": getattr(p, "processed", 0),
                "total": getattr(p, "total", 0),
                "skipped": getattr(p, "skipped", 0),
                "failed": getattr(p, "failed", 0),
                "current_file": getattr(p, "current_file", ""),
                "message": getattr(p, "message", ""),
                "percent": getattr(p, "percent", 0.0),
            })

        try:
            result = self.db.scan_and_index_directory(
                self.folder_path,
                progress_callback=progress_cb,
                cancel_token=lambda: self._is_cancelled,
            )
            self.finished_signal.emit(result)
        except Exception as exc:
            logger.exception("DxfIndexWorker error: %s", exc)
            self.error_signal.emit(str(exc))


class DxfMatchWorker(QThread):
    """
    Worker thread for executing DXF matching:
    - Mode A: PNG stencil -> DXF search
    - Mode B: Reference DXF -> DXF search
    """

    results_signal = Signal(list)
    error_signal = Signal(str)

    def __init__(
        self,
        db: Any,
        mode: str,  # 'stencil' or 'reference_dxf'
        query_data: Any,  # np.ndarray mask or reference DXF Path
        top_k: int = 5,
        candidate_limit: int = 30,
        chamfer_threshold: float = 0.12,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.mode = mode
        self.query_data = query_data
        self.top_k = top_k
        self.candidate_limit = candidate_limit
        self.chamfer_threshold = chamfer_threshold

    def run(self) -> None:
        try:
            from engine.dxf_matcher import search_by_png_stencil, search_by_reference_dxf
            from app.core.search import dxf_match_to_search_result

            if self.mode == "stencil":
                matches = search_by_png_stencil(
                    self.db,
                    self.query_data,
                    top_k=self.top_k,
                    candidate_limit=self.candidate_limit,
                    chamfer_threshold=self.chamfer_threshold,
                )
            elif self.mode == "reference_dxf":
                matches = search_by_reference_dxf(
                    self.db,
                    self.query_data,
                    top_k=self.top_k,
                    candidate_limit=self.candidate_limit,
                    chamfer_threshold=self.chamfer_threshold,
                )
            else:
                raise ValueError(f"Unknown match mode: {self.mode}")

            if hasattr(matches, "matches"):
                matches = matches.matches

            search_results = [dxf_match_to_search_result(m) for m in matches]
            self.results_signal.emit(search_results)
        except Exception as exc:
            logger.exception("DxfMatchWorker error: %s", exc)
            self.error_signal.emit(str(exc))


class LicenseSyncWorker(QThread):
    """Background worker that validates the license online without blocking the UI."""

    sync_finished = Signal(object)  # LicenseCheckResult

    def __init__(self, license_manager, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.license_manager = license_manager

    def run(self) -> None:
        verdict = self.license_manager.validate_online()
        self.sync_finished.emit(verdict)



