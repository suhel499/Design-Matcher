"""
PySide6 asynchronous worker threads for non-blocking UI integration.
Separated from core engine so the core engine remains UI-agnostic.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PySide6.QtCore import QThread, Signal

from .config import EngineConfig, config
from .index_store import IndexStore
from .indexer import IndexStats, index_folders
from .matcher import Matcher, SearchResponse

logger = logging.getLogger(__name__)


class SearchWorker(QThread):
    """Asynchronous worker for performing design matching without freezing the GUI."""

    started_search = Signal()
    finished = Signal(object)  # Emits SearchResponse
    error = Signal(str)

    def __init__(
        self,
        photo_bgr: np.ndarray,
        top_k: int = config.TOP_K,
        already_cropped: bool = False,
        matcher: Optional[Matcher] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.photo_bgr = photo_bgr
        self.top_k = top_k
        self.already_cropped = already_cropped
        self.matcher = matcher

    def run(self) -> None:
        self.started_search.emit()
        try:
            m = self.matcher or Matcher()
            response = m.search(
                self.photo_bgr,
                top_k=self.top_k,
                already_cropped=self.already_cropped,
            )
            self.finished.emit(response)
        except Exception as exc:
            logger.exception("SearchWorker error: %s", exc)
            self.error.emit(str(exc))


class IndexWorker(QThread):
    """Asynchronous worker for indexing catalog PDF folders."""

    started_indexing = Signal()
    progress = Signal(str, int, int, str)  # (current_file, page, total_pages, message)
    finished = Signal(object)  # Emits IndexStats
    error = Signal(str)

    def __init__(
        self,
        folders: List[Union[str, Path]],
        store: Optional[IndexStore] = None,
        cfg: EngineConfig = config,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.folders = folders
        self.store = store
        self.cfg = cfg
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Signals cancellation to the running indexing loop."""
        self._cancel_event.set()

    def run(self) -> None:
        self.started_indexing.emit()

        def _on_progress(cur_file: str, cur_p: int, tot_p: int, msg: str) -> None:
            self.progress.emit(cur_file, cur_p, tot_p, msg)

        try:
            stats = index_folders(
                folders=self.folders,
                progress_cb=_on_progress,
                cancel_event=self._cancel_event,
                store=self.store,
                cfg=self.cfg,
            )
            self.finished.emit(stats)
        except Exception as exc:
            logger.exception("IndexWorker error: %s", exc)
            self.error.emit(str(exc))
