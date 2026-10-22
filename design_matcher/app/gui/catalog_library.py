"""
CatalogLibraryPage — Fast Indexed Catalog Browser.

Features:
 - Paginated DB fetch (PAGE_SIZE tiles per page) – no full table scan on open.
 - Background thumbnail loading via QThreadPool – zero UI freeze.
 - Live search/filter by design code, filename, type (All / PDF / DXF).
 - Dual view: Gallery (thumbnail grid) and Files (table).
 - Load-More button to pull the next page without rebuilding the entire grid.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import (
    QRunnable,
    QThreadPool,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.gui.pages import PAGE_STYLESHEET, create_page_header

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background thumbnail loader
# ---------------------------------------------------------------------------

class _ThumbnailTask(QRunnable):
    """
    Loads a single thumbnail from disk off the GUI thread and delivers the
    result via a plain Python callback.  QTimer.singleShot(0) inside the
    callback hops the pixmap back to the main thread safely.
    """

    def __init__(
        self,
        tile_id: int,
        thumb_path: str,
        data_dir: str,
        callback: Callable[[int, QPixmap], None],
    ) -> None:
        super().__init__()
        self.tile_id = tile_id
        self.thumb_path = thumb_path
        self.data_dir = data_dir
        self.callback = callback
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            abs_path = (
                self.thumb_path
                if os.path.isabs(self.thumb_path)
                else os.path.join(self.data_dir, self.thumb_path)
            )
            pix = QPixmap(abs_path)
            if pix.isNull():
                pix = _placeholder()
        except Exception:
            pix = _placeholder()
        self.callback(self.tile_id, pix)


def _placeholder() -> QPixmap:
    pix = QPixmap(160, 150)
    pix.fill(QColor("#1E293B"))
    return pix


# ---------------------------------------------------------------------------
# Tile card widget
# ---------------------------------------------------------------------------

class _TileCard(QFrame):
    """168 x 210 px card: lazy thumbnail + design-code label + source badge."""

    clicked = Signal(dict)

    _W, _H = 168, 210
    _TW, _TH = 160, 150

    def __init__(self, tile_data: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tile = tile_data
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build()
        self.setStyleSheet(
            "QFrame { background:#1E293B; border:1px solid #334155; border-radius:8px; }"
            "QFrame:hover { border:1px solid #38BDF8; background:#243248; }"
        )

    def _build(self) -> None:
        lo = QVBoxLayout(self)
        lo.setContentsMargins(4, 4, 4, 6)
        lo.setSpacing(4)

        self._thumb = QLabel(self)
        self._thumb.setFixedSize(self._TW, self._TH)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet(
            "background:#0B1120; border-radius:4px; color:#334155; font-size:10px;"
        )
        self._thumb.setText("loading...")
        lo.addWidget(self._thumb, 0, Qt.AlignmentFlag.AlignHCenter)

        d_code = (
            self._tile.get("design_code")
            or self._tile.get("design_number")
            or ""
        )
        src = str(self._tile.get("source_type", "pdf")).upper()
        page = self._tile.get("page_number", "")
        display = str(d_code or f"{src} p.{page}")[:24]

        lbl = QLabel(display, self)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setMaximumWidth(self._W - 8)
        lbl.setStyleSheet(
            "color:#E2E8F0; font-size:10px; font-weight:600; background:transparent;"
        )
        lo.addWidget(lbl)

        badge_color = "#38BDF8" if src == "PDF" else "#A78BFA"
        badge = QLabel(src, self)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"color:{badge_color}; font-size:9px; font-weight:700; background:transparent;"
        )
        lo.addWidget(badge)

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(
            self._TW, self._TH,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb.setPixmap(scaled)
        self._thumb.setText("")

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._tile)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Flow-grid container (manual positioning, no Qt layout manager needed)
# ---------------------------------------------------------------------------

class _FlowGrid(QWidget):
    """Wraps fixed-size cards into responsive rows, recalculated on resize."""

    def __init__(self, h_gap: int = 10, v_gap: int = 10, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._h_gap = h_gap
        self._v_gap = v_gap
        self._cards: List[_TileCard] = []

    def add_card(self, card: _TileCard) -> None:
        card.setParent(self)
        card.show()
        self._cards.append(card)
        self._relayout()

    def clear_cards(self) -> None:
        for c in self._cards:
            c.hide()
            c.deleteLater()
        self._cards.clear()
        self.setMinimumHeight(self._v_gap * 2)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        if not self._cards:
            self.setMinimumHeight(self._v_gap * 2)
            return
        cw = self._cards[0].width()
        ch = self._cards[0].height()
        w = self.width()
        cols = max(1, (w - self._h_gap) // (cw + self._h_gap))
        for i, card in enumerate(self._cards):
            col = i % cols
            row = i // cols
            card.move(
                self._h_gap + col * (cw + self._h_gap),
                self._v_gap + row * (ch + self._v_gap),
            )
        total_rows = (len(self._cards) + cols - 1) // cols
        self.setMinimumHeight(self._v_gap + total_rows * (ch + self._v_gap))


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

class CatalogLibraryPage(QWidget):
    """
    Browseable gallery for all indexed PDF catalog tiles and DXF designs.

    Signals
    -------
    back_requested  -- user clicked "Back to Matcher"
    tile_selected   -- user clicked a tile card; carries tile metadata dict
    """

    back_requested = Signal()
    tile_selected = Signal(dict)

    PAGE_SIZE = 48   # tiles fetched per DB page-load

    def __init__(
        self,
        catalog_index: Any,
        dxf_db: Any,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.catalog_index = catalog_index
        self.dxf_db = dxf_db
        self._data_dir = str(getattr(catalog_index, "data_dir", "")) if catalog_index else ""

        self._offset = 0
        self._filter_type = "all"    # "all" | "pdf" | "dxf"
        self._search_text = ""
        self._card_map: Dict[int, _TileCard] = {}

        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(4)

        self.setStyleSheet(PAGE_STYLESHEET)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        hdr = create_page_header(
            "Catalog Library",
            subtitle="Browse all indexed tiles  PDF catalogs and DXF designs",
            on_back=self.back_requested.emit,
        )
        root.addWidget(hdr)

        # Toolbar
        toolbar = QFrame()
        toolbar.setObjectName("libToolbar")
        toolbar.setStyleSheet(
            "QFrame#libToolbar { background:#0F172A; border-bottom:1px solid #1E293B; }"
        )
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(20, 9, 20, 9)
        tb.setSpacing(10)

        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Search by design code, filename...")
        self._search_box.setFixedHeight(34)
        self._search_box.setStyleSheet(
            "QLineEdit { background:#1E293B; color:#F8FAFC; border:1px solid #334155;"
            " border-radius:6px; padding:0 10px; font-size:13px; }"
            "QLineEdit:focus { border-color:#38BDF8; }"
        )
        self._search_box.textChanged.connect(self._on_search_changed)
        tb.addWidget(self._search_box, 3)

        self._type_btns: Dict[str, QPushButton] = {}
        for label, key in [("All", "all"), ("PDF", "pdf"), ("DXF", "dxf")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(key == "all")
            btn.setFixedHeight(32)
            btn.setStyleSheet(
                "QPushButton { background:#1E293B; color:#94A3B8; border:1px solid #334155;"
                " border-radius:6px; padding:0 16px; font-size:12px; font-weight:600; }"
                "QPushButton:checked { background:#2563EB; color:#FFF; border-color:#3B82F6; }"
                "QPushButton:hover:!checked { background:#334155; color:#F8FAFC; }"
            )
            btn.clicked.connect(lambda _chk, k=key: self._set_type_filter(k))
            self._type_btns[key] = btn
            tb.addWidget(btn)

        tb.addStretch()
        self._stats_lbl = QLabel("", toolbar)
        self._stats_lbl.setStyleSheet("color:#64748B; font-size:12px; background:transparent;")
        tb.addWidget(self._stats_lbl)
        root.addWidget(toolbar)

        # View tabs
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabBar::tab { padding:5px 18px; font-size:12px; font-weight:600; }"
        )
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Tab 0 - Gallery
        gallery_host = QWidget()
        gv = QVBoxLayout(gallery_host)
        gv.setContentsMargins(0, 0, 0, 0)
        gv.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border:none; background:#0B1120; }")

        self._grid = _FlowGrid(h_gap=10, v_gap=10)
        self._grid.setStyleSheet("background:#0B1120;")
        self._scroll.setWidget(self._grid)
        gv.addWidget(self._scroll, 1)

        bbar = QFrame()
        bbar.setStyleSheet("QFrame { background:#0F172A; border-top:1px solid #1E293B; }")
        bb = QHBoxLayout(bbar)
        bb.setContentsMargins(20, 7, 20, 7)

        self._count_lbl = QLabel("", bbar)
        self._count_lbl.setStyleSheet("color:#64748B; font-size:12px;")
        bb.addWidget(self._count_lbl)
        bb.addStretch()

        self._more_btn = QPushButton("Load More Tiles")
        self._more_btn.setFixedHeight(32)
        self._more_btn.setStyleSheet(
            "QPushButton { background:#1E293B; color:#38BDF8; border:1px solid #334155;"
            " border-radius:6px; padding:0 18px; font-size:12px; font-weight:700; }"
            "QPushButton:hover { background:#2563EB; color:#FFF; }"
            "QPushButton:disabled { color:#334155; border-color:#1E293B; }"
        )
        self._more_btn.clicked.connect(self._load_next_page)
        bb.addWidget(self._more_btn)
        gv.addWidget(bbar)
        self._tabs.addTab(gallery_host, "Gallery")

        # Tab 1 - Files
        files_host = QWidget()
        fv = QVBoxLayout(files_host)
        fv.setContentsMargins(16, 12, 16, 12)

        self._files_tbl = QTableWidget()
        self._files_tbl.setColumnCount(5)
        self._files_tbl.setHorizontalHeaderLabels(
            ["File Name", "Type", "Pages / Designs", "Tiles", "Full Path"]
        )
        hdr = self._files_tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._files_tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._files_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._files_tbl.setAlternatingRowColors(True)
        self._files_tbl.setStyleSheet("QTableWidget { alternate-background-color:#131e30; }")
        fv.addWidget(self._files_tbl, 1)
        self._tabs.addTab(files_host, "Files")

        root.addWidget(self._tabs, 1)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Full reset: clear gallery, pagination = 0, load first page."""
        self._offset = 0
        self._card_map.clear()
        self._grid.clear_cards()
        self._more_btn.setEnabled(True)
        self._more_btn.setText("Load More Tiles")
        self._load_next_page()
        self._populate_files_table()
        self._update_stats()

    # ------------------------------------------------------------------
    # DB
    # ------------------------------------------------------------------

    def _where_clause(self) -> tuple:
        cond: List[str] = []
        params: List[Any] = []
        if self._filter_type == "pdf":
            cond.append("source_type = ?")
            params.append("pdf")
        elif self._filter_type == "dxf":
            cond.append("source_type = ?")
            params.append("dxf")
        if self._search_text:
            q = f"%{self._search_text}%"
            cond.append("(design_code LIKE ? OR design_number LIKE ? OR pdf_path LIKE ?)")
            params += [q, q, q]
        where = ("WHERE " + " AND ".join(cond)) if cond else ""
        return where, params

    def _fetch_page(self, offset: int, limit: int) -> List[dict]:
        if not (self.catalog_index and self.catalog_index.db):
            return []
        where, params = self._where_clause()
        sql = (
            "SELECT id, pdf_path, page_number, tile_index, thumbnail_path,"
            " design_code, design_number, source_type, crop_type, folder"
            f" FROM tiles {where} ORDER BY id LIMIT ? OFFSET ?"
        )
        params = list(params) + [limit, offset]
        try:
            with self.catalog_index.db._get_connection() as conn:
                return [dict(r) for r in conn.execute(sql, params).fetchall()]
        except Exception as exc:
            logger.warning("CatalogLibraryPage DB error: %s", exc)
            return []

    def _count_matching(self) -> int:
        if not (self.catalog_index and self.catalog_index.db):
            return 0
        where, params = self._where_clause()
        try:
            with self.catalog_index.db._get_connection() as conn:
                r = conn.execute(f"SELECT COUNT(*) FROM tiles {where}", list(params)).fetchone()
                return int(r[0]) if r else 0
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Gallery
    # ------------------------------------------------------------------

    def _load_next_page(self) -> None:
        rows = self._fetch_page(self._offset, self.PAGE_SIZE)
        if not rows:
            self._more_btn.setEnabled(False)
            self._more_btn.setText("All tiles loaded")
            self._update_count_lbl()
            return

        for tile in rows:
            card = _TileCard(tile)
            card.clicked.connect(self.tile_selected.emit)
            self._grid.add_card(card)
            tid = tile.get("id")
            if tid is not None:
                self._card_map[int(tid)] = card
                thumb = str(tile.get("thumbnail_path") or "")
                if thumb:
                    task = _ThumbnailTask(
                        tile_id=int(tid),
                        thumb_path=thumb,
                        data_dir=self._data_dir,
                        callback=self._deliver_thumbnail,
                    )
                    self._pool.start(task)

        self._offset += len(rows)
        has_more = len(rows) == self.PAGE_SIZE
        self._more_btn.setEnabled(has_more)
        self._more_btn.setText("Load More Tiles" if has_more else "All tiles loaded")
        self._update_count_lbl()

    def _deliver_thumbnail(self, tile_id: int, pixmap: QPixmap) -> None:
        """Thread-safe: hop pixmap delivery back to GUI thread via QTimer."""
        card = self._card_map.get(tile_id)
        if card is not None:
            QTimer.singleShot(0, lambda c=card, p=pixmap: c.set_thumbnail(p))

    def _update_count_lbl(self) -> None:
        total = self._count_matching()
        shown = len(self._card_map)
        self._count_lbl.setText(f"Showing {shown:,} of {total:,} tiles")

    def _update_stats(self) -> None:
        total = self._count_matching()
        self._stats_lbl.setText(f"{total:,} tiles in index")

    # ------------------------------------------------------------------
    # Filter / search
    # ------------------------------------------------------------------

    def _set_type_filter(self, key: str) -> None:
        self._filter_type = key
        for k, btn in self._type_btns.items():
            btn.setChecked(k == key)
        self._reset_and_reload()

    def _on_search_changed(self, text: str) -> None:
        self._search_text = text.strip()
        if not hasattr(self, "_debounce"):
            self._debounce = QTimer(self)
            self._debounce.setSingleShot(True)
            self._debounce.setInterval(300)
            self._debounce.timeout.connect(self._reset_and_reload)
        self._debounce.start()

    def _reset_and_reload(self) -> None:
        self._offset = 0
        self._card_map.clear()
        self._grid.clear_cards()
        self._more_btn.setEnabled(True)
        self._more_btn.setText("Load More Tiles")
        self._load_next_page()
        self._update_stats()

    # ------------------------------------------------------------------
    # Files table
    # ------------------------------------------------------------------

    def _populate_files_table(self) -> None:
        rows: List[tuple] = []

        if self.catalog_index and self.catalog_index.db:
            try:
                with self.catalog_index.db._get_connection() as conn:
                    cat_rows = conn.execute(
                        "SELECT filename, extension, page_count, metadata_status, visual_status, path"
                        " FROM catalogue_files ORDER BY filename"
                    ).fetchall()
                    if cat_rows:
                        for r in cat_rows:
                            r = dict(r) if not isinstance(r, dict) else r
                            ext = str(r.get("extension", "")).upper().replace(".", "")
                            v_st = r.get("visual_status", "pending")
                            m_st = r.get("metadata_status", "pending")
                            status_txt = "Ready" if v_st == "completed" else f"Meta:{m_st} | Vis:{v_st}"
                            rows.append((
                                r.get("filename", ""),
                                ext,
                                r.get("page_count", 0),
                                status_txt,
                                r.get("path", ""),
                            ))
                    else:
                        for r in conn.execute(
                            "SELECT pdf_path, page_count, tile_count"
                            " FROM indexed_files ORDER BY pdf_path"
                        ).fetchall():
                            r = dict(r) if not isinstance(r, dict) else r
                            p = Path(r.get("pdf_path", ""))
                            rows.append((
                                p.name, "PDF",
                                r.get("page_count", 0),
                                r.get("tile_count", 0),
                                str(p),
                            ))
            except Exception as e:
                logger.debug("Library files table PDF error: %s", e)

        if not rows and self.dxf_db and hasattr(self.dxf_db, "_get_connection"):
            try:
                with self.dxf_db._get_connection() as conn:
                    for r in conn.execute(
                        "SELECT file_path, design_count FROM indexed_dxf_files LIMIT 2000"
                    ).fetchall():
                        r = dict(r) if not isinstance(r, dict) else r
                        p = Path(r.get("file_path", ""))
                        cnt = r.get("design_count", 0)
                        rows.append((p.name, "DXF", cnt, f"{cnt * 2} tiles", str(p)))
            except Exception as e:
                logger.debug("Library files table DXF error: %s", e)

        self._files_tbl.setRowCount(len(rows))
        for i, (name, typ, pages, tiles, path) in enumerate(rows):
            self._files_tbl.setItem(i, 0, QTableWidgetItem(str(name)))
            self._files_tbl.setItem(i, 1, QTableWidgetItem(str(typ)))
            self._files_tbl.setItem(i, 2, QTableWidgetItem(str(pages)))
            self._files_tbl.setItem(i, 3, QTableWidgetItem(str(tiles)))
            self._files_tbl.setItem(i, 4, QTableWidgetItem(str(path)))
        self._files_tbl.resizeColumnsToContents()

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:
            self._populate_files_table()
