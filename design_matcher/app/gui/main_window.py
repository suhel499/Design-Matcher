"""
Main application window for Design Matcher.
PySide6 modern desktop UI integrating customer photo drop zone, mask preview,
multi-rotation vector search, result cards, and background catalog indexing.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from dxf_viewer import DxfPreviewPage, DxfViewerWidget
from app.core.image_crop import warp_four_points
from app.core.index import CatalogIndex, get_default_data_dir
from app.core.preprocess import (
    convert_photo_to_stencil_mask,
    detect_sheet_quadrilateral,
    detect_stencil_polarity,
    enhance_photo_contrast_and_sharpness,
    four_point_perspective_transform,
    normalize_stencil,
)
from app.core.licensing.manager import LicenseManager
from app.core.search import DesignSearcher, SearchResult
from app.gui.license_dialog import LicenseDialog, show_subscription_expired_dialog
from app.gui.widgets import (
    CornerWarpDialog,
    CropDialog,
    PhotoDropArea,
    ResultCardWidget,
    SettingsDialog,
    cv2_to_qpixmap,
)
from app.gui.workers import (
    DxfIndexWorker,
    DxfMatchWorker,
    IndexWorker,
    LicenseSyncWorker,
    PreprocessWorker,
    SearchWorker,
)
from engine.dxf_database import DxfDatabase
from engine.dxf_normalization import normalize_dxf_component
from engine.dxf_parser import parse_dxf_drawing

logger = logging.getLogger(__name__)

STYLE_SHEET = """
QMainWindow {
    background-color: #F8FAFC;
}

QLabel {
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1E293B;
}

#headerFrame {
    background-color: #0F172A;
    border-bottom: 1px solid #1E293B;
    padding: 10px 16px;
}

#headerTitle {
    color: #FFFFFF;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

#headerStats {
    color: #94A3B8;
    font-size: 12px;
}

#panelTitle {
    font-size: 14px;
    font-weight: 700;
    color: #334155;
    margin-bottom: 4px;
}

#photoDropArea {
    background-color: #FFFFFF;
    border: 2px dashed #CBD5E1;
    border-radius: 8px;
}

#photoDropArea:hover {
    border-color: #3B82F6;
    background-color: #F8FAFC;
}

#maskFrame {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 6px;
}

#resultCard {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin-bottom: 8px;
}

#resultCard:hover {
    border-color: #93C5FD;
    background-color: #FAFAFA;
}

QPushButton {
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 12px;
}

QPushButton#searchBtn {
    background-color: #2563EB;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 13px;
    padding: 10px;
    border-radius: 6px;
    border: none;
}

QPushButton#searchBtn:hover {
    background-color: #1D4ED8;
}

QPushButton#searchBtn:disabled {
    background-color: #94A3B8;
}

QPushButton#toolBtn {
    background-color: #FFFFFF;
    color: #334155;
    border: 1px solid #CBD5E1;
    border-radius: 5px;
    padding: 5px 8px;
    font-weight: 500;
    font-size: 11px;
}

QPushButton#toolBtn:hover {
    background-color: #F1F5F9;
    border-color: #94A3B8;
}

QPushButton#toolBtn:checked {
    background-color: #EFF6FF;
    color: #1D4ED8;
    border-color: #3B82F6;
    font-weight: 600;
}

QComboBox#modeCombo {
    background-color: #FFFFFF;
    color: #1E293B;
    border: 1px solid #CBD5E1;
    border-radius: 5px;
    padding: 4px 6px;
    font-size: 11px;
    font-weight: 500;
}

QPushButton#headerBtn {
    background-color: #1E293B;
    color: #F1F5F9;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 5px 12px;
    font-weight: 600;
    font-size: 12px;
}

QPushButton#headerBtn:hover {
    background-color: #334155;
}

QPushButton#filterTabBtn {
    background-color: #F1F5F9;
    color: #475569;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 600;
}

QPushButton#filterTabBtn:hover {
    background-color: #E2E8F0;
    color: #1E293B;
}

QPushButton#filterTabBtn:checked {
    background-color: #2563EB;
    color: #FFFFFF;
    border: 1px solid #2563EB;
}

QPushButton#mainNavTab {
    background-color: #1E293B;
    color: #94A3B8;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

QPushButton#mainNavTab:hover {
    background-color: #334155;
    color: #F8FAFC;
    border-color: #475569;
}

QPushButton#mainNavTab:checked {
    background-color: #2563EB;
    color: #FFFFFF;
    border: 1px solid #3B82F6;
    font-weight: 700;
}

QPushButton#mainNavTab:disabled,
QPushButton#mainNavTab[locked="true"] {
    background-color: #0F172A;
    color: #64748B;
    border: 1px dashed #334155;
}
"""


class MainWindow(QMainWindow):
    """Primary application window for Design Matcher desktop app."""

    def __init__(self, data_dir: Optional[Path | str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Design Matcher - CNC Cut-Design Search")
        self.resize(1320, 800)
        self.setMinimumSize(1000, 680)

        # Core engines
        self.catalog_index = CatalogIndex(data_dir=data_dir)
        self.searcher = DesignSearcher(self.catalog_index)
        self.dxf_db = DxfDatabase(self.catalog_index.data_dir / "dxf_library.db")
        self.license_manager = LicenseManager(data_dir=self.catalog_index.data_dir)

        # State
        self.search_mode: str = "mode_a"  # "mode_a": Photo/Stencil -> DXF, "mode_b": Reference DXF -> DXF
        self.reference_dxf_path: Optional[str] = None
        self.reference_dxf_components: List[Any] = []
        self.reference_selected_comp_idx: int = 0
        self.original_bgr: Optional[np.ndarray] = None
        self.raw_photo_bgr: Optional[np.ndarray] = None
        self.current_photo_bgr: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.mask_inverted: bool = False
        self.enhance_enabled: bool = True
        self.current_mode: str = "auto"
        self.index_worker: Optional[IndexWorker] = None
        self.search_worker: Optional[SearchWorker] = None
        self.preprocess_worker: Optional[PreprocessWorker] = None
        self.dxf_index_worker: Optional[DxfIndexWorker] = None
        self.dxf_match_worker: Optional[DxfMatchWorker] = None
        self.license_sync_worker: Optional[LicenseSyncWorker] = None
        self.license_sync_timer = QTimer(self)
        self.license_sync_timer.timeout.connect(self._sync_license_with_server)
        self.license_sync_timer.start(30 * 60 * 1000)  # background check every 30 mins

        # Viewer & Results state
        self.dxf_viewer_window: Optional[Any] = None
        self.result_cards: List[ResultCardWidget] = []
        self.last_dxf_results: List[SearchResult] = []
        self.last_pdf_results: List[SearchResult] = []
        self.current_filter_mode: str = "all"

        self._setup_ui()
        self.update_catalog_stats()
        self._update_license_indicator()
        self._check_license_on_startup()

    def _setup_ui(self) -> None:
        self.setStyleSheet(STYLE_SHEET)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Top Header Bar
        header = QFrame(self)
        header.setObjectName("headerFrame")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 8, 16, 8)

        title_vbox = QVBoxLayout()
        title_row = QHBoxLayout()
        title_label = QLabel("DESIGN MATCHER", header)
        title_label.setObjectName("headerTitle")
        title_row.addWidget(title_label)
        title_row.addSpacing(16)

        # Top Navigation Tabs: 1. Matcher  2. DXF Preview
        self.nav_tab_matcher = QPushButton("🔍 1. Matcher", header)
        self.nav_tab_matcher.setObjectName("mainNavTab")
        self.nav_tab_matcher.setCheckable(True)
        self.nav_tab_matcher.setChecked(True)
        self.nav_tab_matcher.setCursor(Qt.CursorShape.PointingHandCursor)

        self.nav_tab_dxf_preview = QPushButton("📐 2. DXF Preview", header)
        self.nav_tab_dxf_preview.setObjectName("mainNavTab")
        self.nav_tab_dxf_preview.setCheckable(True)
        self.nav_tab_dxf_preview.setChecked(False)
        self.nav_tab_dxf_preview.setCursor(Qt.CursorShape.PointingHandCursor)
        self.nav_tab_dxf_preview.installEventFilter(self)

        self.nav_button_group = QButtonGroup(self)
        self.nav_button_group.addButton(self.nav_tab_matcher, 0)
        self.nav_button_group.addButton(self.nav_tab_dxf_preview, 1)
        self.nav_tab_matcher.clicked.connect(lambda *_: self._on_main_tab_changed(0))
        self.nav_tab_dxf_preview.clicked.connect(self._on_dxf_preview_tab_clicked)

        # Aliases for convenience
        self.tab_matcher_btn = self.nav_tab_matcher
        self.tab_preview_btn = self.nav_tab_dxf_preview

        title_row.addWidget(self.nav_tab_matcher)
        title_row.addWidget(self.nav_tab_dxf_preview)
        title_row.addStretch()
        title_vbox.addLayout(title_row)

        self.stats_label = QLabel("Connecting to catalog database...", header)
        self.stats_label.setObjectName("headerStats")
        title_vbox.addWidget(self.stats_label)
        header_layout.addLayout(title_vbox)

        header_layout.addStretch()

        # Text Shortcut Search Box for Design Code (PD 010, etc.)
        code_search_box = QHBoxLayout()
        code_search_box.setSpacing(6)
        self.code_search_input = QLineEdit(header)
        self.code_search_input.setPlaceholderText("Search Design Code (e.g. PD 010)...")
        self.code_search_input.setStyleSheet(
            "QLineEdit { background: #1E293B; color: #F8FAFC; border: 1px solid #475569; "
            "border-radius: 4px; padding: 5px 10px; font-size: 12px; min-width: 210px; } "
            "QLineEdit:focus { border: 1px solid #7C3AED; background: #0F172A; }"
        )
        self.code_search_input.returnPressed.connect(self._search_by_design_code)
        code_search_box.addWidget(self.code_search_input)

        code_search_btn = QPushButton("Search Code", header)
        code_search_btn.setObjectName("headerBtn")
        code_search_btn.setStyleSheet(
            "QPushButton { background: #7C3AED; color: white; border: none; border-radius: 4px; "
            "padding: 5px 12px; font-weight: 600; font-size: 11px; } "
            "QPushButton:hover { background: #6D28D9; }"
        )
        code_search_btn.clicked.connect(self._search_by_design_code)
        code_search_box.addWidget(code_search_btn)
        header_layout.addLayout(code_search_box)

        # Action buttons in header
        dxf_index_btn = QPushButton("Index DXF Library", header)
        dxf_index_btn.setObjectName("headerBtn")
        dxf_index_btn.setStyleSheet(
            "QPushButton { background: #5B21B6; color: white; border: 1px solid #7C3AED; "
            "padding: 5px 12px; font-weight: 600; font-size: 12px; border-radius: 5px; } "
            "QPushButton:hover { background: #4C1D95; }"
        )
        dxf_index_btn.clicked.connect(self._start_dxf_indexing_flow)
        header_layout.addWidget(dxf_index_btn)

        index_btn = QPushButton("Index Catalogs", header)
        index_btn.setObjectName("headerBtn")
        index_btn.clicked.connect(self._start_indexing_flow)
        header_layout.addWidget(index_btn)

        settings_btn = QPushButton("Settings", header)
        settings_btn.setObjectName("headerBtn")
        settings_btn.clicked.connect(self._open_settings)
        header_layout.addWidget(settings_btn)

        self.license_btn = QPushButton("🔑 License", header)
        self.license_btn.setObjectName("headerBtn")
        self.license_btn.setStyleSheet(
            "QPushButton { background: #1E293B; color: #38BDF8; border: 1px solid #0284C7; "
            "padding: 5px 12px; font-weight: 600; font-size: 12px; border-radius: 5px; } "
            "QPushButton:hover { background: #0369A1; color: white; }"
        )
        self.license_btn.clicked.connect(self._open_license_dialog)
        header_layout.addWidget(self.license_btn)

        main_layout.addWidget(header)

        # 2. Main Body Splitter (Left: Photo & Controls, Right: Results)
        splitter = QSplitter(Qt.Orientation.Horizontal, central_widget)
        splitter.setHandleWidth(1)

        # --- LEFT PANEL: Photo Drop, Tools & Stencil Mask ---
        left_widget = QWidget(splitter)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(18, 16, 12, 16)
        left_layout.setSpacing(10)

        # Mode A / Mode B Tab Switcher
        mode_btn_box = QHBoxLayout()
        mode_btn_box.setSpacing(6)
        self.mode_a_tab_btn = QPushButton("📷 Mode A: Photo / Stencil", left_widget)
        self.mode_a_tab_btn.setObjectName("filterTabBtn")
        self.mode_a_tab_btn.setCheckable(True)
        self.mode_a_tab_btn.setChecked(True)
        self.mode_a_tab_btn.clicked.connect(lambda: self._set_search_mode("mode_a"))
        mode_btn_box.addWidget(self.mode_a_tab_btn)

        self.mode_b_tab_btn = QPushButton("📐 Mode B: Reference DXF", left_widget)
        self.mode_b_tab_btn.setObjectName("filterTabBtn")
        self.mode_b_tab_btn.setCheckable(True)
        self.mode_b_tab_btn.setChecked(False)
        self.mode_b_tab_btn.clicked.connect(lambda: self._set_search_mode("mode_b"))
        mode_btn_box.addWidget(self.mode_b_tab_btn)
        left_layout.addLayout(mode_btn_box)

        self.left_panel_title = QLabel("CUSTOMER PHOTO / STENCIL", left_widget)
        self.left_panel_title.setObjectName("panelTitle")
        left_layout.addWidget(self.left_panel_title)

        # Drop area
        self.drop_area = PhotoDropArea(left_widget)
        self.drop_area.photo_loaded.connect(self._on_photo_loaded)
        left_layout.addWidget(self.drop_area, 1)

        # Multi-component panel selector for reference DXF
        self.ref_comp_container = QWidget(left_widget)
        ref_comp_layout = QHBoxLayout(self.ref_comp_container)
        ref_comp_layout.setContentsMargins(0, 0, 0, 0)
        ref_comp_layout.addWidget(QLabel("Design Panel:", self.ref_comp_container))
        self.ref_comp_combo = QComboBox(self.ref_comp_container)
        self.ref_comp_combo.setObjectName("modeCombo")
        self.ref_comp_combo.currentIndexChanged.connect(self._on_reference_component_changed)
        ref_comp_layout.addWidget(self.ref_comp_combo, 1)
        self.ref_comp_container.setVisible(False)
        left_layout.addWidget(self.ref_comp_container)

        # Photo tools container (tools rows 1 and 2)
        self.photo_tools_container = QWidget(left_widget)
        photo_tools_vbox = QVBoxLayout(self.photo_tools_container)
        photo_tools_vbox.setContentsMargins(0, 0, 0, 0)
        photo_tools_vbox.setSpacing(8)

        # Tools row 1: Geometry transformations
        tools_layout = QHBoxLayout()
        tools_layout.setSpacing(6)

        self.rotate_btn = QPushButton("Rotate 90°", self.photo_tools_container)
        self.rotate_btn.setObjectName("toolBtn")
        self.rotate_btn.clicked.connect(self._rotate_photo)
        self.rotate_btn.setEnabled(False)
        tools_layout.addWidget(self.rotate_btn)

        self.warp_btn = QPushButton("4-Corner Warp", self.photo_tools_container)
        self.warp_btn.setObjectName("toolBtn")
        self.warp_btn.clicked.connect(self._open_warp_dialog)
        self.warp_btn.setEnabled(False)
        tools_layout.addWidget(self.warp_btn)

        self.crop_btn = QPushButton("Crop Rect", self.photo_tools_container)
        self.crop_btn.setObjectName("toolBtn")
        self.crop_btn.clicked.connect(self._open_crop_dialog)
        self.crop_btn.setEnabled(False)
        tools_layout.addWidget(self.crop_btn)

        self.clear_btn = QPushButton("Clear", self.photo_tools_container)
        self.clear_btn.setObjectName("toolBtn")
        self.clear_btn.clicked.connect(self._clear_photo)
        self.clear_btn.setEnabled(False)
        tools_layout.addWidget(self.clear_btn)

        photo_tools_vbox.addLayout(tools_layout)

        # Tools row 2: Processing Mode & Deblur Enhancement
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(6)

        mode_lbl = QLabel("Mode:", self.photo_tools_container)
        mode_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #475569;")
        mode_layout.addWidget(mode_lbl)

        self.mode_combo = QComboBox(self.photo_tools_container)
        self.mode_combo.setObjectName("modeCombo")
        self.mode_combo.addItems([
            "Auto (Smart - All Backgrounds)",
            "Raw Metal Sheet",
            "Installed Gate / Product",
            "Backlit Silhouette",
            "Mixed Lighting (Sky & Shadows)",
            "Blurry / WhatsApp Photo",
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo, 1)

        self.enhance_btn = QPushButton("Enhance Details", self.photo_tools_container)
        self.enhance_btn.setObjectName("toolBtn")
        self.enhance_btn.setCheckable(True)
        self.enhance_btn.setChecked(True)
        self.enhance_btn.clicked.connect(self._toggle_enhance)
        self.enhance_btn.setEnabled(False)
        mode_layout.addWidget(self.enhance_btn)

        self.invert_btn = QPushButton("Invert", self.photo_tools_container)
        self.invert_btn.setObjectName("toolBtn")
        self.invert_btn.clicked.connect(self._toggle_invert_mask)
        self.invert_btn.setEnabled(False)
        mode_layout.addWidget(self.invert_btn)

        photo_tools_vbox.addLayout(mode_layout)
        left_layout.addWidget(self.photo_tools_container)

        # Stencil Mask Preview header with Polarity Badge
        mask_header = QHBoxLayout()
        mask_title = QLabel("NORMALIZED STENCIL MASK", left_widget)
        mask_title.setObjectName("panelTitle")
        mask_header.addWidget(mask_title)
        mask_header.addStretch()

        self.polarity_badge = QLabel("Polarity: Auto", left_widget)
        self.polarity_badge.setStyleSheet(
            "background: #F1F5F9; color: #64748B; border: 1px solid #CBD5E1; "
            "font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px;"
        )
        mask_header.addWidget(self.polarity_badge)
        left_layout.addLayout(mask_header)

        self.mask_frame = QFrame(left_widget)
        self.mask_frame.setObjectName("maskFrame")
        self.mask_frame.setFixedHeight(170)
        mask_frame_layout = QVBoxLayout(self.mask_frame)
        mask_frame_layout.setContentsMargins(6, 6, 6, 6)

        self.mask_label = QLabel(self.mask_frame)
        self.mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_label.setText("Load a photo or DXF above to generate mask preview")
        self.mask_label.setStyleSheet("color: #64748B; font-size: 11px;")
        mask_frame_layout.addWidget(self.mask_label)
        left_layout.addWidget(self.mask_frame)

        # Primary Search Button
        self.search_btn = QPushButton("SEARCH DXF LIBRARY (FROM STENCIL)", left_widget)
        self.search_btn.setObjectName("searchBtn")
        self.search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.search_btn.setEnabled(False)
        self.search_btn.clicked.connect(self._start_search)
        left_layout.addWidget(self.search_btn)

        splitter.addWidget(left_widget)

        # --- RIGHT PANEL: Results ---
        right_widget = QWidget(splitter)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(12, 16, 18, 16)
        right_layout.setSpacing(10)

        results_header_layout = QHBoxLayout()
        results_title = QLabel("MATCHED CATALOG DESIGNS", right_widget)
        results_title.setObjectName("panelTitle")
        results_header_layout.addWidget(results_title)

        self.results_count_label = QLabel("0 results", right_widget)
        self.results_count_label.setStyleSheet("color: #64748B; font-size: 12px;")
        results_header_layout.addWidget(self.results_count_label)
        results_header_layout.addStretch()
        right_layout.addLayout(results_header_layout)

        # Quick format filter tabs: All | Top 5 DXF | Top 5 PDF
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(6)

        self.tab_all_btn = QPushButton("All Matches (Top 5 DXF + Top 5 PDF)", right_widget)
        self.tab_all_btn.setObjectName("filterTabBtn")
        self.tab_all_btn.setCheckable(True)
        self.tab_all_btn.setChecked(True)
        self.tab_all_btn.clicked.connect(lambda: self._set_results_filter("all"))
        filter_bar.addWidget(self.tab_all_btn)

        self.tab_dxf_btn = QPushButton("📐 Top 5 DXF (Vectors)", right_widget)
        self.tab_dxf_btn.setObjectName("filterTabBtn")
        self.tab_dxf_btn.setCheckable(True)
        self.tab_dxf_btn.clicked.connect(lambda: self._set_results_filter("dxf"))
        filter_bar.addWidget(self.tab_dxf_btn)

        self.tab_pdf_btn = QPushButton("📄 Top 5 PDF Catalogs", right_widget)
        self.tab_pdf_btn.setObjectName("filterTabBtn")
        self.tab_pdf_btn.setCheckable(True)
        self.tab_pdf_btn.clicked.connect(lambda: self._set_results_filter("pdf"))
        filter_bar.addWidget(self.tab_pdf_btn)

        filter_bar.addStretch()
        right_layout.addLayout(filter_bar)

        # Results scroll area
        self.scroll_area = QScrollArea(right_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("background: transparent;")

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(8)

        # Placeholder message
        self.placeholder_label = QLabel(
            "Drop a customer photo on the left and click 'Search Closest Design' to find matching catalog panels.",
            self.results_container,
        )
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet("color: #94A3B8; font-size: 13px; margin-top: 80px;")
        self.results_layout.addWidget(self.placeholder_label)
        self.results_layout.addStretch()

        self.scroll_area.setWidget(self.results_container)
        right_layout.addWidget(self.scroll_area)

        splitter.addWidget(right_widget)

        # --- 3. RIGHT PREVIEW PANE: In-App DXF / PDF Live Viewer ---
        self.preview_panel = QWidget(splitter)
        preview_layout = QVBoxLayout(self.preview_panel)
        preview_layout.setContentsMargins(8, 16, 16, 16)
        preview_layout.setSpacing(8)

        preview_header = QHBoxLayout()
        preview_title = QLabel("IN-APP DRAWING PREVIEW", self.preview_panel)
        preview_title.setObjectName("panelTitle")
        preview_header.addWidget(preview_title)
        preview_header.addStretch()
        preview_layout.addLayout(preview_header)

        self.preview_stack = QStackedWidget(self.preview_panel)
        self.preview_stack.setFrameShape(QFrame.Shape.NoFrame)

        # Page 0: Placeholder
        placeholder_frame = QFrame(self.preview_stack)
        placeholder_frame.setStyleSheet("background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 8px;")
        pl_layout = QVBoxLayout(placeholder_frame)
        pl_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl_icon = QLabel("📐", placeholder_frame)
        pl_icon.setStyleSheet("font-size: 36px;")
        pl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl_layout.addWidget(pl_icon)
        pl_text = QLabel(
            "Select any DXF or PDF result to view it here.\n\n"
            "• Crisp vector zoom & pan\n"
            "• Matched design auto-highlight\n"
            "• Interactive dimension measurement\n"
            "• Layer visibility controls",
            placeholder_frame,
        )
        pl_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl_text.setStyleSheet("color: #64748B; font-size: 12px; line-height: 1.5;")
        pl_layout.addWidget(pl_text)
        self.preview_stack.addWidget(placeholder_frame)

        # Page 1: In-App DXF Viewer
        from dxf_viewer import DxfViewerWidget
        self.embedded_dxf_viewer = DxfViewerWidget(self.preview_stack)
        self.embedded_dxf_viewer.expand_window_signal.connect(self._open_standalone_dxf_viewer_from_embedded)
        self.preview_stack.addWidget(self.embedded_dxf_viewer)

        # Page 2: PDF Page Preview
        self.pdf_preview_frame = QFrame(self.preview_stack)
        self.pdf_preview_frame.setStyleSheet("background: #0F172A; border-radius: 8px; border: 1px solid #334155;")
        pdf_prev_layout = QVBoxLayout(self.pdf_preview_frame)
        self.pdf_preview_label = QLabel(self.pdf_preview_frame)
        self.pdf_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pdf_prev_layout.addWidget(self.pdf_preview_label)
        self.preview_stack.addWidget(self.pdf_preview_frame)

        preview_layout.addWidget(self.preview_stack)
        splitter.addWidget(self.preview_panel)

        # Proportions: Photo 28%, Results 36%, Preview 36%
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 5)

        # Central Stacked Widget:
        # Page 0: Matcher (Design Matcher working properly)
        # Page 1: DXF Preview (Folders list on left side, detected designs, vector canvas, tools)
        self.main_stack = QStackedWidget(central_widget)

        # Tab 1: Matcher Page
        self.matcher_page = QWidget(self.main_stack)
        matcher_layout = QVBoxLayout(self.matcher_page)
        matcher_layout.setContentsMargins(0, 0, 0, 0)
        matcher_layout.setSpacing(0)
        matcher_layout.addWidget(splitter, 1)
        self.main_stack.addWidget(self.matcher_page)

        # Tab 2: DXF Preview Page
        self.dxf_preview_page = DxfPreviewPage(parent=self.main_stack)
        self.main_stack.addWidget(self.dxf_preview_page)

        main_layout.addWidget(self.main_stack, 1)

        # 3. Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Select or drop a customer photo to begin.")

    def update_catalog_stats(self) -> None:
        """Refreshes the header statistics bar with DXF and PDF breakdown."""
        dxf_stats = self.dxf_db.get_stats()
        dxf_files = dxf_stats.get("total_files", 0)
        dxf_comps = dxf_stats.get("total_components", 0)
        pdf_files = self.catalog_index.db.get_total_file_count()
        pdf_vectors = self.catalog_index.total_vectors
        self.stats_label.setText(
            f"DXF Library: {dxf_files:,} files ({dxf_comps:,} components)  |  "
            f"PDF Catalog: {pdf_files:,} catalogs ({pdf_vectors:,} tiles)"
        )

    def _set_search_mode(self, mode: str) -> None:
        """Switches between Mode A (Photo/Stencil) and Mode B (Reference DXF)."""
        self.search_mode = mode
        self.mode_a_tab_btn.setChecked(mode == "mode_a")
        self.mode_b_tab_btn.setChecked(mode == "mode_b")

        if mode == "mode_a":
            self.left_panel_title.setText("CUSTOMER PHOTO / STENCIL")
            self.photo_tools_container.setVisible(True)
            self.ref_comp_container.setVisible(False)
            self.search_btn.setText("SEARCH DXF LIBRARY (FROM STENCIL)")
            self.search_btn.setEnabled(self.current_mask is not None)
            self.status_bar.showMessage("Switched to Mode A: Photo / PNG Stencil -> DXF Search.")
        else:
            self.left_panel_title.setText("REFERENCE DXF DESIGN")
            self.photo_tools_container.setVisible(False)
            if len(self.reference_dxf_components) > 1:
                self.ref_comp_container.setVisible(True)
            self.search_btn.setText("SEARCH DXF LIBRARY (FROM REFERENCE DXF)")
            self.search_btn.setEnabled(bool(self.reference_dxf_path))
            self.status_bar.showMessage("Switched to Mode B: Reference DXF -> DXF Search.")

    def _load_reference_dxf(self, file_path: str) -> None:
        """Parses reference DXF CAD file, extracts components, and prepares Mode B search."""
        p = Path(file_path).resolve()
        if not p.is_file():
            return

        if self.search_mode != "mode_b":
            self._set_search_mode("mode_b")

        self.status_bar.showMessage(f"Parsing reference DXF: {p.name}...")
        try:
            parsed = parse_dxf_drawing(p)
            self.reference_dxf_path = str(p)
            self.reference_dxf_components = parsed.components
            self.reference_selected_comp_idx = 0

            # Update component dropdown
            self.ref_comp_combo.blockSignals(True)
            self.ref_comp_combo.clear()
            for comp in parsed.components:
                self.ref_comp_combo.addItem(f"{comp.name} ({comp.entity_count} entities)")
            self.ref_comp_combo.setCurrentIndex(0)
            self.ref_comp_combo.blockSignals(False)
            self.ref_comp_container.setVisible(len(parsed.components) > 1)

            # Normalize selected component and display mask preview
            if parsed.components:
                norm_comp = normalize_dxf_component(parsed.components[0])
                self.current_mask = norm_comp.canonical_mask
                self.polarity_badge.setText("Vector Geometry")
                self.polarity_badge.setStyleSheet(
                    "background: #EDE9FE; color: #6D28D9; border: 1px solid #DDD6FE; "
                    "font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px;"
                )
                self._display_mask_preview()

            self.clear_btn.setEnabled(True)
            self.search_btn.setEnabled(True)
            self.search_btn.setText("SEARCH DXF LIBRARY (FROM REFERENCE DXF)")
            self.status_bar.showMessage(
                f"Reference DXF ready: {p.name} ({len(parsed.components)} component(s), {parsed.total_entity_count} entities)."
            )
        except Exception as exc:
            logger.exception("Error parsing reference DXF: %s", exc)
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.warning(self, "Invalid DXF File", f"Unable to parse reference DXF CAD file:\n\n{exc}")
            self.status_bar.showMessage(f"DXF parse error: {exc}")

    def _on_reference_component_changed(self, idx: int) -> None:
        """Selects a specific sub-panel component from reference DXF."""
        if 0 <= idx < len(self.reference_dxf_components):
            self.reference_selected_comp_idx = idx
            comp = self.reference_dxf_components[idx]
            norm_comp = normalize_dxf_component(comp)
            self.current_mask = norm_comp.canonical_mask
            self._display_mask_preview()
            self.status_bar.showMessage(f"Selected reference design: {comp.name}")

    def _on_photo_loaded(self, bgr_img: np.ndarray, file_path: str) -> None:
        self.original_bgr = bgr_img.copy() if bgr_img is not None else None
        self.raw_photo_bgr = bgr_img.copy() if bgr_img is not None else None
        self.current_photo_bgr = bgr_img.copy() if bgr_img is not None else None
        fname = Path(file_path).name if file_path else "Custom file"

        if file_path and Path(file_path).suffix.lower() == ".dxf":
            self._load_reference_dxf(file_path)
            return

        if self.current_photo_bgr is None:
            return

        if self.search_mode != "mode_a":
            self._set_search_mode("mode_a")

        self.status_bar.showMessage(f"Loaded photo: {fname}. Normalizing stencil mask...")
        self.rotate_btn.setEnabled(True)
        self.warp_btn.setEnabled(True)
        self.crop_btn.setEnabled(True)
        self.enhance_btn.setEnabled(True)
        self.invert_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self._start_preprocess_worker(self.current_photo_bgr, crop=True)

    def _clear_photo(self) -> None:
        """Clears the loaded photo or DXF and resets mask preview and controls."""
        if self.preprocess_worker is not None and self.preprocess_worker.isRunning():
            self.preprocess_worker.quit()
            self.preprocess_worker.wait()
        self.original_bgr = None
        self.raw_photo_bgr = None
        self.current_photo_bgr = None
        self.current_mask = None
        self.mask_inverted = False
        self.reference_dxf_path = None
        self.reference_dxf_components = []
        self.reference_selected_comp_idx = 0
        self.ref_comp_container.setVisible(False)
        self.drop_area.clear_image()
        self.mask_label.clear()
        self.mask_label.setText("Load a photo or DXF above to generate mask preview")
        self.polarity_badge.setText("Polarity: Auto")
        self.polarity_badge.setStyleSheet(
            "background: #F1F5F9; color: #64748B; border: 1px solid #CBD5E1; "
            "font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px;"
        )
        self.rotate_btn.setEnabled(False)
        self.warp_btn.setEnabled(False)
        self.crop_btn.setEnabled(False)
        self.enhance_btn.setEnabled(False)
        self.invert_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.search_btn.setEnabled(False)

        # Clear existing result cards
        self.result_cards.clear()
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
        self.results_count_label.setText("0 matches")

        if hasattr(self, "preview_stack"):
            self.preview_stack.setCurrentIndex(0)
        if hasattr(self, "embedded_dxf_viewer"):
            self.embedded_dxf_viewer.scene.clear()

        self.status_bar.showMessage("Photo cleared. Drag or select a customer photo to begin.")

    def _rotate_photo(self) -> None:
        if self.current_photo_bgr is None:
            return
        self.current_photo_bgr = cv2.rotate(self.current_photo_bgr, cv2.ROTATE_90_CLOCKWISE)
        if self.raw_photo_bgr is not None:
            self.raw_photo_bgr = cv2.rotate(self.raw_photo_bgr, cv2.ROTATE_90_CLOCKWISE)
        if self.original_bgr is not None:
            self.original_bgr = cv2.rotate(self.original_bgr, cv2.ROTATE_90_CLOCKWISE)
        self.drop_area.set_image(self.current_photo_bgr)
        self._update_mask_preview()

    def _open_warp_dialog(self) -> None:
        src = getattr(self, "original_bgr", None)
        if src is None:
            src = getattr(self, "raw_photo_bgr", None)
        if src is None:
            src = self.current_photo_bgr
        if src is None:
            return
        dlg = CornerWarpDialog(
            src,
            parent=self,
            preset_mode_idx=self.mode_combo.currentIndex(),
            is_inverted=self.mask_inverted,
            enhance=self.enhance_enabled,
        )
        if dlg.exec() and dlg.cropped_bgr is not None:
            self.current_photo_bgr = dlg.cropped_bgr
            self.mask_inverted = dlg.is_inverted
            self.enhance_enabled = dlg.enhance_check.isChecked()
            self.enhance_btn.setChecked(self.enhance_enabled)

            # Sync mode combo without firing redundant recalculations
            new_mode_idx = dlg.mode_combo.currentIndex()
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(new_mode_idx)
            self.mode_combo.blockSignals(False)
            mode_map = {
                0: "auto",
                1: "raw_sheet",
                2: "installed_product",
                3: "backlit",
                4: "mixed_lighting",
                5: "blurry",
            }
            self.current_mode = mode_map.get(new_mode_idx, "auto")

            self.drop_area.set_image(self.current_photo_bgr)
            # Pipeline order: feed sharp full-res crop to normalize_stencil with crop=False
            self._start_preprocess_worker(self.current_photo_bgr, crop=False)
            self.status_bar.showMessage("4-corner perspective warp applied with sub-pixel precision!")
            if getattr(dlg, "trigger_search", False):
                self._start_search()

    def _open_crop_dialog(self) -> None:
        src = getattr(self, "original_bgr", None)
        if src is None:
            src = getattr(self, "raw_photo_bgr", None)
        if src is None:
            src = self.current_photo_bgr
        if src is None:
            return
        dlg = CropDialog(
            src,
            parent=self,
            preset_mode_idx=self.mode_combo.currentIndex(),
            is_inverted=self.mask_inverted,
            enhance=self.enhance_enabled,
        )
        if dlg.exec() and dlg.cropped_bgr is not None:
            self.current_photo_bgr = dlg.cropped_bgr
            self.mask_inverted = dlg.is_inverted
            self.enhance_enabled = dlg.enhance_check.isChecked()
            self.enhance_btn.setChecked(self.enhance_enabled)

            # Sync mode combo without firing redundant recalculations
            new_mode_idx = dlg.mode_combo.currentIndex()
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(new_mode_idx)
            self.mode_combo.blockSignals(False)
            mode_map = {
                0: "auto",
                1: "raw_sheet",
                2: "installed_product",
                3: "backlit",
                4: "mixed_lighting",
                5: "blurry",
            }
            self.current_mode = mode_map.get(new_mode_idx, "auto")

            self.drop_area.set_image(self.current_photo_bgr)
            # Pipeline order: feed sharp full-res crop to normalize_stencil with crop=False
            self._start_preprocess_worker(self.current_photo_bgr, crop=False)
            self.status_bar.showMessage("Panel cropped losslessly from original image!")
            if getattr(dlg, "trigger_search", False):
                self._start_search()

    def _on_mode_changed(self, idx: int) -> None:
        mode_map = {
            0: "auto",
            1: "raw_sheet",
            2: "installed_product",
            3: "backlit",
            4: "mixed_lighting",
            5: "blurry",
        }
        self.current_mode = mode_map.get(idx, "auto")
        if self.current_mode == "blurry":
            self.enhance_btn.setChecked(True)
            self.enhance_enabled = True
        self._update_mask_preview()
        self.status_bar.showMessage(f"Processing mode switched to: {self.mode_combo.currentText()}")

    def _toggle_enhance(self) -> None:
        self.enhance_enabled = self.enhance_btn.isChecked()
        self._update_mask_preview()
        status = "Deblur & detail enhancement enabled" if self.enhance_enabled else "Deblur & enhancement disabled"
        self.status_bar.showMessage(status)

    def _toggle_invert_mask(self) -> None:
        self.mask_inverted = not self.mask_inverted
        if self.current_mask is not None:
            self.current_mask = cv2.bitwise_not(self.current_mask)
            is_normal_pol = detect_stencil_polarity(self.current_mask)
            self.polarity_badge.setText("Metal: Solid (White)" if is_normal_pol else "Metal: Dark (Inverted)")
            self._display_mask_preview()
        else:
            self._update_mask_preview(recalculate=True)

    def _start_preprocess_worker(self, bgr: Optional[np.ndarray], crop: bool = True) -> None:
        """Runs stencil mask normalization in a background QThread to keep the UI fully responsive."""
        if bgr is None or bgr.size == 0:
            return

        # In headless / offscreen test mode, run synchronously
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            mask, info = normalize_stencil(bgr, crop=crop)
            self._on_preprocess_finished(mask, info)
            return

        if self.preprocess_worker is not None and self.preprocess_worker.isRunning():
            self.preprocess_worker.quit()
            self.preprocess_worker.wait()

        self.mask_label.setText("Normalizing stencil mask...")
        self.search_btn.setEnabled(False)
        self.status_bar.showMessage("Normalizing stencil mask in background...")

        self.preprocess_worker = PreprocessWorker(bgr, crop=crop, parent=self)
        self.preprocess_worker.finished_signal.connect(self._on_preprocess_finished)
        self.preprocess_worker.error_signal.connect(self._on_preprocess_error)
        self.preprocess_worker.start()

    def _on_preprocess_finished(self, mask: np.ndarray, info: dict) -> None:
        self.current_mask = mask
        is_inv = info.get("polarity_inverted", False)
        self.auto_polarity_inverted = is_inv
        if not is_inv:
            self.polarity_badge.setText("Metal: Solid (White)")
            self.polarity_badge.setStyleSheet(
                "background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; "
                "font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px;"
            )
        else:
            self.polarity_badge.setText("Metal: Inverted to White")
            self.polarity_badge.setStyleSheet(
                "background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; "
                "font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px;"
            )
        self._display_mask_preview()
        self.search_btn.setEnabled(True)
        self.status_bar.showMessage("Stencil mask ready. Click 'SEARCH CLOSEST DESIGN' to find matches.")

    def _on_preprocess_error(self, err_msg: str) -> None:
        self.status_bar.showMessage(f"Mask normalization error: {err_msg}")
        self.mask_label.setText(f"Error normalizing mask: {err_msg}")
        self.search_btn.setEnabled(True)

    def _display_mask_preview(self) -> None:
        """Displays current_mask scaled to fit inside preview keeping aspect ratio."""
        if self.current_mask is None or self.current_mask.size == 0:
            self.mask_label.clear()
            self.mask_label.setText("Load a photo above to generate mask preview")
            return

        gray = np.ascontiguousarray(self.current_mask)
        mh, mw = gray.shape
        # Convert uint8 array to QImage Format_Grayscale8
        q_img = QImage(gray.data, mw, mh, mw, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)

        dpr = self.devicePixelRatio()
        target_w = max(50, int((self.mask_frame.width() - 20) * dpr))
        target_h = max(50, int((self.mask_frame.height() - 20) * dpr))
        scaled = pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(dpr)
        self.mask_label.setPixmap(scaled)

    def _update_mask_preview(self, recalculate: bool = True) -> None:
        if self.current_photo_bgr is None:
            return
        if recalculate or self.current_mask is None:
            self._start_preprocess_worker(self.current_photo_bgr, crop=True)
            return

        self._display_mask_preview()

    def _start_search(self) -> None:
        # 0. License Verification Guard
        op_name = "search_dxf" if self.search_mode == "mode_b" else "search_pdf"
        allowed, lic_msg = self.license_manager.is_operation_allowed(op_name)
        if not allowed:
            show_subscription_expired_dialog(self, self.license_manager, lic_msg)
            self._update_license_indicator()
            return

        top_k_str = self.catalog_index.db.get_setting("top_k", "5")
        top_k = int(top_k_str) if top_k_str else 5
        chamfer_str = self.catalog_index.db.get_setting("chamfer_threshold", "0.12")
        chamfer_th = float(chamfer_str) if chamfer_str else 0.12
        cand_str = self.catalog_index.db.get_setting("candidate_limit", "30")
        cand_limit = int(cand_str) if cand_str else 30

        if self.search_mode == "mode_b":
            if not self.reference_dxf_path or not Path(self.reference_dxf_path).is_file():
                QMessageBox.warning(
                    self,
                    "No Reference DXF",
                    "Please select or drop a reference .dxf file to search the DXF library.",
                )
                return

            dxf_stats = self.dxf_db.get_stats()
            if dxf_stats.get("total_components", 0) == 0:
                QMessageBox.warning(
                    self,
                    "Empty DXF Library Index",
                    "The DXF Library index is currently empty.\nPlease click 'Index DXF Library' at the top to index your DXF folder first.",
                )
                return

            self.search_btn.setEnabled(False)
            self.search_btn.setText("MATCHING DXF DESIGNS...")
            self.status_bar.showMessage("Running Stage 1 candidate retrieval & Stage 2 geometric Chamfer verification...")

            self.dxf_match_worker = DxfMatchWorker(
                db=self.dxf_db,
                mode="reference_dxf",
                query_data=self.reference_dxf_path,
                top_k=top_k,
                candidate_limit=cand_limit,
                chamfer_threshold=chamfer_th,
                parent=self,
            )
            self.dxf_match_worker.results_signal.connect(self._on_dxf_search_completed)
            self.dxf_match_worker.error_signal.connect(self._on_search_error)
            self.dxf_match_worker.start()

        else:
            # Mode A: Stencil -> DXF Library (and PDF Catalogs if indexed)
            if self.current_mask is None:
                return

            dxf_stats = self.dxf_db.get_stats()
            pdf_vecs = self.catalog_index.total_vectors

            if dxf_stats.get("total_components", 0) == 0 and pdf_vecs == 0:
                QMessageBox.warning(
                    self,
                    "Empty Index",
                    "Neither the DXF library nor the PDF catalog is indexed.\nPlease click 'Index DXF Library' or 'Index Catalogs' at the top first.",
                )
                return

            self.search_btn.setEnabled(False)
            self.search_btn.setText("SEARCHING CATALOG...")
            self.status_bar.showMessage("Searching DXF library and PDF catalogs for matching stencil...")

            # Dispatch DXF match worker if DXF library indexed
            if dxf_stats.get("total_components", 0) > 0:
                self.dxf_match_worker = DxfMatchWorker(
                    db=self.dxf_db,
                    mode="stencil",
                    query_data=self.current_mask,
                    top_k=top_k,
                    candidate_limit=cand_limit,
                    chamfer_threshold=chamfer_th,
                    parent=self,
                )
                self.dxf_match_worker.results_signal.connect(self._on_dxf_search_completed)
                self.dxf_match_worker.error_signal.connect(self._on_search_error)
                self.dxf_match_worker.start()

            # Dispatch PDF vector searcher if PDF catalog indexed
            if pdf_vecs > 0:
                self.search_worker = SearchWorker(self.searcher, self.current_mask, top_k=top_k)
                self.search_worker.results_signal.connect(self._on_pdf_search_completed)
                self.search_worker.error_signal.connect(self._on_search_error)
                self.search_worker.start()

    def _set_results_filter(self, mode: str) -> None:
        self.current_filter_mode = mode
        self.tab_all_btn.setChecked(mode == "all")
        self.tab_dxf_btn.setChecked(mode == "dxf")
        self.tab_pdf_btn.setChecked(mode == "pdf")
        self._render_results()

    def _render_results(self) -> None:
        # Clear existing result cards
        self.result_cards.clear()
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        dxf_matches = self.last_dxf_results
        pdf_matches = self.last_pdf_results

        if not dxf_matches and not pdf_matches:
            self.results_count_label.setText("0 matches")
            no_match = QLabel("No matching designs found in DXF or PDF catalogs.", self.results_container)
            no_match.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_match.setStyleSheet("color: #64748B; font-size: 13px; margin-top: 50px;")
            self.results_layout.addWidget(no_match)
            self.results_layout.addStretch()
            return

        self.results_count_label.setText(f"{len(dxf_matches)} DXF + {len(pdf_matches)} PDF matches")
        mode = getattr(self, "current_filter_mode", "all")

        # 1. DXF CAD Results Section
        if mode in ("all", "dxf"):
            dxf_section = QFrame(self.results_container)
            dxf_section.setStyleSheet(
                "background: #F5F3FF; border-left: 4px solid #7C3AED; border-radius: 6px; padding: 8px 14px; margin-top: 4px; margin-bottom: 4px;"
            )
            dxf_sec_layout = QHBoxLayout(dxf_section)
            dxf_sec_layout.setContentsMargins(0, 0, 0, 0)
            dxf_sec_title = QLabel(f"📐 DXF CAD Files (Top {len(dxf_matches)} Vector Matches)", dxf_section)
            dxf_sec_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #5B21B6;")
            dxf_sec_layout.addWidget(dxf_sec_title)
            dxf_sec_layout.addStretch()
            dxf_hint = QLabel("CNC Laser / Plasma Vector Cut Files", dxf_section)
            dxf_hint.setStyleSheet("font-size: 11px; color: #7C3AED; font-weight: 600;")
            dxf_sec_layout.addWidget(dxf_hint)
            self.results_layout.addWidget(dxf_section)

            if dxf_matches:
                for res in dxf_matches:
                    card = ResultCardWidget(
                        res,
                        self.catalog_index.data_dir,
                        self.results_container,
                        query_mask=self.current_mask,
                    )
                    card.card_clicked.connect(self._on_result_card_clicked)
                    card.card_double_clicked.connect(self._on_result_card_double_clicked)
                    card.open_viewer_requested.connect(self._open_standalone_dxf_viewer)
                    self.result_cards.append(card)
                    self.results_layout.addWidget(card)
            else:
                empty_card = QFrame(self.results_container)
                empty_card.setStyleSheet("background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 6px; padding: 12px;")
                ec_layout = QVBoxLayout(empty_card)
                lbl = QLabel(
                    "No DXF vector files indexed yet in catalog.\nPlace your .dxf files in your catalog folder and click 'Index Catalogs' at the top to search them.",
                    empty_card,
                )
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet("color: #64748B; font-size: 12px;")
                ec_layout.addWidget(lbl)
                self.results_layout.addWidget(empty_card)

        # 2. PDF Catalog Results Section
        if mode in ("all", "pdf"):
            pdf_section = QFrame(self.results_container)
            pdf_section.setStyleSheet(
                "background: #EFF6FF; border-left: 4px solid #2563EB; border-radius: 6px; padding: 8px 14px; margin-top: 14px; margin-bottom: 4px;"
            )
            pdf_sec_layout = QHBoxLayout(pdf_section)
            pdf_sec_layout.setContentsMargins(0, 0, 0, 0)
            pdf_sec_title = QLabel(f"📄 PDF Catalog Files (Top {len(pdf_matches)} Page Matches)", pdf_section)
            pdf_sec_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #1D4ED8;")
            pdf_sec_layout.addWidget(pdf_sec_title)
            pdf_sec_layout.addStretch()
            pdf_hint = QLabel("Catalog Design Panels", pdf_section)
            pdf_hint.setStyleSheet("font-size: 11px; color: #2563EB; font-weight: 600;")
            pdf_sec_layout.addWidget(pdf_hint)
            self.results_layout.addWidget(pdf_section)

            if pdf_matches:
                for res in pdf_matches:
                    card = ResultCardWidget(
                        res,
                        self.catalog_index.data_dir,
                        self.results_container,
                        query_mask=self.current_mask,
                    )
                    card.card_clicked.connect(self._on_result_card_clicked)
                    card.card_double_clicked.connect(self._on_result_card_double_clicked)
                    card.open_viewer_requested.connect(self._open_standalone_dxf_viewer)
                    self.result_cards.append(card)
                    self.results_layout.addWidget(card)
            else:
                empty_card = QFrame(self.results_container)
                empty_card.setStyleSheet("background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 6px; padding: 12px;")
                ec_layout = QVBoxLayout(empty_card)
                lbl = QLabel("No matching PDF catalog designs found.", empty_card)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet("color: #64748B; font-size: 12px;")
                ec_layout.addWidget(lbl)
                self.results_layout.addWidget(empty_card)

        self.results_layout.addStretch()

        # Auto-select the top result in the viewer preview
        if self.result_cards:
            self._on_result_card_clicked(self.result_cards[0].result)

    def _on_result_card_clicked(self, result: SearchResult) -> None:
        """Handles single click on any result card: updates selection styling and routes preview."""
        for card in self.result_cards:
            card.set_selected(card.result == result)

        is_dxf = result.pdf_path.lower().endswith(".dxf") or getattr(result, "file_type", "pdf") == "dxf"
        if is_dxf:
            self._display_dxf_preview(result)
        else:
            self._display_pdf_preview(result)

    def _on_result_card_double_clicked(self, result: SearchResult) -> None:
        """Handles double click on result card: opens standalone viewer window for DXF or opens PDF."""
        is_dxf = result.pdf_path.lower().endswith(".dxf") or getattr(result, "file_type", "pdf") == "dxf"
        if is_dxf:
            self._open_standalone_dxf_viewer(result)
        else:
            from app.gui.widgets import open_file_at_page
            open_file_at_page(result.pdf_path, result.page_number)

    def _display_dxf_preview(self, result: SearchResult) -> None:
        """Loads DXF drawing into embedded viewer and auto-frames matched design bbox."""
        self._current_selected_result = result
        self.preview_stack.setCurrentIndex(1)
        self.embedded_dxf_viewer.load_file(
            result.pdf_path,
            matched_bbox=getattr(result, "dxf_bbox", None),
            design_code=getattr(result, "design_code", "") or result.design_number,
            query_mask=self.current_mask,
        )
        self.status_bar.showMessage(
            f"Previewing DXF: {result.pdf_name} | Match: {result.design_code or 'Panel'} ({result.confidence_pct:.1f}%)"
        )

    def _display_pdf_preview(self, result: SearchResult) -> None:
        """Displays thumbnail / rendered catalog page for PDF result in preview panel."""
        self._current_selected_result = result
        self.preview_stack.setCurrentIndex(2)

        pixmap: Optional[QPixmap] = None
        if result.thumbnail_path:
            full_thumb = self.catalog_index.data_dir / result.thumbnail_path
            if full_thumb.is_file():
                pixmap = QPixmap(str(full_thumb))

        if pixmap and not pixmap.isNull():
            dpr = self.devicePixelRatio()
            target_w = max(50, int((self.pdf_preview_frame.width() - 20) * dpr))
            target_h = max(50, int((self.pdf_preview_frame.height() - 20) * dpr))
            scaled = pixmap.scaled(
                target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            scaled.setDevicePixelRatio(dpr)
            self.pdf_preview_label.setPixmap(scaled)
        else:
            self.pdf_preview_label.setText(
                f"Page {result.page_number}\n{result.pdf_name}\n\n(No raster thumbnail stored)"
            )
            self.pdf_preview_label.setStyleSheet("color: #94A3B8; font-size: 13px;")

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Intercepts interactions on locked/disabled controls to provide
        immediate license feedback and subscription renewal actions.
        """
        if watched == getattr(self, "nav_tab_dxf_preview", None):
            if event.type() == QEvent.Type.MouseButtonPress:
                if isinstance(event, QMouseEvent) and event.button() == Qt.MouseButton.LeftButton:
                    allowed, lic_msg = self.license_manager.is_operation_allowed("dxf_preview")
                    if not allowed:
                        self.nav_tab_matcher.setChecked(True)
                        self.nav_tab_dxf_preview.setChecked(False)
                        show_subscription_expired_dialog(self, self.license_manager, lic_msg)
                        self._update_license_indicator()
                        return True
        return super().eventFilter(watched, event)

    def _on_dxf_preview_tab_clicked(self) -> None:
        """
        Handler when the user clicks '2. DXF Preview'.
        Validates license validity before switching to the DXF Preview view.
        """
        allowed, lic_msg = self.license_manager.is_operation_allowed("dxf_preview")
        if not allowed:
            self.nav_tab_matcher.setChecked(True)
            self.nav_tab_dxf_preview.setChecked(False)
            self.main_stack.setCurrentIndex(0)
            show_subscription_expired_dialog(self, self.license_manager, lic_msg)
            self._update_license_indicator()
            return
        self._on_main_tab_changed(1)

    def _on_main_tab_changed(self, tab_id: int) -> None:
        """Switches between '1. Matcher' and '2. DXF Preview' tabs."""
        if tab_id == 1:
            allowed, lic_msg = self.license_manager.is_operation_allowed("dxf_preview")
            if not allowed:
                self.nav_tab_matcher.setChecked(True)
                self.nav_tab_dxf_preview.setChecked(False)
                self.main_stack.setCurrentIndex(0)
                show_subscription_expired_dialog(self, self.license_manager, lic_msg)
                self._update_license_indicator()
                return

        self.nav_tab_matcher.setChecked(tab_id == 0)
        self.nav_tab_dxf_preview.setChecked(tab_id == 1)
        self.main_stack.setCurrentIndex(tab_id)
        if tab_id == 1:
            # Auto-load test11.dxf or first file if nothing loaded yet in DXF Preview
            if self.dxf_preview_page.current_dxf_path is None:
                p_dxf = Path.cwd() / "test11.dxf"
                if p_dxf.is_file():
                    self.dxf_preview_page.load_file(p_dxf)

    def switch_to_dxf_preview(
        self,
        dxf_path: Path | str,
        matched_bbox: Optional[Tuple[float, float, float, float]] = None,
        matched_code: Optional[str] = None,
    ) -> None:
        """Switches active view to '2. DXF Preview' tab and frames the specified drawing / design."""
        allowed, lic_msg = self.license_manager.is_operation_allowed("dxf_preview")
        if not allowed:
            show_subscription_expired_dialog(self, self.license_manager, lic_msg)
            self._update_license_indicator()
            return

        self._on_main_tab_changed(1)
        self.dxf_preview_page.load_file(dxf_path, matched_bbox=matched_bbox, matched_code=matched_code)

    def _open_standalone_dxf_viewer(self, result: Optional[SearchResult] = None) -> None:
        """Opens DXF in the dedicated DXF Preview tab and focuses matched design."""
        allowed, lic_msg = self.license_manager.is_operation_allowed("dxf_preview")
        if not allowed:
            show_subscription_expired_dialog(self, self.license_manager, lic_msg)
            self._update_license_indicator()
            return

        target_res = result or getattr(self, "_current_selected_result", None)
        if target_res is None:
            return
        self.switch_to_dxf_preview(
            target_res.pdf_path,
            matched_bbox=getattr(target_res, "dxf_bbox", None),
            matched_code=getattr(target_res, "design_code", "") or target_res.design_number,
        )

    def _open_standalone_dxf_viewer_from_embedded(self) -> None:
        """Slot for the embedded viewer's 'Full Window' / expand button."""
        self._open_standalone_dxf_viewer(getattr(self, "_current_selected_result", None))

    def _search_by_design_code(self) -> None:
        """
        Executes immediate text shortcut search by design code (e.g. 'PD 010')
        without running vector embedding image search.
        """
        allowed, lic_msg = self.license_manager.is_operation_allowed("code_search")
        if not allowed:
            show_subscription_expired_dialog(self, self.license_manager, lic_msg)
            self._update_license_indicator()
            return

        query = self.code_search_input.text().strip()
        if not query:
            return
        self.status_bar.showMessage(f"Searching catalog for design code '{query}'...")
        results = self.searcher.search_by_code(query, limit=20)
        self.last_dxf_results = [r for r in results if r.file_type == "dxf"]
        self.last_pdf_results = [r for r in results if r.file_type == "pdf"]
        self._render_results()
        total_found = len(results)
        if total_found > 0:
            self.status_bar.showMessage(f"Found {total_found} design(s) matching code '{query}'.")
        else:
            self.status_bar.showMessage(f"No designs found matching code '{query}'.")

    def _on_dxf_search_completed(self, results: List[SearchResult]) -> None:
        self.search_btn.setEnabled(True)
        self.search_btn.setText(
            "SEARCH DXF LIBRARY (FROM REFERENCE DXF)"
            if self.search_mode == "mode_b"
            else "SEARCH DXF LIBRARY (FROM STENCIL)"
        )
        self.last_dxf_results = results
        dxf_count = len(self.last_dxf_results)
        pdf_count = len(self.last_pdf_results)

        self.tab_all_btn.setText(f"All ({dxf_count} DXF + {pdf_count} PDF)")
        self.tab_dxf_btn.setText(f"📐 Top DXF (Vectors) [{dxf_count}]")
        self.tab_pdf_btn.setText(f"📄 Top PDF Catalogs [{pdf_count}]")

        self._render_results()

        if results:
            top_res = results[0]
            status_tag = getattr(top_res, "match_status", "Match")
            self.status_bar.showMessage(
                f"DXF search complete! Top match: {top_res.pdf_name} [{status_tag}] with {top_res.confidence_pct:.1f}% confidence."
            )
        else:
            self.status_bar.showMessage("DXF search finished. No reliable geometric matches found.")

    def _on_pdf_search_completed(self, results: List[SearchResult]) -> None:
        self.search_btn.setEnabled(True)
        self.last_pdf_results = [r for r in results if not r.pdf_path.lower().endswith(".dxf")]
        dxf_count = len(self.last_dxf_results)
        pdf_count = len(self.last_pdf_results)

        self.tab_all_btn.setText(f"All ({dxf_count} DXF + {pdf_count} PDF)")
        self.tab_dxf_btn.setText(f"📐 Top DXF (Vectors) [{dxf_count}]")
        self.tab_pdf_btn.setText(f"📄 Top PDF Catalogs [{pdf_count}]")

        self._render_results()

    def _on_search_completed(self, results: List[SearchResult]) -> None:
        self.search_btn.setEnabled(True)
        self.search_btn.setText(
            "SEARCH DXF LIBRARY (FROM REFERENCE DXF)"
            if self.search_mode == "mode_b"
            else "SEARCH DXF LIBRARY (FROM STENCIL)"
        )

        # Separate into DXF CAD and PDF Catalog results
        self.last_dxf_results = [
            r for r in results
            if r.pdf_path.lower().endswith(".dxf") or getattr(r, "file_type", "pdf") == "dxf"
        ]
        self.last_pdf_results = [
            r for r in results
            if not (r.pdf_path.lower().endswith(".dxf") or getattr(r, "file_type", "pdf") == "dxf")
        ]

        dxf_count = len(self.last_dxf_results)
        pdf_count = len(self.last_pdf_results)

        self.tab_all_btn.setText(f"All ({dxf_count} DXF + {pdf_count} PDF)")
        self.tab_dxf_btn.setText(f"📐 Top DXF (Vectors) [{dxf_count}]")
        self.tab_pdf_btn.setText(f"📄 Top PDF Catalogs [{pdf_count}]")

        self._render_results()

        if results:
            top_res = results[0]
            ext_lbl = "DXF" if top_res.pdf_path.lower().endswith(".dxf") else "PDF"
            self.status_bar.showMessage(
                f"Search complete! Top {ext_lbl} match: {top_res.pdf_name} with {top_res.confidence_pct:.1f}% confidence."
            )
        else:
            self.status_bar.showMessage("Search finished. No matches found.")

    def _on_search_error(self, err_msg: str) -> None:
        self.search_btn.setEnabled(True)
        self.search_btn.setText(
            "SEARCH DXF LIBRARY (FROM REFERENCE DXF)"
            if self.search_mode == "mode_b"
            else "SEARCH DXF LIBRARY (FROM STENCIL)"
        )
        QMessageBox.critical(self, "Search Error", f"An error occurred during search:\n{err_msg}")
        self.status_bar.showMessage("Search failed.")

    def _start_dxf_indexing_flow(self) -> None:
        """Initiates the background DXF CAD library indexing workflow."""
        allowed, lic_msg = self.license_manager.is_operation_allowed("index_dxf")
        if not allowed:
            show_subscription_expired_dialog(self, self.license_manager, lic_msg)
            self._update_license_indicator()
            return

        dxf_root = self.catalog_index.db.get_setting("dxf_root", "")
        if not dxf_root or not Path(dxf_root).is_dir():
            folder = QFileDialog.getExistingDirectory(self, "Select DXF CAD Library Directory")
            if not folder:
                return
            dxf_root = folder
            self.catalog_index.db.set_setting("dxf_root", dxf_root)

        progress_dialog = QProgressDialog("Scanning DXF CAD files...", "Cancel", 0, 100, self)
        progress_dialog.setWindowTitle("Indexing DXF CAD Library")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)

        self.dxf_index_worker = DxfIndexWorker(self.dxf_db, dxf_root, parent=self)

        def on_progress(info: dict) -> None:
            if progress_dialog.wasCanceled():
                if self.dxf_index_worker:
                    self.dxf_index_worker.cancel()
                return

            pct = int(info.get("percent", 0.0))
            progress_dialog.setValue(pct)
            msg = info.get("message", "")
            progress_dialog.setLabelText(msg)
            self.status_bar.showMessage(msg)

        def on_finished(prog_res: Any) -> None:
            progress_dialog.close()
            self.update_catalog_stats()
            stats = self.dxf_db.get_stats()
            QMessageBox.information(
                self,
                "DXF Indexing Complete",
                f"Successfully indexed DXF library!\n\n"
                f"Files scanned: {getattr(prog_res, 'total', 0):,}\n"
                f"Newly indexed/updated: {getattr(prog_res, 'processed', 0) - getattr(prog_res, 'skipped', 0) - getattr(prog_res, 'failed', 0):,}\n"
                f"Unchanged (skipped): {getattr(prog_res, 'skipped', 0):,}\n"
                f"Errors/unsupported: {getattr(prog_res, 'failed', 0):,}\n\n"
                f"Total indexed DXF files in DB: {stats.get('total_files', 0):,}\n"
                f"Total indexed components in DB: {stats.get('total_components', 0):,}",
            )
            self.status_bar.showMessage(
                f"DXF Indexing complete. {stats.get('total_components', 0):,} components ready for instant search."
            )

        def on_error(err_str: str) -> None:
            progress_dialog.close()
            QMessageBox.critical(self, "DXF Indexing Error", f"DXF Indexing failed:\n{err_str}")
            self.status_bar.showMessage("DXF Indexing failed.")

        progress_dialog.canceled.connect(self.dxf_index_worker.cancel)
        self.dxf_index_worker.progress_signal.connect(on_progress)
        self.dxf_index_worker.finished_signal.connect(on_finished)
        self.dxf_index_worker.error_signal.connect(on_error)
        self.dxf_index_worker.start()

    def _start_indexing_flow(self) -> None:
        """Initiates the background catalog indexing workflow."""
        allowed, lic_msg = self.license_manager.is_operation_allowed("index_pdf")
        if not allowed:
            show_subscription_expired_dialog(self, self.license_manager, lic_msg)
            self._update_license_indicator()
            return

        root_dir = self.catalog_index.db.get_setting("catalog_root", "")
        if not root_dir or not Path(root_dir).is_dir():
            # Prompt user to select directory
            self._open_settings()
            root_dir = self.catalog_index.db.get_setting("catalog_root", "")
            if not root_dir or not Path(root_dir).is_dir():
                return

        dpi_str = self.catalog_index.db.get_setting("dpi", "200")
        dpi = int(dpi_str) if dpi_str else 200

        progress_dialog = QProgressDialog("Scanning PDF & DXF catalogs...", "Cancel", 0, 100, self)
        progress_dialog.setWindowTitle("Indexing Catalog Library")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)

        self.index_worker = IndexWorker(self.catalog_index, root_dir, dpi=dpi)

        def on_progress(info: dict) -> None:
            if progress_dialog.wasCanceled():
                if self.index_worker:
                    self.index_worker.cancel()
                return

            pct = int(info.get("percent", 0.0))
            progress_dialog.setValue(pct)
            msg = info.get("message", "")
            progress_dialog.setLabelText(msg)
            self.status_bar.showMessage(msg)

        def on_finished(new_tiles: int) -> None:
            progress_dialog.close()
            self.update_catalog_stats()
            QMessageBox.information(
                self,
                "Indexing Complete",
                f"Successfully indexed catalog library!\n\n"
                f"Newly indexed vectors: {new_tiles:,}\n"
                f"Total vectors in index: {self.catalog_index.total_vectors:,}\n"
                f"Total PDF files: {self.catalog_index.db.get_total_file_count()}",
            )
            self.status_bar.showMessage(f"Indexing complete. {new_tiles} new design vectors added.")

        def on_error(err_str: str) -> None:
            progress_dialog.close()
            QMessageBox.critical(self, "Indexing Error", f"Indexing failed:\n{err_str}")
            self.status_bar.showMessage("Indexing failed.")

        progress_dialog.canceled.connect(self.index_worker.cancel)
        self.index_worker.progress_signal.connect(on_progress)
        self.index_worker.finished_signal.connect(on_finished)
        self.index_worker.error_signal.connect(on_error)
        self.index_worker.start()

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.catalog_index.db, self)
        if dlg.exec():
            self.update_catalog_stats()

    def _update_license_indicator(self) -> None:
        """Updates header license button appearance and warning notices."""
        verdict = self.license_manager.check_license()
        if verdict.is_valid:
            days = verdict.days_remaining
            if days <= 7:
                self.license_btn.setText(f"⚠️ License ({days}d left)")
                self.license_btn.setStyleSheet(
                    "QPushButton { background: #78350F; color: #FCD34D; border: 1px solid #D97706; "
                    "padding: 5px 12px; font-weight: 700; font-size: 12px; border-radius: 5px; } "
                    "QPushButton:hover { background: #B45309; color: white; }"
                )
            else:
                self.license_btn.setText(f"🔑 Active ({days}d)")
                self.license_btn.setStyleSheet(
                    "QPushButton { background: #064E3B; color: #6EE7B7; border: 1px solid #059669; "
                    "padding: 5px 12px; font-weight: 600; font-size: 12px; border-radius: 5px; } "
                    "QPushButton:hover { background: #047857; color: white; }"
                )
            if hasattr(self, "nav_tab_dxf_preview"):
                self.nav_tab_dxf_preview.setEnabled(True)
                self.nav_tab_dxf_preview.setText("📐 2. DXF Preview")
                self.nav_tab_dxf_preview.setToolTip(
                    "Full DXF drawing viewer with vector zoom, layers, and interactive measurement"
                )
                self.nav_tab_dxf_preview.setProperty("locked", False)
                self.nav_tab_dxf_preview.style().polish(self.nav_tab_dxf_preview)
        else:
            if verdict.is_expired:
                self.license_btn.setText("🔒 License Expired")
            elif verdict.status.value == "SUSPENDED":
                self.license_btn.setText("⚠️ License Suspended")
            elif verdict.status.value == "REVOKED":
                self.license_btn.setText("🔒 License Revoked")
            else:
                self.license_btn.setText("🔑 Activate License")
            self.license_btn.setStyleSheet(
                "QPushButton { background: #7F1D1D; color: #FCA5A5; border: 1px solid #DC2626; "
                "padding: 5px 12px; font-weight: 700; font-size: 12px; border-radius: 5px; } "
                "QPushButton:hover { background: #B91C1C; color: white; }"
            )
            if hasattr(self, "nav_tab_dxf_preview"):
                self.nav_tab_dxf_preview.setEnabled(False)
                self.nav_tab_dxf_preview.setText("🔒 2. DXF Preview (Locked)")
                self.nav_tab_dxf_preview.setToolTip(
                    "DXF Preview is locked: Subscription expired or unlicensed. Click to renew / activate."
                )
                self.nav_tab_dxf_preview.setProperty("locked", True)
                self.nav_tab_dxf_preview.style().polish(self.nav_tab_dxf_preview)

            # Revert to Tab 0 if currently on DXF Preview tab
            if hasattr(self, "main_stack") and self.main_stack.currentIndex() == 1:
                self._on_main_tab_changed(0)

    def _open_license_dialog(self) -> None:
        """Opens the License & Subscription Management dialog."""
        dlg = LicenseDialog(self.license_manager, self)
        dlg.license_updated.connect(self._update_license_indicator)
        dlg.exec()
        self._update_license_indicator()

    def _check_license_on_startup(self) -> None:
        """Emits notification if license is expiring soon or expired, and triggers background online heartbeat."""
        verdict = self.license_manager.check_license()
        if not verdict.is_valid:
            if verdict.is_expired:
                self.status_bar.showMessage("⚠️ Subscription has expired. Matching operations are locked. Click 'License' to renew.")
            elif verdict.status.value == "SUSPENDED":
                self.status_bar.showMessage("⚠️ Subscription is suspended by administrator. Click 'License' for details.")
            elif verdict.status.value == "REVOKED":
                self.status_bar.showMessage("🔒 Subscription has been revoked. Click 'License' for details.")
            else:
                self.status_bar.showMessage("ℹ️ Software is unlicensed. Click 'Activate License' in the top bar to get started.")
        else:
            warn_days = self.license_manager.get_expiration_warning_days()
            if warn_days is not None:
                self.status_bar.showMessage(
                    f"⚠️ Subscription Reminder: Your license expires in {verdict.days_remaining} day(s). Click 'License' to renew."
                )

        # Trigger background online validation heartbeat if a license is installed
        if self.license_manager.has_local_license():
            QTimer.singleShot(1500, self._sync_license_with_server)

    def _sync_license_with_server(self) -> None:
        """Runs a background online heartbeat check with the licensing server."""
        if not self.license_manager.has_local_license():
            return
        if self.license_sync_worker and self.license_sync_worker.isRunning():
            return
        self.license_sync_worker = LicenseSyncWorker(self.license_manager, self)
        self.license_sync_worker.sync_finished.connect(self._on_license_sync_finished)
        self.license_sync_worker.start()

    def _on_license_sync_finished(self, verdict) -> None:
        """Called when background server heartbeat check completes."""
        self._update_license_indicator()
        if not verdict.is_valid:
            if verdict.status.value == "SUSPENDED":
                self.status_bar.showMessage("⚠️ Notice: This subscription has been suspended by the administrator.")
            elif verdict.status.value == "REVOKED":
                self.status_bar.showMessage("🔒 Notice: This subscription has been revoked by the administrator.")

    def closeEvent(self, event) -> None:
        """Ensures worker threads terminate cleanly upon window closure."""
        if self.license_sync_worker and self.license_sync_worker.isRunning():
            self.license_sync_worker.wait(1000)
        if self.search_worker and self.search_worker.isRunning():
            self.search_worker.wait(1000)
        if self.dxf_match_worker and self.dxf_match_worker.isRunning():
            self.dxf_match_worker.wait(1000)
        if self.index_worker and self.index_worker.isRunning():
            self.index_worker.cancel()
            self.index_worker.wait(1500)
        if self.dxf_index_worker and self.dxf_index_worker.isRunning():
            self.dxf_index_worker.cancel()
            self.dxf_index_worker.wait(1500)
        event.accept()


def run_gui(data_dir: Optional[Path | str] = None) -> None:
    """Launches the PySide6 desktop GUI."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("Design Matcher")
    app.setOrganizationName("DesignMatcher")

    window = MainWindow(data_dir=data_dir)
    window.show()
    sys.exit(app.exec())
