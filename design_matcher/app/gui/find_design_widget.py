"""
find_design_widget.py - Modern PySide6 UI for the 'Find Design' Stencil & Matching Workflow.

Integrates:
1. Reference Upload & Stencil Studio:
   - Drag & Drop zone (PNG, JPG, DXF, DWG, PDF).
   - Real-time threshold and smoothing sliders.
   - Interactive Crop Region (via CropDialog).
   - Live 512x512 Stencil Preview with contour & point stats.
   - Stencil Export (PNG, SVG, DXF) and Save/Load.
2. Target File Search Controls:
   - Search Scope: Current Explorer Folder or Custom Folder.
   - Similarity Threshold Slider.
   - Background Indexing & Multi-Stage Matching Worker with Progress & Cancel.
3. Results Panel:
   - Result cards with thumbnails, collision-free names (folder/file.ext),
     similarity %, sheet/page, rotation & mirror badges.
   - Click-to-zoom in CAD viewer.
   - Export to CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QByteArray, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.design_finder import (
    DesignFinderCache,
    DesignMatchWorker,
    MatchResult,
    TargetCandidate,
    TargetIndexWorker,
    export_matches_to_csv,
    extract_file_candidates,
)
from app.core.file_scanner import scan_cad_files
from app.core.pdf_render import render_pdf_page
from app.core.stencil import (
    CANVAS_SIZE,
    Stencil,
    create_stencil_from_raster,
    create_stencil_from_vector,
    export_stencil_dxf,
    export_stencil_png,
    export_stencil_svg,
)
from app.gui.widgets import CropDialog, cv2_to_qpixmap

logger = logging.getLogger(__name__)


class StencilDropZone(QFrame):
    """Interactive drag-and-drop file upload zone for reference stencil images or CAD files."""

    file_dropped = Signal(Path)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            StencilDropZone {
                background-color: #0F172A;
                border: 2px dashed #334155;
                border-radius: 8px;
                padding: 12px;
            }
            StencilDropZone:hover {
                background-color: #1E293B;
                border-color: #38BDF8;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(6)

        self.icon_lbl = QLabel("📥", self)
        self.icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_lbl.setStyleSheet("font-size: 26px; color: #38BDF8; background: transparent;")
        layout.addWidget(self.icon_lbl)

        self.text_lbl = QLabel("Drag & Drop Reference File Here\n(PNG, JPG, DXF, DWG, PDF)", self)
        self.text_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_lbl.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 600; background: transparent;")
        layout.addWidget(self.text_lbl)

        self.btn_browse = QPushButton("📁 Browse Reference...", self)
        self.btn_browse.setStyleSheet(
            """
            QPushButton {
                background: #1E293B;
                color: #F8FAFC;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #334155;
                border-color: #38BDF8;
            }
            """
        )
        self.btn_browse.clicked.connect(self._browse_file)
        layout.addWidget(self.btn_browse)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._browse_file()

    def _browse_file(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Design File",
            "",
            "Design Files (*.png *.jpg *.jpeg *.bmp *.dxf *.dwg *.pdf);;Images (*.png *.jpg *.jpeg *.bmp);;CAD Files (*.dxf *.dwg);;PDF Files (*.pdf);;All Files (*.*)",
        )
        if fpath and Path(fpath).is_file():
            self.file_dropped.emit(Path(fpath).resolve())

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                """
                StencilDropZone {
                    background-color: #1E293B;
                    border: 2px dashed #38BDF8;
                    border-radius: 8px;
                    padding: 12px;
                }
                """
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            """
            StencilDropZone {
                background-color: #0F172A;
                border: 2px dashed #334155;
                border-radius: 8px;
                padding: 12px;
            }
            StencilDropZone:hover {
                background-color: #1E293B;
                border-color: #38BDF8;
            }
            """
        )

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path_str = urls[0].toLocalFile()
            if path_str and Path(path_str).is_file():
                self.file_dropped.emit(Path(path_str).resolve())
        event.acceptProposedAction()
        self.dragLeaveEvent(None)


class MatchResultCard(QFrame):
    """Individual clickable card displaying a match result."""

    clicked = Signal(object)  # MatchResult

    def __init__(self, match: MatchResult, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.match = match
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        # Style card based on confidence badge
        if match.similarity_score >= 85.0:
            border_color = "#10B981"  # Emerald green
            badge_bg = "#064E3B"
            badge_fg = "#34D399"
        elif match.similarity_score >= 70.0:
            border_color = "#F59E0B"  # Amber
            badge_bg = "#78350F"
            badge_fg = "#FBBF24"
        else:
            border_color = "#3B82F6"  # Blue
            badge_bg = "#1E3A8A"
            badge_fg = "#93C5FD"

        self.setStyleSheet(
            f"""
            MatchResultCard {{
                background-color: #0F172A;
                border: 1px solid #334155;
                border-left: 4px solid {border_color};
                border-radius: 6px;
                padding: 6px;
            }}
            MatchResultCard:hover {{
                background-color: #1E293B;
                border-color: #38BDF8;
            }}
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # 1. Thumbnail
        self.thumb_lbl = QLabel(self)
        self.thumb_lbl.setFixedSize(64, 64)
        self.thumb_lbl.setStyleSheet("background: #020617; border: 1px solid #1E293B; border-radius: 4px;")
        self.thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if match.thumbnail_png:
            arr = cv2.imdecode(np.frombuffer(match.thumbnail_png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if arr is not None:
                pix = cv2_to_qpixmap(arr)
                self.thumb_lbl.setPixmap(pix.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.thumb_lbl.setText("CAD")

        layout.addWidget(self.thumb_lbl)

        # 2. Metadata info column
        meta_box = QVBoxLayout()
        meta_box.setSpacing(2)

        # Title: Display name (folder/file.ext)
        title_box = QHBoxLayout()
        title_box.setSpacing(6)
        self.title_lbl = QLabel(match.display_name, self)
        self.title_lbl.setStyleSheet("color: #FFFFFF; font-weight: 700; font-size: 11px;")
        self.title_lbl.setToolTip(f"Full Path: {match.file_path}")
        title_box.addWidget(self.title_lbl)

        # Rank badge
        self.rank_lbl = QLabel(f"#{match.rank}", self)
        self.rank_lbl.setStyleSheet("color: #94A3B8; font-size: 10px; font-weight: 600;")
        title_box.addWidget(self.rank_lbl)
        title_box.addStretch()
        meta_box.addLayout(title_box)

        # Details: Page / Sheet / Label
        details_txt = f"{match.file_type.upper()}"
        if match.page_or_sheet > 0:
            details_txt += f" • Page {match.page_or_sheet}"
        if match.label:
            details_txt += f" • {match.label}"
        self.details_lbl = QLabel(details_txt, self)
        self.details_lbl.setStyleSheet("color: #94A3B8; font-size: 10px;")
        meta_box.addWidget(self.details_lbl)

        # Badges row: Rotation, Mirror
        badge_box = QHBoxLayout()
        badge_box.setSpacing(4)

        if match.rotation_deg != 0:
            rot_lbl = QLabel(f"↻ {match.rotation_deg}°", self)
            rot_lbl.setStyleSheet("background: #1E293B; color: #CBD5E1; border-radius: 3px; padding: 1px 4px; font-size: 9px;")
            badge_box.addWidget(rot_lbl)

        if match.is_mirrored:
            mir_lbl = QLabel("⇋ Mirrored", self)
            mir_lbl.setStyleSheet("background: #1E293B; color: #CBD5E1; border-radius: 3px; padding: 1px 4px; font-size: 9px;")
            badge_box.addWidget(mir_lbl)

        badge_box.addStretch()
        meta_box.addLayout(badge_box)
        layout.addLayout(meta_box, stretch=1)

        # 3. Match Score column
        score_box = QVBoxLayout()
        score_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_box.setSpacing(2)

        self.score_lbl = QLabel(f"{match.similarity_score:.1f}%", self)
        self.score_lbl.setStyleSheet(f"color: {border_color}; font-size: 15px; font-weight: 800;")
        self.score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_box.addWidget(self.score_lbl)

        self.badge_lbl = QLabel(match.status_badge, self)
        self.badge_lbl.setStyleSheet(f"background: {badge_bg}; color: {badge_fg}; border-radius: 3px; padding: 2px 5px; font-size: 9px; font-weight: 700;")
        self.badge_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_box.addWidget(self.badge_lbl)

        layout.addLayout(score_box)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.match)


class FindDesignWidget(QWidget):
    """
    Complete Find Design tool panel designed for docking in DxfPreviewPage.
    Provides reference stencil generation, target folder indexing,
    multi-stage matching, and interactive results navigation.
    """

    match_selected = Signal(Path, int, tuple, int, bool)  # path, page_or_sheet, bbox, rotation, is_mirrored

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_stencil: Optional[Stencil] = None
        self.current_ref_path: Optional[Path] = None
        self.current_raw_bgr: Optional[np.ndarray] = None
        self.target_folder: Optional[Path] = None
        self.cache = DesignFinderCache()

        self._index_worker: Optional[TargetIndexWorker] = None
        self._match_worker: Optional[DesignMatchWorker] = None
        self._matches: List[MatchResult] = []

        self._setup_ui()

    def set_current_cad_folder(self, folder: Path | str) -> None:
        """Sets the active CAD explorer folder as the default search target."""
        p = Path(folder).resolve()
        if p.is_dir():
            self.target_folder = p
            self.lbl_target_folder.setText(f"Target: {p.name}/")
            self.lbl_target_folder.setToolTip(str(p))

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Header Title
        title_box = QHBoxLayout()
        self.title_lbl = QLabel("🎯 FIND DESIGN IN CAD & PDF", self)
        self.title_lbl.setStyleSheet("color: #38BDF8; font-size: 12px; font-weight: 800; letter-spacing: 0.5px;")
        title_box.addWidget(self.title_lbl)
        title_box.addStretch()
        main_layout.addLayout(title_box)

        # Main Scroll Area containing Sections
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 4, 0)
        scroll_layout.setSpacing(10)

        # -------------------------------------------------------------
        # STEP 1: REFERENCE STENCIL STUDIO
        # -------------------------------------------------------------
        sec1_frame = QFrame(scroll_content)
        sec1_frame.setStyleSheet("background: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 6px;")
        sec1_layout = QVBoxLayout(sec1_frame)
        sec1_layout.setSpacing(8)

        sec1_hdr = QLabel("1. REFERENCE DESIGN STENCIL", sec1_frame)
        sec1_hdr.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 700;")
        sec1_layout.addWidget(sec1_hdr)

        # Drop Zone
        self.drop_zone = StencilDropZone(sec1_frame)
        self.drop_zone.file_dropped.connect(self._on_reference_file_selected)
        sec1_layout.addWidget(self.drop_zone)

        # Live Stencil Preview Card (512x512 scaled)
        preview_box = QHBoxLayout()
        preview_box.setSpacing(8)

        self.stencil_preview_lbl = QLabel(sec1_frame)
        self.stencil_preview_lbl.setFixedSize(140, 140)
        self.stencil_preview_lbl.setStyleSheet(
            "background: #020617; border: 1px solid #334155; border-radius: 6px;"
        )
        self.stencil_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stencil_preview_lbl.setText("No Stencil")
        preview_box.addWidget(self.stencil_preview_lbl)

        # Controls next to preview
        ctrl_box = QVBoxLayout()
        ctrl_box.setSpacing(4)

        # Threshold slider
        th_box = QHBoxLayout()
        th_lbl = QLabel("Threshold:", sec1_frame)
        th_lbl.setStyleSheet("color: #CBD5E1; font-size: 10px;")
        th_box.addWidget(th_lbl)
        self.val_th_lbl = QLabel("128", sec1_frame)
        self.val_th_lbl.setStyleSheet("color: #38BDF8; font-size: 10px; font-weight: 700;")
        th_box.addWidget(self.val_th_lbl)
        ctrl_box.addLayout(th_box)

        self.slider_th = QSlider(Qt.Orientation.Horizontal, sec1_frame)
        self.slider_th.setRange(0, 255)
        self.slider_th.setValue(128)
        self.slider_th.valueChanged.connect(self._on_params_changed)
        ctrl_box.addWidget(self.slider_th)

        # Smoothing slider
        sm_box = QHBoxLayout()
        sm_lbl = QLabel("Smoothing:", sec1_frame)
        sm_lbl.setStyleSheet("color: #CBD5E1; font-size: 10px;")
        sm_box.addWidget(sm_lbl)
        self.val_sm_lbl = QLabel("1.5", sec1_frame)
        self.val_sm_lbl.setStyleSheet("color: #38BDF8; font-size: 10px; font-weight: 700;")
        sm_box.addWidget(self.val_sm_lbl)
        ctrl_box.addLayout(sm_box)

        self.slider_sm = QSlider(Qt.Orientation.Horizontal, sec1_frame)
        self.slider_sm.setRange(5, 50)  # 0.5 to 5.0
        self.slider_sm.setValue(15)
        self.slider_sm.valueChanged.connect(self._on_params_changed)
        ctrl_box.addWidget(self.slider_sm)

        # Invert Checkbox
        self.chk_invert = QCheckBox("Invert Polarity", sec1_frame)
        self.chk_invert.setStyleSheet("QCheckBox { color: #CBD5E1; font-size: 10px; }")
        self.chk_invert.toggled.connect(self._on_params_changed)
        ctrl_box.addWidget(self.chk_invert)

        # Stats label
        self.lbl_stencil_stats = QLabel("Points: 0 • Contours: 0", sec1_frame)
        self.lbl_stencil_stats.setStyleSheet("color: #64748B; font-size: 10px;")
        ctrl_box.addWidget(self.lbl_stencil_stats)

        preview_box.addLayout(ctrl_box)
        sec1_layout.addLayout(preview_box)

        # Stencil Action Buttons (Crop, Save, Export)
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(4)

        self.btn_crop = QPushButton("✂ Crop Region...", sec1_frame)
        self.btn_crop.setStyleSheet(
            "QPushButton { background: #1E293B; color: #F8FAFC; border: 1px solid #334155; border-radius: 4px; padding: 4px 6px; font-size: 10px; } "
            "QPushButton:hover { background: #334155; }"
        )
        self.btn_crop.clicked.connect(self._open_crop_tool)
        btn_row1.addWidget(self.btn_crop)

        self.btn_save_stencil = QPushButton("💾 Save", sec1_frame)
        self.btn_save_stencil.setStyleSheet(
            "QPushButton { background: #1E293B; color: #F8FAFC; border: 1px solid #334155; border-radius: 4px; padding: 4px 6px; font-size: 10px; } "
            "QPushButton:hover { background: #334155; }"
        )
        self.btn_save_stencil.clicked.connect(self._save_stencil_dialog)
        btn_row1.addWidget(self.btn_save_stencil)

        self.btn_load_stencil = QPushButton("📂 Load", sec1_frame)
        self.btn_load_stencil.setStyleSheet(
            "QPushButton { background: #1E293B; color: #F8FAFC; border: 1px solid #334155; border-radius: 4px; padding: 4px 6px; font-size: 10px; } "
            "QPushButton:hover { background: #334155; }"
        )
        self.btn_load_stencil.clicked.connect(self._load_stencil_dialog)
        btn_row1.addWidget(self.btn_load_stencil)

        # Export dropdown
        self.btn_export = QToolButton(sec1_frame)
        self.btn_export.setText("Export Stencil ▾")
        self.btn_export.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_export.setStyleSheet(
            "QToolButton { background: #059669; color: white; border: none; border-radius: 4px; padding: 4px 6px; font-size: 10px; font-weight: 700; } "
            "QToolButton:hover { background: #047857; }"
        )
        exp_menu = QMenu(self.btn_export)
        exp_menu.addAction("Export as PNG (512x512)", lambda: self._export_stencil_action("png"))
        exp_menu.addAction("Export as SVG (Vector Paths)", lambda: self._export_stencil_action("svg"))
        exp_menu.addAction("Export as DXF (AutoCAD Polylines)", lambda: self._export_stencil_action("dxf"))
        self.btn_export.setMenu(exp_menu)
        btn_row1.addWidget(self.btn_export)

        sec1_layout.addLayout(btn_row1)
        scroll_layout.addWidget(sec1_frame)

        # -------------------------------------------------------------
        # STEP 2: TARGET SEARCH SCOPE & THRESHOLD
        # -------------------------------------------------------------
        sec2_frame = QFrame(scroll_content)
        sec2_frame.setStyleSheet("background: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 6px;")
        sec2_layout = QVBoxLayout(sec2_frame)
        sec2_layout.setSpacing(6)

        sec2_hdr = QLabel("2. SEARCH TARGETS & THRESHOLD", sec2_frame)
        sec2_hdr.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 700;")
        sec2_layout.addWidget(sec2_hdr)

        # Target Folder row
        tf_row = QHBoxLayout()
        self.lbl_target_folder = QLabel("Target: Current CAD Folder", sec2_frame)
        self.lbl_target_folder.setStyleSheet("color: #E2E8F0; font-size: 10px; font-weight: 600;")
        tf_row.addWidget(self.lbl_target_folder, stretch=1)

        self.btn_choose_target = QPushButton("Browse...", sec2_frame)
        self.btn_choose_target.setStyleSheet(
            "QPushButton { background: #1E293B; color: #CBD5E1; border: 1px solid #334155; border-radius: 4px; padding: 2px 6px; font-size: 10px; } "
            "QPushButton:hover { background: #334155; }"
        )
        self.btn_choose_target.clicked.connect(self._browse_custom_target_folder)
        tf_row.addWidget(self.btn_choose_target)
        sec2_layout.addLayout(tf_row)

        # Similarity Threshold Slider
        st_row = QHBoxLayout()
        st_lbl = QLabel("Min Similarity:", sec2_frame)
        st_lbl.setStyleSheet("color: #CBD5E1; font-size: 10px;")
        st_row.addWidget(st_lbl)
        self.lbl_sim_val = QLabel("65%", sec2_frame)
        self.lbl_sim_val.setStyleSheet("color: #10B981; font-size: 11px; font-weight: 700;")
        st_row.addWidget(self.lbl_sim_val)
        st_row.addStretch()
        sec2_layout.addLayout(st_row)

        self.slider_sim = QSlider(Qt.Orientation.Horizontal, sec2_frame)
        self.slider_sim.setRange(40, 95)
        self.slider_sim.setValue(65)
        self.slider_sim.valueChanged.connect(lambda v: self.lbl_sim_val.setText(f"{v}%"))
        sec2_layout.addWidget(self.slider_sim)

        # Search / Cancel Buttons
        act_row = QHBoxLayout()
        act_row.setSpacing(6)

        self.btn_find = QPushButton("🔍 Find Design in Files", sec2_frame)
        self.btn_find.setStyleSheet(
            """
            QPushButton {
                background: #2563EB;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 7px 12px;
                font-weight: 700;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #1D4ED8;
            }
            QPushButton:disabled {
                background: #334155;
                color: #64748B;
            }
            """
        )
        self.btn_find.clicked.connect(self.start_find_design)
        act_row.addWidget(self.btn_find, stretch=3)

        self.btn_cancel = QPushButton("⏹ Cancel", sec2_frame)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            """
            QPushButton {
                background: #1E293B;
                color: #EF4444;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 7px 10px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #334155;
            }
            QPushButton:disabled {
                color: #64748B;
            }
            """
        )
        self.btn_cancel.clicked.connect(self.cancel_operation)
        act_row.addWidget(self.btn_cancel, stretch=1)

        sec2_layout.addLayout(act_row)

        # Progress bar & Status Text
        self.progress_bar = QProgressBar(sec2_frame)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #020617; border-radius: 4px; } "
            "QProgressBar::chunk { background: #38BDF8; border-radius: 4px; }"
        )
        self.progress_bar.setVisible(False)
        sec2_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Ready. Upload or drop a reference design above.", sec2_frame)
        self.status_lbl.setStyleSheet("color: #64748B; font-size: 10px;")
        sec2_layout.addWidget(self.status_lbl)

        scroll_layout.addWidget(sec2_frame)

        # -------------------------------------------------------------
        # STEP 3: MATCHED RESULTS LIST & CSV EXPORT
        # -------------------------------------------------------------
        sec3_frame = QFrame(scroll_content)
        sec3_frame.setStyleSheet("background: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 6px;")
        sec3_layout = QVBoxLayout(sec3_frame)
        sec3_layout.setSpacing(6)

        res_hdr_box = QHBoxLayout()
        self.lbl_results_hdr = QLabel("3. MATCH RESULTS (0 found)", sec3_frame)
        self.lbl_results_hdr.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 700;")
        res_hdr_box.addWidget(self.lbl_results_hdr)
        res_hdr_box.addStretch()

        self.btn_export_csv = QPushButton("📄 Export CSV", sec3_frame)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.setStyleSheet(
            "QPushButton { background: #1E293B; color: #CBD5E1; border: 1px solid #334155; border-radius: 4px; padding: 2px 6px; font-size: 10px; } "
            "QPushButton:hover { background: #334155; }"
        )
        self.btn_export_csv.clicked.connect(self._export_csv_action)
        res_hdr_box.addWidget(self.btn_export_csv)
        sec3_layout.addLayout(res_hdr_box)

        # Container for Match Result Cards
        self.results_container = QWidget(sec3_frame)
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(6)

        self.no_results_lbl = QLabel("No matches found yet. Run 'Find Design in Files'.", self.results_container)
        self.no_results_lbl.setStyleSheet("color: #64748B; font-size: 10px; padding: 10px;")
        self.no_results_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.no_results_lbl)

        sec3_layout.addWidget(self.results_container)
        scroll_layout.addWidget(sec3_frame)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # Stencil Studio Handlers
    # -------------------------------------------------------------------------

    def _on_reference_file_selected(self, file_path: Path) -> None:
        """Called when a file is dropped or selected via browse dialog."""
        self.current_ref_path = file_path
        ext = file_path.suffix.lower()

        try:
            if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                bgr = cv2.imread(str(file_path))
                if bgr is None:
                    raise ValueError(f"Could not read image file: {file_path.name}")
                self.current_raw_bgr = bgr
                self._regenerate_raster_stencil()
            elif ext in (".dxf", ".dwg", ".pdf"):
                self.current_raw_bgr = None
                self.status_lbl.setText(f"Generating vector stencil from {file_path.name}...")
                stencil = create_stencil_from_vector(file_path)
                self._set_active_stencil(stencil)
            else:
                QMessageBox.warning(self, "Unsupported Format", f"Unsupported reference file type: {ext}")
        except Exception as exc:
            logger.exception("Error processing reference file: %s", exc)
            QMessageBox.critical(self, "Error Loading Reference", f"Failed to process {file_path.name}:\n{exc}")

    def _on_params_changed(self) -> None:
        """Triggered when threshold or smoothing sliders move."""
        th = self.slider_th.value()
        sm = self.slider_sm.value() / 10.0
        self.val_th_lbl.setText(str(th))
        self.val_sm_lbl.setText(f"{sm:.1f}")

        if self.current_raw_bgr is not None:
            self._regenerate_raster_stencil()

    def _regenerate_raster_stencil(self) -> None:
        if self.current_raw_bgr is None:
            return
        th = self.slider_th.value()
        sm = self.slider_sm.value() / 10.0
        inv = self.chk_invert.isChecked()

        try:
            stencil = create_stencil_from_raster(
                self.current_raw_bgr,
                threshold_val=th,
                smoothing_val=sm,
                invert=inv,
            )
            self._set_active_stencil(stencil)
        except Exception as exc:
            logger.debug("Failed updating stencil: %s", exc)

    def _set_active_stencil(self, stencil: Stencil) -> None:
        self.current_stencil = stencil
        # Update preview
        pix = cv2_to_qpixmap(stencil.bitmap)
        self.stencil_preview_lbl.setPixmap(
            pix.scaled(138, 138, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        # Update stats
        self.lbl_stencil_stats.setText(
            f"Contours: {stencil.contour_count} • Pts: {stencil.raw_point_count} • AR: {stencil.aspect_ratio:.2f}"
        )
        self.status_lbl.setText(f"Stencil ready ({stencil.contour_count} contours).")

    def _open_crop_tool(self) -> None:
        """Opens interactive crop studio to isolate a specific sub-region."""
        if self.current_raw_bgr is None and self.current_stencil is not None:
            # If from vector, convert stencil bitmap to 3-channel for cropping
            self.current_raw_bgr = cv2.cvtColor(self.current_stencil.bitmap, cv2.COLOR_GRAY2BGR)

        if self.current_raw_bgr is None:
            QMessageBox.information(self, "No Image", "Please upload or drop a reference design first.")
            return

        dlg = CropDialog(self.current_raw_bgr, parent=self)
        if dlg.exec() and dlg.cropped_bgr is not None:
            self.current_raw_bgr = dlg.cropped_bgr
            self._regenerate_raster_stencil()

    def _save_stencil_dialog(self) -> None:
        if self.current_stencil is None:
            QMessageBox.information(self, "No Stencil", "Please create or upload a stencil first.")
            return

        fpath, _ = QFileDialog.getSaveFileName(
            self, "Save Stencil Package", "stencil_design.json", "Stencil JSON (*.json)"
        )
        if fpath:
            try:
                self.current_stencil.save_stencil(Path(fpath))
                QMessageBox.information(self, "Saved", f"Stencil successfully saved to:\n{fpath}")
            except Exception as exc:
                QMessageBox.critical(self, "Save Error", f"Failed to save stencil:\n{exc}")

    def _load_stencil_dialog(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Load Stencil Package", "", "Stencil JSON (*.json)"
        )
        if fpath and Path(fpath).is_file():
            try:
                st = Stencil.load_stencil(Path(fpath))
                self.current_raw_bgr = cv2.cvtColor(st.bitmap, cv2.COLOR_GRAY2BGR)
                self._set_active_stencil(st)
                QMessageBox.information(self, "Loaded", f"Loaded stencil from {Path(fpath).name}")
            except Exception as exc:
                QMessageBox.critical(self, "Load Error", f"Failed to load stencil:\n{exc}")

    def _export_stencil_action(self, fmt: str) -> None:
        if self.current_stencil is None:
            QMessageBox.information(self, "No Stencil", "Please create or upload a stencil first.")
            return

        def_name = f"stencil_export.{fmt}"
        if fmt == "png":
            filter_str = "PNG Image (*.png)"
            func = export_stencil_png
        elif fmt == "svg":
            filter_str = "SVG Vector (*.svg)"
            func = export_stencil_svg
        elif fmt == "dxf":
            filter_str = "AutoCAD DXF (*.dxf)"
            func = export_stencil_dxf
        else:
            return

        fpath, _ = QFileDialog.getSaveFileName(self, f"Export Stencil as {fmt.upper()}", def_name, filter_str)
        if fpath:
            try:
                func(self.current_stencil, Path(fpath))
                QMessageBox.information(self, "Exported", f"Successfully exported stencil to:\n{fpath}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", f"Failed to export stencil:\n{exc}")

    # -------------------------------------------------------------------------
    # Search & Matching Handlers
    # -------------------------------------------------------------------------

    def _browse_custom_target_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select Target Folder to Search")
        if chosen and Path(chosen).is_dir():
            self.set_current_cad_folder(Path(chosen))

    def start_find_design(self) -> None:
        """Starts background target file indexing and matching pipeline."""
        if self.current_stencil is None:
            QMessageBox.warning(self, "No Stencil", "Please create or upload a reference design stencil first.")
            return

        if self.target_folder is None or not self.target_folder.is_dir():
            QMessageBox.warning(self, "Target Folder Missing", "Please select a valid target folder to search.")
            return

        # 1. Discover target files (.dxf, .dwg, .pdf)
        allowed_exts = {".dxf", ".dwg", ".pdf"}
        target_files = scan_cad_files(self.target_folder, allowed_extensions=allowed_exts, recursive=True)

        if not target_files:
            QMessageBox.information(
                self,
                "No CAD / PDF Files",
                f"No DXF, DWG, or PDF files found in:\n{self.target_folder}",
            )
            return

        self.btn_find.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_lbl.setText(f"Scanning {len(target_files)} target files...")

        # Clear previous result cards
        self._clear_results()

        # Step 1: Run background indexing worker to ensure all files are indexed in SQLite
        custom_oda = None
        try:
            from app.core.db import CatalogDatabase
            custom_oda = CatalogDatabase().get_setting("oda_converter_path")
        except Exception:
            pass

        self._index_worker = TargetIndexWorker(
            target_files,
            cache=self.cache,
            custom_oda_path=custom_oda,
            parent=self,
        )
        self._index_worker.progress_signal.connect(self._on_index_progress)
        self._index_worker.status_signal.connect(self.status_lbl.setText)
        self._index_worker.finished_signal.connect(lambda f_cnt, c_cnt: self._on_indexing_finished(target_files))
        self._index_worker.error_signal.connect(lambda p, err: logger.warning("Index error on %s: %s", p, err))
        self._index_worker.start()

    def cancel_operation(self) -> None:
        """Cancels running index or match workers."""
        if self._index_worker and self._index_worker.isRunning():
            self._index_worker.cancel()
        if self._match_worker and self._match_worker.isRunning():
            self._match_worker.cancel()
        self.status_lbl.setText("Operation cancelled.")
        self.btn_find.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)

    def _on_index_progress(self, current: int, total: int, fname: str) -> None:
        pct = int((current / max(1, total)) * 50)  # Indexing is 0-50%
        self.progress_bar.setValue(pct)
        self.status_lbl.setText(f"Indexing ({current}/{total}): {fname}")

    def _on_indexing_finished(self, target_files: List[Path]) -> None:
        """Called when background target indexing completes. Proceeds to multi-stage matching."""
        self.status_lbl.setText("Loading candidates from cache...")
        candidates = self.cache.get_all_candidates_for_files(target_files)

        if not candidates:
            self.status_lbl.setText("No design candidates could be extracted from target files.")
            self.btn_find.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self.progress_bar.setVisible(False)
            return

        self.progress_bar.setValue(50)
        self.status_lbl.setText(f"Matching stencil against {len(candidates)} candidates...")

        min_conf = float(self.slider_sim.value())

        # Step 2: Run background match worker
        self._match_worker = DesignMatchWorker(
            stencil=self.current_stencil,
            candidates=candidates,
            min_confidence=min_conf,
            top_k=50,
            root_folder=self.target_folder,
            parent=self,
        )
        self._match_worker.status_signal.connect(self.status_lbl.setText)
        self._match_worker.finished_signal.connect(self._on_matching_finished)
        self._match_worker.error_signal.connect(self._on_matching_error)
        self._match_worker.start()

    def _on_matching_finished(self, matches: List[MatchResult]) -> None:
        self.progress_bar.setValue(100)
        self.btn_find.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._matches = matches

        self.lbl_results_hdr.setText(f"3. MATCH RESULTS ({len(matches)} found)")
        self.btn_export_csv.setEnabled(len(matches) > 0)

        if not matches:
            self.no_results_lbl.setVisible(True)
            self.no_results_lbl.setText("No designs matched the minimum similarity threshold.")
            self.status_lbl.setText("Search finished. 0 matches found.")
            return

        self.no_results_lbl.setVisible(False)
        self.status_lbl.setText(f"Found {len(matches)} matching designs!")

        # Populate cards
        for match in matches:
            card = MatchResultCard(match, parent=self.results_container)
            card.clicked.connect(self._on_result_card_clicked)
            self.results_layout.addWidget(card)

    def _on_matching_error(self, err_msg: str) -> None:
        self.btn_find.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_lbl.setText(f"Matching error: {err_msg}")
        QMessageBox.critical(self, "Matching Error", f"Search encountered an error:\n{err_msg}")

    def _clear_results(self) -> None:
        # Remove all existing result card widgets
        while self.results_layout.count() > 1:
            item = self.results_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self.no_results_lbl.setVisible(True)
        self.no_results_lbl.setText("Searching target files...")
        self.btn_export_csv.setEnabled(False)
        self.lbl_results_hdr.setText("3. MATCH RESULTS (searching...)")

    def _on_result_card_clicked(self, match: MatchResult) -> None:
        """Triggered when user clicks a match card."""
        fpath = Path(match.file_path)
        logger.info("Match card selected: %s, page/sheet=%d, bbox=%s", fpath.name, match.page_or_sheet, match.bbox)
        self.match_selected.emit(fpath, match.page_or_sheet, match.bbox, match.rotation_deg, match.is_mirrored)

    def _export_csv_action(self) -> None:
        if not self._matches:
            return
        fpath, _ = QFileDialog.getSaveFileName(
            self, "Export Results to CSV", "design_matches.csv", "CSV Files (*.csv)"
        )
        if fpath:
            try:
                export_matches_to_csv(self._matches, fpath)
                QMessageBox.information(self, "Exported", f"Exported {len(self._matches)} matches to:\n{fpath}")
            except Exception as exc:
                QMessageBox.critical(self, "CSV Export Error", f"Failed exporting CSV:\n{exc}")
