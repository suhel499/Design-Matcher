"""
Custom UI widgets for Design Matcher:
- Photo drop zone with preview
- Stencil mask preview
- Modern result cards with thumbnails and action buttons
- Settings dialog with persistence
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.core.image_crop import (
    crop_rect,
    image_to_view_coords,
    load_image_with_exif,
    order_points,
    sharpen,
    view_to_image_coords,
    warp_four_points,
)
from app.core.preprocess import (
    convert_customer_crop_to_stencil_mask,
    detect_sheet_quadrilateral,
    detect_stencil_polarity,
    four_point_perspective_transform,
    normalize_stencil,
)
from app.core.search import SearchResult

logger = logging.getLogger(__name__)


def cv2_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """Converts an OpenCV BGR or Grayscale image array to a QPixmap."""
    if cv_img is None or cv_img.size == 0:
        return QPixmap()

    if cv_img.ndim == 2:
        gray = np.ascontiguousarray(cv_img)
        h, w = gray.shape
        bytes_per_line = w
        q_img = QImage(gray.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
    else:
        # Convert BGR to RGB
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    return QPixmap.fromImage(q_img.copy())


def open_file_at_page(pdf_path: str, page_number: int) -> None:
    """
    Cross-platform utility to open a PDF or DXF file, attempting to pass page number where supported.
    Provides clear error dialog if the file was moved or deleted since indexing.
    """
    p = Path(pdf_path).resolve()
    if not p.is_file():
        ext = "DXF CAD" if p.suffix.lower() == ".dxf" else "PDF Catalog"
        QMessageBox.warning(
            None,
            f"{ext} File Not Found",
            f"The design file could not be found at its indexed path:\n\n{pdf_path}\n\n"
            f"It may have been moved, renamed, or deleted since indexing.\n"
            f"Please run 'Index DXF Library' or re-index your catalogs.",
        )
        return

    system = platform.system()
    try:
        if system == "Windows":
            # On Windows, try opening via default program
            os.startfile(str(p))
        elif system == "Darwin":
            # macOS
            subprocess.run(["open", str(p)], check=False)
        else:
            # Linux
            subprocess.run(["xdg-open", str(p)], check=False)
    except Exception as exc:
        QMessageBox.critical(None, "Error Opening File", f"Could not open file: {exc}")


def open_containing_folder(file_path: str) -> None:
    """Reveals the file or containing folder in the operating system's native file manager."""
    p = Path(file_path).resolve()
    target = p if p.exists() else p.parent
    if not target.exists():
        QMessageBox.warning(
            None,
            "Folder Not Found",
            f"The target path or folder could not be found:\n\n{p}\n\n"
            f"It may have been moved, renamed, or deleted since indexing.",
        )
        return

    system = platform.system()
    try:
        if system == "Windows":
            if p.exists():
                subprocess.run(["explorer", f"/select,{str(p)}"], check=False)
            else:
                subprocess.run(["explorer", str(target)], check=False)
        elif system == "Darwin":
            if p.exists():
                subprocess.run(["open", "-R", str(p)], check=False)
            else:
                subprocess.run(["open", str(target)], check=False)
        else:
            folder_to_open = p.parent if p.is_file() else target
            subprocess.run(["xdg-open", str(folder_to_open)], check=False)
    except Exception as exc:
        logger.error("Error opening containing folder: %s", exc)



class PhotoDropArea(QFrame):
    """
    Drag and drop photo upload area with visual feedback and file picker fallback.
    """

    photo_loaded = Signal(np.ndarray, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.current_image_bgr: Optional[np.ndarray] = None
        self.current_file_path: str = ""

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("photoDropArea")
        self.setMinimumSize(280, 260)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: none; background: transparent;")

        self.text_label = QLabel("Drag & Drop Customer Photo\nor Click to Browse", self)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet("color: #718096; font-size: 13px; font-weight: 500;")

        self.subtext_label = QLabel("Supports JPG, PNG, WEBP, BMP, DXF (CAD)", self)
        self.subtext_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtext_label.setStyleSheet("color: #A0AEC0; font-size: 11px;")

        layout.addWidget(self.preview_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.subtext_label)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for u in urls:
                ext = Path(u.toLocalFile()).suffix.lower()
                if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".dxf"}:
                    event.acceptProposedAction()
                    self.setStyleSheet("#photoDropArea { border: 2px dashed #3B82F6; background: #EFF6FF; }")
                    return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet("")

    def dropEvent(self, event: QDropEvent) -> None:
        self.setStyleSheet("")
        urls = event.mimeData().urls()
        for u in urls:
            path_str = u.toLocalFile()
            if self.load_image_from_file(path_str):
                event.acceptProposedAction()
                return

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Customer Photo or DXF File",
                "",
                "Images & CAD Files (*.jpg *.jpeg *.png *.webp *.bmp *.dxf);;DXF Files (*.dxf);;Image Files (*.jpg *.jpeg *.png *.webp *.bmp);;All Files (*.*)",
            )
            if file_path:
                self.load_image_from_file(file_path)

    def load_image_from_file(self, file_path: str) -> bool:
        p = Path(file_path)
        if not p.is_file():
            return False

        if p.suffix.lower() == ".dxf":
            try:
                from app.core.dxf_render import render_dxf_preview
                bgr = render_dxf_preview(p, target_size=800)
            except Exception as exc:
                QMessageBox.warning(self, "Invalid DXF File", f"Unable to parse DXF CAD file:\n{exc}")
                return False
        else:
            try:
                bgr = load_image_with_exif(p)
            except Exception:
                bgr = None

        if bgr is None:
            QMessageBox.warning(self, "Invalid Image", f"Unable to read image file:\n{file_path}")
            return False

        self.current_image_bgr = bgr
        self.current_file_path = str(p.resolve())
        self.update_preview()
        self.photo_loaded.emit(self.current_image_bgr, self.current_file_path)
        return True

    def set_image(self, bgr_img: np.ndarray, file_path: str = "") -> None:
        self.current_image_bgr = bgr_img
        if file_path:
            self.current_file_path = file_path
        self.update_preview()

    def clear_image(self) -> None:
        self.current_image_bgr = None
        self.current_file_path = ""
        self.update_preview()

    def update_preview(self) -> None:
        if self.current_image_bgr is None:
            self.preview_label.clear()
            self.text_label.setVisible(True)
            self.subtext_label.setVisible(True)
            return

        self.text_label.setVisible(False)
        self.subtext_label.setVisible(False)

        pixmap = cv2_to_qpixmap(self.current_image_bgr)
        dpr = self.devicePixelRatio()
        target_w = max(50, int((self.width() - 24) * dpr))
        target_h = max(50, int((self.height() - 24) * dpr))
        scaled = pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(dpr)
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.current_image_bgr is not None:
            self.update_preview()


class ResultCardWidget(QFrame):
    """
    Individual card displaying matched catalog design tile information and actions.
    """

    card_clicked = Signal(object)
    card_double_clicked = Signal(object)
    open_viewer_requested = Signal(object)

    def __init__(
        self,
        result: SearchResult,
        data_dir: Path,
        parent: Optional[QWidget] = None,
        query_mask: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(parent)
        self.result = result
        self.data_dir = data_dir
        self.query_mask = query_mask
        self.is_selected = False

        self.setObjectName("resultCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._setup_ui()

    def set_selected(self, selected: bool) -> None:
        self.is_selected = selected
        if selected:
            self.setStyleSheet(
                "QFrame#resultCard { background-color: #F8FAFC; border: 2px solid #2563EB; "
                "border-radius: 8px; margin-bottom: 8px; }"
            )
        else:
            self.setStyleSheet("")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.card_clicked.emit(self.result)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.card_double_clicked.emit(self.result)
        super().mouseDoubleClickEvent(event)

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(14)

        is_dxf = self.result.pdf_path.lower().endswith(".dxf") or getattr(self.result, "file_type", "pdf") == "dxf"

        # 1. Thumbnail
        self.thumb_label = QLabel(self)
        self.thumb_label.setFixedSize(96, 96)
        self.thumb_label.setStyleSheet("background: #1E293B; border-radius: 6px; border: 1px solid #E2E8F0;")
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        thumb_loaded = False
        raw_match = getattr(self.result, "raw_dxf_match", None)
        if raw_match and getattr(raw_match, "preview_png", None):
            pix = QPixmap()
            if pix.loadFromData(raw_match.preview_png):
                scaled = pix.scaled(92, 92, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.thumb_label.setPixmap(scaled)
                thumb_loaded = True
        elif self.result.thumbnail_path:
            full_thumb_p = self.data_dir / self.result.thumbnail_path
            if full_thumb_p.is_file():
                pix = QPixmap(str(full_thumb_p))
                if not pix.isNull():
                    scaled = pix.scaled(92, 92, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.thumb_label.setPixmap(scaled)
                    thumb_loaded = True

        if not thumb_loaded:
            self.thumb_label.setText("DXF Vector" if is_dxf else "No Preview")
            self.thumb_label.setStyleSheet("color: #94A3B8; font-size: 10px; background: #1E293B; border-radius: 6px;")

        layout.addWidget(self.thumb_label)

        # 2. Metadata details
        details_layout = QVBoxLayout()
        details_layout.setSpacing(4)
        file_exists = Path(self.result.pdf_path).is_file()

        # Title row: Rank badge + Filename + Format badge + Match Status Badge + Missing alert
        title_row = QHBoxLayout()
        rank_badge_text = f"DXF #{self.result.rank}" if is_dxf else f"PDF #{self.result.rank}"
        rank_badge = QLabel(rank_badge_text, self)
        rank_bg = "#7C3AED" if is_dxf else "#2563EB"
        rank_badge.setStyleSheet(
            f"background: {rank_bg}; color: white; font-weight: bold; font-size: 11px; "
            "padding: 2px 6px; border-radius: 4px;"
        )
        title_row.addWidget(rank_badge)

        file_name_label = QLabel(self.result.pdf_name, self)
        file_name_label.setStyleSheet("font-size: 13px; font-weight: 700; color: #0F172A;")
        file_name_label.setToolTip(self.result.pdf_path)
        title_row.addWidget(file_name_label, 1)

        format_pill = QLabel("DXF CAD" if is_dxf else "PDF Catalog", self)
        if is_dxf:
            format_pill.setStyleSheet(
                "background: #EDE9FE; color: #6D28D9; border: 1px solid #DDD6FE; "
                "font-weight: 700; font-size: 11px; padding: 2px 7px; border-radius: 4px;"
            )
        else:
            format_pill.setStyleSheet(
                "background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; "
                "font-weight: 700; font-size: 11px; padding: 2px 7px; border-radius: 4px;"
            )
        title_row.addWidget(format_pill)

        # Match Status Badge
        status_text = getattr(self.result, "match_status", None)
        if status_text:
            status_badge = QLabel(status_text, self)
            if status_text == "Verified Match":
                status_badge.setStyleSheet(
                    "background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; "
                    "font-weight: 700; font-size: 11px; padding: 2px 7px; border-radius: 4px;"
                )
            elif status_text == "Possible Match":
                status_badge.setStyleSheet(
                    "background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; "
                    "font-weight: 700; font-size: 11px; padding: 2px 7px; border-radius: 4px;"
                )
            else:
                status_badge.setStyleSheet(
                    "background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; "
                    "font-weight: 700; font-size: 11px; padding: 2px 7px; border-radius: 4px;"
                )
            title_row.addWidget(status_badge)

        if not file_exists:
            missing_badge = QLabel("⚠️ Moved / Missing", self)
            missing_badge.setStyleSheet(
                "background: #FEE2E2; color: #DC2626; border: 1px solid #FCA5A5; "
                "font-weight: 700; font-size: 10px; padding: 2px 5px; border-radius: 4px;"
            )
            title_row.addWidget(missing_badge)

        # Region badge if matched on a sub-panel
        if getattr(self.result, "matched_region", "Full") and self.result.matched_region != "Full":
            region_badge = QLabel(f"Section: {self.result.matched_region}", self)
            region_badge.setStyleSheet(
                "background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; "
                "font-weight: 600; font-size: 11px; padding: 2px 7px; border-radius: 4px;"
            )
            title_row.addWidget(region_badge)

        # Confidence pill
        conf_pill = QLabel(f"{self.result.confidence_pct:.1f}% Match", self)
        pill_color = "#10B981" if self.result.confidence_pct >= 85 else "#F59E0B"
        conf_pill.setStyleSheet(
            f"background: {pill_color}; color: white; font-weight: 600; font-size: 11px; "
            f"padding: 2px 8px; border-radius: 10px;"
        )
        title_row.addWidget(conf_pill)
        details_layout.addLayout(title_row)

        # Sub-details
        if is_dxf:
            code_str = getattr(self.result, "design_code", None) or self.result.design_number or "Unlabeled"
            idx_str = getattr(self.result, "design_index_info", None) or f"design {self.result.tile_index}"
            page_info = f"Design: {code_str}  |  {idx_str}"
            if getattr(self.result, "geometric_distance", None) is not None:
                page_info += f"  |  Chamfer: {self.result.geometric_distance:.4f}"
            if getattr(self.result, "is_mirrored", False):
                page_info += "  |  Mirrored: Yes"
        else:
            page_info = f"Page {self.result.page_number}  |  Tile #{self.result.tile_index}"
            if self.result.design_number:
                page_info += f"  |  Design: {self.result.design_number}"
        if self.result.best_rotation_deg > 0:
            page_info += f"  |  Rotated: {self.result.best_rotation_deg}°"
        if self.result.iou is not None:
            page_info += f"  |  Shape IoU: {self.result.iou:.2f}"

        page_label = QLabel(page_info, self)
        page_label.setStyleSheet("color: #475569; font-size: 12px; font-weight: 500;")
        details_layout.addWidget(page_label)

        # Full path label with tooltip
        path_label = QLabel(f"Path: {self.result.pdf_path}", self)
        path_label.setToolTip(self.result.pdf_path)
        path_label.setStyleSheet("color: #64748B; font-size: 11px;")
        details_layout.addWidget(path_label)

        # Action buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        if is_dxf:
            btn_viewer = QPushButton("Open Viewer", self)
            btn_viewer.setStyleSheet(
                "QPushButton { background: #2563EB; color: white; border: none; border-radius: 4px; padding: 4px 10px; font-size: 11px; font-weight: 600; } "
                "QPushButton:hover { background: #1D4ED8; }"
            )
            btn_viewer.clicked.connect(lambda: self.open_viewer_requested.emit(self.result))
            btn_row.addWidget(btn_viewer)

            btn_open = QPushButton("Open in App", self)
            btn_open.setStyleSheet(
                "QPushButton { background: #059669; color: white; border: none; border-radius: 4px; padding: 4px 10px; font-size: 11px; font-weight: 600; } "
                "QPushButton:hover { background: #047857; }"
            )
            btn_open.clicked.connect(lambda: open_file_at_page(self.result.pdf_path, 1))
            btn_row.addWidget(btn_open)

            btn_folder = QPushButton("Folder", self)
            btn_folder.setStyleSheet(
                "QPushButton { background: #EDE9FE; color: #6D28D9; border: 1px solid #DDD6FE; border-radius: 4px; padding: 4px 8px; font-size: 11px; font-weight: 600; } "
                "QPushButton:hover { background: #DDD6FE; }"
            )
            btn_folder.clicked.connect(lambda: open_containing_folder(self.result.pdf_path))
            btn_row.addWidget(btn_folder)

            btn_stencil = QPushButton("Mask Preview", self)
            btn_stencil.setStyleSheet(
                "QPushButton { background: #7C3AED; color: white; border: none; border-radius: 4px; padding: 4px 10px; font-size: 11px; font-weight: 600; } "
                "QPushButton:hover { background: #6D28D9; }"
            )
            btn_stencil.clicked.connect(self._show_stencil_preview)
            btn_row.addWidget(btn_stencil)
        else:
            btn_open = QPushButton("Open PDF", self)
            btn_open.setStyleSheet(
                "QPushButton { background: #2563EB; color: white; border: none; border-radius: 4px; padding: 4px 10px; font-size: 11px; font-weight: 600; } "
                "QPushButton:hover { background: #1D4ED8; }"
            )
            btn_open.clicked.connect(lambda: open_file_at_page(self.result.pdf_path, self.result.page_number))
            btn_row.addWidget(btn_open)

            btn_folder = QPushButton("Containing Folder", self)
            btn_folder.setStyleSheet(
                "QPushButton { background: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; border-radius: 4px; padding: 4px 8px; font-size: 11px; } "
                "QPushButton:hover { background: #E2E8F0; }"
            )
            btn_folder.clicked.connect(lambda: open_containing_folder(self.result.pdf_path))
            btn_row.addWidget(btn_folder)

            btn_stencil = QPushButton("Mask Preview", self)
            btn_stencil.setStyleSheet(
                "QPushButton { background: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; border-radius: 4px; padding: 4px 8px; font-size: 11px; } "
                "QPushButton:hover { background: #E2E8F0; }"
            )
            btn_stencil.clicked.connect(self._show_stencil_preview)
            btn_row.addWidget(btn_stencil)

        btn_copy = QPushButton("Copy Path", self)
        btn_copy.setStyleSheet(
            "QPushButton { background: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; border-radius: 4px; padding: 4px 8px; font-size: 11px; } "
            "QPushButton:hover { background: #E2E8F0; }"
        )
        btn_copy.clicked.connect(self._copy_path_to_clipboard)
        btn_row.addWidget(btn_copy)

        btn_row.addStretch()
        details_layout.addLayout(btn_row)

        layout.addLayout(details_layout, 1)

    def _show_stencil_preview(self) -> None:
        dlg = StencilMaskPreviewDialog(self.result, self.data_dir, self.query_mask, parent=self)
        dlg.exec()

    def _copy_path_to_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        clipboard.setText(self.result.pdf_path)
        ext = "DXF" if self.result.pdf_path.lower().endswith(".dxf") else "PDF"
        QMessageBox.information(self, "Copied", f"{ext} file path copied to clipboard!")


class StencilMaskPreviewDialog(QDialog):
    """
    Side-by-side comparison modal showing:
    1. Query stencil mask (from customer's photo)
    2. Matched design stencil mask
    3. Rendered design / CAD vector preview with matched label and metadata.
    """

    def __init__(
        self,
        result: SearchResult,
        data_dir: Path,
        query_mask: Optional[np.ndarray] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.result = result
        self.data_dir = data_dir
        self.query_mask = query_mask

        is_dxf = self.result.pdf_path.lower().endswith(".dxf") or getattr(self.result, "file_type", "pdf") == "dxf"
        code_lbl = getattr(self.result, "design_code", None) or self.result.design_number or "Panel"
        self.setWindowTitle(f"Stencil Mask Comparison - {code_lbl} ({self.result.pdf_name})")
        self.resize(850, 480)
        self._setup_ui(is_dxf, code_lbl)

    def _setup_ui(self, is_dxf: bool, code_lbl: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # Header info
        header_box = QHBoxLayout()
        title = QLabel(f"Matched Design: {code_lbl}", self)
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #0F172A;")
        header_box.addWidget(title)

        header_box.addStretch()
        conf_pill = QLabel(f"{self.result.confidence_pct:.1f}% Confidence", self)
        conf_pill.setStyleSheet(
            "background: #10B981; color: white; font-weight: 700; font-size: 12px; "
            "padding: 4px 10px; border-radius: 12px;"
        )
        header_box.addWidget(conf_pill)
        layout.addLayout(header_box)

        # 3 Panels Side-by-Side: Query Mask | Matched Mask | CAD / Catalog Preview
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(14)

        # 1. Query Mask
        q_frame = QFrame(self)
        q_frame.setStyleSheet("background: #0F172A; border-radius: 8px; border: 1px solid #334155;")
        q_vbox = QVBoxLayout(q_frame)
        q_lbl_title = QLabel("Customer Query Mask", q_frame)
        q_lbl_title.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: 600;")
        q_lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        q_vbox.addWidget(q_lbl_title)

        q_img_lbl = QLabel(q_frame)
        q_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        q_img_lbl.setMinimumSize(224, 224)
        if self.query_mask is not None and self.query_mask.size > 0:
            pix = cv2_to_qpixmap(self.query_mask)
            q_img_lbl.setPixmap(pix.scaled(224, 224, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            q_img_lbl.setText("Text Shortcut Search\n(No Photo Query)")
            q_img_lbl.setStyleSheet("color: #64748B; font-size: 12px;")
        q_vbox.addWidget(q_img_lbl, 1)
        panels_layout.addWidget(q_frame)

        # 2. Matched Stencil Mask
        m_frame = QFrame(self)
        m_frame.setStyleSheet("background: #0F172A; border-radius: 8px; border: 1px solid #334155;")
        m_vbox = QVBoxLayout(m_frame)
        m_lbl_title = QLabel("Matched Stencil Mask", m_frame)
        m_lbl_title.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: 600;")
        m_lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        m_vbox.addWidget(m_lbl_title)

        m_img_lbl = QLabel(m_frame)
        m_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        m_img_lbl.setMinimumSize(224, 224)

        mask_loaded = False
        raw_match = getattr(self.result, "raw_dxf_match", None)
        if raw_match and getattr(raw_match, "canonical_mask_png", None):
            pix = QPixmap()
            if pix.loadFromData(raw_match.canonical_mask_png):
                m_img_lbl.setPixmap(pix.scaled(224, 224, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                mask_loaded = True
        elif self.result.thumbnail_path:
            full_thumb_p = self.data_dir / self.result.thumbnail_path
            mask_p = full_thumb_p.with_name(full_thumb_p.stem + "_mask.png")
            if mask_p.is_file():
                pix = QPixmap(str(mask_p))
                if not pix.isNull():
                    m_img_lbl.setPixmap(pix.scaled(224, 224, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    mask_loaded = True
            elif full_thumb_p.is_file():
                pix = QPixmap(str(full_thumb_p))
                if not pix.isNull():
                    m_img_lbl.setPixmap(pix.scaled(224, 224, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    mask_loaded = True

        if not mask_loaded:
            m_img_lbl.setText("No Mask Stored")
            m_img_lbl.setStyleSheet("color: #64748B; font-size: 12px;")
        m_vbox.addWidget(m_img_lbl, 1)
        panels_layout.addWidget(m_frame)

        # 3. CAD / Vector Preview
        c_frame = QFrame(self)
        c_frame.setStyleSheet("background: #1E293B; border-radius: 8px; border: 1px solid #334155;")
        c_vbox = QVBoxLayout(c_frame)
        c_lbl_title = QLabel("CAD Vector Preview" if is_dxf else "Catalog Page Preview", c_frame)
        c_lbl_title.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: 600;")
        c_lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c_vbox.addWidget(c_lbl_title)

        c_img_lbl = QLabel(c_frame)
        c_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c_img_lbl.setMinimumSize(224, 224)
        if raw_match and getattr(raw_match, "preview_png", None):
            pix = QPixmap()
            if pix.loadFromData(raw_match.preview_png):
                c_img_lbl.setPixmap(pix.scaled(224, 224, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        elif self.result.thumbnail_path:
            full_thumb_p = self.data_dir / self.result.thumbnail_path
            if full_thumb_p.is_file():
                pix = QPixmap(str(full_thumb_p))
                if not pix.isNull():
                    c_img_lbl.setPixmap(pix.scaled(224, 224, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        c_vbox.addWidget(c_img_lbl, 1)

        info_lbl = QLabel(
            f"Code: {code_lbl}\nFile: {self.result.pdf_name}\n{getattr(self.result, 'design_index_info', '') or ''}",
            c_frame,
        )
        info_lbl.setStyleSheet("color: #CBD5E1; font-size: 11px; padding: 4px;")
        info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c_vbox.addWidget(info_lbl)

        panels_layout.addWidget(c_frame)
        layout.addLayout(panels_layout, 1)

        # Close button
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        close_btn = QPushButton("Close Preview", self)
        close_btn.setStyleSheet(
            "background: #2563EB; color: white; font-weight: 600; padding: 6px 16px; border-radius: 4px;"
        )
        close_btn.clicked.connect(self.accept)
        btn_box.addWidget(close_btn)
        layout.addLayout(btn_box)


class SettingsDialog(QDialog):
    """Configuration dialog for catalog root folder, DXF folders, DPI, and search preferences."""

    def __init__(self, catalog_db, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.db = catalog_db
        self.setWindowTitle("Design Matcher - Settings")
        self.setFixedSize(540, 420)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # 1. Root folder
        layout.addWidget(QLabel("Catalog Root Directory:", self))
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit(self)
        self.folder_input.setPlaceholderText("Select folder containing PDF catalogs...")
        folder_layout.addWidget(self.folder_input)

        browse_btn = QPushButton("Browse...", self)
        browse_btn.clicked.connect(self._browse_folder)
        folder_layout.addWidget(browse_btn)
        layout.addLayout(folder_layout)

        # 1b. DXF Folders Directory
        layout.addWidget(QLabel("DXF Folders Directory (Optional if inside root):", self))
        dxf_folder_layout = QHBoxLayout()
        self.dxf_folder_input = QLineEdit(self)
        self.dxf_folder_input.setPlaceholderText("Select folder containing DXF CAD files...")
        dxf_folder_layout.addWidget(self.dxf_folder_input)

        dxf_browse_btn = QPushButton("Browse...", self)
        dxf_browse_btn.clicked.connect(self._browse_dxf_folder)
        dxf_folder_layout.addWidget(dxf_browse_btn)
        layout.addLayout(dxf_folder_layout)

        # 1c. Include DXF Toggle
        self.include_dxf_cb = QCheckBox("Include DXF CAD files in search results", self)
        self.include_dxf_cb.setStyleSheet("font-size: 12px; font-weight: 500; color: #1E293B;")
        layout.addWidget(self.include_dxf_cb)

        # 2. Rendering DPI
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("PDF Rendering DPI:", self))
        self.dpi_combo = QComboBox(self)
        self.dpi_combo.addItems(["150 DPI (Fast)", "200 DPI (Recommended)", "300 DPI (High Detail)"])
        self.dpi_combo.setCurrentIndex(1)
        dpi_layout.addWidget(self.dpi_combo)
        layout.addLayout(dpi_layout)

        # 3. Top-k results
        topk_layout = QHBoxLayout()
        topk_layout.addWidget(QLabel("Number of Search Results:", self))
        self.topk_spin = QSpinBox(self)
        self.topk_spin.setRange(1, 20)
        self.topk_spin.setValue(5)
        topk_layout.addWidget(self.topk_spin)
        layout.addLayout(topk_layout)

        # 3b. Chamfer distance tolerance
        chamfer_layout = QHBoxLayout()
        chamfer_layout.addWidget(QLabel("DXF Chamfer Tolerance (0.05 tight - 0.25 loose):", self))
        self.chamfer_spin = QDoubleSpinBox(self)
        self.chamfer_spin.setRange(0.02, 0.50)
        self.chamfer_spin.setSingleStep(0.01)
        self.chamfer_spin.setValue(0.12)
        chamfer_layout.addWidget(self.chamfer_spin)
        layout.addLayout(chamfer_layout)

        # 3c. Candidate pool limit
        cand_layout = QHBoxLayout()
        cand_layout.addWidget(QLabel("Verification Candidate Limit (Stage 1 -> Stage 2):", self))
        self.cand_spin = QSpinBox(self)
        self.cand_spin.setRange(5, 100)
        self.cand_spin.setValue(30)
        cand_layout.addWidget(self.cand_spin)
        layout.addLayout(cand_layout)

        # 4. Buttons
        layout.addStretch()
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save Settings", self)
        save_btn.setStyleSheet("background: #2563EB; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px;")
        save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Catalog Root Directory", self.folder_input.text())
        if folder:
            self.folder_input.setText(folder)

    def _browse_dxf_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select DXF CAD Directory", self.dxf_folder_input.text())
        if folder:
            self.dxf_folder_input.setText(folder)

    def _load_settings(self) -> None:
        root_dir = self.db.get_setting("catalog_root", "")
        self.folder_input.setText(root_dir)

        dxf_dir = self.db.get_setting("dxf_root", "")
        self.dxf_folder_input.setText(dxf_dir)

        inc_dxf = self.db.get_setting("include_dxf", "true").lower() in ("true", "1", "yes")
        self.include_dxf_cb.setChecked(inc_dxf)

        dpi_val = self.db.get_setting("dpi", "200")
        if dpi_val == "150":
            self.dpi_combo.setCurrentIndex(0)
        elif dpi_val == "300":
            self.dpi_combo.setCurrentIndex(2)
        else:
            self.dpi_combo.setCurrentIndex(1)

        top_k_str = self.db.get_setting("top_k", "5")
        top_k = int(top_k_str) if top_k_str else 5
        self.topk_spin.setValue(top_k)

        chamfer_str = self.db.get_setting("chamfer_threshold", "0.12")
        chamfer_val = float(chamfer_str) if chamfer_str else 0.12
        self.chamfer_spin.setValue(chamfer_val)

        cand_str = self.db.get_setting("candidate_limit", "30")
        cand_val = int(cand_str) if cand_str else 30
        self.cand_spin.setValue(cand_val)

    def _save_settings(self) -> None:
        self.db.set_setting("catalog_root", self.folder_input.text().strip())
        self.db.set_setting("dxf_root", self.dxf_folder_input.text().strip())
        self.db.set_setting("include_dxf", "true" if self.include_dxf_cb.isChecked() else "false")
        dpi_map = {0: "150", 1: "200", 2: "300"}
        self.db.set_setting("dpi", dpi_map.get(self.dpi_combo.currentIndex(), "200"))
        self.db.set_setting("top_k", str(self.topk_spin.value()))
        self.db.set_setting("chamfer_threshold", str(self.chamfer_spin.value()))
        self.db.set_setting("candidate_limit", str(self.cand_spin.value()))
        self.accept()


class CropImageLabel(QLabel):
    """Compatibility crop image label."""

    def __init__(self, image_bgr: np.ndarray, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.image_bgr = image_bgr
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()

        h, w = image_bgr.shape[:2]
        self.scale_factor = 1.0
        max_dim = max(h, w)
        if max_dim > 900:
            self.scale_factor = 900.0 / float(max_dim)
            display_w = int(w * self.scale_factor)
            display_h = int(h * self.scale_factor)
            resized = cv2.resize(image_bgr, (display_w, display_h), interpolation=cv2.INTER_AREA)
            self.setPixmap(cv2_to_qpixmap(resized))
        else:
            self.setPixmap(cv2_to_qpixmap(image_bgr))

        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event) -> None:
        if not self.origin.isNull():
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def reset_selection(self) -> None:
        self.rubber_band.hide()
        self.origin = QPoint()

    def get_selected_crop_rect(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.rubber_band.isVisible():
            return None
        geom = self.rubber_band.geometry()
        inv_scale = 1.0 / self.scale_factor
        x = int(geom.x() * inv_scale)
        y = int(geom.y() * inv_scale)
        w = int(geom.width() * inv_scale)
        h = int(geom.height() * inv_scale)

        img_h, img_w = self.image_bgr.shape[:2]
        x1 = max(0, min(img_w - 1, x))
        y1 = max(0, min(img_h - 1, y))
        x2 = max(0, min(img_w, x + w))
        y2 = max(0, min(img_h, y + h))

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        return (x1, y1, x2 - x1, y2 - y1)


class StudioCanvasWidget(QWidget):
    """
    Advanced interactive canvas providing:
    - Mode A: Rectangular Crop with 8-handle resize & drag-to-move
    - Mode B: 4-Corner Perspective Warp pins with quad polygon
    - Smooth mouse wheel zoom (0.25x to 6.0x) and pan (right/middle drag or spacebar)
    - Auto-detection of sheet & door panel boundaries
    - Real-time signals for instant stencil mask updating
    """

    selection_changed = Signal()
    points_changed = Signal()  # compatibility with WarpCanvasWidget

    HANDLE_COLORS = [
        QColor("#06B6D4"),  # TL - Cyan
        QColor("#F97316"),  # TR - Orange
        QColor("#10B981"),  # BR - Emerald
        QColor("#8B5CF6"),  # BL - Purple
    ]
    HANDLE_NAMES = ["TL", "TR", "BR", "BL"]

    def __init__(
        self,
        image_bgr: np.ndarray,
        initial_mode: str = "rect",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.image_bgr = image_bgr
        self.img_h, self.img_w = image_bgr.shape[:2]
        self.qimage = self._create_qimage(image_bgr)
        self.mode = initial_mode  # "rect" or "warp"

        self.setMouseTracking(True)
        self.setMinimumSize(540, 480)

        # Viewport zoom & pan
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_pos = QPoint()

        # Rect selection state: [x, y, w, h] in image coords
        mx = max(10, int(self.img_w * 0.05))
        my = max(10, int(self.img_h * 0.05))
        self.crop_rect = [float(mx), float(my), float(max(20, self.img_w - 2 * mx)), float(max(20, self.img_h - 2 * my))]
        self.active_rect_handle: Optional[int] = None
        self.hovered_rect_handle: Optional[int] = None
        self.is_moving_rect = False
        self.is_creating_rect = False
        self.drag_start_pt = QPointF()
        self.drag_start_rect = list(self.crop_rect)

        # Warp selection state: 4 points in image coords [TL, TR, BR, BL]
        self.pts_img = np.zeros((4, 2), dtype=np.float32)
        self.reset_corners_to_defaults()
        self.active_warp_handle: Optional[int] = None
        self.hovered_warp_handle: Optional[int] = None

        self.handle_radius_px = 9

    def _create_qimage(self, bgr: np.ndarray) -> QImage:
        self._qimage_buffer = np.ascontiguousarray(bgr)
        h, w = self._qimage_buffer.shape[:2]
        bytes_per_line = w * 3
        return QImage(self._qimage_buffer.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)

    def set_mode(self, mode: str) -> None:
        if mode in {"rect", "warp"} and mode != self.mode:
            self.mode = mode
            self.update()
            self.selection_changed.emit()

    def reset_corners_to_defaults(self, margin_ratio: float = 0.05) -> None:
        mx = self.img_w * margin_ratio
        my = self.img_h * margin_ratio
        self.pts_img = np.array(
            [
                [mx, my],                              # TL
                [self.img_w - mx, my],                 # TR
                [self.img_w - mx, self.img_h - my],    # BR
                [mx, self.img_h - my],                 # BL
            ],
            dtype=np.float32,
        )
        self.update()
        self.selection_changed.emit()
        self.points_changed.emit()

    def reset_rect_to_defaults(self, margin_ratio: float = 0.05) -> None:
        mx = self.img_w * margin_ratio
        my = self.img_h * margin_ratio
        self.crop_rect = [mx, my, max(20.0, self.img_w - 2 * mx), max(20.0, self.img_h - 2 * my)]
        self.update()
        self.selection_changed.emit()

    def set_points(self, pts_4x2: np.ndarray) -> None:
        if pts_4x2 is not None and len(pts_4x2) == 4:
            self.pts_img = pts_4x2.astype(np.float32).copy()
            self.update()
            self.selection_changed.emit()
            self.points_changed.emit()

    def reset_zoom_and_pan(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update()

    def auto_detect_panel(self) -> bool:
        quad = detect_sheet_quadrilateral(self.image_bgr)
        if quad is not None:
            self.pts_img = quad.copy()
            min_x = max(0.0, float(np.min(quad[:, 0])))
            max_x = min(float(self.img_w), float(np.max(quad[:, 0])))
            min_y = max(0.0, float(np.min(quad[:, 1])))
            max_y = min(float(self.img_h), float(np.max(quad[:, 1])))
            self.crop_rect = [min_x, min_y, max(20.0, max_x - min_x), max(20.0, max_y - min_y)]
            self.update()
            self.selection_changed.emit()
            self.points_changed.emit()
            return True
        return False

    def auto_focus_cutout_pattern(self) -> bool:
        """Finds decorative laser-cut pattern inside image and snaps crop_rect to it."""
        from app.core.preprocess import auto_focus_cutout_box
        box = auto_focus_cutout_box(self.image_bgr)
        if box is not None:
            bx, by, bw, bh = box
            self.crop_rect = [float(bx), float(by), float(bw), float(bh)]
            self.mode = "rect"
            self.update()
            self.selection_changed.emit()
            return True
        return False

    def get_cropped_image(self, sharpen_output: bool = True) -> Optional[np.ndarray]:
        if self.mode == "rect":
            return crop_rect(self.image_bgr, self.crop_rect)
        else:
            return warp_four_points(
                self.image_bgr,
                self.pts_img,
                sharpen_output=sharpen_output,
                upscale_small=True,
            )

    def _get_transform_params(self) -> Tuple[float, int, int, int, int]:
        pad = 20
        avail_w = max(50, self.width() - (2 * pad))
        avail_h = max(50, self.height() - (2 * pad))
        base_scale = min(avail_w / self.img_w, avail_h / self.img_h)
        scale = base_scale * self.zoom
        disp_w = int(self.img_w * scale)
        disp_h = int(self.img_h * scale)
        offset_x = (self.width() - disp_w) // 2 + self.pan_x
        offset_y = (self.height() - disp_h) // 2 + self.pan_y
        return scale, offset_x, offset_y, disp_w, disp_h

    def _img_to_widget(self, ix: float, iy: float) -> QPointF:
        scale, off_x, off_y, _, _ = self._get_transform_params()
        vx, vy = image_to_view_coords(ix, iy, scale, off_x, off_y)
        return QPointF(vx, vy)

    def _widget_to_img(self, wx: float, wy: float) -> Tuple[float, float]:
        scale, off_x, off_y, _, _ = self._get_transform_params()
        return view_to_image_coords(wx, wy, scale, off_x, off_y, self.img_w, self.img_h, clamp=True)

    def _get_rect_handle_points(self) -> List[Tuple[float, float]]:
        rx, ry, rw, rh = self.crop_rect
        return [
            (rx, ry),                   # 0: TL
            (rx + rw / 2.0, ry),        # 1: T
            (rx + rw, ry),              # 2: TR
            (rx + rw, ry + rh / 2.0),   # 3: R
            (rx + rw, ry + rh),         # 4: BR
            (rx + rw / 2.0, ry + rh),   # 5: B
            (rx, ry + rh),              # 6: BL
            (rx, ry + rh / 2.0),        # 7: L
        ]

    def _find_rect_handle_near(self, pos: QPoint) -> Optional[int]:
        handles = self._get_rect_handle_points()
        threshold = self.handle_radius_px + 7
        for i, (hx, hy) in enumerate(handles):
            wpt = self._img_to_widget(hx, hy)
            dist = np.hypot(pos.x() - wpt.x(), pos.y() - wpt.y())
            if dist <= threshold:
                return i

        # Check inside rectangle
        rx, ry, rw, rh = self.crop_rect
        p_tl = self._img_to_widget(rx, ry)
        p_br = self._img_to_widget(rx + rw, ry + rh)
        rect_w = QRectF(p_tl, p_br).normalized()
        if rect_w.contains(pos):
            return -1  # inside box
        return None

    def _find_warp_handle_near(self, pos: QPoint) -> Optional[int]:
        threshold = self.handle_radius_px + 8
        for i in range(4):
            wpt = self._img_to_widget(self.pts_img[i, 0], self.pts_img[i, 1])
            dist = np.hypot(pos.x() - wpt.x(), pos.y() - wpt.y())
            if dist <= threshold:
                return i
        return None

    def _update_hover_cursor(self, pos: QPoint) -> None:
        if self.mode == "warp":
            hover = self._find_warp_handle_near(pos)
            self.hovered_warp_handle = hover
            self.setCursor(Qt.CursorShape.OpenHandCursor if hover is not None else Qt.CursorShape.ArrowCursor)
            self.update()
        else:
            h_idx = self._find_rect_handle_near(pos)
            self.hovered_rect_handle = h_idx if (h_idx is not None and h_idx >= 0) else None
            if h_idx is None:
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif h_idx == -1:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif h_idx in {0, 4}:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif h_idx in {2, 6}:
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif h_idx in {1, 5}:
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif h_idx in {3, 7}:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = np.clip(self.zoom * factor, 0.25, 6.0)
        if abs(new_zoom - self.zoom) > 1e-4:
            self.zoom = float(new_zoom)
            self.update()
            self.selection_changed.emit()

    def mousePressEvent(self, event) -> None:
        # Right click or middle click for panning
        if event.button() in {Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton}:
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "warp":
                idx = self._find_warp_handle_near(event.pos())
                if idx is not None:
                    self.active_warp_handle = idx
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                else:
                    self.is_panning = True
                    self.pan_start_pos = event.pos()
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                h_idx = self._find_rect_handle_near(event.pos())
                ix, iy = self._widget_to_img(event.pos().x(), event.pos().y())
                self.drag_start_pt = QPointF(ix, iy)
                self.drag_start_rect = list(self.crop_rect)

                if h_idx is not None and h_idx >= 0:
                    self.active_rect_handle = h_idx
                elif h_idx == -1:
                    self.is_moving_rect = True
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    self.is_creating_rect = True
                    self.crop_rect = [ix, iy, 1.0, 1.0]

    def mouseMoveEvent(self, event) -> None:
        if self.is_panning:
            dx = event.pos().x() - self.pan_start_pos.x()
            dy = event.pos().y() - self.pan_start_pos.y()
            self.pan_start_pos = event.pos()
            self.pan_x += dx
            self.pan_y += dy
            self.update()
            return

        if self.mode == "warp":
            if self.active_warp_handle is not None:
                ix, iy = self._widget_to_img(event.pos().x(), event.pos().y())
                self.pts_img[self.active_warp_handle] = [ix, iy]
                self.update()
                self.selection_changed.emit()
                self.points_changed.emit()
            else:
                self._update_hover_cursor(event.pos())
        else:
            ix, iy = self._widget_to_img(event.pos().x(), event.pos().y())
            if self.active_rect_handle is not None:
                sx, sy, sw, sh = self.drag_start_rect
                h = self.active_rect_handle

                if h == 0:  # TL
                    nx = min(sx + sw - 10, ix)
                    ny = min(sy + sh - 10, iy)
                    self.crop_rect = [nx, ny, sx + sw - nx, sy + sh - ny]
                elif h == 1:  # T
                    ny = min(sy + sh - 10, iy)
                    self.crop_rect = [sx, ny, sw, sy + sh - ny]
                elif h == 2:  # TR
                    nx2 = max(sx + 10, ix)
                    ny = min(sy + sh - 10, iy)
                    self.crop_rect = [sx, ny, nx2 - sx, sy + sh - ny]
                elif h == 3:  # R
                    nx2 = max(sx + 10, ix)
                    self.crop_rect = [sx, sy, nx2 - sx, sh]
                elif h == 4:  # BR
                    nx2 = max(sx + 10, ix)
                    ny2 = max(sy + 10, iy)
                    self.crop_rect = [sx, sy, nx2 - sx, ny2 - sy]
                elif h == 5:  # B
                    ny2 = max(sy + 10, iy)
                    self.crop_rect = [sx, sy, sw, ny2 - sy]
                elif h == 6:  # BL
                    nx = min(sx + sw - 10, ix)
                    ny2 = max(sy + 10, iy)
                    self.crop_rect = [nx, sy, sx + sw - nx, ny2 - sy]
                elif h == 7:  # L
                    nx = min(sx + sw - 10, ix)
                    self.crop_rect = [nx, sy, sx + sw - nx, sh]

                self.update()
                self.selection_changed.emit()
            elif self.is_moving_rect:
                dx = ix - self.drag_start_pt.x()
                dy = iy - self.drag_start_pt.y()
                sx, sy, sw, sh = self.drag_start_rect
                nx = max(0.0, min(float(self.img_w - sw), sx + dx))
                ny = max(0.0, min(float(self.img_h - sh), sy + dy))
                self.crop_rect = [nx, ny, sw, sh]
                self.update()
                self.selection_changed.emit()
            elif self.is_creating_rect:
                x1 = min(self.drag_start_pt.x(), ix)
                y1 = min(self.drag_start_pt.y(), iy)
                x2 = max(self.drag_start_pt.x(), ix)
                y2 = max(self.drag_start_pt.y(), iy)
                self.crop_rect = [x1, y1, max(10.0, x2 - x1), max(10.0, y2 - y1)]
                self.update()
                self.selection_changed.emit()
            else:
                self._update_hover_cursor(event.pos())

    def mouseReleaseEvent(self, event) -> None:
        self.is_panning = False
        self.active_warp_handle = None
        self.active_rect_handle = None
        self.is_moving_rect = False
        self.is_creating_rect = False
        self._update_hover_cursor(event.pos())
        self.selection_changed.emit()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Background canvas
        painter.fillRect(self.rect(), QColor("#0B1120"))

        scale, off_x, off_y, disp_w, disp_h = self._get_transform_params()
        if disp_w <= 0 or disp_h <= 0:
            return

        dest_rect = QRect(off_x, off_y, disp_w, disp_h)
        painter.drawImage(dest_rect, self.qimage)

        if self.mode == "warp":
            poly = QPolygonF([self._img_to_widget(self.pts_img[i, 0], self.pts_img[i, 1]) for i in range(4)])
            painter.setPen(QPen(QColor(6, 182, 212, 220), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(6, 182, 212, 45)))
            painter.drawPolygon(poly)

            # Draw pins
            font = QFont("Segoe UI", 9, QFont.Weight.Bold)
            painter.setFont(font)

            for i in range(4):
                wpt = self._img_to_widget(self.pts_img[i, 0], self.pts_img[i, 1])
                color = self.HANDLE_COLORS[i]
                is_active = i == self.active_warp_handle
                is_hover = i == self.hovered_warp_handle

                r = self.handle_radius_px + (3 if (is_active or is_hover) else 0)

                painter.setPen(QPen(QColor("#FFFFFF"), 2))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(wpt, r, r)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#FFFFFF")))
                painter.drawEllipse(wpt, 3, 3)

                label = self.HANDLE_NAMES[i]
                painter.setPen(QPen(QColor("#FFFFFF")))
                painter.drawText(int(wpt.x() + r + 5), int(wpt.y() + 4), label)
        else:
            rx, ry, rw, rh = self.crop_rect
            p_tl = self._img_to_widget(rx, ry)
            p_br = self._img_to_widget(rx + rw, ry + rh)
            rect_w = QRectF(p_tl, p_br).normalized()

            # Selection rect
            painter.setPen(QPen(QColor("#2563EB"), 2, Qt.PenStyle.SolidLine))
            painter.setBrush(QBrush(QColor(37, 99, 235, 35)))
            painter.drawRect(rect_w)

            # 8 Handles
            handles = self._get_rect_handle_points()
            for i, (hx, hy) in enumerate(handles):
                wpt = self._img_to_widget(hx, hy)
                is_hover = i == self.hovered_rect_handle or i == self.active_rect_handle
                r = self.handle_radius_px + (2 if is_hover else 0)

                painter.setPen(QPen(QColor("#2563EB"), 2))
                painter.setBrush(QBrush(QColor("#FFFFFF") if not is_hover else QColor("#93C5FD")))
                painter.drawEllipse(wpt, r, r)


class AdvancedCropStudio(QDialog):
    """
    Advanced Crop & Perspective Studio.
    Provides:
    - Interactive 8-handle rectangular crop & 4-corner perspective warp
    - Zoom and pan across high-res customer gate/door photos
    - Live side-by-side stencil mask preview with real-time tuning
    - Presets for Raw Sheet on Floor, Installed Gate/Door, Backlit Silhouettes, and Blurry Photos
    - Sensitivity slider, margin trim, and polarity invert tools
    """

    def __init__(
        self,
        image_bgr: np.ndarray,
        initial_mode: str = "rect",
        parent: Optional[QWidget] = None,
        preset_mode_idx: int = 0,
        is_inverted: bool = False,
        enhance: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Crop & Perspective Studio - Design Matcher")
        self.resize(1080, 720)
        self.original_bgr = image_bgr
        self.initial_mode = initial_mode
        self.initial_mode_idx = preset_mode_idx
        self.cropped_bgr: Optional[np.ndarray] = None
        self.warped_bgr: Optional[np.ndarray] = None  # alias
        self.final_mask: Optional[np.ndarray] = None
        self.trigger_search: bool = False
        self.is_inverted: bool = is_inverted
        self.initial_enhance: bool = enhance

        self._setup_ui()
        QTimer.singleShot(100, self._update_live_mask_preview)

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        # 1. Top Bar: Modes & Zoom Controls
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        mode_lbl = QLabel("Crop Mode:", self)
        mode_lbl.setStyleSheet("font-weight: 700; color: #1E293B; font-size: 12px;")
        top_bar.addWidget(mode_lbl)

        self.rect_mode_btn = QPushButton("Rectangle Crop (8 Handles)", self)
        self.rect_mode_btn.setCheckable(True)
        self.rect_mode_btn.setChecked(self.initial_mode == "rect")
        self.rect_mode_btn.clicked.connect(lambda: self._switch_mode("rect"))
        top_bar.addWidget(self.rect_mode_btn)

        self.warp_mode_btn = QPushButton("4-Corner Perspective Warp", self)
        self.warp_mode_btn.setCheckable(True)
        self.warp_mode_btn.setChecked(self.initial_mode == "warp")
        self.warp_mode_btn.clicked.connect(lambda: self._switch_mode("warp"))
        top_bar.addWidget(self.warp_mode_btn)

        top_bar.addSpacing(16)

        zoom_in_btn = QPushButton("+ Zoom In", self)
        zoom_in_btn.clicked.connect(self._zoom_in)
        top_bar.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("- Zoom Out", self)
        zoom_out_btn.clicked.connect(self._zoom_out)
        top_bar.addWidget(zoom_out_btn)

        fit_btn = QPushButton("Fit Image", self)
        fit_btn.clicked.connect(self._fit_image)
        top_bar.addWidget(fit_btn)

        top_bar.addSpacing(16)

        auto_detect_btn = QPushButton("Auto-Detect Panel", self)
        auto_detect_btn.setStyleSheet("background: #0284C7; color: white; font-weight: 600; padding: 4px 10px; border-radius: 4px;")
        auto_detect_btn.clicked.connect(self._auto_detect)
        top_bar.addWidget(auto_detect_btn)

        focus_btn = QPushButton("🎯 Auto-Focus Pattern", self)
        focus_btn.setToolTip("Automatically snaps crop box around the decorative laser-cut pattern (ignoring handles and borders)")
        focus_btn.setStyleSheet("background: #059669; color: white; font-weight: 600; padding: 4px 10px; border-radius: 4px;")
        focus_btn.clicked.connect(self._auto_focus_pattern)
        top_bar.addWidget(focus_btn)

        reset_btn = QPushButton("Reset Selection", self)
        reset_btn.clicked.connect(self._reset_selection)
        top_bar.addWidget(reset_btn)

        top_bar.addStretch()

        self.stats_lbl = QLabel("", self)
        self.stats_lbl.setStyleSheet("color: #64748B; font-size: 11px; font-weight: 600;")
        top_bar.addWidget(self.stats_lbl)

        main_layout.addLayout(top_bar)

        # 2. Main Central Splitter: Canvas (Left) + Live Mask Studio (Right)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Left: Canvas
        self.canvas = StudioCanvasWidget(self.original_bgr, initial_mode=self.initial_mode, parent=splitter)
        self.canvas.selection_changed.connect(self._on_selection_changed)
        splitter.addWidget(self.canvas)

        # Right: Live Mask Studio Panel
        right_panel = QFrame(splitter)
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        right_panel.setStyleSheet("background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;")
        right_panel.setMinimumWidth(320)
        right_panel.setMaximumWidth(380)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        mask_hdr = QHBoxLayout()
        mask_title = QLabel("LIVE STENCIL MASK", right_panel)
        mask_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #1E293B;")
        mask_hdr.addWidget(mask_title)
        mask_hdr.addStretch()

        self.polarity_badge = QLabel("Metal: Solid (White)", right_panel)
        self.polarity_badge.setStyleSheet(
            "background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px;"
        )
        mask_hdr.addWidget(self.polarity_badge)
        right_layout.addLayout(mask_hdr)

        self.mask_frame = QFrame(right_panel)
        self.mask_frame.setStyleSheet("background: #0F172A; border: 1px solid #334155; border-radius: 6px;")
        self.mask_frame.setFixedHeight(220)
        mask_frame_layout = QVBoxLayout(self.mask_frame)
        mask_frame_layout.setContentsMargins(6, 6, 6, 6)

        self.mask_label = QLabel(self.mask_frame)
        self.mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mask_frame_layout.addWidget(self.mask_label)
        right_layout.addWidget(self.mask_frame)

        # Preset mode selector
        preset_lbl = QLabel("Target Photo Type:", right_panel)
        preset_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #475569;")
        right_layout.addWidget(preset_lbl)

        self.mode_combo = QComboBox(right_panel)
        self.mode_combo.addItems([
            "Auto (Smart - All Backgrounds)",
            "Raw Sheet on Cardboard / Floor",
            "Installed Gate / Safety Door",
            "Backlit Silhouette (Daylight Behind)",
            "Mixed Lighting (Sky & Shadows)",
            "Blurry / WhatsApp Photo",
        ])
        if 0 <= self.initial_mode_idx < self.mode_combo.count():
            self.mode_combo.setCurrentIndex(self.initial_mode_idx)
        self.mode_combo.currentIndexChanged.connect(self._update_live_mask_preview)
        right_layout.addWidget(self.mode_combo)

        # Sensitivity / Threshold Slider
        sens_row = QHBoxLayout()
        sens_lbl = QLabel("Cutout Sensitivity:", right_panel)
        sens_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #475569;")
        sens_row.addWidget(sens_lbl)
        sens_row.addStretch()
        self.sens_val_lbl = QLabel("Normal (0)", right_panel)
        self.sens_val_lbl.setStyleSheet("font-size: 11px; color: #2563EB; font-weight: 600;")
        sens_row.addWidget(self.sens_val_lbl)
        right_layout.addLayout(sens_row)

        self.thresh_slider = QSlider(Qt.Orientation.Horizontal, right_panel)
        self.thresh_slider.setRange(-40, 40)
        self.thresh_slider.setValue(0)
        self.thresh_slider.valueChanged.connect(self._on_slider_changed)
        right_layout.addWidget(self.thresh_slider)

        # Cleaning toggles
        self.trim_check = QCheckBox("Auto-trim 2% margins (removes frame bleed)", right_panel)
        self.trim_check.setChecked(True)
        self.trim_check.stateChanged.connect(self._update_live_mask_preview)
        right_layout.addWidget(self.trim_check)

        self.enhance_check = QCheckBox("Deblur & sharpen aperture edges", right_panel)
        self.enhance_check.setChecked(self.initial_enhance)
        self.enhance_check.stateChanged.connect(self._update_live_mask_preview)
        right_layout.addWidget(self.enhance_check)

        self.sharpen_check = QCheckBox("Gentle unsharp mask after warp", right_panel)
        self.sharpen_check.setChecked(True)
        self.sharpen_check.stateChanged.connect(self._update_live_mask_preview)
        right_layout.addWidget(self.sharpen_check)

        invert_btn = QPushButton("Invert Polarity (Black / White)", right_panel)
        invert_btn.clicked.connect(self._toggle_invert)
        right_layout.addWidget(invert_btn)

        right_layout.addStretch()

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter, 1)

        # 3. Bottom Actions
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        bottom_row.addWidget(cancel_btn)

        bottom_row.addStretch()

        apply_btn = QPushButton("Apply Crop", self)
        apply_btn.setStyleSheet(
            "QPushButton { background: #475569; color: white; font-weight: 700; padding: 7px 18px; border-radius: 5px; } "
            "QPushButton:hover { background: #334155; }"
        )
        apply_btn.clicked.connect(self._apply_crop_only)
        bottom_row.addWidget(apply_btn)

        apply_search_btn = QPushButton("Apply & Search Closest Design", self)
        apply_search_btn.setStyleSheet(
            "QPushButton { background: #2563EB; color: white; font-weight: 700; padding: 7px 22px; border-radius: 5px; font-size: 13px; } "
            "QPushButton:hover { background: #1D4ED8; }"
        )
        apply_search_btn.clicked.connect(self._apply_and_search)
        bottom_row.addWidget(apply_search_btn)

        main_layout.addLayout(bottom_row)

    def _switch_mode(self, mode: str) -> None:
        self.rect_mode_btn.setChecked(mode == "rect")
        self.warp_mode_btn.setChecked(mode == "warp")
        self.canvas.set_mode(mode)
        self._update_live_mask_preview()

    def _zoom_in(self) -> None:
        self.canvas.zoom = float(np.clip(self.canvas.zoom * 1.25, 0.25, 6.0))
        self.canvas.update()
        self._update_stats()

    def _zoom_out(self) -> None:
        self.canvas.zoom = float(np.clip(self.canvas.zoom / 1.25, 0.25, 6.0))
        self.canvas.update()
        self._update_stats()

    def _fit_image(self) -> None:
        self.canvas.reset_zoom_and_pan()
        self._update_stats()

    def _reset_selection(self) -> None:
        if self.canvas.mode == "warp":
            self.canvas.reset_corners_to_defaults()
        else:
            self.canvas.reset_rect_to_defaults()

    def _auto_detect(self) -> None:
        detected = self.canvas.auto_detect_panel()
        if detected:
            self._update_live_mask_preview()
            QMessageBox.information(
                self,
                "Panel Detected",
                "Prominent metal panel boundary detected automatically! Adjust pins/edges if needed.",
            )
        else:
            QMessageBox.information(
                self,
                "Auto Detection",
                "Could not cleanly detect 4 corners automatically. Please drag the handles manually.",
            )

    def _auto_focus_pattern(self) -> None:
        if self.canvas.mode != "rect":
            self._switch_mode("rect")
        detected = self.canvas.auto_focus_cutout_pattern()
        if detected:
            self._update_live_mask_preview()
            self._update_stats()
            QMessageBox.information(
                self,
                "Pattern Focused",
                "Successfully focused crop box on the decorative laser-cut pattern!\nDoor handles, locks, and frames were excluded.",
            )
        else:
            QMessageBox.information(
                self,
                "Auto-Focus Pattern",
                "Could not cleanly detect an isolated pattern cluster. Please adjust the 8 handles manually over the pattern.",
            )

    def _on_selection_changed(self) -> None:
        self._update_stats()
        self._update_live_mask_preview()

    def _update_stats(self) -> None:
        if self.canvas.mode == "rect":
            _, _, w, h = [int(v) for v in self.canvas.crop_rect]
            ar = (w / h) if h > 0 else 1.0
            self.stats_lbl.setText(f"Crop: {w} × {h} px (Ratio 1:{1.0/ar:.2f}) | Zoom: {int(self.canvas.zoom * 100)}%")
        else:
            self.stats_lbl.setText(f"4-Corner Quad Warp | Zoom: {int(self.canvas.zoom * 100)}%")

    def _on_slider_changed(self, val: int) -> None:
        self.sens_val_lbl.setText(f"{val:+d}")
        self._update_live_mask_preview()

    def _toggle_invert(self) -> None:
        self.is_inverted = not self.is_inverted
        self._update_live_mask_preview()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._apply_crop_only()
            return
        elif event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def _update_live_mask_preview(self) -> None:
        sharpen_flag = getattr(self, "sharpen_check", None)
        sharpen_val = sharpen_flag.isChecked() if sharpen_flag else True
        crop = self.canvas.get_cropped_image(sharpen_output=sharpen_val)
        if crop is None or crop.size == 0:
            return

        self.cropped_bgr = crop
        self.warped_bgr = crop

        # Pipeline order: full-res crop/warp -> optional unsharp -> normalize_stencil(crop, crop=False)
        mask, info = normalize_stencil(crop, crop=False)
        self.final_mask = mask

        pixmap = cv2_to_qpixmap(mask)
        dpr = self.devicePixelRatio()
        target_w = max(40, int((self.mask_frame.width() - 16) * dpr))
        target_h = max(40, int((self.mask_frame.height() - 16) * dpr))
        scaled = pixmap.scaled(
            target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        scaled.setDevicePixelRatio(dpr)
        self.mask_label.setPixmap(scaled)

        is_normal = detect_stencil_polarity(mask)
        if is_normal:
            self.polarity_badge.setText("Metal: Solid (White) ✅")
            self.polarity_badge.setStyleSheet(
                "background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px;"
            )
        else:
            self.polarity_badge.setText("Metal: Dark (Inverted) ⚠️")
            self.polarity_badge.setStyleSheet(
                "background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px;"
            )

    def _apply_crop_only(self) -> None:
        self.trigger_search = False
        self._apply_warp()

    def _apply_and_search(self) -> None:
        self.trigger_search = True
        self._apply_warp()

    def _apply_warp(self) -> None:
        sharpen_flag = getattr(self, "sharpen_check", None)
        sharpen_val = sharpen_flag.isChecked() if sharpen_flag else True
        crop = self.canvas.get_cropped_image(sharpen_output=sharpen_val)
        if crop is not None and crop.size > 0:
            self.cropped_bgr = crop
            self.warped_bgr = crop
            mask, info = normalize_stencil(crop, crop=False)
            self.final_mask = mask
            self.accept()
        else:
            QMessageBox.warning(self, "Invalid Selection", "Please make a valid selection before applying.")


# Compatibility wrappers
class WarpCanvasWidget(StudioCanvasWidget):
    """Backward compatibility alias for WarpCanvasWidget."""
    def __init__(self, image_bgr: np.ndarray, parent: Optional[QWidget] = None) -> None:
        super().__init__(image_bgr, initial_mode="warp", parent=parent)


class CornerWarpDialog(AdvancedCropStudio):
    """Backward compatibility alias for CornerWarpDialog."""
    def __init__(
        self,
        image_bgr: np.ndarray,
        parent: Optional[QWidget] = None,
        preset_mode_idx: int = 0,
        is_inverted: bool = False,
        enhance: bool = True,
    ) -> None:
        super().__init__(
            image_bgr,
            initial_mode="warp",
            parent=parent,
            preset_mode_idx=preset_mode_idx,
            is_inverted=is_inverted,
            enhance=enhance,
        )


class CropDialog(AdvancedCropStudio):
    """Backward compatibility alias for CropDialog."""
    def __init__(
        self,
        image_bgr: np.ndarray,
        parent: Optional[QWidget] = None,
        preset_mode_idx: int = 0,
        is_inverted: bool = False,
        enhance: bool = True,
    ) -> None:
        super().__init__(
            image_bgr,
            initial_mode="rect",
            parent=parent,
            preset_mode_idx=preset_mode_idx,
            is_inverted=is_inverted,
            enhance=enhance,
        )

