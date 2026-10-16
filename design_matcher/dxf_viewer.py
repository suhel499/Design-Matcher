"""
In-App DXF Vector Viewer Module for Design Matcher.

Provides hardware-accelerated, crisp vector graphics rendering of 2D CAD DXF files
using Qt QGraphicsView / QGraphicsScene architecture:
- DxfScene: Builds and manages vector items from ezdxf documents.
- DxfView: QGraphicsView subclass with Y-flip, smooth wheel zoom, panning, and measure tool.
- DxfViewerWidget: Embeddable composite widget with toolbar, layer panel, design list, and status bar.
- DxfViewerWindow: Standalone enlarged viewer window.
"""

from __future__ import annotations

import collections
import datetime
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import ezdxf
from ezdxf import colors as ez_colors
from ezdxf import path as ez_path
from PySide6.QtCore import (
    QEvent,
    QPoint,
    QPointF,
    QRectF,
    QSettings,
    QSize,
    Qt,
    QThread,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QDesktopServices,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dxf_ingest import Design, extract_designs, load_dxf

logger = logging.getLogger(__name__)

# AutoCAD $INSUNITS unit names
INSUNITS_MAP: Dict[int, Tuple[str, str]] = {
    0: ("Unitless", ""),
    1: ("Inches", "in"),
    2: ("Feet", "ft"),
    3: ("Miles", "mi"),
    4: ("Millimeters", "mm"),
    5: ("Centimeters", "cm"),
    6: ("Meters", "m"),
    7: ("Kilometers", "km"),
    8: ("Microinches", "μin"),
    9: ("Mils", "mil"),
    10: ("Yards", "yd"),
    11: ("Angstroms", "Å"),
    12: ("Nanometers", "nm"),
    13: ("Microns", "μm"),
    14: ("Decimeters", "dm"),
}

DXF_VERSION_MAP: Dict[str, str] = {
    "AC1009": "AutoCAD R12",
    "AC1012": "AutoCAD R13",
    "AC1014": "AutoCAD R14",
    "AC1015": "AutoCAD 2000",
    "AC1018": "AutoCAD 2004",
    "AC1021": "AutoCAD 2007",
    "AC1024": "AutoCAD 2010",
    "AC1027": "AutoCAD 2013",
    "AC1032": "AutoCAD 2018",
}

DEFAULT_DARK_BG = QColor("#0F172A")    # Deep slate navy
DEFAULT_LIGHT_BG = QColor("#FFFFFF")   # Pure white
HIGHLIGHT_ORANGE = QColor("#F97316")   # Vibrant orange for matched panel
HIGHLIGHT_FILL = QColor(249, 115, 22, 35)
MEASURE_CYAN = QColor("#06B6D4")       # Electric cyan dimension line


def get_insunits_info(code: int) -> Tuple[str, str]:
    """Returns (friendly_name, abbreviation) for AutoCAD $INSUNITS code."""
    return INSUNITS_MAP.get(code, ("Millimeters", "mm"))


def resolve_dxf_color(
    color_code: int,
    true_color: Optional[int] = None,
    layer_color_code: int = 7,
    layer_true_color: Optional[int] = None,
    is_dark_mode: bool = True,
) -> QColor:
    """
    Resolves entity and layer color to a Qt QColor:
    - 256 = BYLAYER (uses layer color)
    - 0 = BYBLOCK (uses layer color)
    - 7 = White on dark background, Black on light background
    - 1-255 = Standard AutoCAD Color Index (ACI)
    - true_color = 24-bit 0xRRGGBB
    """
    # 1. Direct 24-bit true color
    if true_color is not None and true_color > 0:
        r = (true_color >> 16) & 0xFF
        g = (true_color >> 8) & 0xFF
        b = true_color & 0xFF
        return QColor(r, g, b)

    # 2. BYLAYER inheritance
    if color_code == 256 or color_code == 0:
        if layer_true_color is not None and layer_true_color > 0:
            r = (layer_true_color >> 16) & 0xFF
            g = (layer_true_color >> 8) & 0xFF
            b = layer_true_color & 0xFF
            return QColor(r, g, b)
        color_code = abs(layer_color_code)

    # 3. ACI 7 (White/Black contrast)
    if color_code == 7:
        return QColor("#FFFFFF") if is_dark_mode else QColor("#0F172A")

    # 4. Standard ACI Palette
    if 1 <= color_code <= 255:
        try:
            rgb = ez_colors.aci2rgb(color_code)
            return QColor(rgb.r, rgb.g, rgb.b)
        except Exception:
            pass

    return QColor("#F8FAFC") if is_dark_mode else QColor("#1E293B")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class LayerInfo:
    """Metadata and item collection for a single DXF CAD layer."""

    def __init__(self, name: str, color: QColor, is_on: bool = True, is_frozen: bool = False):
        self.name = name
        self.color = color
        self.is_on = is_on
        self.is_frozen = is_frozen
        self.entity_count = 0
        self.items: List[QGraphicsItem] = []


class ParsedDxfData:
    """Container for parsed DXF vector geometry, layers, and extracted designs."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.doc: Optional[ezdxf.document.Drawing] = None
        self.units_code: int = 4  # mm default
        self.units_name: str = "Millimeters"
        self.units_suffix: str = "mm"
        self.drawing_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0)
        self.total_entity_count: int = 0
        self.layers: Dict[str, LayerInfo] = {}
        self.designs: List[Design] = []
        self.is_simplified: bool = False
        self.dxf_version_code: str = ""
        self.dxf_version_name: str = ""
        self.entity_counts_by_type: Dict[str, int] = {}
        self.file_size_bytes: int = 0

        # Pre-grouped paths: List of (path, layer_name, color, is_text)
        self.vector_items: List[Tuple[QPainterPath, str, QColor]] = []
        self.text_items: List[Tuple[str, float, float, float, float, str, QColor]] = []

    @property
    def total_entities(self) -> int:
        return self.total_entity_count


# ---------------------------------------------------------------------------
# Background DXF Loader & LRU Cache
# ---------------------------------------------------------------------------

class DxfSceneCache:
    """LRU memory cache holding up to 5 parsed DXF drawings."""

    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self._cache: collections.OrderedDict[Path, ParsedDxfData] = collections.OrderedDict()

    def get(self, path: Path | str) -> Optional[ParsedDxfData]:
        p = Path(path).resolve()
        if p in self._cache:
            self._cache.move_to_end(p)
            return self._cache[p]
        return None

    def put(self, path: Path | str, data: ParsedDxfData) -> None:
        p = Path(path).resolve()
        if p in self._cache:
            self._cache.move_to_end(p)
        else:
            if len(self._cache) >= self.capacity:
                self._cache.popitem(last=False)
        self._cache[p] = data

    def clear(self) -> None:
        self._cache.clear()


GLOBAL_DXF_CACHE = DxfSceneCache(capacity=5)


def parse_dxf_document(doc: ezdxf.document.Drawing, path: Optional[Path | str] = None) -> ParsedDxfData:
    """
    Parses an ezdxf Drawing document into a structured ParsedDxfData object
    containing vectorized paths, text labels, layer metadata, and design boundaries.
    """
    p_path = Path(path).resolve() if path is not None else Path("untitled.dxf")
    data = ParsedDxfData(p_path)
    data.doc = doc

    # File size and DXF version
    try:
        data.file_size_bytes = p_path.stat().st_size if p_path.is_file() else 0
    except Exception:
        data.file_size_bytes = 0

    ver_code = str(getattr(doc, "dxfversion", "") or "")
    data.dxf_version_code = ver_code
    data.dxf_version_name = DXF_VERSION_MAP.get(ver_code, ver_code or "AutoCAD DXF")

    # Units
    try:
        insunits = int(doc.header.get("$INSUNITS", 4))
        data.units_code = insunits
        uname, usuff = get_insunits_info(insunits)
        data.units_name = uname
        data.units_suffix = usuff
    except Exception:
        data.units_code = 4
        data.units_name = "Millimeters"
        data.units_suffix = "mm"

    # Layer collection
    layer_colors_map: Dict[str, Tuple[int, Optional[int]]] = {}
    for layer in doc.layers:
        lname = layer.dxf.name
        c_idx = layer.color
        is_off = layer.is_off()
        is_frozen = layer.is_frozen()
        tc = getattr(layer.dxf, "true_color", None)
        layer_colors_map[lname] = (abs(c_idx), tc)
        data.layers[lname] = LayerInfo(
            name=lname,
            color=resolve_dxf_color(c_idx, true_color=tc, is_dark_mode=True),
            is_on=not is_off,
            is_frozen=is_frozen,
        )

    # Extract segmented designs using dxf_ingest
    try:
        data.designs = extract_designs(doc)
    except Exception as exc:
        logger.warning("Design extraction warning for %s: %s", p_path.name, exc)
        data.designs = []

    # Parse geometry entities from modelspace
    msp = doc.modelspace()
    all_entities = list(msp)
    data.total_entity_count = len(all_entities)

    # High-entity simplification check (> 200k)
    if data.total_entity_count > 200000:
        data.is_simplified = True

    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    def _process_entity(ent: Any, depth: int = 0) -> None:
        nonlocal min_x, min_y, max_x, max_y
        if depth > 8:
            return

        dxftype = ent.dxftype()
        data.entity_counts_by_type[dxftype] = data.entity_counts_by_type.get(dxftype, 0) + 1
        layer_name = getattr(ent.dxf, "layer", "0")

        if dxftype == "INSERT":
            try:
                for sub in ent.virtual_entities():
                    # Sub-entities on layer 0 inherit the block INSERT layer
                    if getattr(sub.dxf, "layer", "0") == "0":
                        sub.dxf.layer = layer_name
                    _process_entity(sub, depth + 1)
            except Exception as sub_exc:
                logger.debug("Virtual entities error on %s: %s", ent, sub_exc)
            return

        if dxftype in ("TEXT", "MTEXT"):
            try:
                ins = getattr(ent.dxf, "insert", (0.0, 0.0, 0.0))
                tx = float(ins[0])
                ty = float(ins[1])
                th = float(getattr(ent.dxf, "height", 10.0))
                trot = float(getattr(ent.dxf, "rotation", 0.0))
                if dxftype == "MTEXT":
                    raw_t = ent.plain_text() if hasattr(ent, "plain_text") else str(getattr(ent.dxf, "text", ""))
                    th = float(getattr(ent.dxf, "char_height", th))
                else:
                    raw_t = str(getattr(ent.dxf, "text", ""))

                if raw_t.strip():
                    c_idx = getattr(ent.dxf, "color", 256)
                    tc = getattr(ent.dxf, "true_color", None)
                    lc_info = layer_colors_map.get(layer_name, (7, None))
                    qcol = resolve_dxf_color(c_idx, tc, lc_info[0], lc_info[1], is_dark_mode=True)
                    data.text_items.append((raw_t.strip(), tx, ty, th, trot, layer_name, qcol))
                    min_x = min(min_x, tx)
                    min_y = min(min_y, ty)
                    max_x = max(max_x, tx + len(raw_t) * th * 0.7)
                    max_y = max(max_y, ty + th)
            except Exception as text_exc:
                logger.debug("Text extraction warning: %s", text_exc)
            return

        # Geometry entities: convert via ezdxf.path.make_path
        try:
            p = ez_path.make_path(ent)
            if len(p) == 0:
                return

            qpath = QPainterPath()
            qpath.moveTo(p.start.x, p.start.y)
            min_x = min(min_x, p.start.x)
            min_y = min(min_y, p.start.y)
            max_x = max(max_x, p.start.x)
            max_y = max(max_y, p.start.y)

            for cmd in p:
                if isinstance(cmd, ez_path.Curve4To):
                    qpath.cubicTo(cmd.ctrl1.x, cmd.ctrl1.y, cmd.ctrl2.x, cmd.ctrl2.y, cmd.end.x, cmd.end.y)
                    min_x = min(min_x, cmd.end.x)
                    min_y = min(min_y, cmd.end.y)
                    max_x = max(max_x, cmd.end.x)
                    max_y = max(max_y, cmd.end.y)
                elif isinstance(cmd, ez_path.Curve3To):
                    qpath.quadTo(cmd.ctrl.x, cmd.ctrl.y, cmd.end.x, cmd.end.y)
                    min_x = min(min_x, cmd.end.x)
                    min_y = min(min_y, cmd.end.y)
                    max_x = max(max_x, cmd.end.x)
                    max_y = max(max_y, cmd.end.y)
                elif isinstance(cmd, ez_path.LineTo):
                    qpath.lineTo(cmd.end.x, cmd.end.y)
                    min_x = min(min_x, cmd.end.x)
                    min_y = min(min_y, cmd.end.y)
                    max_x = max(max_x, cmd.end.x)
                    max_y = max(max_y, cmd.end.y)
                elif isinstance(cmd, ez_path.MoveTo):
                    qpath.moveTo(cmd.end.x, cmd.end.y)

            if getattr(p, "is_closed", False):
                qpath.closeSubpath()

            c_idx = getattr(ent.dxf, "color", 256)
            tc = getattr(ent.dxf, "true_color", None)
            lc_info = layer_colors_map.get(layer_name, (7, None))
            qcol = resolve_dxf_color(c_idx, tc, lc_info[0], lc_info[1], is_dark_mode=True)

            data.vector_items.append((qpath, layer_name, qcol))

            if layer_name in data.layers:
                data.layers[layer_name].entity_count += 1
        except Exception as geom_exc:
            logger.debug("Failed converting entity %s to path: %s", dxftype, geom_exc)

    for ent in all_entities:
        _process_entity(ent, 0)

    if min_x <= max_x and min_y <= max_y:
        data.drawing_bounds = (min_x, min_y, max_x - min_x, max_y - min_y)
    elif data.designs:
        # Fall back to union of designs
        all_bx = min(d.x for d in data.designs)
        all_by = min(d.y for d in data.designs)
        all_mx = max(d.x + d.width for d in data.designs)
        all_my = max(d.y + d.height for d in data.designs)
        data.drawing_bounds = (all_bx, all_by, all_mx - all_bx, all_my - all_by)
    else:
        data.drawing_bounds = (0.0, 0.0, 100.0, 100.0)

    # If no separate panels extracted, provide fallback design so designs list is never empty
    if not data.designs:
        bx, by, bw, bh = data.drawing_bounds
        if bw > 0 and bh > 0:
            data.designs = [
                Design(
                    design_id=f"{p_path.stem}_full",
                    design_code=p_path.stem,
                    bbox=(bx, by, bx + bw, by + bh),
                    source_path=str(p_path),
                    entity_count=data.total_entity_count,
                    drawing_units=data.units_code,
                )
            ]

    return data


class DxfLoaderWorker(QThread):
    """Background worker thread that parses DXF files without blocking the UI."""

    loaded_signal = Signal(ParsedDxfData)
    error_signal = Signal(str, str)  # (friendly_msg, tech_details)

    def __init__(self, path: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.path = Path(path).resolve()

    def run(self) -> None:
        # Check LRU cache first
        cached = GLOBAL_DXF_CACHE.get(self.path)
        if cached is not None:
            self.loaded_signal.emit(cached)
            return

        if not self.path.is_file():
            self.error_signal.emit(
                f"CAD file not found:\n{self.path.name}",
                f"File does not exist at path:\n{self.path}",
            )
            return

        if self.path.suffix.lower() == ".dwg":
            self.error_signal.emit(
                f"DWG format is not directly supported for '{self.path.name}'.",
                "Please open this drawing in AutoCAD or DWG TrueView and export/save it as DXF format.",
            )
            return

        try:
            doc = load_dxf(self.path)
            parsed = parse_dxf_document(doc, self.path)
            GLOBAL_DXF_CACHE.put(self.path, parsed)
            self.loaded_signal.emit(parsed)
        except Exception as exc:
            logger.exception("Error loading DXF file %s: %s", self.path.name, exc)
            self.error_signal.emit(
                f"Failed to open DXF file '{self.path.name}'.\nThe file may be corrupted or invalid.",
                str(exc),
            )


# ---------------------------------------------------------------------------
# Graphics Scene (DxfScene)
# ---------------------------------------------------------------------------

class DxfScene(QGraphicsScene):
    """QGraphicsScene hosting DXF vector geometry, layers, and interactive overlays."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.is_dark_mode: bool = True
        self.parsed_data: Optional[ParsedDxfData] = None

        # Layer tracking: layer_name -> list of QGraphicsItems
        self.layer_items: Dict[str, List[QGraphicsItem]] = {}

        # Design panel boundaries
        self.design_rect_items: List[QGraphicsRectItem] = []

        # Highlight items for active matched design
        self.highlight_frame: Optional[QGraphicsRectItem] = None
        self.highlight_label: Optional[QGraphicsSimpleTextItem] = None

        # Measurement items
        self.measure_line_item: Optional[QGraphicsItem] = None
        self.measure_text_item: Optional[QGraphicsSimpleTextItem] = None

        self._init_scene_appearance()

    @property
    def drawing_bounds(self) -> QRectF:
        if self.parsed_data is not None:
            bx, by, bw, bh = self.parsed_data.drawing_bounds
            return QRectF(bx, by, bw, bh)
        return self.itemsBoundingRect()

    @property
    def layer_info(self) -> Dict[str, LayerInfo]:
        return self.parsed_data.layers if self.parsed_data else {}

    @property
    def items_by_layer(self) -> Dict[str, List[QGraphicsItem]]:
        return self.layer_items

    @property
    def aci7_color(self) -> QColor:
        return QColor(255, 255, 255) if self.is_dark_mode else QColor(15, 23, 42)

    def set_background_mode(self, mode: str) -> None:
        self.set_dark_mode(mode.lower() == "dark")

    def load_doc(self, doc: ezdxf.document.Drawing, is_dark_mode: bool = True) -> ParsedDxfData:
        """Parses and renders an ezdxf Document directly into this scene."""
        data = parse_dxf_document(doc)
        self.build_scene(data, is_dark_mode=is_dark_mode)
        return data

    def _init_scene_appearance(self) -> None:
        bg_col = DEFAULT_DARK_BG if self.is_dark_mode else DEFAULT_LIGHT_BG
        self.setBackgroundBrush(QBrush(bg_col))

    def build_scene(self, data: ParsedDxfData, is_dark_mode: bool = True) -> None:
        """Constructs all vector items, text items, and overlays from parsed DXF data."""
        self.clear()
        self.parsed_data = data
        self.is_dark_mode = is_dark_mode
        self._init_scene_appearance()

        self.layer_items = {lname: [] for lname in data.layers.keys()}
        self.design_rect_items.clear()
        self.highlight_frame = None
        self.highlight_label = None
        self.measure_line_item = None
        self.measure_text_item = None

        # 1. Batch vector geometry paths per (layer, color) for high performance
        path_batches: Dict[Tuple[str, int], QPainterPath] = {}
        for qpath, lname, qcol in data.vector_items:
            key = (lname, qcol.rgb())
            if key not in path_batches:
                path_batches[key] = QPainterPath()
            path_batches[key].addPath(qpath)

        for (lname, col_rgb), batch_path in path_batches.items():
            item = QGraphicsPathItem(batch_path)
            qcol = QColor.fromRgb(col_rgb)

            # Contrast adaptation for ACI 7 (white on dark, black on light)
            if col_rgb == 0xFFFFFF and not self.is_dark_mode:
                qcol = QColor("#0F172A")
            elif col_rgb == 0x000000 and self.is_dark_mode:
                qcol = QColor("#FFFFFF")

            pen = QPen(qcol, 1.0, Qt.PenStyle.SolidLine)
            pen.setCosmetic(True)  # Line thickness stays invariant under zoom
            item.setPen(pen)
            item.setBrush(Qt.BrushStyle.NoBrush)

            self.addItem(item)
            self.layer_items.setdefault(lname, []).append(item)

        # 2. Add text items
        for txt, tx, ty, th, trot, lname, qcol in data.text_items:
            titem = QGraphicsSimpleTextItem(txt)
            font = QFont("Segoe UI", max(8, int(th)))
            font.setStyleHint(QFont.StyleHint.SansSerif)
            titem.setFont(font)

            # Invert text Y-axis locally so it renders upright in the Y-flipped QGraphicsView
            t = QTransform()
            t.translate(tx, ty)
            if trot != 0.0:
                t.rotate(-trot)
            t.scale(1.0, -1.0)
            titem.setTransform(t)

            if qcol.rgb() == 0xFFFFFF and not self.is_dark_mode:
                qcol = QColor("#0F172A")
            elif qcol.rgb() == 0x000000 and self.is_dark_mode:
                qcol = QColor("#FFFFFF")

            titem.setBrush(QBrush(qcol))
            self.addItem(titem)
            self.layer_items.setdefault(lname, []).append(titem)

        # 3. Add selectable design panels
        for idx, d in enumerate(data.designs):
            rect_item = QGraphicsRectItem(QRectF(d.x, d.y, d.width, d.height))
            p_col = QColor("#64748B" if self.is_dark_mode else "#94A3B8")
            pen = QPen(p_col, 1.0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            rect_item.setPen(pen)
            rect_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            rect_item.setData(0, idx + 1)
            rect_item.setData(1, d.design_code or f"Design #{idx + 1}")
            rect_item.setZValue(10.0)
            self.addItem(rect_item)
            self.design_rect_items.append(rect_item)

        # 4. Initialize highlight frame item
        self.highlight_frame = QGraphicsRectItem()
        h_pen = QPen(HIGHLIGHT_ORANGE, 2.5, Qt.PenStyle.SolidLine)
        h_pen.setCosmetic(True)
        self.highlight_frame.setPen(h_pen)
        self.highlight_frame.setBrush(QBrush(HIGHLIGHT_FILL))
        self.highlight_frame.setZValue(20.0)
        self.highlight_frame.setVisible(False)
        self.addItem(self.highlight_frame)

        # 5. Initialize highlight label item
        self.highlight_label = QGraphicsSimpleTextItem()
        lbl_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        self.highlight_label.setFont(lbl_font)
        self.highlight_label.setBrush(QBrush(HIGHLIGHT_ORANGE))
        self.highlight_label.setZValue(25.0)
        self.highlight_label.setVisible(False)
        self.addItem(self.highlight_label)

    def set_layer_visible(self, layer_name: str, visible: bool) -> None:
        """Toggles visibility of all items on a specific layer."""
        for item in self.layer_items.get(layer_name, []):
            item.setVisible(visible)

    def set_dark_mode(self, is_dark: bool) -> None:
        """Switches between dark CAD mode and light blueprint mode with contrast adaptation."""
        self.is_dark_mode = is_dark
        self._init_scene_appearance()
        if self.parsed_data is not None:
            self.build_scene(self.parsed_data, is_dark_mode=is_dark)

    def highlight_design(self, bbox: Tuple[float, float, float, float], code_label: str) -> None:
        """Highlights the active design bounding box with an orange frame and text badge."""
        if self.highlight_frame is None or self.highlight_label is None:
            return

        bx, by, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            self.clear_highlight()
            return

        self.highlight_frame.setRect(QRectF(bx, by, bw, bh))
        self.highlight_frame.setVisible(True)

        # Label tag placed right below panel border in drawing coordinates
        self.highlight_label.setText(f"★ {code_label}" if code_label else "Matched Design")
        t = QTransform()
        t.translate(bx, by - 12.0)
        t.scale(1.0, -1.0)  # Upright in Y-flipped view
        self.highlight_label.setTransform(t)
        self.highlight_label.setVisible(True)

    def clear_highlight(self) -> None:
        """Removes the highlight frame and label."""
        if self.highlight_frame is not None:
            self.highlight_frame.setVisible(False)
        if self.highlight_label is not None:
            self.highlight_label.setVisible(False)


# ---------------------------------------------------------------------------
# Vector Graphics View (DxfView)
# ---------------------------------------------------------------------------

class DxfView(QGraphicsView):
    """
    Hardware-accelerated QGraphicsView with:
    - Viewport Y-inversion (scale(1, -1)) for true CAD world coordinates.
    - Mouse wheel zooming anchored at cursor.
    - Middle/left mouse drag panning.
    - Keyboard navigation (+ / - / 0, arrow keys).
    - Interactive measure tool (two-point Euclidean distance).
    """

    cursor_moved_signal = Signal(float, float)            # (drawing_x, drawing_y)
    zoom_changed_signal = Signal(float)                    # zoom_percentage
    design_selected_signal = Signal(int, str, tuple)       # (index, code, bbox)
    measure_updated_signal = Signal(float, float, float)   # (dist, dx, dy)

    def __init__(self, scene: DxfScene, parent: Optional[QWidget] = None):
        super().__init__(scene, parent)
        self.dxf_scene = scene

        # Render hints & performance optimizations
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMouseTracking(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # Flip the Y axis so DXF +Y points UP on screen
        self.scale(1.0, -1.0)

        # Interaction state
        self._is_panning: bool = False
        self._pan_start_pos: QPoint = QPoint()
        self._measure_mode: bool = False
        self._measure_p1: Optional[QPointF] = None
        self._measure_p2: Optional[QPointF] = None
        self._temp_measure_line: Optional[QGraphicsItem] = None

    @property
    def measure_mode(self) -> bool:
        return self._measure_mode

    @measure_mode.setter
    def measure_mode(self, enabled: bool) -> None:
        self._measure_mode = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.clear_measurement()

    def set_measure_mode(self, enabled: bool) -> None:
        self.measure_mode = enabled

    @property
    def is_measuring(self) -> bool:
        return self._measure_mode

    @property
    def measure_start_scene(self) -> Optional[QPointF]:
        return self._measure_p1

    @measure_start_scene.setter
    def measure_start_scene(self, pt: Optional[QPointF]) -> None:
        self._measure_p1 = pt

    @property
    def measure_end_scene(self) -> Optional[QPointF]:
        return self._measure_p2

    @measure_end_scene.setter
    def measure_end_scene(self, pt: Optional[QPointF]) -> None:
        self._measure_p2 = pt

    def zoom_to_fit(self) -> None:
        """Fits the entire drawing scene into view."""
        rect = self.scene().itemsBoundingRect()
        if not rect.isEmpty() and rect.width() > 0 and rect.height() > 0:
            margin = max(rect.width(), rect.height()) * 0.05
            target = rect.adjusted(-margin, -margin, margin, margin)
            self.fitInView(target, Qt.AspectRatioMode.KeepAspectRatio)
            self._maintain_flipped_y()
            self._emit_current_zoom()

    def fit_all(self) -> None:
        """Alias for zoom_to_fit."""
        self.zoom_to_fit()

    def zoom_to_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        margin_ratio: float = 0.12,
        padding_ratio: Optional[float] = None,
    ) -> None:
        """Smoothly frames a design bounding box with padding."""
        if padding_ratio is not None:
            margin_ratio = padding_ratio

        bx, by, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            self.zoom_to_fit()
            return

        pad_w = bw * margin_ratio
        pad_h = bh * margin_ratio
        target = QRectF(bx - pad_w, by - pad_h, bw + 2 * pad_w, bh + 2 * pad_h)
        self.fitInView(target, Qt.AspectRatioMode.KeepAspectRatio)
        self._maintain_flipped_y()
        self._emit_current_zoom()

    def zoom_in(self) -> None:
        self.scale(1.25, 1.25)
        self._emit_current_zoom()

    def zoom_out(self) -> None:
        self.scale(0.8, 0.8)
        self._emit_current_zoom()

    def reset_zoom(self) -> None:
        """Resets view zoom matrix to 1:1 scale (preserving Y-flip)."""
        self.resetTransform()
        self.scale(1.0, -1.0)
        self._emit_current_zoom()

    def export_image(self, file_path: str | Path, width: int = 2400, background_color: Optional[QColor] = None) -> bool:
        """Renders the entire drawing scene to high-resolution PNG image with upright CAD orientation."""
        rect = self.dxf_scene.itemsBoundingRect()
        if rect.isEmpty() or rect.width() <= 0 or rect.height() <= 0:
            return False

        # Add small margin padding (1.5%) so perimeter linework is not clipped against image edges
        pad_x = max(1.0, rect.width() * 0.015)
        pad_y = max(1.0, rect.height() * 0.015)
        padded_rect = rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)

        aspect = padded_rect.height() / padded_rect.width()
        height = max(100, int(width * aspect))
        img = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        bg = background_color or self.dxf_scene.backgroundBrush().color()
        if not bg.isValid() or bg.alpha() == 0:
            bg = DEFAULT_DARK_BG if self.dxf_scene.is_dark_mode else DEFAULT_LIGHT_BG
        img.fill(bg)

        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Invert Y to match the CAD viewport's upright orientation (cancels out CAD world Y-inversion)
        painter.translate(0, height)
        painter.scale(1.0, -1.0)

        # Render the graphics scene into the high-res image
        self.dxf_scene.render(painter, QRectF(0, 0, width, height), padded_rect)
        painter.end()

        # Note: PySide6 type stubs incorrectly annotate QImage.save's format as bytes | None,
        # but the Shiboken runtime requires str (b'PNG' throws ValueError). cast(Any, ...) satisfies both.
        return img.save(str(file_path), cast(Any, "PNG"))

    def _maintain_flipped_y(self) -> None:
        """Ensures the view transform matrix maintains its negative Y axis orientation."""
        t = self.transform()
        if t.m22() > 0:
            self.scale(1.0, -1.0)

    def _emit_current_zoom(self) -> None:
        zoom_pct = abs(self.transform().m11()) * 100.0
        self.zoom_changed_signal.emit(zoom_pct)

    # Mouse Events
    def wheelEvent(self, event) -> None:
        """Zooms centered at cursor position."""
        factor = 1.15 if event.angleDelta().y() > 0 else 0.87
        self.scale(factor, factor)
        self._maintain_flipped_y()
        self._emit_current_zoom()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and not self._measure_mode):
            # Check if clicked on a design panel rect
            scene_pos = self.mapToScene(event.pos())
            clicked_panel = self._find_panel_at(scene_pos)

            if clicked_panel is not None and event.button() == Qt.MouseButton.LeftButton:
                idx = clicked_panel.data(0)
                code = clicked_panel.data(1)
                r = clicked_panel.rect()
                bbox = (r.x(), r.y(), r.width(), r.height())
                self.design_selected_signal.emit(idx, code, bbox)

            self._is_panning = True
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if self._measure_mode and event.button() == Qt.MouseButton.LeftButton:
            scene_pt = self.mapToScene(event.pos())
            if self._measure_p1 is None:
                # Set Point 1
                self._measure_p1 = scene_pt
                self.clear_measurement()
            else:
                # Set Point 2 and lock measurement
                self._measure_p2 = scene_pt
                self._draw_measurement(locked=True)
                dx = abs(self._measure_p2.x() - self._measure_p1.x())
                dy = abs(self._measure_p2.y() - self._measure_p1.y())
                dist = math.hypot(dx, dy)
                self.measure_updated_signal.emit(dist, dx, dy)
                # Reset for next measurement
                self._measure_p1 = None
                self._measure_p2 = None
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        scene_pt = self.mapToScene(event.pos())
        self.cursor_moved_signal.emit(scene_pt.x(), scene_pt.y())

        if self._is_panning:
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if self._measure_mode and self._measure_p1 is not None:
            # Dynamic rubber-band dimension line
            self._measure_p2 = scene_pt
            self._draw_measurement(locked=False)
            dx = abs(self._measure_p2.x() - self._measure_p1.x())
            dy = abs(self._measure_p2.y() - self._measure_p1.y())
            dist = math.hypot(dx, dy)
            self.measure_updated_signal.emit(dist, dx, dy)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.CrossCursor if self._measure_mode else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        """Double click fits whole drawing to view."""
        if event.button() == Qt.MouseButton.LeftButton and not self._measure_mode:
            scene_pt = self.mapToScene(event.pos())
            panel = self._find_panel_at(scene_pt)
            if panel is not None:
                r = panel.rect()
                self.zoom_to_bbox((r.x(), r.y(), r.width(), r.height()))
            else:
                self.zoom_to_fit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_in()
            event.accept()
        elif key == Qt.Key.Key_Minus:
            self.zoom_out()
            event.accept()
        elif key == Qt.Key.Key_0:
            self.zoom_to_fit()
            event.accept()
        elif key == Qt.Key.Key_Left:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - 40)
            event.accept()
        elif key == Qt.Key.Key_Right:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + 40)
            event.accept()
        elif key == Qt.Key.Key_Up:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - 40)
            event.accept()
        elif key == Qt.Key.Key_Down:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + 40)
            event.accept()
        elif key == Qt.Key.Key_Escape:
            if self._measure_mode:
                self.measure_mode = False
            self.clear_measurement()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _find_panel_at(self, pt: QPointF) -> Optional[QGraphicsRectItem]:
        for item in self.dxf_scene.design_rect_items:
            if item.rect().contains(pt):
                return item
        return None

    def _draw_measurement(self, locked: bool = False) -> None:
        if self._measure_p1 is None or self._measure_p2 is None:
            return

        p1 = self._measure_p1
        p2 = self._measure_p2
        dx = abs(p2.x() - p1.x())
        dy = abs(p2.y() - p1.y())
        dist = math.hypot(dx, dy)

        # Clear previous rubberband items
        self.clear_measurement()

        # Dimension line path with perpendicular tick end-caps
        path = QPainterPath()
        path.moveTo(p1)
        path.lineTo(p2)

        # End ticks (length = 6 px in drawing units)
        angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
        perp = angle + math.pi / 2.0
        tick_len = max(2.0, dist * 0.02)
        tx = tick_len * math.cos(perp)
        ty = tick_len * math.sin(perp)

        path.moveTo(p1.x() - tx, p1.y() - ty)
        path.lineTo(p1.x() + tx, p1.y() + ty)
        path.moveTo(p2.x() - tx, p2.y() - ty)
        path.lineTo(p2.x() + tx, p2.y() + ty)

        line_item = QGraphicsPathItem(path)
        pen_style = Qt.PenStyle.SolidLine if locked else Qt.PenStyle.DashLine
        pen = QPen(MEASURE_CYAN, 2.0, pen_style)
        pen.setCosmetic(True)
        line_item.setPen(pen)
        line_item.setZValue(30.0)
        self.scene().addItem(line_item)
        self._temp_measure_line = line_item

        # Dimension readout text
        unit_s = self.dxf_scene.parsed_data.units_suffix if self.dxf_scene.parsed_data else "mm"
        mid_x = (p1.x() + p2.x()) / 2.0
        mid_y = (p1.y() + p2.y()) / 2.0

        txt_item = QGraphicsSimpleTextItem(f"{dist:.2f} {unit_s}")
        txt_font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        txt_item.setFont(txt_font)
        txt_item.setBrush(QBrush(MEASURE_CYAN))

        t = QTransform()
        t.translate(mid_x, mid_y + 4.0)
        t.scale(1.0, -1.0)  # Upright in Y-flipped view
        txt_item.setTransform(t)
        txt_item.setZValue(35.0)

        self.scene().addItem(txt_item)
        self.dxf_scene.measure_text_item = txt_item

    def clear_measurement(self) -> None:
        if self._temp_measure_line is not None:
            self.scene().removeItem(self._temp_measure_line)
            self._temp_measure_line = None
        if self.dxf_scene.measure_text_item is not None:
            self.scene().removeItem(self.dxf_scene.measure_text_item)
            self.dxf_scene.measure_text_item = None


# ---------------------------------------------------------------------------
# Composite DXF Viewer Widget (DxfViewerWidget)
# ---------------------------------------------------------------------------

class DxfViewerWidget(QWidget):
    """
    All-in-one DXF viewer widget with:
    - Vector canvas view
    - Layer visibility control panel
    - Extracted designs browser
    - Precision zoom, pan, and measure tools
    - Dark / Light contrast mode toggle
    - 'Open in Application' launcher with settings persistence
    - Side-by-side customer query mask comparison panel
    """

    open_external_signal = Signal(str)
    expand_window_signal = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_dxf_path: Optional[Path] = None
        self.matched_bbox: Optional[Tuple[float, float, float, float]] = None
        self.matched_code: Optional[str] = None
        self.query_mask_bgr: Optional[Any] = None
        self.is_dark_mode: bool = True
        self.external_cad_app: str = ""

        self.scene = DxfScene(self)
        self.view = DxfView(self.scene, self)
        self.loader_worker: Optional[DxfLoaderWorker] = None

        self._load_app_settings()
        self._setup_ui()
        self._wire_signals()

    def _load_app_settings(self) -> None:
        settings = QSettings("DesignMatcher", "DxfViewer")
        self.is_dark_mode = bool(settings.value("dark_mode", True, type=bool))
        self.external_cad_app = str(settings.value("external_cad_app", "", type=str))

    def _save_app_settings(self) -> None:
        settings = QSettings("DesignMatcher", "DxfViewer")
        settings.setValue("dark_mode", self.is_dark_mode)
        settings.setValue("external_cad_app", self.external_cad_app)

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Top CAD Navigation Toolbar
        self.toolbar = QToolBar(self)
        self.toolbar.setStyleSheet(
            "QToolBar { background: #1E293B; border-bottom: 1px solid #334155; padding: 4px; spacing: 4px; } "
            "QToolButton { background: #0F172A; color: #F1F5F9; border: 1px solid #334155; border-radius: 4px; "
            "padding: 4px 8px; font-size: 11px; font-weight: 600; } "
            "QToolButton:hover { background: #334155; border-color: #475569; } "
            "QToolButton:checked { background: #2563EB; border-color: #3B82F6; color: white; }"
        )

        self.btn_fit = QToolButton(self.toolbar)
        self.btn_fit.setText("⛶ Fit All (0)")
        self.btn_fit.setToolTip("Fit entire drawing into view [Key: 0]")
        self.toolbar.addWidget(self.btn_fit)

        self.btn_zoom_match = QToolButton(self.toolbar)
        self.btn_zoom_match.setText("🎯 Zoom to Match")
        self.btn_zoom_match.setToolTip("Zoom to matched design panel")
        self.toolbar.addWidget(self.btn_zoom_match)

        self.btn_zoom_in = QToolButton(self.toolbar)
        self.btn_zoom_in.setText("➕ Zoom In (+)")
        self.toolbar.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QToolButton(self.toolbar)
        self.btn_zoom_out.setText("➖ Zoom Out (-)")
        self.toolbar.addWidget(self.btn_zoom_out)

        self.toolbar.addSeparator()

        self.btn_measure = QToolButton(self.toolbar)
        self.btn_measure.setCheckable(True)
        self.btn_measure.setText("📏 Measure")
        self.btn_measure.setToolTip("Click two points to measure distance in drawing units [Esc to exit]")
        self.toolbar.addWidget(self.btn_measure)

        self.btn_theme = QToolButton(self.toolbar)
        self.btn_theme.setText("🌙 Dark" if self.is_dark_mode else "☀️ Light")
        self.btn_theme.setToolTip("Toggle dark / light CAD background")
        self.toolbar.addWidget(self.btn_theme)

        self.btn_compare = QToolButton(self.toolbar)
        self.btn_compare.setCheckable(True)
        self.btn_compare.setText("🖼 Query Mask")
        self.btn_compare.setToolTip("Toggle side-by-side customer photo mask comparison")
        self.toolbar.addWidget(self.btn_compare)

        self.toolbar.addSeparator()

        # "Open in Application" Split Dropdown Button
        self.btn_open_app = QToolButton(self.toolbar)
        self.btn_open_app.setText("🚀 Open in Application")
        self.btn_open_app.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_open_app.setStyleSheet(
            "QToolButton { background: #059669; color: white; border: none; font-weight: 700; padding: 4px 10px; } "
            "QToolButton:hover { background: #047857; }"
        )

        app_menu = QMenu(self.btn_open_app)
        self.act_open_default = app_menu.addAction("Open with Default Associated CAD App")
        self.act_choose_app = app_menu.addAction("Choose Program (AutoCAD, Laser Software)...")
        app_menu.addSeparator()
        self.act_show_folder = app_menu.addAction("Show in Folder (Explorer)")
        self.act_copy_path = app_menu.addAction("Copy DXF File Path")
        self.btn_open_app.setMenu(app_menu)
        self.toolbar.addWidget(self.btn_open_app)

        self.toolbar.addSeparator()

        self.btn_expand = QToolButton(self.toolbar)
        self.btn_expand.setText("⤢ Open Viewer")
        self.btn_expand.setToolTip("Open in full standalone viewer window")
        self.toolbar.addWidget(self.btn_expand)

        main_layout.addWidget(self.toolbar)

        # 2. Central Splitter Area
        self.central_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.central_splitter.setHandleWidth(2)

        # Left side panel: Designs List (top) + Layers (bottom)
        self.side_panel = QWidget(self.central_splitter)
        self.side_panel.setMinimumWidth(180)
        self.side_panel.setMaximumWidth(280)
        side_layout = QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(6, 6, 6, 6)
        side_layout.setSpacing(6)

        # Designs List Header
        des_lbl = QLabel("📐 DETECTED DESIGNS", self.side_panel)
        des_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #64748B;")
        side_layout.addWidget(des_lbl)

        self.designs_list = QListWidget(self.side_panel)
        self.designs_list.setStyleSheet(
            "QListWidget { background: #0F172A; color: #F8FAFC; border: 1px solid #334155; border-radius: 4px; font-size: 11px; } "
            "QListWidget::item { padding: 4px 6px; } "
            "QListWidget::item:selected { background: #2563EB; color: white; font-weight: 600; }"
        )
        side_layout.addWidget(self.designs_list, 1)

        # Layers Header
        layer_header = QHBoxLayout()
        layer_lbl = QLabel("📑 CAD LAYERS", self.side_panel)
        layer_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #64748B;")
        layer_header.addWidget(layer_lbl)
        layer_header.addStretch()

        btn_all_layers = QPushButton("All", self.side_panel)
        btn_all_layers.setStyleSheet("font-size: 9px; padding: 1px 4px;")
        btn_all_layers.clicked.connect(self._toggle_all_layers_on)
        layer_header.addWidget(btn_all_layers)
        side_layout.addLayout(layer_header)

        # Layers Scroll Area
        self.layers_scroll = QScrollArea(self.side_panel)
        self.layers_scroll.setWidgetResizable(True)
        self.layers_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.layers_container = QWidget()
        self.layers_layout = QVBoxLayout(self.layers_container)
        self.layers_layout.setContentsMargins(2, 2, 2, 2)
        self.layers_layout.setSpacing(3)
        self.layers_layout.addStretch()
        self.layers_scroll.setWidget(self.layers_container)
        side_layout.addWidget(self.layers_scroll, 1)

        self.central_splitter.addWidget(self.side_panel)

        # Center: Graphics View with Loading Overlay
        self.view_container = QWidget(self.central_splitter)
        vbox_view = QVBoxLayout(self.view_container)
        vbox_view.setContentsMargins(0, 0, 0, 0)
        vbox_view.setSpacing(0)
        vbox_view.addWidget(self.view)

        # Loading overlay frame
        self.loading_overlay = QFrame(self.view_container)
        self.loading_overlay.setStyleSheet("background: rgba(15, 23, 42, 0.85); border-radius: 8px;")
        ol_layout = QVBoxLayout(self.loading_overlay)
        ol_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label = QLabel("Loading DXF Vector Geometry...", self.loading_overlay)
        self.loading_label.setStyleSheet("color: #FFFFFF; font-size: 14px; font-weight: 600;")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ol_layout.addWidget(self.loading_label)

        self.progress_bar = QProgressBar(self.loading_overlay)
        self.progress_bar.setRange(0, 0)  # Indeterminate spinner
        self.progress_bar.setFixedWidth(200)
        ol_layout.addWidget(self.progress_bar)
        self.loading_overlay.setVisible(False)

        self.central_splitter.addWidget(self.view_container)

        # Right side panel: Query Photo Mask Comparison (Collapsible)
        self.compare_panel = QFrame(self.central_splitter)
        self.compare_panel.setStyleSheet("background: #0F172A; border-left: 1px solid #334155;")
        self.compare_panel.setMinimumWidth(220)
        self.compare_panel.setMaximumWidth(320)
        cmp_layout = QVBoxLayout(self.compare_panel)
        cmp_layout.setContentsMargins(10, 10, 10, 10)

        cmp_title = QLabel("CUSTOMER QUERY MASK", self.compare_panel)
        cmp_title.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 700;")
        cmp_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cmp_layout.addWidget(cmp_title)

        self.compare_img_label = QLabel(self.compare_panel)
        self.compare_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.compare_img_label.setStyleSheet("border: 1px solid #334155; border-radius: 6px;")
        cmp_layout.addWidget(self.compare_img_label, 1)

        self.compare_info_label = QLabel("1:1 Visual Alignment", self.compare_panel)
        self.compare_info_label.setStyleSheet("color: #64748B; font-size: 10px;")
        self.compare_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cmp_layout.addWidget(self.compare_info_label)

        self.compare_panel.setVisible(False)
        self.central_splitter.addWidget(self.compare_panel)

        # Set default splitter proportions (Side Panel: 1, View: 4)
        self.central_splitter.setStretchFactor(0, 1)
        self.central_splitter.setStretchFactor(1, 4)

        main_layout.addWidget(self.central_splitter, 1)

        # 3. Bottom Status Bar
        self.status_bar = QStatusBar(self)
        self.status_bar.setStyleSheet("QStatusBar { background: #1E293B; color: #94A3B8; font-size: 11px; padding: 2px 8px; }")

        self.lbl_coord = QLabel("X: 0.00  Y: 0.00 mm", self)
        self.lbl_coord.setStyleSheet("color: #F8FAFC; font-weight: 600;")
        self.status_bar.addWidget(self.lbl_coord)

        self.lbl_zoom = QLabel("Zoom: 100%", self)
        self.status_bar.addWidget(self.lbl_zoom)

        self.lbl_units = QLabel("Units: Millimeters (mm)", self)
        self.status_bar.addWidget(self.lbl_units)

        self.lbl_entities = QLabel("0 entities", self)
        self.status_bar.addWidget(self.lbl_entities)

        self.lbl_file = QLabel("No DXF loaded", self)
        self.lbl_file.setStyleSheet("color: #38BDF8; font-weight: 600;")
        self.status_bar.addPermanentWidget(self.lbl_file)

        main_layout.addWidget(self.status_bar)

    def _wire_signals(self) -> None:
        self.btn_fit.clicked.connect(self.view.zoom_to_fit)
        self.btn_zoom_match.clicked.connect(self._zoom_to_matched_design)
        self.btn_zoom_in.clicked.connect(self.view.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view.zoom_out)
        self.btn_measure.toggled.connect(self._toggle_measure_mode)
        self.btn_theme.clicked.connect(self._toggle_theme)
        self.btn_compare.toggled.connect(self._toggle_compare_mask)
        self.btn_expand.clicked.connect(self.expand_window_signal.emit)

        # Application menu actions
        self.btn_open_app.clicked.connect(self._open_in_application)
        self.act_open_default.triggered.connect(self._open_in_default_app)
        self.act_choose_app.triggered.connect(self._choose_custom_app)
        self.act_show_folder.triggered.connect(self._show_in_folder)
        self.act_copy_path.triggered.connect(self._copy_path)

        # View signals
        self.view.cursor_moved_signal.connect(self._on_cursor_moved)
        self.view.zoom_changed_signal.connect(self._on_zoom_changed)
        self.view.design_selected_signal.connect(self._on_design_clicked_in_view)
        self.view.measure_updated_signal.connect(self._on_measure_updated)

        # Designs list click
        self.designs_list.currentRowChanged.connect(self._on_design_list_selected)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.loading_overlay.isVisible():
            self.loading_overlay.setGeometry(self.view_container.rect())

    # -----------------------------------------------------------------------
    # Loading & Display
    # -----------------------------------------------------------------------

    @property
    def status_label(self) -> QLabel:
        """Alias returning the primary status / filename display label."""
        return self.lbl_file

    def load_file(
        self,
        dxf_path: Path | str,
        matched_bbox: Optional[Tuple[float, float, float, float]] = None,
        matched_code: Optional[str] = None,
        query_mask: Optional[Any] = None,
        design_code: Optional[str] = None,
    ) -> None:
        """Loads and visualizes a DXF CAD drawing asynchronously."""
        code = design_code if design_code is not None else matched_code
        path = Path(dxf_path).resolve()
        self.current_dxf_path = path
        self.matched_bbox = matched_bbox
        self.matched_code = code
        self.query_mask_bgr = query_mask

        self.lbl_file.setText(path.name)
        self._update_query_mask_display()

        # Check LRU cache synchronously first for instant switching
        cached = GLOBAL_DXF_CACHE.get(path)
        if cached is not None:
            self._on_dxf_loaded(cached)
            return

        # Show loading overlay
        self.loading_label.setText(f"Loading '{path.name}'...")
        self.loading_overlay.setGeometry(self.view_container.rect())
        self.loading_overlay.setVisible(True)

        if self.loader_worker is not None and self.loader_worker.isRunning():
            self.loader_worker.quit()
            self.loader_worker.wait()

        self.loader_worker = DxfLoaderWorker(path, self)
        self.loader_worker.loaded_signal.connect(self._on_dxf_loaded)
        self.loader_worker.error_signal.connect(self._on_dxf_error)
        self.loader_worker.start()

    def _on_dxf_loaded(self, data: ParsedDxfData) -> None:
        self.loading_overlay.setVisible(False)
        self.scene.build_scene(data, is_dark_mode=self.is_dark_mode)

        # Update status bar info
        self.lbl_units.setText(f"Units: {data.units_name} ({data.units_suffix})")
        self.lbl_entities.setText(f"{data.total_entity_count:,} entities" + (" [High Entity Mode]" if data.is_simplified else ""))

        # Populate designs list
        self.designs_list.blockSignals(True)
        self.designs_list.clear()
        matched_row = -1
        for idx, d in enumerate(data.designs):
            code_str = d.design_code or f"Design #{idx + 1}"
            is_target = False
            if self.matched_bbox is not None:
                bx, by, bw, bh = self.matched_bbox
                if abs(d.x - bx) < 1.0 and abs(d.y - by) < 1.0:
                    is_target = True

            item_text = f"★ {code_str} (Match)" if is_target else f"#{idx + 1}: {code_str}"
            item = QListWidgetItem(item_text)
            if is_target:
                item.setForeground(QBrush(HIGHLIGHT_ORANGE))
                matched_row = idx
            self.designs_list.addItem(item)
        self.designs_list.blockSignals(False)

        # Populate layers panel
        self._populate_layers(data.layers)

        # Highlight and zoom to matched design or fit whole drawing
        if self.matched_bbox is not None:
            self.scene.highlight_design(self.matched_bbox, self.matched_code or "")
            self.view.zoom_to_bbox(self.matched_bbox)
            if matched_row >= 0:
                self.designs_list.setCurrentRow(matched_row)
        else:
            self.view.zoom_to_fit()

    def _on_dxf_error(self, friendly_msg: str, tech_details: str) -> None:
        self.loading_overlay.setVisible(False)
        self.lbl_file.setText(f"Error: {friendly_msg.splitlines()[0]}")
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen" and "pytest" not in sys.modules:
            dlg = QMessageBox(self)
            dlg.setIcon(QMessageBox.Icon.Warning)
            dlg.setWindowTitle("DXF Ingestion Notice")
            dlg.setText(friendly_msg)
            if tech_details:
                dlg.setDetailedText(tech_details)
            dlg.exec()
        else:
            logger.warning("DXF Notice: %s | %s", friendly_msg, tech_details)

    def _populate_layers(self, layers: Dict[str, LayerInfo]) -> None:
        # Clear existing checkboxes
        while self.layers_layout.count() > 1:
            item = self.layers_layout.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.deleteLater()
                child_layout = item.layout()
                if child_layout is not None:
                    while child_layout.count() > 0:
                        child_item = child_layout.takeAt(0)
                        if child_item is not None:
                            cw = child_item.widget()
                            if cw is not None:
                                cw.deleteLater()

        for lname, linfo in sorted(layers.items(), key=lambda x: x[0].lower()):
            row = QHBoxLayout()
            chk = QCheckBox(f"{lname} ({linfo.entity_count})", self.layers_container)
            chk.setChecked(linfo.is_on and not linfo.is_frozen)
            chk.setStyleSheet(f"color: {linfo.color.name()}; font-size: 11px; font-weight: 500;")
            chk.toggled.connect(lambda checked, name=lname: self.scene.set_layer_visible(name, checked))
            row.addWidget(chk)
            row.addStretch()
            self.layers_layout.insertLayout(self.layers_layout.count() - 1, row)

    def _toggle_all_layers_on(self) -> None:
        for i in range(self.layers_layout.count() - 1):
            item = self.layers_layout.itemAt(i)
            if item is not None:
                layout = item.layout()
                if layout is not None:
                    first_item = layout.itemAt(0)
                    if first_item is not None:
                        chk = first_item.widget()
                        if isinstance(chk, QCheckBox):
                            chk.setChecked(True)

    # -----------------------------------------------------------------------
    # Interactive Actions & Navigation
    # -----------------------------------------------------------------------

    def _zoom_to_matched_design(self) -> None:
        if self.matched_bbox is not None:
            self.scene.highlight_design(self.matched_bbox, self.matched_code or "")
            self.view.zoom_to_bbox(self.matched_bbox)
        else:
            self.view.zoom_to_fit()

    def _toggle_measure_mode(self, enabled: bool) -> None:
        self.view.measure_mode = enabled
        if enabled:
            unit_s = self.scene.parsed_data.units_suffix if self.scene.parsed_data else "mm"
            self.status_bar.showMessage(f"Measure Tool Active: Click two points to measure distance ({unit_s}). Press Esc to cancel.", 6000)
        else:
            self.status_bar.clearMessage()

    def _toggle_theme(self) -> None:
        self.is_dark_mode = not self.is_dark_mode
        self.btn_theme.setText("🌙 Dark" if self.is_dark_mode else "☀️ Light")
        self.scene.set_dark_mode(self.is_dark_mode)
        self._save_app_settings()

    def _toggle_compare_mask(self, checked: bool) -> None:
        self.compare_panel.setVisible(checked)
        if checked:
            self._update_query_mask_display()

    def _update_query_mask_display(self) -> None:
        if self.query_mask_bgr is not None and getattr(self.query_mask_bgr, "size", 0) > 0:
            import cv2
            h, w = self.query_mask_bgr.shape[:2]
            rgb = cv2.cvtColor(self.query_mask_bgr, cv2.COLOR_BGR2RGB) if len(self.query_mask_bgr.shape) == 3 else cv2.cvtColor(self.query_mask_bgr, cv2.COLOR_GRAY2RGB)
            from PySide6.QtGui import QImage
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.compare_img_label.setPixmap(pix.scaled(240, 240, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.compare_img_label.setText("No Query Mask Available\n(Text Shortcut Search)")
            self.compare_img_label.setStyleSheet("color: #64748B; font-size: 11px;")

    def _on_cursor_moved(self, x: float, y: float) -> None:
        unit_s = self.scene.parsed_data.units_suffix if self.scene.parsed_data else "mm"
        self.lbl_coord.setText(f"X: {x:.2f}  Y: {y:.2f} {unit_s}")

    def _on_zoom_changed(self, zoom_pct: float) -> None:
        self.lbl_zoom.setText(f"Zoom: {zoom_pct:.0f}%")

    def _on_measure_updated(self, dist: float, dx: float, dy: float) -> None:
        unit_s = self.scene.parsed_data.units_suffix if self.scene.parsed_data else "mm"
        self.status_bar.showMessage(f"📏 Distance: {dist:.2f} {unit_s}  (ΔX: {dx:.2f}, ΔY: {dy:.2f})")

    def _on_design_clicked_in_view(self, index: int, code: str, bbox: tuple) -> None:
        self.matched_bbox = bbox
        self.matched_code = code
        self.scene.highlight_design(bbox, code)
        self.designs_list.blockSignals(True)
        self.designs_list.setCurrentRow(index - 1)
        self.designs_list.blockSignals(False)
        self.status_bar.showMessage(f"Selected Design #{index}: '{code}' | BBox: {tuple(round(v, 1) for v in bbox)}")

    def _on_design_list_selected(self, row: int) -> None:
        if self.scene.parsed_data and 0 <= row < len(self.scene.parsed_data.designs):
            d = self.scene.parsed_data.designs[row]
            bbox = (d.x, d.y, d.width, d.height)
            self.matched_bbox = bbox
            self.matched_code = d.design_code
            self.scene.highlight_design(bbox, d.design_code or f"Design #{row + 1}")
            self.view.zoom_to_bbox(bbox)

    # -----------------------------------------------------------------------
    # External Application Launcher
    # -----------------------------------------------------------------------

    def _open_in_application(self) -> None:
        if self.external_cad_app and Path(self.external_cad_app).is_file():
            self._launch_custom_app(self.external_cad_app)
        else:
            self._open_in_default_app()

    def _open_in_default_app(self) -> None:
        if self.current_dxf_path is None or not self.current_dxf_path.is_file():
            QMessageBox.warning(self, "File Not Found", f"The DXF file could not be found:\n{self.current_dxf_path}")
            return

        target_str = str(self.current_dxf_path)
        try:
            if sys.platform.startswith("win"):
                os.startfile(target_str)
            else:
                from PySide6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(target_str))
        except Exception as exc:
            logger.warning("Default open failed for %s (%s). Falling back to QDesktopServices.", target_str, exc)
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(target_str))

    def _choose_custom_app(self) -> None:
        exe_filter = "Executables (*.exe *.app);;All Files (*)" if sys.platform.startswith("win") else "Applications (*);;All Files (*)"
        chosen, _ = QFileDialog.getOpenFileName(self, "Select CAD / CAM Application", "", exe_filter)
        if chosen and Path(chosen).is_file():
            self.external_cad_app = chosen
            self._save_app_settings()
            self._launch_custom_app(chosen)

    def _launch_custom_app(self, exe_path: str) -> None:
        if self.current_dxf_path is None or not self.current_dxf_path.is_file():
            QMessageBox.warning(self, "File Not Found", f"The DXF file could not be found:\n{self.current_dxf_path}")
            return
        try:
            subprocess.Popen([exe_path, str(self.current_dxf_path)])
        except Exception as exc:
            QMessageBox.warning(self, "Launch Error", f"Failed to start '{exe_path}':\n{exc}")

    def _show_in_folder(self) -> None:
        if self.current_dxf_path is None or not self.current_dxf_path.is_file():
            QMessageBox.warning(self, "File Not Found", f"The DXF file could not be found:\n{self.current_dxf_path}")
            return
        target_str = str(self.current_dxf_path)
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(f'explorer /select,"{target_str}"')
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", target_str])
            else:
                subprocess.Popen(["xdg-open", str(self.current_dxf_path.parent)])
        except Exception as exc:
            logger.warning("Could not open containing folder: %s", exc)

    def _copy_path(self) -> None:
        if self.current_dxf_path is not None:
            QApplication.clipboard().setText(str(self.current_dxf_path))
            self.status_bar.showMessage("DXF file path copied to clipboard!", 3000)


# ---------------------------------------------------------------------------
# Standalone Enlarged Viewer Window (DxfViewerWindow)
# ---------------------------------------------------------------------------

class DxfViewerWindow(QMainWindow):
    """Full-featured standalone CAD viewer window with window geometry persistence."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("DXF CAD Vector Viewer - Design Matcher")
        self.resize(1100, 750)
        self.setMinimumSize(800, 550)

        self.viewer_widget = DxfViewerWidget(self)
        self.setCentralWidget(self.viewer_widget)

        # Hide expand button in the standalone window
        self.viewer_widget.btn_expand.setVisible(False)

        self._restore_window_state()

    def load_file(
        self,
        dxf_path: Path | str,
        matched_bbox: Optional[Tuple[float, float, float, float]] = None,
        matched_code: Optional[str] = None,
        query_mask: Optional[Any] = None,
        design_code: Optional[str] = None,
    ) -> None:
        code = design_code if design_code is not None else matched_code
        p = Path(dxf_path)
        code_tag = f" - [{code}]" if code else ""
        self.setWindowTitle(f"DXF CAD Viewer - {p.name}{code_tag}")
        self.viewer_widget.load_file(dxf_path, matched_bbox=matched_bbox, matched_code=code, query_mask=query_mask)

    def closeEvent(self, event) -> None:
        self._save_window_state()
        # Clean up scene to free memory
        self.viewer_widget.scene.clear()
        super().closeEvent(event)

    def _restore_window_state(self) -> None:
        settings = QSettings("DesignMatcher", "DxfViewerWindow")
        geom = settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)

    def _save_window_state(self) -> None:
        settings = QSettings("DesignMatcher", "DxfViewerWindow")
        settings.setValue("geometry", self.saveGeometry())


# ---------------------------------------------------------------------------
# Left Side Panel: Folders & DXF Files Browser (DxfFolderExplorerWidget)
# ---------------------------------------------------------------------------

class DxfFolderExplorerWidget(QWidget):
    """
    Folder and file list browser on the left side of the DXF Preview workspace.
    Allows browsing directories, quick-jumping to Project / Catalog root,
    real-time filename search/filtering, and selecting DXF files for preview.
    """

    file_selected = Signal(Path)

    def __init__(self, initial_folder: Optional[Path | str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        if initial_folder is not None and Path(initial_folder).is_dir():
            self.current_folder = Path(initial_folder).resolve()
        else:
            settings = QSettings("DesignMatcher", "DxfViewer")
            saved_folder = settings.value("last_dxf_folder", "", type=str)
            if saved_folder and Path(saved_folder).is_dir():
                self.current_folder = Path(saved_folder).resolve()
            else:
                self.current_folder = Path.cwd().resolve()

        self._all_dxf_files: List[Path] = []
        self._setup_ui()
        self.refresh()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 1. Header Title & Refresh
        title_box = QHBoxLayout()
        self.title_lbl = QLabel("📁 DXF FOLDERS & FILES", self)
        self.title_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.5px;")
        title_box.addWidget(self.title_lbl)
        title_box.addStretch()

        self.btn_refresh = QToolButton(self)
        self.btn_refresh.setText("🔄")
        self.btn_refresh.setToolTip("Refresh Folder Contents")
        self.btn_refresh.setStyleSheet(
            "QToolButton { background: #1E293B; color: #CBD5E1; border: 1px solid #334155; border-radius: 4px; padding: 2px 5px; } "
            "QToolButton:hover { background: #334155; }"
        )
        self.btn_refresh.clicked.connect(self.refresh)
        title_box.addWidget(self.btn_refresh)
        layout.addLayout(title_box)

        # 2. Browse Folder Button
        self.btn_browse = QPushButton("📁 Browse Folder...", self)
        self.btn_browse.setStyleSheet(
            "QPushButton { background: #2563EB; color: white; border: none; border-radius: 4px; padding: 6px 10px; font-weight: 600; font-size: 11px; } "
            "QPushButton:hover { background: #1D4ED8; }"
        )
        self.btn_browse.clicked.connect(self._browse_folder_dialog)
        layout.addWidget(self.btn_browse)

        # 3. Current Path Label
        self.path_label = QLabel(self)
        self.path_label.setStyleSheet("color: #64748B; font-size: 10px; padding: 2px 0px;")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)

        # 4. Search / Filter Input
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("🔍 Filter DXF files...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setStyleSheet(
            "QLineEdit { background: #0F172A; color: #F8FAFC; border: 1px solid #334155; border-radius: 4px; padding: 4px 8px; font-size: 11px; } "
            "QLineEdit:focus { border-color: #3B82F6; }"
        )
        self.search_input.textChanged.connect(self._filter_files)
        layout.addWidget(self.search_input)

        # 5. Options Row: Recursive Subfolders & Count Label
        opts_row = QHBoxLayout()
        self.chk_recursive = QCheckBox("Include Subfolders", self)
        self.chk_recursive.setChecked(True)
        self.chk_recursive.setStyleSheet("QCheckBox { color: #94A3B8; font-size: 10px; }")
        self.chk_recursive.toggled.connect(self.refresh)
        opts_row.addWidget(self.chk_recursive)

        opts_row.addStretch()
        self.count_label = QLabel("0 files", self)
        self.count_label.setStyleSheet("color: #64748B; font-size: 10px;")
        opts_row.addWidget(self.count_label)
        layout.addLayout(opts_row)

        # 6. DXF Files List
        self.file_list = QListWidget(self)
        self.file_list.setStyleSheet(
            "QListWidget { background: #0F172A; color: #F8FAFC; border: 1px solid #334155; border-radius: 6px; font-size: 11px; } "
            "QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #1E293B; border-radius: 4px; margin-bottom: 2px; } "
            "QListWidget::item:hover { background: #1E293B; } "
            "QListWidget::item:selected { background: #2563EB; color: #FFFFFF; font-weight: 600; }"
        )
        self.file_list.itemClicked.connect(self._on_item_clicked)
        self.file_list.currentItemChanged.connect(self._on_current_item_changed)
        layout.addWidget(self.file_list, 1)

    def _browse_folder_dialog(self) -> None:
        start_dir = str(self.current_folder) if self.current_folder.is_dir() else str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select Folder Containing DXF CAD Files", start_dir)
        if chosen and Path(chosen).is_dir():
            self.set_folder(Path(chosen))

    def set_folder(self, folder: Path | str) -> None:
        p = Path(folder).resolve()
        if p.is_dir():
            self.current_folder = p
            # Persist selected folder across app sessions
            QSettings("DesignMatcher", "DxfViewer").setValue("last_dxf_folder", str(p))
            self.refresh()

    def refresh(self) -> None:
        p = self.current_folder
        self.path_label.setText(f"📂 {p.name}")
        self.path_label.setToolTip(str(p))

        # Scan for DXF files
        dxf_files: List[Path] = []
        try:
            if self.chk_recursive.isChecked():
                for ext in ("*.dxf", "*.DXF"):
                    dxf_files.extend(list(p.rglob(ext)))
            else:
                for ext in ("*.dxf", "*.DXF"):
                    dxf_files.extend(list(p.glob(ext)))
        except Exception as exc:
            logger.warning("Error scanning DXF folder %s: %s", p, exc)

        # Deduplicate & sort by name
        seen = set()
        unique_files: List[Path] = []
        for f in dxf_files:
            rp = f.resolve()
            if rp not in seen and f.is_file():
                seen.add(rp)
                unique_files.append(f)
        unique_files.sort(key=lambda x: x.name.lower())
        self._all_dxf_files = unique_files
        self._filter_files(self.search_input.text())

    def _filter_files(self, query: str = "") -> None:
        self.file_list.blockSignals(True)
        self.file_list.clear()

        q = query.strip().lower()
        matched = [f for f in self._all_dxf_files if not q or q in f.name.lower()]

        for f in matched:
            try:
                sz = f.stat().st_size
                sz_str = f"{sz / 1024:.1f} KB" if sz < 1024 * 1024 else f"{sz / (1024 * 1024):.1f} MB"
                mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%b %d, %H:%M")
            except Exception:
                sz_str = "0 KB"
                mtime = ""

            item_text = f"📐 {f.name}\n   {sz_str}  •  {mtime}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, f)
            item.setToolTip(f"{f.name}\nPath: {f}\nSize: {sz_str}")
            self.file_list.addItem(item)

        self.count_label.setText(f"{len(matched)} of {len(self._all_dxf_files)} DXF file(s)")
        if not matched:
            empty_item = QListWidgetItem("No DXF files found in folder.\nClick 'Browse Folder' to select CAD directory.")
            empty_item.setFlags(Qt.ItemFlag.NoItemFlags)
            empty_item.setForeground(QBrush(QColor("#64748B")))
            self.file_list.addItem(empty_item)

        self.file_list.blockSignals(False)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        f = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(f, Path) and f.is_file():
            self.file_selected.emit(f)

    def _on_current_item_changed(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if current is not None:
            self._on_item_clicked(current)

    def select_file(self, target_path: Path | str) -> None:
        tp = Path(target_path).resolve()
        for idx in range(self.file_list.count()):
            item = self.file_list.item(idx)
            f = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(f, Path) and f.resolve() == tp:
                self.file_list.blockSignals(True)
                self.file_list.setCurrentRow(idx)
                self.file_list.blockSignals(False)
                break

    def apply_theme(self, is_dark: bool) -> None:
        bg_main = "#0F172A" if is_dark else "#FFFFFF"
        border_col = "#334155" if is_dark else "#CBD5E1"
        text_primary = "#F8FAFC" if is_dark else "#0F172A"
        text_muted = "#94A3B8" if is_dark else "#64748B"
        btn_bg = "#1E293B" if is_dark else "#F1F5F9"
        btn_text = "#E2E8F0" if is_dark else "#1E293B"
        btn_hover = "#334155" if is_dark else "#E2E8F0"
        item_hover = "#1E293B" if is_dark else "#F1F5F9"

        self.setStyleSheet(f"background-color: {bg_main};")
        self.title_lbl.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {text_muted}; letter-spacing: 0.5px;")


        self.btn_refresh.setStyleSheet(
            f"QToolButton {{ background: {btn_bg}; color: {btn_text}; border: 1px solid {border_col}; border-radius: 4px; padding: 2px 5px; }} "
            f"QToolButton:hover {{ background: {btn_hover}; }}"
        )
        self.path_label.setStyleSheet(f"color: {text_muted}; font-size: 10px; padding: 2px 0px;")
        self.chk_recursive.setStyleSheet(f"QCheckBox {{ color: {text_muted}; font-size: 10px; }}")
        self.count_label.setStyleSheet(f"color: {text_muted}; font-size: 10px;")

        self.search_input.setStyleSheet(
            f"QLineEdit {{ background: {bg_main}; color: {text_primary}; border: 1px solid {border_col}; border-radius: 4px; padding: 4px 8px; font-size: 11px; }} "
            f"QLineEdit:focus {{ border-color: #3B82F6; }}"
        )
        self.file_list.setStyleSheet(
            f"QListWidget {{ background: {bg_main}; color: {text_primary}; border: 1px solid {border_col}; border-radius: 6px; font-size: 11px; }} "
            f"QListWidget::item {{ padding: 6px 8px; border-bottom: 1px solid {border_col}; border-radius: 4px; margin-bottom: 2px; }} "
            f"QListWidget::item:hover {{ background: {item_hover}; }} "
            f"QListWidget::item:selected {{ background: #2563EB; color: #FFFFFF; font-weight: 600; }}"
        )


# ---------------------------------------------------------------------------
# Left Side Panel: Detected Designs Browser (DxfDetectedDesignsWidget)
# ---------------------------------------------------------------------------

class DxfDetectedDesignsWidget(QWidget):
    """
    Displays list of detected laser-cut / CNC designs extracted from the active DXF drawing.
    Allows searching by design code, clicking to frame and highlight on canvas.
    """

    design_selected = Signal(int, str, tuple)  # (index, code, bbox)
    fit_sheet_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.designs: List[Design] = []
        self.unit_suffix: str = "mm"
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header Title
        hdr_layout = QHBoxLayout()
        self.title_lbl = QLabel("📐 DETECTED DESIGNS (0)", self)
        self.title_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.5px;")
        hdr_layout.addWidget(self.title_lbl)
        hdr_layout.addStretch()
        layout.addLayout(hdr_layout)

        # Search filter
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("Filter designs (e.g. PD 451)...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setStyleSheet(
            "QLineEdit { background: #0F172A; color: #F8FAFC; border: 1px solid #334155; border-radius: 4px; padding: 4px 8px; font-size: 11px; } "
            "QLineEdit:focus { border-color: #3B82F6; }"
        )
        self.search_input.textChanged.connect(self._filter_designs)
        layout.addWidget(self.search_input)

        # Designs List Widget
        self.designs_list = QListWidget(self)
        self.designs_list.setStyleSheet(
            "QListWidget { background: #0F172A; color: #F8FAFC; border: 1px solid #334155; border-radius: 6px; font-size: 11px; } "
            "QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #1E293B; border-radius: 4px; margin-bottom: 2px; } "
            "QListWidget::item:hover { background: #1E293B; } "
            "QListWidget::item:selected { background: #2563EB; color: #FFFFFF; font-weight: 600; }"
        )
        self.designs_list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.designs_list, 1)

        # Action Buttons
        btn_box = QHBoxLayout()
        self.btn_zoom = QPushButton("🎯 Zoom to Selected", self)
        self.btn_zoom.setStyleSheet(
            "QPushButton { background: #1E293B; color: #F1F5F9; border: 1px solid #334155; border-radius: 4px; padding: 4px 8px; font-size: 11px; font-weight: 600; } "
            "QPushButton:hover { background: #334155; }"
        )
        self.btn_zoom.clicked.connect(self._zoom_current)
        btn_box.addWidget(self.btn_zoom)

        self.btn_fit = QPushButton("⛶ Fit Sheet", self)
        self.btn_fit.setStyleSheet(
            "QPushButton { background: #1E293B; color: #F1F5F9; border: 1px solid #334155; border-radius: 4px; padding: 4px 8px; font-size: 11px; font-weight: 600; } "
            "QPushButton:hover { background: #334155; }"
        )
        self.btn_fit.clicked.connect(self.fit_sheet_requested.emit)
        btn_box.addWidget(self.btn_fit)
        layout.addLayout(btn_box)

    def set_designs(self, designs: List[Design], unit_suffix: str = "mm") -> None:
        self.designs = designs
        self.unit_suffix = unit_suffix
        self.title_lbl.setText(f"📐 DETECTED DESIGNS ({len(designs)})")
        self._filter_designs(self.search_input.text())

    def _filter_designs(self, query: str = "") -> None:
        self.designs_list.blockSignals(True)
        self.designs_list.clear()

        q = query.strip().lower()
        matched: List[Tuple[int, Design]] = []
        for idx, d in enumerate(self.designs):
            code_str = d.design_code or f"Design #{idx + 1}"
            if not q or q in code_str.lower():
                matched.append((idx, d))

        for idx, d in matched:
            code_str = d.design_code or f"Design #{idx + 1}"
            w = max(0.0, d.width)
            h = max(0.0, d.height)
            item_text = (
                f"🏷️ {code_str}\n"
                f"   W: {w:.1f} × H: {h:.1f} {self.unit_suffix}  •  {d.entity_count} entities"
            )
            item = QListWidgetItem(item_text)
            bbox = (d.x, d.y, d.width, d.height)
            item.setData(Qt.ItemDataRole.UserRole, (idx, code_str, bbox))
            item.setToolTip(f"{code_str}\nWidth: {w:.2f} {self.unit_suffix}\nHeight: {h:.2f} {self.unit_suffix}\nPosition: ({d.x:.2f}, {d.y:.2f})")
            self.designs_list.addItem(item)

        if not matched:
            if not self.designs:
                item = QListWidgetItem("No distinct panel designs detected.\nViewing full DXF drawing bounds.")
            else:
                item = QListWidgetItem(f"No designs matching '{query}'")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            item.setForeground(QBrush(QColor("#64748B")))
            self.designs_list.addItem(item)

        self.designs_list.blockSignals(False)

    def _on_row_changed(self, row: int) -> None:
        item = self.designs_list.item(row)
        if item is not None:
            data = item.data(Qt.ItemDataRole.UserRole)
            if data is not None:
                idx, code, bbox = data
                self.design_selected.emit(idx, code, bbox)

    def _zoom_current(self) -> None:
        row = self.designs_list.currentRow()
        if row >= 0:
            self._on_row_changed(row)

    def highlight_design(self, index: int) -> None:
        for r in range(self.designs_list.count()):
            item = self.designs_list.item(r)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data is not None and data[0] == index:
                self.designs_list.blockSignals(True)
                self.designs_list.setCurrentRow(r)
                self.designs_list.blockSignals(False)
                break

    def apply_theme(self, is_dark: bool) -> None:
        bg_main = "#0F172A" if is_dark else "#FFFFFF"
        border_col = "#334155" if is_dark else "#CBD5E1"
        text_primary = "#F8FAFC" if is_dark else "#0F172A"
        text_muted = "#94A3B8" if is_dark else "#64748B"
        btn_bg = "#1E293B" if is_dark else "#F1F5F9"
        btn_text = "#F1F5F9" if is_dark else "#1E293B"
        btn_hover = "#334155" if is_dark else "#E2E8F0"
        item_hover = "#1E293B" if is_dark else "#F1F5F9"

        self.setStyleSheet(f"background-color: {bg_main};")
        self.title_lbl.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {text_muted}; letter-spacing: 0.5px;")

        self.search_input.setStyleSheet(
            f"QLineEdit {{ background: {bg_main}; color: {text_primary}; border: 1px solid {border_col}; border-radius: 4px; padding: 4px 8px; font-size: 11px; }} "
            f"QLineEdit:focus {{ border-color: #3B82F6; }}"
        )
        self.designs_list.setStyleSheet(
            f"QListWidget {{ background: {bg_main}; color: {text_primary}; border: 1px solid {border_col}; border-radius: 6px; font-size: 11px; }} "
            f"QListWidget::item {{ padding: 6px 8px; border-bottom: 1px solid {border_col}; border-radius: 4px; margin-bottom: 2px; }} "
            f"QListWidget::item:hover {{ background: {item_hover}; }} "
            f"QListWidget::item:selected {{ background: #2563EB; color: #FFFFFF; font-weight: 600; }}"
        )
        btn_style = (
            f"QPushButton {{ background: {btn_bg}; color: {btn_text}; border: 1px solid {border_col}; border-radius: 4px; padding: 4px 8px; font-size: 11px; font-weight: 600; }} "
            f"QPushButton:hover {{ background: {btn_hover}; }}"
        )
        self.btn_zoom.setStyleSheet(btn_style)
        self.btn_fit.setStyleSheet(btn_style)


# ---------------------------------------------------------------------------
# Right Side Panel: Layers & Properties Inspector (DxfInspectorWidget)
# ---------------------------------------------------------------------------

class DxfInspectorWidget(QWidget):
    """
    Inspector drawer on the right side of the DXF Preview workspace.
    Tab 1: CAD Layers checklist with color swatches & visibility toggles.
    Tab 2: Technical Drawing Properties & Entity Geometry Breakdown.
    """

    layer_toggled = Signal(str, bool)
    show_all_layers = Signal()
    hide_all_layers = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.is_dark: bool = True
        self._last_parsed_data: Optional[ParsedDxfData] = None
        self._last_layers: Optional[Dict[str, LayerInfo]] = None
        self._last_sel_design: Optional[Tuple[str, Tuple[float, float, float, float], str]] = None
        self._setup_ui()
        self.apply_theme(True)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tab_widget = QTabWidget(self)

        # Tab 1: CAD Layers
        self.layers_tab = QWidget()
        layers_vbox = QVBoxLayout(self.layers_tab)
        layers_vbox.setContentsMargins(8, 8, 8, 8)
        layers_vbox.setSpacing(6)

        layer_actions = QHBoxLayout()
        self.lbl_layers = QLabel("📑 CAD LAYERS", self.layers_tab)
        layer_actions.addWidget(self.lbl_layers)
        layer_actions.addStretch()

        self.btn_show_all = QPushButton("Show All", self.layers_tab)
        self.btn_show_all.clicked.connect(self.show_all_layers.emit)
        layer_actions.addWidget(self.btn_show_all)

        self.btn_hide_all = QPushButton("Hide All", self.layers_tab)
        self.btn_hide_all.clicked.connect(self.hide_all_layers.emit)
        layer_actions.addWidget(self.btn_hide_all)
        layers_vbox.addLayout(layer_actions)

        self.layers_scroll = QScrollArea(self.layers_tab)
        self.layers_scroll.setWidgetResizable(True)
        self.layers_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.layers_container = QWidget()
        self.layers_layout = QVBoxLayout(self.layers_container)
        self.layers_layout.setContentsMargins(0, 0, 0, 0)
        self.layers_layout.setSpacing(4)
        self.layers_layout.addStretch()
        self.layers_scroll.setWidget(self.layers_container)
        layers_vbox.addWidget(self.layers_scroll, 1)

        self.tab_widget.addTab(self.layers_tab, "📑 CAD Layers")

        # Tab 2: Properties Inspector
        self.props_tab = QWidget()
        props_vbox = QVBoxLayout(self.props_tab)
        props_vbox.setContentsMargins(8, 8, 8, 8)
        props_vbox.setSpacing(8)

        self.props_scroll = QScrollArea(self.props_tab)
        self.props_scroll.setWidgetResizable(True)
        self.props_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.props_container = QWidget()
        self.props_layout = QVBoxLayout(self.props_container)
        self.props_layout.setContentsMargins(0, 0, 0, 0)
        self.props_layout.setSpacing(10)

        # Group 1: File Information
        self.grp_file = QGroupBox("File Information", self.props_container)
        file_vbox = QVBoxLayout(self.grp_file)
        file_vbox.setSpacing(6)
        self.lbl_prop_name = QLabel("Name: -", self.grp_file)
        self.lbl_prop_size = QLabel("Size: -", self.grp_file)
        self.lbl_prop_version = QLabel("AutoCAD Version: -", self.grp_file)
        for lbl in (self.lbl_prop_name, self.lbl_prop_size, self.lbl_prop_version):
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            file_vbox.addWidget(lbl)
        self.props_layout.addWidget(self.grp_file)

        # Group 2: Bounds & Units
        self.grp_extents = QGroupBox("Drawing Bounds & Units", self.props_container)
        ext_vbox = QVBoxLayout(self.grp_extents)
        ext_vbox.setSpacing(6)
        self.lbl_prop_units = QLabel("Units: Millimeters (mm)", self.grp_extents)
        self.lbl_prop_bounds = QLabel("Extents: -", self.grp_extents)
        self.lbl_prop_dims = QLabel("Dimensions: -", self.grp_extents)
        for lbl in (self.lbl_prop_units, self.lbl_prop_bounds, self.lbl_prop_dims):
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            ext_vbox.addWidget(lbl)
        self.props_layout.addWidget(self.grp_extents)

        # Group 3: Entity Geometry Breakdown
        self.grp_entities = QGroupBox("Entities Breakdown", self.props_container)
        ent_vbox = QVBoxLayout(self.grp_entities)
        ent_vbox.setSpacing(6)
        self.lbl_prop_total_ent = QLabel("Total Entities: 0", self.grp_entities)
        self.lbl_prop_total_ent.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        ent_vbox.addWidget(self.lbl_prop_total_ent)

        self.lbl_prop_breakdown = QLabel("Lines: 0\nArcs: 0\nCircles: 0\nSplines: 0\nPolylines: 0", self.grp_entities)
        self.lbl_prop_breakdown.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        ent_vbox.addWidget(self.lbl_prop_breakdown)
        self.props_layout.addWidget(self.grp_entities)

        # Group 4: Selected Design Info
        self.grp_sel_design = QGroupBox("Active Selected Design", self.props_container)
        des_vbox = QVBoxLayout(self.grp_sel_design)
        des_vbox.setSpacing(6)
        self.lbl_prop_sel_code = QLabel("Code: None selected", self.grp_sel_design)
        self.lbl_prop_sel_dims = QLabel("Dimensions: -", self.grp_sel_design)
        self.lbl_prop_sel_area = QLabel("Area: -", self.grp_sel_design)
        for lbl in (self.lbl_prop_sel_code, self.lbl_prop_sel_dims, self.lbl_prop_sel_area):
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            des_vbox.addWidget(lbl)
        self.props_layout.addWidget(self.grp_sel_design)

        self.props_layout.addStretch()
        self.props_scroll.setWidget(self.props_container)
        props_vbox.addWidget(self.props_scroll, 1)

        self.tab_widget.addTab(self.props_tab, "ℹ️ CAD Inspector")
        layout.addWidget(self.tab_widget)

    def apply_theme(self, is_dark: bool) -> None:
        self.is_dark = is_dark

        bg_main = "#0F172A" if is_dark else "#FFFFFF"
        bg_card = "#1E293B" if is_dark else "#F8FAFC"
        border_col = "#334155" if is_dark else "#CBD5E1"
        title_col = "#94A3B8" if is_dark else "#475569"
        text_primary = "#F8FAFC" if is_dark else "#0F172A"
        btn_bg = "#1E293B" if is_dark else "#F1F5F9"
        btn_hover = "#334155" if is_dark else "#E2E8F0"
        btn_text = "#F1F5F9" if is_dark else "#1E293B"
        tab_bg = "#1E293B" if is_dark else "#F1F5F9"
        tab_sel_bg = "#0F172A" if is_dark else "#FFFFFF"
        tab_text = "#94A3B8" if is_dark else "#64748B"
        tab_sel_text = "#38BDF8" if is_dark else "#0284C7"

        # Explicitly style container and tabs to prevent light palette leakage
        self.setStyleSheet(f"background-color: {bg_main};")

        tab_style = (
            f"QTabWidget::pane {{ border: 1px solid {border_col}; background-color: {bg_main}; border-radius: 6px; }} "
            f"QTabBar::tab {{ background: {tab_bg}; color: {tab_text}; font-size: 11px; font-weight: 600; padding: 6px 14px; border: 1px solid {border_col}; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }} "
            f"QTabBar::tab:selected {{ background: {tab_sel_bg}; color: {tab_sel_text}; font-weight: 700; border-bottom: 2px solid #3B82F6; }} "
            f"QTabBar::tab:hover {{ background: {btn_hover}; }}"
        )
        self.tab_widget.setStyleSheet(tab_style)

        # Tab 1: Layers
        self.layers_tab.setStyleSheet(f"background-color: {bg_main};")
        self.layers_scroll.setStyleSheet(f"QScrollArea {{ background-color: {bg_main}; border: none; }}")
        self.layers_container.setStyleSheet(f"background-color: {bg_main};")

        self.lbl_layers.setStyleSheet(f"color: {text_primary}; font-weight: 700; font-size: 11px; letter-spacing: 0.5px;")
        btn_style = (
            f"QPushButton {{ background: {btn_bg}; color: {btn_text}; border: 1px solid {border_col}; border-radius: 4px; padding: 3px 8px; font-size: 10px; font-weight: 600; }} "
            f"QPushButton:hover {{ background: {btn_hover}; }}"
        )
        self.btn_show_all.setStyleSheet(btn_style)
        self.btn_hide_all.setStyleSheet(btn_style)

        # Tab 2: Properties
        self.props_tab.setStyleSheet(f"background-color: {bg_main};")
        self.props_scroll.setStyleSheet(f"QScrollArea {{ background-color: {bg_main}; border: none; }}")
        self.props_container.setStyleSheet(f"background-color: {bg_main};")

        grp_style = (
            f"QGroupBox {{ background-color: {bg_card}; color: {title_col}; font-weight: 700; font-size: 11px; border: 1px solid {border_col}; border-radius: 6px; margin-top: 10px; padding: 12px 10px 10px 10px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; background: {bg_card}; color: {title_col}; }}"
        )
        self.grp_file.setStyleSheet(grp_style)
        self.grp_extents.setStyleSheet(grp_style)
        self.grp_entities.setStyleSheet(grp_style)

        orange_border = "#EA580C" if is_dark else "#F97316"
        orange_bg = "#1E293B" if is_dark else "#FFF7ED"
        orange_title = "#FB923C" if is_dark else "#C2410C"
        grp_design_style = (
            f"QGroupBox {{ background-color: {orange_bg}; color: {orange_title}; font-weight: 700; font-size: 11px; border: 1px solid {orange_border}; border-radius: 6px; margin-top: 10px; padding: 12px 10px 10px 10px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; background: {orange_bg}; color: {orange_title}; }}"
        )
        self.grp_sel_design.setStyleSheet(grp_design_style)

        # Re-render content with active theme palette
        if self._last_layers is not None:
            self.populate_layers(self._last_layers)
        if self._last_parsed_data is not None:
            self.populate_properties(self._last_parsed_data)
        if self._last_sel_design is not None:
            code, bbox, u = self._last_sel_design
            self.update_selected_design(code, bbox, u)

    def _create_layer_swatch(self, color: QColor) -> QIcon:
        size = 14
        pix = QPixmap(size, size)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(color))
        border_col = QColor(255, 255, 255, 160) if self.is_dark else QColor(15, 23, 42, 140)
        p.setPen(QPen(border_col, 1))
        p.drawRoundedRect(1, 1, size - 2, size - 2, 3, 3)
        p.end()
        return QIcon(pix)

    def populate_layers(self, layers: Dict[str, LayerInfo]) -> None:
        self._last_layers = layers
        while self.layers_layout.count() > 1:
            item = self.layers_layout.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.deleteLater()

        text_col = "#F8FAFC" if self.is_dark else "#0F172A"
        hover_col = "#1E293B" if self.is_dark else "#F1F5F9"

        for lname, linfo in sorted(layers.items(), key=lambda x: x[0].lower()):
            chk = QCheckBox(f"{lname}  ({linfo.entity_count})", self.layers_container)
            chk.setChecked(linfo.is_on and not linfo.is_frozen)
            chk.setIcon(self._create_layer_swatch(linfo.color))
            chk.setIconSize(QSize(14, 14))
            chk.setStyleSheet(
                f"QCheckBox {{ color: {text_col}; font-size: 11px; font-weight: 600; padding: 4px 6px; border-radius: 4px; }} "
                f"QCheckBox:hover {{ background-color: {hover_col}; }} "
            )
            chk.toggled.connect(lambda checked, name=lname: self.layer_toggled.emit(name, checked))
            self.layers_layout.insertWidget(self.layers_layout.count() - 1, chk)

    def set_all_layers_visible(self, visible: bool) -> None:
        for i in range(self.layers_layout.count() - 1):
            item = self.layers_layout.itemAt(i)
            if item is not None:
                w = item.widget()
                if isinstance(w, QCheckBox):
                    w.setChecked(visible)

    def populate_properties(self, data: ParsedDxfData) -> None:
        self._last_parsed_data = data
        p = data.file_path
        sz = data.file_size_bytes
        sz_str = f"{sz / 1024:.1f} KB" if sz < 1024 * 1024 else f"{sz / (1024 * 1024):.1f} MB"
        u = data.units_suffix
        bx, by, bw, bh = data.drawing_bounds

        muted = "#94A3B8" if self.is_dark else "#64748B"
        val = "#F8FAFC" if self.is_dark else "#0F172A"
        accent_blue = "#38BDF8" if self.is_dark else "#0284C7"
        accent_green = "#34D399" if self.is_dark else "#059669"
        accent_amber = "#FBBF24" if self.is_dark else "#D97706"

        self.lbl_prop_name.setText(f'<span style="color: {muted}; font-weight: 600;">Name:</span> <span style="color: {val}; font-weight: 700;">{p.name}</span>')
        self.lbl_prop_size.setText(f'<span style="color: {muted}; font-weight: 600;">Size:</span> <span style="color: {accent_blue}; font-weight: 700;">{sz_str}</span>')
        self.lbl_prop_version.setText(f'<span style="color: {muted}; font-weight: 600;">DXF Version:</span> <span style="color: {val}; font-weight: 600;">{data.dxf_version_name} ({data.dxf_version_code})</span>')

        self.lbl_prop_units.setText(f'<span style="color: {muted}; font-weight: 600;">Units:</span> <span style="color: {accent_green}; font-weight: 700;">{data.units_name} ({u})</span>')
        self.lbl_prop_bounds.setText(f'<span style="color: {muted}; font-weight: 600;">Extents:</span> <span style="color: {val}; font-weight: 600;">[{bx:.1f}, {by:.1f}] to [{bx + bw:.1f}, {by + bh:.1f}]</span>')
        self.lbl_prop_dims.setText(f'<span style="color: {muted}; font-weight: 600;">Dimensions:</span> <span style="color: {accent_amber}; font-weight: 700;">{bw:.1f} × {bh:.1f} {u}</span>')

        self.lbl_prop_total_ent.setText(f'<span style="color: {accent_blue}; font-size: 12px; font-weight: 700;">Total Entities: {data.total_entity_count}</span>')
        breakdown_lines = []
        for etype, cnt in sorted(data.entity_counts_by_type.items(), key=lambda x: -x[1]):
            breakdown_lines.append(f'<span style="color: {val}; font-size: 11px;">• {etype}: <b>{cnt:,}</b></span>')
        if breakdown_lines:
            self.lbl_prop_breakdown.setText("<br>".join(breakdown_lines[:12]))
        else:
            self.lbl_prop_breakdown.setText(f'<span style="color: {muted};">No entities parsed</span>')

    def update_selected_design(self, code: str, bbox: Tuple[float, float, float, float], unit_suffix: str = "mm") -> None:
        self._last_sel_design = (code, bbox, unit_suffix)
        bx, by, bw, bh = bbox
        area = bw * bh

        lbl_col = "#FDBA74" if self.is_dark else "#C2410C"
        val_col = "#FFFFFF" if self.is_dark else "#7C2D12"
        area_col = "#FED7AA" if self.is_dark else "#9A3412"

        self.lbl_prop_sel_code.setText(f'<span style="color: {lbl_col}; font-weight: 600;">Code:</span> <span style="color: {val_col}; font-weight: 700; font-size: 12px;">{code or "Unlabeled"}</span>')
        self.lbl_prop_sel_dims.setText(f'<span style="color: {lbl_col}; font-weight: 600;">Dimensions:</span> <span style="color: {val_col}; font-weight: 700;">{bw:.2f} × {bh:.2f} {unit_suffix}</span>')
        self.lbl_prop_sel_area.setText(f'<span style="color: {lbl_col}; font-weight: 600;">Area:</span> <span style="color: {area_col}; font-weight: 700;">{area:,.1f} sq {unit_suffix}</span>')


# ---------------------------------------------------------------------------
# Dedicated Full Workspace Page (DxfPreviewPage)
# ---------------------------------------------------------------------------

class DxfPreviewPage(QWidget):
    """
    Dedicated, full-screen capable DXF Vector CAD Browser & Preview Workspace.
    Point 2 development:
    - Left side: Folder Explorer (top) + Detected Designs list (bottom).
    - Center: Ultra-crisp vector graphics view with zoom, pan, coordinates.
    - Top: Modern CAD toolbar with Fit All, Zoom Design, In, Out, 100%,
      interactive precision Measure Tool, Dark/Light Mode, Export PNG,
      and Open in CAD Application.
    - Right side: Collapsible CAD Layers & Properties Inspector drawer.
    - Bottom: Real-time CAD status bar with units, coords, entity count.
    """

    open_external_signal = Signal(str)

    def __init__(self, initial_folder: Optional[Path | str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_dxf_path: Optional[Path] = None
        self.matched_bbox: Optional[Tuple[float, float, float, float]] = None
        self.matched_code: Optional[str] = None
        self.is_dark_mode: bool = True
        self.external_cad_app: str = ""

        self.scene = DxfScene(self)
        self.view = DxfView(self.scene, self)
        self.loader_worker: Optional[DxfLoaderWorker] = None

        self._load_app_settings()
        self._setup_ui(initial_folder)
        self._wire_signals()

    def _load_app_settings(self) -> None:
        settings = QSettings("DesignMatcher", "DxfViewer")
        self.is_dark_mode = bool(settings.value("dark_mode", True, type=bool))
        self.external_cad_app = str(settings.value("external_cad_app", "", type=str))

    def _save_app_settings(self) -> None:
        settings = QSettings("DesignMatcher", "DxfViewer")
        settings.setValue("dark_mode", self.is_dark_mode)
        settings.setValue("external_cad_app", self.external_cad_app)

    def _setup_ui(self, initial_folder: Optional[Path | str] = None) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Top CAD Navigation Toolbar
        self.toolbar = QToolBar(self)
        self.toolbar.setStyleSheet(
            "QToolBar { background: #1E293B; border-bottom: 1px solid #334155; padding: 4px; spacing: 4px; } "
            "QToolButton { background: #0F172A; color: #F1F5F9; border: 1px solid #334155; border-radius: 4px; "
            "padding: 4px 8px; font-size: 11px; font-weight: 600; } "
            "QToolButton:hover { background: #334155; border-color: #475569; } "
            "QToolButton:checked { background: #2563EB; border-color: #3B82F6; color: white; }"
        )

        self.btn_fit = QToolButton(self.toolbar)
        self.btn_fit.setText("⛶ Fit All (0)")
        self.btn_fit.setToolTip("Fit entire drawing into view [Key: 0]")
        self.toolbar.addWidget(self.btn_fit)

        self.btn_zoom_design = QToolButton(self.toolbar)
        self.btn_zoom_design.setText("🎯 Zoom Design")
        self.btn_zoom_design.setToolTip("Zoom to selected detected design")
        self.toolbar.addWidget(self.btn_zoom_design)

        self.btn_zoom_in = QToolButton(self.toolbar)
        self.btn_zoom_in.setText("➕ Zoom In (+)")
        self.toolbar.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QToolButton(self.toolbar)
        self.btn_zoom_out.setText("➖ Zoom Out (-)")
        self.toolbar.addWidget(self.btn_zoom_out)

        self.btn_zoom_100 = QToolButton(self.toolbar)
        self.btn_zoom_100.setText("1:1 (100%)")
        self.btn_zoom_100.setToolTip("Reset to 100% scale")
        self.toolbar.addWidget(self.btn_zoom_100)

        self.toolbar.addSeparator()

        self.btn_measure = QToolButton(self.toolbar)
        self.btn_measure.setCheckable(True)
        self.btn_measure.setText("📏 Measure")
        self.btn_measure.setToolTip("Click two points to measure distance in drawing units [Key: M / Esc to exit]")
        self.toolbar.addWidget(self.btn_measure)

        self.btn_clear_measure = QToolButton(self.toolbar)
        self.btn_clear_measure.setText("✖ Clear")
        self.btn_clear_measure.setToolTip("Clear active measurement lines from drawing")
        self.toolbar.addWidget(self.btn_clear_measure)

        self.btn_theme = QToolButton(self.toolbar)
        self.btn_theme.setText("🌙 Dark" if self.is_dark_mode else "☀️ Light")
        self.btn_theme.setToolTip("Toggle dark / light CAD contrast background")
        self.toolbar.addWidget(self.btn_theme)

        self.toolbar.addSeparator()

        self.btn_layers_toggle = QToolButton(self.toolbar)
        self.btn_layers_toggle.setCheckable(True)
        self.btn_layers_toggle.setChecked(True)
        self.btn_layers_toggle.setText("📑 CAD Layers")
        self.btn_layers_toggle.setToolTip("Toggle CAD Layers & Properties Inspector panel")
        self.toolbar.addWidget(self.btn_layers_toggle)

        self.btn_export_png = QToolButton(self.toolbar)
        self.btn_export_png.setText("📷 Export PNG")
        self.btn_export_png.setToolTip("Render and save high-resolution vector drawing as PNG image")
        self.toolbar.addWidget(self.btn_export_png)

        self.toolbar.addSeparator()

        # Open in Application dropdown
        self.btn_open_app = QToolButton(self.toolbar)
        self.btn_open_app.setText("🚀 Open in CAD App")
        self.btn_open_app.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_open_app.setStyleSheet(
            "QToolButton { background: #059669; color: white; border: none; font-weight: 700; padding: 4px 10px; } "
            "QToolButton:hover { background: #047857; }"
        )

        app_menu = QMenu(self.btn_open_app)
        self.act_open_default = app_menu.addAction("Open with Default Associated CAD App")
        self.act_choose_app = app_menu.addAction("Choose Program (AutoCAD, Laser Software)...")
        app_menu.addSeparator()
        self.act_show_folder = app_menu.addAction("Show in Folder (Explorer / Finder)")
        self.act_copy_path = app_menu.addAction("Copy DXF File Path")
        self.btn_open_app.setMenu(app_menu)
        self.toolbar.addWidget(self.btn_open_app)

        main_layout.addWidget(self.toolbar)

        # 2. Main Body Splitter (4 Columns side-by-side)
        # Column 1: Folders & DXF Files
        # Column 2: Detected Designs List
        # Column 3: Vector CAD Preview Canvas
        # Column 4: CAD Layers & Properties Inspector
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.main_splitter.setHandleWidth(2)

        # Column 1: DXF Folders and Files Browser
        self.folder_explorer = DxfFolderExplorerWidget(initial_folder=initial_folder, parent=self.main_splitter)
        self.folder_explorer.setMinimumWidth(200)
        self.folder_explorer.setMaximumWidth(380)
        self.main_splitter.addWidget(self.folder_explorer)

        # Column 2: Detected Designs Panel
        self.detected_designs = DxfDetectedDesignsWidget(parent=self.main_splitter)
        self.detected_designs.setMinimumWidth(210)
        self.detected_designs.setMaximumWidth(400)
        self.main_splitter.addWidget(self.detected_designs)

        # Column 3: Center Drawing Preview Viewport with Loading Overlay
        self.view_container = QWidget(self.main_splitter)
        self.view_container.setMinimumWidth(320)
        vbox_view = QVBoxLayout(self.view_container)
        vbox_view.setContentsMargins(0, 0, 0, 0)
        vbox_view.setSpacing(0)
        vbox_view.addWidget(self.view)

        self.loading_overlay = QFrame(self.view_container)
        self.loading_overlay.setStyleSheet("background: rgba(15, 23, 42, 0.85); border-radius: 8px;")
        ol_layout = QVBoxLayout(self.loading_overlay)
        ol_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label = QLabel("Loading DXF Vector Geometry...", self.loading_overlay)
        self.loading_label.setStyleSheet("color: #FFFFFF; font-size: 14px; font-weight: 600;")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ol_layout.addWidget(self.loading_label)

        self.progress_bar = QProgressBar(self.loading_overlay)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedWidth(200)
        ol_layout.addWidget(self.progress_bar)
        self.loading_overlay.setVisible(False)

        self.main_splitter.addWidget(self.view_container)

        # Column 4: CAD Layers & Properties Inspector
        self.inspector = DxfInspectorWidget(parent=self.main_splitter)
        self.inspector.setMinimumWidth(200)
        self.inspector.setMaximumWidth(360)
        self.main_splitter.addWidget(self.inspector)

        # Proportions: Col 1 (18%), Col 2 (20%), Col 3 (44%), Col 4 (18%)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setStretchFactor(2, 5)
        self.main_splitter.setStretchFactor(3, 2)

        main_layout.addWidget(self.main_splitter, 1)

        # 3. Bottom Status Bar
        self.status_bar = QStatusBar(self)
        self.status_bar.setStyleSheet("QStatusBar { background: #1E293B; color: #94A3B8; font-size: 11px; padding: 2px 8px; }")

        self.lbl_coord = QLabel("X: 0.00  Y: 0.00 mm", self)
        self.lbl_coord.setStyleSheet("color: #F8FAFC; font-weight: 600;")
        self.status_bar.addWidget(self.lbl_coord)

        self.lbl_zoom = QLabel("Zoom: 100%", self)
        self.status_bar.addWidget(self.lbl_zoom)

        self.lbl_units = QLabel("Units: Millimeters (mm)", self)
        self.status_bar.addWidget(self.lbl_units)

        self.lbl_stats = QLabel("0 designs detected  •  0 entities", self)
        self.status_bar.addWidget(self.lbl_stats)

        self.lbl_file = QLabel("No DXF loaded", self)
        self.lbl_file.setStyleSheet("color: #38BDF8; font-weight: 600;")
        self.status_bar.addPermanentWidget(self.lbl_file)

        main_layout.addWidget(self.status_bar)
        self._apply_page_theme()

    def _wire_signals(self) -> None:
        # Toolbar actions
        self.btn_fit.clicked.connect(self.view.zoom_to_fit)
        self.btn_zoom_design.clicked.connect(self._zoom_to_selected_design)
        self.btn_zoom_in.clicked.connect(self.view.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view.zoom_out)
        self.btn_zoom_100.clicked.connect(self.view.reset_zoom)
        self.btn_measure.toggled.connect(self._toggle_measure_mode)
        self.btn_clear_measure.clicked.connect(self.view.clear_measurement)
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_layers_toggle.toggled.connect(self.inspector.setVisible)
        self.btn_export_png.clicked.connect(self.export_png_image)

        # External application launcher
        self.btn_open_app.clicked.connect(self._open_in_application)
        self.act_open_default.triggered.connect(self._open_in_default_app)
        self.act_choose_app.triggered.connect(self._choose_custom_app)
        self.act_show_folder.triggered.connect(self._show_in_folder)
        self.act_copy_path.triggered.connect(self._copy_path)

        # Folder explorer -> file selection
        self.folder_explorer.file_selected.connect(self.load_file)

        # Detected designs list -> design selection
        self.detected_designs.design_selected.connect(self._on_design_selected_from_list)
        self.detected_designs.fit_sheet_requested.connect(self.view.zoom_to_fit)

        # Inspector layers
        self.inspector.layer_toggled.connect(self.scene.set_layer_visible)
        self.inspector.show_all_layers.connect(lambda: self._set_all_layers(True))
        self.inspector.hide_all_layers.connect(lambda: self._set_all_layers(False))

        # View signals
        self.view.cursor_moved_signal.connect(self._on_cursor_moved)
        self.view.zoom_changed_signal.connect(self._on_zoom_changed)
        self.view.design_selected_signal.connect(self._on_design_clicked_in_view)
        self.view.measure_updated_signal.connect(self._on_measure_updated)

    def _set_all_layers(self, visible: bool) -> None:
        self.inspector.set_all_layers_visible(visible)
        if self.scene.parsed_data:
            for lname in self.scene.parsed_data.layers.keys():
                self.scene.set_layer_visible(lname, visible)

    def load_file(
        self,
        dxf_path: Path | str,
        matched_bbox: Optional[Tuple[float, float, float, float]] = None,
        matched_code: Optional[str] = None,
        design_code: Optional[str] = None,
    ) -> None:
        """Loads and visualizes a DXF CAD drawing asynchronously."""
        code = design_code if design_code is not None else matched_code
        path = Path(dxf_path).resolve()
        self.current_dxf_path = path
        self.matched_bbox = matched_bbox
        self.matched_code = code

        self.lbl_file.setText(path.name)
        self.folder_explorer.select_file(path)

        # Check LRU cache synchronously first for instant switching
        cached = GLOBAL_DXF_CACHE.get(path)
        if cached is not None:
            self._on_dxf_loaded(cached)
            return

        # Show loading overlay
        self.loading_label.setText(f"Loading '{path.name}'...")
        self.loading_overlay.setGeometry(self.view_container.rect())
        self.loading_overlay.setVisible(True)

        if self.loader_worker is not None and self.loader_worker.isRunning():
            self.loader_worker.quit()
            self.loader_worker.wait()

        self.loader_worker = DxfLoaderWorker(path, self)
        self.loader_worker.loaded_signal.connect(self._on_dxf_loaded)
        self.loader_worker.error_signal.connect(self._on_dxf_error)
        self.loader_worker.start()

    def _on_dxf_loaded(self, data: ParsedDxfData) -> None:
        self.loading_overlay.setVisible(False)
        self.scene.build_scene(data, is_dark_mode=self.is_dark_mode)

        # Update status bar info
        u = data.units_suffix
        self.lbl_units.setText(f"Units: {data.units_name} ({u})")
        des_cnt = len(data.designs)
        self.lbl_stats.setText(f"{des_cnt} design(s) detected  •  {data.total_entity_count:,} entities")

        # Populate left detected designs list
        self.detected_designs.set_designs(data.designs, unit_suffix=u)

        # Populate right inspector drawer
        self.inspector.populate_layers(data.layers)
        self.inspector.populate_properties(data)

        # Highlight and zoom to matched design or fit whole drawing
        if self.matched_bbox is not None:
            self.scene.highlight_design(self.matched_bbox, self.matched_code or "")
            self.view.zoom_to_bbox(self.matched_bbox)
            self.inspector.update_selected_design(self.matched_code or "", self.matched_bbox, u)
        elif data.designs:
            # Highlight first detected design by default
            first_d = data.designs[0]
            bbox = (first_d.x, first_d.y, first_d.width, first_d.height)
            self.scene.highlight_design(bbox, first_d.design_code or "Design #1")
            self.view.zoom_to_fit()
            self.inspector.update_selected_design(first_d.design_code or "Design #1", bbox, u)
        else:
            self.view.zoom_to_fit()

    def _on_dxf_error(self, friendly_msg: str, tech_details: str) -> None:
        self.loading_overlay.setVisible(False)
        self.lbl_file.setText(f"Error: {friendly_msg.splitlines()[0]}")
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen" and "pytest" not in sys.modules:
            dlg = QMessageBox(self)
            dlg.setIcon(QMessageBox.Icon.Warning)
            dlg.setWindowTitle("DXF Notice")
            dlg.setText(friendly_msg)
            if tech_details:
                dlg.setDetailedText(tech_details)
            dlg.exec()
        else:
            logger.warning("DXF Notice: %s | %s", friendly_msg, tech_details)

    def _on_design_selected_from_list(self, index: int, code: str, bbox: tuple) -> None:
        self.matched_bbox = bbox
        self.matched_code = code
        self.scene.highlight_design(bbox, code)
        self.view.zoom_to_bbox(bbox)
        u = self.scene.parsed_data.units_suffix if self.scene.parsed_data else "mm"
        self.inspector.update_selected_design(code, bbox, u)
        self.status_bar.showMessage(f"Focused Design #{index}: '{code}' | Bounds: {tuple(round(v, 1) for v in bbox)}", 4000)

    def _on_design_clicked_in_view(self, index: int, code: str, bbox: tuple) -> None:
        self.matched_bbox = bbox
        self.matched_code = code
        self.scene.highlight_design(bbox, code)
        self.detected_designs.highlight_design(index)
        u = self.scene.parsed_data.units_suffix if self.scene.parsed_data else "mm"
        self.inspector.update_selected_design(code, bbox, u)
        self.status_bar.showMessage(f"Selected Design #{index}: '{code}' | Bounds: {tuple(round(v, 1) for v in bbox)}", 4000)

    def _zoom_to_selected_design(self) -> None:
        if self.matched_bbox is not None:
            self.scene.highlight_design(self.matched_bbox, self.matched_code or "")
            self.view.zoom_to_bbox(self.matched_bbox)
        else:
            self.view.zoom_to_fit()

    def _toggle_measure_mode(self, enabled: bool) -> None:
        self.view.measure_mode = enabled
        if enabled:
            unit_s = self.scene.parsed_data.units_suffix if self.scene.parsed_data else "mm"
            self.status_bar.showMessage(
                f"📏 Measure Mode Active: Click Point 1, then click Point 2 to measure distance ({unit_s}). Press Esc to cancel.",
                7000,
            )
        else:
            self.status_bar.clearMessage()

    def _apply_page_theme(self) -> None:
        bg_main = "#0B0F19" if self.is_dark_mode else "#F8FAFC"
        bar_bg = "#1E293B" if self.is_dark_mode else "#F1F5F9"
        border_col = "#334155" if self.is_dark_mode else "#CBD5E1"
        btn_bg = "#0F172A" if self.is_dark_mode else "#FFFFFF"
        btn_text = "#F1F5F9" if self.is_dark_mode else "#1E293B"
        btn_hover = "#334155" if self.is_dark_mode else "#E2E8F0"
        status_text = "#94A3B8" if self.is_dark_mode else "#64748B"

        self.setStyleSheet(f"DxfPreviewPage {{ background-color: {bg_main}; }}")

        self.toolbar.setStyleSheet(
            f"QToolBar {{ background: {bar_bg}; border-bottom: 1px solid {border_col}; padding: 4px; spacing: 4px; }} "
            f"QToolButton {{ background: {btn_bg}; color: {btn_text}; border: 1px solid {border_col}; border-radius: 4px; "
            f"padding: 4px 8px; font-size: 11px; font-weight: 600; }} "
            f"QToolButton:hover {{ background: {btn_hover}; border-color: #3B82F6; }} "
            f"QToolButton:checked {{ background: #2563EB; border-color: #3B82F6; color: white; }}"
        )
        self.btn_open_app.setStyleSheet(
            "QToolButton { background: #059669; color: white; border: none; font-weight: 700; padding: 4px 10px; border-radius: 4px; } "
            "QToolButton:hover { background: #047857; }"
        )

        self.main_splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {border_col}; }} QSplitter {{ background-color: {bg_main}; }}"
        )
        self.view_container.setStyleSheet(f"background-color: {bg_main};")

        self.status_bar.setStyleSheet(
            f"QStatusBar {{ background: {bar_bg}; color: {status_text}; font-size: 11px; padding: 2px 8px; border-top: 1px solid {border_col}; }}"
        )
        coord_col = "#F8FAFC" if self.is_dark_mode else "#0F172A"
        self.lbl_coord.setStyleSheet(f"color: {coord_col}; font-weight: 600;")
        file_col = "#38BDF8" if self.is_dark_mode else "#0284C7"
        self.lbl_file.setStyleSheet(f"color: {file_col}; font-weight: 600;")

        # Forward to child widgets
        self.folder_explorer.apply_theme(self.is_dark_mode)
        self.detected_designs.apply_theme(self.is_dark_mode)
        self.inspector.apply_theme(self.is_dark_mode)

    def toggle_theme(self) -> None:
        self.is_dark_mode = not self.is_dark_mode
        self.btn_theme.setText("🌙 Dark" if self.is_dark_mode else "☀️ Light")
        self.scene.set_dark_mode(self.is_dark_mode)
        self._apply_page_theme()
        self._save_app_settings()

    def export_png_image(self) -> None:
        if self.current_dxf_path is None:
            QMessageBox.information(self, "No Drawing Loaded", "Please select and load a DXF file before exporting.")
            return

        default_name = f"{self.current_dxf_path.stem}_render.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export DXF Render as High-Resolution PNG",
            default_name,
            "PNG Image (*.png);;All Files (*)",
        )
        if save_path:
            success = self.view.export_image(save_path, width=2800)
            if success:
                self.status_bar.showMessage(f"✓ Exported PNG image successfully: {Path(save_path).name}", 6000)
            else:
                QMessageBox.warning(self, "Export Failed", "Could not render drawing to PNG image.")

    def _on_cursor_moved(self, x: float, y: float) -> None:
        unit_s = self.scene.parsed_data.units_suffix if self.scene.parsed_data else "mm"
        self.lbl_coord.setText(f"X: {x:.2f}  Y: {y:.2f} {unit_s}")

    def _on_zoom_changed(self, zoom_pct: float) -> None:
        self.lbl_zoom.setText(f"Zoom: {zoom_pct:.0f}%")

    def _on_measure_updated(self, dist: float, dx: float, dy: float) -> None:
        unit_s = self.scene.parsed_data.units_suffix if self.scene.parsed_data else "mm"
        angle_deg = math.degrees(math.atan2(dy, dx)) if (dx != 0 or dy != 0) else 0.0
        self.status_bar.showMessage(
            f"📏 Distance: {dist:.2f} {unit_s}  (ΔX: {dx:.2f}, ΔY: {dy:.2f}, ∠{angle_deg:.1f}°)",
            8000,
        )

    # -----------------------------------------------------------------------
    # External Application Launchers
    # -----------------------------------------------------------------------

    def _open_in_application(self) -> None:
        if self.external_cad_app and Path(self.external_cad_app).is_file():
            self._launch_custom_app(self.external_cad_app)
        else:
            self._open_in_default_app()

    def _open_in_default_app(self) -> None:
        if self.current_dxf_path is None or not self.current_dxf_path.is_file():
            QMessageBox.warning(self, "File Not Found", f"The DXF file could not be found:\n{self.current_dxf_path}")
            return
        target_str = str(self.current_dxf_path)
        try:
            if sys.platform.startswith("win"):
                os.startfile(target_str)
            else:
                from PySide6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(target_str))
        except Exception as exc:
            logger.warning("Default open failed for %s (%s). Falling back to QDesktopServices.", target_str, exc)
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(target_str))

    def _choose_custom_app(self) -> None:
        exe_filter = "Executables (*.exe *.app);;All Files (*)" if sys.platform.startswith("win") else "Applications (*);;All Files (*)"
        chosen, _ = QFileDialog.getOpenFileName(self, "Select CAD / CAM Application", "", exe_filter)
        if chosen and Path(chosen).is_file():
            self.external_cad_app = chosen
            self._save_app_settings()
            self._launch_custom_app(chosen)

    def _launch_custom_app(self, exe_path: str) -> None:
        if self.current_dxf_path is None or not self.current_dxf_path.is_file():
            QMessageBox.warning(self, "File Not Found", f"The DXF file could not be found:\n{self.current_dxf_path}")
            return
        try:
            subprocess.Popen([exe_path, str(self.current_dxf_path)])
        except Exception as exc:
            QMessageBox.warning(self, "Launch Error", f"Failed to start '{exe_path}':\n{exc}")

    def _show_in_folder(self) -> None:
        if self.current_dxf_path is None or not self.current_dxf_path.is_file():
            QMessageBox.warning(self, "File Not Found", f"The DXF file could not be found:\n{self.current_dxf_path}")
            return
        target_str = str(self.current_dxf_path)
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(f'explorer /select,"{target_str}"')
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", target_str])
            else:
                subprocess.Popen(["xdg-open", str(self.current_dxf_path.parent)])
        except Exception as exc:
            logger.warning("Could not open containing folder: %s", exc)

    def _copy_path(self) -> None:
        if self.current_dxf_path is not None:
            QApplication.clipboard().setText(str(self.current_dxf_path))
            self.status_bar.showMessage("✓ DXF file path copied to clipboard!", 3000)

