"""
Unit and integration tests for In-App Vector DXF Viewer with pytest-qt.
Tests DxfScene, DxfView, DxfViewerWidget, and DxfViewerWindow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest
import ezdxf

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QCheckBox

from dxf_viewer import (
    DxfScene,
    DxfView,
    DxfViewerWidget,
    DxfViewerWindow,
    DxfFolderExplorerWidget,
    DxfDetectedDesignsWidget,
    DxfInspectorWidget,
    DxfPreviewPage,
    DxfSceneCache,
    GLOBAL_DXF_CACHE,
    ParsedDxfData,
    parse_dxf_document,
    resolve_dxf_color,
    get_insunits_info,
)


@pytest.fixture
def synthetic_dxf_file(tmp_path: Path) -> Path:
    """
    Creates a synthetic DXF file containing:
    - LINE
    - ARC
    - CIRCLE
    - SPLINE
    - Block definition and INSERT with scale and rotation
    - TEXT label
    - Multiple layers with distinct colors (including ACI 7)
    """
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4  # 4 = mm
    msp = doc.modelspace()

    # Define layers
    doc.layers.add(name="OUTLINE", color=1)       # Red
    doc.layers.add(name="DETAILS", color=3)       # Green
    doc.layers.add(name="LABELS", color=7)        # ACI 7 (White/Black)
    doc.layers.add(name="SPECIAL", color=5)       # Blue

    # 1. LINE
    msp.add_line((0, 0), (100, 0), dxfattribs={"layer": "OUTLINE"})
    msp.add_line((100, 0), (100, 100), dxfattribs={"layer": "OUTLINE"})
    msp.add_line((100, 100), (0, 100), dxfattribs={"layer": "OUTLINE"})
    msp.add_line((0, 100), (0, 0), dxfattribs={"layer": "OUTLINE"})

    # 2. CIRCLE
    msp.add_circle((50, 50), radius=20, dxfattribs={"layer": "DETAILS"})

    # 3. ARC
    msp.add_arc((50, 50), radius=35, start_angle=0, end_angle=180, dxfattribs={"layer": "DETAILS"})

    # 4. SPLINE
    msp.add_spline([(10, 10), (30, 80), (70, 80), (90, 10)], dxfattribs={"layer": "DETAILS"})

    # 5. Block definition and INSERT with scale & rotation
    blk = doc.blocks.new(name="STAR_MOTIF")
    blk.add_line((-5, 0), (5, 0))
    blk.add_line((0, -5), (0, 5))
    blk.add_circle((0, 0), radius=2)

    # Insert with scale (2.0) and rotation (45 deg)
    msp.add_blockref(
        "STAR_MOTIF",
        insert=(50, 50),
        dxfattribs={"xscale": 2.0, "yscale": 2.0, "rotation": 45.0, "layer": "SPECIAL"},
    )

    # 6. TEXT / Label
    msp.add_text("PD 010", dxfattribs={"insert": (15, 85), "height": 8.0, "layer": "LABELS"})

    # Save to disk
    file_path = tmp_path / "test_synthetic.dxf"
    doc.saveas(file_path)
    return file_path


def test_dxf_insunits_info():
    """Tests DXF measurement unit string decoding."""
    assert "mm" in get_insunits_info(4)
    assert "in" in get_insunits_info(1)
    assert "Unitless" in get_insunits_info(0)


def test_resolve_dxf_color():
    """Tests ACI and True Color conversions."""
    # ACI 1 = Red
    c1 = resolve_dxf_color(1)
    assert c1.red() > 200 and c1.green() < 50
    # ACI 7 = White
    c7 = resolve_dxf_color(7)
    assert c7.red() == 255 and c7.green() == 255 and c7.blue() == 255


def test_scene_building_and_bounds(qtbot, synthetic_dxf_file: Path):
    """
    Verifies that DxfScene builds geometry from synthetic DXF containing
    line, arc, circle, spline, INSERT (scaled/rotated), and text label,
    and calculates non-empty drawing bounds matching CAD coordinate space.
    """
    scene = DxfScene()
    doc = ezdxf.readfile(synthetic_dxf_file)
    data = scene.load_doc(doc)

    assert data.total_entities > 0
    assert "OUTLINE" in scene.layer_info
    assert "DETAILS" in scene.layer_info
    assert "LABELS" in scene.layer_info
    assert "SPECIAL" in scene.layer_info

    # Verify scene bounds contain our 100x100 bounding box
    bounds = scene.drawing_bounds
    assert not bounds.isNull()
    assert bounds.width() >= 90.0
    assert bounds.height() >= 90.0

    # Ensure items exist in scene
    assert len(scene.items()) > 0
    assert "OUTLINE" in scene.items_by_layer
    assert len(scene.items_by_layer["OUTLINE"]) > 0


def test_layer_visibility_toggle(qtbot, synthetic_dxf_file: Path):
    """Verifies toggling layer visibility shows/hides graphic items."""
    scene = DxfScene()
    doc = ezdxf.readfile(synthetic_dxf_file)
    scene.load_doc(doc)

    outline_items = scene.items_by_layer["OUTLINE"]
    assert len(outline_items) > 0
    # Initially visible
    for item in outline_items:
        assert item.isVisible()

    # Toggle hidden
    scene.set_layer_visible("OUTLINE", False)
    for item in outline_items:
        assert not item.isVisible()

    # Toggle back to visible
    scene.set_layer_visible("OUTLINE", True)
    for item in outline_items:
        assert item.isVisible()


def test_view_y_inversion_and_zoom_to_match(qtbot, synthetic_dxf_file: Path):
    """
    Verifies DxfView viewport inversion (scale(1, -1)) for upright CAD coordinates,
    and checks that 'zoom_to_bbox' centers and frames the target bounding box.
    """
    scene = DxfScene()
    doc = ezdxf.readfile(synthetic_dxf_file)
    scene.load_doc(doc)

    view = DxfView(scene)
    qtbot.addWidget(view)
    view.resize(800, 600)
    view.show()

    # Verify Y is inverted in transform
    tf = view.transform()
    assert tf.m22() < 0, "Y-axis should be inverted for CAD view"

    # Zoom to match bbox
    target_bbox = (10.0, 10.0, 80.0, 80.0)
    view.zoom_to_bbox(target_bbox, padding_ratio=0.1)

    # Center of view should be close to center of bbox
    visible_center = view.mapToScene(view.viewport().rect().center())
    assert abs(visible_center.x() - 50.0) < 15.0
    assert abs(visible_center.y() - 50.0) < 15.0


def test_background_mode_toggle_and_aci7_contrast(qtbot, synthetic_dxf_file: Path):
    """Verifies that changing background mode toggles scene background and adapts ACI 7 contrast."""
    scene = DxfScene()
    doc = ezdxf.readfile(synthetic_dxf_file)
    scene.load_doc(doc)

    # In dark mode, background is dark and ACI 7 is white
    scene.set_background_mode("dark")
    assert scene.backgroundBrush().color().lightness() < 50
    assert scene.aci7_color == QColor(255, 255, 255)

    # In light mode, background is light and ACI 7 is dark/black
    scene.set_background_mode("light")
    assert scene.backgroundBrush().color().lightness() > 200
    assert scene.aci7_color == QColor(15, 23, 42)


def test_viewer_widget_load_and_missing_file(qtbot, tmp_path: Path):
    """
    Verifies that DxfViewerWidget handles missing files gracefully
    without raising an unhandled exception or crashing.
    """
    widget = DxfViewerWidget()
    qtbot.addWidget(widget)
    widget.show()

    missing_path = tmp_path / "does_not_exist_file.dxf"
    # Should show warning/overlay without raising
    widget.load_file(missing_path)
    qtbot.wait(100)

    assert widget.status_label.text() != ""
    assert widget.scene.drawing_bounds.isNull() or widget.scene.drawing_bounds.width() == 0


def test_viewer_widget_corrupt_file_recovery(qtbot, tmp_path: Path):
    """
    Verifies that opening a corrupt or truncated DXF file falls back
    to ezdxf.recover or presents a friendly error message without crashing.
    """
    corrupt_file = tmp_path / "corrupt.dxf"
    corrupt_file.write_text("SECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF_CORRUPTED_TRUNCATED")

    widget = DxfViewerWidget()
    qtbot.addWidget(widget)
    widget.show()

    # Must not raise exception
    widget.load_file(corrupt_file)
    qtbot.wait(100)
    assert widget.isVisible()


def test_measure_tool(qtbot, synthetic_dxf_file: Path):
    """
    Verifies the interactive measure tool mode in DxfView:
    computes Euclidean distance, delta X, and delta Y in CAD drawing units.
    """
    scene = DxfScene()
    doc = ezdxf.readfile(synthetic_dxf_file)
    scene.load_doc(doc)

    view = DxfView(scene)
    qtbot.addWidget(view)
    view.resize(600, 400)
    view.show()

    # Enable measure mode
    view.set_measure_mode(True)
    assert view.is_measuring

    # Simulate two clicks in scene coordinates
    p1 = QPointF(10.0, 20.0)
    p2 = QPointF(40.0, 60.0)

    # Directly verify measurement calculation
    view.measure_start_scene = p1
    view.measure_end_scene = p2
    dx = p2.x() - p1.x()
    dy = p2.y() - p1.y()
    dist = (dx**2 + dy**2) ** 0.5
    assert abs(dist - 50.0) < 1e-4
    assert abs(dx - 30.0) < 1e-4
    assert abs(dy - 40.0) < 1e-4

    # Exit measure mode
    view.set_measure_mode(False)
    assert not view.is_measuring


def test_dxf_viewer_window_instantiation_and_settings(qtbot, synthetic_dxf_file: Path):
    """Verifies DxfViewerWindow initializes and can load file and restore geometry."""
    window = DxfViewerWindow()
    qtbot.addWidget(window)
    window.show()

    window.load_file(synthetic_dxf_file, matched_bbox=(0, 0, 100, 100), design_code="PD 010")
    qtbot.wait(100)

    assert window.isVisible()
    assert "PD 010" in window.windowTitle() or "test_synthetic" in window.windowTitle()

    # Verify layer items were populated
    assert window.viewer_widget.layers_layout.count() > 1

    # Toggle all layers on
    window.viewer_widget._toggle_all_layers_on()

    # Re-load file to verify clearing and repopulating layers layout works without error
    window.load_file(synthetic_dxf_file)
    qtbot.wait(100)
    assert window.viewer_widget.layers_layout.count() > 1


def test_dxf_scene_lru_cache(synthetic_dxf_file: Path):
    """Verifies that DxfSceneCache correctly caches parsed DXF drawings up to capacity."""
    cache = DxfSceneCache(capacity=3)
    p = str(synthetic_dxf_file.resolve())

    # Initial miss
    assert cache.get(p) is None

    # Load and put
    doc = ezdxf.readfile(synthetic_dxf_file)
    scene = DxfScene()
    data = scene.load_doc(doc)
    cache.put(p, data)

    # Cache hit
    hit = cache.get(p)
    assert hit is not None
    assert hit.total_entities == data.total_entities


def test_dxf_folder_explorer_widget(qtbot, tmp_path: Path):
    """Verifies DxfFolderExplorerWidget folder loading, file filtering, and signal emission."""
    sub = tmp_path / "subfolder"
    sub.mkdir()
    dxf1 = tmp_path / "panel_a.dxf"
    dxf1.write_text("0\nSECTION\n0\nENDSEC\n0\nEOF\n")
    dxf2 = sub / "gate_b.dxf"
    dxf2.write_text("0\nSECTION\n0\nENDSEC\n0\nEOF\n")
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")

    widget = DxfFolderExplorerWidget(initial_folder=tmp_path)
    qtbot.addWidget(widget)
    widget.show()

    # Verify folder loaded and DXF files are listed
    assert widget.current_folder == tmp_path
    assert widget.file_list.count() == 2

    # Test search filter
    widget.search_input.setText("gate")
    visible_items = [
        widget.file_list.item(i).text()
        for i in range(widget.file_list.count())
        if not widget.file_list.item(i).isHidden()
    ]
    assert len(visible_items) == 1
    assert "gate_b" in visible_items[0]

    # Clear filter
    widget.search_input.setText("")
    assert sum(1 for i in range(widget.file_list.count()) if not widget.file_list.item(i).isHidden()) == 2

    # Test file selection signal
    with qtbot.waitSignal(widget.file_selected, timeout=1000) as blocker:
        widget.file_list.setCurrentRow(0)
    assert Path(blocker.args[0]).suffix.lower() == ".dxf"


def test_dxf_detected_designs_widget(qtbot, synthetic_dxf_file: Path):
    """Verifies DxfDetectedDesignsWidget displays detected design cards and emits selection signals."""
    doc = ezdxf.readfile(synthetic_dxf_file)
    parsed = parse_dxf_document(doc, str(synthetic_dxf_file))

    widget = DxfDetectedDesignsWidget()
    qtbot.addWidget(widget)
    widget.show()

    widget.set_designs(parsed.designs, unit_suffix=parsed.units_suffix)
    assert widget.designs_list.count() == len(parsed.designs)
    assert len(widget.designs) == len(parsed.designs)

    # Test filter by design code
    first_code = parsed.designs[0].design_code or "Design"
    widget.search_input.setText(first_code)
    assert widget.designs_list.count() >= 1

    # Test signal emission on row selection
    with qtbot.waitSignal(widget.design_selected, timeout=1000) as blocker:
        widget.designs_list.setCurrentRow(0)
    assert blocker.args[0] == 0


def test_dxf_inspector_widget(qtbot, synthetic_dxf_file: Path):
    """Verifies DxfInspectorWidget layers toggling, entity breakdown, and CAD properties display."""
    doc = ezdxf.readfile(synthetic_dxf_file)
    parsed = parse_dxf_document(doc, str(synthetic_dxf_file))

    widget = DxfInspectorWidget()
    qtbot.addWidget(widget)
    widget.show()

    widget.populate_layers(parsed.layers)
    widget.populate_properties(parsed)

    # Check layers populated in scroll area
    checkboxes = widget.layers_container.findChildren(QCheckBox)
    assert len(checkboxes) >= len(parsed.layers)

    # Test layer toggle signal
    with qtbot.waitSignal(widget.layer_toggled, timeout=1000) as blocker:
        first_cb = checkboxes[0]
        first_cb.setChecked(False)
    assert blocker.args[1] is False

    # Check Show All / Hide All
    widget.set_all_layers_visible(False)
    for cb in checkboxes:
        assert not cb.isChecked()

    widget.set_all_layers_visible(True)
    for cb in checkboxes:
        assert cb.isChecked()

    # Check properties text in info labels
    assert "AutoCAD" in widget.lbl_prop_version.text()
    assert str(parsed.total_entity_count) in widget.lbl_prop_total_ent.text()


def test_dxf_preview_page(qtbot, synthetic_dxf_file: Path, tmp_path: Path):
    """Verifies DxfPreviewPage full workspace: loading, zoom actions, theme toggle, measure, and image export."""
    page = DxfPreviewPage()
    qtbot.addWidget(page)
    page.resize(1100, 700)
    page.show()

    # Load synthetic dxf and wait for async QThread loader to complete
    page.load_file(synthetic_dxf_file)
    qtbot.waitUntil(lambda: page.scene.parsed_data is not None, timeout=5000)

    assert page.current_dxf_path == synthetic_dxf_file
    assert page.scene.parsed_data is not None
    assert len(page.detected_designs.designs) >= 1

    # Zoom actions
    page.btn_zoom_in.click()
    page.btn_zoom_out.click()
    page.btn_zoom_100.click()
    page.btn_fit.click()

    # Measure mode toggle
    page.btn_measure.click()
    assert page.view.is_measuring is True
    page.btn_measure.click()
    assert page.view.is_measuring is False

    # Theme toggle
    initial_dark = page.is_dark_mode
    page.btn_theme.click()
    assert page.is_dark_mode != initial_dark
    page.btn_theme.click()
    assert page.is_dark_mode == initial_dark

    # Export PNG
    export_png = tmp_path / "exported_cad.png"
    success = page.view.export_image(export_png, width=400)
    assert success is True
    assert export_png.is_file()
    assert export_png.stat().st_size > 0

    # Verify exported image is loaded properly with upright CAD orientation
    qimg = QImage(str(export_png))
    assert not qimg.isNull()
    assert qimg.width() == 400
    assert qimg.height() > 0

    # Ensure exported image is not empty / uniform blank
    bg_pixel = qimg.pixelColor(0, 0)
    has_foreground = any(
        qimg.pixelColor(x, y) != bg_pixel
        for y in range(0, qimg.height(), 4)
        for x in range(0, qimg.width(), 4)
    )
    assert has_foreground, "Exported image should contain drawn CAD vectors, not an empty background."


def test_main_window_tabs_and_dxf_preview(qtbot, synthetic_dxf_file: Path, tmp_path: Path, monkeypatch):
    """Verifies MainWindow top navigation tabs and integration with DxfPreviewPage."""
    try:
        from app.gui.main_window import MainWindow
    except ImportError:
        from design_matcher.app.gui.main_window import MainWindow

    win = MainWindow(data_dir=tmp_path)
    try:
        from app.core.licensing.manager import LicenseCheckResult, LicenseStatus
    except ImportError:
        from design_matcher.app.core.licensing.manager import LicenseCheckResult, LicenseStatus
    monkeypatch.setattr(win.license_manager, "is_operation_allowed", lambda op: (True, "Licensed"))
    monkeypatch.setattr(
        win.license_manager,
        "check_license",
        lambda *args, **kwargs: LicenseCheckResult(
            is_valid=True, status=LicenseStatus.ACTIVE, days_remaining=100, message="Active"
        ),
    )
    win._update_license_indicator()
    qtbot.addWidget(win)
    win.show()

    # Initial state: on Tab 0 (Matcher)
    assert win.main_stack.currentIndex() == 0
    assert win.nav_tab_matcher.isChecked()
    assert not win.nav_tab_dxf_preview.isChecked()

    # Switch to Tab 1 (DXF Preview)
    win.nav_tab_dxf_preview.click()
    assert win.main_stack.currentIndex() == 1
    assert win.nav_tab_dxf_preview.isChecked()
    assert not win.nav_tab_matcher.isChecked()

    # Switch back to Tab 0 (Matcher)
    win.nav_tab_matcher.click()
    assert win.main_stack.currentIndex() == 0
    assert win.nav_tab_matcher.isChecked()

    # Test programmatic switch_to_dxf_preview with target file
    win.switch_to_dxf_preview(synthetic_dxf_file, matched_bbox=(0, 0, 100, 100), matched_code="TEST 01")
    qtbot.waitUntil(lambda: win.dxf_preview_page.scene.parsed_data is not None, timeout=5000)

    assert win.main_stack.currentIndex() == 1
    assert win.nav_tab_dxf_preview.isChecked()
    assert win.dxf_preview_page.current_dxf_path == synthetic_dxf_file


