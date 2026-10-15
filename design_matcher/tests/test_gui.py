"""
Tests for GUI widgets and MainWindow in headless offscreen environment.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.gui.widgets import CornerWarpDialog, WarpCanvasWidget, CropDialog
from app.gui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_warp_canvas_widget(qapp):
    """Tests WarpCanvasWidget points initialization, reset, and transformations."""
    dummy = np.full((400, 300, 3), 128, dtype=np.uint8)
    canvas = WarpCanvasWidget(dummy)

    # 4 points initialized
    assert canvas.pts_img.shape == (4, 2)
    # Check bounds
    assert np.all(canvas.pts_img[:, 0] >= 0)
    assert np.all(canvas.pts_img[:, 0] <= 300)
    assert np.all(canvas.pts_img[:, 1] >= 0)
    assert np.all(canvas.pts_img[:, 1] <= 400)

    # Custom points set
    custom_pts = np.array([[10, 10], [290, 15], [285, 390], [15, 385]], dtype=np.float32)
    canvas.set_points(custom_pts)
    assert np.allclose(canvas.pts_img, custom_pts)


def test_corner_warp_dialog(qapp):
    """Tests CornerWarpDialog initialization and warp application."""
    dummy = np.full((300, 200, 3), 200, dtype=np.uint8)
    dlg = CornerWarpDialog(dummy)

    assert dlg.canvas is not None
    dlg._apply_warp()
    assert dlg.warped_bgr is not None
    assert dlg.warped_bgr.ndim == 3


def test_main_window_modes_and_controls(qapp, tmp_path):
    """Tests MainWindow mode switching, enhance toggle, and photo loading."""
    win = MainWindow(data_dir=tmp_path)

    # Verify mode combo (6 modes: Auto, Raw Sheet, Installed Gate, Backlit Silhouette, Mixed Lighting, Blurry)
    assert win.mode_combo.count() == 6
    win.mode_combo.setCurrentIndex(5)  # Blurry mode
    assert win.current_mode == "blurry"
    assert win.enhance_enabled is True

    # Load a test photo
    test_photo = np.full((250, 180, 3), 220, dtype=np.uint8)
    win._on_photo_loaded(test_photo, "test_sample.jpg")

    assert win.current_photo_bgr is not None
    assert win.current_mask is not None
    assert win.current_mask.shape == (250, 180)

    # Verify clear photo
    win._clear_photo()
    assert win.current_photo_bgr is None
    assert win.current_mask is None


def test_advanced_crop_studio(qapp):
    """Tests AdvancedCropStudio in rectangle crop and 4-corner warp modes."""
    from app.gui.widgets import AdvancedCropStudio

    dummy = np.full((400, 300, 3), 160, dtype=np.uint8)
    # Add a bright rectangular feature to crop
    dummy[80:320, 60:240] = (240, 240, 240)

    # 1. Test in rect mode
    studio_rect = AdvancedCropStudio(dummy, initial_mode="rect")
    assert studio_rect.canvas.mode == "rect"
    studio_rect._update_live_mask_preview()
    assert studio_rect.final_mask is not None
    assert studio_rect.final_mask.shape[0] > 0
    studio_rect._apply_crop_only()
    assert studio_rect.cropped_bgr is not None
    assert studio_rect.trigger_search is False

    # 2. Test in warp mode
    studio_warp = AdvancedCropStudio(dummy, initial_mode="warp")
    assert studio_warp.canvas.mode == "warp"
    studio_warp._update_live_mask_preview()
    assert studio_warp.final_mask is not None
    studio_warp._apply_and_search()
    assert studio_warp.cropped_bgr is not None
    assert studio_warp.trigger_search is True


def test_main_window_rotation_and_inversion(qapp, tmp_path):
    """Tests rotation updates raw_photo_bgr and inversion bitwise flips without resetting mask."""
    import cv2
    win = MainWindow(data_dir=tmp_path)
    test_photo = np.full((300, 200, 3), 200, dtype=np.uint8)
    win._on_photo_loaded(test_photo, "sample.jpg")

    # Rotate 90 degrees clockwise -> shape becomes (200, 300, 3)
    win._rotate_photo()
    assert win.current_photo_bgr.shape == (200, 300, 3)
    assert win.raw_photo_bgr.shape == (200, 300, 3)

    # Test direct inversion
    mask_before = win.current_mask.copy()
    win._toggle_invert_mask()
    assert win.mask_inverted is True
    # Inverted mask should be bitwise NOT
    assert np.array_equal(win.current_mask, cv2.bitwise_not(mask_before))


def test_dialog_state_synchronization(qapp, tmp_path):
    """Tests that CropDialog initializes with MainWindow's settings and its tuned final_mask is kept."""
    win = MainWindow(data_dir=tmp_path)
    test_photo = np.full((300, 200, 3), 180, dtype=np.uint8)
    win._on_photo_loaded(test_photo, "sample.jpg")
    win.mode_combo.setCurrentIndex(3)  # Backlit
    win.mask_inverted = True

    # Check CropDialog initialization
    dlg = CropDialog(
        win.current_photo_bgr,
        parent=win,
        preset_mode_idx=win.mode_combo.currentIndex(),
        is_inverted=win.mask_inverted,
        enhance=win.enhance_enabled,
    )
    assert dlg.mode_combo.currentIndex() == 3
    assert dlg.is_inverted is True
    assert dlg.enhance_check.isChecked() == win.enhance_enabled


def test_main_window_search_modes_and_dxf(qapp, tmp_path):
    """Tests MainWindow Mode A vs Mode B switching and reference DXF loading."""
    import ezdxf
    win = MainWindow(data_dir=tmp_path)

    # 1. Check default is Mode A
    assert win.search_mode == "mode_a"
    assert not win.photo_tools_container.isHidden()
    assert win.search_btn.text() == "SEARCH DXF LIBRARY (FROM STENCIL)"

    # 2. Switch to Mode B
    win._set_search_mode("mode_b")
    assert win.search_mode == "mode_b"
    assert win.photo_tools_container.isHidden()
    assert win.search_btn.text() == "SEARCH DXF LIBRARY (FROM REFERENCE DXF)"

    # 3. Create a test DXF file and load it as reference
    dxf_file = tmp_path / "ref_test.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)], close=True)
    doc.saveas(str(dxf_file))

    win._load_reference_dxf(str(dxf_file))
    assert win.reference_dxf_path == str(dxf_file)
    assert len(win.reference_dxf_components) >= 1
    assert win.search_btn.isEnabled()
    assert win.current_mask is not None
    assert win.current_mask.shape == (256, 256)

    # Switch back to Mode A
    win._set_search_mode("mode_a")
    assert win.search_mode == "mode_a"
    assert not win.photo_tools_container.isHidden()



def test_result_card_dxf_badges(qapp, tmp_path):
    """Tests ResultCardWidget properly displays DXF preview, match status, and geometric distance."""
    from app.core.search import SearchResult
    from app.gui.widgets import ResultCardWidget
    from engine.dxf_matcher import DxfMatch

    dxf_path = tmp_path / "sample.dxf"
    dxf_path.write_text("0\nSECTION\n0\nENDSEC\n0\nEOF\n")

    raw_match = DxfMatch(
        rank=1,
        file_path=str(dxf_path),
        filename="sample.dxf",
        component_name="Panel 1",
        component_index=0,
        confidence_score=94.5,
        match_status="Verified Match",
        geometric_distance=0.038,
        retrieval_similarity=0.92,
        raster_similarity=0.88,
        aspect_ratio=2.0,
        entity_count=12,
        rotation_deg=90,
        is_mirrored=False,
    )

    res = SearchResult(
        rank=1,
        score=0.945,
        confidence_pct=94.5,
        tile_id=0,
        pdf_path=str(dxf_path),
        pdf_name="sample.dxf",
        folder=str(tmp_path),
        page_number=1,
        tile_index=0,
        crop_type="dxf",
        bbox=[0, 0, 100, 100],
        design_number="Panel 1",
        thumbnail_path=None,
        best_rotation_deg=90,
        iou=0.88,
        chamfer=0.038,
        matched_region="Panel 1",
        file_type="dxf",
        design_code="Panel 1",
        match_status="Verified Match",
        geometric_distance=0.038,
        retrieval_similarity=0.92,
        is_mirrored=False,
        raw_dxf_match=raw_match,
    )

    card = ResultCardWidget(res, tmp_path)
    assert card is not None
    # Check that rank and badges were created
    assert "DXF #1" in [w.text() for w in card.findChildren(object) if hasattr(w, "text")]
    assert "Verified Match" in [w.text() for w in card.findChildren(object) if hasattr(w, "text")]


def test_settings_dialog_dxf_options(qapp, tmp_path):
    """Tests SettingsDialog reads and persists chamfer threshold and candidate limit."""
    from app.core.db import CatalogDatabase
    from app.gui.widgets import SettingsDialog

    db = CatalogDatabase(tmp_path / "test_settings.db")
    dlg = SettingsDialog(db)

    # Defaults
    assert abs(dlg.chamfer_spin.value() - 0.12) < 1e-4
    assert dlg.cand_spin.value() == 30

    # Change and save
    dlg.chamfer_spin.setValue(0.18)
    dlg.cand_spin.setValue(50)
    dlg._save_settings()

    assert float(db.get_setting("chamfer_threshold", "0.12")) == pytest.approx(0.18)
    assert int(db.get_setting("candidate_limit", "30")) == 50


def test_dxf_matcher_imports_and_worker(qapp, tmp_path):
    """Tests that search functions can be imported from engine.dxf_matcher and DxfMatchWorker runs."""
    import cv2
    from engine.dxf_database import DxfDatabase
    from engine.dxf_matcher import search_by_png_stencil, search_by_reference_dxf
    from app.gui.workers import DxfMatchWorker

    db = DxfDatabase(tmp_path / "test_worker.db")
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 50, 255, -1)

    # Test module-level stencil search function
    results = search_by_png_stencil(db, mask, top_k=5)
    assert isinstance(results, list)

    # Test DxfMatchWorker in stencil mode
    worker = DxfMatchWorker(db=db, mode="stencil", query_data=mask, top_k=5)
    worker_results = []
    worker_errors = []
    worker.results_signal.connect(lambda res: worker_results.extend(res))
    worker.error_signal.connect(lambda err: worker_errors.append(err))

    worker.run()
    assert len(worker_errors) == 0


def test_start_dxf_indexing_flow_qfiledialog(qapp, tmp_path, monkeypatch):
    """Tests MainWindow._start_dxf_indexing_flow uses QFileDialog without NameError."""
    from PySide6.QtWidgets import QFileDialog
    from app.gui.main_window import MainWindow
    from app.gui.workers import DxfIndexWorker

    win = MainWindow(data_dir=tmp_path)
    dxf_dir = tmp_path / "dxf_library"
    dxf_dir.mkdir(parents=True, exist_ok=True)

    # Mock QFileDialog.getExistingDirectory, license check, and worker.start to test GUI logic safely
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(dxf_dir))
    monkeypatch.setattr(DxfIndexWorker, "start", lambda self: None)
    monkeypatch.setattr(win.license_manager, "is_operation_allowed", lambda op: (True, "Licensed"))

    # Should run without NameError: name 'QFileDialog' is not defined
    win._start_dxf_indexing_flow()
    assert win.dxf_index_worker is not None



