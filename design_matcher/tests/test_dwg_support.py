"""
Unit tests for DWG file reading, embedded preview extraction, and on-select viewing.
"""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from app.core.dwg_reader import (
    extract_dwg_embedded_thumbnail,
    generate_dwg_cad_card,
    get_dwg_version,
    is_dwg_file,
    read_dwg_preview,
)
from app.core.stage_a_extractor import extract_single_file_metadata
from dxf_ingest import process_dxf_file
from dxf_viewer import (
    DxfFolderExplorerWidget,
    DxfLoaderWorker,
    DxfScene,
    ParsedDxfData,
)


@pytest.fixture
def synthetic_dwg_file(tmp_path: Path) -> Path:
    """
    Creates a valid synthetic AutoCAD 2000 (AC1015) DWG file with an embedded BMP thumbnail.
    """
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:, :] = (30, 25, 20)
    cv2.circle(img, (80, 60), 35, (220, 220, 220), 2)
    _, bmp_bytes = cv2.imencode(".bmp", img)

    dwg_bytes = bytearray(b"AC1015\x00\x00\x00\x00\x00\x00\x00")
    img_offset = 64
    dwg_bytes[13:17] = struct.pack("<I", img_offset)
    dwg_bytes.extend(b"\x00" * (img_offset - len(dwg_bytes)))

    # Image record: 1 byte count=1, 1 byte type=1 (BMP), 4 bytes length
    dwg_bytes.append(1)
    dwg_bytes.append(1)
    dwg_bytes.extend(struct.pack("<I", len(bmp_bytes)))
    dwg_bytes.extend(bmp_bytes)

    dwg_path = tmp_path / "test_panel.dwg"
    dwg_path.write_bytes(dwg_bytes)
    return dwg_path


def test_is_dwg_file():
    assert is_dwg_file("sample.dwg")
    assert is_dwg_file("path/to/DRAWING.DWG")
    assert not is_dwg_file("sample.dxf")
    assert not is_dwg_file("sample.pdf")


def test_get_dwg_version(synthetic_dwg_file: Path):
    code, name = get_dwg_version(synthetic_dwg_file)
    assert code == "AC1015"
    assert "AutoCAD 2000" in name


def test_extract_dwg_embedded_thumbnail(synthetic_dwg_file: Path):
    thumb = extract_dwg_embedded_thumbnail(synthetic_dwg_file)
    assert thumb is not None
    assert thumb.shape == (120, 160, 3)

    preview = read_dwg_preview(synthetic_dwg_file)
    assert preview is not None
    assert preview.shape == (120, 160, 3)


def test_generate_dwg_cad_card(tmp_path: Path):
    dwg_dummy = tmp_path / "empty.dwg"
    dwg_dummy.write_bytes(b"AC1032\x00\x00\x00")
    card = generate_dwg_cad_card(dwg_dummy, "AutoCAD 2018")
    assert card is not None
    assert card.shape == (360, 480, 3)


def test_stage_a_dwg_extraction(synthetic_dwg_file: Path):
    stat = synthetic_dwg_file.stat()
    rec = extract_single_file_metadata(
        str(synthetic_dwg_file),
        modified_ns=stat.st_mtime_ns,
        file_size=stat.st_size,
    )
    assert rec["metadata_status"] == "completed"
    assert rec["extension"] == ".dwg"
    assert "AutoCAD 2000" in rec["searchable_metadata"]
    assert rec["content_hash"] is not None


def test_dxf_ingest_process_dwg(synthetic_dwg_file: Path):
    designs = process_dxf_file(synthetic_dwg_file, px_height=400)
    assert len(designs) == 1
    d = designs[0]
    assert d.design_code == synthetic_dwg_file.stem
    assert d.mask is not None
    assert d.mask_inverted is not None
    assert d.rendered_image is not None


def test_dxf_loader_worker_dwg(qtbot, synthetic_dwg_file: Path):
    worker = DxfLoaderWorker(synthetic_dwg_file)
    with qtbot.waitSignal(worker.loaded_signal, timeout=5000) as blocker:
        worker.start()

    parsed = blocker.args[0]
    assert isinstance(parsed, ParsedDxfData)
    assert parsed.raster_pixmap is not None
    assert not parsed.raster_pixmap.isNull()
    assert parsed.raster_image is not None
    assert "DWG" in parsed.dxf_version_name


def test_dxf_scene_dwg_pixmap(qtbot, synthetic_dwg_file: Path):
    parsed = ParsedDxfData(synthetic_dwg_file)
    img = np.full((100, 150, 3), (25, 20, 15), dtype=np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, 150, 100, 150 * 3, QImage.Format.Format_RGB888)
    parsed.raster_pixmap = QPixmap.fromImage(qimg)
    parsed.drawing_bounds = (0.0, 0.0, 150.0, 100.0)

    scene = DxfScene()
    scene.build_scene(parsed, is_dark_mode=True)
    assert len(scene.items()) > 0


def test_dxf_folder_explorer_dwg(qtbot, tmp_path: Path, synthetic_dwg_file: Path):
    # Copy a dxf and dwg into the same folder
    dxf_sample = tmp_path / "panel_a.dxf"
    dxf_sample.write_text("0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF\n")

    explorer = DxfFolderExplorerWidget(initial_folder=tmp_path)
    qtbot.addWidget(explorer)
    explorer.refresh()

    # Verify both files are discovered in the list
    assert explorer.file_list.count() == 2
    item_texts = [explorer.file_list.item(i).text() for i in range(explorer.file_list.count())]
    assert any("[DWG]" in t for t in item_texts)
    assert any("[DXF]" in t for t in item_texts)

    # Test selecting the DWG file
    with qtbot.waitSignal(explorer.file_selected, timeout=3000) as blocker:
        explorer.select_file(synthetic_dwg_file)
        # Click the item
        for i in range(explorer.file_list.count()):
            item = explorer.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == synthetic_dwg_file:
                explorer._on_item_clicked(item)
                break

    selected_path = blocker.args[0]
    assert selected_path == synthetic_dwg_file


def test_convert_dwg_to_dxf_interface(tmp_path: Path):
    from app.core.dwg_reader import convert_dwg_to_dxf, find_dwg2dxf_binary
    dummy_dwg = tmp_path / "sample.dwg"
    dummy_dwg.write_bytes(b"AC1015" + b"\x00" * 200)

    # Calling convert_dwg_to_dxf should not crash and return None or Path
    result = convert_dwg_to_dxf(dummy_dwg)
    assert result is None or isinstance(result, Path)


def test_dxf_loader_worker_dwg_vector_fallback(qtbot, tmp_path: Path, monkeypatch):
    from app.core import dwg_reader

    # Simulate converted DXF returned from DWG
    simulated_dxf = tmp_path / "simulated.dxf"
    simulated_dxf.write_text(
        "0\nSECTION\n2\nENTITIES\n0\nLINE\n8\n0\n10\n0.0\n20\n0.0\n11\n100.0\n21\n100.0\n0\nENDSEC\n0\nEOF\n"
    )

    dwg_file = tmp_path / "panel_vector.dwg"
    dwg_file.write_bytes(b"AC1032" + b"\x00" * 100)

    monkeypatch.setattr(dwg_reader, "convert_dwg_to_dxf", lambda p: simulated_dxf)

    worker = DxfLoaderWorker(dwg_file)
    with qtbot.waitSignal(worker.loaded_signal, timeout=5000) as blocker:
        worker.start()

    parsed = blocker.args[0]
    assert isinstance(parsed, ParsedDxfData)
    # When vector conversion succeeds, vector_items are populated!
    assert len(parsed.vector_items) > 0
    assert "DWG" in parsed.dxf_version_name


def test_real_sample_dwg_vector_rendering(qtbot, tmp_path: Path):
    """
    Validates end-to-end vector conversion, spline parsing, and high-res rendering
    on the user's real sample DWG file if present.
    """
    sample_dwg = Path("/Users/suhel/Downloads/javed/DWG/shimla 2pc new more 11x48 2mm..dwg")
    if not sample_dwg.is_file():
        pytest.skip(f"Sample DWG not found at {sample_dwg}")

    from app.core.dwg_reader import convert_dwg_to_dxf
    from dxf_viewer import DxfView, DxfScene, load_dxf, parse_dxf_document

    converted_dxf = convert_dwg_to_dxf(sample_dwg)
    assert converted_dxf is not None and converted_dxf.is_file()

    doc = load_dxf(converted_dxf)
    parsed = parse_dxf_document(doc, sample_dwg)

    # Verify splines and polylines preserved from DWG
    assert parsed.entity_counts_by_type.get("SPLINE", 0) >= 90
    assert parsed.entity_counts_by_type.get("LWPOLYLINE", 0) >= 2
    assert len(parsed.vector_items) >= 94

    # Verify clean drawing bounds (frames are ~836 x ~1219 mm)
    bx, by, bw, bh = parsed.drawing_bounds
    assert 800.0 <= bw <= 860.0
    assert 1200.0 <= bh <= 1250.0

    # Build scene and verify vector items (DxfScene batches vector paths per (layer, color))
    scene = DxfScene()
    scene.build_scene(parsed, is_dark_mode=True)
    assert len(parsed.vector_items) >= 94
    assert len(scene.items()) >= 1

    # Test high-res export image (min 3200px on long side)
    view = DxfView(scene)
    qtbot.addWidget(view)
    out_png = tmp_path / "sample_dwg_render_test.png"
    ok = view.export_image(out_png, width=3200)
    assert ok is True
    assert out_png.is_file()
    assert out_png.stat().st_size > 50000  # Non-trivial high-res image

    # Verify dimensions: long side must be >= 3200px
    qimg = QImage(str(out_png))
    assert not qimg.isNull()
    assert max(qimg.width(), qimg.height()) >= 3200


def test_point_entity_crosshair_rendering():
    """
    Tests that POINT entities generate a crosshair '+' marker instead of being dropped.
    """
    import ezdxf
    from dxf_viewer import parse_dxf_document, DxfScene

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_point((50.0, 75.0), dxfattribs={"layer": "MARKERS", "color": 1})

    parsed = parse_dxf_document(doc)
    assert parsed.entity_counts_by_type.get("POINT", 0) == 1
    assert len(parsed.vector_items) == 1

    # Verify crosshair bounds around (50, 75)
    qpath, layer_name, _ = parsed.vector_items[0]
    assert layer_name == "MARKERS"
    br = qpath.boundingRect()
    assert br.center().x() == pytest.approx(50.0, abs=0.1)
    assert br.center().y() == pytest.approx(75.0, abs=0.1)


def test_drawing_bounds_outlier_rejection():
    """
    Tests that stray far-away points do not corrupt the drawing bounds.
    """
    import ezdxf
    from dxf_viewer import parse_dxf_document

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Draw 10 lines clustered between (0, 0) and (100, 100)
    for i in range(10):
        msp.add_line((i * 10, 0), (i * 10, 100))

    # Add 1 stray outlier entity far away at (50000, 50000)
    msp.add_point((50000.0, 50000.0))

    parsed = parse_dxf_document(doc)
    bx, by, bw, bh = parsed.drawing_bounds

    # The bounds must be framed around the cluster (0, 0, 90, 100), not the 50000 outlier
    assert bw < 200.0
    assert bh < 200.0
    assert bx < 10.0
    assert by < 10.0


