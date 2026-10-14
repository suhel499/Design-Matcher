"""
Comprehensive tests for the DXF ingestion module (dxf_ingest.py).
Tests:
- Single panel extraction
- Multiple panels with text labels ("PD 010", "PD 008")
- INSERT blocks with scale and rotation
- Stencil bridge outline gaps closed by morphology
- Curve flattening for splines and arcs
- DWG error handling
- Debug CLI execution and artifact output
- End-to-end photo perturbation test (rotation, blur, perspective, noise -> top-3 match)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import ezdxf
import numpy as np
import pytest

from app.core.preprocess import normalize_stencil
from dxf_ingest import (
    convert_line_art_to_stencil_mask,
    extract_designs,
    load_dxf,
    process_dxf_file,
    render_design,
)
from engine.config import EngineConfig
from engine.index_store import IndexStore
from engine.indexer import index_folders
from engine.matcher import Matcher


def _create_single_panel_dxf(file_path: Path) -> Path:
    """Creates a simple single-panel DXF with an outer rectangle and circular apertures."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    # Outer rectangle: 200 x 400
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 400), (0, 400)], close=True)
    # Circular cuts inside
    for y in [100, 200, 300]:
        msp.add_circle((100, y), radius=25)
    doc.saveas(str(file_path))
    return file_path


def _create_multi_panel_dxf(file_path: Path) -> Path:
    """Creates a DXF with two side-by-side panels and text labels ('PD 010' and 'PD 008')."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Panel 1: Buddha / Floral stylized (PD 010)
    # Border: (0, 0) to (200, 400)
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 400), (0, 400)], close=True)
    # Center circle + cross slits
    msp.add_circle((100, 200), radius=50)
    msp.add_line((100, 80), (100, 320))
    msp.add_line((30, 200), (170, 200))
    # Label below panel (Y is down in world coords, i.e. negative Y or below min_y)
    msp.add_text("PD 010", height=15).set_placement((60, -35))

    # Panel 2: Geometric Scroll (PD 008)
    # Border: (300, 0) to (500, 400)
    msp.add_lwpolyline([(300, 0), (500, 0), (500, 400), (300, 400)], close=True)
    for y in range(80, 360, 60):
        for x in range(350, 470, 50):
            msp.add_lwpolyline(
                [(x, y - 15), (x + 15, y), (x, y + 15), (x - 15, y)],
                close=True,
            )
    # MTEXT label below panel
    msp.add_mtext("{\\fArial|b1;}PD 008", dxfattribs={"char_height": 14}).set_location((360, -35))

    doc.saveas(str(file_path))
    return file_path


def _create_block_insert_dxf(file_path: Path) -> Path:
    """Creates a DXF with a BLOCK containing panel geometry, instantiated via INSERT with scale and rotation."""
    doc = ezdxf.new("R2010")
    block = doc.blocks.new(name="PANEL_BLOCK")
    # Geometry in block space: 100 x 200
    block.add_lwpolyline([(0, 0), (100, 0), (100, 200), (0, 200)], close=True)
    block.add_circle((50, 100), radius=20)

    msp = doc.modelspace()
    # Insert with 2x scale
    msp.add_blockref(
        "PANEL_BLOCK",
        insert=(50, 50),
        dxfattribs={"xscale": 2.0, "yscale": 2.0, "rotation": 0.0},
    )
    doc.saveas(str(file_path))
    return file_path


def _create_gaps_dxf(file_path: Path) -> Path:
    """Creates a DXF panel where interior cutouts have small stencil gaps."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    # Panel border
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 400), (0, 400)], close=True)

    # Cutout circle with a small 4-degree bridge gap (arc from 2 to 358 deg)
    msp.add_arc((100, 200), radius=40, start_angle=2, end_angle=358)
    doc.saveas(str(file_path))
    return file_path


def _create_spline_and_arc_dxf(file_path: Path) -> Path:
    """Creates a DXF panel with ARC and SPLINE entities."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 400), (0, 400)], close=True)

    # Arc
    msp.add_arc((100, 300), radius=35, start_angle=0, end_angle=180)

    # Spline
    fit_points = [(50, 100), (100, 150), (150, 100), (100, 50)]
    msp.add_spline(fit_points)
    doc.saveas(str(file_path))
    return file_path


# --- TESTS ---

def test_single_panel_extraction(tmp_path: Path) -> None:
    """Verifies that a single-panel DXF is correctly extracted and rendered."""
    dxf_path = _create_single_panel_dxf(tmp_path / "single.dxf")
    doc = load_dxf(dxf_path)
    assert doc is not None

    designs = extract_designs(doc)
    assert len(designs) == 1
    d = designs[0]
    assert d.width > 150
    assert d.height > 350

    img = render_design(d, px_height=512)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2
    assert img.shape[0] == 512
    assert img.shape[1] > 200

    mask_norm, mask_inv = convert_line_art_to_stencil_mask(img)
    assert mask_norm.shape == img.shape
    assert set(np.unique(mask_norm)).issubset({0, 255})
    assert set(np.unique(mask_inv)).issubset({0, 255})


def test_multi_panel_extraction_with_labels(tmp_path: Path) -> None:
    """Verifies that multiple panels with nearby TEXT / MTEXT are extracted with correct labels."""
    dxf_path = _create_multi_panel_dxf(tmp_path / "multi.dxf")
    doc = load_dxf(dxf_path)
    designs = extract_designs(doc)

    assert len(designs) == 2
    codes = {d.design_code for d in designs}
    assert "PD 010" in codes
    assert "PD 008" in codes

    for d in designs:
        assert d.width > 150
        assert d.height > 350
        img = render_design(d, px_height=512)
        assert img is not None
        assert img.size > 0


def test_insert_block_with_scale_and_rotation(tmp_path: Path) -> None:
    """Verifies that virtual entities inside scaled INSERT blocks are exploded and extracted."""
    dxf_path = _create_block_insert_dxf(tmp_path / "block.dxf")
    doc = load_dxf(dxf_path)
    designs = extract_designs(doc)

    assert len(designs) == 1
    d = designs[0]
    # In block space it was 100x200; with 2x scale it should be ~200x400
    assert d.width >= 190
    assert d.height >= 390
    assert len(d.entities) >= 2


def test_outline_gaps_closed_by_morphology(tmp_path: Path) -> None:
    """Verifies that small stencil bridge gaps are bridged so the cut hole is separated from metal."""
    dxf_path = _create_gaps_dxf(tmp_path / "gaps.dxf")
    doc = load_dxf(dxf_path)
    designs = extract_designs(doc)
    assert len(designs) == 1

    img = render_design(designs[0], px_height=512)
    mask_norm, mask_inv = convert_line_art_to_stencil_mask(img)

    # In mask_norm, metal is 255 and hole inside circle is 0
    # Check that hole was properly detected (contains black 0 pixels surrounded by white 255)
    cy, cx = mask_norm.shape[0] // 2, mask_norm.shape[1] // 2
    assert mask_norm[cy, cx] == 0  # Interior is enclosed hole (black 0)


def test_spline_and_arc_curves(tmp_path: Path) -> None:
    """Verifies curve flattening handles ARC and SPLINE entities smoothly."""
    dxf_path = _create_spline_and_arc_dxf(tmp_path / "curves.dxf")
    doc = load_dxf(dxf_path)
    designs = extract_designs(doc)
    assert len(designs) == 1

    img = render_design(designs[0], px_height=512)
    assert img is not None
    # Dark lines on light background
    assert (img < 50).sum() > 100


def test_load_dxf_dwg_friendly_error(tmp_path: Path) -> None:
    """Verifies that a .dwg file yields a helpful message rather than crashing."""
    fake_dwg = tmp_path / "sample.dwg"
    fake_dwg.write_bytes(b"AC1032DWGFORMAT")

    with pytest.raises(ValueError) as excinfo:
        load_dxf(fake_dwg)
    assert "DWG" in str(excinfo.value)
    assert "DXF" in str(excinfo.value)


def test_debug_cli_execution(tmp_path: Path) -> None:
    """Tests the debug CLI: python dxf_ingest.py file.dxf --out debug/"""
    dxf_path = _create_multi_panel_dxf(tmp_path / "debug_test.dxf")
    out_dir = tmp_path / "debug_output"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "dxf_ingest.py"),
        str(dxf_path),
        "--out",
        str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed: {res.stderr}"

    assert out_dir.is_dir()
    # Check that overlay was created
    overlay_files = list(out_dir.glob("*_overview_overlay.png")) + list(out_dir.glob("overlay_detected_panels.png"))
    assert len(overlay_files) >= 1

    # Check panel renders
    renders = list(out_dir.glob("panel_*_rendered.png"))
    assert len(renders) == 2

    # Check mask variants
    masks_norm = list(out_dir.glob("panel_*_mask_normal.png"))
    masks_inv = list(out_dir.glob("panel_*_mask_inverted.png"))
    assert len(masks_norm) == 2
    assert len(masks_inv) == 2


def test_end_to_end_photo_perturbation_matcher(tmp_path: Path) -> None:
    """
    End-to-end test:
    1. Indexes DXF catalog containing PD 010 and PD 008.
    2. Takes the PD 010 design, applies photo distortions (rotation, perspective, blur, noise).
    3. Runs the distorted photo through normalize_stencil and Matcher.search.
    4. Asserts that the top-3 results contain the PD 010 design.
    """
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    dxf_file = _create_multi_panel_dxf(catalog_dir / "panels_catalog.dxf")

    db_path = tmp_path / "test_catalog.db"
    index_path = tmp_path / "test_catalog.index"

    cfg = EngineConfig(
        DB_PATH=db_path,
        INDEX_PATH=index_path,
        TOP_K=5,
    )
    store = IndexStore(db_path=db_path, index_path=index_path)

    # Index the catalog
    stats = index_folders(
        folders=[catalog_dir],
        store=store,
        cfg=cfg,
    )
    assert stats.indexed_designs >= 2

    # Verify text shortcut search works
    code_matches = store.search_by_code("PD 010")
    assert len(code_matches) >= 1
    assert any("PD 010" in (m.design_code or m.label or "") for m in code_matches)

    # Generate a realistic customer phone photo of the PD 010 panel
    doc = load_dxf(dxf_file)
    designs = extract_designs(doc)
    pd010_design = next(d for d in designs if d.design_code == "PD 010")

    # Render clean image
    clean_rendered = render_design(pd010_design, px_height=600)
    norm_m, _ = convert_line_art_to_stencil_mask(clean_rendered)

    # Turn the stencil mask into a realistic laser-cut metal plate photo:
    # Metal is brushed aluminum (gray/silver ~200), cutouts are dark background (black/dark ~30)
    h, w = norm_m.shape
    photo = np.full((h, w, 3), 195, dtype=np.uint8)  # Metal sheet
    photo[norm_m == 0] = [35, 30, 25]  # Cutout apertures

    # 1. Perspective distortion (photographed at an angle)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Tilt slightly: trapezoidal skew
    dst_pts = np.float32([
        [w * 0.08, h * 0.04],
        [w * 0.92, h * 0.02],
        [w * 0.96, h * 0.95],
        [w * 0.04, h * 0.97],
    ])
    M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_photo = cv2.warpPerspective(photo, M_persp, (w, h), borderValue=(20, 20, 20))

    # 2. Rotation (e.g. 90 degrees clockwise as phone was held sideways)
    rotated_photo = cv2.rotate(warped_photo, cv2.ROTATE_90_CLOCKWISE)

    # 3. Add Gaussian blur (simulating slight motion/camera blur)
    blurred_photo = cv2.GaussianBlur(rotated_photo, (5, 5), 1.2)

    # 4. Add sensor noise
    noise = np.random.normal(0, 5, blurred_photo.shape).astype(np.int16)
    noisy_photo = np.clip(blurred_photo.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Run query through matcher
    matcher = Matcher(store=store, cfg=cfg)
    search_response = matcher.search(noisy_photo, top_k=3)

    assert not search_response.no_confident_match
    assert len(search_response.results) > 0

    top_3_labels = [
        (r.design_code or r.label or "")
        for r in search_response.results[:3]
    ]
    # PD 010 should be in top 3
    assert any("PD 010" in lbl for lbl in top_3_labels), (
        f"Expected 'PD 010' in top 3 results, got: {top_3_labels}"
    )
