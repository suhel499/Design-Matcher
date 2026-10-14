"""
Unit tests for DXF CAD parsing, stencil mask rendering, indexing, and Top 5-5 dual format search.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import ezdxf
import numpy as np
from PIL import Image
import pytest

from engine.config import EngineConfig
from engine.dxf_render import (
    generate_dxf_thumbnail,
    get_dxf_metadata,
    render_dxf_preview,
    render_dxf_to_stencil,
)
from engine.features import get_default_embedder
from engine.index_store import IndexStore
from engine.indexer import index_folders
from engine.matcher import Matcher


def create_sample_dxf(file_path: Path, pattern_id: int = 0) -> Path:
    """Creates a synthetic DXF laser cut panel file with distinct apertures."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Outer boundary rectangle (200 x 400 mm plate)
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 400), (0, 400)], close=True)

    if pattern_id == 0:
        # Array of circular holes
        for y in range(50, 360, 50):
            for x in range(40, 170, 40):
                msp.add_circle((x, y), radius=12)
    elif pattern_id == 1:
        # Concentric rings
        for r in range(25, 90, 20):
            msp.add_circle((100, 200), radius=r)
    elif pattern_id == 2:
        # Diamond grid
        for y in range(60, 350, 60):
            for x in range(50, 160, 50):
                msp.add_lwpolyline(
                    [(x, y - 18), (x + 18, y), (x, y + 18), (x - 18, y)],
                    close=True,
                )
    elif pattern_id == 3:
        # Vertical slotted slits
        for x in range(30, 180, 25):
            msp.add_line((x, 40), (x, 360))
    elif pattern_id == 4:
        # Horizontal slotted slits
        for y in range(40, 370, 25):
            msp.add_line((30, y), (170, y))
    else:
        # Diagonal cross cuts
        for d in range(50, 360, 40):
            msp.add_line((30, d), (170, d + 30))

    doc.saveas(str(file_path))
    return file_path


def create_sample_pdf(file_path: Path, pattern_id: int = 0) -> Path:
    """Creates a synthetic PDF catalog page containing a laser panel design."""
    page_bgr = np.full((600, 400, 3), (25, 20, 15), dtype=np.uint8)
    panel = np.full((400, 200, 3), (218, 218, 222), dtype=np.uint8)  # Metal plate
    p_mask = np.full((400, 200), 255, dtype=np.uint8)

    if pattern_id == 0:
        for y in range(50, 360, 50):
            for x in range(40, 170, 40):
                cv2.circle(p_mask, (x, y), 12, 0, -1)
    elif pattern_id == 1:
        for r in range(25, 90, 20):
            cv2.circle(p_mask, (100, 200), r, 0, 8)
    elif pattern_id == 2:
        for y in range(60, 350, 60):
            for x in range(50, 160, 50):
                pts = np.array([[x, y - 18], [x + 18, y], [x, y + 18], [x - 18, y]], np.int32)
                cv2.fillPoly(p_mask, [pts], 0)
    elif pattern_id == 3:
        for x in range(30, 180, 25):
            cv2.line(p_mask, (x, 40), (x, 360), 0, 6)
    else:
        for y in range(40, 370, 25):
            cv2.line(p_mask, (30, y), (170, y), 0, 6)

    panel[p_mask == 0] = (25, 20, 15)
    page_bgr[100:500, 100:300] = panel

    rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img.save(str(file_path), "PDF", resolution=150.0)
    return file_path


def test_dxf_rendering_and_metadata(tmp_path: Path):
    """Tests DXF metadata extraction, stencil binarization, and preview generation."""
    dxf_file = tmp_path / "panel_01.dxf"
    create_sample_dxf(dxf_file, pattern_id=0)

    # 1. Metadata
    meta = get_dxf_metadata(dxf_file)
    assert meta["width"] == 200.0
    assert meta["height"] == 400.0
    assert meta["path_count"] >= 1

    # 2. Stencil mask
    stencil, info = render_dxf_to_stencil(dxf_file, target_size=800)
    assert stencil.shape[0] == 800
    assert stencil.shape[1] == 400
    assert info["closed_loops"] >= 1
    # Metal material is white (255)
    assert stencil[20, 20] == 255
    # Cutout center is black (0)
    assert stencil[100, 80] == 0

    # 3. Preview image
    preview = render_dxf_preview(dxf_file, target_size=800)
    assert preview.ndim == 3
    assert preview.shape[:2] == stencil.shape[:2]

    # 4. Thumbnail PNG bytes
    thumb_png = generate_dxf_thumbnail(stencil, thumb_size=120)
    assert isinstance(thumb_png, bytes)
    assert thumb_png.startswith(b"\x89PNG")


def test_engine_dxf_and_pdf_dual_top5_search(tmp_path: Path):
    """
    Indexes 6 DXF files and 6 PDF files, then verifies that search returns:
    - Exactly Top 5 DXF results
    - Exactly Top 5 PDF results
    - Accurate top match for the query
    """
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()

    # Create 6 DXFs and 6 PDFs
    for i in range(6):
        create_sample_dxf(catalog_dir / f"cad_panel_{i:02d}.dxf", pattern_id=i)
        create_sample_pdf(catalog_dir / f"cat_panel_{i:02d}.pdf", pattern_id=i)

    cfg = EngineConfig(
        DB_PATH=tmp_path / "test.db",
        INDEX_PATH=tmp_path / "test.index",
        MIN_CONFIDENCE=20.0,
    )
    store = IndexStore(db_path=cfg.DB_PATH, index_path=cfg.INDEX_PATH)
    embedder = get_default_embedder("classical", cfg)

    stats = index_folders([catalog_dir], store=store, embedder=embedder, cfg=cfg)
    assert stats.scanned_files == 12
    assert stats.indexed_files == 12
    assert stats.indexed_designs == 12

    matcher = Matcher(store=store, embedder=embedder, cfg=cfg)

    # Search with query photo corresponding to DXF pattern 0 (holes array)
    query_bgr = render_dxf_preview(catalog_dir / "cad_panel_00.dxf", target_size=400)
    # Add minor noise / rotation to simulate camera capture
    query_rot = cv2.rotate(query_bgr, cv2.ROTATE_90_CLOCKWISE)

    resp = matcher.search(query_rot, top_k=5, already_cropped=True)

    assert not resp.no_confident_match
    assert len(resp.dxf_results) == 5
    assert len(resp.pdf_results) == 5

    # Top DXF match should be cad_panel_00.dxf
    top_dxf = resp.dxf_results[0]
    assert Path(top_dxf.pdf_path).name == "cad_panel_00.dxf"
    assert top_dxf.file_type == "dxf"
    assert top_dxf.score > 80.0

    # Top PDF match should also identify pattern 0 (cat_panel_00.pdf)
    top_pdf = resp.pdf_results[0]
    assert Path(top_pdf.pdf_path).name == "cat_panel_00.pdf"
    assert top_pdf.file_type == "pdf"
    assert top_pdf.score > 70.0


def test_app_core_dxf_indexing_and_dual_results(tmp_path: Path):
    """Tests CatalogIndex & DesignSearcher indexing of DXF and dual top results."""
    from app.core.index import CatalogIndex
    from app.core.search import DesignSearcher

    cat_dir = tmp_path / "catalog_library"
    cat_dir.mkdir()
    data_dir = tmp_path / "app_data"

    create_sample_dxf(cat_dir / "design_a.dxf", pattern_id=0)
    create_sample_pdf(cat_dir / "design_a.pdf", pattern_id=0)

    cat_idx = CatalogIndex(data_dir=data_dir)
    tiles_count = cat_idx.index_catalog(cat_dir)
    assert tiles_count >= 2

    searcher = DesignSearcher(catalog_index=cat_idx)
    # Search with stencil mask from pattern 0
    stencil, _ = render_dxf_to_stencil(cat_dir / "design_a.dxf", target_size=400)
    results = searcher.search_mask(stencil, top_k=5)

    assert len(results) >= 2
    dxf_res = [r for r in results if r.file_type == "dxf"]
    pdf_res = [r for r in results if r.file_type == "pdf"]

    assert len(dxf_res) >= 1
    assert len(pdf_res) >= 1
    assert dxf_res[0].pdf_name == "design_a.dxf"
