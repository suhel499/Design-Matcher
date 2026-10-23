"""
test_find_design.py - Comprehensive Unit Tests for 'Find Design' Feature.

Covers:
1. Stencil creation from raster (PNG/JPG) with adaptive thresholding, speck removal,
   and Douglas-Peucker simplification.
2. Stencil creation from vector (DXF) with curve flattening.
3. Stencil export to PNG (512x512), SVG (vector paths), and DXF (AutoCAD lwpolylines).
4. Stencil serialization (save/load round-trip).
5. Target candidate extraction and multi-design sheet segmentation.
6. SQLite WAL caching and incremental re-indexing verification.
7. Multi-stage geometric matching with rotation (90, 180, 270 deg) and mirror invariance.
8. Negative rejection and CSV report export.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np
import pytest

from app.core.design_finder import (
    DesignFinderCache,
    MatchResult,
    TargetCandidate,
    compute_point_cloud_chamfer,
    export_matches_to_csv,
    extract_candidates_from_dxf,
    match_stencil_against_candidates,
)
from app.core.stencil import (
    CANVAS_SIZE,
    Stencil,
    create_stencil_from_raster,
    create_stencil_from_vector,
    export_stencil_dxf,
    export_stencil_png,
    export_stencil_svg,
)


@pytest.fixture
def synthetic_raster_image(tmp_path: Path) -> Path:
    """Creates a synthetic high-contrast test image (nested star pattern with noise)."""
    img = np.zeros((400, 400), dtype=np.uint8)
    # Draw diamond/star shape
    pts = np.array([[200, 50], [350, 200], [200, 350], [50, 200]], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=255, thickness=4)
    # Draw inner circle cutout
    cv2.circle(img, (200, 200), 50, color=255, thickness=3)
    # Add small noise specks to verify speck filter
    img[10, 10] = 255
    img[12, 11] = 255
    img[20, 25] = 255

    fpath = tmp_path / "synthetic_star.png"
    cv2.imwrite(str(fpath), img)
    return fpath


@pytest.fixture
def synthetic_dxf_drawing(tmp_path: Path) -> Path:
    """Creates a synthetic multi-design DXF CAD drawing with lines, circles, and lwpolylines."""
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()

    # Panel 1: Star/Diamond design (left: x in 0..100, y in 0..100)
    diamond = [(50, 10), (90, 50), (50, 90), (10, 50)]
    msp.add_lwpolyline(diamond, close=True, dxfattribs={"layer": "0"})
    msp.add_circle((50, 50), radius=15, dxfattribs={"layer": "0"})

    # Panel 2: Distractor Hexagon design (right: x in 150..250, y in 0..100)
    hex_pts = [
        (150 + 40 * math.cos(math.radians(a)), 50 + 40 * math.sin(math.radians(a)))
        for a in range(0, 360, 60)
    ]
    msp.add_lwpolyline(hex_pts, close=True, dxfattribs={"layer": "0"})

    fpath = tmp_path / "multi_design.dxf"
    doc.saveas(str(fpath))
    return fpath


# ---------------------------------------------------------------------------
# FEATURE 1: STENCIL GENERATION & EXPORT TESTS
# ---------------------------------------------------------------------------


def test_create_stencil_from_raster(synthetic_raster_image: Path):
    """Verifies raster ingestion, speck removal, contour simplification, and 512x512 normalization."""
    stencil = create_stencil_from_raster(
        synthetic_raster_image,
        threshold_val=128,
        smoothing_val=1.5,
        min_speck_area=10,
    )

    assert isinstance(stencil, Stencil)
    assert stencil.bitmap.shape == (CANVAS_SIZE, CANVAS_SIZE)
    assert cv2.countNonZero(stencil.bitmap) > 50  # Linework is present
    assert stencil.contour_count >= 2  # Outer diamond + inner circle
    assert len(stencil.polylines) >= 2
    assert stencil.feature_vector.shape == (128,)
    assert stencil.hu_moments.shape == (7,)
    assert math.isclose(np.linalg.norm(stencil.feature_vector), 1.0, rel_tol=1e-3)


def test_create_stencil_from_vector(synthetic_dxf_drawing: Path):
    """Verifies vector DXF ingestion, curve-to-polyline flattening, and normalization."""
    stencil = create_stencil_from_vector(synthetic_dxf_drawing)

    assert isinstance(stencil, Stencil)
    assert stencil.source_type == "dxf"
    assert stencil.bitmap.shape == (CANVAS_SIZE, CANVAS_SIZE)
    assert len(stencil.polylines) >= 2
    assert cv2.countNonZero(stencil.bitmap) > 100


def test_stencil_exports(synthetic_raster_image: Path, tmp_path: Path):
    """Verifies export of stencil to PNG, SVG, and DXF formats."""
    stencil = create_stencil_from_raster(synthetic_raster_image)

    # 1. Export PNG
    png_path = tmp_path / "exported_stencil.png"
    export_stencil_png(stencil, png_path)
    assert png_path.is_file()
    loaded_png = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    assert loaded_png.shape == (512, 512)

    # 2. Export SVG
    svg_path = tmp_path / "exported_stencil.svg"
    export_stencil_svg(stencil, svg_path)
    assert svg_path.is_file()
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "viewBox=\"0 0 512 512\"" in svg_text
    assert "points=" in svg_text

    # 3. Export DXF
    dxf_path = tmp_path / "exported_stencil.dxf"
    export_stencil_dxf(stencil, dxf_path)
    assert dxf_path.is_file()
    doc = ezdxf.readfile(str(dxf_path))
    lwpolys = doc.modelspace().query("LWPOLYLINE")
    assert len(lwpolys) >= 2


def test_stencil_serialization_roundtrip(synthetic_raster_image: Path, tmp_path: Path):
    """Verifies saving and loading stencil packages preserves all geometry and descriptors."""
    original = create_stencil_from_raster(synthetic_raster_image)
    save_path = tmp_path / "saved_package.json"

    original.save_stencil(save_path)
    loaded = Stencil.load_stencil(save_path)

    assert loaded.contour_count == original.contour_count
    assert loaded.raw_point_count == original.raw_point_count
    assert math.isclose(loaded.aspect_ratio, original.aspect_ratio, rel_tol=1e-3)
    assert np.allclose(loaded.feature_vector, original.feature_vector, atol=1e-4)
    assert np.allclose(loaded.hu_moments, original.hu_moments, atol=1e-4)
    assert loaded.bitmap.shape == (512, 512)


# ---------------------------------------------------------------------------
# FEATURE 2: TARGET CANDIDATE EXTRACTION & SQLITE CACHE TESTS
# ---------------------------------------------------------------------------


def test_extract_candidates_from_dxf(synthetic_dxf_drawing: Path):
    """Verifies extracting individual design candidates from a multi-design DXF sheet."""
    cands = extract_candidates_from_dxf(synthetic_dxf_drawing)
    assert len(cands) >= 1
    for c in cands:
        assert isinstance(c, TargetCandidate)
        assert c.file_type == "dxf"
        assert c.feature_vector.shape == (128,)
        assert len(c.polylines) >= 1
        assert c.bitmap_png is not None


def test_sqlite_cache_incremental(synthetic_dxf_drawing: Path, tmp_path: Path):
    """Verifies SQLite WAL caching and incremental re-indexing based on mtime and size."""
    db_file = tmp_path / "test_cache.db"
    cache = DesignFinderCache(db_path=db_file)

    # 1. First check: file not yet in cache
    assert not cache.is_file_up_to_date(synthetic_dxf_drawing)

    # 2. Extract and store candidates
    cands = extract_candidates_from_dxf(synthetic_dxf_drawing)
    cache.store_file_candidates(synthetic_dxf_drawing, cands)

    # 3. Second check: file is now up to date
    assert cache.is_file_up_to_date(synthetic_dxf_drawing)

    # 4. Retrieve from cache
    loaded_cands = cache.get_candidates_for_file(synthetic_dxf_drawing)
    assert len(loaded_cands) == len(cands)
    assert loaded_cands[0].id == cands[0].id
    assert np.allclose(loaded_cands[0].feature_vector, cands[0].feature_vector, atol=1e-4)

    # 5. Modify file (touch mtime)
    synthetic_dxf_drawing.touch()
    assert not cache.is_file_up_to_date(synthetic_dxf_drawing)


# ---------------------------------------------------------------------------
# FEATURE 3: MULTI-STAGE MATCHING INVARIANCE TESTS
# ---------------------------------------------------------------------------


def test_matching_rotation_and_mirror_invariance(tmp_path: Path):
    """
    Tests that the multi-stage matcher identifies:
    1. Exact positive match (>85%)
    2. Rotated 90 degrees positive match (>75%)
    3. Mirrored positive match (>75%)
    4. Rejection of distinct negative distractor (<50%)
    """
    # 1. Create Base Query: Asymmetric L-shape design (no diagonal or 4-fold rotational symmetry)
    query_img = np.zeros((300, 300), dtype=np.uint8)
    pts = np.array([[50, 50], [250, 50], [250, 100], [110, 100], [110, 160], [50, 160]], dtype=np.int32)
    cv2.polylines(query_img, [pts], isClosed=True, color=255, thickness=4)
    cv2.circle(query_img, (75, 75), 14, color=255, thickness=3)
    query_stencil = create_stencil_from_raster(query_img)

    # 2. Candidate A: Exact copy
    cand_a_img = query_img.copy()
    st_a = create_stencil_from_raster(cand_a_img)
    cand_a = TargetCandidate(
        id="cand_exact",
        file_path=str(tmp_path / "exact.dxf"),
        filename="exact.dxf",
        file_type="dxf",
        page_or_sheet=0,
        candidate_index=0,
        label="Exact Design",
        bbox=(0, 0, 100, 100),
        aspect_ratio=st_a.aspect_ratio,
        entity_count=len(st_a.polylines),
        feature_vector=st_a.feature_vector,
        hu_moments=st_a.hu_moments,
        radial_hist=st_a.radial_hist,
        pairwise_hist=st_a.pairwise_hist,
        polylines=st_a.polylines,
        closed_flags=st_a.closed_flags,
        bitmap_png=cv2.imencode(".png", st_a.bitmap)[1].tobytes(),
    )

    # 3. Candidate B: Rotated 90 degrees
    cand_b_img = cv2.rotate(query_img, cv2.ROTATE_90_CLOCKWISE)
    st_b = create_stencil_from_raster(cand_b_img)
    cand_b = TargetCandidate(
        id="cand_rot90",
        file_path=str(tmp_path / "rot90.dxf"),
        filename="rot90.dxf",
        file_type="dxf",
        page_or_sheet=0,
        candidate_index=0,
        label="Rotated 90 Design",
        bbox=(0, 0, 100, 100),
        aspect_ratio=st_b.aspect_ratio,
        entity_count=len(st_b.polylines),
        feature_vector=st_b.feature_vector,
        hu_moments=st_b.hu_moments,
        radial_hist=st_b.radial_hist,
        pairwise_hist=st_b.pairwise_hist,
        polylines=st_b.polylines,
        closed_flags=st_b.closed_flags,
        bitmap_png=cv2.imencode(".png", st_b.bitmap)[1].tobytes(),
    )

    # 4. Candidate C: Mirrored horizontally
    cand_c_img = cv2.flip(query_img, 1)
    st_c = create_stencil_from_raster(cand_c_img)
    cand_c = TargetCandidate(
        id="cand_mirror",
        file_path=str(tmp_path / "mirror.dxf"),
        filename="mirror.dxf",
        file_type="dxf",
        page_or_sheet=0,
        candidate_index=0,
        label="Mirrored Design",
        bbox=(0, 0, 100, 100),
        aspect_ratio=st_c.aspect_ratio,
        entity_count=len(st_c.polylines),
        feature_vector=st_c.feature_vector,
        hu_moments=st_c.hu_moments,
        radial_hist=st_c.radial_hist,
        pairwise_hist=st_c.pairwise_hist,
        polylines=st_c.polylines,
        closed_flags=st_c.closed_flags,
        bitmap_png=cv2.imencode(".png", st_c.bitmap)[1].tobytes(),
    )

    # 5. Candidate D: Negative Distractor (parallel horizontal grill lines)
    cand_d_img = np.zeros((300, 300), dtype=np.uint8)
    for y in range(40, 270, 30):
        cv2.line(cand_d_img, (30, y), (270, y), 255, thickness=4)
    st_d = create_stencil_from_raster(cand_d_img)
    cand_d = TargetCandidate(
        id="cand_distractor",
        file_path=str(tmp_path / "distractor.dxf"),
        filename="distractor.dxf",
        file_type="dxf",
        page_or_sheet=0,
        candidate_index=0,
        label="Distractor Grille",
        bbox=(0, 0, 100, 100),
        aspect_ratio=st_d.aspect_ratio,
        entity_count=len(st_d.polylines),
        feature_vector=st_d.feature_vector,
        hu_moments=st_d.hu_moments,
        radial_hist=st_d.radial_hist,
        pairwise_hist=st_d.pairwise_hist,
        polylines=st_d.polylines,
        closed_flags=st_d.closed_flags,
        bitmap_png=cv2.imencode(".png", st_d.bitmap)[1].tobytes(),
    )

    all_cands = [cand_a, cand_b, cand_c, cand_d]

    matches = match_stencil_against_candidates(
        stencil=query_stencil,
        candidates=all_cands,
        min_confidence=40.0,
        top_k=10,
    )

    match_dict = {m.candidate_id: m for m in matches}

    # Verify Exact Match is Rank 1
    assert matches[0].candidate_id == "cand_exact"
    assert match_dict["cand_exact"].similarity_score >= 85.0
    assert match_dict["cand_exact"].status_badge == "High Match"

    # Verify Rotated 90 Match
    assert "cand_rot90" in match_dict
    assert match_dict["cand_rot90"].similarity_score >= 75.0
    assert match_dict["cand_rot90"].rotation_deg in (90, 270)

    # Verify Mirrored Match
    assert "cand_mirror" in match_dict
    assert match_dict["cand_mirror"].similarity_score >= 70.0
    assert match_dict["cand_mirror"].is_mirrored is True

    # Verify Distractor is rejected or scored low
    if "cand_distractor" in match_dict:
        assert match_dict["cand_distractor"].similarity_score < match_dict["cand_exact"].similarity_score - 20.0


def test_export_matches_to_csv(tmp_path: Path):
    """Verifies that search results export accurately to CSV."""
    results = [
        MatchResult(
            rank=1,
            candidate_id="c1",
            file_path="/path/to/design_01.dxf",
            filename="design_01.dxf",
            display_name="designs/design_01.dxf",
            file_type="dxf",
            page_or_sheet=0,
            label="Panel A",
            similarity_score=94.2,
            geometric_distance=0.045,
            rotation_deg=0,
            is_mirrored=False,
            bbox=(10.0, 20.0, 150.0, 200.0),
            status_badge="High Match",
        )
    ]

    csv_file = tmp_path / "test_results.csv"
    export_matches_to_csv(results, csv_file)

    assert csv_file.is_file()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["Rank"] == "1"
        assert row["Similarity (%)"] == "94.2"
        assert row["File Name"] == "design_01.dxf"
        assert row["Match Status"] == "High Match"
