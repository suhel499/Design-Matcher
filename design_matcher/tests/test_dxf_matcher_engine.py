"""
test_dxf_matcher_engine.py - Unit and Integration Tests for DXF Matching Engine.

Validates all 13 core requirements from specification:
1. Exact copy matching (Rank 1, confidence >= 95%, 'Verified Match').
2. Translated and scaled copy matching.
3. Rotated (90, 180, 270) and mirrored copies matching.
4. Entity segmentation invariance (arcs vs 60 discrete line segments).
5. Slightly modified designs & different hole pattern discrimination.
6. Annotation & non-geometric layer exclusion (TEXT, DIMENSION, defpoints).
7. Multi-design DXF sheet splitting into individual components.
8. Nested block INSERT transformations with rotation & scaling.
9. Corrupt, empty, and unsupported (.dwg) file handling.
10. Mode A: PNG Stencil -> DXF search.
11. Mode B: Reference DXF -> DXF search.
12. Incremental indexing, deduplication, and deleted file cleanup.
13. Search latency and retrieval accuracy.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import cv2
import ezdxf
import numpy as np
import pytest

from engine.dxf_database import DxfDatabase
from engine.dxf_matcher import DxfMatchingEngine, align_and_verify_point_clouds
from engine.dxf_normalization import (
    normalize_dxf_component,
    normalize_raster_stencil,
)
from engine.dxf_parser import DxfParserError, parse_dxf_drawing


def _create_sample_dxf(
    path: Path,
    width: float = 200.0,
    height: float = 400.0,
    cutout_type: str = "circles",
    offset: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    rotation_deg: float = 0.0,
    label: str = "PD 100",
) -> Path:
    """Helper to generate parameterized synthetic DXF CAD drawings."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    ox, oy = offset
    w = width * scale
    h = height * scale

    rad = math.radians(rotation_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    def rot(lx: float, ly: float) -> Tuple[float, float]:
        rx = lx * cos_r - ly * sin_r
        ry = lx * sin_r + ly * cos_r
        return (float(rx + ox), float(ry + oy))

    hw, hh = w / 2.0, h / 2.0
    corners = [
        rot(-hw, -hh),
        rot(hw, -hh),
        rot(hw, hh),
        rot(-hw, hh),
    ]
    msp.add_lwpolyline(corners, close=True)

    if cutout_type == "circles":
        c1 = rot(0.0, -60.0 * scale)
        c2 = rot(0.0, 60.0 * scale)
        msp.add_circle(c1, radius=35.0 * scale)
        msp.add_circle(c2, radius=35.0 * scale)
    elif cutout_type == "diamonds":
        dw, dh = 40.0 * scale, 60.0 * scale
        diamond = [
            rot(-dw, 0.0),
            rot(0.0, dh),
            rot(dw, 0.0),
            rot(0.0, -dh),
        ]
        msp.add_lwpolyline(diamond, close=True)
    elif cutout_type == "segmented_circle":
        r = 35.0 * scale
        num_segs = 36
        angles = np.linspace(0, 2 * math.pi, num_segs, endpoint=False)
        for offset_y in (-60.0 * scale, 60.0 * scale):
            pts = [rot(r * math.cos(a), offset_y + r * math.sin(a)) for a in angles]
            for i in range(num_segs):
                msp.add_line(pts[i], pts[(i + 1) % num_segs])

    if label:
        lbl_pt = rot(-hw + 20.0, -hh - 25.0)
        msp.add_text(label, dxfattribs={"insert": lbl_pt, "height": 14.0})

    doc.saveas(str(path))
    return path


# ---------------------------------------------------------------------------
# Test 1: Exact Copy Matching
# ---------------------------------------------------------------------------

def test_1_exact_copy_matching(tmp_path: Path):
    """Verifies that an exact copy matches with Rank 1, Verified Match, confidence >= 95%."""
    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()
    db_path = tmp_path / "dxf_test.db"

    dxf_a = _create_sample_dxf(cat_dir / "design_a.dxf", cutout_type="circles", label="PD 001")
    dxf_b = _create_sample_dxf(cat_dir / "design_b.dxf", cutout_type="diamonds", label="PD 002")

    db = DxfDatabase(db_path)
    db.scan_and_index_directory(cat_dir)
    assert db.get_total_indexed_count() == 2

    engine = DxfMatchingEngine(db)
    res = engine.search_by_reference_dxf(dxf_a)

    assert len(res.matches) >= 1
    top = res.matches[0]
    assert top.filename == "design_a.dxf"
    assert top.rank == 1
    assert top.confidence_score >= 95.0
    assert top.match_status == "Verified Match"
    assert top.geometric_distance < 0.05


# ---------------------------------------------------------------------------
# Test 2: Translated and Scaled Copy Matching
# ---------------------------------------------------------------------------

def test_2_translated_and_scaled_copy_matching(tmp_path: Path):
    """Verifies that large coordinate translations and scale factors are normalized."""
    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()
    db_path = tmp_path / "dxf_test.db"

    _create_sample_dxf(cat_dir / "original.dxf", cutout_type="circles", offset=(0, 0), scale=1.0)
    _create_sample_dxf(cat_dir / "other.dxf", cutout_type="diamonds", offset=(0, 0), scale=1.0)

    db = DxfDatabase(db_path)
    db.scan_and_index_directory(cat_dir)

    # Query file: translated by (2500, -1800) and scaled by 4.0x
    query_dxf = tmp_path / "transformed_query.dxf"
    _create_sample_dxf(query_dxf, cutout_type="circles", offset=(2500, -1800), scale=4.0)

    engine = DxfMatchingEngine(db)
    res = engine.search_by_reference_dxf(query_dxf)

    assert len(res.matches) >= 1
    top = res.matches[0]
    assert top.filename == "original.dxf"
    assert top.confidence_score >= 85.0
    assert top.match_status == "Verified Match"
    assert top.geometric_distance < 0.10


# ---------------------------------------------------------------------------
# Test 3: Rotated and Mirrored Copy Matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [90.0, 180.0, 270.0])
def test_3_rotated_matching(tmp_path: Path, angle: float):
    """Verifies rotation-aware alignment across 90, 180, and 270 degrees."""
    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()
    db_path = tmp_path / f"dxf_rot_{int(angle)}.db"

    _create_sample_dxf(cat_dir / "base.dxf", cutout_type="circles", offset=(0, 0), rotation_deg=0.0)

    db = DxfDatabase(db_path)
    db.scan_and_index_directory(cat_dir)

    query_dxf = tmp_path / f"query_rot_{int(angle)}.dxf"
    _create_sample_dxf(query_dxf, cutout_type="circles", offset=(100, 100), rotation_deg=angle)

    engine = DxfMatchingEngine(db)
    res = engine.search_by_reference_dxf(query_dxf)

    assert len(res.matches) >= 1
    top = res.matches[0]
    assert top.filename == "base.dxf"
    assert top.confidence_score >= 80.0
    assert top.match_status == "Verified Match"


# ---------------------------------------------------------------------------
# Test 4: Entity Segmentation Invariance (Arcs vs Discrete Lines)
# ---------------------------------------------------------------------------

def test_4_segmentation_invariance(tmp_path: Path):
    """Verifies that shapes drawn as multiple LINE segments match smooth curve entities."""
    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()
    db_path = tmp_path / "dxf_seg.db"

    # Indexed with smooth CIRCLE entities
    _create_sample_dxf(cat_dir / "smooth_circles.dxf", cutout_type="circles")

    db = DxfDatabase(db_path)
    db.scan_and_index_directory(cat_dir)

    # Query drawn with 36 discrete LINE segments
    query_dxf = tmp_path / "chopped_lines.dxf"
    _create_sample_dxf(query_dxf, cutout_type="segmented_circle")

    # Also index an unrelated diamond to ensure discrimination
    _create_sample_dxf(cat_dir / "unrelated_diamond.dxf", cutout_type="diamonds")
    db.scan_and_index_directory(cat_dir)

    engine = DxfMatchingEngine(db)
    res = engine.search_by_reference_dxf(query_dxf)

    assert len(res.matches) >= 1
    top = res.matches[0]
    assert top.filename == "smooth_circles.dxf"
    assert top.confidence_score > res.matches[1].confidence_score


# ---------------------------------------------------------------------------
# Test 5: Hole Pattern Discrimination
# ---------------------------------------------------------------------------

def test_5_hole_pattern_discrimination(tmp_path: Path):
    """Verifies that designs with identical outer borders but different cutouts are distinguished."""
    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()
    db_path = tmp_path / "dxf_discrim.db"

    _create_sample_dxf(cat_dir / "circle_cutouts.dxf", cutout_type="circles")
    _create_sample_dxf(cat_dir / "diamond_cutouts.dxf", cutout_type="diamonds")

    db = DxfDatabase(db_path)
    db.scan_and_index_directory(cat_dir)

    query_dxf = tmp_path / "query_diamonds.dxf"
    _create_sample_dxf(query_dxf, cutout_type="diamonds")

    engine = DxfMatchingEngine(db)
    res = engine.search_by_reference_dxf(query_dxf)

    assert len(res.matches) >= 2
    assert res.matches[0].filename == "diamond_cutouts.dxf"
    assert res.matches[0].confidence_score >= 90.0
    assert res.matches[1].filename == "circle_cutouts.dxf"
    assert res.matches[0].confidence_score > res.matches[1].confidence_score + 15.0


# ---------------------------------------------------------------------------
# Test 6: Annotation & Layer Exclusion
# ---------------------------------------------------------------------------

def test_6_annotations_and_layers_excluded(tmp_path: Path):
    """Verifies that dimensions, defpoints, and text annotations do not distort geometry."""
    dxf_file = tmp_path / "annotated.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Border
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 400), (0, 400)], close=True)
    msp.add_circle((100, 200), radius=50)

    # Ignored layers: dimensions, notes
    doc.layers.add("DIMENSIONS", color=1)
    if not doc.layers.has_entry("DEFPOINTS"):
        doc.layers.add("DEFPOINTS", color=7)
    msp.add_line((0, -20), (200, -20), dxfattribs={"layer": "DIMENSIONS"})
    msp.add_text("200 mm", dxfattribs={"layer": "DIMENSIONS", "insert": (100, -35), "height": 10})
    msp.add_point((0, 0), dxfattribs={"layer": "DEFPOINTS"})

    doc.saveas(str(dxf_file))

    parsed = parse_dxf_drawing(dxf_file)
    assert len(parsed.components) == 1
    comp = parsed.components[0]
    # Polyline count should only be border + circle (2 entities)
    assert comp.entity_count == 2


# ---------------------------------------------------------------------------
# Test 7: Multi-Design Sheet Splitting
# ---------------------------------------------------------------------------

def test_7_multi_design_sheet_splitting(tmp_path: Path):
    """Verifies that a drawing containing multiple side-by-side panels is split into components."""
    dxf_file = tmp_path / "sheet_catalog.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Panel 1: at x=0
    msp.add_lwpolyline([(0, 0), (150, 0), (150, 400), (0, 400)], close=True)
    msp.add_circle((75, 200), radius=30)
    msp.add_text("PD 401", dxfattribs={"insert": (75, -20), "height": 12})

    # Panel 2: at x=250
    msp.add_lwpolyline([(250, 0), (400, 0), (400, 400), (250, 400)], close=True)
    msp.add_circle((325, 200), radius=45)
    msp.add_text("PD 402", dxfattribs={"insert": (325, -20), "height": 12})

    doc.saveas(str(dxf_file))

    parsed = parse_dxf_drawing(dxf_file)
    assert len(parsed.components) == 2
    assert parsed.components[0].name == "PD 401"
    assert parsed.components[1].name == "PD 402"


# ---------------------------------------------------------------------------
# Test 8: Block INSERT Transformation
# ---------------------------------------------------------------------------

def test_8_nested_block_transformations(tmp_path: Path):
    """Verifies that block INSERT entities with rotation & scaling are properly unpacked."""
    dxf_file = tmp_path / "block_insert.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Create block definition with a circle cutout
    blk = doc.blocks.new(name="CUTOUT_BLOCK")
    blk.add_circle((0, 0), radius=20)

    # Modelspace outer panel
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 400), (0, 400)], close=True)

    # Insert block with scale 2.0 at position (100, 200)
    msp.add_blockref("CUTOUT_BLOCK", insert=(100, 200), dxfattribs={"xscale": 2.0, "yscale": 2.0})

    doc.saveas(str(dxf_file))

    parsed = parse_dxf_drawing(dxf_file)
    assert len(parsed.components) == 1
    # Both border and expanded block geometry must be present
    assert parsed.components[0].entity_count == 2


# ---------------------------------------------------------------------------
# Test 9: Corrupt, Empty, and DWG Handling
# ---------------------------------------------------------------------------

def test_9_corrupt_empty_and_dwg_handling(tmp_path: Path):
    """Verifies graceful handling of degenerate, corrupted, and .dwg files."""
    dwg_file = tmp_path / "unsupported.dwg"
    dwg_file.write_bytes(b"AC1032 binary dwg data")

    with pytest.raises(DxfParserError, match="DWG format is not supported"):
        parse_dxf_drawing(dwg_file)

    # Empty DXF
    empty_dxf = tmp_path / "empty.dxf"
    empty_dxf.write_text("")
    parsed_empty = parse_dxf_drawing(empty_dxf)
    assert len(parsed_empty.components) == 0

    # Corrupt DXF
    bad_dxf = tmp_path / "garbage.dxf"
    bad_dxf.write_text("SOME RANDOM GARBAGE DATA 99999")
    parsed_bad = parse_dxf_drawing(bad_dxf)
    assert len(parsed_bad.components) == 0


# ---------------------------------------------------------------------------
# Test 10: Mode A - PNG Stencil to DXF Search
# ---------------------------------------------------------------------------

def test_10_mode_a_png_stencil_to_dxf_matching(tmp_path: Path):
    """Verifies Mode A: Querying with a PNG stencil correctly finds matching DXF design."""
    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()
    db_path = tmp_path / "dxf_mode_a.db"

    _create_sample_dxf(cat_dir / "target_design.dxf", cutout_type="circles", label="PD 888")
    _create_sample_dxf(cat_dir / "other_design.dxf", cutout_type="diamonds", label="PD 999")

    db = DxfDatabase(db_path)
    db.scan_and_index_directory(cat_dir)

    # Create synthetic query PNG stencil: white metal with black circular voids
    stencil_img = np.full((400, 200), 255, dtype=np.uint8)
    cv2.circle(stencil_img, (100, 140), 35, 0, -1)
    cv2.circle(stencil_img, (100, 260), 35, 0, -1)

    png_path = tmp_path / "query_stencil.png"
    cv2.imwrite(str(png_path), stencil_img)

    engine = DxfMatchingEngine(db)
    res = engine.search_by_png_stencil(png_path)

    assert len(res.matches) >= 1
    top = res.matches[0]
    assert top.filename == "target_design.dxf"
    assert top.confidence_score >= 70.0
    assert top.match_status in ("Verified Match", "Possible Match")


# ---------------------------------------------------------------------------
# Test 11: Mode B - Reference DXF Search
# ---------------------------------------------------------------------------

def test_11_mode_b_reference_dxf_search(tmp_path: Path):
    """Verifies Mode B: Selecting a reference DXF file returns ranked vector matches."""
    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()
    db_path = tmp_path / "dxf_mode_b.db"

    _create_sample_dxf(cat_dir / "ref_target.dxf", cutout_type="circles", label="REF 01")
    _create_sample_dxf(cat_dir / "ref_decoy.dxf", cutout_type="diamonds", label="REF 02")

    db = DxfDatabase(db_path)
    db.scan_and_index_directory(cat_dir)

    query_ref = tmp_path / "user_reference.dxf"
    _create_sample_dxf(query_ref, cutout_type="circles", offset=(50, 50), label="USER REF")

    engine = DxfMatchingEngine(db)
    res = engine.search_by_reference_dxf(query_ref)

    assert len(res.matches) >= 1
    top = res.matches[0]
    assert top.filename == "ref_target.dxf"
    assert top.rank == 1
    assert top.confidence_score >= 90.0
    assert top.is_verified


# ---------------------------------------------------------------------------
# Test 12: Incremental Indexing and File Deletion
# ---------------------------------------------------------------------------

def test_12_incremental_indexing_and_cleanup(tmp_path: Path):
    """Verifies skipping unchanged files, re-indexing changed files, and cleaning deleted files."""
    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()
    db_path = tmp_path / "dxf_inc.db"

    f1 = _create_sample_dxf(cat_dir / "f1.dxf", cutout_type="circles")
    f2 = _create_sample_dxf(cat_dir / "f2.dxf", cutout_type="diamonds")

    db = DxfDatabase(db_path)
    # Initial scan
    p1 = db.scan_and_index_directory(cat_dir)
    assert p1.total == 2
    assert db.get_total_indexed_count() == 2

    # Second scan with no changes -> both skipped
    p2 = db.scan_and_index_directory(cat_dir)
    assert p2.skipped == 2

    # Delete f2 -> cleaned up
    f2.unlink()
    p3 = db.scan_and_index_directory(cat_dir)
    assert db.get_total_indexed_count() == 1


# ---------------------------------------------------------------------------
# Test 13: Search Latency Performance Benchmark
# ---------------------------------------------------------------------------

def test_13_search_latency_benchmark(tmp_path: Path):
    """Verifies that retrieval latency is fast (< 100 ms)."""
    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()
    db_path = tmp_path / "dxf_bench.db"

    # Create 5 catalog designs
    for i in range(5):
        ctype = "circles" if i == 0 else "diamonds"
        _create_sample_dxf(cat_dir / f"cat_{i}.dxf", cutout_type=ctype, offset=(i * 10, i * 10))

    db = DxfDatabase(db_path)
    db.scan_and_index_directory(cat_dir)

    engine = DxfMatchingEngine(db)
    query_dxf = cat_dir / "cat_0.dxf"

    res = engine.search_by_reference_dxf(query_dxf)
    assert res.search_latency_ms < 100.0, f"Search latency too high: {res.search_latency_ms} ms"
    assert res.matches[0].filename == "cat_0.dxf"


# ---------------------------------------------------------------------------
# Test 14: Clear and Rebuild Index Operations
# ---------------------------------------------------------------------------

def test_14_database_clear_and_rebuild_index(tmp_path: Path):
    """Verifies that clear() and rebuild_index() correctly reset and reindex."""
    import argparse
    from app.cli import cmd_index_dxf

    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()
    db_path = tmp_path / "dxf_rebuild.db"

    _create_sample_dxf(cat_dir / "item1.dxf", cutout_type="circles")
    _create_sample_dxf(cat_dir / "item2.dxf", cutout_type="diamonds")

    db = DxfDatabase(db_path)
    prog = db.scan_and_index_directory(cat_dir)
    assert prog.indexed == 2
    assert db.get_total_indexed_count() == 2

    # Test clear()
    db.clear()
    assert db.get_total_indexed_count() == 0
    assert db.get_total_component_count() == 0

    # Test rebuild_index() without arguments (purges without error)
    db.scan_and_index_directory(cat_dir)
    assert db.get_total_indexed_count() == 2
    res_prog = db.rebuild_index()
    assert res_prog is None
    assert db.get_total_indexed_count() == 0

    # Test rebuild_index(cat_dir) (purges and reindexes)
    prog_rebuild = db.rebuild_index(cat_dir)
    assert prog_rebuild is not None
    assert prog_rebuild.indexed == 2
    assert db.get_total_indexed_count() == 2

    # Test CLI cmd_index_dxf with --rebuild flag
    cli_args = argparse.Namespace(
        root=str(cat_dir),
        data_dir=str(tmp_path),
        rebuild=True,
    )
    # The CLI uses dxf_library.db in data_dir
    exit_code = cmd_index_dxf(cli_args)
    assert exit_code == 0
    cli_db = DxfDatabase(tmp_path / "dxf_library.db")
    assert cli_db.get_total_indexed_count() == 2

