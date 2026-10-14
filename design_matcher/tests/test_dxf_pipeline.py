"""
Comprehensive unit and integration tests for DXF CAD integration.
Tests 1 to 8:
1. Border rectangle + circle cut-out + island circle + TEXT label.
2. Closed shape drawn as 40 separate LINE segments with tiny gaps (< tolerance).
3. Block INSERT (rotated and scaled) containing a cut-out.
4. Drawing with 3 designs side by side, labels, and left-to-right order.
5. Y-orientation: asymmetrical "F"-like shape not mirrored.
6. End-to-end: 20 random DXF designs, augmented queries (top-1 >= 90%, top-5 >= 98%), dedup linking.
7. Incremental indexing for DXF (unchanged skipped, modified re-indexed, deleted removed).
8. .dwg file skipped with index_error, corrupt DXF handled gracefully without crashing scan.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import cv2
import ezdxf
import numpy as np
from PIL import Image
import pytest

from engine.config import EngineConfig
from engine.dxf_reader import DxfReaderError, read_dxf
from engine.dxf_segment import segment_dxf_drawing
from engine.features import canonical_mask, get_default_embedder
from engine.index_store import IndexStore
from engine.indexer import find_catalog_files, index_folders
from engine.linework_to_mask import linework_to_mask
from engine.matcher import Matcher, open_source, render_dxf_preview, search_by_code


# ---------------------------------------------------------------------------
# Test 1: Border + Circle Hole + Island + Label
# ---------------------------------------------------------------------------

def test_1_border_cutout_island_and_label(tmp_path: Path):
    """
    Test 1: Border rectangle + circle cut-out + smaller circle island + TEXT label.
    Assert panel corner = 255, inside circle = 0, inside island = 255,
    outside border = 0, and label detected.
    """
    dxf_file = tmp_path / "test1_panel.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Outer border: 200 x 400 mm
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 400), (0, 400)], close=True)

    # Cutout circle: center (100, 200), radius 60
    msp.add_circle((100, 200), radius=60)

    # Island circle inside cutout: center (100, 200), radius 20
    msp.add_circle((100, 200), radius=20)

    # Text label placed below panel
    msp.add_text("PD 010", dxfattribs={"insert": (100, -30), "height": 15})

    doc.saveas(str(dxf_file))

    # Parse and segment
    dxf_data = read_dxf(dxf_file)
    designs = segment_dxf_drawing(dxf_data)

    assert len(designs) == 1
    d = designs[0]
    assert d.label == "PD 010"

    mask = d.mask
    h, w = mask.shape[:2]

    # Map drawing coordinates (x, y) to image coordinates:
    # Scale and margin from mask
    min_x, min_y, max_x, max_y = d.drawing_bounds
    scale = (max(h, w) - 32) / max(max_x - min_x, max_y - min_y)

    def to_img(x: float, y: float) -> Tuple[int, int]:
        ix = int(round(16 + (x - min_x) * scale))
        iy = int(round(16 + (max_y - y) * scale))
        return max(0, min(w - 1, ix)), max(0, min(h - 1, iy))

    # Corner inside panel: (20, 20)
    cx, cy = to_img(20, 20)
    assert mask[cy, cx] == 255, "Panel corner should be metal (255)"

    # Inside circle hole (between r=20 and r=60): (100, 240)
    hx, hy = to_img(100, 240)
    assert mask[hy, hx] == 0, "Inside circle cutout should be void (0)"

    # Inside island: (100, 200)
    ix, iy = to_img(100, 200)
    assert mask[iy, ix] == 255, "Island center should be metal (255)"

    # Outside panel border: (0, 0) in mask canvas (margin is 16px)
    assert mask[2, 2] == 0, "Outside panel border should be background (0)"


# ---------------------------------------------------------------------------
# Test 2: 40 Separate LINE Segments with tiny gaps (< tolerance)
# ---------------------------------------------------------------------------

def test_2_segmented_polyline_gap_tolerance(tmp_path: Path):
    """
    Test 2: Closed shape drawn as 40 separate LINE segments with tiny gaps (< tolerance).
    Assert it is snapped and closed into a hole.
    """
    dxf_file = tmp_path / "test2_segments.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Outer border: 200 x 200 mm
    msp.add_lwpolyline([(0, 0), (200, 0), (200, 200), (0, 200)], close=True)

    # Circular hole of radius 50 at center (100, 100) drawn as 40 separate lines with small gaps
    N = 40
    r = 50.0
    cx, cy = 100.0, 100.0
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    for i in range(N):
        a1 = angles[i]
        a2 = angles[(i + 1) % N]
        p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
        # Add tiny gap at endpoint (0.02 mm, well below 0.1% tolerance of ~0.28 mm)
        p2 = (cx + r * math.cos(a2) + 0.015, cy + r * math.sin(a2) - 0.015)
        msp.add_line(p1, p2)

    doc.saveas(str(dxf_file))

    dxf_data = read_dxf(dxf_file)
    designs = segment_dxf_drawing(dxf_data)
    assert len(designs) == 1

    mask = designs[0].mask
    h, w = mask.shape[:2]

    # Center of hole (100, 100) should be void (0)
    center_val = mask[h // 2, w // 2]
    assert center_val == 0, "Center of 40-segment circle must be cut-out hole (0)"

    # Corner of panel should be metal (255)
    corner_val = mask[h // 10, w // 10]
    assert corner_val == 255, "Panel corner must be metal (255)"


# ---------------------------------------------------------------------------
# Test 3: Block INSERT (Rotated and Scaled) with Cutout
# ---------------------------------------------------------------------------

def test_3_block_insert_rotated_scaled(tmp_path: Path):
    """
    Test 3: Block INSERT (rotated and scaled) containing a cut-out.
    Assert virtual entities are recursively expanded with transform and included.
    """
    dxf_file = tmp_path / "test3_block.dxf"
    doc = ezdxf.new("R2010")

    # Define a block containing a cut-out diamond
    blk = doc.blocks.new(name="DIAMOND_CUT")
    blk.add_lwpolyline([(-15, 0), (0, 15), (15, 0), (0, -15)], close=True)

    msp = doc.modelspace()
    # Panel boundary: 300 x 300 mm
    msp.add_lwpolyline([(0, 0), (300, 0), (300, 300), (0, 300)], close=True)

    # Insert block at center (150, 150) rotated 45 degrees and scaled 2.0x
    msp.add_blockref("DIAMOND_CUT", insert=(150, 150), dxfattribs={"rotation": 45.0, "xscale": 2.0, "yscale": 2.0})

    doc.saveas(str(dxf_file))

    dxf_data = read_dxf(dxf_file)
    assert dxf_data.diagnostics["inserts"] >= 1

    designs = segment_dxf_drawing(dxf_data)
    assert len(designs) == 1

    mask = designs[0].mask
    h, w = mask.shape[:2]

    # Center at (150, 150) must be black (0) cutout
    assert mask[h // 2, w // 2] == 0, "Block insert cut-out must be black hole (0)"
    assert mask[h // 10, w // 10] == 255, "Panel corner must be metal (255)"


# ---------------------------------------------------------------------------
# Test 4: 3 Designs Side-by-Side with Labels
# ---------------------------------------------------------------------------

def test_4_three_designs_side_by_side(tmp_path: Path):
    """
    Test 4: A drawing with 3 designs side by side, each with its own label.
    Assert 3 designs, right labels, and left-to-right order.
    """
    dxf_file = tmp_path / "test4_multi.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Create 3 designs horizontally at x=0..100, x=150..250, x=300..400
    expected_labels = ["PD 001", "PD 002", "PD 003"]
    x_offsets = [0.0, 150.0, 300.0]

    for x_off, lbl in zip(x_offsets, expected_labels):
        # Panel border
        msp.add_lwpolyline([(x_off, 0), (x_off + 100, 0), (x_off + 100, 200), (x_off, 200)], close=True)
        # Unique aperture
        msp.add_circle((x_off + 50, 100), radius=25)
        # Text label below panel
        msp.add_text(lbl, dxfattribs={"insert": (x_off + 30, -25), "height": 12})

    doc.saveas(str(dxf_file))

    dxf_data = read_dxf(dxf_file)
    designs = segment_dxf_drawing(dxf_data)

    assert len(designs) == 3, f"Expected 3 segmented designs, got {len(designs)}"

    # Assert left-to-right order and matching labels
    for i in range(3):
        assert designs[i].label == expected_labels[i]
        assert designs[i].bbox[0] == pytest.approx(x_offsets[i], abs=5.0)

    # Ensure strictly increasing X positions
    assert designs[0].bbox[0] < designs[1].bbox[0] < designs[2].bbox[0]


# ---------------------------------------------------------------------------
# Test 5: Y-Orientation (Asymmetrical "F"-like shape not mirrored)
# ---------------------------------------------------------------------------

def test_5_y_orientation_not_mirrored(tmp_path: Path):
    """
    Test 5: Y-orientation: an asymmetrical shape (letter "F"-like polygon)
    must not be mirrored in the mask.
    In DXF (Y up):
    - Top arm has higher Y than bottom arm.
    In Image (Y down):
    - Top arm must be at LOWER image rows (higher on screen).
    """
    dxf_file = tmp_path / "test5_orientation.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Panel border: 0..100, 0..100
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 100), (0, 100)], close=True)

    # Letter "F" cutout hole:
    # Stem: x=20..35, y=20..80
    # Top arm: x=35..65, y=65..80  (high Y in DXF)
    # Mid arm: x=35..50, y=45..60  (mid Y in DXF)
    f_outline = [
        (20, 20), (35, 20), (35, 45), (55, 45), (55, 60), (35, 60),
        (35, 65), (75, 65), (75, 80), (20, 80)
    ]
    msp.add_lwpolyline(f_outline, close=True)

    doc.saveas(str(dxf_file))

    dxf_data = read_dxf(dxf_file)
    designs = segment_dxf_drawing(dxf_data)
    assert len(designs) == 1

    mask = designs[0].mask
    h, w = mask.shape[:2]

    # The top arm (x=70, y=72 in DXF) should be in upper half of image (row < h/2)
    # The bottom of stem (x=25, y=25 in DXF) should be in lower half of image (row > h/2)
    # Check that cutouts (value 0) are located in upper half for the long arm
    cutout_pts = np.argwhere(mask == 0)
    assert len(cutout_pts) > 0

    # Top arm tip: in DXF, x=70, y=72.
    # In true image, it should be at right columns (x > w/2) and upper rows (y < h/2).
    top_arm_cutouts = np.argwhere((mask == 0) & (np.arange(w) > w * 0.6))
    assert len(top_arm_cutouts) > 0

    # All cutouts in rightmost section (top arm) must be in top half of image (row < h * 0.5)
    mean_top_arm_row = float(top_arm_cutouts[:, 0].mean())
    assert mean_top_arm_row < h * 0.45, "Top arm must appear in top rows (non-mirrored Y orientation)"


# ---------------------------------------------------------------------------
# Test 6: End-to-End Search (20 Designs, Augmented Query, Dedup Linking)
# ---------------------------------------------------------------------------

def test_6_end_to_end_search_and_dedup(tmp_path: Path):
    """
    Test 6: End-to-end:
    - Index folder with 20 distinct DXF designs.
    - Also create 1 PDF catalog page sharing the design with DXF #0.
    - Assert Top-1 >= 90% and Top-5 >= 98% under rotation/blur/warp query.
    - Assert that the PDF and DXF of the same design are linked as same_design_group.
    """
    cat_dir = tmp_path / "catalog_20"
    cat_dir.mkdir()

    # Generate 20 distinct DXF designs
    for i in range(20):
        dxf_p = cat_dir / f"design_{i:02d}.dxf"
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        # Panel border 200x200
        msp.add_lwpolyline([(0, 0), (200, 0), (200, 200), (0, 200)], close=True)

        if i == 0:
            for r in range(35, 175, 35):
                for c in range(35, 175, 35):
                    msp.add_circle((c, r), radius=10)
        elif i == 1:
            for rad in range(20, 85, 20):
                msp.add_circle((100, 100), radius=rad)
        elif i == 2:
            for r in range(35, 175, 45):
                for c in range(35, 175, 45):
                    msp.add_lwpolyline([(c, r - 15), (c + 15, r), (c, r + 15), (c - 15, r)], close=True)
        elif i == 3:
            for y in range(30, 180, 30):
                shift = 15 if (y // 30) % 2 == 1 else 0
                for x in range(30 + shift, 180, 30):
                    msp.add_circle((x, y), radius=9)
        elif i == 4:
            for d in range(-100, 250, 25):
                msp.add_line((d, 25), (d + 150, 175))
        elif i == 5:
            for d in range(-100, 250, 25):
                msp.add_line((d, 175), (d + 150, 25))
        elif i == 6:
            for y in range(40, 170, 35):
                for x in range(40, 170, 35):
                    msp.add_line((x - 10, y), (x + 10, y))
                    msp.add_line((x, y - 10), (x, y + 10))
        elif i == 7:
            for y in range(40, 170, 30):
                pts = [(x, float(y + 8 * np.sin(x / 12.0))) for x in range(25, 176, 5)]
                msp.add_lwpolyline(pts, close=False)
        elif i == 8:
            for x in range(35, 175, 25):
                msp.add_line((x, 25), (x, 175))
        elif i == 9:
            for y in range(35, 175, 25):
                msp.add_line((25, y), (175, y))
        elif i == 10:
            for ang in range(0, 360, 20):
                rad = np.radians(ang)
                msp.add_line((100, 100), (100 + 65 * np.cos(rad), 100 + 65 * np.sin(rad)))
        elif i == 11:
            for r in range(30, 170, 35):
                for c in range(30, 170, 35):
                    if ((r // 35) + (c // 35)) % 2 == 0:
                        msp.add_lwpolyline([(c, r), (c + 20, r), (c + 20, r + 20), (c, r + 20)], close=True)
        elif i == 12:
            for y in range(35, 175, 30):
                msp.add_lwpolyline([(30, y), (100, y + 15), (170, y)], close=False)
        elif i == 13:
            for y in range(35, 175, 35):
                for x in range(35, 175, 35):
                    r_sz = 6 + (x % 8)
                    msp.add_circle((x, y), radius=r_sz)
        elif i == 14:
            for y in range(40, 170, 35):
                for x in range(40, 170, 35):
                    msp.add_lwpolyline([(x, y - 10), (x + 10, y + 10), (x - 10, y + 10)], close=True)
        elif i == 15:
            for y in range(35, 175, 30):
                msp.add_line((30, y), (170, y))
            for x in range(35, 175, 30):
                msp.add_line((x, 30), (x, 170))
        elif i == 16:
            for s in range(20, 80, 20):
                msp.add_lwpolyline([(100 - s, 100 - s), (100 + s, 100 - s),
                                    (100 + s, 100 + s), (100 - s, 100 + s)], close=True)
        elif i == 17:
            for y in range(40, 170, 35):
                for x in range(40, 170, 35):
                    msp.add_ellipse((x, y), major_axis=(12, 0), ratio=0.5)
        elif i == 18:
            for y in range(40, 170, 35):
                for x in range(40, 170, 35):
                    msp.add_line((x - 10, y - 10), (x + 10, y + 10))
                    msp.add_line((x - 10, y + 10), (x + 10, y - 10))
        else:
            rng = np.random.RandomState(42 + i)
            for _ in range(25):
                rx = float(rng.randint(30, 170))
                ry = float(rng.randint(30, 170))
                msp.add_circle((rx, ry), radius=8)

        msp.add_text(f"CODE {i:03d}", dxfattribs={"insert": (80, -25), "height": 14})
        doc.saveas(str(dxf_p))

    # Create 1 PDF page with the identical design as design_00.dxf
    d0_data = read_dxf(cat_dir / "design_00.dxf")
    d0_design = segment_dxf_drawing(d0_data)[0]
    d0_mask = d0_design.mask

    pdf_p = cat_dir / "shared_catalog.pdf"
    ph, pw = d0_mask.shape[:2]
    page_bgr = np.full((ph + 100, pw + 100, 3), (25, 20, 15), dtype=np.uint8)
    page_bgr[50 : 50 + ph, 50 : 50 + pw][d0_mask == 255] = (218, 218, 222)
    rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(str(pdf_p), "PDF", resolution=150.0)

    cfg = EngineConfig(
        DB_PATH=tmp_path / "test6.db",
        INDEX_PATH=tmp_path / "test6.index",
        MIN_CONFIDENCE=15.0,
    )
    store = IndexStore(db_path=cfg.DB_PATH, index_path=cfg.INDEX_PATH)
    embedder = get_default_embedder("classical", cfg)

    stats = index_folders([cat_dir], store=store, embedder=embedder, cfg=cfg)
    assert stats.indexed_files >= 21
    assert stats.indexed_designs >= 21

    matcher = Matcher(store=store, embedder=embedder, cfg=cfg)

    # Verify Dedup: DXF design_00 and PDF shared_catalog should be linked
    dxf_0_record = store.conn.cursor().execute(
        "SELECT id, same_design_group FROM designs WHERE source_type='dxf' AND label='CODE 000'"
    ).fetchone()
    assert dxf_0_record is not None
    group_id = dxf_0_record["same_design_group"]
    assert group_id is not None, "Identical design in DXF and PDF must be linked under same_design_group"

    # Query with augmented versions of all 20 DXF designs across rotation, mirror, blur, warp
    top1_correct = 0
    top5_correct = 0

    for idx in range(20):
        target_name = f"design_{idx:02d}.dxf"
        preview = render_dxf_preview(cat_dir / target_name, target_size=400)

        # Diverse perturbations (rotated, mirrored, blurred, noisy, lit, perspective-warped)
        mode = idx % 6
        if mode == 0:
            aug = cv2.rotate(preview, cv2.ROTATE_90_CLOCKWISE)
        elif mode == 1:
            aug = cv2.flip(preview, 1)  # Horizontal mirror
        elif mode == 2:
            aug = cv2.GaussianBlur(preview, (5, 5), 1.2)
        elif mode == 3:
            noise = np.random.normal(0, 10, preview.shape).astype(np.int16)
            aug = np.clip(preview.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif mode == 4:
            h, w = preview.shape[:2]
            grad = np.linspace(0.8, 1.2, w).reshape(1, w, 1)
            aug = np.clip(preview.astype(float) * grad, 0, 255).astype(np.uint8)
        else:
            # Perspective warped (skewed 4-corners)
            h, w = preview.shape[:2]
            src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst_pts = np.float32([[10, 8], [w - 8, 12], [w - 12, h - 10], [6, h - 8]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            aug = cv2.warpPerspective(preview, M, (w, h), borderValue=(25, 20, 15))

        resp = matcher.search(aug, top_k=5, already_cropped=True)
        assert not resp.no_confident_match

        candidate_names = [Path(r.pdf_path).name for r in resp.results]
        top_cand = resp.results[0]
        # Candidate is correct if it matches target or is linked via same_design_group
        is_top1 = (
            Path(top_cand.pdf_path).name == target_name
            or any(Path(rel.pdf_path).name == target_name for rel in top_cand.related)
        )
        if is_top1:
            top1_correct += 1
        else:
            print(f"DEBUG: Target {target_name} missed top-1! Top cand: {Path(top_cand.pdf_path).name}, score: {top_cand.score:.2f}, mode: {mode}")

        is_top5 = (
            target_name in candidate_names[:5]
            or any(
                any(Path(rel.pdf_path).name == target_name for rel in r.related)
                for r in resp.results[:5]
            )
        )
        if is_top5:
            top5_correct += 1
        else:
            print(f"DEBUG: Target {target_name} MISSED TOP-5! Candidate names: {candidate_names}")

    top1_acc = float(top1_correct) / 20.0
    top5_acc = float(top5_correct) / 20.0
    assert top1_acc >= 0.90, f"Top-1 accuracy was {top1_acc * 100:.1f}%"
    assert top5_acc >= 0.98, f"Top-5 accuracy was {top5_acc * 100:.1f}%"

    # Verify search_by_code
    code_results = matcher.search_by_code("code000")
    assert len(code_results) >= 1
    assert code_results[0].label == "CODE 000"
    # Related linked results should contain the PDF
    assert len(code_results[0].related) >= 1


# ---------------------------------------------------------------------------
# Test 7: Incremental Indexing for DXF
# ---------------------------------------------------------------------------

def test_7_incremental_indexing_dxf(tmp_path: Path):
    """
    Test 7: Incremental indexing for DXF:
    - Unchanged files skipped.
    - Modified files re-indexed.
    - Deleted files removed from SQLite and FAISS.
    """
    cat_dir = tmp_path / "incremental_cat"
    cat_dir.mkdir()

    dxf_a = cat_dir / "panel_a.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 100), (0, 100)], close=True)
    msp.add_circle((50, 50), radius=15)
    doc.saveas(str(dxf_a))

    cfg = EngineConfig(
        DB_PATH=tmp_path / "inc.db",
        INDEX_PATH=tmp_path / "inc.index",
    )
    store = IndexStore(db_path=cfg.DB_PATH, index_path=cfg.INDEX_PATH)
    embedder = get_default_embedder("classical", cfg)

    # 1. Initial index
    s1 = index_folders([cat_dir], store=store, embedder=embedder, cfg=cfg)
    assert s1.indexed_files == 1
    assert s1.skipped_files == 0
    assert store.index.ntotal == 1

    # 2. Second index (unchanged)
    s2 = index_folders([cat_dir], store=store, embedder=embedder, cfg=cfg)
    assert s2.skipped_files == 1
    assert s2.indexed_files == 0
    assert store.index.ntotal == 1

    # 3. Modify DXF
    doc2 = ezdxf.new("R2010")
    msp2 = doc2.modelspace()
    msp2.add_lwpolyline([(0, 0), (150, 0), (150, 150), (0, 150)], close=True)
    msp2.add_circle((75, 75), radius=25)
    doc2.saveas(str(dxf_a))

    s3 = index_folders([cat_dir], store=store, embedder=embedder, cfg=cfg)
    assert s3.indexed_files == 1
    assert s3.skipped_files == 0
    assert store.index.ntotal == 1

    # 4. Delete DXF
    dxf_a.unlink()
    s4 = index_folders([cat_dir], store=store, embedder=embedder, cfg=cfg)
    assert s4.deleted_files == 1
    assert store.index.ntotal == 0


# ---------------------------------------------------------------------------
# Test 8: DWG Skipped & Corrupt DXF Scan Resilience
# ---------------------------------------------------------------------------

def test_8_dwg_skip_and_corrupt_dxf_resilience(tmp_path: Path):
    """
    Test 8:
    - .dwg file is skipped with index_error message 'DWG not supported: please export to DXF'.
    - Corrupt DXF does not crash the scan, records error, and valid DXF is still indexed.
    """
    cat_dir = tmp_path / "resilience_cat"
    cat_dir.mkdir()

    # Valid DXF
    valid_dxf = cat_dir / "valid.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 100), (0, 100)], close=True)
    doc.saveas(str(valid_dxf))

    # Unsupported DWG dummy file
    dwg_file = cat_dir / "unsupported.dwg"
    dwg_file.write_bytes(b"AC1032 dummy binary dwg header")

    # Corrupt DXF file
    corrupt_dxf = cat_dir / "broken.dxf"
    corrupt_dxf.write_text("NOT A VALID DXF FILE GARBAGE CONTENT 12345")

    cfg = EngineConfig(
        DB_PATH=tmp_path / "resilience.db",
        INDEX_PATH=tmp_path / "resilience.index",
    )
    store = IndexStore(db_path=cfg.DB_PATH, index_path=cfg.INDEX_PATH)
    embedder = get_default_embedder("classical", cfg)

    stats = index_folders([cat_dir], store=store, embedder=embedder, cfg=cfg)

    # Valid DXF must be indexed
    assert stats.indexed_files == 1
    assert store.index.ntotal >= 1

    # Check index_errors table
    cur = store.conn.cursor()
    cur.execute("SELECT pdf_path, error_message FROM index_errors")
    errors = cur.fetchall()

    dwg_err = [e for e in errors if "unsupported.dwg" in e["pdf_path"]]
    assert len(dwg_err) >= 1
    assert "DWG not supported: please export to DXF" in dwg_err[0]["error_message"]

    # Corrupt DXF error recorded without crashing
    broken_err = [e for e in errors if "broken.dxf" in e["pdf_path"]]
    assert len(broken_err) >= 1
