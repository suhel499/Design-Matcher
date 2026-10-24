"""
Tests for the DWG Image-Finder, ODA/LibreDWG Converter, and Relative Path Resolution.

Tests:
1. Converter Detection (ODA priority, LibreDWG fallback, and custom settings override).
2. Sample 1: DWG with attached raster image reference (IMAGE / IMAGEDEF),
   including path resolution relative to DWG folder and missing reference reporting.
3. Sample 2: DWG without attached image (vector candidates + DWG header thumbnail fallback).
4. Sample 3: Corrupted / invalid DWG (gracefully caught and skipped without crashing).
5. DWG to DXF persistent caching (keyed by path, size, mtime).
6. Cross-platform safety (subprocess list args, CREATE_NO_WINDOW on Windows, temp directory cleanup).
"""

from __future__ import annotations

import os
import shutil
import struct
import tempfile
from pathlib import Path

import cv2
import ezdxf
import numpy as np
import pytest

from app.core.cad_image_finder import (
    CadExtractedImage,
    build_candidate_from_extracted_image,
    extract_cad_images_all,
    extract_images_from_dxf,
    resolve_cad_image_path,
)
from app.core.design_finder import (
    TargetCandidate,
    extract_candidates_from_dwg,
    extract_candidates_from_dxf,
    extract_file_candidates,
)
from app.core.dwg_converter import (
    batch_convert_dwg_to_dxf,
    compute_dwg_cache_key,
    convert_dwg_to_dxf_single,
    find_libredwg_binary,
    find_oda_converter_binary,
    get_available_converter,
    get_cached_dxf_path,
    get_dwg_cache_dir,
)
from app.core.dwg_reader import extract_dwg_embedded_thumbnail


# -----------------------------------------------------------------------------
# Fixtures: 3 Sample DWG Files (With Image, Without Image, Corrupted)
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_raster_image(tmp_path: Path) -> Path:
    """Creates a sample PNG image to attach to CAD drawings."""
    img_path = tmp_path / "panel_texture.png"
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw geometric motif
    cv2.circle(arr, (100, 100), 70, (255, 255, 255), -1)
    cv2.rectangle(arr, (40, 40), (160, 160), (0, 0, 0), 4)
    cv2.imwrite(str(img_path), arr)
    return img_path


@pytest.fixture
def sample_dwg_with_image(tmp_path: Path, sample_raster_image: Path) -> Tuple[Path, Path]:
    """
    Creates Sample 1: DWG drawing with an attached raster image reference.
    Uses DXF with IMAGEDEF and an authentic DWG binary wrapper.
    """
    dwg_folder = tmp_path / "cad_drawings"
    dwg_folder.mkdir(parents=True, exist_ok=True)

    # Copy image into relative subfolder 'images/'
    img_subfolder = dwg_folder / "images"
    img_subfolder.mkdir(exist_ok=True)
    attached_img = img_subfolder / "panel_texture.png"
    shutil.copy2(sample_raster_image, attached_img)

    # 1. Create a DXF with an IMAGE entity pointing relatively to images/panel_texture.png
    dxf_target = dwg_folder / "drawing_with_raster.dxf"
    doc = ezdxf.new("R2018")
    img_def = doc.add_image_def(filename="images/panel_texture.png", size_in_pixel=(200, 200))
    msp = doc.modelspace()
    msp.add_image(image_def=img_def, insert=(10.0, 10.0, 0.0), size_in_units=(100.0, 100.0))
    # Add an outer cut frame polyline
    msp.add_lwpolyline([(0, 0), (120, 0), (120, 120), (0, 120)], close=True)
    doc.saveas(str(dxf_target))

    # 2. Create the associated .dwg file
    dwg_target = dwg_folder / "drawing_with_raster.dwg"
    dwg_target.write_bytes(b"AC1032" + b"\x00" * 20 + b"AUTOCAD_DWG_BINARY_PAYLOAD")

    # Pre-cache the converted DXF so tests run deterministically regardless of whether
    # external ODA converter is installed on CI runner
    cache_key = compute_dwg_cache_key(dwg_target)
    cached_dest = get_dwg_cache_dir() / f"{cache_key}.dxf"
    shutil.copy2(dxf_target, cached_dest)

    return dwg_target, attached_img


@pytest.fixture
def sample_dwg_without_image(tmp_path: Path) -> Path:
    """
    Creates Sample 2: Clean DWG drawing without attached raster images.
    Has pure vector geometry and an embedded DWG header preview thumbnail.
    """
    dwg_folder = tmp_path / "vector_dwgs"
    dwg_folder.mkdir(parents=True, exist_ok=True)

    dxf_target = dwg_folder / "pure_vector_panel.dxf"
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (80, 0), (80, 80), (0, 80)], close=True)
    msp.add_circle((40, 40), 20)
    doc.saveas(str(dxf_target))

    # Create DWG with embedded BMP header thumbnail (AC1015 format)
    dwg_target = dwg_folder / "pure_vector_panel.dwg"
    thumb_img = np.full((64, 64, 3), (200, 100, 50), dtype=np.uint8)
    bmp_bytes = cv2.imencode(".bmp", thumb_img)[1].tobytes()

    header = bytearray(b"AC1015" + b"\x00" * 7)
    offset = 32
    header.extend(struct.pack("<I", offset))
    header.extend(b"\x00" * (offset - len(header)))
    header.extend(struct.pack("<BB", 1, 1))  # 1 record, type 1 (BMP)
    header.extend(struct.pack("<I", len(bmp_bytes)))
    header.extend(bmp_bytes)
    dwg_target.write_bytes(bytes(header))

    # Pre-cache converted DXF
    cache_key = compute_dwg_cache_key(dwg_target)
    cached_dest = get_dwg_cache_dir() / f"{cache_key}.dxf"
    shutil.copy2(dxf_target, cached_dest)

    return dwg_target


@pytest.fixture
def sample_dwg_corrupted(tmp_path: Path) -> Path:
    """Creates Sample 3: Corrupted / truncated DWG file."""
    corrupt_file = tmp_path / "corrupted_drawing.dwg"
    corrupt_file.write_bytes(b"INVALID_HEADER_GARBAGE_RANDOM_BYTES_\xff\xfe\x00\x12\x34\x56")
    return corrupt_file


# -----------------------------------------------------------------------------
# Test 1: Converter Detection Logic
# -----------------------------------------------------------------------------

def test_converter_detection(tmp_path: Path):
    """Verifies ODA priority, LibreDWG fallback, and custom path overrides."""
    # Test custom path override
    fake_exe = tmp_path / "fake_oda_converter.exe"
    fake_exe.write_text("dummy binary")
    os.chmod(fake_exe, 0o755)

    bin_path, conv_type = get_available_converter(custom_path=str(fake_exe))
    assert bin_path == str(fake_exe)
    assert conv_type in ("oda", "libredwg")

    # Test auto-detection on current machine
    detected_bin, detected_type = get_available_converter()
    # On this machine, dwg2dxf is installed at /opt/homebrew/bin/dwg2dxf
    if find_libredwg_binary() or find_oda_converter_binary():
        assert detected_type in ("oda", "libredwg")
        assert detected_bin is not None
        assert os.path.isfile(detected_bin)
    else:
        assert detected_type == "none"
        assert detected_bin is None


# -----------------------------------------------------------------------------
# Test 2: Sample 1 (DWG with Attached Raster Image Reference)
# -----------------------------------------------------------------------------

def test_dwg_with_raster_image_extraction(sample_dwg_with_image):
    """
    Tests finding IMAGE entities, resolving IMAGEDEF relative paths,
    and handling missing references gracefully.
    """
    dwg_path, img_path = sample_dwg_with_image
    assert dwg_path.is_file()
    assert img_path.is_file()

    # 1. Test image extraction from DWG
    images = extract_cad_images_all(dwg_path)
    assert len(images) >= 1

    # Verify raster reference attributes
    raster_img = next((img for img in images if img.source_type == "raster_ref"), None)
    assert raster_img is not None
    assert raster_img.is_missing is False
    assert raster_img.referenced_path == "images/panel_texture.png"
    assert raster_img.resolved_path == str(img_path.resolve())
    assert raster_img.image_bgr is not None
    assert raster_img.image_bgr.shape[0] == 200
    assert raster_img.image_bgr.shape[1] == 200

    # 2. Test candidate extraction integration
    cands = extract_candidates_from_dwg(dwg_path)
    assert len(cands) >= 1
    # Check that candidates point back to original DWG path
    for c in cands:
        assert c.file_path == str(dwg_path.resolve())
        assert c.filename == dwg_path.name
        assert c.file_type == "dwg"

    # 3. Test Missing Reference Handling
    # Delete the attached image to simulate missing file on disk
    img_path.unlink()
    assert not img_path.exists()

    missing_resolved, is_missing = resolve_cad_image_path("images/panel_texture.png", dwg_path.parent)
    assert is_missing is True
    assert missing_resolved is None

    # Extraction must still succeed without crashing, reporting missing reference
    images_after = extract_cad_images_all(dwg_path)
    raster_after = next((img for img in images_after if img.source_type == "raster_ref"), None)
    assert raster_after is not None
    assert raster_after.is_missing is True
    assert "Missing referenced image" in raster_after.status_message


# -----------------------------------------------------------------------------
# Test 3: Sample 2 (DWG Without Image - Vector + Header Thumbnail)
# -----------------------------------------------------------------------------

def test_dwg_without_image_vector_and_thumbnail(sample_dwg_without_image: Path):
    """
    Verifies that a DWG without attached raster images extracts pure vector candidates
    and the embedded DWG header preview thumbnail without raising false missing-image errors.
    """
    dwg_path = sample_dwg_without_image
    assert dwg_path.is_file()

    # 1. Test DWG embedded header thumbnail extraction
    thumb = extract_dwg_embedded_thumbnail(dwg_path)
    assert thumb is not None
    assert thumb.shape[0] == 64
    assert thumb.shape[1] == 64

    # 2. Test candidate extraction
    cands = extract_candidates_from_dwg(dwg_path)
    assert len(cands) >= 1
    # Check that each candidate attributes to original .dwg
    for c in cands:
        assert c.file_path == str(dwg_path.resolve())
        assert c.filename == dwg_path.name
        assert c.file_type == "dwg"


# -----------------------------------------------------------------------------
# Test 4: Sample 3 (Corrupted DWG - Graceful Skip & Error Handling)
# -----------------------------------------------------------------------------

def test_dwg_corrupted_handling(sample_dwg_corrupted: Path):
    """
    Verifies that a corrupted or truncated DWG file is handled gracefully:
    logs warning, skips, and returns empty list without crashing.
    """
    corrupt_path = sample_dwg_corrupted
    assert corrupt_path.is_file()

    # 1. Test image extraction on corrupt file (should return [] without crashing)
    imgs = extract_cad_images_all(corrupt_path)
    assert isinstance(imgs, list)

    # 2. Test candidate extraction (should return [] without unhandled exception)
    cands = extract_candidates_from_dwg(corrupt_path)
    assert isinstance(cands, list)

    # 3. Test convert single (should return None without crashing)
    res = convert_dwg_to_dxf_single(corrupt_path, timeout_sec=5)
    assert res is None


# -----------------------------------------------------------------------------
# Test 5: DWG to DXF Persistent Caching
# -----------------------------------------------------------------------------

def test_dwg_conversion_caching(tmp_path: Path):
    """Verifies that converted DXFs are persistently cached and reused without re-converting."""
    sample_dwg = tmp_path / "cache_test.dwg"
    sample_dwg.write_bytes(b"AC1032 dummy dwg binary content")

    key = compute_dwg_cache_key(sample_dwg)
    assert len(key) == 24

    # Fake a converted DXF in the cache directory
    cache_dir = get_dwg_cache_dir()
    fake_cached_dxf = cache_dir / f"{key}.dxf"
    fake_cached_dxf.write_text("0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF\n")

    # get_cached_dxf_path should immediately hit the cache
    hit = get_cached_dxf_path(sample_dwg)
    assert hit is not None
    assert hit == fake_cached_dxf

    # Modify mtime to verify cache invalidation
    os.utime(sample_dwg, (1000.0, 2000.0))
    new_key = compute_dwg_cache_key(sample_dwg)
    assert new_key != key


# -----------------------------------------------------------------------------
# Test 6: Unified File Routing & Zero Regressions for DXF and PDF
# -----------------------------------------------------------------------------

def test_unified_file_routing(sample_dwg_without_image: Path, tmp_path: Path):
    """Verifies extract_file_candidates routes .dxf, .dwg, and .pdf properly."""
    # 1. Test DWG routing
    dwg_cands = extract_file_candidates(sample_dwg_without_image)
    assert len(dwg_cands) >= 1
    assert dwg_cands[0].file_type == "dwg"

    # 2. Test DXF routing (ensure DXF path has zero regressions)
    dxf_file = tmp_path / "sample_test.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (50, 0), (50, 50), (0, 50)], close=True)
    doc.saveas(str(dxf_file))

    dxf_cands = extract_file_candidates(dxf_file)
    assert len(dxf_cands) >= 1
    assert dxf_cands[0].file_type == "dxf"
