"""
Unit and benchmark tests for the Production Two-Stage Indexing Engine.
Tests:
- Profiling utilities (StageTimer, measure_stage, ProfilingSession)
- Stage A metadata extraction (PDF, DXF, DWG rejection)
- SQLite FTS5 full-text search and synchronization triggers
- Incremental indexing change detection (0ms skip of unchanged files)
- End-to-end TwoStageIndexingEngine execution
"""

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from app.core.db import CatalogDatabase
from app.core.indexing_engine import TwoStageIndexingEngine, TwoStageProgress
from app.core.profiling import ProfilingSession, StageTimer, measure_stage
from app.core.stage_a_extractor import (
    extract_dxf_metadata,
    extract_metadata_batch,
    extract_single_file_metadata,
)


# -----------------------------------------------------------------------------
# Profiling Suite Tests
# -----------------------------------------------------------------------------

def test_stage_timer_and_profiling_session():
    session = ProfilingSession("test_session")

    with session.timer("fast_stage"):
        time.sleep(0.01)

    with session.timer("fast_stage"):
        time.sleep(0.02)

    stats = session.stages.get("fast_stage")
    assert stats is not None
    assert stats.count == 2
    assert stats.total_seconds >= 0.025
    assert stats.avg_seconds > 0
    assert stats.min_seconds <= stats.max_seconds

    # Test measure_stage
    res = measure_stage("calc", lambda x, y: x + y, 5, 10)
    assert res == 15

    session.finish(items_processed=10, tiles_generated=25)
    summary = session.get_summary()
    assert summary["items_processed"] == 10
    assert summary["tiles_generated"] == 25
    assert "fast_stage" in summary["stages"]

    report_str = session.format_report()
    assert "Indexing Benchmark Report" in report_str
    assert "fast_stage" in report_str


# -----------------------------------------------------------------------------
# Stage A Extractor Tests
# -----------------------------------------------------------------------------

def test_stage_a_dwg_handling(tmp_path):
    dwg_file = tmp_path / "panel_sample.dwg"
    dwg_file.write_bytes(b"AC1032 dummy binary dwg content")

    rec = extract_single_file_metadata(str(dwg_file), modified_ns=1000, file_size=100)
    assert rec["metadata_status"] == "completed"
    assert rec["extension"] == ".dwg"
    assert "AutoCAD 2018" in rec["searchable_metadata"]
    assert rec["content_hash"] is not None


def test_stage_a_dxf_extraction(tmp_path):
    import ezdxf

    dxf_file = tmp_path / "laser_grill.dxf"
    doc = ezdxf.new("R2010")
    doc.layers.add("PANEL_FRAME")
    doc.layers.add("CUT_INTERIOR")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 100), (0, 100)], close=True)
    msp.add_text("DESIGN-FLORAL-01", dxfattribs={"height": 5.0})
    doc.saveas(str(dxf_file))

    stat = dxf_file.stat()
    rec = extract_single_file_metadata(str(dxf_file), modified_ns=stat.st_mtime_ns, file_size=stat.st_size)

    assert rec["metadata_status"] == "completed"
    assert rec["extension"] == ".dxf"
    assert "laser_grill" in rec["searchable_metadata"]
    assert "PANEL_FRAME" in rec["searchable_metadata"]
    assert "DESIGN-FLORAL-01" in rec["searchable_metadata"]
    assert rec["content_hash"] is not None


# -----------------------------------------------------------------------------
# Database Manifest & FTS5 Tests
# -----------------------------------------------------------------------------

def test_db_catalogue_manifest_and_fts(tmp_path):
    db_file = tmp_path / "test_manifest.db"
    db = CatalogDatabase(db_file, use_mysql=False)

    records = [
        {
            "path": "/catalog/panels/screen_lotus.dxf",
            "filename": "screen_lotus.dxf",
            "extension": ".dxf",
            "file_size": 45000,
            "modified_ns": 1000000000,
            "content_hash": "hash_lotus_1",
            "page_count": 1,
            "searchable_metadata": "screen_lotus floral lattice panel decorative steel",
            "metadata_status": "completed",
            "visual_status": "completed",
        },
        {
            "path": "/catalog/gates/modern_geometric.pdf",
            "filename": "modern_geometric.pdf",
            "extension": ".pdf",
            "file_size": 120000,
            "modified_ns": 2000000000,
            "content_hash": "hash_geom_2",
            "page_count": 4,
            "searchable_metadata": "modern_geometric architectural gate entrance aluminum",
            "metadata_status": "completed",
            "visual_status": "pending",
        },
    ]

    db.upsert_catalogue_files_batch(records)

    # Verify retrieval
    all_files = db.get_all_catalogue_files()
    assert len(all_files) == 2

    # Verify pending visual query
    pending = db.get_pending_visual_files()
    assert len(pending) == 1
    assert pending[0]["filename"] == "modern_geometric.pdf"

    # Verify FTS5 search
    lotus_matches = db.search_catalogue_fts("lotus")
    assert len(lotus_matches) == 1
    assert lotus_matches[0]["filename"] == "screen_lotus.dxf"

    floral_matches = db.search_catalogue_fts("floral lattice")
    assert len(floral_matches) == 1

    arch_matches = db.search_catalogue_fts("architectural")
    assert len(arch_matches) == 1
    assert arch_matches[0]["filename"] == "modern_geometric.pdf"

    # Test update trigger sync
    db.update_catalogue_visual_status("/catalog/gates/modern_geometric.pdf", "completed")
    pending_after = db.get_pending_visual_files()
    assert len(pending_after) == 0


# -----------------------------------------------------------------------------
# Incremental Scan & Engine Workflow
# -----------------------------------------------------------------------------

def test_incremental_scanner_skip(tmp_path):
    import ezdxf

    # Create dummy catalog directory
    cat_dir = tmp_path / "my_catalogs"
    cat_dir.mkdir()

    dxf1 = cat_dir / "panel_1.dxf"
    doc1 = ezdxf.new("R2010")
    msp1 = doc1.modelspace()
    msp1.add_line((0, 0), (50, 50))
    doc1.saveas(str(dxf1))

    # Mock CatalogIndex
    class DummyEmbedder:
        def embed_masks(self, masks, batch_size=32):
            return np.ones((len(masks), 384), dtype=np.float32)

    class DummyIndex:
        def __init__(self, db_p, data_p):
            self.data_dir = data_p
            self.thumbnails_dir = data_p / "thumbnails"
            self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
            self.db = CatalogDatabase(db_p, use_mysql=False)
            self.embedder = DummyEmbedder()
            self.total_vectors = 0

        def add_vectors(self, embeddings, ids):
            self.total_vectors += len(ids)

        def remove_vectors(self, ids):
            self.total_vectors = max(0, self.total_vectors - len(ids))

        def save_index(self):
            pass

    dummy_idx = DummyIndex(tmp_path / "cat.db", tmp_path / "data")
    engine = TwoStageIndexingEngine(dummy_idx, max_workers=1)

    # First run (Stage A metadata only)
    summary_a = engine.run_two_stage_index(cat_dir, mode="metadata_only")
    assert summary_a["items_processed"] == 1

    rec = dummy_idx.db.get_catalogue_file(str(dxf1.resolve()))
    assert rec is not None
    assert rec["metadata_status"] == "completed"

    # Mark as completed visual
    dummy_idx.db.update_catalogue_visual_status(str(dxf1.resolve()), "completed")

    # Second run (Unchanged file should be skipped)
    summary_b = engine.run_two_stage_index(cat_dir, mode="full")
    # File is unchanged, so 0 tiles generated
    assert summary_b["tiles_generated"] == 0
