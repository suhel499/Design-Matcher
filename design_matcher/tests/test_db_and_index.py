"""
Unit tests for SQLite database operations and FAISS vector indexing.
"""

from pathlib import Path
import numpy as np
import pytest

from app.core.db import CatalogDatabase
from app.core.embed import EMBEDDING_DIM
from app.core.index import CatalogIndex


def test_sqlite_db_operations(tmp_path: Path):
    """Tests SQLite metadata storage, cascading deletions, and queries."""
    db_file = tmp_path / "test_catalog.db"
    db = CatalogDatabase(db_file)

    # Record a PDF file
    pdf_path = str(tmp_path / "catalogs" / "vol1.pdf")
    db.record_indexed_file(
        pdf_path=pdf_path,
        folder="catalogs",
        file_hash="hash12345",
        file_size=10240,
        last_modified=1700000000.0,
        page_count=2,
        tile_count=5,
    )

    rec = db.get_indexed_file(pdf_path)
    assert rec is not None
    assert rec["file_hash"] == "hash12345"
    assert rec["page_count"] == 2

    # Insert tiles batch
    tiles_data = [
        {
            "pdf_path": pdf_path,
            "folder": "catalogs",
            "page_number": 1,
            "tile_index": 0,
            "bbox": [10, 20, 100, 150],
            "crop_type": "full",
            "design_number": "D-01",
            "thumbnail_path": "thumbs/1.jpg",
            "file_hash": "hash12345",
        },
        {
            "pdf_path": pdf_path,
            "folder": "catalogs",
            "page_number": 1,
            "tile_index": 0,
            "bbox": [10, 20, 100, 75],
            "crop_type": "top_half",
            "design_number": "D-01",
            "thumbnail_path": "thumbs/1_top.jpg",
            "file_hash": "hash12345",
        },
    ]

    tile_ids = db.insert_tiles_batch(tiles_data)
    assert len(tile_ids) == 2
    assert tile_ids[0] > 0
    assert tile_ids[1] > tile_ids[0]

    # Query tile by ID
    t1 = db.get_tile(tile_ids[0])
    assert t1 is not None
    assert t1["bbox"] == [10, 20, 100, 150]
    assert t1["design_number"] == "D-01"

    # Query multiple tiles
    tiles_map = db.get_tiles_by_ids(tile_ids)
    assert len(tiles_map) == 2
    assert tile_ids[0] in tiles_map

    # Query with >500 IDs to test chunking
    large_id_list = list(range(1, 600))
    large_map = db.get_tiles_by_ids(large_id_list)
    assert isinstance(large_map, dict)
    assert tile_ids[0] in large_map

    # Test deletion
    deleted_ids = db.delete_file_records(pdf_path)
    assert set(deleted_ids) == set(tile_ids)
    assert db.get_tile(tile_ids[0]) is None
    assert db.get_indexed_file(pdf_path) is None


def test_faiss_vector_index(tmp_path: Path):
    """Tests FAISS IndexIDMap2 vector insertion, search, removal, and serialization."""
    idx = CatalogIndex(data_dir=tmp_path)

    # Create dummy normalized vectors
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((3, EMBEDDING_DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    ids = [101, 102, 103]
    idx.add_vectors(vecs, ids)
    assert idx.total_vectors == 3

    # Search with first vector as query
    query = vecs[0:1]
    scores, res_ids = idx.search_vectors(query, top_k=3)

    assert res_ids[0, 0] == 101
    assert scores[0, 0] > 0.9999  # Identical cosine similarity ~ 1.0

    # Save to disk and reload
    idx.save_index()
    assert idx.faiss_path.exists()

    idx_reloaded = CatalogIndex(data_dir=tmp_path)
    assert idx_reloaded.total_vectors == 3

    # Test removing an ID
    idx_reloaded.remove_vectors([102])
    assert idx_reloaded.total_vectors == 2

    # Verify ID 102 is no longer retrieved
    _, updated_ids = idx_reloaded.search_vectors(query, top_k=3)
    assert 102 not in updated_ids[0]
