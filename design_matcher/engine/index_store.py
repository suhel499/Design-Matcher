"""
Persistent storage layer combining SQLite and FAISS.
- SQLite stores metadata, design coordinates, file hashes, thumbnails, and canonical masks.
- FAISS IndexFlatIP (via IndexIDMap) stores L2-normalized vectors mapped to design IDs.
- Provides atomic saves, transactions, incremental updates, and removal.
- Supports both PDF and DXF vector CAD designs with schema migration v2.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np

from .config import config
from .models import DesignRecord

logger = logging.getLogger(__name__)


class IndexStore:
    """Manages SQLite catalog database and FAISS vector index."""

    def __init__(
        self,
        db_path: Path = config.DB_PATH,
        index_path: Path = config.INDEX_PATH,
        dimension: int = 1282,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.index_path = Path(index_path).resolve()
        self.dimension = dimension
        self._lock = threading.Lock()

        # Ensure parent directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_sqlite()

        # Initialize or load FAISS index
        self.index: faiss.IndexIDMap = self._init_faiss()

    def _init_sqlite(self) -> None:
        """Creates tables, applies schema migrations (user_version v2), and configures indexes."""
        with self.conn:
            cur = self.conn.cursor()
            cur.execute("PRAGMA user_version")
            ver = cur.fetchone()[0]

            # 1. Base tables
            self.conn.executescript("""
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS pdf_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    folder TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime REAL NOT NULL,
                    sha1 TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS dxf_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    folder TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime REAL NOT NULL,
                    sha1 TEXT NOT NULL,
                    entity_count INTEGER NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS index_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_path TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    design_id INTEGER NOT NULL,
                    correct INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)

            # 2. Check designs table schema
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='designs'")
            has_designs = cur.fetchone() is not None

            if not has_designs:
                # Create designs with complete v2 schema
                self.conn.executescript("""
                    CREATE TABLE designs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pdf_id INTEGER,
                        dxf_id INTEGER,
                        page_no INTEGER,
                        x INTEGER NOT NULL,
                        y INTEGER NOT NULL,
                        w INTEGER NOT NULL,
                        h INTEGER NOT NULL,
                        thumb BLOB NOT NULL,
                        mask_hash TEXT NOT NULL,
                        aspect_ratio REAL DEFAULT 1.0,
                        fill_ratio REAL DEFAULT 0.5,
                        canonical_mask BLOB,
                        source_type TEXT DEFAULT 'pdf',
                        label TEXT,
                        bbox_x REAL,
                        bbox_y REAL,
                        bbox_w REAL,
                        bbox_h REAL,
                        same_design_group INTEGER,
                        FOREIGN KEY (pdf_id) REFERENCES pdf_files(id) ON DELETE CASCADE,
                        FOREIGN KEY (dxf_id) REFERENCES dxf_files(id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_designs_pdf_id ON designs(pdf_id);
                    CREATE INDEX IF NOT EXISTS idx_designs_dxf_id ON designs(dxf_id);
                    CREATE INDEX IF NOT EXISTS idx_designs_label ON designs(label);
                    CREATE INDEX IF NOT EXISTS idx_designs_group ON designs(same_design_group);
                    PRAGMA user_version = 2;
                """)
            elif ver < 2:
                # Migrate existing designs table to v2
                logger.info("Migrating SQLite designs table to v2 (DXF support + nullable pdf_id)...")
                cur.execute("PRAGMA table_info(designs)")
                existing_cols = {row["name"] for row in cur.fetchall()}

                self.conn.execute("PRAGMA foreign_keys = OFF")
                self.conn.executescript("""
                    CREATE TABLE designs_v2 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pdf_id INTEGER,
                        dxf_id INTEGER,
                        page_no INTEGER,
                        x INTEGER NOT NULL,
                        y INTEGER NOT NULL,
                        w INTEGER NOT NULL,
                        h INTEGER NOT NULL,
                        thumb BLOB NOT NULL,
                        mask_hash TEXT NOT NULL,
                        aspect_ratio REAL DEFAULT 1.0,
                        fill_ratio REAL DEFAULT 0.5,
                        canonical_mask BLOB,
                        source_type TEXT DEFAULT 'pdf',
                        label TEXT,
                        bbox_x REAL,
                        bbox_y REAL,
                        bbox_w REAL,
                        bbox_h REAL,
                        same_design_group INTEGER,
                        FOREIGN KEY (pdf_id) REFERENCES pdf_files(id) ON DELETE CASCADE,
                        FOREIGN KEY (dxf_id) REFERENCES dxf_files(id) ON DELETE CASCADE
                    );
                """)

                # Copy existing data
                col_source = "source_type" if "source_type" in existing_cols else "'pdf'"
                self.conn.execute(f"""
                    INSERT INTO designs_v2 (
                        id, pdf_id, page_no, x, y, w, h, thumb, mask_hash,
                        aspect_ratio, fill_ratio, canonical_mask, source_type
                    )
                    SELECT id, pdf_id, page_no, x, y, w, h, thumb, mask_hash,
                           aspect_ratio, fill_ratio, canonical_mask, {col_source}
                    FROM designs;
                """)

                self.conn.executescript("""
                    DROP TABLE designs;
                    ALTER TABLE designs_v2 RENAME TO designs;
                    CREATE INDEX IF NOT EXISTS idx_designs_pdf_id ON designs(pdf_id);
                    CREATE INDEX IF NOT EXISTS idx_designs_dxf_id ON designs(dxf_id);
                    CREATE INDEX IF NOT EXISTS idx_designs_label ON designs(label);
                    CREATE INDEX IF NOT EXISTS idx_designs_group ON designs(same_design_group);
                    PRAGMA user_version = 2;
                """)
                self.conn.execute("PRAGMA foreign_keys = ON")

            # Verify and apply DXF schema extension columns (migration-safe)
            cur.execute("PRAGMA table_info(designs)")
            existing_cols = {row["name"] for row in cur.fetchall()}
            for col_name, col_def in [
                ("source_type", "TEXT DEFAULT 'pdf'"),
                ("design_code", "TEXT"),
                ("dxf_bbox", "TEXT"),
                ("polarity_variant", "INTEGER DEFAULT 0"),
                ("entity_count", "INTEGER DEFAULT 0"),
                ("design_index_info", "TEXT"),
            ]:
                if col_name not in existing_cols:
                    self.conn.execute(f"ALTER TABLE designs ADD COLUMN {col_name} {col_def}")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_designs_design_code ON designs(design_code)")

    def _init_faiss(self) -> faiss.IndexIDMap:
        """Loads an existing FAISS index or creates a new IndexIDMap."""
        if self.index_path.is_file() and self.index_path.stat().st_size > 0:
            try:
                loaded = faiss.read_index(str(self.index_path))
                if isinstance(loaded, faiss.IndexIDMap):
                    logger.info("Loaded FAISS IndexIDMap from %s (%d vectors)", self.index_path, loaded.ntotal)
                    return loaded
                else:
                    mapped = faiss.IndexIDMap(loaded)
                    logger.info("Wrapped loaded FAISS index into IndexIDMap (%d vectors)", mapped.ntotal)
                    return mapped
            except Exception as exc:
                logger.warning("Failed to read FAISS index (%s). Recreating new index.", exc)

        sub_index = faiss.IndexFlatIP(self.dimension)
        return faiss.IndexIDMap(sub_index)

    def save(self) -> None:
        """Commits SQLite transaction and saves FAISS index atomically to disk."""
        with self._lock:
            self.conn.commit()
            if self.index is not None:
                tmp_path = self.index_path.with_name(f"{self.index_path.name}.tmp.{os.getpid()}")
                faiss.write_index(self.index, str(tmp_path))
                os.replace(tmp_path, self.index_path)

    def close(self) -> None:
        """Saves state and closes connections."""
        try:
            self.save()
        except Exception:
            pass
        self.conn.close()

    # ---------------------------------------------------------
    # PDF Operations
    # ---------------------------------------------------------

    def get_pdf_record_by_path(self, path: str) -> Optional[dict]:
        """Retrieves an existing pdf_files record by its path."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM pdf_files WHERE path = ?", (str(path),))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_indexed_pdf_paths(self) -> Dict[str, dict]:
        """Returns map of {path: dict(row)} for all indexed PDFs."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, path, folder, size, mtime, sha1, page_count FROM pdf_files")
        return {row["path"]: dict(row) for row in cur.fetchall()}

    def upsert_pdf_record(
        self,
        path: str,
        folder: str,
        size: int,
        mtime: float,
        sha1: str,
        page_count: int,
    ) -> int:
        """Inserts or updates a pdf_files record and returns its ID."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id FROM pdf_files WHERE path = ?", (str(path),))
            existing = cur.fetchone()
            if existing:
                pdf_id = int(existing["id"])
                cur.execute(
                    """
                    UPDATE pdf_files
                    SET folder = ?, size = ?, mtime = ?, sha1 = ?, page_count = ?, indexed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (folder, size, mtime, sha1, page_count, pdf_id),
                )
                return pdf_id
            else:
                cur.execute(
                    """
                    INSERT INTO pdf_files (path, folder, size, mtime, sha1, page_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (str(path), folder, size, mtime, sha1, page_count),
                )
                if cur.lastrowid is None:
                    raise RuntimeError("INSERT into pdf_files did not return a lastrowid")
                return int(cur.lastrowid)

    def remove_pdf(self, pdf_id: int) -> int:
        """
        Removes a PDF and all its designs from SQLite and FAISS.
        Returns the number of removed designs.
        """
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id FROM designs WHERE pdf_id = ?", (pdf_id,))
            design_ids = [int(row["id"]) for row in cur.fetchall()]

            if design_ids and self.index.ntotal > 0:
                id_array = np.array(design_ids, dtype=np.int64)
                sel = faiss.IDSelectorArray(id_array)
                self.index.remove_ids(sel)

            cur.execute("DELETE FROM designs WHERE pdf_id = ?", (pdf_id,))
            cur.execute("DELETE FROM pdf_files WHERE id = ?", (pdf_id,))
            self.conn.commit()
            return len(design_ids)

    # ---------------------------------------------------------
    # DXF Operations
    # ---------------------------------------------------------

    def get_dxf_record_by_path(self, path: str) -> Optional[dict]:
        """Retrieves an existing dxf_files record by its path."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM dxf_files WHERE path = ?", (str(path),))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_indexed_dxf_paths(self) -> Dict[str, dict]:
        """Returns map of {path: dict(row)} for all indexed DXF files."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, path, folder, size, mtime, sha1, entity_count FROM dxf_files")
        return {row["path"]: dict(row) for row in cur.fetchall()}

    def upsert_dxf_record(
        self,
        path: str,
        folder: str,
        size: int,
        mtime: float,
        sha1: str,
        entity_count: int,
    ) -> int:
        """Inserts or updates a dxf_files record and returns its ID."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id FROM dxf_files WHERE path = ?", (str(path),))
            existing = cur.fetchone()
            if existing:
                dxf_id = int(existing["id"])
                cur.execute(
                    """
                    UPDATE dxf_files
                    SET folder = ?, size = ?, mtime = ?, sha1 = ?, entity_count = ?, indexed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (folder, size, mtime, sha1, entity_count, dxf_id),
                )
                return dxf_id
            else:
                cur.execute(
                    """
                    INSERT INTO dxf_files (path, folder, size, mtime, sha1, entity_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (str(path), folder, size, mtime, sha1, entity_count),
                )
                if cur.lastrowid is None:
                    raise RuntimeError("INSERT did not return a valid row ID")
                return int(cur.lastrowid)

    def remove_dxf(self, dxf_id: int) -> int:
        """
        Removes a DXF file and all its designs from SQLite and FAISS.
        Returns the number of removed designs.
        """
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id FROM designs WHERE dxf_id = ?", (dxf_id,))
            design_ids = [int(row["id"]) for row in cur.fetchall()]

            if design_ids and self.index.ntotal > 0:
                id_array = np.array(design_ids, dtype=np.int64)
                sel = faiss.IDSelectorArray(id_array)
                self.index.remove_ids(sel)

            cur.execute("DELETE FROM designs WHERE dxf_id = ?", (dxf_id,))
            cur.execute("DELETE FROM dxf_files WHERE id = ?", (dxf_id,))
            self.conn.commit()
            return len(design_ids)

    # ---------------------------------------------------------
    # Design Operations & Multi-format Storage
    # ---------------------------------------------------------

    def add_designs(
        self,
        records: List[dict],
        vectors: np.ndarray,
    ) -> List[int]:
        """
        Adds multiple designs and their embedding vectors atomically.

        Args:
            records: List of dicts with design attributes (supports both PDF and DXF).
            vectors: (N, D) float32 numpy array, L2 normalized.

        Returns:
            List of generated design IDs.
        """
        if not records or len(vectors) == 0:
            return []

        inserted_ids: List[int] = []
        with self._lock:
            cur = self.conn.cursor()
            for rec in records:
                lbl_val = rec.get("design_code") or rec.get("label")
                d_bbox = rec.get("dxf_bbox")
                if isinstance(d_bbox, (list, tuple)) and len(d_bbox) >= 4:
                    bx, by, bw, bh = d_bbox[0], d_bbox[1], d_bbox[2], d_bbox[3]
                    dxf_bbox_str = f"{bx},{by},{bw},{bh}"
                elif rec.get("bbox_x") is not None:
                    bx, by, bw, bh = rec.get("bbox_x"), rec.get("bbox_y"), rec.get("bbox_w"), rec.get("bbox_h")
                    dxf_bbox_str = f"{bx},{by},{bw},{bh}"
                else:
                    bx = by = bw = bh = None
                    dxf_bbox_str = None

                cur.execute(
                    """
                    INSERT INTO designs (
                        pdf_id, dxf_id, source_type, page_no, label, design_code,
                        bbox_x, bbox_y, bbox_w, bbox_h, dxf_bbox, polarity_variant, entity_count,
                        design_index_info, x, y, w, h, thumb, mask_hash,
                        aspect_ratio, fill_ratio, canonical_mask, same_design_group
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rec.get("pdf_id"),
                        rec.get("dxf_id"),
                        rec.get("source_type", "pdf"),
                        rec.get("page_no"),
                        lbl_val,
                        lbl_val,
                        bx,
                        by,
                        bw,
                        bh,
                        dxf_bbox_str,
                        int(rec.get("polarity_variant", 0)),
                        int(rec.get("entity_count", 0)),
                        str(rec.get("design_index_info", "")),
                        rec["x"],
                        rec["y"],
                        rec["w"],
                        rec["h"],
                        rec["thumb"],
                        rec["mask_hash"],
                        rec.get("aspect_ratio", 1.0),
                        rec.get("fill_ratio", 0.5),
                        rec.get("canonical_mask"),
                        rec.get("same_design_group"),
                    ),
                )
                if cur.lastrowid is None:
                    raise RuntimeError("INSERT into designs did not return a lastrowid")
                inserted_ids.append(int(cur.lastrowid))

            # Add to FAISS index
            ids_arr = np.array(inserted_ids, dtype=np.int64)
            vecs_float = np.ascontiguousarray(vectors, dtype=np.float32)
            self.index.add_with_ids(vecs_float, ids_arr)

        return inserted_ids

    def search_vectors(
        self,
        query_vectors: np.ndarray,
        top_k: int = config.RERANK_N,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches FAISS index with batch query vectors.

        Returns:
            (scores, indices): shapes (N, top_k), scores are float32 inner products.
        """
        with self._lock:
            if self.index.ntotal == 0:
                n_queries = len(query_vectors)
                return np.full((n_queries, top_k), -1.0, dtype=np.float32), np.full(
                    (n_queries, top_k), -1, dtype=np.int64
                )

            q_vecs = np.ascontiguousarray(query_vectors, dtype=np.float32)
            scores, indices = self.index.search(q_vecs, min(top_k, self.index.ntotal))
            return scores, indices

    def get_designs_batch(self, design_ids: List[int]) -> Dict[int, DesignRecord]:
        """Fetches full DesignRecord models for given design IDs across both PDF and DXF sources."""
        if not design_ids:
            return {}

        clean_ids = [int(i) for i in design_ids]
        placeholders = ",".join("?" for _ in clean_ids)
        sql = f"""
            SELECT d.id, d.pdf_id,
                   COALESCE(p.path, x.path) AS file_path,
                   COALESCE(p.folder, x.folder) AS folder,
                   d.page_no,
                   d.x, d.y, d.w, d.h, d.thumb, d.mask_hash,
                   d.aspect_ratio, d.fill_ratio, d.canonical_mask,
                   COALESCE(d.source_type, 'pdf') AS source_type,
                   d.dxf_id, COALESCE(d.design_code, d.label) AS label,
                   COALESCE(d.design_code, d.label) AS design_code,
                   d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
                   d.same_design_group,
                   COALESCE(d.polarity_variant, 0) AS polarity_variant,
                   COALESCE(d.entity_count, 0) AS entity_count,
                   COALESCE(d.design_index_info, '') AS design_index_info
            FROM designs d
            LEFT JOIN pdf_files p ON d.pdf_id = p.id
            LEFT JOIN dxf_files x ON d.dxf_id = x.id
            WHERE d.id IN ({placeholders})
        """
        cur = self.conn.cursor()
        cur.execute(sql, clean_ids)
        results: Dict[int, DesignRecord] = {}
        for row in cur.fetchall():
            s_type = str(row["source_type"])
            f_path = str(row["file_path"]) if row["file_path"] is not None else ""
            bbox_dxf = None
            if row["bbox_x"] is not None:
                bbox_dxf = (
                    float(row["bbox_x"]),
                    float(row["bbox_y"]),
                    float(row["bbox_w"]),
                    float(row["bbox_h"]),
                )

            rec = DesignRecord(
                id=int(row["id"]),
                pdf_id=int(row["pdf_id"]) if row["pdf_id"] is not None else None,
                pdf_path=f_path,
                folder=str(row["folder"]) if row["folder"] is not None else "",
                page_no=int(row["page_no"]) if row["page_no"] is not None else None,
                x=int(row["x"]),
                y=int(row["y"]),
                w=int(row["w"]),
                h=int(row["h"]),
                thumb_png=bytes(row["thumb"]),
                mask_hash=str(row["mask_hash"]),
                aspect_ratio=float(row["aspect_ratio"]),
                fill_ratio=float(row["fill_ratio"]),
                canonical_mask_png=bytes(row["canonical_mask"]) if row["canonical_mask"] else None,
                source_type=s_type,
                dxf_id=int(row["dxf_id"]) if row["dxf_id"] is not None else None,
                label=str(row["label"]) if row["label"] is not None else None,
                design_code=str(row["design_code"]) if row["design_code"] is not None else None,
                bbox_dxf=bbox_dxf,
                same_design_group=int(row["same_design_group"]) if row["same_design_group"] is not None else None,
                polarity_variant=int(row["polarity_variant"]),
                entity_count=int(row["entity_count"]),
                design_index_info=str(row["design_index_info"]) if row["design_index_info"] else None,
            )
            results[rec.id] = rec
        return results

    def search_by_code(self, text: str, limit: int = 20) -> List[DesignRecord]:
        """
        Performs case-insensitive, space-insensitive, and symbol-insensitive match on
        design labels (e.g. 'PD010', 'pd 010', 'PD-010').
        Returns matched DesignRecords ordered by match quality.
        """
        if not text or not text.strip():
            return []

        clean_query = re.sub(r"[\s\-_#]", "", text).lower()
        if not clean_query:
            return []

        cur = self.conn.cursor()
        cur.execute("""
            SELECT d.id, d.pdf_id,
                   COALESCE(p.path, x.path) AS file_path,
                   COALESCE(p.folder, x.folder) AS folder,
                   d.page_no, d.x, d.y, d.w, d.h, d.thumb, d.mask_hash,
                   d.aspect_ratio, d.fill_ratio, d.canonical_mask,
                   COALESCE(d.source_type, 'pdf') AS source_type,
                   d.dxf_id,
                   COALESCE(d.design_code, d.label) AS label,
                   COALESCE(d.design_code, d.label) AS design_code,
                   d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
                   d.same_design_group,
                   COALESCE(d.polarity_variant, 0) AS polarity_variant,
                   COALESCE(d.entity_count, 0) AS entity_count
            FROM designs d
            LEFT JOIN pdf_files p ON d.pdf_id = p.id
            LEFT JOIN dxf_files x ON d.dxf_id = x.id
            WHERE (d.label IS NOT NULL AND TRIM(d.label) != '')
               OR (d.design_code IS NOT NULL AND TRIM(d.design_code) != '')
        """)

        candidates: List[Tuple[int, DesignRecord]] = []
        for row in cur.fetchall():
            raw_lbl = str(row["design_code"] or row["label"])
            clean_lbl = re.sub(r"[\s\-_#]", "", raw_lbl).lower()
            if not clean_lbl:
                continue

            score = 0
            if clean_lbl == clean_query:
                score = 100
            elif clean_lbl.startswith(clean_query):
                score = 80
            elif clean_query in clean_lbl:
                score = 60

            if score > 0:
                s_type = str(row["source_type"])
                f_path = str(row["file_path"]) if row["file_path"] is not None else ""
                bbox_dxf = None
                if row["bbox_x"] is not None:
                    bbox_dxf = (
                        float(row["bbox_x"]),
                        float(row["bbox_y"]),
                        float(row["bbox_w"]),
                        float(row["bbox_h"]),
                    )

                rec = DesignRecord(
                    id=int(row["id"]),
                    pdf_id=int(row["pdf_id"]) if row["pdf_id"] is not None else None,
                    pdf_path=f_path,
                    folder=str(row["folder"]) if row["folder"] is not None else "",
                    page_no=int(row["page_no"]) if row["page_no"] is not None else None,
                    x=int(row["x"]),
                    y=int(row["y"]),
                    w=int(row["w"]),
                    h=int(row["h"]),
                    thumb_png=bytes(row["thumb"]),
                    mask_hash=str(row["mask_hash"]),
                    aspect_ratio=float(row["aspect_ratio"]),
                    fill_ratio=float(row["fill_ratio"]),
                    canonical_mask_png=bytes(row["canonical_mask"]) if row["canonical_mask"] else None,
                    source_type=s_type,
                    dxf_id=int(row["dxf_id"]) if row["dxf_id"] is not None else None,
                    label=str(row["label"]) if row["label"] is not None else None,
                    design_code=str(row["design_code"]) if row["design_code"] is not None else None,
                    bbox_dxf=bbox_dxf,
                    same_design_group=int(row["same_design_group"]) if row["same_design_group"] is not None else None,
                    polarity_variant=int(row["polarity_variant"]),
                    entity_count=int(row["entity_count"]),
                )
                candidates.append((score, rec))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [c[1] for c in candidates[:limit]]

    def link_same_design_group(self, design_id_1: int, design_id_2: int) -> int:
        """Links two designs under a common same_design_group identifier."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, same_design_group FROM designs WHERE id IN (?, ?)", (design_id_1, design_id_2))
            rows = cur.fetchall()
            existing_groups = [r["same_design_group"] for r in rows if r["same_design_group"] is not None]
            group_id = min(existing_groups) if existing_groups else min(design_id_1, design_id_2)

            # Collect all group IDs that need to be merged (the two design IDs plus any existing group IDs)
            groups_to_merge = list({design_id_1, design_id_2} | set(existing_groups))
            placeholders = ",".join("?" for _ in groups_to_merge)
            cur.execute(
                f"""
                UPDATE designs
                SET same_design_group = ?
                WHERE id IN (?, ?) OR (same_design_group IS NOT NULL AND same_design_group IN ({placeholders}))
                """,
                [group_id, design_id_1, design_id_2] + groups_to_merge,
            )
            self.conn.commit()
            return group_id

    def get_related_designs(self, group_id: int, exclude_id: int) -> List[DesignRecord]:
        """Fetches linked designs in the same design group, excluding the specified ID."""
        if group_id is None:
            return []
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM designs WHERE same_design_group = ? AND id != ?", (group_id, exclude_id))
        rel_ids = [int(r["id"]) for r in cur.fetchall()]
        if not rel_ids:
            return []
        batch = self.get_designs_batch(rel_ids)
        return list(batch.values())

    def record_error(self, pdf_path: str, error_message: str) -> None:
        """Logs an indexing error into index_errors table."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO index_errors (pdf_path, error_message) VALUES (?, ?)",
                (str(pdf_path), str(error_message)),
            )
            self.conn.commit()

    def record_feedback(self, query_hash: str, design_id: int, correct: bool) -> None:
        """Records user match feedback for accuracy reporting."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO feedback (query_hash, design_id, correct) VALUES (?, ?, ?)",
                (str(query_hash), int(design_id), 1 if correct else 0),
            )
            self.conn.commit()
