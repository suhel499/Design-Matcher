"""
Database layer for Design Matcher.
Manages metadata storage in SQLite, mapping between vector IDs in FAISS and tile catalog details.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional, overload

logger = logging.getLogger(__name__)


class CatalogDatabase:
    """Manages SQLite metadata storage for indexed catalog PDFs, pages, and tiles."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection configured with row factory and foreign keys."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def init_db(self) -> None:
        """Initializes tables and indices if they do not already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Table for tracking indexed PDF files
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    pdf_path TEXT PRIMARY KEY,
                    folder TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    last_modified REAL NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    tile_count INTEGER NOT NULL DEFAULT 0,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # Table for individual design tiles and sub-regions
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_path TEXT NOT NULL,
                    folder TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    tile_index INTEGER NOT NULL,
                    bbox TEXT NOT NULL,
                    crop_type TEXT NOT NULL DEFAULT 'full',
                    design_number TEXT,
                    thumbnail_path TEXT,
                    file_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pdf_path) REFERENCES indexed_files (pdf_path) ON DELETE CASCADE
                );
                """
            )

            # Indexes for high performance lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_pdf_path ON tiles (pdf_path);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_file_hash ON tiles (file_hash);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexed_hash ON indexed_files (file_hash);")

            # App configuration table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

            # Search history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    search_mode TEXT NOT NULL,
                    query_text TEXT,
                    thumbnail_b64 TEXT,
                    results_count INTEGER DEFAULT 0,
                    top_match_code TEXT,
                    top_match_score REAL DEFAULT 0.0
                );
                """
            )

            # Migration check for DXF support columns on tiles table
            cursor.execute("PRAGMA table_info(tiles);")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            for col_name, col_def in [
                ("source_type", "TEXT DEFAULT 'pdf'"),
                ("design_code", "TEXT"),
                ("dxf_bbox", "TEXT"),
                ("polarity_variant", "INTEGER DEFAULT 0"),
                ("entity_count", "INTEGER DEFAULT 0"),
                ("design_index_info", "TEXT"),
            ]:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE tiles ADD COLUMN {col_name} {col_def};")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_design_code ON tiles(design_code);")
            conn.commit()
            logger.debug("Database initialized at %s", self.db_path)

    def get_indexed_file(self, pdf_path: str | Path) -> Optional[dict[str, Any]]:
        """Retrieves an indexed file record by path."""
        norm_path = str(Path(pdf_path).resolve())
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM indexed_files WHERE pdf_path = ?", (norm_path,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_indexed_files(self) -> list[dict[str, Any]]:
        """Retrieves all indexed PDF records."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM indexed_files ORDER BY pdf_path").fetchall()
            return [dict(r) for r in rows]

    def record_indexed_file(
        self,
        pdf_path: str | Path,
        folder: str,
        file_hash: str,
        file_size: int,
        last_modified: float,
        page_count: int,
        tile_count: int,
    ) -> None:
        """Inserts or updates an indexed file record."""
        norm_path = str(Path(pdf_path).resolve())
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO indexed_files (
                    pdf_path, folder, file_hash, file_size, last_modified, page_count, tile_count, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(pdf_path) DO UPDATE SET
                    folder = excluded.folder,
                    file_hash = excluded.file_hash,
                    file_size = excluded.file_size,
                    last_modified = excluded.last_modified,
                    page_count = excluded.page_count,
                    tile_count = excluded.tile_count,
                    indexed_at = CURRENT_TIMESTAMP;
                """,
                (norm_path, folder, file_hash, file_size, last_modified, page_count, tile_count),
            )
            conn.commit()

    def insert_tile(
        self,
        pdf_path: str | Path,
        folder: str,
        page_number: int,
        tile_index: int,
        bbox: tuple[int, int, int, int] | list[int],
        crop_type: str = "full",
        design_number: Optional[str] = None,
        thumbnail_path: Optional[str] = None,
        file_hash: str = "",
    ) -> int:
        """
        Inserts a tile record and returns its auto-assigned ID.
        This integer ID matches the vector ID in FAISS.
        """
        norm_path = str(Path(pdf_path).resolve())
        bbox_json = json.dumps(list(bbox))
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tiles (
                    pdf_path, folder, page_number, tile_index, bbox,
                    crop_type, design_number, thumbnail_path, file_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    norm_path,
                    folder,
                    page_number,
                    tile_index,
                    bbox_json,
                    crop_type,
                    design_number,
                    thumbnail_path,
                    file_hash,
                ),
            )
            conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("INSERT into tiles did not return a lastrowid")
            return int(cursor.lastrowid)

    def insert_tiles_batch(self, tiles_data: list[dict[str, Any]]) -> list[int]:
        """
        Inserts multiple tile records in a single transaction.
        Returns the list of generated tile IDs in corresponding order.
        """
        if not tiles_data:
            return []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            ids = []
            for item in tiles_data:
                norm_path = str(Path(item["pdf_path"]).resolve())
                bbox_json = json.dumps(list(item["bbox"]))
                dxf_b = item.get("dxf_bbox")
                dxf_b_json = json.dumps(list(dxf_b)) if isinstance(dxf_b, (list, tuple)) else dxf_b
                cursor.execute(
                    """
                    INSERT INTO tiles (
                        pdf_path, folder, page_number, tile_index, bbox,
                        crop_type, design_number, thumbnail_path, file_hash,
                        source_type, design_code, dxf_bbox, polarity_variant, entity_count, design_index_info
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        norm_path,
                        item.get("folder", ""),
                        item["page_number"],
                        item["tile_index"],
                        bbox_json,
                        item.get("crop_type", "full"),
                        item.get("design_number"),
                        item.get("thumbnail_path"),
                        item.get("file_hash", ""),
                        item.get("source_type", "pdf"),
                        item.get("design_code") or item.get("design_number"),
                        dxf_b_json,
                        int(item.get("polarity_variant", 0)),
                        int(item.get("entity_count", 0)),
                        str(item.get("design_index_info", "")),
                    ),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("INSERT into tiles did not return a lastrowid")
                ids.append(int(cursor.lastrowid))
            conn.commit()
            return ids

    def delete_file_records(self, pdf_path: str | Path) -> list[int]:
        """
        Deletes all tile records and the file entry for the given PDF path.
        Returns the list of deleted tile IDs so they can be removed from FAISS.
        """
        norm_path = str(Path(pdf_path).resolve())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Fetch IDs before deletion
            cursor.execute("SELECT id FROM tiles WHERE pdf_path = ?", (norm_path,))
            deleted_ids = [int(row[0]) for row in cursor.fetchall()]

            cursor.execute("DELETE FROM tiles WHERE pdf_path = ?", (norm_path,))
            cursor.execute("DELETE FROM indexed_files WHERE pdf_path = ?", (norm_path,))
            conn.commit()
            return deleted_ids

    def get_tile(self, tile_id: int) -> Optional[dict[str, Any]]:
        """Retrieves tile metadata by its ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM tiles WHERE id = ?", (tile_id,)).fetchone()
            if not row:
                return None
            data = dict(row)
            data["bbox"] = json.loads(data["bbox"])
            return data

    def get_tiles_by_ids(self, tile_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Retrieves multiple tile records keyed by tile ID."""
        if not tile_ids:
            return {}
        results: dict[int, dict[str, Any]] = {}
        chunk_size = 500
        with self._get_connection() as conn:
            for i in range(0, len(tile_ids), chunk_size):
                chunk = tile_ids[i : i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT * FROM tiles WHERE id IN ({placeholders})", chunk
                ).fetchall()
                for r in rows:
                    item = dict(r)
                    item["bbox"] = json.loads(item["bbox"])
                    results[item["id"]] = item
        return results

    def get_total_tile_count(self) -> int:
        """Returns the total number of tiles indexed."""
        with self._get_connection() as conn:
            val = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()
            return int(val[0]) if val else 0

    def get_total_file_count(self) -> int:
        """Returns the total number of PDF files indexed."""
        with self._get_connection() as conn:
            val = conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()
            return int(val[0]) if val else 0

    def get_all_tile_ids(self) -> list[int]:
        """Returns all tile IDs currently in the database."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id FROM tiles ORDER BY id").fetchall()
            return [int(r[0]) for r in rows]

    def cleanup_missing_files(self, existing_paths: set[str]) -> tuple[list[str], list[int]]:
        """
        Removes records for files that no longer exist on disk.
        Returns:
            (deleted_pdf_paths, deleted_tile_ids)
        """
        all_indexed = self.get_all_indexed_files()
        deleted_paths: list[str] = []
        all_deleted_tile_ids: list[int] = []

        for record in all_indexed:
            path_str = record["pdf_path"]
            if path_str not in existing_paths:
                tile_ids = self.delete_file_records(path_str)
                deleted_paths.append(path_str)
                all_deleted_tile_ids.extend(tile_ids)

        return deleted_paths, all_deleted_tile_ids

    @overload
    def get_setting(self, key: str, default: str) -> str: ...

    @overload
    def get_setting(self, key: str, default: None = None) -> Optional[str]: ...

    @overload
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]: ...

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieves an application setting value."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            if row is not None and row[0] is not None:
                return str(row[0])
            return default

    def set_setting(self, key: str, value: str) -> None:
        """Sets or updates an application setting value."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
            conn.commit()

    def search_by_code(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """
        Performs case-insensitive, space-insensitive exact or prefix match on design codes.
        Returns matched tile dictionaries ordered by match quality.
        """
        import re

        if not query or not query.strip():
            return []
        clean_q = re.sub(r"[\s\-_#]", "", query).lower()
        if not clean_q:
            return []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tiles
                WHERE (design_code IS NOT NULL AND TRIM(design_code) != '')
                   OR (design_number IS NOT NULL AND TRIM(design_number) != '')
            """)
            matches = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict["bbox"] = json.loads(row_dict["bbox"])
                d_code = str(row_dict.get("design_code") or row_dict.get("design_number") or "")
                clean_c = re.sub(r"[\s\-_#]", "", d_code).lower()
                if not clean_c:
                    continue

                score = 0
                if clean_c == clean_q:
                    score = 100
                elif clean_c.startswith(clean_q):
                    score = 80
                elif clean_q in clean_c:
                    score = 60

                if score > 0:
                    matches.append((score, row_dict))

            matches.sort(key=lambda item: item[0], reverse=True)
            return [m[1] for m in matches[:limit]]

    def add_search_history(
        self,
        search_mode: str,
        query_text: str = "",
        thumbnail_b64: Optional[str] = None,
        results_count: int = 0,
        top_match_code: Optional[str] = None,
        top_match_score: float = 0.0,
    ) -> int:
        """Records a completed search into search history."""
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO search_history (timestamp, search_mode, query_text, thumbnail_b64, results_count, top_match_code, top_match_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, search_mode, query_text, thumbnail_b64, results_count, top_match_code, top_match_score),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def get_search_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieves past searches ordered by newest first."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM search_history ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def delete_search_history(self, entry_id: int) -> None:
        """Deletes a single search history item by ID."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM search_history WHERE id = ?", (entry_id,))
            conn.commit()

    def clear_all_search_history(self) -> None:
        """Purges all records from search history."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM search_history")
            conn.commit()

