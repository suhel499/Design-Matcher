"""
Database layer for Design Matcher.
Manages metadata storage in MySQL (production/normalized) and SQLite (fallback/testing),
mapping between vector IDs in FAISS and tile catalog details.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional, overload

logger = logging.getLogger(__name__)

try:
    import pymysql
    from pymysql.cursors import DictCursor
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False


class RowDict(dict):
    """
    Dictionary subclass providing both key-based and 0-indexed lookup,
    ensuring 100% interoperability between MySQL query results and sqlite3.Row.
    """
    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, int):
            return list(self.values())[item]
        return super().__getitem__(item)


if PYMYSQL_AVAILABLE:
    class RowDictCursor(DictCursor):
        dict_type = RowDict
else:
    RowDictCursor = None  # type: ignore


class MySQLConnectionWrapper:
    """Thread-safe context manager wrapping PyMySQL connections."""
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def __enter__(self) -> MySQLConnectionWrapper:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type:
                self._conn.rollback()
            else:
                self._conn.commit()
        finally:
            self._conn.close()

    @staticmethod
    def _translate_query(sql: str) -> str:
        # Convert ? placeholders to %s for PyMySQL parameter binding
        translated = sql.replace("?", "%s")
        # Translate SQLite upsert for indexed_files
        if "ON CONFLICT(pdf_path) DO UPDATE SET" in translated:
            translated = re.sub(
                r"ON CONFLICT\(pdf_path\) DO UPDATE SET.*",
                "ON DUPLICATE KEY UPDATE folder = VALUES(folder), file_hash = VALUES(file_hash), file_size = VALUES(file_size), last_modified = VALUES(last_modified), page_count = VALUES(page_count), tile_count = VALUES(tile_count), indexed_at = CURRENT_TIMESTAMP;",
                translated,
                flags=re.DOTALL,
            )
        # Translate SQLite upsert for settings
        elif "ON CONFLICT(key) DO UPDATE SET" in translated or "ON CONFLICT(`key`)" in translated:
            translated = "INSERT INTO settings (`key`, `value`) VALUES (%s, %s) ON DUPLICATE KEY UPDATE `value` = VALUES(`value`);"
        # Escape MySQL reserved keyword `key`
        translated = re.sub(r"\bsettings\s*\(\s*key\s*,", "settings (`key`,", translated)
        translated = re.sub(r"\bWHERE\s+key\s*=", "WHERE `key` =", translated)
        translated = re.sub(r"\bSELECT\s+value\s+FROM\s+settings", "SELECT `value` FROM settings", translated)
        return translated

    def execute(self, sql: str, params: Any = None) -> Any:
        cur = self._conn.cursor()
        t_sql = self._translate_query(sql)
        cur.execute(t_sql, params or ())
        return cur

    def cursor(self) -> Any:
        cur = self._conn.cursor()
        orig_execute = cur.execute
        def wrapped_execute(sql: str, params: Any = None) -> Any:
            return orig_execute(self._translate_query(sql), params or ())
        cur.execute = wrapped_execute
        return cur

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()


class CatalogDatabase:
    """Manages metadata storage (MySQL primary, SQLite fallback) for indexed catalog PDFs, pages, and tiles."""

    def __init__(self, db_path: Optional[Path | str] = None, use_mysql: Optional[bool] = None) -> None:
        if db_path is None:
            try:
                from app.core.licensing.hardware import get_secure_app_data_dir
                db_path = get_secure_app_data_dir() / "catalog_metadata.db"
            except Exception:
                try:
                    from design_matcher.app.core.licensing.hardware import get_secure_app_data_dir
                    db_path = get_secure_app_data_dir() / "catalog_metadata.db"
                except Exception:
                    db_path = Path.home() / ".design_matcher" / "catalog_metadata.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if use_mysql is not None:
            self.use_mysql = use_mysql
        elif "PYTEST_CURRENT_TEST" in os.environ and "USE_MYSQL" not in os.environ:
            self.use_mysql = False
        else:
            self.use_mysql = os.environ.get("USE_MYSQL", "true").lower() in ("true", "1", "yes")

        self.mysql_config = {
            "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
            "port": int(os.environ.get("MYSQL_PORT", "3306")),
            "user": os.environ.get("MYSQL_USER", "root"),
            "password": os.environ.get("MYSQL_PASSWORD", "root@123"),
            "database": os.environ.get("MYSQL_DATABASE", "design_matcher_catalog"),
            "charset": "utf8mb4",
            "autocommit": False,
        }

        self.is_mysql = False
        if self.use_mysql and PYMYSQL_AVAILABLE:
            try:
                conn = self._create_raw_mysql_connection()
                conn.close()
                self.is_mysql = True
                logger.info("CatalogDatabase initialized with MySQL (%s:%d/%s)",
                            self.mysql_config["host"], self.mysql_config["port"], self.mysql_config["database"])
            except Exception as exc:
                logger.warning("Could not connect to MySQL (%s). Falling back to SQLite: %s", exc, self.db_path)
                self.is_mysql = False

        self.init_db()

    def _create_raw_mysql_connection(self) -> Any:
        return pymysql.connect(
            host=self.mysql_config["host"],
            port=self.mysql_config["port"],
            user=self.mysql_config["user"],
            password=self.mysql_config["password"],
            database=self.mysql_config["database"],
            charset=self.mysql_config["charset"],
            cursorclass=RowDictCursor,
            autocommit=self.mysql_config["autocommit"],
        )

    def _get_connection(self) -> Any:
        """Returns a connection configured with row factory/dict cursor and foreign keys."""
        if self.is_mysql:
            conn = self._create_raw_mysql_connection()
            return MySQLConnectionWrapper(conn)
        else:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            return conn

    def init_db(self) -> None:
        """Initializes tables and indices if they do not already exist."""
        if self.is_mysql:
            self._init_mysql_db()
        else:
            self._init_sqlite_db()

    def _init_mysql_db(self) -> None:
        """Creates MySQL tables with normalized schemas, InnoDB engine, and utf8mb4 charset."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    pdf_path VARCHAR(700) PRIMARY KEY,
                    folder VARCHAR(500) NOT NULL,
                    file_hash VARCHAR(128) NOT NULL,
                    file_size BIGINT NOT NULL,
                    last_modified DOUBLE NOT NULL,
                    page_count INT NOT NULL DEFAULT 0,
                    tile_count INT NOT NULL DEFAULT 0,
                    indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_indexed_hash (file_hash)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tiles (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    pdf_path VARCHAR(700) NOT NULL,
                    folder VARCHAR(500) NOT NULL,
                    page_number INT NOT NULL,
                    tile_index INT NOT NULL,
                    bbox TEXT NOT NULL,
                    crop_type VARCHAR(64) NOT NULL DEFAULT 'full',
                    design_number VARCHAR(255),
                    thumbnail_path VARCHAR(700),
                    file_hash VARCHAR(128) NOT NULL,
                    source_type VARCHAR(64) DEFAULT 'pdf',
                    design_code VARCHAR(255),
                    dxf_bbox TEXT,
                    polarity_variant INT DEFAULT 0,
                    entity_count INT DEFAULT 0,
                    design_index_info TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_tiles_pdf_path (pdf_path),
                    INDEX idx_tiles_file_hash (file_hash),
                    INDEX idx_tiles_design_code (design_code),
                    CONSTRAINT fk_tiles_pdf_path FOREIGN KEY (pdf_path) REFERENCES indexed_files (pdf_path) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    `key` VARCHAR(191) PRIMARY KEY,
                    `value` LONGTEXT NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS search_history (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    timestamp VARCHAR(64) NOT NULL,
                    search_mode VARCHAR(64) NOT NULL,
                    query_text TEXT,
                    thumbnail_b64 LONGTEXT,
                    results_count INT DEFAULT 0,
                    top_match_code VARCHAR(255),
                    top_match_score DOUBLE DEFAULT 0.0,
                    INDEX idx_search_history_timestamp (timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS contact_submissions (
                    id VARCHAR(64) PRIMARY KEY,
                    name VARCHAR(150) NOT NULL,
                    email VARCHAR(255) NOT NULL,
                    subject VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'NEW',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_contact_status (status),
                    INDEX idx_contact_created (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """
            )
            # Auto-sync from SQLite if MySQL tables are empty and SQLite database has indexed data
            if self.db_path.exists():
                try:
                    cursor.execute("SELECT COUNT(*) FROM tiles;")
                    res = cursor.fetchone()
                    m_count = list(res.values())[0] if isinstance(res, dict) else res[0]
                    if m_count == 0:
                        self._sync_from_sqlite_to_mysql(cursor)
                except Exception as sync_err:
                    logger.warning("Could not auto-sync SQLite data to MySQL: %s", sync_err)

            conn.commit()
            logger.debug("MySQL Catalog database initialized.")

    def _sync_from_sqlite_to_mysql(self, mcur: Any) -> None:
        """Copies existing SQLite metadata into MySQL on first boot."""
        try:
            sconn = sqlite3.connect(str(self.db_path))
            sconn.row_factory = sqlite3.Row
            scur = sconn.cursor()

            # Copy settings
            scur.execute("SELECT key, value FROM settings;")
            for r in scur.fetchall():
                mcur.execute(
                    "INSERT INTO settings (`key`, `value`) VALUES (%s, %s) ON DUPLICATE KEY UPDATE `value`=VALUES(`value`);",
                    (r["key"], r["value"]),
                )

            # Copy indexed_files
            scur.execute("SELECT pdf_path, folder, file_hash, file_size, last_modified, page_count, tile_count, indexed_at FROM indexed_files;")
            for r in scur.fetchall():
                mcur.execute(
                    """
                    INSERT INTO indexed_files (pdf_path, folder, file_hash, file_size, last_modified, page_count, tile_count, indexed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE folder=VALUES(folder), file_hash=VALUES(file_hash);
                    """,
                    (r["pdf_path"], r["folder"], r["file_hash"], r["file_size"], r["last_modified"], r["page_count"], r["tile_count"], r["indexed_at"]),
                )

            # Copy tiles
            scur.execute("SELECT id, pdf_path, folder, page_number, tile_index, bbox, crop_type, design_number, thumbnail_path, file_hash, source_type, design_code, dxf_bbox, polarity_variant, entity_count, design_index_info FROM tiles;")
            tiles = scur.fetchall()
            if tiles:
                batch = []
                for r in tiles:
                    batch.append((
                        r["id"], r["pdf_path"], r["folder"], r["page_number"], r["tile_index"],
                        r["bbox"], r["crop_type"], r["design_number"], r["thumbnail_path"],
                        r["file_hash"], r["source_type"] or "pdf", r["design_code"],
                        r["dxf_bbox"], r["polarity_variant"] or 0, r["entity_count"] or 0,
                        r["design_index_info"]
                    ))
                mcur.executemany(
                    """
                    INSERT INTO tiles (id, pdf_path, folder, page_number, tile_index, bbox, crop_type, design_number, thumbnail_path, file_hash, source_type, design_code, dxf_bbox, polarity_variant, entity_count, design_index_info)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE pdf_path=VALUES(pdf_path);
                    """,
                    batch,
                )
            sconn.close()
            logger.info("Auto-synced %d tiles and metadata from SQLite to MySQL.", len(tiles) if tiles else 0)
        except Exception as exc:
            logger.warning("Error during SQLite to MySQL sync: %s", exc)

    def _init_sqlite_db(self) -> None:
        """Initializes SQLite fallback tables and indices."""
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

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS contact_submissions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'NEW',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_contact_status ON contact_submissions(status);")

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
            if isinstance(data.get("bbox"), str):
                try:
                    data["bbox"] = json.loads(data["bbox"])
                except Exception:
                    pass
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
                    if isinstance(item.get("bbox"), str):
                        try:
                            item["bbox"] = json.loads(item["bbox"])
                        except Exception:
                            pass
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
                if isinstance(row_dict.get("bbox"), str):
                    try:
                        row_dict["bbox"] = json.loads(row_dict["bbox"])
                    except Exception:
                        pass
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

    def save_contact_submission(
        self,
        name: str,
        email: str,
        subject: str,
        message: str,
    ) -> str:
        """
        Saves a contact query or inquiry to the database using parameterized queries.
        Returns the generated unique submission ID.
        """
        import uuid
        submission_id = str(uuid.uuid4())
        clean_name = name.strip()
        clean_email = email.strip().lower()
        clean_subject = subject.strip()
        clean_msg = message.strip()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO contact_submissions (id, name, email, subject, message, status)
                VALUES (?, ?, ?, ?, ?, 'NEW')
                """,
                (submission_id, clean_name, clean_email, clean_subject, clean_msg),
            )
            conn.commit()

        # If connected to MySQL, also guarantee persistence in the central server authority database
        if getattr(self, "is_mysql", False):
            try:
                raw_conn = self._create_raw_mysql_connection()
                with raw_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO design_matcher_licensing.contact_submissions
                            (id, name, email, subject, message, status, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, 'NEW', NOW(), NOW())
                        ON DUPLICATE KEY UPDATE status = 'NEW', updated_at = NOW()
                        """,
                        (submission_id, clean_name, clean_email, clean_subject, clean_msg),
                    )
                raw_conn.commit()
                raw_conn.close()
                logger.info("Successfully persisted contact submission %s into server DB (design_matcher_licensing)", submission_id)
            except Exception as exc:
                logger.warning("Could not mirror contact submission to server DB: %s", exc)

        return submission_id

    def get_contact_submissions(self, status: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieves saved contact queries, optionally filtered by status ('NEW', 'IN_PROGRESS', 'RESOLVED')."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM contact_submissions WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status.upper(), limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM contact_submissions ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def update_contact_submission_status(self, submission_id: str, status: str) -> None:
        """Updates the status of a contact submission."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE contact_submissions SET status = ? WHERE id = ?",
                (status.upper(), submission_id),
            )
            conn.commit()


