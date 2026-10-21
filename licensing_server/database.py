"""
Database connection, session management, and Base declarative class for Licensing Server.
Supports PostgreSQL for production and SQLite for local development/testing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from licensing_server.config import settings

logger = logging.getLogger(__name__)

db_url = settings.DATABASE_URL
engine_args = {}

if "mysql" in db_url:
    try:
        test_args = {
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "pool_size": 2,
            "max_overflow": 0,
        }
        test_engine = create_engine(db_url, **test_args)
        with test_engine.connect():
            pass
        test_engine.dispose()
        engine_args = {
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "pool_size": 10,
            "max_overflow": 20,
        }
    except Exception as exc:
        sqlite_file = Path(__file__).resolve().parent.parent / "licensing.db"
        logger.warning(
            "Could not connect to MySQL (%s). Falling back to SQLite: %s",
            exc,
            sqlite_file,
        )
        db_url = f"sqlite:///{sqlite_file.as_posix()}"
        engine_args = {"connect_args": {"check_same_thread": False}}
elif db_url.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}

engine = create_engine(
    db_url,
    **engine_args,
    echo=settings.DEBUG,
)

# Enable foreign keys and WAL mode for SQLite
if db_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency that yields a database session and ensures clean closure."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
