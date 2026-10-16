"""
Configuration settings for the Design Matcher Licensing Server.
Loads environment variables with safe defaults for development and production.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field

_DEFAULT_DB_FILE = Path(__file__).resolve().parent.parent / "licensing.db"


class Settings(BaseModel):
    # Server Info
    APP_NAME: str = "Design Matcher License Authority"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Database: PostgreSQL for production, SQLite for local testing
    DATABASE_URL: str = Field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL",
            f"sqlite:///{_DEFAULT_DB_FILE}",
        )
    )

    # Cryptographic keys
    # Ed25519 Private key in PEM format (authoritative master signing key)
    LICENSE_PRIVATE_KEY_PEM: str = Field(
        default_factory=lambda: os.environ.get(
            "LICENSE_PRIVATE_KEY_PEM",
            """-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIE2rNittAgKnTeO5ygvgWDHRCPwe5NztKkZj0TNpXNdc
-----END PRIVATE KEY-----""",
        )
    )

    # Ed25519 Public key in PEM format
    LICENSE_PUBLIC_KEY_PEM: str = Field(
        default_factory=lambda: os.environ.get(
            "LICENSE_PUBLIC_KEY_PEM",
            """-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAbbgL6nULdFQo/qmy+Si/t/fa2F/+jBX2oQ5bkjlSjOA=
-----END PUBLIC KEY-----""",
        )
    )

    # JWT Authentication for Admin API
    JWT_SECRET: str = Field(
        default_factory=lambda: os.environ.get(
            "JWT_SECRET", "change-this-in-production-super-secret-jwt-key-2026"
        )
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Product Identifier
    DEFAULT_PRODUCT_CODE: str = "design-matcher"
    DEFAULT_PRODUCT_NAME: str = "Laser Cut Design Matcher Desktop"

    # Default Admin User
    INITIAL_ADMIN_USERNAME: str = Field(
        default_factory=lambda: os.environ.get("INITIAL_ADMIN_USERNAME", "admin")
    )
    INITIAL_ADMIN_PASSWORD: str = Field(
        default_factory=lambda: os.environ.get("INITIAL_ADMIN_PASSWORD", "AdminPass@2026")
    )
    INITIAL_ADMIN_EMAIL: str = Field(
        default_factory=lambda: os.environ.get("INITIAL_ADMIN_EMAIL", "admin@designmatcher.local")
    )


settings = Settings()
