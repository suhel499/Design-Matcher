"""
Security, password hashing, and JWT token authentication for Licensing Server.
Uses PBKDF2-HMAC-SHA256 for secure password storage and HMAC-SHA256 for JWT tokens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from licensing_server.config import settings
from licensing_server.database import get_db

security_bearer = HTTPBearer(auto_error=False)


# -----------------------------------------------------------------------------
# Password Hashing (PBKDF2-HMAC-SHA256)
# -----------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hashes a password with PBKDF2-HMAC-SHA256 and a cryptographically random salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    )
    return f"pbkdf2:sha256:100000${salt}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against the stored PBKDF2 hash."""
    try:
        parts = hashed_password.split("$")
        if len(parts) != 3:
            return False
        algo_iter, salt, stored_hash = parts
        _, _, iterations_str = algo_iter.split(":")
        iterations = int(iterations_str)

        key = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        )
        return hmac.compare_digest(key.hex(), stored_hash)
    except Exception:
        return False


# -----------------------------------------------------------------------------
# JWT Tokens (HMAC-SHA256)
# -----------------------------------------------------------------------------

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data_str: str) -> bytes:
    padding = 4 - (len(data_str) % 4)
    if padding and padding != 4:
        data_str += "=" * padding
    return base64.urlsafe_b64decode(data_str.encode("ascii"))


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generates an HMAC-SHA256 signed JWT token."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"iat": int(now.timestamp()), "exp": int(expire.timestamp())})

    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(to_encode, separators=(",", ":")).encode("utf-8"))

    msg = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(settings.JWT_SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decodes and verifies a JWT token. Raises HTTPException if invalid or expired."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")

        header_b64, payload_b64, sig_b64 = parts
        msg = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(settings.JWT_SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
        actual_sig = _b64url_decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Signature verification failed")

        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        exp = payload.get("exp")
        if exp and datetime.now(timezone.utc).timestamp() > exp:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")

        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token validation error: {exc}")


def get_current_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    db: Session = Depends(get_db),
):
    """Dependency verifying that the request originates from an active authenticated admin user."""
    from licensing_server.models import AdminUser

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject")

    user = db.query(AdminUser).filter(AdminUser.username == username, AdminUser.is_active == True).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin user not found or inactive")

    return user
