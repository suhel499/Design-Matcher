"""
Server-side cryptographic helpers for generating signed license files.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    from app.core.licensing.crypto import (
        canonical_json_bytes,
        load_private_key_from_pem,
        sign_payload,
    )
except ImportError:
    from design_matcher.app.core.licensing.crypto import (
        canonical_json_bytes,
        load_private_key_from_pem,
        sign_payload,
    )
from licensing_server.config import settings


def generate_license_key() -> str:
    """
    Generates a cryptographically strong, human-readable license key.
    Format: DM-YYYY-XXXX-XXXX-XXXX
    """
    year = datetime.now(timezone.utc).year
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # omit ambiguous O, 0, 1, I
    c1 = "".join(secrets.choice(chars) for _ in range(4))
    c2 = "".join(secrets.choice(chars) for _ in range(4))
    c3 = "".join(secrets.choice(chars) for _ in range(4))
    return f"DM-{year}-{c1}-{c2}-{c3}"


def build_and_sign_license_payload(
    license_id: str,
    license_key: str,
    customer_id: str,
    customer_name: str,
    product_code: str,
    plan_type: str,
    issued_at: datetime,
    expires_at: datetime,
    max_activations: int,
    status: str,
    machine_fingerprint: Optional[str] = None,
    features: Optional[list[str]] = None,
    nonce: Optional[str] = None,
    private_key_pem: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds the authoritative canonical license payload and signs it using the Ed25519 private key.
    Returns:
        Dict[str, Any]: The complete SignedLicenseFile structure ready to deliver to the client.
    """
    if features is None:
        features = ["dxf_matching", "pdf_catalog", "code_search", "dxf_export"]

    priv_pem = private_key_pem or settings.LICENSE_PRIVATE_KEY_PEM

    payload: Dict[str, Any] = {
        "license_id": license_id,
        "license_key": license_key,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "product_id": product_code,
        "plan_type": plan_type,
        "issued_at": issued_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "max_activations": max_activations,
        "status": status,
        "hardware_binding": machine_fingerprint,
        "features": features,
        "nonce": nonce or secrets.token_hex(16),
    }

    sig_b64 = sign_payload(payload, priv_pem)

    signed_file = {
        "version": 1,
        "algorithm": "Ed25519",
        "payload": payload,
        "signature": sig_b64,
    }
    return signed_file
