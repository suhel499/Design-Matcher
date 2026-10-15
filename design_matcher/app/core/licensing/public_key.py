"""
Public verification key for Design Matcher.
This file is embedded directly into the client desktop binary.
The private signing key is NEVER embedded in client binaries or distributed to customers.
"""

from __future__ import annotations

import os

# Production Ed25519 public key (PEM format)
DEFAULT_EMBEDDED_PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAbbgL6nULdFQo/qmy+Si/t/fa2F/+jBX2oQ5bkjlSjOA=
-----END PUBLIC KEY-----
"""


def get_verification_public_key() -> str:
    """
    Returns the active Ed25519 public key for license verification.
    Can be overridden via DESIGN_MATCHER_PUBLIC_KEY for local test fixtures.
    """
    override = os.environ.get("DESIGN_MATCHER_PUBLIC_KEY")
    if override and override.strip():
        return override.strip()
    return DEFAULT_EMBEDDED_PUBLIC_KEY_PEM.strip()
