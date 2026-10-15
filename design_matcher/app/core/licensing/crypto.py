"""
Ed25519 Cryptographic Utilities for Design Matcher Licensing.
Provides keypair generation, canonical JSON serialization, digital signing,
and public key verification for online and offline licenses.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger(__name__)


def generate_keypair() -> Tuple[str, str]:
    """
    Generates a new Ed25519 keypair.
    Returns:
        Tuple[str, str]: (private_key_pem, public_key_pem) as UTF-8 strings.
    """
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")

    return priv_pem, pub_pem


def load_private_key_from_pem(pem_str: str) -> ed25519.Ed25519PrivateKey:
    """Loads an Ed25519 private key from PEM formatted text."""
    key = serialization.load_pem_private_key(pem_str.encode("utf-8"), password=None)
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise ValueError("Key in PEM is not an Ed25519 private key")
    return key


def load_public_key_from_pem(pem_str: str) -> ed25519.Ed25519PublicKey:
    """Loads an Ed25519 public key from PEM formatted text."""
    key = serialization.load_pem_public_key(pem_str.encode("utf-8"))
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise ValueError("Key in PEM is not an Ed25519 public key")
    return key


def canonical_json_bytes(data: Dict[str, Any]) -> bytes:
    """
    Serializes a dictionary into canonical deterministic JSON bytes.
    Keys are strictly sorted with compact delimiters and UTF-8 encoding.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_payload(payload: Dict[str, Any], private_key: ed25519.Ed25519PrivateKey | str) -> str:
    """
    Digitally signs a payload dictionary with an Ed25519 private key.
    Returns the signature encoded as a standard base64 string.
    """
    if isinstance(private_key, str):
        private_key = load_private_key_from_pem(private_key)

    canonical_bytes = canonical_json_bytes(payload)
    signature_bytes = private_key.sign(canonical_bytes)
    return base64.b64encode(signature_bytes).decode("ascii")


def verify_payload(payload: Dict[str, Any], signature_b64: str, public_key: ed25519.Ed25519PublicKey | str) -> bool:
    """
    Verifies that the given payload matches the digital signature under the public key.
    Returns True if valid, False if signature is invalid or payload has been tampered with.
    """
    try:
        if isinstance(public_key, str):
            public_key = load_public_key_from_pem(public_key)

        canonical_bytes = canonical_json_bytes(payload)
        signature_bytes = base64.b64decode(signature_b64.encode("ascii"))
        public_key.verify(signature_bytes, canonical_bytes)
        return True
    except (InvalidSignature, ValueError, Exception) as exc:
        logger.warning("License signature verification failed: %s", exc)
        return False
