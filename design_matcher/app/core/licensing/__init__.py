"""
Licensing subsystem package for Design Matcher.
"""

from app.core.licensing.crypto import (
    canonical_json_bytes,
    generate_keypair,
    load_private_key_from_pem,
    load_public_key_from_pem,
    sign_payload,
    verify_payload,
)
from app.core.licensing.hardware import get_device_info, get_hardware_fingerprint
from app.core.licensing.manager import LicenseManager
from app.core.licensing.models import (
    LicenseCheckResult,
    LicensePayload,
    LicenseStatus,
    OfflineActivationRequest,
    OfflineActivationResponse,
    PlanType,
    SignedLicenseFile,
)
from app.core.licensing.public_key import get_verification_public_key

__all__ = [
    "canonical_json_bytes",
    "generate_keypair",
    "load_private_key_from_pem",
    "load_public_key_from_pem",
    "sign_payload",
    "verify_payload",
    "get_hardware_fingerprint",
    "get_device_info",
    "LicenseManager",
    "LicenseCheckResult",
    "LicensePayload",
    "LicenseStatus",
    "OfflineActivationRequest",
    "OfflineActivationResponse",
    "PlanType",
    "SignedLicenseFile",
    "get_verification_public_key",
]
