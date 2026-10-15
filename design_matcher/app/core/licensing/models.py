"""
Data models and enumerations for Design Matcher Licensing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PlanType(str, Enum):
    ONE_YEAR = "1_year"
    THREE_YEARS = "3_years"
    FIVE_YEARS = "5_years"
    CUSTOM = "custom"


class LicenseStatus(str, Enum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    SUSPENDED = "SUSPENDED"
    REVOKED = "REVOKED"


class LicensePayload(BaseModel):
    """Authoritative digital payload signed with Ed25519."""
    license_id: str
    license_key: str
    customer_id: str
    customer_name: str
    product_id: str = "design-matcher"
    plan_type: str = PlanType.ONE_YEAR.value
    issued_at: str  # ISO 8601 UTC
    expires_at: str  # ISO 8601 UTC
    max_activations: int = 1
    status: str = LicenseStatus.ACTIVE.value
    hardware_binding: Optional[str] = None  # Bound machine fingerprint
    features: List[str] = Field(default_factory=lambda: ["dxf_matching", "pdf_catalog", "code_search", "dxf_export"])
    nonce: str


class SignedLicenseFile(BaseModel):
    """Complete signed license file containing canonical payload and Ed25519 signature."""
    version: int = 1
    algorithm: str = "Ed25519"
    payload: Dict[str, Any]
    signature: str  # Base64 string


class OfflineActivationRequest(BaseModel):
    """Payload exported by client machine for air-gapped activation."""
    version: int = 1
    license_key: str
    machine_fingerprint: str
    machine_name: str
    os_info: str
    request_timestamp: str
    request_nonce: str


class OfflineActivationResponse(BaseModel):
    """Payload returned by administrator to air-gapped client."""
    version: int = 1
    license_file: SignedLicenseFile


@dataclass
class LicenseCheckResult:
    """Detailed verdict produced by LicenseManager verification."""
    is_valid: bool
    status: LicenseStatus = LicenseStatus.ACTIVE
    is_expired: bool = False
    is_device_mismatched: bool = False
    is_rollback_detected: bool = False
    is_revoked_or_suspended: bool = False
    is_grace_period_active: bool = False
    days_remaining: int = 0
    license_id: str = ""
    license_key: str = ""
    customer_name: str = ""
    expires_at: Optional[datetime] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
