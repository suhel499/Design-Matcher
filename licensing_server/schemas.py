"""
Pydantic schemas for the Licensing API requests and responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Customer Schemas
# -----------------------------------------------------------------------------

class CustomerCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    email: str = Field(..., min_length=5, max_length=255)
    company: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None


class CustomerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    email: str
    company: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# -----------------------------------------------------------------------------
# Product Schemas
# -----------------------------------------------------------------------------

class ProductCreate(BaseModel):
    product_code: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., min_length=2, max_length=255)
    description: Optional[str] = None


class ProductResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    product_code: str
    name: str
    description: Optional[str] = None
    is_active: bool
    created_at: datetime


# -----------------------------------------------------------------------------
# License & Subscription Schemas
# -----------------------------------------------------------------------------

class LicenseCreate(BaseModel):
    customer_id: str
    product_code: str = "design-matcher"
    plan_type: str = "1_year"  # 1_year, 3_years, 5_years, custom
    custom_starts_at: Optional[datetime] = None
    custom_expires_at: Optional[datetime] = None
    max_activations: int = Field(default=1, ge=1, le=50)
    notes: Optional[str] = None


class LicenseStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(ACTIVE|EXPIRED|SUSPENDED|REVOKED)$")
    reason: Optional[str] = None


class LicenseRenewalRequest(BaseModel):
    plan_type: str = "1_year"  # 1_year, 3_years, 5_years, custom
    duration_days: Optional[int] = None
    custom_expires_at: Optional[datetime] = None
    payment_reference: Optional[str] = None
    notes: Optional[str] = None


class LicenseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    license_key: str
    customer_id: str
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    product_code: str
    plan_type: str
    status: str
    max_activations: int
    active_device_count: int = 0
    starts_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# -----------------------------------------------------------------------------
# Client Endpoints (Activation, Validation, Deactivation)
# -----------------------------------------------------------------------------

class ClientActivationRequest(BaseModel):
    license_key: str
    machine_fingerprint: str
    machine_name: str = "Unknown-Computer"
    os_info: str = "Unknown-OS"
    nonce: str


class ClientActivationResponse(BaseModel):
    status: str
    message: str
    license_file: Dict[str, Any]


class ClientValidationRequest(BaseModel):
    license_key: str
    machine_fingerprint: str
    nonce: str


class ClientValidationResponse(BaseModel):
    status: str
    message: str
    license_file: Optional[Dict[str, Any]] = None


class ClientDeactivationRequest(BaseModel):
    license_key: str
    machine_fingerprint: str
    nonce: str


class ClientTransferRequest(BaseModel):
    license_key: str
    old_machine_fingerprint: str
    new_machine_fingerprint: str
    new_machine_name: str = "Transferred-Computer"
    reason: Optional[str] = None


# -----------------------------------------------------------------------------
# Offline Activation Schemas
# -----------------------------------------------------------------------------

class OfflineSignRequest(BaseModel):
    request_text: str  # .dmreq JSON payload text


class OfflineSignResponse(BaseModel):
    status: str
    license_key: str
    machine_fingerprint: str
    expires_at: datetime
    signed_license_file: Dict[str, Any]
    download_filename: str


# -----------------------------------------------------------------------------
# Admin & Audit Schemas
# -----------------------------------------------------------------------------

class AdminLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    actor: str
    action: str
    resource_type: str
    resource_id: str
    details_json: str
    ip_address: Optional[str] = None
    timestamp: datetime


class DeviceActivationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    license_id: str
    license_key: Optional[str] = None
    customer_name: Optional[str] = None
    machine_fingerprint: str
    machine_name: str
    os_info: str
    ip_address: Optional[str] = None
    activated_at: datetime
    last_checkin_at: datetime
    is_active: bool
    deactivated_at: Optional[datetime] = None
