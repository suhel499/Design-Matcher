"""
Administrative API endpoints for customer profiles, license generation,
plan renewals, air-gapped offline signing, device management, and audit reporting.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from licensing_server.config import settings
from licensing_server.crypto import (
    build_and_sign_license_payload,
    generate_license_key,
)
from licensing_server.database import get_db
from licensing_server.models import (
    AdminUser,
    AuditLog,
    Customer,
    DeviceActivation,
    License,
    OfflineLicenseFile,
    Product,
    RenewalTransaction,
    Subscription,
    ensure_utc,
    utc_now,
)
from licensing_server.schemas import (
    AdminLoginRequest,
    AuditLogResponse,
    CustomerCreate,
    CustomerResponse,
    DeviceActivationResponse,
    LicenseCreate,
    LicenseRenewalRequest,
    LicenseResponse,
    LicenseStatusUpdate,
    OfflineSignRequest,
    OfflineSignResponse,
    ProductCreate,
    ProductResponse,
    TokenResponse,
)
from licensing_server.security import (
    create_access_token,
    get_current_admin,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin Licensing"])


def calculate_plan_duration(plan_type: str, starts_at: datetime) -> Tuple[datetime, int]:
    """Calculates expiration date and duration in days based on subscription plan."""
    if plan_type == "1_year":
        days = 365
        return starts_at + timedelta(days=days), days
    elif plan_type == "3_years":
        days = 365 * 3
        return starts_at + timedelta(days=days), days
    elif plan_type == "5_years":
        days = 365 * 5 + 1  # accounts for leap year
        return starts_at + timedelta(days=days), days
    else:
        # Default fallback
        days = 365
        return starts_at + timedelta(days=days), days


# -----------------------------------------------------------------------------
# Admin Authentication
# -----------------------------------------------------------------------------

@router.post("/auth/login", response_model=TokenResponse)
def admin_login(
    req: AdminLoginRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Authenticates an administrator and returns a JWT access token."""
    user = db.query(AdminUser).filter(AdminUser.username == req.username).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect admin username or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin account is deactivated",
        )

    token = create_access_token({"sub": user.username, "role": user.role})

    # Log audit
    audit = AuditLog(
        actor=user.username,
        action="ADMIN_LOGIN",
        resource_type="admin_user",
        resource_id=user.id,
        details_json=json.dumps({"ip": request.client.host if request.client else None}),
        ip_address=request.client.host if request.client else None,
        timestamp=utc_now(),
    )
    db.add(audit)
    db.commit()

    return TokenResponse(
        access_token=token,
        username=user.username,
        role=user.role,
    )


# -----------------------------------------------------------------------------
# Customers Management
# -----------------------------------------------------------------------------

@router.get("/customers", response_model=List[CustomerResponse])
def list_customers(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """Lists customers with optional substring search by name, email, or company."""
    query = db.query(Customer)
    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(
            (Customer.name.ilike(term)) | (Customer.email.ilike(term)) | (Customer.company.ilike(term))
        )
    return query.order_by(Customer.created_at.desc()).offset(skip).limit(limit).all()


@router.post("/customers", response_model=CustomerResponse)
def create_customer(
    req: CustomerCreate,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """Registers a new customer profile."""
    existing = db.query(Customer).filter(Customer.email == req.email.strip().lower()).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A customer with this email already exists.")

    cust = Customer(
        name=req.name.strip(),
        email=req.email.strip().lower(),
        company=req.company.strip() if req.company else None,
        phone=req.phone.strip() if req.phone else None,
        notes=req.notes.strip() if req.notes else None,
    )
    db.add(cust)
    db.flush()

    audit = AuditLog(
        actor=admin.username,
        action="CREATE_CUSTOMER",
        resource_type="customer",
        resource_id=cust.id,
        details_json=json.dumps({"name": cust.name, "email": cust.email}),
        timestamp=utc_now(),
    )
    db.add(audit)
    db.commit()
    db.refresh(cust)
    return cust


# -----------------------------------------------------------------------------
# License Creation & Assignment
# -----------------------------------------------------------------------------

@router.post("/licenses", response_model=LicenseResponse)
def create_license(
    req: LicenseCreate,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """
    Generates a cryptographically secure license for a customer with specified duration.
    Supports 1 year, 3 years, 5 years, or custom start/expiration dates.
    """
    customer = db.query(Customer).filter(Customer.id == req.customer_id).first()
    if not customer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found.")

    product = db.query(Product).filter(Product.product_code == req.product_code).first()
    if not product:
        # Create default product if not seeded
        product = Product(
            product_code=req.product_code,
            name=settings.DEFAULT_PRODUCT_NAME,
            description="Desktop Application",
        )
        db.add(product)
        db.flush()

    now = utc_now()
    starts_at = req.custom_starts_at if (req.plan_type == "custom" and req.custom_starts_at) else now
    if req.plan_type == "custom" and req.custom_expires_at:
        expires_at = req.custom_expires_at
    else:
        expires_at, _ = calculate_plan_duration(req.plan_type, starts_at)

    if expires_at <= starts_at:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Expiration date must be after start date.")

    # Generate unique license key
    license_key = generate_license_key()
    while db.query(License).filter(License.license_key == license_key).first() is not None:
        license_key = generate_license_key()

    license_obj = License(
        license_key=license_key,
        customer_id=customer.id,
        product_id=product.id,
        plan_type=req.plan_type,
        status="ACTIVE",
        max_activations=req.max_activations,
        notes=req.notes,
    )
    db.add(license_obj)
    db.flush()

    subscription = Subscription(
        license_id=license_obj.id,
        starts_at=starts_at,
        expires_at=expires_at,
        status="ACTIVE",
        is_active=True,
    )
    db.add(subscription)

    audit = AuditLog(
        actor=admin.username,
        action="CREATE_LICENSE",
        resource_type="license",
        resource_id=license_obj.id,
        details_json=json.dumps({
            "key": license_key,
            "plan": req.plan_type,
            "starts_at": starts_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "customer": customer.name,
        }),
        timestamp=utc_now(),
    )
    db.add(audit)
    db.commit()

    return LicenseResponse(
        id=license_obj.id,
        license_key=license_obj.license_key,
        customer_id=customer.id,
        customer_name=customer.name,
        customer_email=customer.email,
        product_code=product.product_code,
        plan_type=license_obj.plan_type,
        status=license_obj.status,
        max_activations=license_obj.max_activations,
        active_device_count=0,
        starts_at=starts_at,
        expires_at=expires_at,
        notes=license_obj.notes,
        created_at=license_obj.created_at,
        updated_at=license_obj.updated_at,
    )


@router.get("/licenses", response_model=List[LicenseResponse])
def list_licenses(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None,
    customer_id: Optional[str] = None,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """Lists licenses with optional status or customer filters."""
    query = db.query(License)
    if status_filter:
        query = query.filter(License.status == status_filter.upper())
    if customer_id:
        query = query.filter(License.customer_id == customer_id)

    licenses = query.order_by(License.created_at.desc()).offset(skip).limit(limit).all()
    results = []
    for lic in licenses:
        active_sub = (
            db.query(Subscription)
            .filter(Subscription.license_id == lic.id, Subscription.is_active == True)
            .order_by(Subscription.expires_at.desc())
            .first()
        )
        active_devices = (
            db.query(DeviceActivation)
            .filter(DeviceActivation.license_id == lic.id, DeviceActivation.is_active == True)
            .count()
        )
        results.append(
            LicenseResponse(
                id=lic.id,
                license_key=lic.license_key,
                customer_id=lic.customer_id,
                customer_name=lic.customer.name if lic.customer else None,
                customer_email=lic.customer.email if lic.customer else None,
                product_code=lic.product.product_code if lic.product else "design-matcher",
                plan_type=lic.plan_type,
                status=lic.status,
                max_activations=lic.max_activations,
                active_device_count=active_devices,
                starts_at=active_sub.starts_at if active_sub else None,
                expires_at=active_sub.expires_at if active_sub else None,
                notes=lic.notes,
                created_at=lic.created_at,
                updated_at=lic.updated_at,
            )
        )
    return results


@router.get("/licenses/{license_id}")
def get_license_details(
    license_id: str,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """Returns complete details of a license including devices and renewal history."""
    lic = db.query(License).filter((License.id == license_id) | (License.license_key == license_id)).first()
    if not lic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="License not found.")

    subs = db.query(Subscription).filter(Subscription.license_id == lic.id).order_by(Subscription.created_at.desc()).all()
    devices = db.query(DeviceActivation).filter(DeviceActivation.license_id == lic.id).all()
    renewals = db.query(RenewalTransaction).filter(RenewalTransaction.license_id == lic.id).order_by(RenewalTransaction.created_at.desc()).all()

    return {
        "license": {
            "id": lic.id,
            "license_key": lic.license_key,
            "customer_id": lic.customer_id,
            "customer_name": lic.customer.name,
            "customer_email": lic.customer.email,
            "product_code": lic.product.product_code,
            "plan_type": lic.plan_type,
            "status": lic.status,
            "max_activations": lic.max_activations,
            "notes": lic.notes,
            "created_at": lic.created_at,
            "updated_at": lic.updated_at,
        },
        "subscriptions": [
            {"id": s.id, "starts_at": s.starts_at, "expires_at": s.expires_at, "status": s.status, "is_active": s.is_active}
            for s in subs
        ],
        "devices": [
            {
                "id": d.id,
                "machine_fingerprint": d.machine_fingerprint,
                "machine_name": d.machine_name,
                "os_info": d.os_info,
                "ip_address": d.ip_address,
                "activated_at": d.activated_at,
                "last_checkin_at": d.last_checkin_at,
                "is_active": d.is_active,
                "deactivated_at": d.deactivated_at,
            }
            for d in devices
        ],
        "renewals": [
            {
                "id": r.id,
                "plan_type": r.plan_type,
                "duration_days": r.duration_days,
                "previous_expiration": r.previous_expiration,
                "new_expiration": r.new_expiration,
                "payment_reference": r.payment_reference,
                "notes": r.notes,
                "created_by": r.created_by,
                "created_at": r.created_at,
            }
            for r in renewals
        ],
    }


# -----------------------------------------------------------------------------
# License Status & Renewal
# -----------------------------------------------------------------------------

@router.patch("/licenses/{license_id}/status")
def update_license_status(
    license_id: str,
    req: LicenseStatusUpdate,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """Updates license status: ACTIVE, SUSPENDED, REVOKED, EXPIRED."""
    lic = db.query(License).filter((License.id == license_id) | (License.license_key == license_id)).first()
    if not lic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="License not found.")

    old_status = lic.status
    lic.status = req.status

    audit = AuditLog(
        actor=admin.username,
        action="UPDATE_LICENSE_STATUS",
        resource_type="license",
        resource_id=lic.id,
        details_json=json.dumps({"old_status": old_status, "new_status": req.status, "reason": req.reason}),
        timestamp=utc_now(),
    )
    db.add(audit)
    db.commit()

    return {"status": "success", "old_status": old_status, "new_status": lic.status}


@router.post("/licenses/{license_id}/renew")
def renew_license(
    license_id: str,
    req: LicenseRenewalRequest,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """
    Extends subscription by 1 year, 3 years, 5 years, or custom dates.
    Handles early renewals (extends from current expiration) and late renewals (extends from now).
    Records payment reference and audit log.
    """
    lic = db.query(License).filter((License.id == license_id) | (License.license_key == license_id)).first()
    if not lic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="License not found.")

    active_sub = (
        db.query(Subscription)
        .filter(Subscription.license_id == lic.id, Subscription.is_active == True)
        .order_by(Subscription.expires_at.desc())
        .first()
    )

    now = utc_now()
    active_exp = ensure_utc(active_sub.expires_at) if active_sub else None
    if active_sub and active_exp and active_exp > now:
        base_date = active_exp  # Early renewal: add to current expiration
        previous_exp = active_exp
    else:
        base_date = now  # Late renewal / expired: add from now
        previous_exp = active_exp if active_exp else now

    if req.plan_type == "custom" and req.custom_expires_at:
        new_expiration = ensure_utc(req.custom_expires_at)
        duration_days = (new_expiration - base_date).days
    else:
        new_expiration, duration_days = calculate_plan_duration(req.plan_type, base_date)

    if new_expiration <= base_date:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="New expiration date must be in the future.")

    # Create new subscription span
    new_sub = Subscription(
        license_id=lic.id,
        starts_at=base_date,
        expires_at=new_expiration,
        status="ACTIVE",
        is_active=True,
    )
    db.add(new_sub)

    # Reactivate license status if it was expired
    if lic.status in ("EXPIRED", "ACTIVE"):
        lic.status = "ACTIVE"

    # Record renewal transaction with payment reference
    tx = RenewalTransaction(
        license_id=lic.id,
        subscription_id=new_sub.id,
        plan_type=req.plan_type,
        duration_days=duration_days,
        previous_expiration=previous_exp,
        new_expiration=new_expiration,
        payment_reference=req.payment_reference,
        notes=req.notes,
        created_by=admin.username,
    )
    db.add(tx)

    audit = AuditLog(
        actor=admin.username,
        action="RENEW_LICENSE",
        resource_type="license",
        resource_id=lic.id,
        details_json=json.dumps({
            "plan_type": req.plan_type,
            "duration_days": duration_days,
            "previous_expiration": previous_exp.isoformat(),
            "new_expiration": new_expiration.isoformat(),
            "payment_reference": req.payment_reference,
        }),
        timestamp=utc_now(),
    )
    db.add(audit)
    db.commit()

    return {
        "status": "success",
        "license_key": lic.license_key,
        "previous_expiration": previous_exp,
        "new_expiration": new_expiration,
        "duration_days": duration_days,
        "plan_type": req.plan_type,
        "payment_reference": req.payment_reference,
    }


# -----------------------------------------------------------------------------
# Offline Air-Gapped Signature Generator
# -----------------------------------------------------------------------------

@router.post("/licenses/offline-sign", response_model=OfflineSignResponse)
def sign_offline_request(
    req: OfflineSignRequest,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """
    Signs an offline activation request (.dmreq) for an air-gapped machine.
    Generates an Ed25519-signed .lic response that can be imported into the desktop app.
    """
    try:
        req_data = json.loads(req.request_text.strip())
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid offline request JSON: {exc}")

    license_key = req_data.get("license_key")
    machine_fingerprint = req_data.get("machine_fingerprint")
    machine_name = req_data.get("machine_name", "Offline-Computer")
    os_info = req_data.get("os_info", "Offline-OS")
    request_nonce = req_data.get("request_nonce")

    if not license_key or not machine_fingerprint or not request_nonce:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request missing license_key, machine_fingerprint, or request_nonce.",
        )

    # Check for replay attack on offline signing nonce
    existing_offline = db.query(OfflineLicenseFile).filter(OfflineLicenseFile.nonce == request_nonce).first()
    if existing_offline:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This offline request nonce has already been processed. Replay rejected.",
        )

    lic = db.query(License).filter(License.license_key == license_key.strip()).first()
    if not lic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="License key not found.")

    if lic.status != "ACTIVE":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"License status is {lic.status}.")

    active_sub = (
        db.query(Subscription)
        .filter(Subscription.license_id == lic.id, Subscription.is_active == True)
        .order_by(Subscription.expires_at.desc())
        .first()
    )
    active_exp = ensure_utc(active_sub.expires_at) if active_sub else None
    active_start = ensure_utc(active_sub.starts_at) if active_sub else None
    if not active_sub or not active_exp or active_exp <= utc_now():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Subscription has expired.")

    # Manage seat activation
    existing_dev = (
        db.query(DeviceActivation)
        .filter(
            DeviceActivation.license_id == lic.id,
            DeviceActivation.machine_fingerprint == machine_fingerprint,
            DeviceActivation.is_active == True,
        )
        .first()
    )
    if not existing_dev:
        active_count = (
            db.query(DeviceActivation)
            .filter(DeviceActivation.license_id == lic.id, DeviceActivation.is_active == True)
            .count()
        )
        if active_count >= lic.max_activations:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Device limit reached ({active_count}/{lic.max_activations}). Deactivate a machine first.",
            )

        new_activation = DeviceActivation(
            license_id=lic.id,
            machine_fingerprint=machine_fingerprint,
            machine_name=machine_name,
            os_info=os_info,
            ip_address="OFFLINE",
            activated_at=utc_now(),
            last_checkin_at=utc_now(),
            is_active=True,
        )
        db.add(new_activation)

    # Build and sign canonical payload with authoritative Ed25519 private key
    signed_license = build_and_sign_license_payload(
        license_id=lic.id,
        license_key=lic.license_key,
        customer_id=lic.customer.id,
        customer_name=lic.customer.name,
        product_code=lic.product.product_code,
        plan_type=lic.plan_type,
        issued_at=active_start,
        expires_at=active_exp,
        max_activations=lic.max_activations,
        status="ACTIVE",
        machine_fingerprint=machine_fingerprint,
        nonce=request_nonce,
    )

    offline_file = OfflineLicenseFile(
        license_id=lic.id,
        machine_fingerprint=machine_fingerprint,
        nonce=request_nonce,
        request_payload=req.request_text,
        signed_response_payload=json.dumps(signed_license),
        issued_at=utc_now(),
        expires_at=active_exp,
        created_by=admin.username,
    )
    db.add(offline_file)

    audit = AuditLog(
        actor=admin.username,
        action="SIGN_OFFLINE_LICENSE",
        resource_type="license",
        resource_id=lic.id,
        details_json=json.dumps({"fingerprint": machine_fingerprint, "expires_at": active_exp.isoformat()}),
        timestamp=utc_now(),
    )
    db.add(audit)
    db.commit()

    return OfflineSignResponse(
        status="success",
        license_key=lic.license_key,
        machine_fingerprint=machine_fingerprint,
        expires_at=active_exp,
        signed_license_file=signed_license,
        download_filename=f"{lic.license_key}_{machine_fingerprint[:8]}.lic",
    )


# -----------------------------------------------------------------------------
# Devices & Seat Management
# -----------------------------------------------------------------------------

@router.get("/devices", response_model=List[DeviceActivationResponse])
def list_devices(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """Lists registered client computers across all licenses."""
    devices = db.query(DeviceActivation).order_by(DeviceActivation.activated_at.desc()).offset(skip).limit(limit).all()
    results = []
    for d in devices:
        results.append(
            DeviceActivationResponse(
                id=d.id,
                license_id=d.license_id,
                license_key=d.license.license_key if d.license else None,
                customer_name=d.license.customer.name if (d.license and d.license.customer) else None,
                machine_fingerprint=d.machine_fingerprint,
                machine_name=d.machine_name,
                os_info=d.os_info,
                ip_address=d.ip_address,
                activated_at=d.activated_at,
                last_checkin_at=d.last_checkin_at,
                is_active=d.is_active,
                deactivated_at=d.deactivated_at,
            )
        )
    return results


@router.post("/devices/{device_id}/deactivate")
def admin_deactivate_device(
    device_id: str,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """Force-deactivates a registered computer seat to facilitate hardware replacement."""
    device = db.query(DeviceActivation).filter(DeviceActivation.id == device_id).first()
    if not device:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device activation record not found.")

    device.is_active = False
    device.deactivated_at = utc_now()

    audit = AuditLog(
        actor=admin.username,
        action="ADMIN_DEACTIVATE_DEVICE",
        resource_type="device",
        resource_id=device.id,
        details_json=json.dumps({"machine": device.machine_name, "fingerprint": device.machine_fingerprint}),
        timestamp=utc_now(),
    )
    db.add(audit)
    db.commit()

    return {"status": "success", "message": f"Device '{device.machine_name}' deactivated."}


# -----------------------------------------------------------------------------
# Audit Logs & Reporting
# -----------------------------------------------------------------------------

@router.get("/audit-logs", response_model=List[AuditLogResponse])
def list_audit_logs(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """Returns recent administrative and security audit events."""
    return db.query(AuditLog).order_by(AuditLog.timestamp.desc()).offset(skip).limit(limit).all()


@router.get("/reports/expirations")
def get_expiration_forecast_report(
    days_ahead: int = 60,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
):
    """Returns licenses expiring within the next N days (for renewal reminders and forecasting)."""
    now = utc_now()
    threshold = now + timedelta(days=days_ahead)

    expiring_subs = (
        db.query(Subscription)
        .join(License)
        .join(Customer)
        .filter(
            Subscription.is_active == True,
            Subscription.expires_at >= now,
            Subscription.expires_at <= threshold,
        )
        .order_by(Subscription.expires_at.asc())
        .all()
    )

    report = []
    for sub in expiring_subs:
        lic = sub.license
        cust = lic.customer
        sub_exp = ensure_utc(sub.expires_at)
        days_left = (sub_exp - now).days if sub_exp else 0
        report.append({
            "license_id": lic.id,
            "license_key": lic.license_key,
            "customer_id": cust.id,
            "customer_name": cust.name,
            "customer_email": cust.email,
            "company": cust.company,
            "plan_type": lic.plan_type,
            "status": lic.status,
            "expires_at": sub_exp,
            "days_remaining": days_left,
        })

    return {
        "generated_at": now,
        "days_ahead": days_ahead,
        "total_expiring": len(report),
        "licenses": report,
    }
