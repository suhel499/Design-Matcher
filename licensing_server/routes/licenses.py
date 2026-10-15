"""
Public and Client API endpoints for online license activation, validation, deactivation, and transfer.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from licensing_server.crypto import build_and_sign_license_payload
from licensing_server.database import get_db
from licensing_server.models import (
    AuditLog,
    Customer,
    DeviceActivation,
    License,
    Product,
    Subscription,
    ensure_utc,
    utc_now,
)
from licensing_server.schemas import (
    ClientActivationRequest,
    ClientActivationResponse,
    ClientDeactivationRequest,
    ClientTransferRequest,
    ClientValidationRequest,
    ClientValidationResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/licenses", tags=["Client Licensing"])


def log_audit(
    db: Session,
    actor: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: dict,
    ip_address: Optional[str] = None,
) -> None:
    """Helper to log security events and audit trails."""
    audit = AuditLog(
        actor=actor,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details_json=json.dumps(details),
        ip_address=ip_address,
        timestamp=utc_now(),
    )
    db.add(audit)


@router.post("/activate", response_model=ClientActivationResponse)
def activate_license(
    req: ClientActivationRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Activates a desktop client machine seat against a license key.
    Enforces status, expiration, and maximum device limits.
    """
    ip = request.client.host if request.client else None
    license_obj = db.query(License).filter(License.license_key == req.license_key.strip()).first()
    if not license_obj:
        log_audit(db, "client", "ACTIVATE_FAILED_NOT_FOUND", "license", req.license_key, {"fingerprint": req.machine_fingerprint}, ip)
        db.commit()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="License key not found.")

    if license_obj.status == "REVOKED":
        log_audit(db, "client", "ACTIVATE_FAILED_REVOKED", "license", license_obj.id, {}, ip)
        db.commit()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This license has been revoked.")

    if license_obj.status == "SUSPENDED":
        log_audit(db, "client", "ACTIVATE_FAILED_SUSPENDED", "license", license_obj.id, {}, ip)
        db.commit()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This license is currently suspended.")

    # Get active subscription
    active_sub = (
        db.query(Subscription)
        .filter(Subscription.license_id == license_obj.id, Subscription.is_active == True)
        .order_by(Subscription.expires_at.desc())
        .first()
    )
    if not active_sub or ensure_utc(active_sub.expires_at) <= utc_now():
        license_obj.status = "EXPIRED"
        db.commit()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This subscription has expired.")

    # Check existing activation on this machine
    existing_dev = (
        db.query(DeviceActivation)
        .filter(
            DeviceActivation.license_id == license_obj.id,
            DeviceActivation.machine_fingerprint == req.machine_fingerprint,
            DeviceActivation.is_active == True,
        )
        .first()
    )

    if existing_dev:
        # Re-activating same machine: update heartbeat and metadata
        existing_dev.machine_name = req.machine_name
        existing_dev.os_info = req.os_info
        existing_dev.last_checkin_at = utc_now()
        existing_dev.ip_address = ip
    else:
        # New machine activation: verify seat limit
        active_count = (
            db.query(DeviceActivation)
            .filter(DeviceActivation.license_id == license_obj.id, DeviceActivation.is_active == True)
            .count()
        )
        if active_count >= license_obj.max_activations:
            log_audit(
                db,
                "client",
                "ACTIVATE_FAILED_DEVICE_LIMIT",
                "license",
                license_obj.id,
                {"active_count": active_count, "max_activations": license_obj.max_activations, "fingerprint": req.machine_fingerprint},
                ip,
            )
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Device activation limit reached ({active_count}/{license_obj.max_activations}). Please deactivate an existing machine or contact support to transfer seats.",
            )

        new_activation = DeviceActivation(
            license_id=license_obj.id,
            machine_fingerprint=req.machine_fingerprint,
            machine_name=req.machine_name,
            os_info=req.os_info,
            ip_address=ip,
            activated_at=utc_now(),
            last_checkin_at=utc_now(),
            is_active=True,
        )
        db.add(new_activation)

    customer = license_obj.customer
    product = license_obj.product

    # Generate Ed25519 digitally signed license file
    signed_license = build_and_sign_license_payload(
        license_id=license_obj.id,
        license_key=license_obj.license_key,
        customer_id=customer.id,
        customer_name=customer.name,
        product_code=product.product_code,
        plan_type=license_obj.plan_type,
        issued_at=ensure_utc(active_sub.starts_at),
        expires_at=ensure_utc(active_sub.expires_at),
        max_activations=license_obj.max_activations,
        status="ACTIVE",
        machine_fingerprint=req.machine_fingerprint,
    )

    log_audit(
        db,
        "client",
        "ACTIVATE_SUCCESS",
        "license",
        license_obj.id,
        {"machine": req.machine_name, "fingerprint": req.machine_fingerprint},
        ip,
    )
    db.commit()

    return ClientActivationResponse(
        status="success",
        message="License activated successfully.",
        license_file=signed_license,
    )


@router.post("/validate", response_model=ClientValidationResponse)
def validate_license(
    req: ClientValidationRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Heartbeat and periodic license validation. Returns authoritative status.
    """
    ip = request.client.host if request.client else None
    license_obj = db.query(License).filter(License.license_key == req.license_key.strip()).first()
    if not license_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="License key not found.")

    if license_obj.status != "ACTIVE":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"License status is {license_obj.status}.")

    # Verify device is active
    device = (
        db.query(DeviceActivation)
        .filter(
            DeviceActivation.license_id == license_obj.id,
            DeviceActivation.machine_fingerprint == req.machine_fingerprint,
            DeviceActivation.is_active == True,
        )
        .first()
    )
    if not device:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Computer is not registered for this license.")

    # Verify subscription expiration
    active_sub = (
        db.query(Subscription)
        .filter(Subscription.license_id == license_obj.id, Subscription.is_active == True)
        .order_by(Subscription.expires_at.desc())
        .first()
    )
    if not active_sub or ensure_utc(active_sub.expires_at) <= utc_now():
        license_obj.status = "EXPIRED"
        db.commit()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Subscription has expired.")

    # Update heartbeat
    device.last_checkin_at = utc_now()
    if ip:
        device.ip_address = ip

    # Return refreshed signed token
    signed_license = build_and_sign_license_payload(
        license_id=license_obj.id,
        license_key=license_obj.license_key,
        customer_id=license_obj.customer.id,
        customer_name=license_obj.customer.name,
        product_code=license_obj.product.product_code,
        plan_type=license_obj.plan_type,
        issued_at=ensure_utc(active_sub.starts_at),
        expires_at=ensure_utc(active_sub.expires_at),
        max_activations=license_obj.max_activations,
        status="ACTIVE",
        machine_fingerprint=req.machine_fingerprint,
    )
    db.commit()

    return ClientValidationResponse(
        status="success",
        message="License is active.",
        license_file=signed_license,
    )


@router.post("/deactivate")
def deactivate_device(
    req: ClientDeactivationRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Deactivates a machine seat to allow license migration to another computer.
    """
    ip = request.client.host if request.client else None
    license_obj = db.query(License).filter(License.license_key == req.license_key.strip()).first()
    if not license_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="License key not found.")

    device = (
        db.query(DeviceActivation)
        .filter(
            DeviceActivation.license_id == license_obj.id,
            DeviceActivation.machine_fingerprint == req.machine_fingerprint,
            DeviceActivation.is_active == True,
        )
        .first()
    )
    if not device:
        return {"status": "ok", "message": "Computer was not actively registered."}

    device.is_active = False
    device.deactivated_at = utc_now()

    log_audit(
        db,
        "client",
        "DEACTIVATE_DEVICE",
        "license",
        license_obj.id,
        {"machine": device.machine_name, "fingerprint": req.machine_fingerprint},
        ip,
    )
    db.commit()
    return {"status": "success", "message": "Machine seat released."}


@router.post("/transfer", response_model=ClientActivationResponse)
def transfer_license(
    req: ClientTransferRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Transfers activation from an old computer to a replacement computer.
    """
    ip = request.client.host if request.client else None
    license_obj = db.query(License).filter(License.license_key == req.license_key.strip()).first()
    if not license_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="License key not found.")

    if license_obj.status != "ACTIVE":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"License is {license_obj.status}.")

    # Deactivate old device if currently active
    old_device = (
        db.query(DeviceActivation)
        .filter(
            DeviceActivation.license_id == license_obj.id,
            DeviceActivation.machine_fingerprint == req.old_machine_fingerprint,
            DeviceActivation.is_active == True,
        )
        .first()
    )
    if old_device:
        old_device.is_active = False
        old_device.deactivated_at = utc_now()

    # Activate new device
    new_device = DeviceActivation(
        license_id=license_obj.id,
        machine_fingerprint=req.new_machine_fingerprint,
        machine_name=req.new_machine_name,
        os_info="Transferred",
        ip_address=ip,
        activated_at=utc_now(),
        last_checkin_at=utc_now(),
        is_active=True,
    )
    db.add(new_device)

    active_sub = (
        db.query(Subscription)
        .filter(Subscription.license_id == license_obj.id, Subscription.is_active == True)
        .order_by(Subscription.expires_at.desc())
        .first()
    )
    if not active_sub or ensure_utc(active_sub.expires_at) <= utc_now():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Subscription has expired.")

    signed_license = build_and_sign_license_payload(
        license_id=license_obj.id,
        license_key=license_obj.license_key,
        customer_id=license_obj.customer.id,
        customer_name=license_obj.customer.name,
        product_code=license_obj.product.product_code,
        plan_type=license_obj.plan_type,
        issued_at=ensure_utc(active_sub.starts_at),
        expires_at=ensure_utc(active_sub.expires_at),
        max_activations=license_obj.max_activations,
        status="ACTIVE",
        machine_fingerprint=req.new_machine_fingerprint,
    )

    log_audit(
        db,
        "client",
        "TRANSFER_DEVICE",
        "license",
        license_obj.id,
        {"old_hw": req.old_machine_fingerprint, "new_hw": req.new_machine_fingerprint, "reason": req.reason},
        ip,
    )
    db.commit()

    return ClientActivationResponse(
        status="success",
        message="License transferred to new computer successfully.",
        license_file=signed_license,
    )


@router.get("/status/{license_key}")
def get_license_public_status(
    license_key: str,
    db: Session = Depends(get_db),
):
    """Public read-only status inquiry for customers."""
    license_obj = db.query(License).filter(License.license_key == license_key.strip()).first()
    if not license_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="License not found.")

    active_sub = (
        db.query(Subscription)
        .filter(Subscription.license_id == license_obj.id, Subscription.is_active == True)
        .order_by(Subscription.expires_at.desc())
        .first()
    )

    return {
        "license_key": license_obj.license_key,
        "product": license_obj.product.name,
        "plan_type": license_obj.plan_type,
        "status": license_obj.status,
        "expires_at": active_sub.expires_at if active_sub else None,
        "max_activations": license_obj.max_activations,
    }
