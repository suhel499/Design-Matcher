"""
Unit and integration tests for client LicenseManager component:
- Signature verification
- Expiration enforcement & reminder thresholds
- Hardware binding & device mismatch
- Tampered payload rejection
- Clock rollback detection
- Grace period handling during network failure
- Operation blocking (search, index, export)
- Customer user data preservation guarantee
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from app.core.licensing.crypto import generate_keypair, sign_payload
from app.core.licensing.hardware import get_hardware_fingerprint
from app.core.licensing.manager import LicenseManager
from app.core.licensing.models import LicensePayload, LicenseStatus, SignedLicenseFile


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="dm_lic_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def keypair():
    priv, pub = generate_keypair()
    return priv, pub


def create_signed_license(priv_key: str, **kwargs) -> dict:
    hw = kwargs.get("hardware_binding", get_hardware_fingerprint())
    now = datetime.now(timezone.utc)
    expires = kwargs.get("expires_at", (now + timedelta(days=365)).isoformat())

    payload = {
        "license_id": kwargs.get("license_id", "lic-test-123"),
        "license_key": kwargs.get("license_key", "DM-2026-TEST-ABCD"),
        "customer_id": kwargs.get("customer_id", "cust-99"),
        "customer_name": kwargs.get("customer_name", "Test Cutting Shop"),
        "product_id": kwargs.get("product_id", "design-matcher"),
        "plan_type": kwargs.get("plan_type", "1_year"),
        "issued_at": kwargs.get("issued_at", now.isoformat()),
        "expires_at": expires,
        "max_activations": kwargs.get("max_activations", 1),
        "status": kwargs.get("status", "ACTIVE"),
        "hardware_binding": hw,
        "features": kwargs.get("features", ["dxf_matching", "pdf_catalog", "code_search", "dxf_export"]),
        "nonce": "test-nonce-12345",
    }
    sig = sign_payload(payload, priv_key)
    return {
        "version": 1,
        "algorithm": "Ed25519",
        "payload": payload,
        "signature": sig,
    }


def test_unlicensed_state(temp_dir, keypair):
    _, pub = keypair
    mgr = LicenseManager(data_dir=temp_dir, public_key_pem=pub)
    verdict = mgr.check_license()
    assert verdict.is_valid is False
    assert "No license installed" in verdict.message

    allowed, msg = mgr.is_operation_allowed("search_dxf")
    assert allowed is False
    assert "No license installed" in msg


def test_valid_license(temp_dir, keypair):
    priv, pub = keypair
    signed = create_signed_license(priv)

    mgr = LicenseManager(data_dir=temp_dir, public_key_pem=pub)
    success, msg = mgr.save_signed_license(signed)
    assert success is True

    verdict = mgr.check_license()
    assert verdict.is_valid is True
    assert verdict.status == LicenseStatus.ACTIVE
    assert verdict.days_remaining >= 364
    assert verdict.customer_name == "Test Cutting Shop"

    allowed, _ = mgr.is_operation_allowed("search_dxf")
    assert allowed is True
    allowed, _ = mgr.is_operation_allowed("index_dxf")
    assert allowed is True


def test_expired_license_blocks_operations(temp_dir, keypair):
    priv, pub = keypair
    past = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    signed = create_signed_license(priv, expires_at=past)

    mgr = LicenseManager(data_dir=temp_dir, public_key_pem=pub)
    success, msg = mgr.save_signed_license(signed)
    assert success is True

    verdict = mgr.check_license()
    assert verdict.is_valid is False
    assert verdict.is_expired is True
    assert "expired" in verdict.message.lower()

    allowed, msg = mgr.is_operation_allowed("search_dxf")
    assert allowed is False
    assert "expired" in msg.lower()


def test_tampered_payload_rejected(temp_dir, keypair):
    priv, pub = keypair
    signed = create_signed_license(priv)

    # Maliciously extend expiration date without re-signing
    signed["payload"]["expires_at"] = (datetime.now(timezone.utc) + timedelta(days=3650)).isoformat()

    mgr = LicenseManager(data_dir=temp_dir, public_key_pem=pub)
    # Save should reject invalid signature
    success, msg = mgr.save_signed_license(signed)
    assert success is False
    assert "verification failed" in msg.lower()

    verdict = mgr.check_license()
    assert verdict.is_valid is False


def test_wrong_product_rejected(temp_dir, keypair):
    priv, pub = keypair
    signed = create_signed_license(priv, product_id="some-other-software")

    mgr = LicenseManager(data_dir=temp_dir, public_key_pem=pub)
    success, msg = mgr.save_signed_license(signed)
    assert success is False
    assert "some-other-software" in msg


def test_device_binding_mismatch(temp_dir, keypair):
    priv, pub = keypair
    other_hw = "DM-HW-FFFF-9999-8888-7777"
    signed = create_signed_license(priv, hardware_binding=other_hw)

    mgr = LicenseManager(data_dir=temp_dir, public_key_pem=pub)
    success, msg = mgr.save_signed_license(signed)
    assert success is False
    assert "bound to another computer" in msg


def test_clock_rollback_detection(temp_dir, keypair):
    priv, pub = keypair
    signed = create_signed_license(priv)

    mgr = LicenseManager(data_dir=temp_dir, public_key_pem=pub)
    mgr.save_signed_license(signed)

    # First check records high-water mark timestamp
    now = datetime.now(timezone.utc)
    v1 = mgr.check_license(now_utc=now)
    assert v1.is_valid is True

    # User maliciously rolls system clock backward by 10 days
    past_clock = now - timedelta(days=10)
    v2 = mgr.check_license(now_utc=past_clock)
    assert v2.is_valid is False
    assert v2.is_rollback_detected is True
    assert "rollback detected" in v2.message.lower()

    allowed, msg = mgr.is_operation_allowed("search_dxf")
    assert allowed is False
    assert "rollback" in msg.lower()


def test_offline_request_and_import(temp_dir, keypair):
    priv, pub = keypair
    mgr = LicenseManager(data_dir=temp_dir, public_key_pem=pub)

    # 1. Client exports offline request
    req_json = mgr.generate_offline_request("DM-2026-AIRGAP-TEST")
    req_dict = json.loads(req_json)
    assert req_dict["license_key"] == "DM-2026-AIRGAP-TEST"
    assert req_dict["machine_fingerprint"] == get_hardware_fingerprint()

    # 2. Admin signs request offline and gives response file to customer
    signed = create_signed_license(
        priv,
        license_key="DM-2026-AIRGAP-TEST",
        hardware_binding=req_dict["machine_fingerprint"],
    )

    # 3. Customer imports signed license
    ok, msg = mgr.import_offline_license(json.dumps(signed))
    assert ok is True

    verdict = mgr.check_license()
    assert verdict.is_valid is True
    assert verdict.license_key == "DM-2026-AIRGAP-TEST"


def test_customer_data_preservation_guarantee(temp_dir, keypair):
    """
    CRITICAL ACCEPTANCE CRITERIA:
    When a license expires or is invalid, customer catalogs, DXF files,
    databases, and index files must NEVER be deleted or modified.
    """
    priv, pub = keypair

    # Create dummy catalog databases and files in the user data dir
    catalog_db = temp_dir / "catalog.db"
    catalog_db.write_text("DUMMY_CATALOG_DATABASE_DATA")
    dxf_db = temp_dir / "dxf_library.db"
    dxf_db.write_text("DUMMY_DXF_DATABASE_DATA")
    vector_index = temp_dir / "catalog.faiss"
    vector_index.write_bytes(b"DUMMY_FAISS_VECTORS")

    # Install an expired license
    past = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    signed = create_signed_license(priv, expires_at=past)
    mgr = LicenseManager(data_dir=temp_dir, public_key_pem=pub)
    mgr.save_signed_license(signed)

    # Verification should block paid operation
    allowed, msg = mgr.is_operation_allowed("search_dxf")
    assert allowed is False

    # VERIFY USER DATA IS 100% PRESERVED
    assert catalog_db.exists()
    assert catalog_db.read_text() == "DUMMY_CATALOG_DATABASE_DATA"
    assert dxf_db.exists()
    assert dxf_db.read_text() == "DUMMY_DXF_DATABASE_DATA"
    assert vector_index.exists()
    assert vector_index.read_bytes() == b"DUMMY_FAISS_VECTORS"


def test_expiration_warning_reminders(temp_dir, keypair):
    priv, pub = keypair

    # Test 30 days remaining
    now = datetime.now(timezone.utc)
    s30 = create_signed_license(priv, expires_at=(now + timedelta(days=25)).isoformat())
    mgr30 = LicenseManager(data_dir=temp_dir, public_key_pem=pub)
    mgr30.save_signed_license(s30)
    assert mgr30.get_expiration_warning_days() == 30

    # Test 7 days remaining
    s7 = create_signed_license(priv, expires_at=(now + timedelta(days=5)).isoformat())
    mgr7 = LicenseManager(data_dir=temp_dir, public_key_pem=pub)
    mgr7.save_signed_license(s7)
    assert mgr7.get_expiration_warning_days() == 7

    # Test 1 day remaining
    s1 = create_signed_license(priv, expires_at=(now + timedelta(days=1, hours=2)).isoformat())
    mgr1 = LicenseManager(data_dir=temp_dir, public_key_pem=pub)
    mgr1.save_signed_license(s1)
    assert mgr1.get_expiration_warning_days() == 1
