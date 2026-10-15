"""
Comprehensive automated tests for Licensing Server REST API,
Ed25519 cryptographic signatures, device limit enforcement, renewals, and offline signing.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.licensing.crypto import verify_payload
from app.core.licensing.public_key import get_verification_public_key
from licensing_server.config import settings
from licensing_server.database import Base, get_db
from licensing_server.main import app, init_db_and_seeds
from licensing_server.models import AdminUser, Customer, License, Product, Subscription
from licensing_server.security import hash_password

from sqlalchemy.pool import StaticPool

# Use an isolated in-memory SQLite test database with StaticPool
TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient connected to an isolated test database with seeded admin."""
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Seed product and admin in test DB
    db = TestingSessionLocal()
    prod = Product(
        product_code="design-matcher",
        name="Laser Cut Design Matcher Desktop",
        description="CAD Suite",
    )
    db.add(prod)
    admin = AdminUser(
        username="admin",
        email="admin@test.local",
        password_hash=hash_password("AdminPass@2026"),
        role="superadmin",
        is_active=True,
    )
    db.add(admin)
    db.commit()
    db.close()

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="module")
def admin_token(client) -> str:
    """Logs in and returns a valid admin JWT bearer token."""
    res = client.post("/api/v1/admin/auth/login", json={"username": "admin", "password": "AdminPass@2026"})
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_admin_login_invalid_password(client):
    res = client.post("/api/v1/admin/auth/login", json={"username": "admin", "password": "WrongPassword"})
    assert res.status_code == 401


def test_customer_lifecycle(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. Create Customer
    payload = {
        "name": "Acme Metal Fab",
        "email": "info@acmemetal.com",
        "company": "Acme Corp",
        "phone": "+1 555-0100",
        "notes": "CNC cut workshop",
    }
    res = client.post("/api/v1/admin/customers", json=payload, headers=headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["name"] == "Acme Metal Fab"
    customer_id = data["id"]

    # 2. Duplicate email rejected
    res_dup = client.post("/api/v1/admin/customers", json=payload, headers=headers)
    assert res_dup.status_code == 400

    # 3. List customers
    res_list = client.get("/api/v1/admin/customers?search=Acme", headers=headers)
    assert res_list.status_code == 200
    assert len(res_list.json()) >= 1


def test_license_creation_plans(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Fetch customer
    cust_res = client.get("/api/v1/admin/customers", headers=headers)
    customer_id = cust_res.json()[0]["id"]

    # 1. Create 1-Year License
    res_1yr = client.post(
        "/api/v1/admin/licenses",
        json={"customer_id": customer_id, "plan_type": "1_year", "max_activations": 1},
        headers=headers,
    )
    assert res_1yr.status_code == 200
    lic_1yr = res_1yr.json()
    assert lic_1yr["plan_type"] == "1_year"
    assert lic_1yr["status"] == "ACTIVE"
    assert lic_1yr["license_key"].startswith("DM-")
    exp_1yr = datetime.fromisoformat(lic_1yr["expires_at"])
    start_1yr = datetime.fromisoformat(lic_1yr["starts_at"])
    assert (exp_1yr - start_1yr).days == 365

    # 2. Create 3-Years License
    res_3yr = client.post(
        "/api/v1/admin/licenses",
        json={"customer_id": customer_id, "plan_type": "3_years", "max_activations": 2},
        headers=headers,
    )
    assert res_3yr.status_code == 200
    lic_3yr = res_3yr.json()
    exp_3yr = datetime.fromisoformat(lic_3yr["expires_at"])
    start_3yr = datetime.fromisoformat(lic_3yr["starts_at"])
    assert (exp_3yr - start_3yr).days == 365 * 3

    # 3. Create Custom Duration License
    custom_start = datetime.now(timezone.utc)
    custom_exp = custom_start + timedelta(days=90)
    res_cust = client.post(
        "/api/v1/admin/licenses",
        json={
            "customer_id": customer_id,
            "plan_type": "custom",
            "custom_starts_at": custom_start.isoformat(),
            "custom_expires_at": custom_exp.isoformat(),
            "max_activations": 1,
        },
        headers=headers,
    )
    assert res_cust.status_code == 200
    lic_cust = res_cust.json()
    assert lic_cust["plan_type"] == "custom"


def test_online_activation_and_signature_verification(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    cust_res = client.get("/api/v1/admin/customers", headers=headers)
    customer_id = cust_res.json()[0]["id"]

    # Create single-seat license
    res_lic = client.post(
        "/api/v1/admin/licenses",
        json={"customer_id": customer_id, "plan_type": "1_year", "max_activations": 1},
        headers=headers,
    )
    lic_key = res_lic.json()["license_key"]

    # 1. Activate Machine 1
    hw_1 = "DM-HW-AAAA-BBBB-CCCC-DDDD"
    res_act = client.post(
        "/api/v1/licenses/activate",
        json={
            "license_key": lic_key,
            "machine_fingerprint": hw_1,
            "machine_name": "Workshop-PC1",
            "os_info": "Windows 11 Pro",
            "nonce": "test-nonce-1",
        },
    )
    assert res_act.status_code == 200, res_act.text
    act_data = res_act.json()
    assert act_data["status"] == "success"
    signed_file = act_data["license_file"]

    # Cryptographic verification of returned license using client public key
    pub_key = get_verification_public_key()
    assert verify_payload(signed_file["payload"], signed_file["signature"], pub_key) is True
    assert signed_file["payload"]["hardware_binding"] == hw_1
    assert signed_file["payload"]["product_id"] == "design-matcher"

    # 2. Duplicate activation on same machine updates last checkin
    res_dup = client.post(
        "/api/v1/licenses/activate",
        json={
            "license_key": lic_key,
            "machine_fingerprint": hw_1,
            "machine_name": "Workshop-PC1-Updated",
            "os_info": "Windows 11 Pro",
            "nonce": "test-nonce-dup",
        },
    )
    assert res_dup.status_code == 200

    # 3. Activation on Machine 2 must fail due to device limit (1 seat)
    hw_2 = "DM-HW-XXXX-YYYY-ZZZZ-WWWW"
    res_exceed = client.post(
        "/api/v1/licenses/activate",
        json={
            "license_key": lic_key,
            "machine_fingerprint": hw_2,
            "machine_name": "Office-Laptop",
            "os_info": "Windows 10",
            "nonce": "test-nonce-2",
        },
    )
    assert res_exceed.status_code == 403
    assert "Device activation limit reached" in res_exceed.json()["detail"]

    # 4. Deactivate Machine 1
    res_deact = client.post(
        "/api/v1/licenses/deactivate",
        json={"license_key": lic_key, "machine_fingerprint": hw_1, "nonce": "nonce-deact"},
    )
    assert res_deact.status_code == 200

    # 5. Machine 2 can now activate because seat was released
    res_act2 = client.post(
        "/api/v1/licenses/activate",
        json={
            "license_key": lic_key,
            "machine_fingerprint": hw_2,
            "machine_name": "Office-Laptop",
            "os_info": "Windows 10",
            "nonce": "test-nonce-after-deact",
        },
    )
    assert res_act2.status_code == 200
    assert res_act2.json()["license_file"]["payload"]["hardware_binding"] == hw_2


def test_license_suspension_and_revocation(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    cust_res = client.get("/api/v1/admin/customers", headers=headers)
    customer_id = cust_res.json()[0]["id"]

    res_lic = client.post(
        "/api/v1/admin/licenses",
        json={"customer_id": customer_id, "plan_type": "1_year", "max_activations": 1},
        headers=headers,
    )
    lic_id = res_lic.json()["id"]
    lic_key = res_lic.json()["license_key"]

    # Suspend license
    res_susp = client.patch(
        f"/api/v1/admin/licenses/{lic_id}/status",
        json={"status": "SUSPENDED", "reason": "Payment overdue"},
        headers=headers,
    )
    assert res_susp.status_code == 200

    # Activation attempt while suspended must be blocked
    res_act = client.post(
        "/api/v1/licenses/activate",
        json={
            "license_key": lic_key,
            "machine_fingerprint": "DM-HW-TEST-SUSP",
            "machine_name": "Test",
            "os_info": "Test OS",
            "nonce": "n1",
        },
    )
    assert res_act.status_code == 403
    assert "suspended" in res_act.json()["detail"]


def test_subscription_renewal(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    cust_res = client.get("/api/v1/admin/customers", headers=headers)
    customer_id = cust_res.json()[0]["id"]

    res_lic = client.post(
        "/api/v1/admin/licenses",
        json={"customer_id": customer_id, "plan_type": "1_year", "max_activations": 1},
        headers=headers,
    )
    lic_id = res_lic.json()["id"]
    initial_exp = datetime.fromisoformat(res_lic.json()["expires_at"])

    # Renew early for +3 years
    res_renew = client.post(
        f"/api/v1/admin/licenses/{lic_id}/renew",
        json={"plan_type": "3_years", "payment_reference": "INV-2026-999"},
        headers=headers,
    )
    assert res_renew.status_code == 200
    renew_data = res_renew.json()
    new_exp = datetime.fromisoformat(renew_data["new_expiration"])
    # Early renewal adds exactly 3*365 days to initial expiration
    assert (new_exp - initial_exp).days == 3 * 365


def test_offline_air_gap_signing(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    cust_res = client.get("/api/v1/admin/customers", headers=headers)
    customer_id = cust_res.json()[0]["id"]

    res_lic = client.post(
        "/api/v1/admin/licenses",
        json={"customer_id": customer_id, "plan_type": "1_year", "max_activations": 1},
        headers=headers,
    )
    lic_key = res_lic.json()["license_key"]

    offline_request_payload = {
        "version": 1,
        "license_key": lic_key,
        "machine_fingerprint": "DM-HW-AIRGAP-0001",
        "machine_name": "CNC-Isolated-Station",
        "os_info": "Windows 10 IoT Enterprise",
        "request_timestamp": datetime.now(timezone.utc).isoformat(),
        "request_nonce": "nonce-offline-unique-1234",
    }

    # Admin signs offline request
    res_sign = client.post(
        "/api/v1/admin/licenses/offline-sign",
        json={"request_text": json.dumps(offline_request_payload)},
        headers=headers,
    )
    assert res_sign.status_code == 200, res_sign.text
    sign_data = res_sign.json()
    signed_file = sign_data["signed_license_file"]

    # Verify signature
    pub_key = get_verification_public_key()
    assert verify_payload(signed_file["payload"], signed_file["signature"], pub_key) is True
    assert signed_file["payload"]["hardware_binding"] == "DM-HW-AIRGAP-0001"

    # Replay attack with same nonce must be rejected
    res_replay = client.post(
        "/api/v1/admin/licenses/offline-sign",
        json={"request_text": json.dumps(offline_request_payload)},
        headers=headers,
    )
    assert res_replay.status_code == 409


def test_audit_logs_and_reports(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Audit logs
    res_audit = client.get("/api/v1/admin/audit-logs", headers=headers)
    assert res_audit.status_code == 200
    assert len(res_audit.json()) > 0

    # Expiration forecast report
    res_report = client.get("/api/v1/admin/reports/expirations?days_ahead=400", headers=headers)
    assert res_report.status_code == 200
    data = res_report.json()
    assert "total_expiring" in data
    assert "licenses" in data
