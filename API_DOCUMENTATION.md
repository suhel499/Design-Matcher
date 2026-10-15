# Design Matcher Licensing Server — REST API Documentation

This document provides the complete API specification for the **Design Matcher Licensing Server**. The server provides authoritative license validation, online/offline activations, subscription management, seat transfers, and administrative operations.

---

## 1. Overview & Architecture

- **Protocol:** HTTPS (TLS 1.3 / 1.2 recommended)
- **Base URL:** `https://licensing.yourdomain.com` (or `http://localhost:8000` for local testing)
- **Data Format:** JSON (`application/json; charset=utf-8`)
- **Authentication:**
  - **Client Endpoints (`/api/v1/licenses/*`):** Publicly accessible, secured via cryptographic license keys, hardware fingerprints, and anti-replay nonces.
  - **Admin Endpoints (`/api/v1/admin/*`):** Protected with Bearer JWT tokens in the `Authorization` header (`Authorization: Bearer <token>`).
- **Signature Algorithm:** Asymmetric **Ed25519** (Edwards-curve Digital Signature Algorithm, RFC 8032).
  - Private key: Securely held by the licensing server (never distributed).
  - Public key: Embedded in client binaries (`design_matcher/app/core/licensing/public_key.py`).

---

## 2. Security & Anti-Abuse Controls

1. **Replay Protection:** Sensitive client activation and validation requests include a cryptographically random client nonce (`nonce`). Responses echo or bind this nonce to ensure the response was freshly signed.
2. **Deterministic Canonicalization:** Payloads are canonicalized before hashing/signing:
   - JSON keys sorted alphabetically (`sort_keys=True`)
   - Compact delimiters (`separators=(',', ':')`)
   - UTF-8 byte encoding
3. **Hardware Fingerprinting:** Multi-factor hardware ID computed from OS installation GUID, Motherboard/BIOS UUID, MAC address, and CPU architecture. Format: `DM-HW-XXXX-XXXX-XXXX-XXXX`.
4. **Clock Rollback Detection:** Desktop client records monotonic timestamps on startup, search, and indexing operations. If system clock is set behind the latest recorded timestamp, protected operations are immediately locked.
5. **Rate Limiting:** Recommended reverse proxy (Nginx/Caddy) rate limiting of 10 req/minute on `/api/v1/licenses/activate` and `/validate` per IP.

---

## 3. Client Licensing Endpoints

### 3.1. Activate License (Online)

Activates a desktop client installation on a specific machine.

- **Endpoint:** `POST /api/v1/licenses/activate`
- **Auth:** None (Requires valid `license_key`)

#### Request Body
```json
{
  "license_key": "DM-1Y-20261014-9B3E-7F12",
  "hardware_id": "DM-HW-A1B2-C3D4-E5F6-7890",
  "machine_name": "WORKSTATION-01",
  "os_info": "Windows 11 Pro 64-bit (10.0.22631)",
  "app_version": "1.0.0",
  "nonce": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

#### Success Response (`200 OK`)
```json
{
  "success": true,
  "message": "Activation successful",
  "signed_license": {
    "payload": {
      "license_id": "DM-1Y-20261014-9B3E-7F12",
      "customer_id": "cust-882194a0-e2b1",
      "customer_name": "Apex Fabrications Ltd",
      "product_id": "design_matcher",
      "plan": "1_year",
      "hardware_id": "DM-HW-A1B2-C3D4-E5F6-7890",
      "issued_at": "2026-10-14T15:00:00Z",
      "expires_at": "2027-10-14T15:00:00Z",
      "status": "active",
      "features": ["dxf_matching", "pdf_catalog", "code_search", "export"],
      "max_activations": 1,
      "grace_period_days": 7
    },
    "signature": "3e9f8a...<128-hex-chars-ed25519-signature>...",
    "algorithm": "Ed25519",
    "version": "1.0"
  },
  "license_status": "active",
  "expires_at": "2027-10-14T15:00:00Z"
}
```

#### Error Responses
- `404 Not Found`: License key not found.
- `400 Bad Request`:
  - `Activation limit reached (1/1 seats used). Transfer or deactivate an existing machine.`
  - `License is suspended: Pending annual audit.`
  - `License has been revoked: Chargeback filed.`
  - `License has expired on 2026-09-30T00:00:00Z.`

---

### 3.2. Validate License Heartbeat (Online)

Periodically called by the desktop client when internet access is available to verify license validity and obtain refreshed digital signatures upon renewals.

- **Endpoint:** `POST /api/v1/licenses/validate`
- **Auth:** None

#### Request Body
```json
{
  "license_key": "DM-1Y-20261014-9B3E-7F12",
  "hardware_id": "DM-HW-A1B2-C3D4-E5F6-7890",
  "machine_name": "WORKSTATION-01",
  "app_version": "1.0.0",
  "nonce": "1a2b3c4d5e6f...",
  "current_signature": "3e9f8a..."
}
```

#### Success Response (`200 OK`)
```json
{
  "valid": true,
  "message": "License is active",
  "license_status": "active",
  "expires_at": "2027-10-14T15:00:00Z",
  "signed_license": {
    "payload": { ... },
    "signature": "...",
    "algorithm": "Ed25519",
    "version": "1.0"
  },
  "server_time": "2026-10-14T20:30:00Z"
}
```

---

### 3.3. Deactivate Machine (Seat Release)

Releases a seat allocation when moving software to a new PC or decommissioning hardware.

- **Endpoint:** `POST /api/v1/licenses/deactivate`
- **Auth:** None

#### Request Body
```json
{
  "license_key": "DM-1Y-20261014-9B3E-7F12",
  "hardware_id": "DM-HW-A1B2-C3D4-E5F6-7890",
  "reason": "Upgraded CNC workstation PC"
}
```

#### Success Response (`200 OK`)
```json
{
  "success": true,
  "message": "Device deactivated successfully"
}
```

---

### 3.4. Transfer License Seat

Transfers an existing active seat from an old hardware ID directly to a new hardware ID.

- **Endpoint:** `POST /api/v1/licenses/transfer`
- **Auth:** None

#### Request Body
```json
{
  "license_key": "DM-1Y-20261014-9B3E-7F12",
  "old_hardware_id": "DM-HW-A1B2-C3D4-E5F6-7890",
  "new_hardware_id": "DM-HW-F9E8-D7C6-B5A4-3210",
  "new_machine_name": "WORKSTATION-02",
  "new_os_info": "Windows 11 Pro 64-bit",
  "reason": "Hardware replacement"
}
```

#### Success Response (`200 OK`)
Returns `ActivationResponse` with freshly signed license bound to `new_hardware_id`.

---

### 3.5. Check License Status (Read-Only)

Lightweight status check endpoint.

- **Endpoint:** `GET /api/v1/licenses/status/{license_key}?hardware_id=DM-HW-...`
- **Auth:** None

#### Success Response (`200 OK`)
```json
{
  "license_key": "DM-1Y-20261014-9B3E-7F12",
  "product_id": "design_matcher",
  "status": "active",
  "plan": "1_year",
  "expires_at": "2027-10-14T15:00:00Z",
  "max_activations": 1,
  "active_activations": 1,
  "is_hardware_active": true
}
```

---

## 4. Administrative Endpoints (`/api/v1/admin/*`)

All administrative endpoints require the HTTP header:
```http
Authorization: Bearer <jwt_access_token>
```

---

### 4.1. Admin Authentication (`POST /api/v1/admin/auth/login`)

Authenticate admin credentials and retrieve a JWT token.

#### Request Body
```json
{
  "username": "admin",
  "password": "SecureAdminPassword123!"
}
```

#### Success Response (`200 OK`)
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "username": "admin",
  "role": "admin"
}
```

---

### 4.2. Customer Management

#### List Customers
- **Endpoint:** `GET /api/v1/admin/customers`
- **Response:** Array of Customer objects.

#### Create Customer
- **Endpoint:** `POST /api/v1/admin/customers`
- **Request Body:**
```json
{
  "name": "Apex Fabrications Ltd",
  "email": "orders@apexfab.co.uk",
  "company": "Apex Group",
  "notes": "CNC Cutting Division"
}
```

---

### 4.3. License Management

#### List Licenses
- **Endpoint:** `GET /api/v1/admin/licenses`
- **Query Parameters:**
  - `customer_id` (optional): Filter by customer ID.
  - `status` (optional): Filter by status (`active`, `expired`, `suspended`, `revoked`).

#### Create License
- **Endpoint:** `POST /api/v1/admin/licenses`
- **Request Body:**
```json
{
  "customer_id": "cust-882194a0-e2b1",
  "product_id": "design_matcher",
  "plan": "3_years",
  "max_activations": 2,
  "custom_start_date": null,
  "custom_expiration_date": null,
  "notes": "2 shopfloor licenses"
}
```
*Note on Plans:* Supported plans are `"1_year"`, `"3_years"`, `"5_years"`, and `"custom"`. If `"custom"` is selected, both `custom_start_date` and `custom_expiration_date` (ISO-8601 strings in UTC) must be provided.

#### Update License Status
- **Endpoint:** `PATCH /api/v1/admin/licenses/{license_id}/status`
- **Request Body:**
```json
{
  "status": "suspended",
  "reason": "Payment overdue 30 days"
}
```

#### Renew License Subscription
- **Endpoint:** `POST /api/v1/admin/licenses/{license_id}/renew`
- **Request Body:**
```json
{
  "plan": "1_year",
  "custom_expiration_date": null,
  "payment_reference": "INV-2027-09412",
  "amount": 1200.00,
  "currency": "USD",
  "notes": "Wire transfer received"
}
```
*Renewal Logic:*
- If the license is currently active, the new duration is appended to the **existing** expiration date.
- If the license has already expired, the new duration starts from **current UTC time**.

---

### 4.4. Air-Gapped / Offline License Signing

Generates an Ed25519 cryptographically signed `.lic` license file from a customer's `.dmreq` offline request file.

- **Endpoint:** `POST /api/v1/admin/offline-sign`
- **Request Body:**
```json
{
  "request_payload_json": "{\"license_key\":\"DM-1Y-20261014-9B3E-7F12\",\"hardware_id\":\"DM-HW-A1B2-C3D4-E5F6-7890\",\"machine_name\":\"AIRGAP-CNC-01\",\"requested_at\":\"2026-10-14T15:00:00Z\",\"nonce\":\"...\"}"
}
```

#### Success Response (`200 OK`)
```json
{
  "success": true,
  "message": "Signed license generated successfully",
  "license_key": "DM-1Y-20261014-9B3E-7F12",
  "signed_license_file_json": "{\n  \"payload\": {\n    \"license_id\": \"DM-1Y-20261014-9B3E-7F12\",\n    ...\n  },\n  \"signature\": \"...\"\n}",
  "filename": "DM-1Y-20261014-9B3E-7F12.lic"
}
```

---

### 4.5. Device Management & Deactivation

#### List Active Devices
- **Endpoint:** `GET /api/v1/admin/devices?license_id=DM-1Y-20261014-9B3E-7F12`

#### Force-Deactivate Device
- **Endpoint:** `POST /api/v1/admin/devices/{device_id}/deactivate`
- **Request Body:**
```json
{
  "reason": "Hard drive crashed, customer replaced PC"
}
```

---

### 4.6. Reports & Audit Logs

#### Expiration Forecast Report
- **Endpoint:** `GET /api/v1/admin/reports/expirations?days_ahead=30`
- **Response:** Array of licenses expiring within the specified window.

#### Security & Audit Logs
- **Endpoint:** `GET /api/v1/admin/audit-logs?limit=100&action=renew`
- **Response:** Array of immutable audit records including action, timestamp, actor, IP address, and details JSON.
