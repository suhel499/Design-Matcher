# Database Architecture & Migration Guide

## 1. Overview
The Design Matcher Licensing System uses a relational database compatible with **PostgreSQL** in production and **SQLite** for local development and offline testing. Schema versioning is managed via **Alembic**.

All records use **UTC timestamps** (`DateTime(timezone=True)`) to ensure unambiguous expiration rules across time zones.

---

## 2. Entity-Relationship Diagram

```
+------------------+          +-----------------------+          +-------------------------+
|    customers     | 1      * |        licenses       | 1      * |      subscriptions      |
|------------------|----------|-----------------------|----------|-------------------------|
| id (PK)          |          | id (PK)               |          | id (PK)                 |
| name             |          | license_key (UQ)      |          | license_id (FK)         |
| email (UQ)       |          | customer_id (FK)      |          | starts_at (UTC)         |
| company          |          | product_id (FK)       |          | expires_at (UTC)        |
| phone            |          | plan_type             |          | status                  |
| notes            |          | status                |          | is_active               |
| created_at (UTC) |          | max_activations       |          | created_at (UTC)        |
| updated_at (UTC) |          | notes                 |          | updated_at (UTC)        |
+------------------+          | created_at (UTC)      |          +-------------------------+
                              | updated_at (UTC)      |                       | 1
                              +-----------------------+                       |
                               | 1                  | 1                       | *
                               |                    |                         |
                               | *                  | *          +-------------------------+
                    +--------------------+    +------------------|   renewal_transactions  |
                    | device_activations |    | offline_licenses |-------------------------|
                    |--------------------|    |------------------| id (PK)                 |
                    | id (PK)            |    | id (PK)          | license_id (FK)         |
                    | license_id (FK)    |    | license_id (FK)  | subscription_id (FK)    |
                    | machine_fingerprint|    | machine_fingerpr | plan_type               |
                    | machine_name       |    | nonce (UQ)       | duration_days           |
                    | os_info            |    | request_payload  | previous_expiration     |
                    | ip_address         |    | signed_response  | new_expiration          |
                    | activated_at (UTC) |    | issued_at (UTC)  | payment_reference       |
                    | last_checkin (UTC) |    | expires_at (UTC) | notes                   |
                    | is_active          |    | created_by       | created_by              |
                    | deactivated_at     |    +------------------| created_at (UTC)        |
                    +--------------------+                       +-------------------------+
```

---

## 3. Table Definitions & Constraints

### 3.1 `customers`
Stores registered commercial accounts and contacts.
- `id` (VARCHAR(36), PK): UUID4 string.
- `name` (VARCHAR(255), NOT NULL): Customer contact or business name.
- `email` (VARCHAR(255), UNIQUE, INDEX): Primary contact email.
- `company` (VARCHAR(255), NULLABLE): Fabrication facility or enterprise name.
- `phone` (VARCHAR(50), NULLABLE): Contact phone number.
- `notes` (TEXT, NULLABLE): Account notes.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): UTC creation time.
- `updated_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): UTC modification time.

### 3.2 `products`
Product catalog identifying licensed software packages.
- `id` (VARCHAR(36), PK): UUID4 string.
- `product_code` (VARCHAR(64), UNIQUE, INDEX): Unique SKU/slug (e.g. `design-matcher`).
- `name` (VARCHAR(255), NOT NULL): Display name.
- `description` (TEXT, NULLABLE): Package details.
- `is_active` (BOOLEAN, DEFAULT TRUE): Product availability flag.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): UTC creation time.

### 3.3 `licenses`
License entitlement master record.
- `id` (VARCHAR(36), PK): UUID4 string.
- `license_key` (VARCHAR(64), UNIQUE, INDEX): Formatted license string (`DM-YYYY-XXXX-XXXX-XXXX`).
- `customer_id` (VARCHAR(36), FK -> `customers.id` ON DELETE CASCADE, INDEX): Associated customer.
- `product_id` (VARCHAR(36), FK -> `products.id` ON DELETE RESTRICT, INDEX): Target product.
- `plan_type` (VARCHAR(32), NOT NULL): `1_year`, `3_years`, `5_years`, `custom`.
- `status` (VARCHAR(32), NOT NULL, INDEX): `ACTIVE`, `EXPIRED`, `SUSPENDED`, `REVOKED`.
- `max_activations` (INTEGER, DEFAULT 1): Maximum concurrent machine seats.
- `notes` (TEXT, NULLABLE): Commercial/PO reference notes.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): UTC creation time.
- `updated_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): UTC modification time.

### 3.4 `subscriptions`
Tracks subscription periods and expiration timestamps.
- `id` (VARCHAR(36), PK): UUID4 string.
- `license_id` (VARCHAR(36), FK -> `licenses.id` ON DELETE CASCADE, INDEX): Target license.
- `starts_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): UTC start date.
- `expires_at` (TIMESTAMP WITH TIME ZONE, NOT NULL, INDEX): Authoritative UTC expiration date.
- `status` (VARCHAR(32), DEFAULT 'ACTIVE'): Subscription status.
- `is_active` (BOOLEAN, DEFAULT TRUE): Active state flag.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): UTC creation time.
- `updated_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): UTC update time.

### 3.5 `device_activations`
Maintains hardware-bound computer seat registrations.
- `id` (VARCHAR(36), PK): UUID4 string.
- `license_id` (VARCHAR(36), FK -> `licenses.id` ON DELETE CASCADE, INDEX): Target license.
- `machine_fingerprint` (VARCHAR(128), NOT NULL, INDEX): SHA-256 hardware hash.
- `machine_name` (VARCHAR(255), NOT NULL): Host computer name.
- `os_info` (VARCHAR(255), NOT NULL): Operating system description.
- `ip_address` (VARCHAR(64), NULLABLE): Last connecting client IP address.
- `activated_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): First activation time.
- `last_checkin_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): Last online validation heartbeat.
- `is_active` (BOOLEAN, DEFAULT TRUE, INDEX): Seat reservation status.
- `deactivated_at` (TIMESTAMP WITH TIME ZONE, NULLABLE): Deactivation timestamp.

### 3.6 `renewal_transactions`
Auditable log of every subscription extension or renewal.
- `id` (VARCHAR(36), PK): UUID4 string.
- `license_id` (VARCHAR(36), FK -> `licenses.id` ON DELETE CASCADE, INDEX): Target license.
- `subscription_id` (VARCHAR(36), FK -> `subscriptions.id` ON DELETE CASCADE, NULLABLE): Resulting subscription.
- `plan_type` (VARCHAR(32), NOT NULL): Extension plan type.
- `duration_days` (INTEGER, NOT NULL): Days added to subscription.
- `previous_expiration` (TIMESTAMP WITH TIME ZONE, NOT NULL): Expiration before renewal.
- `new_expiration` (TIMESTAMP WITH TIME ZONE, NOT NULL): Expiration after renewal.
- `payment_reference` (VARCHAR(128), NULLABLE): Invoice ID, wire reference, or Stripe charge ID.
- `notes` (TEXT, NULLABLE): Administrative notes.
- `created_by` (VARCHAR(128), NOT NULL): Admin username who authorized renewal.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): Transaction timestamp.

### 3.7 `offline_license_files`
Log of air-gapped activation response files issued.
- `id` (VARCHAR(36), PK): UUID4 string.
- `license_id` (VARCHAR(36), FK -> `licenses.id` ON DELETE CASCADE, INDEX): Target license.
- `machine_fingerprint` (VARCHAR(128), NOT NULL, INDEX): Bound hardware identifier.
- `nonce` (VARCHAR(64), UNIQUE, INDEX): Anti-replay unique request nonce.
- `request_payload` (TEXT, NOT NULL): Original `.dmreq` JSON text.
- `signed_response_payload` (TEXT, NOT NULL): Generated `.lic` JSON text.
- `issued_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): Issuance timestamp.
- `expires_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): License expiration timestamp.
- `created_by` (VARCHAR(128), NOT NULL): Admin username.

### 3.8 `audit_logs`
Immutable audit log of all security and administrative events.
- `id` (VARCHAR(36), PK): UUID4 string.
- `actor` (VARCHAR(128), NOT NULL): Username or "client" / "system".
- `action` (VARCHAR(64), NOT NULL, INDEX): Event type (e.g. `CREATE_LICENSE`, `ACTIVATE_SUCCESS`, `DEACTIVATE_DEVICE`).
- `resource_type` (VARCHAR(64), NOT NULL, INDEX): `license`, `customer`, `device`, etc.
- `resource_id` (VARCHAR(128), NOT NULL, INDEX): Identifier of the affected entity.
- `details_json` (TEXT, NOT NULL): Structured JSON event details.
- `ip_address` (VARCHAR(64), NULLABLE): Source IP.
- `timestamp` (TIMESTAMP WITH TIME ZONE, NOT NULL, INDEX): UTC event timestamp.

### 3.9 `admin_users`
Administrator credentials and roles.
- `id` (VARCHAR(36), PK): UUID4 string.
- `username` (VARCHAR(64), UNIQUE, INDEX): Admin login username.
- `email` (VARCHAR(255), UNIQUE, INDEX): Admin email.
- `password_hash` (VARCHAR(255), NOT NULL): PBKDF2-HMAC-SHA256 salted password hash.
- `role` (VARCHAR(32), DEFAULT 'admin'): `superadmin`, `admin`, `support`.
- `is_active` (BOOLEAN, DEFAULT TRUE): Account status.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): Account creation timestamp.

---

## 4. Running Migrations

To apply database migrations to your PostgreSQL or SQLite instance:

```bash
# Set database URL (optional, defaults to sqlite:///./licensing.db)
export DATABASE_URL="postgresql://dm_user:SecretPass@localhost:5432/dm_licensing"

# Run migrations
alembic -c licensing_server/alembic.ini upgrade head
```

To create a new migration after modifying models:
```bash
alembic -c licensing_server/alembic.ini revision --autogenerate -m "describe_changes"
```
