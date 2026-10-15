# Administrator & Operations Guide — Subscription & License Management

This guide covers day-to-day operations for administrators managing customer accounts, issuing subscription licenses, processing renewals, handling air-gapped offline activations, managing device seats, and reviewing security audit logs.

---

## 1. Accessing the Administration Web Portal

1. Open your web browser and navigate to:
   - **Local / Staging:** `http://localhost:8000/admin`
   - **Production:** `https://licensing.yourcompany.com/admin`
2. Enter your administrator username and password.
3. Upon login, your session is authenticated via an industry-standard signed JWT stored securely in session storage.

---

## 2. Managing Customers

Every subscription license is assigned to a specific customer profile.

### 2.1. Creating a New Customer
1. Navigate to the **Customers** tab.
2. Click **+ Add Customer**.
3. Fill in:
   - **Customer Name:** Full individual or primary contact name.
   - **Email Address:** Billing or operational contact email.
   - **Company / Organization:** Legal business name (e.g., `Apex Metalworks Ltd`).
   - **Notes:** Optional internal details (e.g., account manager, contract number).
4. Click **Create Customer**.

---

## 3. Creating & Issuing Subscription Licenses

### 3.1. Supported Subscription Plans
The system supports 4 subscription durations:
- **1 Year:** Automatically sets expiration to exactly 365 days from creation (UTC).
- **3 Years:** Sets expiration to 3 * 365 days (1,095 days) from creation (UTC).
- **5 Years:** Sets expiration to 5 * 365 days (1,825 days) from creation (UTC).
- **Custom Dates:** Allows administrators to specify arbitrary start and expiration dates (useful for multi-month trials, co-terming with corporate fiscal years, or multi-year custom agreements).

### 3.2. Generating a License
1. Navigate to the **Licenses** tab.
2. Click **+ Generate License**.
3. Select the customer from the dropdown.
4. Choose the subscription plan (**1 Year**, **3 Years**, **5 Years**, or **Custom**).
5. If **Custom** is selected, enter the Start Date and Expiration Date in `YYYY-MM-DD` format.
6. Set **Seat Limit (Max Activations)**: The number of distinct workstations permitted to activate simultaneously (default is `1`).
7. Enter optional notes (e.g., PO number or shop floor workstation tag).
8. Click **Generate License**.
9. The unique license key (e.g., `DM-3Y-20261014-A82F-C3B1`) will be created and displayed in the table. Copy and deliver this key to the customer.

---

## 4. License Status Management

Licenses can be in one of four states:
- **Active:** Normal operation. Desktop clients can validate and use all features.
- **Expired:** Subscription duration elapsed. Paid operations (search, indexing, export) are locked. Customer files and search history remain 100% intact.
- **Suspended:** Administrative temporary hold (e.g., billing dispute, audit). Client displays a suspended notification and locks protected features.
- **Revoked:** Permanent cancellation (e.g., fraudulent purchase, refund issued). Cannot be reactivated.

To change a license's status:
1. Find the license in the **Licenses** tab.
2. Click the status button or action menu.
3. Select **Suspend**, **Activate**, or **Revoke**.
4. Enter an audit reason. The status is updated immediately and logged to the audit trail.

---

## 5. Subscription Renewals

When a customer pays to extend their subscription:
1. Navigate to the **Licenses** tab.
2. Find the customer's license and click **Renew**.
3. Choose the extension plan (**1 Year**, **3 Years**, **5 Years**, or **Custom Expiration Date**).
4. Enter the **Payment Reference** (e.g., `INV-2027-1049` or Stripe Charge ID).
5. Enter the **Amount** and **Currency** (e.g., `1200.00` `USD`).
6. Click **Confirm Renewal**.

### Renewal Expiration Rules:
- **If the license is currently Active:** The new duration is added to the **existing** expiration date. The customer loses zero paid days.
- **If the license has already Expired:** The new subscription starts immediately from the current time.
- Next time the customer's desktop app connects to the internet (or imports an updated `.lic` file), the new expiration date is automatically updated.

---

## 6. Air-Gapped / Offline Activation Workflow

For customers with CNC factory workstations that are strictly air-gapped without internet access:

### Step 1: Customer Exports Request
1. The customer opens `Design Matcher` on their air-gapped Windows PC.
2. In the License Dialog, they enter their License Key and click **Offline Activation**.
3. They click **Generate & Save Request (.dmreq)** and save the file to a USB thumb drive.

### Step 2: Administrator Signs the Request
1. The customer sends the `.dmreq` file to you (via email or support portal).
2. Open the Admin Web Portal and click the **Air-Gap Signer** tab.
3. Click **Browse** and select the customer's `.dmreq` file (or paste its JSON text).
4. Click **Verify & Sign License Response**.
5. The portal validates that:
   - The license key exists and is Active.
   - The customer's hardware ID does not exceed the allowed seat limit.
   - The subscription is not expired.
6. The server generates an Ed25519 digital signature and downloads the signed `.lic` file (e.g., `DM-1Y-20261014-9B3E-7F12.lic`).

### Step 3: Customer Imports Signed File
1. Email or provide the `.lic` file back to the customer.
2. On their air-gapped PC, the customer clicks **Import Signed License (.lic)** and selects the file.
3. The software verifies the digital signature using its embedded public key, confirms the hardware binding matches the PC, and instantly unlocks the application.

---

## 7. Managing Computer Activations & Seat Transfers

### 7.1. Viewing Active Devices
1. Navigate to the **Devices** tab.
2. View the table of all registered machines, including:
   - Machine Name (e.g., `CNC-OPERATOR-PC`)
   - Hardware Fingerprint (`DM-HW-XXXX-XXXX-XXXX-XXXX`)
   - Operating System & App Version
   - Activation Date & Last Seen Timestamp

### 7.2. Force-Deactivating a Lost or Replaced Machine
If a customer's computer motherboard fails or a laptop is stolen:
1. Locate the device in the **Devices** tab.
2. Click **Deactivate**.
3. Enter the reason (e.g., `Customer reported motherboard failure; releasing seat for replacement PC`).
4. The seat is released immediately, allowing the customer to activate their replacement workstation.

---

## 8. Expiration Forecast & Renewal Reports

1. Navigate to the **Expirations** tab.
2. Filter by timeframe (e.g., Next 30 Days, Next 15 Days, Next 7 Days).
3. The table displays all licenses due for renewal, customer contact details, and current status.
4. Export or use this list to proactively contact customers before their license expires.

---

## 9. Security & Audit Logging

Every sensitive action is permanently recorded in the immutable audit log:
- Administrative logins
- License creation
- Subscription renewals with payment references
- Status changes (Suspensions / Revocations)
- Device activations and deactivations
- Offline license signings

To review logs:
1. Navigate to the **Audit Logs** tab.
2. Search by License ID, Customer ID, or Action (`activate`, `validate`, `renew`, `suspend`, `offline_sign`).
3. Each record includes the timestamp (UTC), IP address, actor, and detailed JSON payload.
