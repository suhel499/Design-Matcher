/**
 * Design Matcher Licensing Authority — Admin Portal Front-End Logic.
 */

const API_BASE = '/api/v1/admin';

let authToken = localStorage.getItem('dm_admin_token');
let currentAdminUser = localStorage.getItem('dm_admin_user') || 'admin';
let customersCache = [];
let lastSignedLic = null;

// DOM Elements
const loginOverlay = document.getElementById('loginOverlay');
const appContainer = document.getElementById('appContainer');
const loginForm = document.getElementById('loginForm');
const loginError = document.getElementById('loginError');
const adminUserDisplay = document.getElementById('adminUserDisplay');
const adminAvatar = document.getElementById('adminAvatar');
const logoutBtn = document.getElementById('logoutBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  if (authToken) {
    showApp();
  } else {
    showLogin();
  }

  setupEventListeners();
});

function showLogin() {
  loginOverlay.style.display = 'flex';
  appContainer.style.display = 'none';
}

function showApp() {
  loginOverlay.style.display = 'none';
  appContainer.style.display = 'flex';
  adminUserDisplay.textContent = currentAdminUser;
  adminAvatar.textContent = currentAdminUser.charAt(0).toUpperCase();
  loadAllDashboardData();
}

async function apiRequest(endpoint, method = 'GET', body = null) {
  const headers = { 'Content-Type': 'application/json' };
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }

  const options = { method, headers };
  if (body) {
    options.body = JSON.stringify(body);
  }

  const response = await fetch(`${API_BASE}${endpoint}`, options);
  if (response.status === 401) {
    // Token expired or invalid
    localStorage.removeItem('dm_admin_token');
    authToken = null;
    showLogin();
    throw new Error('Session expired. Please log in again.');
  }

  if (!response.ok) {
    let errorDetail = response.statusText;
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || JSON.stringify(errJson);
    } catch (_) {}
    throw new Error(errorDetail);
  }

  return await response.json();
}

function setupEventListeners() {
  // Login Form
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    loginError.textContent = '';
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;

    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Authentication failed');
      }

      const data = await res.json();
      authToken = data.access_token;
      currentAdminUser = data.username;
      localStorage.setItem('dm_admin_token', authToken);
      localStorage.setItem('dm_admin_user', currentAdminUser);
      showApp();
    } catch (err) {
      loginError.textContent = err.message;
    }
  });

  // Logout
  logoutBtn.addEventListener('click', () => {
    localStorage.removeItem('dm_admin_token');
    localStorage.removeItem('dm_admin_user');
    authToken = null;
    showLogin();
  });

  // Tab Navigation
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');

      const titleMap = {
        'tab-overview': 'Licensing Dashboard',
        'tab-licenses': 'License Management',
        'tab-customers': 'Customer Directory',
        'tab-offline': 'Offline Air-Gap Response Signer',
        'tab-devices': 'Computer Activations',
        'tab-expirations': 'Subscription Expirations & Renewals',
        'tab-audit': 'System Audit Logs'
      };
      document.getElementById('pageTitle').textContent = titleMap[tabId] || 'Dashboard';
    });
  });

  // Modals Open / Close
  document.getElementById('btnNewCustomer').addEventListener('click', () => openModal('customerModal'));
  document.getElementById('btnNewLicense').addEventListener('click', () => {
    populateCustomerDropdown();
    openModal('licenseModal');
  });

  document.querySelectorAll('[data-close]').forEach(btn => {
    btn.addEventListener('click', () => closeModal(btn.getAttribute('data-close')));
  });

  // Custom Plan Type Toggle in Create License Form
  document.getElementById('licPlanType').addEventListener('change', (e) => {
    const customDiv = document.getElementById('customDatesGroup');
    customDiv.style.display = e.target.value === 'custom' ? 'block' : 'none';
  });

  // Custom Plan Type Toggle in Renew Form
  document.getElementById('renewPlanType').addEventListener('change', (e) => {
    const customDiv = document.getElementById('renewCustomDateGroup');
    customDiv.style.display = e.target.value === 'custom' ? 'block' : 'none';
  });

  // Create Customer Form Submit
  document.getElementById('createCustomerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
      name: document.getElementById('custName').value.trim(),
      email: document.getElementById('custEmail').value.trim(),
      company: document.getElementById('custCompany').value.trim() || null,
      phone: document.getElementById('custPhone').value.trim() || null,
      notes: document.getElementById('custNotes').value.trim() || null
    };

    try {
      await apiRequest('/customers', 'POST', payload);
      closeModal('customerModal');
      document.getElementById('createCustomerForm').reset();
      loadCustomers();
      loadOverviewStats();
    } catch (err) {
      alert(`Failed to create customer: ${err.message}`);
    }
  });

  // Create License Form Submit
  document.getElementById('createLicenseForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const planType = document.getElementById('licPlanType').value;
    const payload = {
      customer_id: document.getElementById('licCustomerSelect').value,
      product_code: 'design-matcher',
      plan_type: planType,
      max_activations: parseInt(document.getElementById('licMaxSeats').value, 10),
      notes: document.getElementById('licNotes').value.trim() || null
    };

    if (planType === 'custom') {
      const s = document.getElementById('licCustomStart').value;
      const exp = document.getElementById('licCustomExp').value;
      if (s) payload.custom_starts_at = new Date(s).toISOString();
      if (exp) payload.custom_expires_at = new Date(exp).toISOString();
    }

    try {
      const res = await apiRequest('/licenses', 'POST', payload);
      closeModal('licenseModal');
      document.getElementById('createLicenseForm').reset();
      loadLicenses();
      loadOverviewStats();
      alert(`License generated successfully!\nKey: ${res.license_key}\nExpires: ${formatDate(res.expires_at)}`);
    } catch (err) {
      alert(`Failed to create license: ${err.message}`);
    }
  });

  // Process Renewal Form Submit
  document.getElementById('renewLicenseForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const licId = document.getElementById('renewLicId').value;
    const planType = document.getElementById('renewPlanType').value;
    const payload = {
      plan_type: planType,
      payment_reference: document.getElementById('renewPaymentRef').value.trim() || null,
      notes: document.getElementById('renewNotes').value.trim() || null
    };

    if (planType === 'custom') {
      const exp = document.getElementById('renewCustomExp').value;
      if (exp) payload.custom_expires_at = new Date(exp).toISOString();
    }

    try {
      const res = await apiRequest(`/licenses/${licId}/renew`, 'POST', payload);
      closeModal('renewModal');
      document.getElementById('renewLicenseForm').reset();
      loadLicenses();
      loadOverviewStats();
      loadExpirations();
      alert(`License renewed successfully!\nNew Expiration: ${formatDate(res.new_expiration)}`);
    } catch (err) {
      alert(`Renewal failed: ${err.message}`);
    }
  });

  // Offline Air-Gap Signer
  document.getElementById('btnSignOffline').addEventListener('click', async () => {
    const rawReq = document.getElementById('offlineRequestInput').value.trim();
    if (!rawReq) {
      alert('Please paste the client .dmreq request JSON.');
      return;
    }

    try {
      const res = await apiRequest('/licenses/offline-sign', 'POST', { request_text: rawReq });
      lastSignedLic = res;
      document.getElementById('offResKey').textContent = res.license_key;
      document.getElementById('offResHw').textContent = res.machine_fingerprint;
      document.getElementById('offResExp').textContent = formatDate(res.expires_at);
      document.getElementById('offlineResponseOutput').value = JSON.stringify(res.signed_license_file, null, 2);
      document.getElementById('offlineResultArea').style.display = 'block';
    } catch (err) {
      alert(`Signing error: ${err.message}`);
    }
  });

  document.getElementById('btnDownloadLic').addEventListener('click', () => {
    if (!lastSignedLic) return;
    const blob = new Blob([JSON.stringify(lastSignedLic.signed_license_file, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = lastSignedLic.download_filename || 'design_matcher.lic';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  document.getElementById('btnCopyLic').addEventListener('click', () => {
    const txt = document.getElementById('offlineResponseOutput').value;
    navigator.clipboard.writeText(txt).then(() => alert('Signed license copied to clipboard!'));
  });

  // Refresh Buttons
  document.getElementById('btnRefreshLicenses').addEventListener('click', loadLicenses);
  document.getElementById('btnRefreshDevices').addEventListener('click', loadDevices);
  document.getElementById('btnRefreshExpirations').addEventListener('click', loadExpirations);
  document.getElementById('btnRefreshAudit').addEventListener('click', loadAuditLogs);
  document.getElementById('licenseStatusFilter').addEventListener('change', loadLicenses);
}

function openModal(modalId) {
  document.getElementById(modalId).classList.add('active');
}

function closeModal(modalId) {
  document.getElementById(modalId).classList.remove('active');
}

function formatDate(isoStr) {
  if (!isoStr) return 'N/A';
  const d = new Date(isoStr);
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC' }) + ' UTC';
}

function formatBadge(status) {
  const s = (status || '').toUpperCase();
  if (s === 'ACTIVE') return `<span class="badge badge-active">ACTIVE</span>`;
  if (s === 'EXPIRED') return `<span class="badge badge-expired">EXPIRED</span>`;
  if (s === 'SUSPENDED') return `<span class="badge badge-suspended">SUSPENDED</span>`;
  return `<span class="badge badge-revoked">${s}</span>`;
}

// -----------------------------------------------------------------------------
// Data Loading
// -----------------------------------------------------------------------------

async function loadAllDashboardData() {
  await Promise.all([
    loadOverviewStats(),
    loadLicenses(),
    loadCustomers(),
    loadDevices(),
    loadExpirations(),
    loadAuditLogs()
  ]);
}

async function loadOverviewStats() {
  try {
    const [licenses, customers, devices, expReport] = await Promise.all([
      apiRequest('/licenses?limit=500'),
      apiRequest('/customers?limit=500'),
      apiRequest('/devices?limit=500'),
      apiRequest('/reports/expirations?days_ahead=30')
    ]);

    const activeLics = licenses.filter(l => l.status === 'ACTIVE').length;
    const activeDevs = devices.filter(d => d.is_active).length;

    document.getElementById('statActiveLicenses').textContent = activeLics;
    document.getElementById('statExpiringSoon').textContent = expReport.total_expiring;
    document.getElementById('statActiveDevices').textContent = activeDevs;
    document.getElementById('statTotalCustomers').textContent = customers.length;

    // Render recent table
    const tbody = document.querySelector('#recentLicensesTable tbody');
    if (licenses.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="loading">No licenses issued yet. Click "+ Generate License" above.</td></tr>`;
      return;
    }

    tbody.innerHTML = licenses.slice(0, 5).map(lic => `
      <tr>
        <td><strong>${lic.license_key}</strong></td>
        <td>${lic.customer_name || 'N/A'}</td>
        <td>${lic.plan_type.replace('_', ' ').toUpperCase()}</td>
        <td>${lic.active_device_count} / ${lic.max_activations}</td>
        <td>${formatDate(lic.expires_at)}</td>
        <td>${formatBadge(lic.status)}</td>
        <td>
          <button class="btn btn-sm btn-outline" onclick="openRenewModal('${lic.id}', '${lic.license_key}', '${escapeHtml(lic.customer_name)}', '${lic.expires_at}')">Renew</button>
        </td>
      </tr>
    `).join('');
  } catch (err) {
    console.error('Failed to load overview:', err);
  }
}

async function loadLicenses() {
  const statusFilter = document.getElementById('licenseStatusFilter').value;
  const endpoint = statusFilter ? `/licenses?status_filter=${statusFilter}` : '/licenses';
  const tbody = document.querySelector('#allLicensesTable tbody');

  try {
    const licenses = await apiRequest(endpoint);
    if (licenses.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="loading">No licenses match filter.</td></tr>`;
      return;
    }

    tbody.innerHTML = licenses.map(lic => `
      <tr>
        <td><strong>${lic.license_key}</strong></td>
        <td>${lic.customer_name || 'N/A'}<br><small style="color: #9ca3af;">${lic.customer_email || ''}</small></td>
        <td>${lic.plan_type.replace('_', ' ').toUpperCase()}</td>
        <td>${lic.active_device_count} / ${lic.max_activations}</td>
        <td>${formatDate(lic.starts_at)}</td>
        <td>${formatDate(lic.expires_at)}</td>
        <td>${formatBadge(lic.status)}</td>
        <td>
          <div style="display: flex; gap: 6px;">
            <button class="btn btn-sm btn-outline" onclick="openRenewModal('${lic.id}', '${lic.license_key}', '${escapeHtml(lic.customer_name)}', '${lic.expires_at}')">Renew</button>
            <button class="btn btn-sm btn-outline" onclick="toggleStatus('${lic.id}', '${lic.status}')">${lic.status === 'ACTIVE' ? 'Suspend' : 'Activate'}</button>
          </div>
        </td>
      </tr>
    `).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" class="error-msg">Failed to load licenses: ${err.message}</td></tr>`;
  }
}

async function loadCustomers() {
  const tbody = document.querySelector('#customersTable tbody');
  try {
    const customers = await apiRequest('/customers');
    customersCache = customers;
    if (customers.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="loading">No customers registered yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = customers.map(c => `
      <tr>
        <td><strong>${c.name}</strong></td>
        <td>${c.email}</td>
        <td>${c.company || '—'}</td>
        <td>${c.phone || '—'}</td>
        <td>${formatDate(c.created_at)}</td>
        <td>
          <button class="btn btn-sm btn-outline" onclick="generateLicenseForCustomer('${c.id}')">+ License</button>
        </td>
      </tr>
    `).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" class="error-msg">Failed to load customers: ${err.message}</td></tr>`;
  }
}

async function loadDevices() {
  const tbody = document.querySelector('#devicesTable tbody');
  try {
    const devices = await apiRequest('/devices');
    if (devices.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="loading">No computers currently registered.</td></tr>`;
      return;
    }

    tbody.innerHTML = devices.map(d => `
      <tr>
        <td><strong>${d.machine_name}</strong></td>
        <td><code>${d.machine_fingerprint}</code></td>
        <td>${d.os_info}</td>
        <td><strong>${d.license_key || '—'}</strong></td>
        <td>${d.customer_name || '—'}</td>
        <td>${formatDate(d.last_checkin_at)}</td>
        <td>${d.is_active ? '<span class="badge badge-active">ACTIVE SEAT</span>' : '<span class="badge badge-expired">DEACTIVATED</span>'}</td>
        <td>
          ${d.is_active ? `<button class="btn btn-sm btn-danger" onclick="deactivateDevice('${d.id}')">Deactivate Seat</button>` : '—'}
        </td>
      </tr>
    `).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" class="error-msg">Failed to load devices: ${err.message}</td></tr>`;
  }
}

async function loadExpirations() {
  const tbody = document.querySelector('#expirationsTable tbody');
  try {
    const rep = await apiRequest('/reports/expirations?days_ahead=60');
    if (rep.licenses.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="loading">No licenses expiring in the next 60 days.</td></tr>`;
      return;
    }

    tbody.innerHTML = rep.licenses.map(l => `
      <tr>
        <td><strong>${l.customer_name}</strong></td>
        <td>${l.company || '—'}</td>
        <td><strong>${l.license_key}</strong></td>
        <td>${l.plan_type.replace('_', ' ').toUpperCase()}</td>
        <td>${formatDate(l.expires_at)}</td>
        <td><strong style="color: ${l.days_remaining <= 7 ? '#f87171' : '#fbbf24'};">${l.days_remaining} days</strong></td>
        <td>${formatBadge(l.status)}</td>
        <td>
          <button class="btn btn-sm btn-primary" onclick="openRenewModal('${l.license_id}', '${l.license_key}', '${escapeHtml(l.customer_name)}', '${l.expires_at}')">Renew Now</button>
        </td>
      </tr>
    `).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" class="error-msg">Failed to load expirations: ${err.message}</td></tr>`;
  }
}

async function loadAuditLogs() {
  const tbody = document.querySelector('#auditTable tbody');
  try {
    const logs = await apiRequest('/audit-logs?limit=50');
    if (logs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="loading">No audit events recorded yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = logs.map(a => `
      <tr>
        <td>${formatDate(a.timestamp)}</td>
        <td><code>${a.actor}</code></td>
        <td><strong>${a.action}</strong></td>
        <td>${a.resource_type}:${a.resource_id.substring(0, 8)}...</td>
        <td><small style="color: #94a3b8;">${a.details_json}</small></td>
        <td>${a.ip_address || '—'}</td>
      </tr>
    `).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" class="error-msg">Failed to load audit logs: ${err.message}</td></tr>`;
  }
}

// -----------------------------------------------------------------------------
// Actions
// -----------------------------------------------------------------------------

function populateCustomerDropdown() {
  const select = document.getElementById('licCustomerSelect');
  select.innerHTML = customersCache.map(c => `
    <option value="${c.id}">${c.name} (${c.company || c.email})</option>
  `).join('');
}

function generateLicenseForCustomer(customerId) {
  populateCustomerDropdown();
  document.getElementById('licCustomerSelect').value = customerId;
  openModal('licenseModal');
}

function openRenewModal(licenseId, key, custName, currentExp) {
  document.getElementById('renewLicId').value = licenseId;
  document.getElementById('renewKeyDisplay').textContent = key;
  document.getElementById('renewCustDisplay').textContent = custName || 'N/A';
  document.getElementById('renewCurrExpDisplay').textContent = formatDate(currentExp);
  openModal('renewModal');
}

async function toggleStatus(licenseId, currentStatus) {
  const newStatus = currentStatus === 'ACTIVE' ? 'SUSPENDED' : 'ACTIVE';
  if (!confirm(`Are you sure you want to change license status to ${newStatus}?`)) return;

  try {
    await apiRequest(`/licenses/${licenseId}/status`, 'PATCH', { status: newStatus, reason: 'Admin toggle' });
    loadLicenses();
    loadOverviewStats();
  } catch (err) {
    alert(`Status update failed: ${err.message}`);
  }
}

async function deactivateDevice(deviceId) {
  if (!confirm('Deactivating this computer will release the seat so the customer can activate a new machine. Continue?')) return;
  try {
    await apiRequest(`/devices/${deviceId}/deactivate`, 'POST');
    loadDevices();
    loadOverviewStats();
  } catch (err) {
    alert(`Deactivation failed: ${err.message}`);
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/'/g, "\\'");
}
