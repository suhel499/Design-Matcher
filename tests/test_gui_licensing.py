"""
PySide6 GUI tests for LicenseDialog and MainWindow licensing indicator.
Uses pytest-qt and QTest to verify user interface states and events.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication

from app.core.licensing.crypto import generate_keypair, sign_payload
from app.core.licensing.hardware import get_hardware_fingerprint
from app.core.licensing.manager import LicenseManager
from app.gui.license_dialog import LicenseDialog, show_subscription_expired_dialog
from app.gui.main_window import MainWindow


@pytest.fixture
def temp_lic_dir():
    d = tempfile.mkdtemp(prefix="dm_gui_lic_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def keypair():
    priv, pub = generate_keypair()
    return priv, pub


def test_license_dialog_unlicensed_ui(qtbot, temp_lic_dir, keypair):
    _, pub = keypair
    mgr = LicenseManager(data_dir=temp_lic_dir, public_key_pem=pub)

    dialog = LicenseDialog(mgr)
    qtbot.addWidget(dialog)

    assert "UNLICENSED" in dialog.status_title.text()
    assert dialog.hw_id_label.text() == get_hardware_fingerprint()
    assert dialog.deactivate_btn.isEnabled() is False


def test_license_dialog_active_license_ui(qtbot, temp_lic_dir, keypair):
    priv, pub = keypair
    mgr = LicenseManager(data_dir=temp_lic_dir, public_key_pem=pub)

    # Save a valid signed license
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(days=180)).isoformat()
    payload = {
        "license_id": "lic-gui-1",
        "license_key": "DM-2026-GUI1-TEST",
        "customer_id": "cust-gui",
        "customer_name": "Precision CNC Laserworks",
        "product_id": "design-matcher",
        "plan_type": "1_year",
        "issued_at": now.isoformat(),
        "expires_at": expires,
        "max_activations": 1,
        "status": "ACTIVE",
        "hardware_binding": get_hardware_fingerprint(),
        "features": ["dxf_matching", "pdf_catalog", "code_search", "dxf_export"],
        "nonce": "gui-nonce-1",
    }
    sig = sign_payload(payload, priv)
    signed_file = {
        "version": 1,
        "algorithm": "Ed25519",
        "payload": payload,
        "signature": sig,
    }
    mgr.save_signed_license(signed_file)

    dialog = LicenseDialog(mgr)
    qtbot.addWidget(dialog)

    assert "ACTIVE SUBSCRIPTION" in dialog.status_title.text()
    assert "Precision CNC Laserworks" in dialog.status_details.text()
    assert dialog.online_key_input.text() == "DM-2026-GUI1-TEST"
    assert dialog.deactivate_btn.isEnabled() is True


def test_main_window_license_indicator_updates(qtbot, temp_lic_dir, keypair, monkeypatch):
    priv, pub = keypair
    monkeypatch.setenv("DESIGN_MATCHER_PUBLIC_KEY", pub)

    window = MainWindow(data_dir=temp_lic_dir)
    qtbot.addWidget(window)

    # Initially unlicensed
    assert "DeActivated" in window.license_btn.text() or "Activate" in window.license_btn.text() or "Expired" in window.license_btn.text()

    # Clicking DeActivated button opens the License Activation page (tab index 3)
    window.license_btn.click()
    assert window.main_stack.currentIndex() == 3

    # Now activate a license
    now = datetime.now(timezone.utc)
    payload = {
        "license_id": "lic-gui-2",
        "license_key": "DM-2026-GUI2-ACTIVE",
        "customer_id": "cust-gui-2",
        "customer_name": "Alpha Cut Co",
        "product_id": "design-matcher",
        "plan_type": "1_year",
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(days=200)).isoformat(),
        "max_activations": 1,
        "status": "ACTIVE",
        "hardware_binding": get_hardware_fingerprint(),
        "features": ["dxf_matching", "pdf_catalog", "code_search", "dxf_export"],
        "nonce": "gui-nonce-2",
    }
    sig = sign_payload(payload, priv)
    window.license_manager.save_signed_license({
        "version": 1,
        "algorithm": "Ed25519",
        "payload": payload,
        "signature": sig,
    })

    window._update_license_indicator()
    assert "Activated" in window.license_btn.text() or "Active" in window.license_btn.text()


def test_dxf_preview_locked_when_unlicensed_or_expired(qtbot, temp_lic_dir, keypair, monkeypatch):
    priv, pub = keypair
    monkeypatch.setenv("DESIGN_MATCHER_PUBLIC_KEY", pub)

    dialog_opened = []
    def mock_expired_dialog(parent, mgr, msg):
        dialog_opened.append(msg)
    monkeypatch.setattr("app.gui.main_window.show_subscription_expired_dialog", mock_expired_dialog)

    win = MainWindow(data_dir=temp_lic_dir)
    qtbot.addWidget(win)

    # 1. Verify button is disabled / locked when unlicensed
    assert win.nav_tab_dxf_preview.isEnabled() is False
    assert "Locked" in win.nav_tab_dxf_preview.text()
    assert win.main_stack.currentIndex() == 0

    # 2. Click the DXF Preview button: validates license upon click and stays on Tab 0
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    QTest.mouseClick(win.nav_tab_dxf_preview, Qt.MouseButton.LeftButton)
    assert len(dialog_opened) == 1
    assert win.main_stack.currentIndex() == 0
    assert win.nav_tab_matcher.isChecked() is True
    assert win.nav_tab_dxf_preview.isChecked() is False

    # 3. Direct programmatic navigation also checks license and stays on Tab 0
    win._on_main_tab_changed(1)
    assert len(dialog_opened) == 2
    assert win.main_stack.currentIndex() == 0

    win.switch_to_dxf_preview("sample.dxf")
    assert len(dialog_opened) == 3
    assert win.main_stack.currentIndex() == 0


def test_dxf_preview_unlocked_with_active_license_and_auto_revert(qtbot, temp_lic_dir, keypair, monkeypatch):
    priv, pub = keypair
    monkeypatch.setenv("DESIGN_MATCHER_PUBLIC_KEY", pub)

    dialog_opened = []
    def mock_expired_dialog(parent, mgr, msg):
        dialog_opened.append(msg)
    monkeypatch.setattr("app.gui.main_window.show_subscription_expired_dialog", mock_expired_dialog)

    win = MainWindow(data_dir=temp_lic_dir)
    qtbot.addWidget(win)

    # Save active license
    now = datetime.now(timezone.utc)
    payload = {
        "license_id": "lic-dxf-ok",
        "license_key": "DM-2026-DXF-ACTIVE",
        "customer_id": "cust-dxf",
        "customer_name": "Laser Labs",
        "product_id": "design-matcher",
        "plan_type": "1_year",
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(days=90)).isoformat(),
        "max_activations": 1,
        "status": "ACTIVE",
        "hardware_binding": get_hardware_fingerprint(),
        "features": ["dxf_matching", "pdf_catalog", "code_search", "dxf_export"],
        "nonce": "nonce-dxf-active",
    }
    sig = sign_payload(payload, priv)
    win.license_manager.save_signed_license({
        "version": 1,
        "algorithm": "Ed25519",
        "payload": payload,
        "signature": sig,
    })

    win._update_license_indicator()

    # 1. Verify button is now enabled and unlocked
    assert win.nav_tab_dxf_preview.isEnabled() is True
    assert "2. DXF Preview" in win.nav_tab_dxf_preview.text()
    assert "Locked" not in win.nav_tab_dxf_preview.text()

    # 2. Clicking switches to Tab 1
    win.nav_tab_dxf_preview.click()
    assert win.main_stack.currentIndex() == 1
    assert win.nav_tab_dxf_preview.isChecked() is True
    assert len(dialog_opened) == 0

    # 3. Simulate license expiration/suspension while on Tab 1
    payload["status"] = "SUSPENDED"
    sig_susp = sign_payload(payload, priv)
    win.license_manager.save_signed_license({
        "version": 1,
        "algorithm": "Ed25519",
        "payload": payload,
        "signature": sig_susp,
    })

    # Trigger update
    win._update_license_indicator()

    # Verify user is immediately kicked back to Tab 0 and button is locked
    assert win.main_stack.currentIndex() == 0
    assert win.nav_tab_matcher.isChecked() is True
    assert win.nav_tab_dxf_preview.isEnabled() is False
    assert "Locked" in win.nav_tab_dxf_preview.text()
