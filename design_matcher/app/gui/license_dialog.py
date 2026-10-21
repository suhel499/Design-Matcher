"""
PySide6 License & Subscription Management Dialog for Design Matcher.
Enables online activation, renewal sync, seat deactivation, offline request export,
and air-gapped signed license file import with an enterprise-grade dark theme layout.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.licensing.hardware import get_device_info, get_hardware_fingerprint
from app.core.licensing.manager import LicenseManager
from app.core.licensing.models import LicenseStatus

logger = logging.getLogger(__name__)

LICENSE_STYLESHEET = """
QDialog {
    background-color: #0F172A;
    color: #F8FAFC;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

QLabel {
    color: #F8FAFC;
    font-size: 12px;
    border: none;
    background: transparent;
}

QLabel#dialogTitle {
    font-size: 17px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.3px;
}

QLabel#dialogSubtitle {
    font-size: 12px;
    color: #94A3B8;
}

/* Card Containers */
QFrame#hwCard {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 16px;
}

QFrame#statusBanner {
    border-radius: 8px;
    padding: 14px 18px;
}

QFrame#stepCard {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 16px;
}

/* Form Inputs */
QLineEdit {
    background-color: #0F172A;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 8px 12px;
    color: #FFFFFF;
    font-size: 13px;
    selection-background-color: #2563EB;
}

QLineEdit:focus {
    border: 1.5px solid #38BDF8;
    background-color: #0F172A;
}

QLineEdit:disabled {
    background-color: #1E293B;
    color: #64748B;
    border-color: #334155;
}

/* Tab Bar */
QTabWidget::pane {
    border: 1px solid #334155;
    border-radius: 0 0 8px 8px;
    background-color: #1E293B;
    top: -1px;
}

QTabBar::tab {
    background: #0F172A;
    border: 1px solid #334155;
    border-bottom: none;
    padding: 9px 20px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #94A3B8;
    font-size: 12px;
    font-weight: 600;
}

QTabBar::tab:selected {
    background: #1E293B;
    color: #38BDF8;
    border-top: 2px solid #38BDF8;
    border-bottom: 1px solid #1E293B;
}

QTabBar::tab:hover:!selected {
    background: #162032;
    color: #F1F5F9;
}

/* Buttons */
QPushButton {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 9px 18px;
    font-size: 12px;
    font-weight: 600;
    min-height: 18px;
}

QPushButton:hover {
    background-color: #1D4ED8;
}

QPushButton:pressed {
    background-color: #1E40AF;
}

QPushButton:disabled {
    background-color: #334155;
    color: #64748B;
}

QPushButton#secondaryBtn {
    background-color: #1E293B;
    color: #E2E8F0;
    border: 1px solid #475569;
}

QPushButton#secondaryBtn:hover {
    background-color: #334155;
    border-color: #64748B;
    color: #FFFFFF;
}

QPushButton#copyBtn {
    background-color: #0F172A;
    color: #38BDF8;
    border: 1px solid #0284C7;
    font-size: 11px;
    font-weight: 600;
    padding: 6px 12px;
}

QPushButton#copyBtn:hover {
    background-color: #0369A1;
    color: #FFFFFF;
}

QPushButton#dangerBtn {
    background-color: rgba(239, 68, 68, 0.12);
    color: #F87171;
    border: 1px solid #EF4444;
    font-size: 12px;
}

QPushButton#dangerBtn:hover {
    background-color: #DC2626;
    color: #FFFFFF;
}

QPushButton#dangerBtn:disabled {
    background-color: transparent;
    color: #64748B;
    border-color: #334155;
}
"""


class LicenseDialog(QDialog):
    """License and subscription status dialog for Design Matcher."""

    license_updated = Signal()

    def __init__(self, license_manager: LicenseManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.license_manager = license_manager
        self.setWindowTitle("Design Matcher — License & Subscription Management")
        self.resize(680, 680)
        self.setMinimumSize(660, 660)
        self.setStyleSheet(LICENSE_STYLESHEET)
        self._setup_ui()
        self.refresh_status()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 20)
        main_layout.setSpacing(14)

        # 1. Header with Icon
        header_layout = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.setSpacing(3)
        title = QLabel("🛡️ LICENSE & SUBSCRIPTION AUTHORITY", self)
        title.setObjectName("dialogTitle")
        header_text.addWidget(title)
        header_layout.addLayout(header_text)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # 2. Status Banner Card
        self.status_banner = QFrame(self)
        self.status_banner.setObjectName("statusBanner")
        banner_layout = QVBoxLayout(self.status_banner)
        banner_layout.setContentsMargins(16, 14, 16, 14)
        banner_layout.setSpacing(6)

        self.status_title = QLabel("Checking license...", self.status_banner)
        self.status_title.setStyleSheet("font-size: 14px; font-weight: 700; border: none; background: transparent;")
        self.status_details = QLabel("", self.status_banner)
        self.status_details.setStyleSheet("font-size: 12px; line-height: 1.4; border: none; background: transparent;")
        self.status_details.setWordWrap(True)

        banner_layout.addWidget(self.status_title)
        banner_layout.addWidget(self.status_details)
        main_layout.addWidget(self.status_banner)

        # 3. Hardware Fingerprint Card
        hw_card = QFrame(self)
        hw_card.setObjectName("hwCard")
        hw_layout = QVBoxLayout(hw_card)
        hw_layout.setContentsMargins(14, 12, 14, 12)
        hw_layout.setSpacing(8)

        hw_header_row = QHBoxLayout()
        hw_caption = QLabel("💻 CURRENT COMPUTER HARDWARE IDENTIFIER", hw_card)
        hw_caption.setStyleSheet("font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.5px;")
        hw_header_row.addWidget(hw_caption)
        hw_header_row.addStretch()
        hw_layout.addLayout(hw_header_row)

        hw_val_row = QHBoxLayout()
        self.hw_id_label = QLabel(get_hardware_fingerprint(), hw_card)
        self.hw_id_label.setStyleSheet(
            "font-family: Menlo, Monaco, Consolas, monospace; font-size: 13px; font-weight: 700; "
            "color: #38BDF8; background-color: #0B1120; border: 1px solid #1E293B; "
            "border-radius: 6px; padding: 7px 12px;"
        )
        self.hw_id_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        copy_hw_btn = QPushButton("📋 Copy ID", hw_card)
        copy_hw_btn.setObjectName("copyBtn")
        copy_hw_btn.setFixedWidth(100)
        copy_hw_btn.setMinimumHeight(32)
        copy_hw_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_hw_btn.clicked.connect(self._copy_hardware_id)

        hw_val_row.addWidget(self.hw_id_label, 1)
        hw_val_row.addWidget(copy_hw_btn)
        hw_layout.addLayout(hw_val_row)

        main_layout.addWidget(hw_card)

        # 4. Tabbed Actions
        self.tabs = QTabWidget(self)
        self.tabs.setMinimumHeight(320)

        # --- Tab 1: Online Activation ---
        tab_online = QWidget()
        online_layout = QVBoxLayout(tab_online)
        online_layout.setContentsMargins(20, 20, 20, 20)
        online_layout.setSpacing(14)

        # Key Input
        key_group = QVBoxLayout()
        key_group.setSpacing(6)
        key_label = QLabel("License Key:", tab_online)
        key_label.setStyleSheet("font-weight: 600; color: #E2E8F0; font-size: 12px;")
        self.online_key_input = QLineEdit(tab_online)
        self.online_key_input.setPlaceholderText("DM-YYYY-XXXX-XXXX-XXXX")
        self.online_key_input.setMinimumHeight(38)
        self.online_key_input.setStyleSheet("font-family: Menlo, Monaco, Consolas, monospace; font-weight: 600; font-size: 13px;")
        key_group.addWidget(key_label)
        key_group.addWidget(self.online_key_input)
        online_layout.addLayout(key_group)

        # Server URL Input
        server_group = QVBoxLayout()
        server_group.setSpacing(6)
        server_label = QLabel("Licensing Server Authority URL:", tab_online)
        server_label.setStyleSheet("font-weight: 600; color: #94A3B8; font-size: 11px;")
        self.online_server_input = QLineEdit(self.license_manager.server_url, tab_online)
        self.online_server_input.setMinimumHeight(36)
        server_group.addWidget(server_label)
        server_group.addWidget(self.online_server_input)
        online_layout.addLayout(server_group)

        # Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self.activate_btn = QPushButton("🚀 Activate Computer Online", tab_online)
        self.activate_btn.setMinimumHeight(40)
        self.activate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.activate_btn.clicked.connect(self._activate_online)

        self.refresh_btn = QPushButton("🔄 Sync / Refresh License", tab_online)
        self.refresh_btn.setObjectName("secondaryBtn")
        self.refresh_btn.setMinimumHeight(40)
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.clicked.connect(self._validate_online)

        btn_row.addWidget(self.activate_btn, 3)
        btn_row.addWidget(self.refresh_btn, 2)
        online_layout.addLayout(btn_row)

        online_layout.addSpacing(6)

        self.deactivate_btn = QPushButton("⚠️ Deactivate Computer (Release Seat for Transfer)", tab_online)
        self.deactivate_btn.setObjectName("dangerBtn")
        self.deactivate_btn.setMinimumHeight(38)
        self.deactivate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.deactivate_btn.clicked.connect(self._deactivate_online)
        online_layout.addWidget(self.deactivate_btn)

        online_layout.addStretch()
        self.tabs.addTab(tab_online, "🌐 Online Activation && Sync")

        # --- Tab 2: Offline Air-Gapped Activation ---
        tab_offline = QWidget()
        offline_layout = QVBoxLayout(tab_offline)
        offline_layout.setContentsMargins(18, 18, 18, 18)
        offline_layout.setSpacing(12)

        # Step 1 Card
        step1_card = QFrame(tab_offline)
        step1_card.setObjectName("stepCard")
        step1_layout = QHBoxLayout(step1_card)
        step1_layout.setContentsMargins(14, 12, 14, 12)
        step1_layout.setSpacing(14)

        step1_info = QVBoxLayout()
        step1_title = QLabel("Step 1: Export Activation Request (.dmreq)", step1_card)
        step1_title.setStyleSheet("font-weight: 700; font-size: 13px; color: #38BDF8;")
        step1_desc = QLabel("Export an offline activation request bound to this PC to send to your administrator.", step1_card)
        step1_desc.setStyleSheet("color: #94A3B8; font-size: 11px;")
        step1_desc.setWordWrap(True)
        step1_info.addWidget(step1_title)
        step1_info.addWidget(step1_desc)

        self.export_req_btn = QPushButton("💾 Export .dmreq", step1_card)
        self.export_req_btn.setObjectName("secondaryBtn")
        self.export_req_btn.setFixedWidth(140)
        self.export_req_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_req_btn.clicked.connect(self._export_offline_request)

        step1_layout.addLayout(step1_info, 1)
        step1_layout.addWidget(self.export_req_btn)
        offline_layout.addWidget(step1_card)

        # Step 2 Card
        step2_card = QFrame(tab_offline)
        step2_card.setObjectName("stepCard")
        step2_layout = QHBoxLayout(step2_card)
        step2_layout.setContentsMargins(14, 12, 14, 12)
        step2_layout.setSpacing(14)

        step2_info = QVBoxLayout()
        step2_title = QLabel("Step 2: Import Signed License File (.lic)", step2_card)
        step2_title.setStyleSheet("font-weight: 700; font-size: 13px; color: #34D399;")
        step2_desc = QLabel("Import the Ed25519 digitally signed license response file generated by your admin.", step2_card)
        step2_desc.setStyleSheet("color: #94A3B8; font-size: 11px;")
        step2_desc.setWordWrap(True)
        step2_info.addWidget(step2_title)
        step2_info.addWidget(step2_desc)

        self.import_lic_btn = QPushButton("📥 Import .lic File", step2_card)
        self.import_lic_btn.setFixedWidth(140)
        self.import_lic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_lic_btn.clicked.connect(self._import_offline_license)

        step2_layout.addLayout(step2_info, 1)
        step2_layout.addWidget(self.import_lic_btn)
        offline_layout.addWidget(step2_card)

        offline_layout.addStretch()
        self.tabs.addTab(tab_offline, "📁 Offline Air-Gapped Activation")

        main_layout.addWidget(self.tabs, 1)

        # 5. Footer with Security Seal & Close Button
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 10, 0, 0)
        security_note = QLabel("🔒 Ed25519 Cryptographically Authoritative • Hardware Bound", self)
        security_note.setStyleSheet("font-size: 11px; color: #64748B;")
        footer.addWidget(security_note)
        footer.addStretch()

        close_btn = QPushButton("Close", self)
        close_btn.setObjectName("secondaryBtn")
        close_btn.setFixedWidth(90)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        main_layout.addLayout(footer)

    def refresh_status(self) -> None:
        """Updates the status banner and controls based on current license verification."""
        verdict = self.license_manager.check_license()

        if verdict.is_valid:
            days = verdict.days_remaining
            if days <= 7:
                # Expiring soon warning
                self.status_banner.setStyleSheet(
                    "QFrame#statusBanner { background-color: rgba(245, 158, 11, 0.15); border: 1.5px solid #F59E0B; }"
                )
                self.status_title.setText(f"⚠️ SUBSCRIPTION EXPIRING SOON ({days} DAYS REMAINING)")
                self.status_title.setStyleSheet("color: #FBBF24; font-size: 14px; font-weight: 700; border: none;")
            else:
                # Active
                self.status_banner.setStyleSheet(
                    "QFrame#statusBanner { background-color: rgba(16, 185, 129, 0.15); border: 1.5px solid #10B981; }"
                )
                self.status_title.setText("✅ ACTIVE SUBSCRIPTION")
                self.status_title.setStyleSheet("color: #34D399; font-size: 14px; font-weight: 700; border: none;")

            exp_str = verdict.expires_at.strftime("%Y-%m-%d %H:%M UTC") if verdict.expires_at else "N/A"
            self.status_details.setText(
                f"License Key: {verdict.license_key}   •   Customer: {verdict.customer_name}\n"
                f"Expiration: {exp_str} ({days} days remaining)"
            )
            self.status_details.setStyleSheet("color: #E2E8F0; font-size: 12px; border: none;")
            self.online_key_input.setText(verdict.license_key)
            self.deactivate_btn.setEnabled(True)
            self.refresh_btn.setEnabled(True)
        else:
            # Invalid or Expired or Suspended
            if verdict.status.value == "SUSPENDED":
                self.status_banner.setStyleSheet(
                    "QFrame#statusBanner { background-color: rgba(234, 88, 12, 0.18); border: 1.5px solid #EA580C; }"
                )
                self.status_title.setText("⚠️ SUBSCRIPTION SUSPENDED")
                self.status_title.setStyleSheet("color: #FB923C; font-size: 14px; font-weight: 700; border: none;")
                self.status_details.setText(
                    f"{verdict.message}\n"
                    "Your saved DXF designs, catalogs, and databases remain safely preserved. Contact administrator to resolve hold."
                )
                self.status_details.setStyleSheet("color: #FED7AA; font-size: 12px; border: none;")
            elif verdict.status.value == "REVOKED":
                self.status_banner.setStyleSheet(
                    "QFrame#statusBanner { background-color: rgba(225, 29, 72, 0.18); border: 1.5px solid #E11D48; }"
                )
                self.status_title.setText("🔒 SUBSCRIPTION REVOKED")
                self.status_title.setStyleSheet("color: #FDA4AF; font-size: 14px; font-weight: 700; border: none;")
                self.status_details.setText(
                    f"{verdict.message}\n"
                    "Saved DXF files and catalog indexes are safely preserved. Please contact administrator."
                )
                self.status_details.setStyleSheet("color: #FECDD3; font-size: 12px; border: none;")
            elif verdict.is_expired:
                self.status_banner.setStyleSheet(
                    "QFrame#statusBanner { background-color: rgba(239, 68, 68, 0.18); border: 1.5px solid #EF4444; }"
                )
                self.status_title.setText("🔒 SUBSCRIPTION EXPIRED")
                self.status_title.setStyleSheet("color: #F87171; font-size: 14px; font-weight: 700; border: none;")
                self.status_details.setText(
                    f"{verdict.message}\n"
                    "Your saved catalogs, DXF files, and indexes remain intact. Renew your subscription or import a new .lic file to continue."
                )
                self.status_details.setStyleSheet("color: #FCA5A5; font-size: 12px; border: none;")
            elif verdict.is_rollback_detected:
                self.status_banner.setStyleSheet(
                    "QFrame#statusBanner { background-color: rgba(245, 158, 11, 0.18); border: 1.5px solid #F59E0B; }"
                )
                self.status_title.setText("⚠️ SYSTEM CLOCK ROLLBACK DETECTED")
                self.status_title.setStyleSheet("color: #FBBF24; font-size: 14px; font-weight: 700; border: none;")
                self.status_details.setText(
                    f"{verdict.message}\n"
                    "Please restore your computer's date and time settings to re-enable paid matching features."
                )
                self.status_details.setStyleSheet("color: #FDE68A; font-size: 12px; border: none;")
            elif verdict.is_device_mismatched:
                self.status_banner.setStyleSheet(
                    "QFrame#statusBanner { background-color: rgba(245, 158, 11, 0.18); border: 1.5px solid #F59E0B; }"
                )
                self.status_title.setText("⚠️ COMPUTER HARDWARE SEAT MISMATCH")
                self.status_title.setStyleSheet("color: #FBBF24; font-size: 14px; font-weight: 700; border: none;")
                self.status_details.setText(
                    f"{verdict.message}\n"
                    "Please transfer your license seat or activate on the designated machine."
                )
                self.status_details.setStyleSheet("color: #FDE68A; font-size: 12px; border: none;")
            else:
                self.status_banner.setStyleSheet(
                    "QFrame#statusBanner { background-color: rgba(239, 68, 68, 0.15); border: 1.5px solid #DC2626; }"
                )
                self.status_title.setText("🔒 UNLICENSED APPLICATION")
                self.status_title.setStyleSheet("color: #F87171; font-size: 14px; font-weight: 700; border: none;")
                self.status_details.setText(
                    f"{verdict.message}\n"
                    "Saved DXF files and catalog indexes are safely preserved. Please enter a valid license key or import a signed .lic file."
                )
                self.status_details.setStyleSheet("color: #FCA5A5; font-size: 12px; border: none;")

            self.deactivate_btn.setEnabled(False)

    def _copy_hardware_id(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(get_hardware_fingerprint())
            QMessageBox.information(self, "Hardware ID Copied", "Computer Hardware ID copied to clipboard.")

    def _activate_online(self) -> None:
        key = self.online_key_input.text().strip()
        server_url = self.online_server_input.text().strip()
        if not key:
            QMessageBox.warning(self, "Missing Key", "Please enter your license key.")
            return

        self.activate_btn.setEnabled(False)
        self.activate_btn.setText("Activating with server...")
        QApplication.processEvents()

        try:
            ok, msg = self.license_manager.activate_online(key, server_url)
            if ok:
                QMessageBox.information(self, "Activation Successful", "License activated and verified successfully!")
                self.refresh_status()
                self.license_updated.emit()
            else:
                QMessageBox.critical(self, "Activation Failed", f"Activation failed:\n{msg}")
        finally:
            self.activate_btn.setEnabled(True)
            self.activate_btn.setText("🚀 Activate Computer Online")

    def _validate_online(self) -> None:
        server_url = self.online_server_input.text().strip()
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("Checking...")
        QApplication.processEvents()

        try:
            verdict = self.license_manager.validate_online(server_url)
            self.refresh_status()
            self.license_updated.emit()
            if verdict.is_valid:
                QMessageBox.information(self, "License Refreshed", f"License verified!\n{verdict.message}")
            else:
                QMessageBox.warning(self, "Validation Alert", f"License check returned:\n{verdict.message}")
        finally:
            self.refresh_btn.setEnabled(True)
            self.refresh_btn.setText("🔄 Sync / Refresh License")

    def _deactivate_online(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Confirm Computer Deactivation",
            "Deactivating this computer will release the seat so you can install it on another computer.\n\n"
            "Paid operations on this computer will be locked until reactivated.\n\n"
            "Do you wish to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        server_url = self.online_server_input.text().strip()
        ok, msg = self.license_manager.deactivate_online(server_url)
        if ok:
            QMessageBox.information(self, "Seat Deactivated", "Computer seat deactivated successfully.")
            self.refresh_status()
            self.license_updated.emit()
        else:
            QMessageBox.critical(self, "Deactivation Failed", f"Could not deactivate seat:\n{msg}")

    def _export_offline_request(self) -> None:
        key = self.online_key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "Missing Key", "Please enter your license key above before generating request.")
            return

        req_json = self.license_manager.generate_offline_request(key)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Activation Request File",
            f"request_{key}.dmreq",
            "Activation Request Files (*.dmreq);;All Files (*)",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(req_json)
                QMessageBox.information(
                    self,
                    "Request Exported",
                    f"Activation request saved to:\n{path}\n\n"
                    "Send this file to your administrator to generate your signed license response.",
                )
            except Exception as exc:
                QMessageBox.critical(self, "Export Failed", f"Failed saving file: {exc}")

    def _import_offline_license(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Signed License File",
            "",
            "License Files (*.lic *.json);;All Files (*)",
        )
        if not path:
            return

        ok, msg = self.license_manager.import_offline_license(path)
        if ok:
            QMessageBox.information(self, "Offline License Imported", "Signed license validated and applied successfully!")
            self.refresh_status()
            self.license_updated.emit()
        else:
            QMessageBox.critical(self, "Import Failed", f"Invalid license file:\n{msg}")


def show_subscription_expired_dialog(parent: QWidget, license_manager: LicenseManager, message: str) -> None:
    """
    Displays a clear subscription expired modal explaining that user files and databases
    are safe, and providing an immediate button to open the License Dialog.
    """
    box = QMessageBox(parent)
    box.setWindowTitle("Design Matcher - Subscription Expired")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setText("<h3>Subscription Expired or Unlicensed</h3>")
    box.setInformativeText(
        f"{message}\n\n"
        "Your saved catalogs, DXF files, indexes, and search history remain 100% safe and intact.\n\n"
        "To resume design matching and indexing, please renew your subscription or import a signed license file."
    )
    manage_btn = box.addButton("Manage / Renew License...", QMessageBox.ButtonRole.ActionRole)
    box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
    box.exec()

    if box.clickedButton() == manage_btn:
        dlg = LicenseDialog(license_manager, parent)
        dlg.exec()
