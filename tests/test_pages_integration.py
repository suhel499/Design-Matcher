"""
Tests for all 10 In-App Pages and MainWindow Integration.
Verifies functionality for:
1. EulaPage (First-run acceptance and review)
2. LicensePage (Activation, plan status, hardware fingerprint)
3. SettingsPage (Catalog folders, thresholds, language, theme)
4. CatalogManagerPage (Indexed counts, files table, reindex signals)
5. HistoryPage (Past searches, deletion, rerun signal)
6. HelpPage (Quick start guide and interactive diagnostics)
7. UpdatesPage (Update checking, changelog)
8. BackupPage (Backup archive, restore, and factory reset)
9. AboutPage (Version specs and third-party licenses)
10. SupportPage (Problem report composer and zero-PII log sanitizer)
11. MainWindow page navigation and search history logging
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget

from app.core.db import CatalogDatabase
from app.core.index import CatalogIndex
from app.core.licensing.manager import LicenseManager
from app.gui.main_window import MainWindow
from app.gui.pages import (
    AboutPage,
    BackupPage,
    CatalogManagerPage,
    ContactPage,
    EulaPage,
    HelpPage,
    HistoryPage,
    LicensePage,
    PrivacyPolicyPage,
    SettingsPage,
    SupportPage,
    TermsConditionsPage,
    UpdatesPage,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_eula_page(qtbot):
    page = EulaPage()
    qtbot.addWidget(page)

    assert not page.btn_continue.isEnabled()
    page.accept_check.setChecked(True)
    assert page.btn_continue.isEnabled()

    accepted_emitted = []
    page.accepted.connect(lambda: accepted_emitted.append(True))
    page.btn_continue.click()
    assert len(accepted_emitted) == 1

    # Test review mode when already accepted
    page.set_already_accepted(True)
    assert not page.btn_back.isHidden()
    assert page.accept_check.isChecked()


def test_license_page(qtbot, temp_dir):
    mgr = LicenseManager(data_dir=temp_dir)
    page = LicensePage(license_manager=mgr)
    qtbot.addWidget(page)

    assert page.lbl_hwid.text() != ""
    assert page.lbl_status.text() != ""


def test_settings_page(qtbot, temp_dir):
    db = CatalogDatabase(temp_dir / "catalog.db")
    idx = CatalogIndex(data_dir=temp_dir)
    page = SettingsPage(db=db, catalog_index=idx)
    qtbot.addWidget(page)

    # Verify layout utilises all width space with margins <= 50px
    assert page.settings_container.maximumWidth() >= 10000
    page_body = page.findChild(QWidget, "pageBody")
    assert page_body is not None
    margins = page_body.layout().contentsMargins()
    assert margins.left() <= 50
    assert margins.right() <= 50

    page.txt_folders.setPlainText("/path/to/dxf_folder")
    page.slider_thresh.setValue(82)
    page.spin_topk.setValue(45)
    page.slider_chamfer.setValue(15)
    page.spin_cand_pool.setValue(60)
    page.chk_rotations.setChecked(False)

    saved_signals = []
    page.settings_saved.connect(lambda: saved_signals.append(True))
    page._save_values(show_alert=False)

    assert len(saved_signals) == 1
    assert db.get_setting("catalog_folders") == "/path/to/dxf_folder"
    assert db.get_setting("similarity_threshold") == "0.82"
    assert db.get_setting("top_k") == "45"
    assert db.get_setting("chamfer_threshold") == "0.15"
    assert db.get_setting("candidate_limit") == "60"
    assert db.get_setting("allow_rotations") == "0"


def test_catalog_manager_page(qtbot, temp_dir):
    idx = CatalogIndex(data_dir=temp_dir)
    dxf_db = MagicMock()
    dxf_db.get_stats.return_value = {"total_files": 12, "total_components": 85}
    dxf_db.get_all_files.return_value = [
        {"file_name": "test1.dxf", "file_type": "DXF", "components_count": 5, "file_path": "/tmp/test1.dxf"}
    ]

    page = CatalogManagerPage(catalog_index=idx, dxf_db=dxf_db)
    qtbot.addWidget(page)

    assert page.table_files.rowCount() == 1
    item = page.table_files.item(0, 0)
    assert item is not None
    assert item.text() == "test1.dxf"

    dxf_sig = []
    pdf_sig = []
    page.reindex_dxf_requested.connect(lambda: dxf_sig.append(True))
    page.reindex_pdf_requested.connect(lambda: pdf_sig.append(True))

    page.reindex_dxf_requested.emit()
    page.reindex_pdf_requested.emit()
    assert len(dxf_sig) == 1
    assert len(pdf_sig) == 1


def test_history_page(qtbot, temp_dir):
    db = CatalogDatabase(temp_dir / "catalog.db")
    db.add_search_history(
        search_mode="mode_a",
        query_text="gate_stencil.png",
        results_count=15,
        top_match_code="PD 101",
        top_match_score=0.88,
    )

    page = HistoryPage(db=db)
    qtbot.addWidget(page)

    assert page.table.rowCount() == 1
    item_query = page.table.item(0, 3)
    assert item_query is not None
    assert item_query.text() == "gate_stencil.png"
    item_top = page.table.item(0, 5)
    assert item_top is not None
    assert "PD 101" in item_top.text()

    reruns = []
    page.rerun_search.connect(lambda mode, query: reruns.append((mode, query)))
    page.table.selectRow(0)
    page._rerun_selected()
    assert len(reruns) == 1
    assert reruns[0][0] == "mode_a"
    assert reruns[0][1] == "gate_stencil.png"

    # Verify edge cases: row with None items does not crash _rerun_selected or _delete_selected
    page.table.setRowCount(2)  # row 1 will have None items
    page.table.selectRow(1)
    page._rerun_selected()
    assert len(reruns) == 1  # No new rerun emitted, no crash

    page._delete_selected()  # Should safely ignore row 1 with None item

    # Normal delete of row 0
    page.table.selectRow(0)
    page._delete_selected()
    assert page.table.rowCount() == 0


def test_help_page_diagnostics(qtbot):
    page = HelpPage()
    qtbot.addWidget(page)

    page._run_diagnostics()
    text = page.txt_diag.toHtml()
    assert "SQLite" in text
    assert "PyTorch" in text or "ezdxf" in text or "FAISS" in text


def test_updates_page(qtbot):
    page = UpdatesPage()
    qtbot.addWidget(page)

    assert "v" in page.lbl_installed.text()
    page._on_update_checked(show_alert=False)
    assert page.txt_changelog.toPlainText() != ""


def test_backup_page(qtbot, temp_dir):
    idx = CatalogIndex(data_dir=temp_dir)
    # create a dummy file in data_dir
    (temp_dir / "test.txt").write_text("hello", encoding="utf-8")

    page = BackupPage(catalog_index=idx)
    qtbot.addWidget(page)

    export_zip = temp_dir / "backup.zip"
    page._perform_backup(str(export_zip), show_alert=False)
    assert export_zip.is_file()


def test_about_page(qtbot):
    page = AboutPage()
    qtbot.addWidget(page)

    assert "Design Matcher" in page.about_text.toPlainText() or "Design Matcher" in page.about_text.toHtml()
    assert "PySide6" in page.lic_text.toPlainText()


def test_support_page_log_sanitizer(qtbot):
    page = SupportPage()
    qtbot.addWidget(page)

    # Test sanitizing raw logs containing personal paths, username, and license keys
    user_name = os.getenv("USER", "testuser")
    raw_sample = f"User /Users/{user_name}/Documents/cad.dxf loaded key DM-PRO-9999-ABCD-1234-WXYZ with IP 192.168.1.55 and email test@company.com"
    sanitized = page._sanitize_logs(raw_sample)

    assert user_name not in sanitized
    assert "/Users/" not in sanitized
    assert "DM-PRO-" not in sanitized
    assert "[LICENSE_KEY_REDACTED]" in sanitized
    assert "[EMAIL_REDACTED]" in sanitized
    assert "[IP_REDACTED]" in sanitized


def test_contact_page_founder_details(qtbot):
    page = ContactPage()
    qtbot.addWidget(page)

    from PySide6.QtWidgets import QLabel
    labels = page.findChildren(QLabel)
    all_text = " ".join([l.text() for l in labels])

    assert "suhel.mulani90@gmail.com" in all_text
    assert "9422602386" in all_text
    assert "nihal.mulani@gmail.com" in all_text
    assert "9423591358" in all_text
    assert "Suhel Mulani" in all_text
    assert "Nihal Mulani" in all_text


def test_contact_page_form_submission_success(qtbot, monkeypatch):
    from unittest.mock import MagicMock
    from PySide6.QtWidgets import QMessageBox

    # Mock QMessageBox.information to prevent blocking modal
    info_mock = MagicMock()
    monkeypatch.setattr(QMessageBox, "information", info_mock)

    from app.core.db import CatalogDatabase
    mock_db = MagicMock(spec=CatalogDatabase)
    mock_db.save_contact_submission.return_value = "ticket-uuid-12345"
    mock_db.is_mysql = True

    page = ContactPage(db=mock_db)
    qtbot.addWidget(page)

    assert page.btn_submit.text() == "Submit"

    page.txt_name.setText("Rajesh Kumar")
    page.txt_email.setText("rajesh@steelworks.com")
    page.txt_summary.setText("Feature Request for CAD Linework")
    page.txt_desc.setText("Can you add automated DXF scaling options for 200 DPI drawings?")

    page._submit_inquiry()

    # Verify popup shown and form cleared
    assert info_mock.called
    call_msg = info_mock.call_args[0][2]
    assert "Thank you, Rajesh Kumar!" in call_msg
    assert "Your inquiry has been securely Submited" in call_msg
    assert "Reference Ticket ID:" in call_msg
    assert "Our engineering team will review it and reply earlyer" in call_msg

    assert page.txt_name.text() == ""
    assert page.txt_email.text() == ""
    assert page.txt_summary.text() == ""
    assert page.txt_desc.toPlainText() == ""


def test_contact_page_validation(qtbot, monkeypatch):
    from unittest.mock import MagicMock
    from PySide6.QtWidgets import QMessageBox

    warn_mock = MagicMock()
    monkeypatch.setattr("app.gui.pages.QMessageBox.warning", warn_mock)
    monkeypatch.setattr(QMessageBox, "warning", warn_mock)

    page = ContactPage()
    qtbot.addWidget(page)

    # Missing fields
    page.txt_name.setText("")
    page._submit_inquiry()
    assert warn_mock.called

    # Malicious script tag
    warn_mock.reset_mock()
    page.txt_name.setText("Valid Name")
    page.txt_email.setText("user@domain.com")
    page.txt_summary.setText("Valid Subject")
    page.txt_desc.setPlainText("<script>alert('xss')</script> test message here")
    page._submit_inquiry()
    assert warn_mock.called


def test_contact_page_server_url_handling(qtbot, monkeypatch):
    from unittest.mock import MagicMock
    from PySide6.QtWidgets import QMessageBox

    info_mock = MagicMock()
    monkeypatch.setattr(QMessageBox, "information", info_mock)

    # Case 1: license_manager is None (must not raise AttributeError)
    mock_db = MagicMock()
    mock_db.save_contact_submission.return_value = "ticket-none-lm"
    page_none = ContactPage(db=mock_db, license_manager=None, server_url=None)
    qtbot.addWidget(page_none)

    page_none.txt_name.setText("Test User")
    page_none.txt_email.setText("test@example.com")
    page_none.txt_summary.setText("Subject")
    page_none.txt_desc.setText("Testing inquiry with None license_manager")
    page_none._submit_inquiry()
    assert info_mock.called

    # Case 2: license_manager is provided with server_url
    info_mock.reset_mock()
    mock_lm = MagicMock()
    mock_lm.server_url = "http://lm-server:8080"
    page_lm = ContactPage(db=mock_db, license_manager=mock_lm, server_url="http://page-server:8080")
    qtbot.addWidget(page_lm)

    page_lm.txt_name.setText("Test User")
    page_lm.txt_email.setText("test@example.com")
    page_lm.txt_summary.setText("Subject")
    page_lm.txt_desc.setText("Testing inquiry with license_manager server_url")
    page_lm._submit_inquiry()
    assert info_mock.called


def test_privacy_and_terms_pages(qtbot):
    privacy_page = PrivacyPolicyPage()
    qtbot.addWidget(privacy_page)
    assert "PRIVACY POLICY" in privacy_page.txt_privacy.toPlainText()
    assert "100% On-Device" in privacy_page.txt_privacy.toPlainText()

    terms_page = TermsConditionsPage()
    qtbot.addWidget(terms_page)
    assert "TERMS & CONDITIONS" in terms_page.txt_terms.toPlainText()
    assert "Commercial License Grant" in terms_page.txt_terms.toPlainText()


def test_main_window_navigation_and_pages(qtbot, temp_dir):
    win = MainWindow(data_dir=temp_dir)
    qtbot.addWidget(win)

    # All 13 views in main_stack:
    # 0: Matcher, 1: DXF Preview, 2: EULA, 3: License, 4: Settings, 5: Help,
    # 6: Updates, 7: Backup, 8: About, 9: Contact, 10: Privacy, 11: Terms, 12: Catalog Library
    assert win.main_stack.count() == 13

    # Navigate to Settings (page 4)
    win.open_page(4)
    assert win.main_stack.currentIndex() == 4

    # Navigate to Help (page 5)
    win.open_page(5)
    assert win.main_stack.currentIndex() == 5

    # Navigate to Contact (page 9)
    win.open_page(9)
    assert win.main_stack.currentIndex() == 9

    # Navigate to Privacy Policy (page 10)
    win.open_page(10)
    assert win.main_stack.currentIndex() == 10

    # Navigate to Terms & Conditions (page 11)
    win.open_page(11)
    assert win.main_stack.currentIndex() == 11

    # Navigate back to Matcher (page 0)
    win.open_page(0)
    assert win.main_stack.currentIndex() == 0
    assert win.nav_tab_matcher.isChecked()

    # Test Theme Switching (Dark <-> Light)
    win.apply_theme(is_dark=False)
    assert win.is_dark is False
    assert win.catalog_index.db.get_setting("interface_theme") == "light"

    win._toggle_theme()
    assert win.is_dark is True
    assert win.catalog_index.db.get_setting("interface_theme") == "dark"

    # Verify search history logging
    win._record_search_to_history(search_mode="test_mode", query_text="test_query", results=[])
    history = win.catalog_index.db.get_search_history(10)
    assert len(history) >= 1
    assert history[0]["search_mode"] == "test_mode"
    assert history[0]["query_text"] == "test_query"


def test_first_run_eula_flow(qtbot, temp_dir, monkeypatch):
    monkeypatch.setenv("TEST_EULA_STARTUP", "1")
    win = MainWindow(data_dir=temp_dir)
    qtbot.addWidget(win)

    # First launch should land on EulaPage (page 2)
    assert win.main_stack.currentIndex() == 2
    assert win.eula_page.accept_check.isChecked() is False

    # Accept EULA
    win.eula_page.accept_check.setChecked(True)
    win.eula_page.btn_continue.click()

    # Should record to DB and transition to Matcher (page 0)
    assert win.catalog_index.db.get_setting("eula_accepted") == "1"
    assert win.main_stack.currentIndex() == 0

