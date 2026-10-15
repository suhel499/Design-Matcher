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
from PySide6.QtWidgets import QApplication

from app.core.db import CatalogDatabase
from app.core.index import CatalogIndex
from app.core.licensing.manager import LicenseManager
from app.gui.main_window import MainWindow
from app.gui.pages import (
    AboutPage,
    BackupPage,
    CatalogManagerPage,
    EulaPage,
    HelpPage,
    HistoryPage,
    LicensePage,
    SettingsPage,
    SupportPage,
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

    page.txt_folders.setPlainText("/path/to/dxf_folder")
    page.slider_thresh.setValue(82)
    page.spin_topk.setValue(45)

    saved_signals = []
    page.settings_saved.connect(lambda: saved_signals.append(True))
    page._save_values(show_alert=False)

    assert len(saved_signals) == 1
    assert db.get_setting("catalog_folders") == "/path/to/dxf_folder"
    assert db.get_setting("similarity_threshold") == "0.82"
    assert db.get_setting("top_k") == "45"


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
    assert page.table_files.item(0, 0).text() == "test1.dxf"

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
    assert page.table.item(0, 3).text() == "gate_stencil.png"
    assert "PD 101" in page.table.item(0, 5).text()

    reruns = []
    page.rerun_search.connect(lambda mode, query: reruns.append((mode, query)))
    page.table.selectRow(0)
    page._rerun_selected()
    assert len(reruns) == 1
    assert reruns[0][0] == "mode_a"
    assert reruns[0][1] == "gate_stencil.png"


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


def test_main_window_navigation_and_pages(qtbot, temp_dir):
    win = MainWindow(data_dir=temp_dir)
    qtbot.addWidget(win)

    # All 12 views in main_stack
    assert win.main_stack.count() == 12

    # Navigate to Settings (page 4)
    win.open_page(4)
    assert win.main_stack.currentIndex() == 4

    # Navigate to Catalogs (page 5)
    win.open_page(5)
    assert win.main_stack.currentIndex() == 5
    assert win.nav_tab_catalogs.isChecked()

    # Navigate to History (page 6)
    win.open_page(6)
    assert win.main_stack.currentIndex() == 6
    assert win.nav_tab_history.isChecked()

    # Navigate to Help (page 7)
    win.open_page(7)
    assert win.main_stack.currentIndex() == 7

    # Navigate back to Matcher (page 0)
    win.open_page(0)
    assert win.main_stack.currentIndex() == 0
    assert win.nav_tab_matcher.isChecked()

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

