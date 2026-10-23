"""
Unit tests for CAD file browser scanning, natural sorting, unique display labels,
and hierarchical tree structure generation (DXF & DWG).
"""

from __future__ import annotations

from pathlib import Path
import pytest

from app.core.file_scanner import (
    build_tree_hierarchy,
    build_unique_display_labels,
    natural_sort_key,
    scan_cad_files,
)


@pytest.fixture
def cad_test_tree(tmp_path: Path) -> Path:
    """
    Creates the test folder structure specified in the acceptance criteria:
      root/a/1.dxf
      root/b/1.dxf
      root/c/1.dxf
      root/d/2.dxf
      root/d/3.dwg
      root/e/1.dwg
    """
    root = tmp_path / "cad_root"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "c").mkdir(parents=True)
    (root / "d").mkdir(parents=True)
    (root / "e").mkdir(parents=True)

    (root / "a" / "1.dxf").write_text("0\nEOF")
    (root / "b" / "1.dxf").write_text("0\nEOF")
    (root / "c" / "1.dxf").write_text("0\nEOF")
    (root / "d" / "2.dxf").write_text("0\nEOF")
    (root / "d" / "3.dwg").write_bytes(b"AC1032")
    (root / "e" / "1.dwg").write_bytes(b"AC1032")

    return root


def test_scan_and_labels_dxf_only(cad_test_tree: Path):
    """
    With DXF only:
    Expected matches: a/1.dxf, b/1.dxf, c/1.dxf, d/2.dxf
    Expected labels: 'a/1.dxf', 'b/1.dxf', 'c/1.dxf', '2.dxf'
    """
    matched = scan_cad_files(cad_test_tree, allowed_extensions=[".dxf"], recursive=True)
    assert len(matched) == 4

    labels = build_unique_display_labels(matched, root=cad_test_tree)
    label_values = set(labels.values())

    assert label_values == {"a/1.dxf", "b/1.dxf", "c/1.dxf", "2.dxf"}


def test_scan_and_labels_dwg_only(cad_test_tree: Path):
    """
    With DWG only:
    Expected matches: d/3.dwg, e/1.dwg
    Both filenames are unique across DWG results, so no parent prefix is needed:
    Expected labels: '3.dwg', '1.dwg'
    """
    matched = scan_cad_files(cad_test_tree, allowed_extensions=[".dwg"], recursive=True)
    assert len(matched) == 2

    labels = build_unique_display_labels(matched, root=cad_test_tree)
    label_values = set(labels.values())

    assert label_values == {"3.dwg", "1.dwg"}


def test_scan_and_labels_all_filter(cad_test_tree: Path):
    """
    With All checked:
    All 6 files matched.
    '1.dxf' appears in 3 folders -> prefixed as 'a/1.dxf', 'b/1.dxf', 'c/1.dxf'.
    '1.dwg', '3.dwg', '2.dxf' are distinct -> shown without folder prefix.
    """
    matched = scan_cad_files(cad_test_tree, allowed_extensions=[".dxf", ".dwg"], recursive=True)
    assert len(matched) == 6

    labels = build_unique_display_labels(matched, root=cad_test_tree)
    label_values = set(labels.values())

    expected = {"a/1.dxf", "b/1.dxf", "c/1.dxf", "2.dxf", "3.dwg", "1.dwg"}
    assert label_values == expected


def test_nested_folders_same_parent_name(tmp_path: Path):
    """
    Acceptance test: two nested folders with the same parent name
    (x/a/1.dxf and y/a/1.dxf).
    Expected: immediate parent 'a' collides, so second parent level is added:
    'x/a/1.dxf' and 'y/a/1.dxf'.
    """
    root = tmp_path / "nested_root"
    (root / "x" / "a").mkdir(parents=True)
    (root / "y" / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)

    f1 = root / "x" / "a" / "1.dxf"
    f2 = root / "y" / "a" / "1.dxf"
    f3 = root / "b" / "1.dxf"
    f4 = root / "standalone.dxf"

    for f in (f1, f2, f3, f4):
        f.write_text("0\nEOF")

    matched = scan_cad_files(root, allowed_extensions=[".dxf"], recursive=True)
    assert len(matched) == 4

    labels = build_unique_display_labels(matched, root=root)

    assert labels[f1] == "x/a/1.dxf"
    assert labels[f2] == "y/a/1.dxf"
    assert labels[f3] == "b/1.dxf"
    assert labels[f4] == "standalone.dxf"


def test_case_insensitive_extensions(tmp_path: Path):
    """
    Acceptance test: extension matching is case-insensitive (.DXF, .Dwg, etc.).
    """
    root = tmp_path / "case_root"
    root.mkdir()
    (root / "file1.DXF").write_text("0\nEOF")
    (root / "file2.Dwg").write_bytes(b"AC1032")
    (root / "file3.DWG").write_bytes(b"AC1032")
    (root / "ignore.pdf").write_bytes(b"%PDF")

    # DXF only
    dxf_matches = scan_cad_files(root, allowed_extensions=[".dxf"], recursive=True)
    assert len(dxf_matches) == 1
    assert dxf_matches[0].name == "file1.DXF"

    # DWG only
    dwg_matches = scan_cad_files(root, allowed_extensions=[".dwg"], recursive=True)
    assert len(dwg_matches) == 2
    dwg_names = {f.name for f in dwg_matches}
    assert dwg_names == {"file2.Dwg", "file3.DWG"}


def test_natural_sorting():
    """
    Verifies that '1.dxf', '2.dxf', '10.dxf' sort numerically instead of alphabetically.
    """
    raw_list = ["10.dxf", "1.dxf", "2.dxf", "20.dxf", "a/10.dxf", "a/2.dxf"]
    sorted_list = sorted(raw_list, key=natural_sort_key)
    expected = ["1.dxf", "2.dxf", "10.dxf", "20.dxf", "a/2.dxf", "a/10.dxf"]
    assert sorted_list == expected


def test_tree_hierarchy_builder(cad_test_tree: Path):
    """
    Verifies build_tree_hierarchy excludes folders with 0 matching files:
    - In DWG only, folders a, b, c (which only contain .dxf) must NOT appear in the tree.
    - Folders d and e contain DWG files and must appear.
    """
    dwg_matches = scan_cad_files(cad_test_tree, allowed_extensions=[".dwg"], recursive=True)
    tree = build_tree_hierarchy(dwg_matches, root=cad_test_tree)

    top_level_children = tree["children"]
    assert "d" in top_level_children
    assert "e" in top_level_children
    assert "a" not in top_level_children
    assert "b" not in top_level_children
    assert "c" not in top_level_children

    # In folder d, only 3.dwg is present (2.dxf is excluded)
    d_children = top_level_children["d"]["children"]
    assert "3.dwg" in d_children
    assert "2.dxf" not in d_children


def test_dxf_folder_explorer_ui_filters(qtbot, cad_test_tree: Path):
    """
    Validates filter checkboxes synchronization and display label formatting in DxfFolderExplorerWidget:
    - [ ] DXF  [ ] DWG  [ ] All
    - Displays unique display names in flat list:
      'a/1.dxf', 'b/1.dxf', 'c/1.dxf', '2.dxf' (DXF only)
      '1.dwg', '3.dwg' (DWG only)
    """
    from dxf_viewer import DxfFolderExplorerWidget

    widget = DxfFolderExplorerWidget(initial_folder=cad_test_tree)
    qtbot.addWidget(widget)

    # Initial state: All checked (DXF + DWG)
    assert widget.chk_all.isChecked()
    assert widget.chk_dxf.isChecked()
    assert widget.chk_dwg.isChecked()
    assert widget.file_list.count() == 6

    # Verify display labels in flat list
    item_texts = [widget.file_list.item(i).text() for i in range(widget.file_list.count())]
    assert any("a/1.dxf" in t for t in item_texts)
    assert any("b/1.dxf" in t for t in item_texts)
    assert any("c/1.dxf" in t for t in item_texts)
    assert any("[DXF] 2.dxf" in t for t in item_texts)
    assert any("[DWG] 1.dwg" in t for t in item_texts)
    assert any("[DWG] 3.dwg" in t for t in item_texts)

    # Test Filter: DXF only (uncheck DWG)
    widget.chk_dwg.setChecked(False)
    assert not widget.chk_all.isChecked()
    assert widget.chk_dxf.isChecked()
    assert widget.file_list.count() == 4
    dxf_texts = [widget.file_list.item(i).text() for i in range(widget.file_list.count())]
    assert all("[DXF]" in t for t in dxf_texts)
    assert any("a/1.dxf" in t for t in dxf_texts)
    assert any("2.dxf" in t for t in dxf_texts)

    # Test Filter: DWG only (uncheck DXF, check DWG)
    widget.chk_dxf.setChecked(False)
    widget.chk_dwg.setChecked(True)
    assert not widget.chk_all.isChecked()
    assert widget.file_list.count() == 2
    dwg_texts = [widget.file_list.item(i).text() for i in range(widget.file_list.count())]
    assert all("[DWG]" in t for t in dwg_texts)
    # Both names are unique across DWG results, so no parent prefix
    assert any("1.dwg" in t for t in dwg_texts)
    assert any("3.dwg" in t for t in dwg_texts)

    # Test Filter: Uncheck both -> empty list with hint
    widget.chk_dwg.setChecked(False)
    assert widget.file_list.count() == 1
    assert "No file types selected" in widget.file_list.item(0).text()

    # Test Filter: Check "All" -> checks both DXF and DWG
    widget.chk_all.setChecked(True)
    assert widget.chk_dxf.isChecked()
    assert widget.chk_dwg.isChecked()
    assert widget.file_list.count() == 6


def test_dxf_folder_explorer_ui_tree_view(qtbot, cad_test_tree: Path):
    """
    Validates Tree View mode in DxfFolderExplorerWidget:
    - Button toggles between Flat List and Tree View.
    - Folders with 0 matching files are pruned.
    - Clicking a file item emits file_selected(Path).
    """
    from dxf_viewer import DxfFolderExplorerWidget
    from PySide6.QtCore import Qt

    widget = DxfFolderExplorerWidget(initial_folder=cad_test_tree)
    qtbot.addWidget(widget)

    # Toggle to Tree View
    widget.btn_tree_view.click()
    assert widget.view_stack.currentIndex() == 1
    assert widget.btn_tree_view.isChecked()
    assert "List View" in widget.btn_tree_view.text()

    # Under All filter: 5 top level folders (a, b, c, d, e)
    assert widget.file_tree.topLevelItemCount() == 5
    top_folder_names = [widget.file_tree.topLevelItem(i).text(0) for i in range(widget.file_tree.topLevelItemCount())]
    assert any("a" in name for name in top_folder_names)
    assert any("d" in name for name in top_folder_names)
    assert any("e" in name for name in top_folder_names)

    # Filter to DWG only: folders a, b, c (which only contain .dxf) must be pruned
    widget.chk_dxf.setChecked(False)
    assert widget.file_tree.topLevelItemCount() == 2
    dwg_folder_names = [widget.file_tree.topLevelItem(i).text(0) for i in range(widget.file_tree.topLevelItemCount())]
    assert any("d" in name for name in dwg_folder_names)
    assert any("e" in name for name in dwg_folder_names)
    assert not any("a" in name for name in dwg_folder_names)

    # Test clicking file leaf in Tree View emits file_selected
    folder_d_item = widget.file_tree.topLevelItem(0)
    file_3_item = folder_d_item.child(0)
    assert "3.dwg" in file_3_item.text(0)

    with qtbot.waitSignal(widget.file_selected, timeout=1000) as blocker:
        widget._on_tree_item_clicked(file_3_item, 0)
    assert blocker.args[0] == (cad_test_tree / "d" / "3.dwg").resolve()

    # Test select_file in tree view
    target_e = (cad_test_tree / "e" / "1.dwg").resolve()
    widget.select_file(target_e)
    selected_tree_item = widget.file_tree.currentItem()
    assert selected_tree_item is not None
    assert selected_tree_item.data(0, Qt.ItemDataRole.UserRole).resolve() == target_e

