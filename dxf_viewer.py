"""
Root shim for In-App DXF Vector Viewer module.
Allows importing from root or running standalone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

dm_dir = Path(__file__).resolve().parent / "design_matcher"
if str(dm_dir) not in sys.path:
    sys.path.insert(0, str(dm_dir))

_impl_path = dm_dir / "dxf_viewer.py"
_spec = importlib.util.spec_from_file_location("design_matcher_dxf_viewer", _impl_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["design_matcher_dxf_viewer"] = _mod
_spec.loader.exec_module(_mod)

DxfScene = _mod.DxfScene
DxfView = _mod.DxfView
DxfViewerWidget = _mod.DxfViewerWidget
DxfViewerWindow = _mod.DxfViewerWindow
DxfSceneCache = _mod.DxfSceneCache
GLOBAL_DXF_CACHE = _mod.GLOBAL_DXF_CACHE
ParsedDxfData = _mod.ParsedDxfData
LayerInfo = _mod.LayerInfo
resolve_dxf_color = _mod.resolve_dxf_color
get_insunits_info = _mod.get_insunits_info
parse_dxf_document = _mod.parse_dxf_document
DxfFolderExplorerWidget = _mod.DxfFolderExplorerWidget
DxfDetectedDesignsWidget = _mod.DxfDetectedDesignsWidget
DxfInspectorWidget = _mod.DxfInspectorWidget
DxfPreviewPage = _mod.DxfPreviewPage

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = DxfViewerWindow()
    if len(sys.argv) > 1:
        win.load_file(sys.argv[1])
    win.show()
    sys.exit(app.exec())
