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
if _impl_path.exists():
    _spec = importlib.util.spec_from_file_location("design_matcher_dxf_viewer", _impl_path)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["design_matcher_dxf_viewer"] = _mod
        _spec.loader.exec_module(_mod)
else:
    _mod = sys.modules.get("design_matcher_dxf_viewer") or sys.modules.get("dxf_viewer")
    if _mod is None:
        try:
            import design_matcher_dxf_viewer as _mod
        except ImportError:
            import dxf_viewer as _mod

DxfScene = getattr(_mod, "DxfScene", None)
DxfView = getattr(_mod, "DxfView", None)
DxfViewerWidget = getattr(_mod, "DxfViewerWidget", None)
DxfViewerWindow = getattr(_mod, "DxfViewerWindow", None)
DxfSceneCache = getattr(_mod, "DxfSceneCache", None)
GLOBAL_DXF_CACHE = getattr(_mod, "GLOBAL_DXF_CACHE", None)
ParsedDxfData = getattr(_mod, "ParsedDxfData", None)
LayerInfo = getattr(_mod, "LayerInfo", None)
resolve_dxf_color = getattr(_mod, "resolve_dxf_color", None)
get_insunits_info = getattr(_mod, "get_insunits_info", None)
parse_dxf_document = getattr(_mod, "parse_dxf_document", None)
DxfFolderExplorerWidget = getattr(_mod, "DxfFolderExplorerWidget", None)
DxfDetectedDesignsWidget = getattr(_mod, "DxfDetectedDesignsWidget", None)
DxfInspectorWidget = getattr(_mod, "DxfInspectorWidget", None)
DxfPreviewPage = getattr(_mod, "DxfPreviewPage", None)

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = DxfViewerWindow()
    if len(sys.argv) > 1:
        win.load_file(sys.argv[1])
    win.show()
    sys.exit(app.exec())
