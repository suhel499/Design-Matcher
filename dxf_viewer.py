"""
Root shim for In-App DXF Vector Viewer module.
Allows importing from root or running standalone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# 1. Resolve candidate directories containing design_matcher/dxf_viewer.py
_this_file = Path(__file__).resolve()
_candidates = [
    _this_file.parent / "design_matcher",
    _this_file.parent,
]

if getattr(sys, "frozen", False):
    _meipass = getattr(sys, "_MEIPASS", None)
    if _meipass:
        _candidates.insert(0, Path(_meipass) / "design_matcher")
        _candidates.insert(1, Path(_meipass))
    _exe_dir = Path(sys.executable).resolve().parent
    _candidates.append(_exe_dir / "design_matcher")
    _candidates.append(_exe_dir / "_internal" / "design_matcher")
    _candidates.append(_exe_dir)

_impl_path = None
for c in _candidates:
    p = c / "dxf_viewer.py"
    if p.exists() and p.resolve() != _this_file:
        _impl_path = p
        if str(c) not in sys.path:
            sys.path.insert(0, str(c))
        break

_mod = None
if _impl_path and _impl_path.exists():
    try:
        _spec = importlib.util.spec_from_file_location("design_matcher_dxf_viewer", _impl_path)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["design_matcher_dxf_viewer"] = _mod
            _spec.loader.exec_module(_mod)
    except Exception:
        _mod = None

if _mod is None:
    _mod = sys.modules.get("design_matcher_dxf_viewer")

if _mod is None:
    try:
        import design_matcher_dxf_viewer as _mod  # type: ignore
    except Exception:
        _mod = None

if _mod and _mod is not sys.modules.get(__name__):
    for _k, _v in _mod.__dict__.items():
        if not _k.startswith("__"):
            globals()[_k] = _v

DxfScene = globals().get("DxfScene")
DxfView = globals().get("DxfView")
DxfViewerWidget = globals().get("DxfViewerWidget")
DxfViewerWindow = globals().get("DxfViewerWindow")
DxfSceneCache = globals().get("DxfSceneCache")
GLOBAL_DXF_CACHE = globals().get("GLOBAL_DXF_CACHE")
ParsedDxfData = globals().get("ParsedDxfData")
LayerInfo = globals().get("LayerInfo")
resolve_dxf_color = globals().get("resolve_dxf_color")
get_insunits_info = globals().get("get_insunits_info")
parse_dxf_document = globals().get("parse_dxf_document")
DxfFolderExplorerWidget = globals().get("DxfFolderExplorerWidget")
DxfDetectedDesignsWidget = globals().get("DxfDetectedDesignsWidget")
DxfInspectorWidget = globals().get("DxfInspectorWidget")
DxfPreviewPage = globals().get("DxfPreviewPage")

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = DxfViewerWindow() if DxfViewerWindow else None
    if win:
        if len(sys.argv) > 1:
            win.load_file(sys.argv[1])
        win.show()
        sys.exit(app.exec())
    else:
        sys.stderr.write("Error: DxfViewerWindow could not be loaded.\n")
        sys.exit(1)
