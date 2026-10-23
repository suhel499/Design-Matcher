"""
Root shim for DXF Ingestion module.
Allows running `python dxf_ingest.py file.dxf --out debug/` or importing from root.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# 1. Resolve candidate directories containing design_matcher/dxf_ingest.py
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
    p = c / "dxf_ingest.py"
    if p.exists() and p.resolve() != _this_file:
        _impl_path = p
        if str(c) not in sys.path:
            sys.path.insert(0, str(c))
        break

_mod = None
if _impl_path and _impl_path.exists():
    try:
        _spec = importlib.util.spec_from_file_location("design_matcher_dxf_ingest", _impl_path)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["design_matcher_dxf_ingest"] = _mod
            _spec.loader.exec_module(_mod)
    except Exception:
        _mod = None

if _mod is None:
    _mod = sys.modules.get("design_matcher_dxf_ingest")

if _mod is None:
    try:
        import design_matcher_dxf_ingest as _mod  # type: ignore
    except Exception:
        _mod = None

if _mod and _mod is not sys.modules.get(__name__):
    for _k, _v in _mod.__dict__.items():
        if not _k.startswith("__"):
            globals()[_k] = _v

DEFAULT_IGNORE_LAYERS = globals().get("DEFAULT_IGNORE_LAYERS")
Design = globals().get("Design")
DxfTextLabel = globals().get("DxfTextLabel")
convert_line_art_to_stencil_mask = globals().get("convert_line_art_to_stencil_mask")
extract_designs = globals().get("extract_designs")
load_dxf = globals().get("load_dxf")
process_dxf_file = globals().get("process_dxf_file")
render_design = globals().get("render_design")
_run_debug_cli = globals().get("_run_debug_cli")

if __name__ == "__main__":
    if callable(_run_debug_cli):
        _run_debug_cli()
    else:
        sys.stderr.write("Error: dxf_ingest debug CLI could not be loaded.\n")
        sys.exit(1)
