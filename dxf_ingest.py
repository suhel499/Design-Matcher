"""
Root shim for DXF Ingestion module.
Allows running `python dxf_ingest.py file.dxf --out debug/` or importing from root.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

dm_dir = Path(__file__).resolve().parent / "design_matcher"
if str(dm_dir) not in sys.path:
    sys.path.insert(0, str(dm_dir))

_impl_path = dm_dir / "dxf_ingest.py"
if _impl_path.exists():
    _spec = importlib.util.spec_from_file_location("design_matcher_dxf_ingest", _impl_path)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["design_matcher_dxf_ingest"] = _mod
        _spec.loader.exec_module(_mod)
else:
    _mod = sys.modules.get("design_matcher_dxf_ingest") or sys.modules.get("dxf_ingest")
    if _mod is None:
        try:
            import design_matcher_dxf_ingest as _mod
        except ImportError:
            import dxf_ingest as _mod

DEFAULT_IGNORE_LAYERS = getattr(_mod, "DEFAULT_IGNORE_LAYERS", None)
Design = getattr(_mod, "Design", None)
DxfTextLabel = getattr(_mod, "DxfTextLabel", None)
convert_line_art_to_stencil_mask = getattr(_mod, "convert_line_art_to_stencil_mask", None)
extract_designs = getattr(_mod, "extract_designs", None)
load_dxf = getattr(_mod, "load_dxf", None)
process_dxf_file = getattr(_mod, "process_dxf_file", None)
render_design = getattr(_mod, "render_design", None)
_run_debug_cli = getattr(_mod, "_run_debug_cli", None)

if __name__ == "__main__":
    _run_debug_cli()
