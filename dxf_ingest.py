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
_spec = importlib.util.spec_from_file_location("design_matcher_dxf_ingest", _impl_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["design_matcher_dxf_ingest"] = _mod
_spec.loader.exec_module(_mod)

DEFAULT_IGNORE_LAYERS = _mod.DEFAULT_IGNORE_LAYERS
Design = _mod.Design
DxfTextLabel = _mod.DxfTextLabel
convert_line_art_to_stencil_mask = _mod.convert_line_art_to_stencil_mask
extract_designs = _mod.extract_designs
load_dxf = _mod.load_dxf
process_dxf_file = _mod.process_dxf_file
render_design = _mod.render_design
_run_debug_cli = _mod._run_debug_cli

if __name__ == "__main__":
    _run_debug_cli()
