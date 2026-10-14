"""
Shim to run tests/test_dxf_viewer.py directly from root tests directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure design_matcher is on sys.path
dm_dir = Path(__file__).resolve().parent.parent / "design_matcher"
if str(dm_dir) not in sys.path:
    sys.path.insert(0, str(dm_dir))

from tests.test_dxf_viewer import *  # noqa: F401, F403
