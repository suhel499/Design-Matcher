"""
Root shim for stencil mask normalization.
"""

from __future__ import annotations

import sys
from pathlib import Path

dm_dir = Path(__file__).resolve().parent / "design_matcher"
if str(dm_dir) not in sys.path:
    sys.path.insert(0, str(dm_dir))

from app.core.preprocess import (  # noqa: F401
    detect_polarity,
    equalize,
    main,
    normalize_stencil,
    remove_specks,
)

if __name__ == "__main__":
    main()
