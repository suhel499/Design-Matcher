"""
CLI entrypoint and module export for stencil mask normalization.
Can be executed as:
    python preprocess.py input.jpg output.png
Or imported as:
    from preprocess import normalize_stencil
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so app.core can be imported
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from app.core.preprocess import (  # noqa: F401
    detect_polarity,
    equalize,
    main,
    normalize_dxf_screenshot,
    normalize_stencil,
    remove_specks,
)

if __name__ == "__main__":
    main()
