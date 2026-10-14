"""
Root-level evaluation CLI launcher.
"""

from __future__ import annotations

import sys
from pathlib import Path

dm_dir = Path(__file__).resolve().parent / "design_matcher"
if str(dm_dir) not in sys.path:
    sys.path.insert(0, str(dm_dir))

from design_matcher.evaluate import main

if __name__ == "__main__":
    main()
