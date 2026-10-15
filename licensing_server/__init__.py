"""
Licensing Server package for Design Matcher.
"""

import sys
from pathlib import Path

# Ensure design_matcher is in sys.path so app.core modules can be imported
_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "design_matcher"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from licensing_server.main import app

__all__ = ["app"]
