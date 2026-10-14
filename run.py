"""
Launcher for Design Matcher.
Run this script to launch the PySide6 desktop GUI application or pass CLI arguments:
    python run.py
    python run.py index <catalog_path>
    python run.py search <photo_path>
    python run.py status
"""

import sys
from pathlib import Path

# Add design_matcher to path
project_dir = Path(__file__).resolve().parent / "design_matcher"
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from app.main import main

if __name__ == "__main__":
    main()
