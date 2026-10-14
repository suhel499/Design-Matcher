"""
Design Matcher - Main Entry Point.
Launches the PySide6 desktop GUI by default, or runs CLI commands if arguments are provided.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so modules can always be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    """Entry point for Design Matcher."""
    if len(sys.argv) > 1:
        # CLI sub-commands (index, search, status, etc.)
        from app.cli import main as cli_main
        cli_main()
    else:
        # Desktop GUI
        from app.gui.main_window import run_gui
        run_gui()


if __name__ == "__main__":
    main()
