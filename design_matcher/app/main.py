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

CLI_COMMANDS = {
    "admin",
    "index",
    "index-dxf",
    "match-dxf",
    "search",
    "status",
    "download-model",
}


def main() -> None:
    """Entry point for Design Matcher."""
    # Check if first argument is an explicit CLI subcommand
    if len(sys.argv) > 1 and sys.argv[1].lower() in CLI_COMMANDS:
        from app.cli import main as cli_main
        cli_main()
    else:
        # Desktop GUI
        from app.gui.main_window import run_gui
        initial_file = None
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            arg_path = Path(sys.argv[1])
            if arg_path.exists():
                initial_file = str(arg_path)
        run_gui(initial_file=initial_file)


if __name__ == "__main__":
    main()
