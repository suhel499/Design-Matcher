"""
Launcher for Design Matcher.
Run this script to launch the PySide6 desktop GUI application or pass CLI arguments:
    python run.py
    python run.py index <catalog_path>
    python run.py search <photo_path>
    python run.py status
"""

import sys
import traceback
from pathlib import Path

# Add design_matcher to path
project_dir = Path(__file__).resolve().parent / "design_matcher"
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))


def _handle_fatal_error(exc: BaseException) -> None:
    """Shows user-friendly diagnostic window if GUI is available, else prints to stderr."""
    tb = traceback.format_exc()
    sys.stderr.write(f"FATAL STARTUP ERROR:\n{exc}\n\n{tb}\n")
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Design Matcher - Startup Error")
        box.setText(f"An unexpected error occurred while launching Design Matcher:\n\n{exc}")
        box.setDetailedText(tb)
        box.exec()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        from app.main import main
        main()
    except Exception as e:
        _handle_fatal_error(e)
        sys.exit(1)

