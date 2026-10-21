"""
Launcher for Design Matcher.
Run this script to launch the PySide6 desktop GUI application or pass CLI arguments:
    python run.py
    python run.py admin
    python run.py index <catalog_path>
    python run.py search <photo_path>
    python run.py status
"""

import sys
import traceback
from pathlib import Path

# Add repo root and design_matcher to sys.path
if not getattr(sys, "frozen", False):
    repo_root = Path(__file__).resolve().parent
    project_dir = repo_root / "design_matcher"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
else:
    # Running inside PyInstaller frozen executable
    if hasattr(sys, "_MEIPASS"):
        meipass = sys._MEIPASS
        if meipass not in sys.path:
            sys.path.insert(0, meipass)
        dm_dir = Path(meipass) / "design_matcher"
        if dm_dir.exists() and str(dm_dir) not in sys.path:
            sys.path.insert(0, str(dm_dir))


def _handle_fatal_error(exc: BaseException) -> None:
    """Shows user-friendly diagnostic window if GUI is available, else prints to stderr."""
    tb = traceback.format_exc()
    sys.stderr.write(f"FATAL STARTUP ERROR:\n{exc}\n\n{tb}\nsys.path:\n{sys.path}\nsys._MEIPASS: {getattr(sys, '_MEIPASS', None)}\n")
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
        try:
            from app.main import main
        except ImportError:
            from design_matcher.app.main import main
        main()
    except Exception as e:
        _handle_fatal_error(e)
        sys.exit(1)

