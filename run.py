"""
Launcher for Design Matcher.
Run this script to launch the PySide6 desktop GUI application or pass CLI arguments:
    python run.py
    python run.py admin
    python run.py index <catalog_path>
    python run.py search <photo_path>
    python run.py status
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# 1. Resolve paths for standard Python execution and PyInstaller frozen bundles
if getattr(sys, "frozen", False):
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
    exe_dir = Path(sys.executable).resolve().parent
    for p in (bundle_root, bundle_root / "design_matcher", exe_dir, exe_dir / "design_matcher"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
else:
    repo_root = Path(__file__).resolve().parent
    project_dir = repo_root / "design_matcher"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

# 2. On Windows, register DLL directories for PyTorch, SciPy, OpenCV, Qt
if sys.platform.startswith("win"):
    base_dirs = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            base_dirs.append(Path(meipass))
        base_dirs.append(Path(sys.executable).resolve().parent)
    else:
        base_dirs.append(Path(__file__).resolve().parent)

    for base in base_dirs:
        for sub in ("", "torch/lib", "scipy/.libs", "PySide6", "cv2"):
            d = (base / sub).resolve()
            if d.is_dir():
                try:
                    os.add_dll_directory(str(d))
                except Exception:
                    pass
                cur_path = os.environ.get("PATH", "")
                if str(d) not in cur_path:
                    os.environ["PATH"] = str(d) + os.path.pathsep + cur_path


def _handle_fatal_error(exc: BaseException) -> None:
    """Shows user-friendly diagnostic window if GUI is available, else prints to stderr."""
    tb = traceback.format_exc()
    if sys.stderr is not None:
        try:
            sys.stderr.write(f"FATAL STARTUP ERROR:\n{exc}\n\n{tb}\n")
        except Exception:
            pass
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Design Matcher - Error")
        box.setText(f"An unexpected error occurred in Design Matcher:\n\n{exc}")
        box.setDetailedText(tb)
        box.exec()
    except Exception:
        pass


def _uncaught_exception_handler(exc_type, exc_value, exc_traceback):
    """Global exception handler for unexpected background or Qt event loop errors."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    _handle_fatal_error(exc_value)


sys.excepthook = _uncaught_exception_handler


if __name__ == "__main__":
    try:
        from app.main import main
        main()
    except Exception as e:
        _handle_fatal_error(e)
        sys.exit(1)
