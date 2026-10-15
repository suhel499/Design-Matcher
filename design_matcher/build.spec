# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build specification for Design Matcher.
Supports bundling on Windows (and macOS/Linux for development).
Configured for offline CPU execution with local model caching.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

# Base directories
project_root = Path.cwd().resolve()
app_dir = project_root / "app"

# Collect datas and hidden imports
datas = []
hiddenimports = [
    "pypdfium2",
    "pypdfium2_raw",
    "faiss",
    "faiss.swigfaiss",
    "torch",
    "torchvision",
    "transformers",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "sqlite3",
    "cv2",
    "PIL",
    "numpy",
    "ezdxf",
    "ezdxf.path",
    "ezdxf.recover",
    "ezdxf.addons",
    "ezdxf.addons.drawing",
    "ezdxf.math",
    "ezdxf.disassemble",
    "ezdxf.tools",
    "dxf_ingest",
    "dxf_viewer",
    "shapely",
    "shapely.geometry",
    "shapely.ops",
    "cryptography",
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "httpx",
    "app.core.licensing",
    "app.core.licensing.crypto",
    "app.core.licensing.hardware",
    "app.core.licensing.manager",
    "app.core.licensing.models",
    "app.core.licensing.public_key",
    "app.gui.license_dialog",
]

# Collect package data files for transformers, torch, pypdfium2, ezdxf, shapely
datas += collect_data_files("pypdfium2")
datas += collect_data_files("transformers")
datas += collect_data_files("torch")
datas += collect_data_files("ezdxf")
datas += collect_data_files("shapely")

# If local model weights are present, bundle them into 'models/dinov2-small'
local_model_dir = project_root / "models" / "dinov2-small"
if local_model_dir.exists():
    datas.append((str(local_model_dir), "models/dinov2-small"))

# Collect submodules
hiddenimports += collect_submodules("transformers")
hiddenimports += collect_submodules("pypdfium2")
hiddenimports += collect_submodules("faiss")
hiddenimports += collect_submodules("ezdxf")
hiddenimports += collect_submodules("shapely")

# Binaries
binaries = []
binaries += collect_dynamic_libs("pypdfium2")
binaries += collect_dynamic_libs("faiss")
binaries += collect_dynamic_libs("shapely")

a = Analysis(
    [str(app_dir / "main.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "IPython", "matplotlib"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# One-folder build (recommended first for fast debugging and startup)
exe_onedir = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DesignMatcher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False for windowed release
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe_onedir,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DesignMatcher",
)
