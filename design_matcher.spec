# -*- mode: python ; coding: utf-8 -*-
"""
Cross-Platform PyInstaller Build Specification for Design Matcher.
Supports native bundling for:
  - Windows: Standalone Folder / Portable EXE (used by Inno Setup installer)
  - macOS: Native .app Application Bundle (used by DMG disk image installer)
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
)

block_cipher = None

# 1. Resolve project directories
spec_dir = Path(SPECPATH).resolve()
if (spec_dir / "design_matcher").exists():
    project_root = spec_dir
    dm_dir = spec_dir / "design_matcher"
elif (spec_dir / "app").exists():
    project_root = spec_dir.parent
    dm_dir = spec_dir
else:
    project_root = Path.cwd().resolve()
    dm_dir = project_root / "design_matcher"

# Platform flags
is_windows = sys.platform.startswith("win")
is_macos = sys.platform == "darwin"

# Choose appropriate icon
icon_file = None
if is_windows and (project_root / "assets" / "icon.ico").exists():
    icon_file = str(project_root / "assets" / "icon.ico")
elif is_macos and (project_root / "assets" / "icon.icns").exists():
    icon_file = str(project_root / "assets" / "icon.icns")
elif (project_root / "assets" / "icon.png").exists():
    icon_file = str(project_root / "assets" / "icon.png")

# 2. Collect Data Files
datas = []
# Bundle assets (icons, brand graphics)
if (project_root / "assets").exists():
    datas.append((str(project_root / "assets"), "assets"))

# Package data files
datas += collect_data_files("pypdfium2")
datas += collect_data_files("transformers")
datas += collect_data_files("torch")
datas += collect_data_files("ezdxf")
datas += collect_data_files("shapely")

# If local model weights are present, bundle them into 'models/dinov2-small'
local_model_dir = project_root / "models" / "dinov2-small"
if local_model_dir.exists():
    datas.append((str(local_model_dir), "models/dinov2-small"))

# 3. Collect Hidden Imports
hiddenimports = [
    # Core GUI & Qt
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",
    "PySide6.QtNetwork",
    "PySide6.QtPrintSupport",
    # Core Data & Image processing
    "sqlite3",
    "cv2",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "numpy",
    # DXF & CAD
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
    # PDF & Geometry
    "pypdfium2",
    "pypdfium2_raw",
    "shapely",
    "shapely.geometry",
    "shapely.ops",
    # Vector Search & Deep Learning
    "faiss",
    "faiss.swigfaiss",
    "torch",
    "torchvision",
    "transformers",
    # Licensing & Cryptography
    "cryptography",
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "httpx",
    "httpcore",
    "h11",
]

# Collect submodules
hiddenimports += collect_submodules("app")
hiddenimports += collect_submodules("engine")
hiddenimports += collect_submodules("transformers")
hiddenimports += collect_submodules("pypdfium2")
hiddenimports += collect_submodules("faiss")
hiddenimports += collect_submodules("ezdxf")
hiddenimports += collect_submodules("shapely")

# 4. Binaries & Dynamic Libraries
binaries = []
binaries += collect_dynamic_libs("pypdfium2")
binaries += collect_dynamic_libs("faiss")
binaries += collect_dynamic_libs("shapely")
try:
    binaries += collect_dynamic_libs("cv2")
except Exception:
    pass

# Collect torchvision extension libraries (_C_stable, image_stable, and .dylibs)
try:
    import torchvision
    tv_dir = Path(torchvision.__file__).parent
    for so_file in tv_dir.glob("*.so"):
        binaries.append((str(so_file), "torchvision"))
    for pyd_file in tv_dir.glob("*.pyd"):
        binaries.append((str(pyd_file), "torchvision"))
    dylibs_dir = tv_dir / ".dylibs"
    if dylibs_dir.exists():
        for dylib in dylibs_dir.glob("*.dylib"):
            binaries.append((str(dylib), "torchvision/.dylibs"))
except Exception:
    pass

# Entrypoint script
entrypoint = str(project_root / "run.py")
if not Path(entrypoint).exists():
    entrypoint = str(dm_dir / "app" / "main.py")

# Ensure design_matcher source files are also present in bundle data
if (dm_dir / "dxf_viewer.py").exists():
    datas.append((str(dm_dir / "dxf_viewer.py"), "design_matcher"))
if (dm_dir / "dxf_ingest.py").exists():
    datas.append((str(dm_dir / "dxf_ingest.py"), "design_matcher"))

# 5. Analysis
a = Analysis(
    [entrypoint],
    pathex=[str(dm_dir), str(project_root)],
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

# 6. Executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DesignMatcher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=is_windows,  # UPX enabled only on Windows; disabled on macOS to avoid code signing corruption
    console=False,   # Windowed GUI application (no terminal popup)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

# 7. Collect folder
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=is_windows,
    upx_exclude=[],
    name="DesignMatcher",
)

# 8. macOS Native .app Bundle
if is_macos:
    app = BUNDLE(
        coll,
        name="DesignMatcher.app",
        icon=icon_file,
        bundle_identifier="com.designmatcher.app",
        info_plist={
            "CFBundleName": "Design Matcher",
            "CFBundleDisplayName": "Design Matcher",
            "CFBundleGetInfoString": "Design Matcher - DXF Pattern & Stencil Matching System",
            "CFBundleIdentifier": "com.designmatcher.app",
            "CFBundleVersion": "1.0.0",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSApplicationCategoryType": "public.app-category.graphics-design",
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "AutoCAD DXF Drawing",
                    "CFBundleTypeRole": "Viewer",
                    "CFBundleTypeExtensions": ["dxf", "DXF"],
                    "LSHandlerRank": "Alternate",
                },
                {
                    "CFBundleTypeName": "Design Matcher License",
                    "CFBundleTypeRole": "Editor",
                    "CFBundleTypeExtensions": ["lic", "dmreq"],
                    "LSHandlerRank": "Owner",
                },
            ],
        },
    )
