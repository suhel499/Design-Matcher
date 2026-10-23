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
    copy_metadata,
)

def safe_collect_data(pkg_name):
    try:
        return collect_data_files(pkg_name)
    except Exception:
        return []

def safe_collect_submodules(pkg_name):
    try:
        return collect_submodules(pkg_name)
    except Exception:
        return []

def safe_copy_metadata(pkg_name):
    try:
        return copy_metadata(pkg_name)
    except Exception:
        return []

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

# 2. Collect Data Files and Package Metadata
datas = []
# Bundle assets (icons, brand graphics)
if (project_root / "assets").exists():
    datas.append((str(project_root / "assets"), "assets"))

# Package metadata required at runtime
for pkg in [
    "transformers",
    "huggingface_hub",
    "safetensors",
    "tokenizers",
    "torch",
    "torchvision",
    "scipy",
    "tqdm",
    "regex",
    "requests",
    "packaging",
    "filelock",
    "numpy",
    "pydantic",
    "pydantic_core",
    "pypdfium2",
    "ezdxf",
    "PySide6",
    "PIL",
]:
    datas += safe_copy_metadata(pkg)

# Package data files
for pkg in [
    "pypdfium2",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "torch",
    "torchvision",
    "scipy",
    "ezdxf",
    "shapely",
    "pydantic",
    "PySide6",
    "cv2",
]:
    datas += safe_collect_data(pkg)

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
    "pymysql",
    "pymysql.cursors",
    "pymysql.constants",
    "cv2",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "numpy",
    # SciPy & Spatial Distance / KDTree (used in Stencil & CAD Matcher)
    "scipy",
    "scipy.spatial",
    "scipy.spatial.cKDTree",
    "scipy.spatial.distance",
    "scipy.spatial.transform",
    "scipy.special",
    "scipy.optimize",
    "scipy.ndimage",
    "scipy.linalg",
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
    "torch.utils",
    "torch.utils.data",
    "torch.nn",
    "torch.nn.functional",
    "torchvision",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "transformers",
    "transformers.models",
    "transformers.models.dinov2",
    "transformers.models.dinov2.modeling_dinov2",
    "transformers.models.auto",
    "transformers.models.auto.modeling_auto",
    "transformers.models.auto.image_processing_auto",
    "transformers.models.auto.configuration_auto",
    "huggingface_hub",
    "huggingface_hub.utils",
    "safetensors",
    "safetensors.torch",
    "tokenizers",
    # Licensing & Cryptography & Validation
    "cryptography",
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "httpx",
    "httpcore",
    "h11",
    "pydantic",
    "pydantic_core",
    "pydantic_core._pydantic_core",
]

# Collect submodules
for pkg in [
    "app",
    "engine",
    "scipy",
    "torch",
    "torchvision",
    "huggingface_hub",
    "safetensors",
    "pypdfium2",
    "faiss",
    "ezdxf",
    "shapely",
    "pymysql",
    "pydantic",
    "pydantic_core",
]:
    hiddenimports += safe_collect_submodules(pkg)


# 4. Binaries & Dynamic Libraries
binaries = []
binaries += collect_dynamic_libs("pypdfium2")
binaries += collect_dynamic_libs("faiss")
binaries += collect_dynamic_libs("shapely")
binaries += collect_dynamic_libs("scipy")
binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("torchvision")
try:
    binaries += collect_dynamic_libs("cv2")
except Exception:
    pass

# On Windows, collect PyTorch native DLLs (c10.dll, torch_cpu.dll, libiomp5md.dll, etc.)
if is_windows:
    try:
        import torch
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists():
            for dll in torch_lib.glob("*.dll"):
                binaries.append((str(dll), "torch/lib"))
                binaries.append((str(dll), "."))
    except Exception:
        pass
    try:
        import scipy
        scipy_dir = Path(scipy.__file__).parent
        for dll in scipy_dir.glob("**/*.dll"):
            binaries.append((str(dll), "scipy/.libs"))
            binaries.append((str(dll), "."))
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
    excludes=["tkinter", "IPython"],
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
    upx=False,  # UPX disabled to ensure binary stability and faster packaging
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
    upx=False,
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
