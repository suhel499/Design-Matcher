# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build specification for Design Matcher.
Delegates to the root design_matcher.spec configuration.
"""

from pathlib import Path

# Spec path inside design_matcher
spec_file = Path(SPECPATH).resolve().parent / "design_matcher.spec"
if spec_file.exists():
    with open(spec_file, "r", encoding="utf-8") as f:
        exec(f.read())
else:
    raise FileNotFoundError(f"Missing master spec at {spec_file}")
