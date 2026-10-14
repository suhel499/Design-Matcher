"""
Pytest configuration and environment setup.
Sets KMP_DUPLICATE_LIB_OK to avoid OpenMP conflicts between torch and faiss-cpu.
"""

import os
import sys
from pathlib import Path

# Prevent OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
