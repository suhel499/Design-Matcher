"""
Design Matcher Application Package.
"""

import os

# Prevent OpenMP runtime conflicts between PyTorch and FAISS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "1.0.2"
