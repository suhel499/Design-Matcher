"""
Utility functions for file hashing, preview rendering, visual diffs, and system integration.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np

from .pdf_render import render_pdf_page

logger = logging.getLogger(__name__)


def compute_file_fingerprint(path: Union[str, Path]) -> Tuple[int, float, str]:
    """
    Computes (size, mtime, sha1) for incremental file indexing.
    Reads first 1 MB and last 1 MB along with file size for fast, collision-resistant hashing.
    """
    p = Path(path).resolve()
    stat = p.stat()
    size = stat.st_size
    mtime = stat.st_mtime

    hasher = hashlib.sha1()
    hasher.update(str(size).encode("utf-8"))
    chunk_size = 1024 * 1024  # 1 MB

    with open(p, "rb") as f:
        head = f.read(chunk_size)
        hasher.update(head)
        if size > chunk_size:
            f.seek(max(0, size - chunk_size))
            tail = f.read(chunk_size)
            hasher.update(tail)

    return size, mtime, hasher.hexdigest()


def render_page_preview(
    pdf_path: Union[str, Path],
    page_no: int,
    bbox: Tuple[int, int, int, int],
    dpi: int = 150,
) -> np.ndarray:
    """
    Renders the PDF page and highlights the matched design bounding box.
    Returns BGR uint8 numpy array.
    """
    page_bgr = render_pdf_page(pdf_path, page_no, dpi=dpi)
    h_page, w_page = page_bgr.shape[:2]

    # Scale bounding box if DPI differs from indexing DPI (200)
    scale = float(dpi) / 200.0
    x, y, w, h = bbox
    sx = int(round(x * scale))
    sy = int(round(y * scale))
    sw = int(round(w * scale))
    sh = int(round(h * scale))

    # Clamp to image bounds
    x1 = max(0, min(w_page - 1, sx))
    y1 = max(0, min(h_page - 1, sy))
    x2 = max(0, min(w_page, sx + sw))
    y2 = max(0, min(h_page, sy + sh))

    annotated = page_bgr.copy()
    if x2 > x1 and y2 > y1:
        # Translucent amber/gold overlay inside candidate box
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), -1)  # BGR amber
        cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)

        # Crisp border rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 3, cv2.LINE_AA)

    return annotated


def overlay_diff(
    query_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    """
    Creates an RGB image comparing query and candidate masks (both 256x256 uint8, 255=metal, 0=voids):
    - Agreement on voids (cutouts): Grey (140, 140, 140)
    - Agreement on metal: Light background (240, 240, 240)
    - Query only cutout (missing in candidate): Red (230, 45, 45)
    - Candidate only cutout (missing in query): Blue (45, 95, 235)
    """
    # Ensure same dimensions
    q = (query_mask > 127).astype(np.uint8)
    c = (candidate_mask > 127).astype(np.uint8)
    if q.shape != c.shape:
        c = cv2.resize(c, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_NEAREST)

    h, w = q.shape[:2]
    rgb = np.full((h, w, 3), 240, dtype=np.uint8)  # Default light metal agreement

    # Cutout flags (0 is cutout)
    q_cut = q == 0
    c_cut = c == 0

    both_cut = q_cut & c_cut
    q_only = q_cut & (~c_cut)
    c_only = (~q_cut) & c_cut

    rgb[both_cut] = [140, 140, 140]  # Agreement on voids -> Grey
    rgb[q_only] = [230, 45, 45]     # Query only -> Red
    rgb[c_only] = [45, 95, 235]     # Candidate only -> Blue

    return rgb


def open_pdf_at_page(pdf_path: Union[str, Path], page_no: int = 0) -> bool:
    """
    Opens the PDF in the operating system's default viewer.
    Handles Windows os.startfile, macOS open, and Linux xdg-open.
    """
    resolved = Path(pdf_path).resolve()
    if not resolved.exists():
        logger.error("Cannot open non-existent PDF: %s", pdf_path)
        return False

    path_str = str(resolved)
    try:
        sys_name = platform.system()
        if sys_name == "Windows":
            # On Windows, os.startfile opens default handler
            os.startfile(path_str)  # type: ignore
            return True
        elif sys_name == "Darwin":
            subprocess.Popen(["open", path_str])
            return True
        else:
            subprocess.Popen(["xdg-open", path_str])
            return True
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", pdf_path, exc)
        return False


def record_feedback(
    query_hash: str,
    design_id: int,
    correct: bool,
    store: Optional[Any] = None,
) -> None:
    """Convenience helper to record user feedback."""
    if store is not None:
        store.record_feedback(query_hash, design_id, correct)
