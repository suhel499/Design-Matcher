"""
PDF rendering engine using pypdfium2.
Free for commercial use (Apache-2.0 / BSD-3-clause). Strictly no PyMuPDF / AGPL.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pypdfium2 as pdfium

logger = logging.getLogger(__name__)


class PdfRenderError(Exception):
    """Raised when a PDF cannot be opened or rendered."""
    pass


def calculate_file_hash(file_path: str | Path) -> str:
    """
    Computes SHA-256 hash of a file efficiently using 64KB chunks.

    Args:
        file_path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_pdf_page_count(pdf_path: str | Path) -> int:
    """
    Returns the number of pages in the given PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Integer count of pages.
    """
    p = Path(pdf_path)
    if not p.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = pdfium.PdfDocument(str(p))
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception as exc:
        raise PdfRenderError(f"Failed to inspect PDF '{pdf_path}': {exc}") from exc


def render_pdf_page(
    pdf_path: str | Path,
    page_index: int,
    dpi: int = 200,
) -> np.ndarray:
    """
    Renders a single PDF page to an OpenCV BGR uint8 numpy array.

    Args:
        pdf_path: Path to the PDF file.
        page_index: Zero-based page index.
        dpi: Rendering DPI (default 200 DPI). Standard PDF unit is 72 DPI.

    Returns:
        numpy.ndarray: BGR image array (H, W, 3) uint8.
    """
    p = Path(pdf_path)
    if not p.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = pdfium.PdfDocument(str(p))
        try:
            if page_index < 0 or page_index >= len(doc):
                raise IndexError(
                    f"Page index {page_index} out of range for PDF '{pdf_path}' with {len(doc)} pages."
                )

            page = doc.get_page(page_index)
            try:
                scale = dpi / 72.0
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                # Convert PIL image to BGR numpy array for OpenCV
                img_array = np.array(pil_image)

                if img_array.ndim == 2:
                    return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4:
                    return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif img_array.shape[2] == 3:
                    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    raise ValueError(f"Unexpected image shape from PDF page: {img_array.shape}")
            finally:
                page.close()
        finally:
            doc.close()
    except Exception as exc:
        if isinstance(exc, (IndexError, FileNotFoundError)):
            raise
        raise PdfRenderError(f"Failed to render page {page_index} of '{pdf_path}': {exc}") from exc
