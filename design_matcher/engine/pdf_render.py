"""
PDF rendering module using pypdfium2.
Free / Apache-2.0 / BSD-3 licensed (avoids AGPL licence of PyMuPDF).
Renders pages directly to BGR numpy arrays at specified DPI.
Handles Unicode paths, Windows long paths, and corrupt/encrypted files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List, Union

import cv2
import numpy as np
import pypdfium2 as pdfium

from .config import config

logger = logging.getLogger(__name__)


class PdfRenderError(Exception):
    """Raised when a PDF cannot be rendered or read."""
    pass


class PdfEncryptedError(PdfRenderError):
    """Raised when a PDF is password protected."""
    pass


def _resolve_pdf_path(path: Union[str, Path]) -> str:
    """Normalizes and resolves the PDF path for Unicode and Windows long paths."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")
    path_str = str(resolved)
    # Support Windows extended-length path prefix if on Windows and path is long
    if os_is_windows := (hasattr(Path, "_windows_flavour") and Path._windows_flavour):
        if not path_str.startswith("\\\\?\\") and len(path_str) >= 250:
            path_str = "\\\\?\\" + path_str
    return path_str


def get_pdf_page_count(pdf_path: Union[str, Path]) -> int:
    """
    Returns the total page count for a given PDF file.
    Raises PdfRenderError or PdfEncryptedError if unreadable or password-protected.
    """
    path_str = _resolve_pdf_path(pdf_path)
    try:
        doc = pdfium.PdfDocument(path_str)
        try:
            return len(doc)
        finally:
            doc.close()
    except pdfium.PdfiumError as exc:
        msg = str(exc).lower()
        if "password" in msg or "encrypted" in msg:
            raise PdfEncryptedError(f"PDF is password protected: {pdf_path}") from exc
        raise PdfRenderError(f"Corrupt or invalid PDF file {pdf_path}: {exc}") from exc
    except Exception as exc:
        raise PdfRenderError(f"Failed to open PDF {pdf_path}: {exc}") from exc


def render_pdf_page(
    pdf_path: Union[str, Path],
    page_no: int,
    dpi: int = config.RENDER_DPI,
) -> np.ndarray:
    """
    Renders a specific page (0-indexed) of a PDF to a BGR numpy array.

    Args:
        pdf_path: Path to the PDF file.
        page_no: 0-indexed page number.
        dpi: Target rendering DPI (default: 200).

    Returns:
        np.ndarray: BGR image array (uint8).
    """
    path_str = _resolve_pdf_path(pdf_path)
    scale = float(dpi) / 72.0

    try:
        doc = pdfium.PdfDocument(path_str)
        try:
            total_pages = len(doc)
            if page_no < 0 or page_no >= total_pages:
                raise ValueError(
                    f"Page index {page_no} out of bounds for {pdf_path} (total: {total_pages})"
                )
            page = doc[page_no]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            rgb_arr = np.array(pil_image)
            # Convert RGB to BGR for OpenCV standard
            if rgb_arr.ndim == 3 and rgb_arr.shape[2] == 3:
                return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            elif rgb_arr.ndim == 3 and rgb_arr.shape[2] == 4:
                return cv2.cvtColor(rgb_arr, cv2.COLOR_RGBA2BGR)
            elif rgb_arr.ndim == 2:
                return cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2BGR)
            return rgb_arr
        finally:
            doc.close()
    except pdfium.PdfiumError as exc:
        msg = str(exc).lower()
        if "password" in msg or "encrypted" in msg:
            raise PdfEncryptedError(f"PDF is encrypted: {pdf_path}") from exc
        raise PdfRenderError(f"Error rendering page {page_no} of {pdf_path}: {exc}") from exc
    except Exception as exc:
        if isinstance(exc, (ValueError, FileNotFoundError, PdfRenderError)):
            raise
        raise PdfRenderError(f"Unexpected error rendering page {page_no} of {pdf_path}: {exc}") from exc


def render_all_pages(
    pdf_path: Union[str, Path],
    dpi: int = config.RENDER_DPI,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yields (page_no, bgr_image) for all pages in the PDF."""
    path_str = _resolve_pdf_path(pdf_path)
    scale = float(dpi) / 72.0
    doc = pdfium.PdfDocument(path_str)
    try:
        total = len(doc)
        for page_no in range(total):
            page = doc[page_no]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            rgb_arr = np.array(pil_image)
            bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            yield page_no, bgr
    finally:
        doc.close()
