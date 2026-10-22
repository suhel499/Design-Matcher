"""
Stage A: Searchable Metadata Extraction Worker.
Fast, picklable extraction of file-level attributes, PDF text/metadata,
and lightweight DXF layer/geometry properties without heavy rasterization.
Strictly decoupled from UI modules and database write operations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_fast_hash(file_path: str | Path) -> str:
    """Computes SHA-256 hash in 64KB blocks."""
    p = Path(file_path)
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_pdf_metadata(file_path: str | Path) -> Tuple[int, str]:
    """
    Extracts page count and searchable text/document tags from a PDF using pypdfium2.
    Does not perform heavy rasterization or OCR.

    Returns:
        (page_count, searchable_text)
    """
    import pypdfium2 as pdfium

    p = Path(file_path)
    doc = pdfium.PdfDocument(str(p))
    try:
        page_count = len(doc)
        searchable_parts: List[str] = [p.stem]

        # Extract embedded PDF metadata dictionary
        try:
            meta = doc.get_metadata_dict()
            for key in ("Title", "Author", "Subject", "Keywords"):
                val = meta.get(key)
                if val and str(val).strip():
                    searchable_parts.append(str(val).strip())
        except Exception:
            pass

        # Extract lightweight embedded text from up to first 3 pages
        max_pages_to_sample = min(page_count, 3)
        for p_idx in range(max_pages_to_sample):
            try:
                page = doc.get_page(p_idx)
                try:
                    textpage = page.get_textpage()
                    try:
                        text = textpage.get_text_range()
                        if text and text.strip():
                            # Truncate to reasonable searchable snippet
                            sample = " ".join(text.split()[:100])
                            if sample:
                                searchable_parts.append(sample)
                    finally:
                        textpage.close()
                finally:
                    page.close()
            except Exception:
                continue

        searchable_text = " ".join(searchable_parts)
        return page_count, searchable_text
    finally:
        doc.close()


def extract_dxf_metadata(file_path: str | Path) -> Tuple[int, str]:
    """
    Extracts layers, entity counts, units, and extents from a DXF drawing using ezdxf.
    Does not render raster stencils.

    Returns:
        (design_count, searchable_text)
    """
    import ezdxf

    p = Path(file_path)
    searchable_parts: List[str] = [p.stem]

    try:
        doc = ezdxf.readfile(str(p))
    except Exception:
        # Fallback to recovery reader if standard read fails
        try:
            from ezdxf import recover
            doc, auditor = recover.readfile(str(p))
        except Exception as exc:
            raise ValueError(f"Could not parse DXF drawing '{p.name}': {exc}") from exc

    try:
        # Collect layer names
        layers = [layer.dxf.name for layer in doc.layers if layer.dxf.name]
        if layers:
            searchable_parts.extend(layers[:30])

        # Inspect modelspace entities
        msp = doc.modelspace()
        entity_count = len(msp)

        # Look for text or mtext annotations in the drawing
        text_samples = []
        for entity in msp:
            dxftype = entity.dxftype()
            if dxftype in ("TEXT", "MTEXT"):
                txt = getattr(entity.dxf, "text", "")
                if txt and txt.strip():
                    text_samples.append(txt.strip())
                    if len(text_samples) >= 15:
                        break

        if text_samples:
            searchable_parts.extend(text_samples)

        searchable_parts.append(f"layers:{len(layers)} entities:{entity_count}")
        searchable_text = " ".join(searchable_parts)
        return 1, searchable_text
    finally:
        pass


def extract_single_file_metadata(
    file_path: str,
    modified_ns: int,
    file_size: int,
) -> Dict[str, Any]:
    """
    Top-level picklable metadata extractor for a single file.
    Catches errors cleanly and marks status accordingly.
    """
    p = Path(file_path)
    ext = p.suffix.lower()
    fname = p.name

    record: Dict[str, Any] = {
        "path": str(p.resolve()),
        "filename": fname,
        "extension": ext,
        "file_size": file_size,
        "modified_ns": modified_ns,
        "content_hash": None,
        "page_count": 0,
        "searchable_metadata": fname,
        "metadata_status": "pending",
        "visual_status": "pending",
        "parser_version": "1.0",
        "error_message": "",
    }

    try:
        # Compute content hash
        content_hash = compute_fast_hash(p)
        record["content_hash"] = content_hash

        if ext == ".dwg":
            from app.core.dwg_reader import get_dwg_version
            ver_code, ver_name = get_dwg_version(p)
            record["page_count"] = 1
            record["searchable_metadata"] = f"{p.stem} dwg autocad cad {ver_name} drawing"
            record["metadata_status"] = "completed"

        elif ext == ".pdf":
            page_count, searchable_text = extract_pdf_metadata(p)
            record["page_count"] = page_count
            record["searchable_metadata"] = searchable_text
            record["metadata_status"] = "completed"

        elif ext == ".dxf":
            design_count, searchable_text = extract_dxf_metadata(p)
            record["page_count"] = design_count
            record["searchable_metadata"] = searchable_text
            record["metadata_status"] = "completed"

        else:
            record["metadata_status"] = "failed"
            record["visual_status"] = "failed"
            record["error_message"] = f"Unsupported file extension '{ext}'"

    except Exception as exc:
        logger.warning("Stage A extraction failed for '%s': %s", fname, exc)
        record["metadata_status"] = "failed"
        record["error_message"] = str(exc)

    return record


def extract_metadata_batch(file_batch: List[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
    """
    Picklable worker function for ProcessPoolExecutor.
    Extracts Stage A metadata for a batch of file tuples (path, modified_ns, file_size).
    """
    results = []
    for fpath, mod_ns, fsize in file_batch:
        results.append(extract_single_file_metadata(fpath, mod_ns, fsize))
    return results
