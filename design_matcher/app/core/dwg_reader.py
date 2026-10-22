"""
DWG File Reader and Preview Generator for Design Matcher.
Enables reading, previewing, and indexing of AutoCAD binary DWG files alongside DXF files.

Supports:
1. Native binary extraction of embedded BMP/PNG thumbnail images directly from DWG headers
   (AutoCAD R13 through R2018+).
2. Headless CAD rendering via LibreOffice Draw (exports high-resolution PNG previews).
3. Vector conversion via ODAFileConverter or dwg2dxf when available.
4. Persistent disk caching of converted previews for instant on-select loading.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard DWG version codes
DWG_VERSIONS: Dict[str, str] = {
    "AC1012": "AutoCAD R13",
    "AC1014": "AutoCAD R14",
    "AC1015": "AutoCAD 2000",
    "AC1018": "AutoCAD 2004",
    "AC1021": "AutoCAD 2007",
    "AC1024": "AutoCAD 2010",
    "AC1027": "AutoCAD 2013",
    "AC1032": "AutoCAD 2018",
}


def get_dwg_cache_dir() -> Path:
    """Returns local cache directory for converted DWG previews and DXF files."""
    base = Path.home() / ".design_matcher" / "dwg_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def is_dwg_file(file_path: Union[str, Path]) -> bool:
    """Checks if file is a DWG based on extension."""
    return Path(file_path).suffix.lower() == ".dwg"


def get_dwg_version(file_path: Union[str, Path]) -> Tuple[str, str]:
    """
    Reads the 6-byte version header from a DWG file.
    Returns:
        (version_code, friendly_name)
    """
    p = Path(file_path)
    if not p.is_file():
        return "", "Unknown"
    try:
        with open(p, "rb") as f:
            header = f.read(6).decode("ascii", errors="ignore")
        name = DWG_VERSIONS.get(header, f"DWG ({header})")
        return header, name
    except Exception:
        return "", "Unknown DWG"


def extract_dwg_embedded_thumbnail(file_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Extracts embedded preview thumbnail (BMP/PNG/DIB) from binary DWG file.
    Zero external dependencies, fast (<5ms).

    Returns:
        np.ndarray BGR image (uint8) or None if no thumbnail embedded.
    """
    p = Path(file_path)
    if not p.is_file():
        return None

    try:
        file_bytes = p.read_bytes()
    except Exception as exc:
        logger.debug("Could not read DWG file %s: %s", p.name, exc)
        return None

    if len(file_bytes) < 32:
        return None

    version = file_bytes[:6].decode("ascii", errors="ignore")

    # Method 1: AC1012 / AC1014 / AC1015 (R13 - 2000) header offset
    if version in ("AC1012", "AC1014", "AC1015") and len(file_bytes) >= 17:
        try:
            offset = struct.unpack("<I", file_bytes[13:17])[0]
            if 17 < offset < len(file_bytes) - 8:
                rec_count = file_bytes[offset]
                if 1 <= rec_count <= 4:
                    rec_type = file_bytes[offset + 1]
                    rec_len = struct.unpack("<I", file_bytes[offset + 2:offset + 6])[0]
                    if 0 < rec_len <= len(file_bytes) - (offset + 6):
                        img_data = file_bytes[offset + 6:offset + 6 + rec_len]
                        # Check for BMP
                        if img_data.startswith(b"BM"):
                            arr = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                            if arr is not None and arr.size > 0:
                                return arr
                        # Check if raw DIB
                        elif img_data.startswith(b"\x28\x00\x00\x00"):
                            bmp_header = b"BM" + struct.pack("<IHHI", 14 + len(img_data), 0, 0, 54)
                            arr = cv2.imdecode(np.frombuffer(bmp_header + img_data, np.uint8), cv2.IMREAD_COLOR)
                            if arr is not None and arr.size > 0:
                                return arr
        except Exception as e:
            logger.debug("Offset method failed for %s: %s", p.name, e)

    # Method 2: Scan for embedded BMP (magic 'BM' with BITMAPINFOHEADER length 40)
    search_limit = min(len(file_bytes), 2 * 1024 * 1024)  # Search first 2MB
    bm_pos = 0
    while True:
        bm_pos = file_bytes.find(b"BM", bm_pos, search_limit)
        if bm_pos == -1 or bm_pos + 54 > len(file_bytes):
            break

        # Check if DIB header size is 40
        if file_bytes[bm_pos + 14:bm_pos + 18] == b"\x28\x00\x00\x00":
            try:
                bmp_size = struct.unpack("<I", file_bytes[bm_pos + 2:bm_pos + 6])[0]
                width = struct.unpack("<i", file_bytes[bm_pos + 18:bm_pos + 22])[0]
                height = abs(struct.unpack("<i", file_bytes[bm_pos + 22:bm_pos + 26])[0])
                planes = struct.unpack("<H", file_bytes[bm_pos + 26:bm_pos + 28])[0]

                if 16 <= width <= 4096 and 16 <= height <= 4096 and planes == 1:
                    slice_len = bmp_size if 54 < bmp_size <= (len(file_bytes) - bm_pos) else min(1024 * 1024, len(file_bytes) - bm_pos)
                    candidate = file_bytes[bm_pos:bm_pos + slice_len]
                    arr = cv2.imdecode(np.frombuffer(candidate, np.uint8), cv2.IMREAD_COLOR)
                    if arr is not None and arr.shape[0] >= 16 and arr.shape[1] >= 16:
                        return arr
            except Exception:
                pass
        bm_pos += 2

    # Method 3: Scan for embedded PNG stream
    png_pos = file_bytes.find(b"\x89PNG\r\n\x1a\n", 0, search_limit)
    if png_pos != -1:
        try:
            candidate = file_bytes[png_pos:png_pos + min(1024 * 1024, len(file_bytes) - png_pos)]
            arr = cv2.imdecode(np.frombuffer(candidate, np.uint8), cv2.IMREAD_COLOR)
            if arr is not None and arr.size > 0:
                return arr
        except Exception:
            pass

    return None


def find_dwg2dxf_binary() -> Optional[str]:
    """Locates dwg2dxf binary from libredwg across standard macOS / Linux paths."""
    candidates = [
        shutil.which("dwg2dxf"),
        "/opt/homebrew/bin/dwg2dxf",
        "/usr/local/bin/dwg2dxf",
        "/opt/homebrew/opt/libredwg/bin/dwg2dxf",
        "/usr/bin/dwg2dxf",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def find_oda_converter_binary() -> Optional[str]:
    """Locates ODAFileConverter binary across standard macOS / Linux paths."""
    candidates = [
        shutil.which("ODAFileConverter"),
        "/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter",
        "/usr/local/bin/ODAFileConverter",
        "/opt/homebrew/bin/ODAFileConverter",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def convert_dwg_to_dxf(file_path: Union[str, Path]) -> Optional[Path]:
    """
    Converts a DWG CAD drawing into a standardized ASCII DXF file.
    Enables native vector CAD rendering, layer visibility, and multi-panel design extraction.
    Caches converted DXF persistently in ~/.design_matcher/dwg_cache/.

    Returns:
        Path to converted .dxf file if successful, or None if conversion failed/tools missing.
    """
    p = Path(file_path).resolve()
    if not p.is_file():
        return None

    stat = p.stat()
    cache_key = hashlib.md5(f"dxf:{p}:{stat.st_size}:{stat.st_mtime}".encode("utf-8")).hexdigest()
    cache_dir = get_dwg_cache_dir()
    cached_dxf = cache_dir / f"{cache_key}.dxf"

    # Return cached DXF if already converted
    if cached_dxf.is_file() and cached_dxf.stat().st_size > 100:
        return cached_dxf

    # Method 1: dwg2dxf from libredwg (fast, native CAD conversion preserving splines and blocks)
    dwg2dxf_bin = find_dwg2dxf_binary()
    if dwg2dxf_bin:
        temp_dxf = cache_dir / f"tmp_{cache_key}.dxf"
        if temp_dxf.exists():
            temp_dxf.unlink(missing_ok=True)
        try:
            # Pass --as r2018 to keep splines, 2D/3D polylines, and blocks intact
            cmd = [dwg2dxf_bin, "-y", "--as", "r2018", "-o", str(temp_dxf), str(p)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if res.returncode != 0 or not temp_dxf.is_file() or temp_dxf.stat().st_size <= 100:
                # Retry with basic -y flag
                cmd_alt = [dwg2dxf_bin, "-y", "-o", str(temp_dxf), str(p)]
                res = subprocess.run(cmd_alt, capture_output=True, text=True, timeout=25)
                if res.returncode != 0 or not temp_dxf.is_file() or temp_dxf.stat().st_size <= 100:
                    cmd_alt2 = [dwg2dxf_bin, "-o", str(temp_dxf), str(p)]
                    subprocess.run(cmd_alt2, capture_output=True, text=True, timeout=25)

            if temp_dxf.is_file() and temp_dxf.stat().st_size > 100:
                temp_dxf.replace(cached_dxf)
                logger.info("Successfully converted %s to DXF via dwg2dxf: %s", p.name, cached_dxf)
                return cached_dxf
        except Exception as exc:
            logger.debug("dwg2dxf conversion failed for %s: %s", p.name, exc)
        finally:
            if temp_dxf.exists():
                temp_dxf.unlink(missing_ok=True)

    # Method 2: ODAFileConverter
    oda_bin = find_oda_converter_binary()
    if oda_bin:
        temp_in = cache_dir / f"oda_in_{cache_key}"
        temp_out = cache_dir / f"oda_out_{cache_key}"
        temp_in.mkdir(parents=True, exist_ok=True)
        temp_out.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(p, temp_in / p.name)
            cmd = [
                oda_bin,
                str(temp_in),
                str(temp_out),
                "ACAD2018",
                "DXF",
                "0",  # recurse: no
                "1",  # audit: yes
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            out_dxfs = list(temp_out.glob("*.dxf"))
            if out_dxfs and out_dxfs[0].stat().st_size > 100:
                shutil.copy2(out_dxfs[0], cached_dxf)
                logger.info("Successfully converted %s to DXF via ODA: %s", p.name, cached_dxf)
                return cached_dxf
        except Exception as exc:
            logger.debug("ODA conversion failed for %s: %s", p.name, exc)
        finally:
            shutil.rmtree(temp_in, ignore_errors=True)
            shutil.rmtree(temp_out, ignore_errors=True)

    # Method 3: LibreOffice headless export to DXF
    lo_bin = shutil.which("libreoffice") or shutil.which("soffice")
    if not lo_bin and Path("/usr/local/bin/libreoffice").is_file():
        lo_bin = "/usr/local/bin/libreoffice"
    if not lo_bin and Path("/opt/homebrew/bin/soffice").is_file():
        lo_bin = "/opt/homebrew/bin/soffice"
    if not lo_bin and Path("/Applications/LibreOffice.app/Contents/MacOS/soffice").is_file():
        lo_bin = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

    if lo_bin:
        temp_out = cache_dir / f"lo_dxf_{cache_key}"
        temp_out.mkdir(parents=True, exist_ok=True)
        try:
            cmd = [
                lo_bin,
                "--headless",
                "--convert-to",
                "dxf",
                str(p),
                "--outdir",
                str(temp_out),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if res.returncode == 0:
                out_dxfs = list(temp_out.glob("*.dxf"))
                if out_dxfs and out_dxfs[0].stat().st_size > 100:
                    shutil.copy2(out_dxfs[0], cached_dxf)
                    logger.info("Successfully converted %s to DXF via LibreOffice: %s", p.name, cached_dxf)
                    return cached_dxf
        except Exception as exc:
            logger.debug("LibreOffice DXF conversion failed for %s: %s", p.name, exc)
        finally:
            shutil.rmtree(temp_out, ignore_errors=True)

    return None


def convert_dwg_to_preview_libreoffice(file_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Renders DWG drawing into an image using LibreOffice Draw headlessly.
    Cached locally by file content hash.
    """
    p = Path(file_path).resolve()
    if not p.is_file():
        return None

    # Locate libreoffice executable
    lo_bin = shutil.which("libreoffice") or shutil.which("soffice")
    if not lo_bin and Path("/usr/local/bin/libreoffice").is_file():
        lo_bin = "/usr/local/bin/libreoffice"
    if not lo_bin and Path("/opt/homebrew/bin/soffice").is_file():
        lo_bin = "/opt/homebrew/bin/soffice"
    if not lo_bin and Path("/Applications/LibreOffice.app/Contents/MacOS/soffice").is_file():
        lo_bin = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

    if not lo_bin:
        return None

    # Compute cache key from path and mtime
    stat = p.stat()
    cache_key = hashlib.md5(f"img:{p}:{stat.st_size}:{stat.st_mtime}".encode("utf-8")).hexdigest()
    cache_dir = get_dwg_cache_dir()
    cached_png = cache_dir / f"{cache_key}.png"

    if cached_png.is_file():
        arr = cv2.imread(str(cached_png))
        if arr is not None and arr.size > 0:
            return arr

    temp_out = cache_dir / f"tmp_{cache_key}"
    temp_out.mkdir(parents=True, exist_ok=True)
    try:
        # Export as PDF first for maximum vector crispness
        cmd_pdf = [
            lo_bin,
            "--headless",
            "--convert-to",
            "pdf",
            str(p),
            "--outdir",
            str(temp_out),
        ]
        res_pdf = subprocess.run(cmd_pdf, capture_output=True, text=True, timeout=20)
        out_pdfs = list(temp_out.glob("*.pdf"))
        if res_pdf.returncode == 0 and out_pdfs:
            pdf_path = out_pdfs[0]
            # Primary PDF renderer: pypdfium2 (vector anti-aliased high-res)
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(str(pdf_path))
                if len(pdf) > 0:
                    page = pdf[0]
                    pil_img = page.render(scale=3.0).to_pil()
                    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    if arr is not None and arr.size > 0:
                        cv2.imwrite(str(cached_png), arr)
                        return arr
            except Exception as pdf_exc:
                logger.debug("pypdfium2 render failed for %s: %s", pdf_path.name, pdf_exc)

            # Fallback 1: macOS sips
            sips_bin = shutil.which("sips") or "/usr/bin/sips"
            target_png = temp_out / f"{pdf_path.stem}.png"
            if os.path.isfile(sips_bin):
                cmd_sips = [
                    sips_bin,
                    "-s",
                    "format",
                    "png",
                    "--resampleHeight",
                    "2048",
                    str(pdf_path),
                    "--out",
                    str(target_png),
                ]
                subprocess.run(cmd_sips, capture_output=True, timeout=10)

            if target_png.is_file() and target_png.stat().st_size > 0:
                shutil.copy2(target_png, cached_png)
                arr = cv2.imread(str(cached_png))
                if arr is not None and arr.size > 0:
                    return arr

        # Fallback to direct PNG export
        cmd_png = [
            lo_bin,
            "--headless",
            "--convert-to",
            "png",
            str(p),
            "--outdir",
            str(temp_out),
        ]
        res_png = subprocess.run(cmd_png, capture_output=True, text=True, timeout=15)
        if res_png.returncode == 0:
            out_pngs = list(temp_out.glob("*.png"))
            if out_pngs:
                shutil.copy2(out_pngs[0], cached_png)
                arr = cv2.imread(str(cached_png))
                if arr is not None and arr.size > 0:
                    return arr
    except Exception as exc:
        logger.debug("LibreOffice DWG conversion failed for %s: %s", p.name, exc)
    finally:
        shutil.rmtree(temp_out, ignore_errors=True)

    return None


def read_dwg_preview(file_path: Union[str, Path], prefer_high_res: bool = True) -> Optional[np.ndarray]:
    """
    High-level entrypoint to retrieve a visual preview for any DWG file.
    Priority order:
    1. LibreOffice Draw / PDF conversion (super crisp high resolution 2048px).
    2. Embedded binary thumbnail (only if high-res fails, avoids pixelation).
    """
    p = Path(file_path)
    if prefer_high_res:
        # Try high-resolution conversion first
        lo_img = convert_dwg_to_preview_libreoffice(p)
        if lo_img is not None and lo_img.shape[0] >= 400 and lo_img.shape[1] >= 400:
            return lo_img

        # Fallback to embedded thumbnail if high-res failed
        thumb = extract_dwg_embedded_thumbnail(p)
        if thumb is not None:
            return thumb

        if lo_img is not None:
            return lo_img
    else:
        thumb = extract_dwg_embedded_thumbnail(p)
        if thumb is not None:
            return thumb
        lo_img = convert_dwg_to_preview_libreoffice(p)
        if lo_img is not None:
            return lo_img

    return None


def generate_dwg_cad_card(file_path: Union[str, Path], version_name: str) -> np.ndarray:
    """
    Generates a dark CAD themed card with drawing metadata when visual preview is absent.
    """
    p = Path(file_path)
    h, w = 360, 480
    card = np.full((h, w, 3), (25, 20, 15), dtype=np.uint8)

    # Grid lines
    for y in range(0, h, 20):
        cv2.line(card, (0, y), (w, y), (35, 30, 25), 1)
    for x in range(0, w, 20):
        cv2.line(card, (x, 0), (x, h), (35, 30, 25), 1)

    # CAD File Icon
    cv2.rectangle(card, (190, 80), (290, 200), (218, 218, 222), 2)
    cv2.line(card, (200, 100), (280, 100), (59, 130, 246), 2)
    cv2.line(card, (200, 130), (280, 130), (59, 130, 246), 2)
    cv2.line(card, (200, 160), (260, 160), (59, 130, 246), 2)

    # Text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(card, "DWG CAD DRAWING", (140, 235), font, 0.65, (255, 255, 255), 2)
    cv2.putText(card, p.name[:32], (70, 275), font, 0.55, (56, 189, 248), 1)
    cv2.putText(card, f"Format: {version_name}", (130, 305), font, 0.45, (148, 163, 184), 1)

    return card
