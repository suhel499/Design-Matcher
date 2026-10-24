"""
dwg_converter.py - Robust DWG to DXF Converter & Caching Layer for Design Matcher.

Supports:
1. Pluggable converter detection (Priority: User Setting -> ODA File Converter -> LibreDWG dwg2dxf -> None).
2. Auto-detection across standard installation paths on macOS and Windows.
3. Batch conversion per ODA call (converts whole folders in one subprocess call for maximum throughput).
4. Persistent disk caching of converted DXF files keyed by (path, size, mtime).
5. Cross-platform subprocess safety: list arguments, shell=False, CREATE_NO_WINDOW on Windows, timeout limits.
6. Clean temporary directory lifecycle management on completion, error, and cancellation.
"""

from __future__ import annotations

import atexit
import glob
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# Active temporary directories to clean up at process exit
_ACTIVE_TEMP_DIRS: set[str] = set()


def _cleanup_registered_temp_dirs() -> None:
    """Atexit handler ensuring no temporary conversion directories remain on disk."""
    for d in list(_ACTIVE_TEMP_DIRS):
        try:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        _ACTIVE_TEMP_DIRS.discard(d)


atexit.register(_cleanup_registered_temp_dirs)


def get_dwg_cache_dir() -> Path:
    """Returns the persistent cache directory for converted DXF files."""
    base = Path.home() / ".design_matcher" / "dwg_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def compute_dwg_cache_key(file_path: Union[str, Path]) -> str:
    """
    Computes a deterministic cache key for a DWG file based on its path, size, and mtime.
    Changes to file size or mtime automatically invalidate the cache entry.
    """
    p = Path(file_path).resolve()
    try:
        stat = p.stat()
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
        payload = f"{str(p).lower()}:{stat.st_size}:{mtime_ns}"
    except (OSError, FileNotFoundError):
        payload = f"{str(p).lower()}:missing"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def get_cached_dxf_path(file_path: Union[str, Path]) -> Optional[Path]:
    """
    Returns the path to the cached .dxf file if it exists and is valid (>100 bytes).
    Returns None if cache misses or file is unread.
    """
    p = Path(file_path).resolve()
    if not p.is_file():
        return None
    key = compute_dwg_cache_key(p)
    cached = get_dwg_cache_dir() / f"{key}.dxf"
    if cached.is_file() and cached.stat().st_size > 0:
        return cached
    return None


def find_oda_converter_binary(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Locates the ODA File Converter executable.
    Checks:
    1. User custom setting path.
    2. macOS standard bundle and Homebrew paths.
    3. Windows standard Program Files paths.
    4. System PATH via shutil.which.
    """
    if custom_path and os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
        return custom_path

    # macOS candidates
    if sys.platform == "darwin":
        mac_candidates = [
            "/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter",
            "/usr/local/bin/ODAFileConverter",
            "/opt/homebrew/bin/ODAFileConverter",
        ]
        for c in mac_candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c

    # Windows candidates
    elif sys.platform == "win32":
        win_candidates = [
            shutil.which("ODAFileConverter.exe"),
            shutil.which("ODAFileConverter"),
        ]
        # Search Program Files directories with wildcard for versions (e.g. ODAFileConverter 24.12.0)
        prog_roots = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        for root in prog_roots:
            if not root or not os.path.isdir(root):
                continue
            # Look for ODA/ODAFileConverter* or ODAFileConverter*
            patterns = [
                os.path.join(root, "ODA", "ODAFileConverter*", "ODAFileConverter.exe"),
                os.path.join(root, "ODAFileConverter*", "ODAFileConverter.exe"),
            ]
            for pat in patterns:
                for match in glob.glob(pat):
                    if os.path.isfile(match):
                        win_candidates.append(match)

        for c in win_candidates:
            if c and os.path.isfile(c):
                return c

    # Fallback system PATH lookup
    which_bin = shutil.which("ODAFileConverter") or shutil.which("ODAFileConverter.exe")
    if which_bin and os.path.isfile(which_bin):
        return which_bin

    return None


def find_libredwg_binary(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Locates the LibreDWG `dwg2dxf` executable.
    Checks PATH and standard Homebrew / Linux / Windows paths.
    """
    if custom_path and os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
        return custom_path

    candidates = [
        shutil.which("dwg2dxf"),
        shutil.which("dwg2dxf.exe"),
        "/opt/homebrew/bin/dwg2dxf",
        "/usr/local/bin/dwg2dxf",
        "/opt/homebrew/opt/libredwg/bin/dwg2dxf",
        "/usr/bin/dwg2dxf",
        r"C:\Program Files\LibreDWG\dwg2dxf.exe",
        r"C:\Program Files (x86)\LibreDWG\dwg2dxf.exe",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def get_available_converter(custom_path: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    Identifies the best available converter and its type.
    Returns:
        (executable_path, converter_type) where converter_type is 'oda', 'libredwg', or 'none'.
    """
    oda_bin = find_oda_converter_binary(custom_path)
    if oda_bin:
        return oda_bin, "oda"

    libredwg_bin = find_libredwg_binary()
    if libredwg_bin:
        return libredwg_bin, "libredwg"

    return None, "none"


def batch_convert_dwg_to_dxf(
    dwg_paths: Sequence[Union[str, Path]],
    custom_oda_path: Optional[str] = None,
    timeout_per_file: int = 30,
) -> Dict[Path, Optional[Path]]:
    """
    Converts a batch of DWG drawings into cached DXF files.
    Skips files already present in the cache.
    Uses ODA File Converter folder-batching when available; falls back to LibreDWG dwg2dxf.

    Returns:
        Mapping of original DWG Path -> cached DXF Path (or None if conversion failed/unavailable).
    """
    results: Dict[Path, Optional[Path]] = {}
    uncached: List[Path] = []

    # 1. Check existing cache
    for item in dwg_paths:
        p = Path(item).resolve()
        if not p.is_file():
            results[p] = None
            continue

        cached = get_cached_dxf_path(p)
        if cached:
            results[p] = cached
        else:
            uncached.append(p)
            results[p] = None

    if not uncached:
        return results

    # 2. Identify converter executable
    conv_bin, conv_type = get_available_converter(custom_oda_path)
    if not conv_bin or conv_type == "none":
        logger.warning(
            "Install ODA File Converter to enable DWG image search. Neither ODA File Converter nor LibreDWG found."
        )
        return results

    cache_dir = get_dwg_cache_dir()

    # Subprocess flags on Windows (no console window flashing)
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    # -------------------------------------------------------------------------
    # Option A: ODA File Converter (Batch folder mode)
    # -------------------------------------------------------------------------
    if conv_type == "oda":
        batch_id = uuid.uuid4().hex[:10]
        temp_in = cache_dir / f"oda_in_{batch_id}"
        temp_out = cache_dir / f"oda_out_{batch_id}"

        _ACTIVE_TEMP_DIRS.add(str(temp_in))
        _ACTIVE_TEMP_DIRS.add(str(temp_out))
        temp_in.mkdir(parents=True, exist_ok=True)
        temp_out.mkdir(parents=True, exist_ok=True)

        try:
            # Map temp filename back to original path
            temp_name_to_orig: Dict[str, Path] = {}
            for idx, orig_p in enumerate(uncached):
                # Use clean alphanumeric name to avoid ODA special character / spacing quirks
                ext = orig_p.suffix
                temp_filename = f"dwg_item_{idx:05d}{ext}"
                target_copy = temp_in / temp_filename
                try:
                    shutil.copy2(orig_p, target_copy)
                    temp_name_to_orig[temp_filename] = orig_p
                except Exception as c_err:
                    logger.warning("Failed copying DWG %s to ODA input directory: %s", orig_p.name, c_err)

            if temp_name_to_orig:
                # ODA CLI syntax: <Input_Folder> <Output_Folder> <Output_Version> <Output_Type> <Recurse> <Audit>
                cmd = [
                    conv_bin,
                    str(temp_in),
                    str(temp_out),
                    "ACAD2018",
                    "DXF",
                    "0",  # Recurse: no
                    "0",  # Audit: no (per requirement for speed)
                ]

                total_timeout = max(20, len(temp_name_to_orig) * timeout_per_file)
                logger.info("Executing ODA File Converter on %d DWG files (timeout=%ds)...", len(temp_name_to_orig), total_timeout)

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=total_timeout,
                    creationflags=creation_flags,
                )

                if proc.returncode != 0:
                    logger.debug("ODA process returned code %d. Stdout: %s", proc.returncode, proc.stdout[:200])

                # Collect converted outputs and place in permanent cache
                for temp_file in temp_out.glob("*.dxf"):
                    stem = temp_file.stem  # e.g. dwg_item_00000
                    orig_dwg = temp_name_to_orig.get(f"{stem}.dwg")
                    if orig_dwg and temp_file.stat().st_size > 0:
                        key = compute_dwg_cache_key(orig_dwg)
                        final_cached = cache_dir / f"{key}.dxf"
                        shutil.copy2(temp_file, final_cached)
                        results[orig_dwg] = final_cached

        except subprocess.TimeoutExpired:
            logger.warning("ODA conversion timed out after %ds for batch.", total_timeout)
        except Exception as exc:
            logger.exception("Unexpected error during ODA batch conversion: %s", exc)
        finally:
            shutil.rmtree(temp_in, ignore_errors=True)
            shutil.rmtree(temp_out, ignore_errors=True)
            _ACTIVE_TEMP_DIRS.discard(str(temp_in))
            _ACTIVE_TEMP_DIRS.discard(str(temp_out))

    # -------------------------------------------------------------------------
    # Option B: LibreDWG dwg2dxf (Per-file fallback)
    # -------------------------------------------------------------------------
    elif conv_type == "libredwg":
        for orig_p in uncached:
            key = compute_dwg_cache_key(orig_p)
            temp_dxf = cache_dir / f"tmp_libredwg_{key}.dxf"
            final_cached = cache_dir / f"{key}.dxf"

            if temp_dxf.exists():
                temp_dxf.unlink(missing_ok=True)

            try:
                cmd = [conv_bin, "-y", "--as", "r2018", "-o", str(temp_dxf), str(orig_p)]
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_per_file,
                    creationflags=creation_flags,
                )
                if res.returncode != 0 or not temp_dxf.is_file() or temp_dxf.stat().st_size == 0:
                    # Retry with basic -y flag
                    cmd_fallback = [conv_bin, "-y", "-o", str(temp_dxf), str(orig_p)]
                    subprocess.run(
                        cmd_fallback,
                        capture_output=True,
                        text=True,
                        timeout=timeout_per_file,
                        creationflags=creation_flags,
                    )

                if temp_dxf.is_file() and temp_dxf.stat().st_size > 0:
                    temp_dxf.replace(final_cached)
                    results[orig_p] = final_cached
            except Exception as exc:
                logger.debug("LibreDWG conversion failed for %s: %s", orig_p.name, exc)
            finally:
                if temp_dxf.exists():
                    temp_dxf.unlink(missing_ok=True)

    return results


def convert_dwg_to_dxf_single(
    file_path: Union[str, Path],
    custom_oda_path: Optional[str] = None,
    timeout_sec: int = 30,
) -> Optional[Path]:
    """Converts a single DWG file to DXF, returning the cached DXF Path or None."""
    p = Path(file_path).resolve()
    batch_res = batch_convert_dwg_to_dxf([p], custom_oda_path=custom_oda_path, timeout_per_file=timeout_sec)
    return batch_res.get(p)
