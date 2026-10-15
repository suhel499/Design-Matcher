"""
Hardware Fingerprinting Module for Design Matcher.
Generates deterministic, stable, multi-factor hardware fingerprints across
Windows, macOS, and Linux to enforce per-seat activation limits securely.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import sys
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _run_cmd(cmd: list[str]) -> Optional[str]:
    """Runs a system command safely and returns cleaned stdout if successful."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
        if res.returncode == 0:
            out = res.stdout.strip()
            return out if out else None
    except Exception as exc:
        logger.debug("Hardware probe command %s failed: %s", cmd, exc)
    return None


def get_windows_machine_guid() -> Optional[str]:
    """Reads MachineGuid from Windows Registry without requiring administrative privileges."""
    if sys.platform != "win32":
        return None
    try:
        import winreg  # type: ignore

        reg: Any = winreg
        with reg.OpenKey(
            reg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            reg.KEY_READ | getattr(reg, "KEY_WOW64_64KEY", 0),
        ) as key:
            val, _ = reg.QueryValueEx(key, "MachineGuid")
            if val:
                return str(val).strip()
    except Exception as exc:
        logger.debug("Failed reading Windows MachineGuid from registry: %s", exc)
    return None


def get_windows_bios_uuid() -> Optional[str]:
    """Retrieves motherboard / BIOS UUID on Windows via PowerShell."""
    if sys.platform != "win32":
        return None
    out = _run_cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance -ClassName Win32_ComputerSystemProduct).UUID"])
    if out and len(out) > 5 and "error" not in out.lower():
        return out.strip()
    return None


def get_macos_platform_uuid() -> Optional[str]:
    """Retrieves IOPlatformUUID on macOS."""
    if sys.platform != "darwin":
        return None
    out = _run_cmd(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
    if out:
        for line in out.splitlines():
            if "IOPlatformUUID" in line:
                parts = line.split("=")
                if len(parts) >= 2:
                    return parts[1].strip().strip('"')
    return None


def get_linux_machine_id() -> Optional[str]:
    """Retrieves Linux machine-id or DMI product UUID."""
    if not sys.platform.startswith("linux"):
        return None
    for path in ["/etc/machine-id", "/var/lib/dbus/machine-id", "/sys/class/dmi/id/product_uuid"]:
        try:
            p = os.path.expanduser(path)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content and len(content) > 6:
                        return content
        except Exception:
            continue
    return None


def get_hardware_fingerprint() -> str:
    """
    Computes a resilient, deterministic SHA-256 hardware identifier.
    Combines primary OS hardware identifiers, MAC address node, and system architecture.
    Formatted as: DM-HWID-XXXX-XXXX-XXXX-XXXX
    """
    factors: list[str] = []

    # 1. OS-specific primary stable hardware IDs
    if sys.platform == "win32":
        guid = get_windows_machine_guid()
        if guid:
            factors.append(f"win_guid:{guid}")
        bios = get_windows_bios_uuid()
        if bios:
            factors.append(f"win_bios:{bios}")
    elif sys.platform == "darwin":
        mac_uuid = get_macos_platform_uuid()
        if mac_uuid:
            factors.append(f"mac_uuid:{mac_uuid}")
    elif sys.platform.startswith("linux"):
        lin_id = get_linux_machine_id()
        if lin_id:
            factors.append(f"linux_id:{lin_id}")

    # 2. Universal network MAC address node
    node = uuid.getnode()
    if node:
        factors.append(f"mac_node:{node}")

    # 3. Fallback: Hostname and Machine arch
    factors.append(f"node:{platform.node()}")
    factors.append(f"machine:{platform.machine()}")

    raw_combined = "|".join(factors).encode("utf-8")
    digest = hashlib.sha256(raw_combined).hexdigest().upper()

    # Format into user-friendly chunks: DM-HW-XXXX-XXXX-XXXX-XXXX
    chunk1 = digest[0:4]
    chunk2 = digest[4:8]
    chunk3 = digest[8:12]
    chunk4 = digest[12:16]
    return f"DM-HW-{chunk1}-{chunk2}-{chunk3}-{chunk4}"


def get_device_info() -> Dict[str, str]:
    """
    Returns metadata about the host computer for license audit tracking.
    """
    return {
        "machine_fingerprint": get_hardware_fingerprint(),
        "machine_name": platform.node() or "Unknown-Computer",
        "os_info": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        "python_version": platform.python_version(),
    }
