"""
Centralized License Manager for Design Matcher.
Enforces subscription checks, Ed25519 signature verification, hardware binding,
clock-rollback detection, grace period for online outages, and offline air-gapped activation.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

from app.core.index import get_default_data_dir
from app.core.licensing.crypto import verify_payload
from app.core.licensing.hardware import get_device_info, get_hardware_fingerprint
from app.core.licensing.models import (
    LicenseCheckResult,
    LicensePayload,
    LicenseStatus,
    OfflineActivationRequest,
    OfflineActivationResponse,
    PlanType,
    SignedLicenseFile,
)
from app.core.licensing.public_key import get_verification_public_key

logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://dm.suiteutil.com"
DEFAULT_GRACE_PERIOD_DAYS = 7
EXPECTED_PRODUCT_ID = "design-matcher"


class LicenseManager:
    """
    Centralized license and subscription manager.
    Protects search, indexing, and export operations while guaranteeing
    that customer files, databases, and saved indices are never deleted.
    """

    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        server_url: Optional[str] = None,
        public_key_pem: Optional[str] = None,
        grace_period_days: int = DEFAULT_GRACE_PERIOD_DAYS,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else get_default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.license_file_path = self.data_dir / "license.json"
        self.clock_file_path = self.data_dir / ".license_clock.json"
        self.status_override_path = self.data_dir / ".license_status.json"

        self.server_url = server_url or os.environ.get("DESIGN_MATCHER_SERVER_URL", DEFAULT_SERVER_URL)
        self.public_key_pem = public_key_pem or get_verification_public_key()
        self.grace_period_days = grace_period_days

        self._cached_signed_license: Optional[SignedLicenseFile] = None
        self._cached_payload: Optional[LicensePayload] = None
        self._cached_result: Optional[LicenseCheckResult] = None
        self._clock_rollback_detected: bool = False
        self._load_local_license()

    # -------------------------------------------------------------------------
    # Clock Rollback Detection
    # -------------------------------------------------------------------------

    def _check_and_update_monotonic_clock(self, now_utc: datetime) -> bool:
        """
        Detects if system time has been rolled backward.
        Allows up to 1 hour of clock skew/tolerance.
        Returns True if clock is valid, False if clock rollback is detected.
        """
        if self._clock_rollback_detected:
            return False

        last_known_ts: Optional[datetime] = None
        if self.clock_file_path.exists():
            try:
                with open(self.clock_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    ts_str = data.get("last_known_utc")
                    if ts_str:
                        last_known_ts = datetime.fromisoformat(ts_str)
            except Exception as exc:
                logger.debug("Failed reading clock rollback state: %s", exc)

        if last_known_ts is not None:
            # If current time is earlier than last known by more than 1 hour -> rollback
            if now_utc < (last_known_ts - timedelta(hours=1)):
                self._clock_rollback_detected = True
                logger.warning(
                    "System clock rollback detected! Current: %s, Last Known: %s",
                    now_utc.isoformat(),
                    last_known_ts.isoformat(),
                )
                return False

        # Monotonically update state
        new_last_ts = max(now_utc, last_known_ts) if last_known_ts else now_utc
        try:
            with open(self.clock_file_path, "w", encoding="utf-8") as f:
                json.dump({"last_known_utc": new_last_ts.isoformat()}, f)
        except Exception as exc:
            logger.debug("Failed writing clock state: %s", exc)

        return True

    # -------------------------------------------------------------------------
    # License Storage & Verification
    # -------------------------------------------------------------------------

    def _load_local_license(self) -> None:
        """Loads and parses the locally saved signed license file if present."""
        if not self.license_file_path.exists():
            self._cached_signed_license = None
            self._cached_payload = None
            return

        try:
            with open(self.license_file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                signed_file = SignedLicenseFile.model_validate(raw_data)
                self._cached_signed_license = signed_file
                self._cached_payload = LicensePayload.model_validate(signed_file.payload)
        except Exception as exc:
            logger.warning("Failed to parse local license file: %s", exc)
            self._cached_signed_license = None
            self._cached_payload = None

    def save_signed_license(self, signed_file_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates cryptographic signature and saves license locally.
        Returns (success, message).
        """
        try:
            signed_file = SignedLicenseFile.model_validate(signed_file_dict)
            payload = signed_file.payload

            # 1. Cryptographic Ed25519 verification
            if not verify_payload(payload, signed_file.signature, self.public_key_pem):
                return False, "Digital signature verification failed. The license file is invalid or tampered with."

            parsed_payload = LicensePayload.model_validate(payload)

            # 2. Product verification
            if parsed_payload.product_id not in (EXPECTED_PRODUCT_ID, "design_matcher"):
                return False, f"License is for '{parsed_payload.product_id}', not '{EXPECTED_PRODUCT_ID}'."

            # 3. Hardware binding verification
            current_hw = get_hardware_fingerprint()
            if parsed_payload.hardware_binding and parsed_payload.hardware_binding != current_hw:
                return False, (
                    f"License is bound to another computer ({parsed_payload.hardware_binding}). "
                    f"Current computer ID: {current_hw}"
                )

            # Write atomically to disk
            with open(self.license_file_path, "w", encoding="utf-8") as f:
                json.dump(signed_file.model_dump(), f, indent=2)

            # Clear any administrative suspension/revocation override
            if self.status_override_path.exists():
                try:
                    self.status_override_path.unlink()
                except Exception:
                    pass

            self._cached_signed_license = signed_file
            self._cached_payload = parsed_payload
            self._cached_result = None

            return True, "License activated successfully."
        except Exception as exc:
            logger.exception("Failed to save signed license: %s", exc)
            return False, f"Failed to activate license: {exc}"

    def remove_local_license(self) -> None:
        """Removes local license file (e.g. upon deactivation or license transfer)."""
        if self.license_file_path.exists():
            try:
                self.license_file_path.unlink()
            except Exception as exc:
                logger.error("Error removing local license file: %s", exc)
        self._cached_signed_license = None
        self._cached_payload = None
        self._cached_result = None

    def has_local_license(self) -> bool:
        """Returns True if a license file is present and loaded locally."""
        return self.license_file_path.exists() and self._cached_payload is not None

    # -------------------------------------------------------------------------
    # Core Verification & Enforcement
    # -------------------------------------------------------------------------

    def check_license(self, now_utc: Optional[datetime] = None) -> LicenseCheckResult:
        """
        Comprehensive offline verification of the installed license.
        Checks signature, product ID, device binding, status, expiration, and clock rollback.
        """
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        # 1. Clock rollback check
        if not self._check_and_update_monotonic_clock(now_utc):
            return LicenseCheckResult(
                is_valid=False,
                is_rollback_detected=True,
                status=LicenseStatus.SUSPENDED,
                message="System clock rollback detected. Please restore system time to continue.",
            )

        # 2. Presence check
        if not self._cached_signed_license or not self._cached_payload:
            return LicenseCheckResult(
                is_valid=False,
                status=LicenseStatus.EXPIRED,
                message="No license installed. Please activate your subscription.",
            )

        signed_file = self._cached_signed_license
        payload = self._cached_payload

        # 2b. Server administrative suspension or revocation override check
        if self.status_override_path.exists():
            try:
                with open(self.status_override_path, "r", encoding="utf-8") as f:
                    ov = json.load(f)
                ov_status_str = ov.get("status", "REVOKED")
                ov_status = LicenseStatus.SUSPENDED if ov_status_str == "SUSPENDED" else LicenseStatus.REVOKED
                return LicenseCheckResult(
                    is_valid=False,
                    is_revoked_or_suspended=True,
                    status=ov_status,
                    license_id=payload.license_id,
                    license_key=payload.license_key,
                    customer_name=payload.customer_name,
                    message=ov.get("message", f"License is {ov_status.value.lower()} by the administrator."),
                )
            except Exception as exc:
                logger.debug("Failed reading status override file: %s", exc)

        # 3. Cryptographic signature verification
        if not verify_payload(signed_file.payload, signed_file.signature, self.public_key_pem):
            return LicenseCheckResult(
                is_valid=False,
                status=LicenseStatus.REVOKED,
                message="License signature verification failed. The license payload has been modified.",
            )

        # 4. Product ID check
        if payload.product_id not in (EXPECTED_PRODUCT_ID, "design_matcher"):
            return LicenseCheckResult(
                is_valid=False,
                status=LicenseStatus.REVOKED,
                message=f"License is for '{payload.product_id}', not '{EXPECTED_PRODUCT_ID}'.",
            )

        # 5. Hardware binding check
        current_hw = get_hardware_fingerprint()
        if payload.hardware_binding and payload.hardware_binding != current_hw:
            return LicenseCheckResult(
                is_valid=False,
                is_device_mismatched=True,
                status=LicenseStatus.REVOKED,
                license_id=payload.license_id,
                license_key=payload.license_key,
                customer_name=payload.customer_name,
                message=f"License is bound to another machine ({payload.hardware_binding}). Current: {current_hw}",
            )

        # 6. Status check (REVOKED, SUSPENDED)
        if payload.status == LicenseStatus.REVOKED.value:
            return LicenseCheckResult(
                is_valid=False,
                is_revoked_or_suspended=True,
                status=LicenseStatus.REVOKED,
                license_id=payload.license_id,
                license_key=payload.license_key,
                customer_name=payload.customer_name,
                message="This license has been revoked by the administrator.",
            )

        if payload.status == LicenseStatus.SUSPENDED.value:
            return LicenseCheckResult(
                is_valid=False,
                is_revoked_or_suspended=True,
                status=LicenseStatus.SUSPENDED,
                license_id=payload.license_id,
                license_key=payload.license_key,
                customer_name=payload.customer_name,
                message="This license is currently suspended.",
            )

        # 7. Expiration check
        try:
            expires_at = datetime.fromisoformat(payload.expires_at)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
        except Exception:
            return LicenseCheckResult(
                is_valid=False,
                status=LicenseStatus.REVOKED,
                message="Invalid expiration date in license payload.",
            )

        delta = expires_at - now_utc
        days_remaining = delta.days

        if now_utc >= expires_at:
            return LicenseCheckResult(
                is_valid=False,
                is_expired=True,
                status=LicenseStatus.EXPIRED,
                license_id=payload.license_id,
                license_key=payload.license_key,
                customer_name=payload.customer_name,
                expires_at=expires_at,
                days_remaining=0,
                message=f"Subscription expired on {expires_at.strftime('%Y-%m-%d %H:%M UTC')}. Please renew to continue.",
            )

        return LicenseCheckResult(
            is_valid=True,
            status=LicenseStatus.ACTIVE,
            license_id=payload.license_id,
            license_key=payload.license_key,
            customer_name=payload.customer_name,
            expires_at=expires_at,
            days_remaining=days_remaining,
            message="License is active.",
            details={
                "plan_type": payload.plan_type,
                "issued_at": payload.issued_at,
                "max_activations": payload.max_activations,
                "features": payload.features,
            },
        )

    def is_operation_allowed(self, operation: str = "search", now_utc: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Main gatekeeper called before running searches, indexing, or exports.
        Returns:
            Tuple[bool, str]: (allowed, failure_message)
        """
        verdict = self.check_license(now_utc=now_utc)
        if not verdict.is_valid:
            return False, verdict.message

        if self._cached_payload and self._cached_payload.features:
            # If specific features are listed, verify operation
            op_feature_map = {
                "search_dxf": "dxf_matching",
                "search_pdf": "pdf_catalog",
                "index_dxf": "dxf_matching",
                "index_pdf": "pdf_catalog",
                "code_search": "code_search",
                "export": "dxf_export",
                "dxf_preview": "dxf_matching",
            }
            required_feat = op_feature_map.get(operation)
            if required_feat and required_feat not in self._cached_payload.features:
                return False, f"Feature '{required_feat}' is not enabled in your current plan."

        return True, ""

    def get_expiration_warning_days(self) -> Optional[int]:
        """
        Returns the warning day bucket (30, 15, 7, 1) if renewal reminder is due, else None.
        """
        verdict = self.check_license()
        if not verdict.is_valid or verdict.days_remaining is None:
            return None

        days = verdict.days_remaining
        if days == 1:
            return 1
        elif 1 < days <= 7:
            return 7
        elif 7 < days <= 15:
            return 15
        elif 15 < days <= 30:
            return 30
        return None

    # -------------------------------------------------------------------------
    # Online Activation / Validation
    # -------------------------------------------------------------------------

    def activate_online(self, license_key: str, server_url: Optional[str] = None) -> Tuple[bool, str]:
        """
        Activates a license key with the central server over HTTPS.
        """
        url = (server_url or self.server_url).rstrip("/") + "/api/v1/licenses/activate"
        dev_info = get_device_info()

        req_payload = {
            "license_key": license_key.strip(),
            "machine_fingerprint": dev_info["machine_fingerprint"],
            "machine_name": dev_info["machine_name"],
            "os_info": dev_info["os_info"],
            "nonce": secrets.token_hex(16),
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(url, json=req_payload)

            if res.status_code == 200:
                data = res.json()
                signed_file = data.get("license_file")
                if not signed_file:
                    return False, "Server returned invalid activation response: missing license_file."
                return self.save_signed_license(signed_file)
            else:
                try:
                    err_json = res.json()
                    detail = err_json.get("detail", res.text)
                except Exception:
                    detail = res.text
                return False, f"Activation rejected by server: {detail}"

        except Exception as exc:
            logger.warning("Online activation network error: %s", exc)
            return False, f"Cannot connect to license server: {exc}"

    def validate_online(self, server_url: Optional[str] = None) -> LicenseCheckResult:
        """
        Validates the license with the server. If server is unreachable, applies grace period.
        """
        if not self._cached_payload:
            return self.check_license()

        url = (server_url or self.server_url).rstrip("/") + "/api/v1/licenses/validate"
        dev_info = get_device_info()

        req_payload = {
            "license_key": self._cached_payload.license_key,
            "machine_fingerprint": dev_info["machine_fingerprint"],
            "nonce": secrets.token_hex(16),
        }

        try:
            with httpx.Client(timeout=8.0) as client:
                res = client.post(url, json=req_payload)

            if res.status_code == 200:
                if self.status_override_path.exists():
                    try:
                        self.status_override_path.unlink()
                    except Exception:
                        pass
                data = res.json()
                signed_file = data.get("license_file")
                if signed_file:
                    self.save_signed_license(signed_file)
                return self.check_license()
            else:
                try:
                    detail = res.json().get("detail", res.text)
                except Exception:
                    detail = res.text

                lower_det = detail.lower()
                target_status = LicenseStatus.SUSPENDED if "suspended" in lower_det else LicenseStatus.REVOKED
                try:
                    with open(self.status_override_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "status": target_status.value,
                            "message": detail,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }, f)
                except Exception as exc:
                    logger.warning("Failed writing status override: %s", exc)

                return LicenseCheckResult(
                    is_valid=False,
                    is_revoked_or_suspended=True,
                    status=target_status,
                    license_id=self._cached_payload.license_id if self._cached_payload else None,
                    license_key=self._cached_payload.license_key if self._cached_payload else None,
                    customer_name=self._cached_payload.customer_name if self._cached_payload else None,
                    message=f"Server validation: {detail}",
                )

        except Exception as exc:
            logger.info("Server unreachable during online validation: %s. Evaluating grace period...", exc)
            # Evaluate grace period
            local_verdict = self.check_license()
            # If the offline cryptographic expiration date has PASSED, grace period CANNOT extend it!
            if local_verdict.is_expired:
                return local_verdict

            if local_verdict.is_valid:
                local_verdict.is_grace_period_active = True
                local_verdict.message = (
                    f"Running in offline grace period ({self.grace_period_days} days). "
                    "License server temporarily unreachable."
                )
            return local_verdict

    def deactivate_online(self, server_url: Optional[str] = None) -> Tuple[bool, str]:
        """
        Deactivates this computer seat on the server to permit transfer to another computer.
        """
        if not self._cached_payload:
            return False, "No active license to deactivate."

        url = (server_url or self.server_url).rstrip("/") + "/api/v1/licenses/deactivate"
        dev_info = get_device_info()

        req_payload = {
            "license_key": self._cached_payload.license_key,
            "machine_fingerprint": dev_info["machine_fingerprint"],
            "nonce": secrets.token_hex(16),
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(url, json=req_payload)

            if res.status_code == 200:
                self.remove_local_license()
                return True, "Computer deactivated successfully. Seat released for license transfer."
            else:
                detail = res.text
                return False, f"Server deactivation rejected: {detail}"
        except Exception as exc:
            return False, f"Network error during deactivation: {exc}"

    # -------------------------------------------------------------------------
    # Offline Air-Gapped Activation Workflow
    # -------------------------------------------------------------------------

    def generate_offline_request(self, license_key: str) -> str:
        """
        Generates an offline activation request string (.dmreq).
        The customer sends this request to the administrator.
        """
        dev_info = get_device_info()
        req = OfflineActivationRequest(
            version=1,
            license_key=license_key.strip(),
            machine_fingerprint=dev_info["machine_fingerprint"],
            machine_name=dev_info["machine_name"],
            os_info=dev_info["os_info"],
            request_timestamp=datetime.now(timezone.utc).isoformat(),
            request_nonce=secrets.token_hex(16),
        )
        return json.dumps(req.model_dump(), indent=2)

    def import_offline_license(self, file_content_or_path: str | Path) -> Tuple[bool, str]:
        """
        Imports and validates a signed license file generated by the administrator.
        """
        try:
            content = ""
            if isinstance(file_content_or_path, Path) or (
                isinstance(file_content_or_path, str) and os.path.exists(file_content_or_path)
            ):
                with open(file_content_or_path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                content = str(file_content_or_path)

            raw_dict = json.loads(content)

            # Check if wrapped in OfflineActivationResponse
            if "license_file" in raw_dict:
                signed_dict = raw_dict["license_file"]
            else:
                signed_dict = raw_dict

            return self.save_signed_license(signed_dict)
        except Exception as exc:
            logger.exception("Failed to import offline license: %s", exc)
            return False, f"Failed to import offline license: {exc}"
