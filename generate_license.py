#!/usr/bin/env python3
"""
CLI Utility to Generate and Sign License Files for Design Matcher.

Usage Examples:
    # 1-Year License for current PC:
    python generate_license.py --customer "Precision Laserworks" --plan 1_year --bind-current-pc

    # 3-Year License for an offline machine:
    python generate_license.py --customer "Apex CNC" --plan 3_years --hw-id "DM-HW-A1B2-C3D4-E5F6-7890" --out apex.lic

    # Custom 90-day trial license:
    python generate_license.py --customer "Trial User" --plan custom --days 90 --out trial.lic
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure root directory and design_matcher are in path
_REPO_ROOT = Path(__file__).resolve().parent
_APP_DIR = _REPO_ROOT / "design_matcher"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from licensing_server.config import settings
from licensing_server.crypto import build_and_sign_license_payload, generate_license_key
from licensing_server.database import Base, SessionLocal, engine
from licensing_server.models import Customer, License, Product, Subscription
from design_matcher.app.core.licensing.hardware import get_hardware_fingerprint


def main():
    parser = argparse.ArgumentParser(description="Generate and cryptographically sign a Design Matcher license file.")
    parser.add_argument("--customer", default="Valued Customer", help="Customer name or company")
    parser.add_argument("--plan", choices=["1_year", "3_years", "5_years", "custom"], default="1_year", help="Subscription plan")
    parser.add_argument("--days", type=int, default=365, help="Duration in days if plan is 'custom'")
    parser.add_argument("--hw-id", default=None, help="Hardware fingerprint to bind (format: DM-HW-XXXX-XXXX-XXXX-XXXX)")
    parser.add_argument("--bind-current-pc", action="store_true", help="Automatically bind to the current machine's hardware ID")
    parser.add_argument("--seats", type=int, default=1, help="Maximum concurrent activations allowed (default: 1)")
    parser.add_argument("--out", default=None, help="Output .lic file path (defaults to <license_key>.lic)")
    parser.add_argument("--save-db", action="store_true", default=True, help="Save record to licensing server database")
    args = parser.parse_args()

    # Determine hardware binding
    hw_binding = args.hw_id
    if args.bind_current_pc:
        hw_binding = get_hardware_fingerprint()
        print(f"[*] Bound to current machine hardware ID: {hw_binding}")

    # Determine duration
    now = datetime.now(timezone.utc)
    if args.plan == "1_year":
        exp = now + timedelta(days=365)
    elif args.plan == "3_years":
        exp = now + timedelta(days=365 * 3)
    elif args.plan == "5_years":
        exp = now + timedelta(days=365 * 5)
    else:
        exp = now + timedelta(days=args.days)

    lic_key = generate_license_key()

    # Optional: Save to local database
    db = SessionLocal()
    try:
        # Check or create customer
        cust = db.query(Customer).filter(Customer.name == args.customer).first()
        if not cust:
            cust = Customer(
                name=args.customer,
                email=f"{args.customer.lower().replace(' ', '')}@example.com",
                company=args.customer,
            )
            db.add(cust)
            db.flush()

        prod = db.query(Product).filter(Product.product_code == settings.DEFAULT_PRODUCT_CODE).first()
        if not prod:
            prod = Product(
                product_code=settings.DEFAULT_PRODUCT_CODE,
                name=settings.DEFAULT_PRODUCT_NAME,
                description="Desktop CAD/DXF matching",
            )
            db.add(prod)
            db.flush()

        lic_record = License(
            license_key=lic_key,
            customer_id=cust.id,
            product_id=prod.id,
            plan_type=args.plan.upper(),
            status="ACTIVE",
            max_activations=args.seats,
            current_activations=1 if hw_binding else 0,
        )
        db.add(lic_record)
        db.flush()

        sub = Subscription(
            license_id=lic_record.id,
            starts_at=now,
            expires_at=exp,
            status="ACTIVE",
            is_active=True,
        )
        db.add(sub)
        db.commit()
        lic_id = lic_record.id
        cust_id = cust.id
    except Exception as exc:
        db.rollback()
        # Fallback to standalone IDs if DB is unavailable
        lic_id = f"lic-{now.strftime('%Y%m%d%H%M%S')}"
        cust_id = f"cust-{now.strftime('%Y%m%d%H%M%S')}"
    finally:
        db.close()

    # Build and sign canonical payload
    signed_license = build_and_sign_license_payload(
        license_id=lic_id,
        license_key=lic_key,
        customer_id=cust_id,
        customer_name=args.customer,
        product_code=settings.DEFAULT_PRODUCT_CODE,
        plan_type=args.plan,
        issued_at=now,
        expires_at=exp,
        max_activations=args.seats,
        status="ACTIVE",
        machine_fingerprint=hw_binding,
    )

    out_path = Path(args.out) if args.out else Path(f"{lic_key}.lic")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(signed_license, f, indent=2)

    print("\n" + "=" * 65)
    print(" LICENSE FILE GENERATED AND SIGNED SUCCESSFULLY!")
    print("=" * 65)
    print(f" License Key    : {lic_key}")
    print(f" Customer       : {args.customer}")
    print(f" Plan           : {args.plan}")
    print(f" Issued At      : {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f" Expires At     : {exp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f" Bound Hardware : {hw_binding or 'Any (will bind on first activation)'}")
    print(f" Output File    : {out_path.resolve()}")
    print("=" * 65)
    print(f"To import into Design Matcher:\n  Click '🔑 License' -> 'Offline Activation' -> 'Import Signed License (.lic)'\n")


if __name__ == "__main__":
    main()
