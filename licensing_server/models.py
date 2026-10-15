"""
SQLAlchemy ORM models for the Licensing System.
Supports customers, products, licenses, subscriptions, device activations,
renewal transactions, offline license files, audit logs, and admin accounts.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship, Mapped

from licensing_server.database import Base


def utc_now() -> datetime:
    """Returns current UTC datetime with timezone awareness."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensures a datetime object is timezone-aware in UTC (fixes SQLite naive datetime issue)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def generate_uuid() -> str:
    """Generates a standard UUID4 hex string."""
    return str(uuid.uuid4())


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = Column(String(255), nullable=False)
    email: Mapped[str] = Column(String(255), unique=True, index=True, nullable=False)
    company: Mapped[Optional[str]] = Column(String(255), nullable=True)
    phone: Mapped[Optional[str]] = Column(String(50), nullable=True)
    notes: Mapped[Optional[str]] = Column(Text, nullable=True)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    licenses = relationship("License", back_populates="customer", cascade="all, delete-orphan")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    product_code: Mapped[str] = Column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = Column(String(255), nullable=False)
    description: Mapped[Optional[str]] = Column(Text, nullable=True)
    is_active: Mapped[bool] = Column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    licenses = relationship("License", back_populates="product")


class License(Base):
    __tablename__ = "licenses"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    license_key: Mapped[str] = Column(String(64), unique=True, index=True, nullable=False)
    customer_id: Mapped[str] = Column(String(36), ForeignKey("customers.id", ondelete="CASCADE"), index=True, nullable=False)
    product_id: Mapped[str] = Column(String(36), ForeignKey("products.id", ondelete="RESTRICT"), index=True, nullable=False)
    plan_type: Mapped[str] = Column(String(32), default="1_year", nullable=False)
    status: Mapped[str] = Column(String(32), default="ACTIVE", index=True, nullable=False)  # ACTIVE, EXPIRED, SUSPENDED, REVOKED
    max_activations: Mapped[int] = Column(Integer, default=1, nullable=False)
    notes: Mapped[Optional[str]] = Column(Text, nullable=True)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    customer = relationship("Customer", back_populates="licenses")
    product = relationship("Product", back_populates="licenses")
    subscriptions = relationship("Subscription", back_populates="license", cascade="all, delete-orphan", order_by="desc(Subscription.created_at)")
    device_activations = relationship("DeviceActivation", back_populates="license", cascade="all, delete-orphan")
    renewal_transactions = relationship("RenewalTransaction", back_populates="license", cascade="all, delete-orphan")
    offline_files = relationship("OfflineLicenseFile", back_populates="license", cascade="all, delete-orphan")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    license_id: Mapped[str] = Column(String(36), ForeignKey("licenses.id", ondelete="CASCADE"), index=True, nullable=False)
    starts_at: Mapped[datetime] = Column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = Column(DateTime(timezone=True), index=True, nullable=False)
    status: Mapped[str] = Column(String(32), default="ACTIVE", nullable=False)
    is_active: Mapped[bool] = Column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    license = relationship("License", back_populates="subscriptions")


class DeviceActivation(Base):
    __tablename__ = "device_activations"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    license_id: Mapped[str] = Column(String(36), ForeignKey("licenses.id", ondelete="CASCADE"), index=True, nullable=False)
    machine_fingerprint: Mapped[str] = Column(String(128), index=True, nullable=False)
    machine_name: Mapped[str] = Column(String(255), default="Unknown-Computer", nullable=False)
    os_info: Mapped[str] = Column(String(255), default="Unknown-OS", nullable=False)
    ip_address: Mapped[Optional[str]] = Column(String(64), nullable=True)
    activated_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    last_checkin_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    is_active: Mapped[bool] = Column(Boolean, default=True, index=True, nullable=False)
    deactivated_at: Mapped[Optional[datetime]] = Column(DateTime(timezone=True), nullable=True)

    license = relationship("License", back_populates="device_activations")


class RenewalTransaction(Base):
    __tablename__ = "renewal_transactions"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    license_id: Mapped[str] = Column(String(36), ForeignKey("licenses.id", ondelete="CASCADE"), index=True, nullable=False)
    subscription_id: Mapped[Optional[str]] = Column(String(36), ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=True)
    plan_type: Mapped[str] = Column(String(32), nullable=False)
    duration_days: Mapped[int] = Column(Integer, nullable=False)
    previous_expiration: Mapped[datetime] = Column(DateTime(timezone=True), nullable=False)
    new_expiration: Mapped[datetime] = Column(DateTime(timezone=True), nullable=False)
    payment_reference: Mapped[Optional[str]] = Column(String(128), nullable=True)
    notes: Mapped[Optional[str]] = Column(Text, nullable=True)
    created_by: Mapped[str] = Column(String(128), default="admin", nullable=False)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    license = relationship("License", back_populates="renewal_transactions")


class OfflineLicenseFile(Base):
    __tablename__ = "offline_license_files"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    license_id: Mapped[str] = Column(String(36), ForeignKey("licenses.id", ondelete="CASCADE"), index=True, nullable=False)
    machine_fingerprint: Mapped[str] = Column(String(128), index=True, nullable=False)
    nonce: Mapped[str] = Column(String(64), unique=True, index=True, nullable=False)
    request_payload: Mapped[str] = Column(Text, nullable=False)
    signed_response_payload: Mapped[str] = Column(Text, nullable=False)
    issued_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    expires_at: Mapped[datetime] = Column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = Column(String(128), default="admin", nullable=False)

    license = relationship("License", back_populates="offline_files")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    actor: Mapped[str] = Column(String(128), default="system", nullable=False)
    action: Mapped[str] = Column(String(64), index=True, nullable=False)  # CREATE_LICENSE, RENEW_LICENSE, ACTIVATE_DEVICE, etc.
    resource_type: Mapped[str] = Column(String(64), index=True, nullable=False)
    resource_id: Mapped[str] = Column(String(128), index=True, nullable=False)
    details_json: Mapped[str] = Column(Text, default="{}", nullable=False)
    ip_address: Mapped[Optional[str]] = Column(String(64), nullable=True)
    timestamp: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, index=True, nullable=False)


class AdminUser(Base):
    __tablename__ = "admin_users"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    username: Mapped[str] = Column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str] = Column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = Column(String(255), nullable=False)
    role: Mapped[str] = Column(String(32), default="admin", nullable=False)  # superadmin, admin, support
    is_active: Mapped[bool] = Column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), default=utc_now, nullable=False)
