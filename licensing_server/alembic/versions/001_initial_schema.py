"""Initial schema for licensing and subscriptions

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-14 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Customers
    op.create_table(
        'customers',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('company', sa.String(length=255), nullable=True),
        sa.Column('phone', sa.String(length=50), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_customers_email', 'customers', ['email'], unique=True)

    # 2. Products
    op.create_table(
        'products',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('product_code', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_products_product_code', 'products', ['product_code'], unique=True)

    # 3. Licenses
    op.create_table(
        'licenses',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('license_key', sa.String(length=64), nullable=False),
        sa.Column('customer_id', sa.String(length=36), nullable=False),
        sa.Column('product_id', sa.String(length=36), nullable=False),
        sa.Column('plan_type', sa.String(length=32), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('max_activations', sa.Integer(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_licenses_license_key', 'licenses', ['license_key'], unique=True)
    op.create_index('ix_licenses_customer_id', 'licenses', ['customer_id'], unique=False)
    op.create_index('ix_licenses_product_id', 'licenses', ['product_id'], unique=False)
    op.create_index('ix_licenses_status', 'licenses', ['status'], unique=False)

    # 4. Subscriptions
    op.create_table(
        'subscriptions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('license_id', sa.String(length=36), nullable=False),
        sa.Column('starts_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['license_id'], ['licenses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_subscriptions_license_id', 'subscriptions', ['license_id'], unique=False)
    op.create_index('ix_subscriptions_expires_at', 'subscriptions', ['expires_at'], unique=False)

    # 5. Device Activations
    op.create_table(
        'device_activations',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('license_id', sa.String(length=36), nullable=False),
        sa.Column('machine_fingerprint', sa.String(length=128), nullable=False),
        sa.Column('machine_name', sa.String(length=255), nullable=False),
        sa.Column('os_info', sa.String(length=255), nullable=False),
        sa.Column('ip_address', sa.String(length=64), nullable=True),
        sa.Column('activated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_checkin_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('deactivated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['license_id'], ['licenses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_device_activations_license_id', 'device_activations', ['license_id'], unique=False)
    op.create_index('ix_device_activations_machine_fingerprint', 'device_activations', ['machine_fingerprint'], unique=False)
    op.create_index('ix_device_activations_is_active', 'device_activations', ['is_active'], unique=False)

    # 6. Renewal Transactions
    op.create_table(
        'renewal_transactions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('license_id', sa.String(length=36), nullable=False),
        sa.Column('subscription_id', sa.String(length=36), nullable=True),
        sa.Column('plan_type', sa.String(length=32), nullable=False),
        sa.Column('duration_days', sa.Integer(), nullable=False),
        sa.Column('previous_expiration', sa.DateTime(timezone=True), nullable=False),
        sa.Column('new_expiration', sa.DateTime(timezone=True), nullable=False),
        sa.Column('payment_reference', sa.String(length=128), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_by', sa.String(length=128), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['license_id'], ['licenses.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['subscription_id'], ['subscriptions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_renewal_transactions_license_id', 'renewal_transactions', ['license_id'], unique=False)

    # 7. Offline License Files
    op.create_table(
        'offline_license_files',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('license_id', sa.String(length=36), nullable=False),
        sa.Column('machine_fingerprint', sa.String(length=128), nullable=False),
        sa.Column('nonce', sa.String(length=64), nullable=False),
        sa.Column('request_payload', sa.Text(), nullable=False),
        sa.Column('signed_response_payload', sa.Text(), nullable=False),
        sa.Column('issued_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_by', sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(['license_id'], ['licenses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_offline_license_files_nonce', 'offline_license_files', ['nonce'], unique=True)
    op.create_index('ix_offline_license_files_license_id', 'offline_license_files', ['license_id'], unique=False)
    op.create_index('ix_offline_license_files_machine_fingerprint', 'offline_license_files', ['machine_fingerprint'], unique=False)

    # 8. Audit Logs
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('actor', sa.String(length=128), nullable=False),
        sa.Column('action', sa.String(length=64), nullable=False),
        sa.Column('resource_type', sa.String(length=64), nullable=False),
        sa.Column('resource_id', sa.String(length=128), nullable=False),
        sa.Column('details_json', sa.Text(), nullable=False),
        sa.Column('ip_address', sa.String(length=64), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_audit_logs_action', 'audit_logs', ['action'], unique=False)
    op.create_index('ix_audit_logs_resource_type', 'audit_logs', ['resource_type'], unique=False)
    op.create_index('ix_audit_logs_resource_id', 'audit_logs', ['resource_id'], unique=False)
    op.create_index('ix_audit_logs_timestamp', 'audit_logs', ['timestamp'], unique=False)

    # 9. Admin Users
    op.create_table(
        'admin_users',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('username', sa.String(length=64), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=32), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_admin_users_username', 'admin_users', ['username'], unique=True)
    op.create_index('ix_admin_users_email', 'admin_users', ['email'], unique=True)


def downgrade() -> None:
    op.drop_table('admin_users')
    op.drop_table('audit_logs')
    op.drop_table('offline_license_files')
    op.drop_table('renewal_transactions')
    op.drop_table('device_activations')
    op.drop_table('subscriptions')
    op.drop_table('licenses')
    op.drop_table('products')
    op.drop_table('customers')
