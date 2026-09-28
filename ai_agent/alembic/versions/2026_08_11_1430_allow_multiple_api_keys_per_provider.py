"""Allow multiple API keys per provider in an organization and add name field

Revision ID: f81a2b3c4d5e
Revises: e4827ce43e2e
Create Date: 2026-08-11 14:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'f81a2b3c4d5e'
down_revision = 'e9b8c7d6e5f4'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Add name column with empty default for existing rows
    op.execute("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS name VARCHAR(255) NOT NULL DEFAULT '';")

    # 2. Backfill existing rows with provider-based default name
    op.execute("UPDATE api_keys SET name = INITCAP(provider) || ' Key' WHERE name = '' OR name IS NULL;")

    # 3. Drop legacy unique indexes that restricted to 1 active key per provider or 1 default key per org
    op.execute("DROP INDEX IF EXISTS uq_api_keys_org_provider_active;")
    op.execute("DROP INDEX IF EXISTS uq_api_keys_org_default;")

    # 4. Create new partial unique index enforcing max ONE default key per (organization_id, provider)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'uq_api_keys_org_provider_default'
            ) THEN
                CREATE UNIQUE INDEX uq_api_keys_org_provider_default ON api_keys (organization_id, provider)
                WHERE is_default = true AND is_active = true;
            END IF;
        END $$;
    """)


def downgrade():
    op.execute("DROP INDEX IF EXISTS uq_api_keys_org_provider_default;")
    op.execute("ALTER TABLE api_keys DROP COLUMN IF EXISTS name;")
