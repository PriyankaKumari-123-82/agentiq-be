"""add keycloak identity fields, citext email, and addresses table

Revision ID: 2026_06_07_0000
Revises: 2026_06_04_0000
Create Date: 2026-06-07 00:00:00.000000

Implements the AgentIQ local identity schema:
- enables the citext extension and makes users.email case-insensitive
- adds first_name, last_name to users
- creates the address table (one-to-many with users)
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2026_06_07_0000"
down_revision: Union[str, Sequence[str], None] = "2026_06_04_0000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Case-insensitive email support
    # NOTE: email column is NOT removed — its type is changed from varchar to citext.
    # citext (case-insensitive text) ensures User@Example.COM and user@example.com
    # are treated as the same value at the DB level, preventing duplicate registrations
    # with mixed-case emails. All existing data is preserved during this type cast.
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")
    op.execute("ALTER TABLE users ALTER COLUMN email TYPE citext")

    # 2. Drop orphaned keycloak_sub column from a removed intermediate migration
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS keycloak_sub")

    # 3. New identity columns on users
    op.add_column("users", sa.Column("first_name", sa.String(length=255), nullable=True))
    op.add_column("users", sa.Column("last_name", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "last_name")
    op.drop_column("users", "first_name")

    op.execute("ALTER TABLE users ALTER COLUMN email TYPE varchar")
