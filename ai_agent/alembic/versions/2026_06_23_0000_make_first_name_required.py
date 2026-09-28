"""Make first_name required on users table

Revision ID: 1636297f7af1
Revises: 2026_06_23_0000
Create Date: 2026-06-25 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "1636297f7af1"
down_revision = "2026_06_23_0000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Backfill existing users with null/empty first_name
    op.execute("UPDATE users SET first_name = 'User' WHERE first_name IS NULL OR first_name = ''")
    # Make first_name non-nullable
    op.alter_column("users", "first_name", existing_type=sa.String(length=255), nullable=False)


def downgrade() -> None:
    op.alter_column("users", "first_name", existing_type=sa.String(length=255), nullable=True)
