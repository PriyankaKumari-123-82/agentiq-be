"""Drop password column from users table

Revision ID: 2026_06_22_0000
Revises: 2026_06_19_0000
Create Date: 2026-06-22 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "2026_06_22_0000"
down_revision = "2026_06_19_0000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("users", "password")


def downgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "password",
            sa.String(255),
            nullable=True
        )
    )
