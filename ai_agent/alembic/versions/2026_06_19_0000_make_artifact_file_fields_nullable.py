"""Make artifact file fields nullable

Revision ID: 2026_06_19_0000
Revises: 2026_06_07_0000
Create Date: 2026-06-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2026_06_19_0000"
down_revision: Union[str, Sequence[str], None] = "2026_06_07_0000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "artifacts",
        "filename",
        existing_type=sa.String(),
        nullable=True,
    )
    op.alter_column(
        "artifacts",
        "storage_key",
        existing_type=sa.String(),
        nullable=True,
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE artifacts
        SET storage_key = ''
        WHERE storage_key IS NULL
        """
    )
    op.execute(
        """
        UPDATE artifacts
        SET filename = ''
        WHERE filename IS NULL
        """
    )
    op.alter_column(
        "artifacts",
        "storage_key",
        existing_type=sa.String(),
        nullable=False,
    )
    op.alter_column(
        "artifacts",
        "filename",
        existing_type=sa.String(),
        nullable=False,
    )
