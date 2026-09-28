"""Add se_table_categories table

Revision ID: 9f7c5e6d8a4b
Revises: se_001
Create Date: 2026-07-14 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "9f7c5e6d8a4b"
down_revision = "se_001"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "se_table_categories",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("database_id", sa.String(), sa.ForeignKey("se_databases.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("category_name", sa.String(), nullable=False, index=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("table_names_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("questions", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.UniqueConstraint("database_id", "category_name", name="uq_se_table_categories_name"),
    )


def downgrade() -> None:
    op.drop_table("se_table_categories")
