"""Add query_history_entries table

Revision ID: b3d7e1a9f2c4
Revises: f1a2b3c4d5e6
Create Date: 2026-08-14 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "b3d7e1a9f2c4"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "query_history_entries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("organization_id", sa.Integer(), sa.ForeignKey("organizations.id"), nullable=True),
        sa.Column("query", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        op.f("ix_query_history_entries_created_by"), "query_history_entries", ["created_by"], unique=False,
    )
    op.create_index(
        op.f("ix_query_history_entries_organization_id"), "query_history_entries", ["organization_id"], unique=False,
    )
    op.create_index(
        op.f("ix_query_history_entries_created_at"), "query_history_entries", ["created_at"], unique=False,
    )


def downgrade():
    op.drop_index(op.f("ix_query_history_entries_created_at"), table_name="query_history_entries")
    op.drop_index(op.f("ix_query_history_entries_organization_id"), table_name="query_history_entries")
    op.drop_index(op.f("ix_query_history_entries_created_by"), table_name="query_history_entries")
    op.drop_table("query_history_entries")
