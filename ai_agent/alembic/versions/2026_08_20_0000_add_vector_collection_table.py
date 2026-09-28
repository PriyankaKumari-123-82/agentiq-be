
"""Add vector_collection table

Revision ID: e2b7c9a4d1f6
Revises: 90bafbc3e0a1
Create Date: 2026-08-20 15:20:00.000000

Introduces the ``vector_collection`` table which stores the Milvus collection
name per organization. Each org gets one row with a generated name of the form
``org_<uuid_hex>`` (32 hex chars, no hyphens).
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "e2b7c9a4d1f6"
down_revision = "90bafbc3e0a1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "vector_collection",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("collection_name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", name="uq_vector_collection_org"),
        sa.UniqueConstraint("collection_name", name="uq_vector_collection_name"),
    )
    op.create_index(
        "ix_vector_collection_organization_id",
        "vector_collection",
        ["organization_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_vector_collection_organization_id", table_name="vector_collection")
    op.drop_table("vector_collection")


