"""Add is_active column to organizations and roles (safety-net)

Revision ID: f1a2b3c4d5e6
Revises: e2a9d4c7b1f3
Create Date: 2026-08-13 09:51:00.000000

Migration e4827ce43e2e was stamped but its DDL never executed, leaving
organizations.is_active and roles.is_active missing from the schema.
This safety-net migration re-applies those columns idempotently so every
environment converges regardless of prior state.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "f1a2b3c4d5e6"
down_revision = "e2a9d4c7b1f3"
branch_labels = None
depends_on = None

# Tables and their expected is_active column from e4827ce43e2e.
_TARGETS = ["organizations", "roles"]


def _has_column(inspector, table: str, column: str) -> bool:
    return any(c["name"] == column for c in inspector.get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    for table in _TARGETS:
        if not inspector.has_table(table):
            continue
        if _has_column(inspector, table, "is_active"):
            continue
        op.add_column(
            table,
            sa.Column(
                "is_active",
                sa.Boolean(),
                server_default="true",
                nullable=True,
            ),
        )
        op.create_index(
            op.f(f"ix_{table}_is_active"), table, ["is_active"], unique=False
        )

    # organization_documents may not exist yet in every environment.
    if inspector.has_table("organization_documents"):
        if not _has_column(inspector, "organization_documents", "is_active"):
            op.add_column(
                "organization_documents",
                sa.Column(
                    "is_active",
                    sa.Boolean(),
                    server_default="true",
                    nullable=True,
                ),
            )
            op.create_index(
                op.f("ix_organization_documents_is_active"),
                "organization_documents",
                ["is_active"],
                unique=False,
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    for table in [*_TARGETS, "organization_documents"]:
        if not inspector.has_table(table):
            continue
        if not _has_column(inspector, table, "is_active"):
            continue
        op.drop_index(op.f(f"ix_{table}_is_active"), table_name=table)
        op.drop_column(table, "is_active")
