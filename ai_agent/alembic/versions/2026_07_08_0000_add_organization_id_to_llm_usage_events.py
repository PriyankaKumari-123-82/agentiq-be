"""add organization_id to llm_usage_events

Revision ID: c1d2e3f4a5b6
Revises: b1c2d3e4f5a6
Create Date: 2026-07-08 09:15:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c1d2e3f4a5b6'
down_revision = 'b1c2d3e4f5a6'
branch_labels = None
depends_on = None


def upgrade():
    # Add organization_id column to llm_usage_events
    op.add_column('llm_usage_events', sa.Column('organization_id', sa.Integer(), nullable=True))
    # Add foreign key constraint
    op.create_foreign_key(
        'fk_llm_usage_events_organization_id',
        'llm_usage_events', 'organizations',
        ['organization_id'], ['id']
    )


def downgrade():
    # Remove foreign key constraint
    op.drop_constraint('fk_llm_usage_events_organization_id', 'llm_usage_events', type_='foreignkey')
    # Remove organization_id column
    op.drop_column('llm_usage_events', 'organization_id')
