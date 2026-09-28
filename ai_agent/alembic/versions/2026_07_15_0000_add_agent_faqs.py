"""add agent_faqs table

Revision ID: d1e2f3a4b5c6
Revises: c1d2e3f4a5b6
Create Date: 2026-07-15 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd1e2f3a4b5c6'
down_revision = '9f7c5e6d8a4b'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'agent_faqs',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('agent_id', sa.String(), sa.ForeignKey('ai_agents.id'), nullable=False, index=True),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('source', sa.String(), nullable=False, server_default='agent_config', index=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )


def downgrade():
    op.drop_table('agent_faqs')
