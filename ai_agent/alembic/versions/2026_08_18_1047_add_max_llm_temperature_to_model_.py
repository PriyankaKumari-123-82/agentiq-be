"""add_max_llm_temperature_to_model_provider

Revision ID: 7436154890c0
Revises: c5e9f3b1a2c5
Create Date: 2026-08-18 10:47:57.320689

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = '7436154890c0'
down_revision = 'd6f0a4b2c3e6'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('model_provider', sa.Column('max_llm_temperature', sa.Float(), nullable=False, server_default='2.0'))
    op.execute("UPDATE model_provider SET max_llm_temperature = 1.0 WHERE provider = 'anthropic'")
    op.execute("UPDATE model_provider SET max_llm_temperature = 1.0 WHERE provider = 'mistral'")
    op.execute("UPDATE model_provider SET max_llm_temperature = 1.0 WHERE provider = 'cohere'")


def downgrade():
    op.drop_column('model_provider', 'max_llm_temperature')
