"""add_websearch_model_field_and_fix_websearch_flags

Revision ID: c5d6e7f8g9h0
Revises: fdafaafcad98
Create Date: 2026-09-02 14:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c5d6e7f8g9h0'
down_revision = 'fdafaafcad98'
branch_labels = None
depends_on = None


def upgrade():
    # Add websearch_model column to ai_agent_settings table
    op.add_column('ai_agent_settings', sa.Column('websearch_model', sa.String(), nullable=True))
    
    # Set ALL models to is_websearch = false first
    op.execute("UPDATE model_provider SET is_websearch = false")
    
    # Then set ONLY specific OpenAI models to true
    openai_websearch_models = [
        'gpt-5.6-sol',
        'gpt-5.6-terra',
        'gpt-5.6-luna',
        'gpt-5.5',
        'gpt-5.5-pro',
        'gpt-5.4',
        'gpt-5.4-pro',
        'gpt-5.4-mini',
        'gpt-4.1',
        'gpt-4.1-mini',
        'gpt-4o',
    ]
    
    for model in openai_websearch_models:
        op.execute(
            f"UPDATE model_provider SET is_websearch = true "
            f"WHERE provider = 'openai' AND model_name = '{model}'"
        )
    
    # Set Tavily models to is_websearch = true
    op.execute("UPDATE model_provider SET is_websearch = true WHERE provider = 'tavily'")


def downgrade():
    # Revert: Set all OpenAI models to is_websearch = true
    op.execute("UPDATE model_provider SET is_websearch = true WHERE provider = 'openai'")
    
    # Remove websearch_model column from ai_agent_settings
    op.drop_column('ai_agent_settings', 'websearch_model')
