"""add web search provider columns

Revision ID: fdafaafcad98
Revises: b9e7f8c2a1d4
Create Date: 2026-08-25 16:25:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column
from datetime import datetime, timezone

# revision identifiers, used by Alembic.
revision = 'fdafaafcad98'
down_revision = 'b9e7f8c2a1d4'  
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. Add columns to model_provider
    op.add_column('model_provider', sa.Column('is_llm', sa.Boolean(), server_default='true', nullable=False))
    op.add_column('model_provider', sa.Column('is_websearch', sa.Boolean(), server_default='false', nullable=False))

    # Data migration for existing OpenAI models to have is_websearch = True
    op.execute("UPDATE model_provider SET is_websearch = true WHERE provider = 'openai'")

    # Update check constraint
    op.execute("ALTER TABLE model_provider DROP CONSTRAINT IF EXISTS ck_model_provider_type")
    op.execute("ALTER TABLE model_provider ADD CONSTRAINT ck_model_provider_type CHECK (type::text = ANY (ARRAY['llm'::character varying, 'embeddings'::character varying, 'websearch'::character varying]::text[]))")


    # Insert Tavily seed row
    op.execute(
        """
        INSERT INTO model_provider 
        (provider, model_name, type, dimension, litellm_prefix, is_default, is_active, supports_temperature, max_llm_temperature, created_at, updated_at, is_llm, is_websearch) 
        VALUES 
        ('tavily', 'tavily-search', 'websearch', NULL, '', false, true, false, 2.0, NOW(), NOW(), false, true)
        """
    )

    # 2. Add columns to ai_agent_settings
    op.add_column('ai_agent_settings', sa.Column('websearch_provider', sa.String(), nullable=True))
    op.add_column('ai_agent_settings', sa.Column('websearch_api_key_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_ai_agent_settings_websearch_api_key_id'), 'ai_agent_settings', ['websearch_api_key_id'], unique=False)
    op.create_foreign_key('fk_ai_agent_settings_websearch_api_key_id_api_keys', 'ai_agent_settings', 'api_keys', ['websearch_api_key_id'], ['id'])

def downgrade() -> None:
    # Reverse ai_agent_settings columns
    op.drop_constraint('fk_ai_agent_settings_websearch_api_key_id_api_keys', 'ai_agent_settings', type_='foreignkey')
    op.drop_index(op.f('ix_ai_agent_settings_websearch_api_key_id'), table_name='ai_agent_settings')
    op.drop_column('ai_agent_settings', 'websearch_api_key_id')
    op.drop_column('ai_agent_settings', 'websearch_provider')

    # Reverse model_provider changes
    op.execute("DELETE FROM model_provider WHERE provider = 'tavily' AND model_name = 'tavily-search'")
    
    op.execute("ALTER TABLE model_provider DROP CONSTRAINT IF EXISTS ck_model_provider_type")
    op.execute("ALTER TABLE model_provider ADD CONSTRAINT ck_model_provider_type CHECK (type::text = ANY (ARRAY['llm'::character varying, 'embeddings'::character varying]::text[]))")
    
    op.drop_column('model_provider', 'is_websearch')
    op.drop_column('model_provider', 'is_llm')
