"""add_model_provider_catalog

Revision ID: a9c7e4f2d8b1
Revises: d1e2f3a4b5c6
Create Date: 2026-07-20 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a9c7e4f2d8b1'
down_revision = 'd1e2f3a4b5c6'
branch_labels = None
depends_on = None


model_provider = sa.table(
    'model_provider',
    sa.column('provider', sa.String()),
    sa.column('model_name', sa.String()),
    sa.column('type', sa.String()),
    sa.column('dimension', sa.Integer()),
    sa.column('litellm_prefix', sa.String()),
    sa.column('is_default', sa.Boolean()),
    sa.column('is_active', sa.Boolean()),
)


# litellm_prefix: the string LiteLLM expects before "/" in the model string.
# is_default=True marks the one fallback model per provider used when an
# agent is created without an explicit model name.
MODEL_PROVIDER_ROWS = [
    # ── Anthropic ────────────────────────────────────────────────────────────
    {'provider': 'anthropic', 'model_name': 'claude-sonnet-5',       'type': 'llm', 'dimension': None, 'litellm_prefix': 'anthropic', 'is_default': True,  'is_active': True},
    {'provider': 'anthropic', 'model_name': 'claude-opus-4-8',       'type': 'llm', 'dimension': None, 'litellm_prefix': 'anthropic', 'is_default': False, 'is_active': True},
    {'provider': 'anthropic', 'model_name': 'claude-haiku-4-5',      'type': 'llm', 'dimension': None, 'litellm_prefix': 'anthropic', 'is_default': False, 'is_active': True},
    # ── OpenAI ───────────────────────────────────────────────────────────────
    {'provider': 'openai', 'model_name': 'gpt-5.6-sol',   'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-5.6-terra', 'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-5.6-luna',  'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-5.5',       'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-5.5-pro',   'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-5.4',       'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-5.4-pro',   'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-5.4-mini',  'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-5.4-nano',  'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-4.1',       'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-4.1-mini',  'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-4.1-nano',  'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-4o',        'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': True,  'is_active': True},
    {'provider': 'openai', 'model_name': 'gpt-4o-mini',   'type': 'llm', 'dimension': None, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    # ── Groq ─────────────────────────────────────────────────────────────────
    {'provider': 'groq', 'model_name': 'llama-3.3-70b-versatile', 'type': 'llm', 'dimension': None, 'litellm_prefix': 'groq', 'is_default': True,  'is_active': True},
    {'provider': 'groq', 'model_name': 'llama-3.1-8b-instant',   'type': 'llm', 'dimension': None, 'litellm_prefix': 'groq', 'is_default': False, 'is_active': True},
    {'provider': 'groq', 'model_name': 'openai/gpt-oss-120b',    'type': 'llm', 'dimension': None, 'litellm_prefix': 'groq', 'is_default': False, 'is_active': True},
    {'provider': 'groq', 'model_name': 'openai/gpt-oss-20b',     'type': 'llm', 'dimension': None, 'litellm_prefix': 'groq', 'is_default': False, 'is_active': True},
    # ── Gemini ───────────────────────────────────────────────────────────────
    {'provider': 'gemini', 'model_name': 'gemini-3.5-flash',      'type': 'llm', 'dimension': None, 'litellm_prefix': 'gemini', 'is_default': True,  'is_active': True},
    {'provider': 'gemini', 'model_name': 'gemini-3.1-flash-lite', 'type': 'llm', 'dimension': None, 'litellm_prefix': 'gemini', 'is_default': False, 'is_active': True},
    {'provider': 'gemini', 'model_name': 'gemini-3-flash-preview','type': 'llm', 'dimension': None, 'litellm_prefix': 'gemini', 'is_default': False, 'is_active': True},
    {'provider': 'gemini', 'model_name': 'gemini-2.5-flash',      'type': 'llm', 'dimension': None, 'litellm_prefix': 'gemini', 'is_default': False, 'is_active': True},
    # ── Cohere ───────────────────────────────────────────────────────────────
    {'provider': 'cohere', 'model_name': 'command-a-plus-05-2026',      'type': 'llm', 'dimension': None, 'litellm_prefix': 'cohere_chat', 'is_default': False, 'is_active': True},
    {'provider': 'cohere', 'model_name': 'command-r7b-12-2024',         'type': 'llm', 'dimension': None, 'litellm_prefix': 'cohere_chat', 'is_default': False, 'is_active': True},
    {'provider': 'cohere', 'model_name': 'command-a-translate-08-2025', 'type': 'llm', 'dimension': None, 'litellm_prefix': 'cohere_chat', 'is_default': False, 'is_active': True},
    {'provider': 'cohere', 'model_name': 'command-a-reasoning-08-2025', 'type': 'llm', 'dimension': None, 'litellm_prefix': 'cohere_chat', 'is_default': False, 'is_active': True},
    {'provider': 'cohere', 'model_name': 'command-r-plus-08-2024',      'type': 'llm', 'dimension': None, 'litellm_prefix': 'cohere_chat', 'is_default': True,  'is_active': True},
    {'provider': 'cohere', 'model_name': 'command-a-vision-07-2025',    'type': 'llm', 'dimension': None, 'litellm_prefix': 'cohere_chat', 'is_default': False, 'is_active': True},
    # ── Mistral ──────────────────────────────────────────────────────────────
    {'provider': 'mistral', 'model_name': 'mistral-tiny',        'type': 'llm', 'dimension': None, 'litellm_prefix': 'mistral', 'is_default': False, 'is_active': True},
    {'provider': 'mistral', 'model_name': 'mistral-medium-3.5',  'type': 'llm', 'dimension': None, 'litellm_prefix': 'mistral', 'is_default': False, 'is_active': True},
    {'provider': 'mistral', 'model_name': 'mistral-small-2603',  'type': 'llm', 'dimension': None, 'litellm_prefix': 'mistral', 'is_default': False, 'is_active': True},
    {'provider': 'mistral', 'model_name': 'mistral-large-2512',  'type': 'llm', 'dimension': None, 'litellm_prefix': 'mistral', 'is_default': True,  'is_active': True},
    {'provider': 'mistral', 'model_name': 'ministral-14b-2512',  'type': 'llm', 'dimension': None, 'litellm_prefix': 'mistral', 'is_default': False, 'is_active': True},
    {'provider': 'mistral', 'model_name': 'ministral-8b-2512',   'type': 'llm', 'dimension': None, 'litellm_prefix': 'mistral', 'is_default': False, 'is_active': True},
    {'provider': 'mistral', 'model_name': 'ministral-3b-2512',   'type': 'llm', 'dimension': None, 'litellm_prefix': 'mistral', 'is_default': False, 'is_active': True},
    # ── Embedding models ─────────────────────────────────────────────────────
    {'provider': 'openai', 'model_name': 'text-embedding-3-large', 'type': 'embeddings', 'dimension': 3072, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'openai', 'model_name': 'text-embedding-3-small', 'type': 'embeddings', 'dimension': 1536, 'litellm_prefix': 'openai', 'is_default': False, 'is_active': True},
    {'provider': 'baai',   'model_name': 'bge-large-en-v1.5',      'type': 'embeddings', 'dimension': 1024, 'litellm_prefix': 'baai',   'is_default': False, 'is_active': True},
]


def upgrade():
    op.create_table(
        'model_provider',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('dimension', sa.Integer(), nullable=True),
        sa.Column('litellm_prefix', sa.String(), nullable=False),
        sa.Column('is_default', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("type IN ('llm', 'embeddings')", name='ck_model_provider_type'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('provider', 'model_name', 'type', name='uq_model_provider_catalog'),
    )
    op.create_index('ix_model_provider_provider_type', 'model_provider', ['provider', 'type'], unique=False)
    op.bulk_insert(model_provider, MODEL_PROVIDER_ROWS)


def downgrade():
    op.drop_index('ix_model_provider_provider_type', table_name='model_provider')
    op.drop_table('model_provider')
