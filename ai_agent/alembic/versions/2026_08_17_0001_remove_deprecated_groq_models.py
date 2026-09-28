"""remove deprecated groq models

Revision ID: c5e9f3b1a2c5
Revises: a1b2c3d4e5f7
Create Date: 2026-08-17 10:45:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c5e9f3b1a2c5'
down_revision = 'a1b2c3d4e5f7'
branch_labels = None
depends_on = None


def upgrade():
    model_provider = sa.table(
        'model_provider',
        sa.column('provider', sa.String),
        sa.column('model_name', sa.String),
        sa.column('is_default', sa.Boolean)
    )
    
    # Remove the deprecated models
    op.execute(
        model_provider.delete().where(
            sa.and_(
                model_provider.c.provider == 'groq',
                model_provider.c.model_name.in_(['llama-3.1-8b-instant', 'llama-3.3-70b-versatile'])
            )
        )
    )
    
    # Set a new default model for groq since llama-3.3-70b-versatile was the default
    op.execute(
        model_provider.update().where(
            sa.and_(
                model_provider.c.provider == 'groq',
                model_provider.c.model_name == 'openai/gpt-oss-120b'
            )
        ).values(is_default=True)
    )


def downgrade():
    model_provider = sa.table(
        'model_provider',
        sa.column('provider', sa.String),
        sa.column('model_name', sa.String),
        sa.column('type', sa.String),
        sa.column('dimension', sa.Integer),
        sa.column('litellm_prefix', sa.String),
        sa.column('is_default', sa.Boolean),
        sa.column('is_active', sa.Boolean),
    )
    
    # Revert the default model change
    op.execute(
        model_provider.update().where(
            sa.and_(
                model_provider.c.provider == 'groq',
                model_provider.c.model_name == 'openai/gpt-oss-120b'
            )
        ).values(is_default=False)
    )
    
    # Re-insert the removed models
    op.bulk_insert(
        model_provider,
        [
            {'provider': 'groq', 'model_name': 'llama-3.3-70b-versatile', 'type': 'llm', 'dimension': None, 'litellm_prefix': 'groq', 'is_default': True,  'is_active': True},
            {'provider': 'groq', 'model_name': 'llama-3.1-8b-instant',   'type': 'llm', 'dimension': None, 'litellm_prefix': 'groq', 'is_default': False, 'is_active': True},
        ]
    )
