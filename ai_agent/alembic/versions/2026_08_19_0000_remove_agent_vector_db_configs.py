"""remove agent vector db configs

Revision ID: 90bafbc3e0a1
Revises: 7436154890c0
Create Date: 2026-08-19 13:10:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '90bafbc3e0a1'
down_revision = '7436154890c0'
branch_labels = None
depends_on = None


def upgrade():
    # Remove vector_store_name from ai_agents
    with op.batch_alter_table('ai_agents', schema=None) as batch_op:
        batch_op.drop_column('vector_store_name')

    # Remove vector configs from ai_agent_settings
    with op.batch_alter_table('ai_agent_settings', schema=None) as batch_op:
        batch_op.drop_column('vector_db_type')
        batch_op.drop_column('vector_chunk_size')
        batch_op.drop_column('vector_chunk_overlap')
        batch_op.drop_column('vector_embedding_model')
        batch_op.drop_column('vector_max_chunks')
        batch_op.drop_column('vector_top_k')
        batch_op.drop_column('vector_similarity_threshold')


def downgrade():
    with op.batch_alter_table('ai_agent_settings', schema=None) as batch_op:
        batch_op.add_column(sa.Column('vector_similarity_threshold', sa.Float(), nullable=False, server_default='0.0'))
        batch_op.add_column(sa.Column('vector_top_k', sa.Integer(), nullable=False, server_default='5'))
        batch_op.add_column(sa.Column('vector_max_chunks', sa.Integer(), nullable=False, server_default='5'))
        batch_op.add_column(sa.Column('vector_embedding_model', sa.String(), nullable=False, server_default='text-embedding-3-small'))
        batch_op.add_column(sa.Column('vector_chunk_overlap', sa.Integer(), nullable=False, server_default='200'))
        batch_op.add_column(sa.Column('vector_chunk_size', sa.Integer(), nullable=False, server_default='1000'))
        batch_op.add_column(sa.Column('vector_db_type', sa.String(), nullable=True))

    with op.batch_alter_table('ai_agents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('vector_store_name', sa.String(), nullable=True))
