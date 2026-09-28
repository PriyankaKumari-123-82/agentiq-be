"""add_org_kb_and_agent_documents

Revision ID: 5983b6791e44
Revises: c1d2e3f4a5b6
Create Date: 2026-07-15 10:29:56.584115

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = '5983b6791e44'
down_revision = 'c1d2e3f4a5b6'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('agent_documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('agent_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['agent_id'], ['ai_agents.id'], ),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id', 'agent_id', name='uq_agent_doc_link')
    )
    op.create_index(op.f('ix_agent_documents_agent_id'), 'agent_documents', ['agent_id'], unique=False)
    op.create_index(op.f('ix_agent_documents_document_id'), 'agent_documents', ['document_id'], unique=False)
    op.create_index(op.f('ix_agent_documents_user_id'), 'agent_documents', ['user_id'], unique=False)

    op.add_column('ai_agents', sa.Column('organization_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_ai_agents_organization_id'), 'ai_agents', ['organization_id'], unique=False)
    op.create_foreign_key('fk_ai_agents_org', 'ai_agents', 'organizations', ['organization_id'], ['id'])

    op.add_column('documents', sa.Column('organization_id', sa.Integer(), nullable=True))
    op.alter_column('documents', 'agent_id',
                    existing_type=sa.VARCHAR(),
                    nullable=True)
    op.create_index(op.f('ix_documents_organization_id'), 'documents', ['organization_id'], unique=False)
    op.create_foreign_key('fk_documents_org', 'documents', 'organizations', ['organization_id'], ['id'])


def downgrade():
    op.drop_constraint('fk_documents_org', 'documents', type_='foreignkey')
    op.drop_index(op.f('ix_documents_organization_id'), table_name='documents')
    op.alter_column('documents', 'agent_id',
                    existing_type=sa.VARCHAR(),
                    nullable=False)
    op.drop_column('documents', 'organization_id')

    op.drop_constraint('fk_ai_agents_org', 'ai_agents', type_='foreignkey')
    op.drop_index(op.f('ix_ai_agents_organization_id'), table_name='ai_agents')
    op.drop_column('ai_agents', 'organization_id')

    op.drop_index(op.f('ix_agent_documents_user_id'), table_name='agent_documents')
    op.drop_index(op.f('ix_agent_documents_document_id'), table_name='agent_documents')
    op.drop_index(op.f('ix_agent_documents_agent_id'), table_name='agent_documents')
    op.drop_table('agent_documents')
