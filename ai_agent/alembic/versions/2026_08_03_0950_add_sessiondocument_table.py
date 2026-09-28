"""Add SessionDocument table

Revision ID: fef8a3fe77d0
Revises: b7e2f3a4c5d6
Create Date: 2026-08-03 09:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision = 'fef8a3fe77d0'
down_revision = 'f9d7c4a1b2e8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('session_documents',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('session_id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
    sa.ForeignKeyConstraint(['session_id'], ['user_sessions.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('document_id', 'session_id', name='uq_session_doc_link')
    )
    op.create_index(op.f('ix_session_documents_document_id'), 'session_documents', ['document_id'], unique=False)
    op.create_index(op.f('ix_session_documents_session_id'), 'session_documents', ['session_id'], unique=False)
    op.create_index(op.f('ix_session_documents_user_id'), 'session_documents', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_session_documents_user_id'), table_name='session_documents')
    op.drop_index(op.f('ix_session_documents_session_id'), table_name='session_documents')
    op.drop_index(op.f('ix_session_documents_document_id'), table_name='session_documents')
    op.drop_table('session_documents')
