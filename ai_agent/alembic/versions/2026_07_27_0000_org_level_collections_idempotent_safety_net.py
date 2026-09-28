"""org_level_collections_idempotent_safety_net

Revision ID: c5e7a9f1d3b2
Revises: b7f2c9a4d6e8
Create Date: 2026-07-27 00:00:00.000000

Re-applies the organization_documents / agent_documents / documents cleanup
from 2026_07_17_0001_org_level_collections.py (d4b8e21a8f9c), but guarded so
it's safe to run regardless of whether that migration's DDL already landed
on a given environment (e.g. via an earlier, later-reverted deploy attempt).
Every step is a no-op if its target state already exists.
"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = 'c5e7a9f1d3b2'
down_revision = 'b7f2c9a4d6e8'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # ------------------------------------------------------------------
    # 1. Create organization_documents table
    # ------------------------------------------------------------------
    if not inspector.has_table('organization_documents'):
        op.create_table(
            'organization_documents',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('document_id', sa.Integer(), nullable=False),
            sa.Column('organization_id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
            sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('document_id', 'organization_id', name='uq_org_doc_link'),
        )
        op.create_index('ix_organization_documents_document_id', 'organization_documents', ['document_id'], unique=False)
        op.create_index('ix_organization_documents_organization_id', 'organization_documents', ['organization_id'], unique=False)
        op.create_index('ix_organization_documents_user_id', 'organization_documents', ['user_id'], unique=False)

    # ------------------------------------------------------------------
    # 2. Migrate existing documents.organization_id → organization_documents
    # ------------------------------------------------------------------
    has_org_id = False
    if inspector.has_table('documents'):
        for col in inspector.get_columns('documents'):
            if col['name'] == 'organization_id':
                has_org_id = True
                break

    if has_org_id:
        op.execute("""
            INSERT INTO organization_documents (document_id, organization_id, user_id, created_at)
            SELECT id, organization_id, user_id, NOW()
            FROM documents
            WHERE organization_id IS NOT NULL
              AND is_active = TRUE
            ON CONFLICT (document_id, organization_id) DO NOTHING
        """)

    # ------------------------------------------------------------------
    # 3. Reset agent_documents unique constraint back to (document_id, agent_id)
    #    (Remove org-scoped constraint if it was applied in a prior migration)
    # ------------------------------------------------------------------
    op.execute("ALTER TABLE agent_documents DROP CONSTRAINT IF EXISTS uq_agent_doc_org_link;")
    op.execute("ALTER TABLE agent_documents DROP CONSTRAINT IF EXISTS fk_agent_documents_org;")
    op.execute("DROP INDEX IF EXISTS ix_agent_documents_organization_id;")
    op.execute("ALTER TABLE agent_documents DROP COLUMN IF EXISTS organization_id;")

    # Ensure clean simple unique constraint exists
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'uq_agent_doc_link'
            ) THEN
                ALTER TABLE agent_documents ADD CONSTRAINT uq_agent_doc_link UNIQUE (document_id, agent_id);
            END IF;
        END $$;
    """)

    # ------------------------------------------------------------------
    # 4. Drop legacy columns from documents table
    # ------------------------------------------------------------------
    # Drop FK and index for agent_id first
    op.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_agent_id_fkey;")
    op.execute("DROP INDEX IF EXISTS ix_documents_agent_id;")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS agent_id;")

    # Drop FK and index for organization_id
    op.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS fk_documents_org;")
    op.execute("DROP INDEX IF EXISTS ix_documents_organization_id;")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS organization_id;")


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Restore organization_id to documents
    if 'organization_id' not in {c['name'] for c in inspector.get_columns('documents')}:
        op.add_column('documents', sa.Column('organization_id', sa.Integer(), nullable=True))
        op.create_index('ix_documents_organization_id', 'documents', ['organization_id'], unique=False)
        op.create_foreign_key('fk_documents_org', 'documents', 'organizations', ['organization_id'], ['id'])

    # Restore agent_id to documents
    if 'agent_id' not in {c['name'] for c in inspector.get_columns('documents')}:
        op.add_column('documents', sa.Column('agent_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        op.create_index('ix_documents_agent_id', 'documents', ['agent_id'], unique=False)
        op.create_foreign_key('documents_agent_id_fkey', 'documents', 'ai_agents', ['agent_id'], ['id'])

    # Drop organization_documents table
    if inspector.has_table('organization_documents'):
        op.drop_index('ix_organization_documents_user_id', table_name='organization_documents')
        op.drop_index('ix_organization_documents_organization_id', table_name='organization_documents')
        op.drop_index('ix_organization_documents_document_id', table_name='organization_documents')
        op.drop_table('organization_documents')
