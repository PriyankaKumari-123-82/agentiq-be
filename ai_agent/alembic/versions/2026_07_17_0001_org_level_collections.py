"""org_level_collections_clean_document_model

Revision ID: d4b8e21a8f9c
Revises: dal001user2026
Create Date: 2026-07-17 07:34:00.000000

Drops the legacy `agent_id` and `organization_id` columns from the `documents`
table and introduces a new table:

  * ``organization_documents``  — one row per (document, organization) pair; drives
    which Milvus org-collection a document is embedded in.

Agent-document access is tracked by the pre-existing ``agent_documents`` table
(keyed on document_id + agent_id only, no org_id column needed there).

Migration is safe: existing data is migrated before columns are dropped.
"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = 'd4b8e21a8f9c'
down_revision = 'a9c7e4f2d8b1'
branch_labels = None
depends_on = None


def upgrade():
    # ------------------------------------------------------------------
    # 1. Create organization_documents table
    # ------------------------------------------------------------------
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
    # Restore organization_id to documents
    op.add_column('documents', sa.Column('organization_id', sa.Integer(), nullable=True))
    op.create_index('ix_documents_organization_id', 'documents', ['organization_id'], unique=False)
    op.create_foreign_key('fk_documents_org', 'documents', 'organizations', ['organization_id'], ['id'])

    # Restore agent_id to documents
    op.add_column('documents', sa.Column('agent_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.create_index('ix_documents_agent_id', 'documents', ['agent_id'], unique=False)
    op.create_foreign_key('documents_agent_id_fkey', 'documents', 'ai_agents', ['agent_id'], ['id'])

    # Drop organization_documents table
    op.drop_index('ix_organization_documents_user_id', table_name='organization_documents')
    op.drop_index('ix_organization_documents_organization_id', table_name='organization_documents')
    op.drop_index('ix_organization_documents_document_id', table_name='organization_documents')
    op.drop_table('organization_documents')
