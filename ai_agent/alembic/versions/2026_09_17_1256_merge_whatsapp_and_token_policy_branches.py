"""merge whatsapp and token policy branches

Revision ID: 1938f126963c
Revises: d4b2e8f1c9a7, 7c4a1e9d2b5f
Create Date: 2026-09-17 12:56:11.269488

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = '1938f126963c'
down_revision = ('d4b2e8f1c9a7', '7c4a1e9d2b5f')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass