"""split user token policy into independent daily/weekly/monthly limits

Revision ID: 7c4a1e9d2b5f
Revises: 590f9415bc6f
Create Date: 2026-09-13 01:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import table, column


# revision identifiers, used by Alembic.
revision = '7c4a1e9d2b5f'
down_revision = '590f9415bc6f'
branch_labels = None
depends_on = None

_TABLE = 'user_token_policies'
_ID_COLUMN = 'user_id'
_DEFAULTS = {"daily": 10000, "weekly": 50000, "monthly": 100000}


def upgrade():
    for period in ("daily", "weekly", "monthly"):
        op.add_column(_TABLE, sa.Column(f'{period}_limit_enabled', sa.Boolean(), nullable=False, server_default=sa.false()))
        op.add_column(_TABLE, sa.Column(f'{period}_max_tokens', sa.Integer(), nullable=False, server_default=str(_DEFAULTS[period])))

    conn = op.get_bind()
    t = table(
        _TABLE,
        column(_ID_COLUMN),
        column('limit_enabled', sa.Boolean()),
        column('max_tokens', sa.Integer()),
        column('reset_period', sa.String()),
        column('daily_limit_enabled', sa.Boolean()),
        column('daily_max_tokens', sa.Integer()),
        column('weekly_limit_enabled', sa.Boolean()),
        column('weekly_max_tokens', sa.Integer()),
        column('monthly_limit_enabled', sa.Boolean()),
        column('monthly_max_tokens', sa.Integer()),
    )
    rows = conn.execute(sa.select(t.c[_ID_COLUMN], t.c.limit_enabled, t.c.max_tokens, t.c.reset_period)).fetchall()
    for row_id, limit_enabled, max_tokens, reset_period in rows:
        period = reset_period if reset_period in ("daily", "weekly", "monthly") else "monthly"
        conn.execute(
            t.update()
            .where(t.c[_ID_COLUMN] == row_id)
            .values(**{
                f"{period}_limit_enabled": bool(limit_enabled),
                f"{period}_max_tokens": max_tokens,
            })
        )

    op.drop_column(_TABLE, 'limit_enabled')
    op.drop_column(_TABLE, 'max_tokens')
    op.drop_column(_TABLE, 'reset_period')


def downgrade():
    op.add_column(_TABLE, sa.Column('limit_enabled', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column(_TABLE, sa.Column('max_tokens', sa.Integer(), nullable=False, server_default='100000'))
    op.add_column(_TABLE, sa.Column('reset_period', sa.String(), nullable=False, server_default='monthly'))

    conn = op.get_bind()
    t = table(
        _TABLE,
        column(_ID_COLUMN),
        column('limit_enabled', sa.Boolean()),
        column('max_tokens', sa.Integer()),
        column('reset_period', sa.String()),
        column('daily_limit_enabled', sa.Boolean()),
        column('daily_max_tokens', sa.Integer()),
        column('weekly_limit_enabled', sa.Boolean()),
        column('weekly_max_tokens', sa.Integer()),
        column('monthly_limit_enabled', sa.Boolean()),
        column('monthly_max_tokens', sa.Integer()),
    )
    rows = conn.execute(sa.select(
        t.c[_ID_COLUMN], t.c.daily_limit_enabled, t.c.daily_max_tokens,
        t.c.weekly_limit_enabled, t.c.weekly_max_tokens,
        t.c.monthly_limit_enabled, t.c.monthly_max_tokens,
    )).fetchall()
    for row_id, d_en, d_max, w_en, w_max, m_en, m_max in rows:
        if m_en:
            limit_enabled, max_tokens, reset_period = True, m_max, "monthly"
        elif w_en:
            limit_enabled, max_tokens, reset_period = True, w_max, "weekly"
        elif d_en:
            limit_enabled, max_tokens, reset_period = True, d_max, "daily"
        else:
            limit_enabled, max_tokens, reset_period = False, m_max, "monthly"
        conn.execute(
            t.update().where(t.c[_ID_COLUMN] == row_id).values(
                limit_enabled=limit_enabled, max_tokens=max_tokens, reset_period=reset_period,
            )
        )

    for period in ("daily", "weekly", "monthly"):
        op.drop_column(_TABLE, f'{period}_limit_enabled')
        op.drop_column(_TABLE, f'{period}_max_tokens')
