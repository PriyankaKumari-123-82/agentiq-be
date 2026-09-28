"""split org/agent token policies into independent daily/weekly/monthly limits

Revision ID: a1b2c3d4e5f6
Revises: d4b2e8f1c9a7
Create Date: 2026-09-13 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import table, column


# revision identifiers, used by Alembic.
revision = '590f9415bc6f'
down_revision = '4ea94ac7e147'
branch_labels = None
depends_on = None


_DEFAULTS = {
    "organization_token_policies": {"daily": 500000, "weekly": 2500000, "monthly": 10000000},
    "agent_token_policies": {"daily": 50000, "weekly": 250000, "monthly": 1000000},
}


def _add_period_columns(table_name: str):
    defaults = _DEFAULTS[table_name]
    for period in ("daily", "weekly", "monthly"):
        op.add_column(table_name, sa.Column(f'{period}_limit_enabled', sa.Boolean(), nullable=False, server_default=sa.false()))
        op.add_column(table_name, sa.Column(f'{period}_max_tokens', sa.Integer(), nullable=False, server_default=str(defaults[period])))


def _migrate_existing_rows(table_name: str, id_column: str):
    """Copy each row's old (limit_enabled, max_tokens, reset_period) into the matching new period columns."""
    conn = op.get_bind()
    t = table(
        table_name,
        column(id_column),
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
    rows = conn.execute(sa.select(t.c[id_column], t.c.limit_enabled, t.c.max_tokens, t.c.reset_period)).fetchall()
    for row_id, limit_enabled, max_tokens, reset_period in rows:
        period = reset_period if reset_period in ("daily", "weekly", "monthly") else "monthly"
        conn.execute(
            t.update()
            .where(t.c[id_column] == row_id)
            .values(**{
                f"{period}_limit_enabled": bool(limit_enabled),
                f"{period}_max_tokens": max_tokens,
            })
        )


def _drop_old_columns(table_name: str):
    op.drop_column(table_name, 'limit_enabled')
    op.drop_column(table_name, 'max_tokens')
    op.drop_column(table_name, 'reset_period')


def upgrade():
    for table_name, id_column in (
        ('organization_token_policies', 'organization_id'),
        ('agent_token_policies', 'agent_id'),
    ):
        _add_period_columns(table_name)
        _migrate_existing_rows(table_name, id_column)
        _drop_old_columns(table_name)


def downgrade():
    for table_name, id_column in (
        ('organization_token_policies', 'organization_id'),
        ('agent_token_policies', 'agent_id'),
    ):
        op.add_column(table_name, sa.Column('limit_enabled', sa.Boolean(), nullable=False, server_default=sa.false()))
        op.add_column(table_name, sa.Column('max_tokens', sa.Integer(), nullable=False, server_default='10000000'))
        op.add_column(table_name, sa.Column('reset_period', sa.String(), nullable=False, server_default='monthly'))

        conn = op.get_bind()
        t = table(
            table_name,
            column(id_column),
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
            t.c[id_column], t.c.daily_limit_enabled, t.c.daily_max_tokens,
            t.c.weekly_limit_enabled, t.c.weekly_max_tokens,
            t.c.monthly_limit_enabled, t.c.monthly_max_tokens,
        )).fetchall()
        for row_id, d_en, d_max, w_en, w_max, m_en, m_max in rows:
            # Collapse back to a single period: prefer whichever is enabled, monthly > weekly > daily.
            if m_en:
                limit_enabled, max_tokens, reset_period = True, m_max, "monthly"
            elif w_en:
                limit_enabled, max_tokens, reset_period = True, w_max, "weekly"
            elif d_en:
                limit_enabled, max_tokens, reset_period = True, d_max, "daily"
            else:
                limit_enabled, max_tokens, reset_period = False, m_max, "monthly"
            conn.execute(
                t.update().where(t.c[id_column] == row_id).values(
                    limit_enabled=limit_enabled, max_tokens=max_tokens, reset_period=reset_period,
                )
            )

        _drop_period_columns(table_name)


def _drop_period_columns(table_name: str):
    for period in ("daily", "weekly", "monthly"):
        op.drop_column(table_name, f'{period}_limit_enabled')
        op.drop_column(table_name, f'{period}_max_tokens')
