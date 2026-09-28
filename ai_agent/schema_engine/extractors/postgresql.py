"""PostgreSQL metadata extractor via direct connection.

This extractor connects directly to a PostgreSQL instance using
``asyncpg`` and queries ``information_schema`` + ``pg_catalog`` for
comprehensive metadata including comments, row counts, and sample values.

Usage::

    extractor = PostgreSQLExtractor(dsn="postgresql://user:pass@host:5432/db")
    meta = await extractor.extract("my-pg-server")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.extractors.base import BaseMetadataExtractor
from ai_agent.schema_engine.models import (
    ColumnMetadata,
    DatabaseMetadata,
    ForeignKeyMetadata,
    IndexMetadata,
    TableMetadata,
)

log = logging.getLogger(__name__)

_TABLES_SQL = """
SELECT
    t.table_schema,
    t.table_name,
    obj_description(pgc.oid) AS table_comment,
    pgc.reltuples::bigint AS row_count
FROM information_schema.tables t
LEFT JOIN pg_catalog.pg_class pgc
    ON pgc.relname = t.table_name
    AND pgc.relnamespace = (
        SELECT oid FROM pg_catalog.pg_namespace WHERE nspname = t.table_schema
    )
WHERE t.table_type = 'BASE TABLE'
  AND t.table_schema NOT IN ({excluded})
ORDER BY t.table_schema, t.table_name;
"""

_COLUMNS_SQL = """
SELECT
    c.table_schema,
    c.table_name,
    c.column_name,
    c.data_type,
    c.is_nullable,
    c.column_default,
    col_description(
        (SELECT oid FROM pg_catalog.pg_class WHERE relname = c.table_name
         AND relnamespace = (SELECT oid FROM pg_catalog.pg_namespace WHERE nspname = c.table_schema)),
        c.ordinal_position
    ) AS column_comment
FROM information_schema.columns c
WHERE c.table_schema NOT IN ({excluded})
ORDER BY c.table_schema, c.table_name, c.ordinal_position;
"""

_PK_SQL = """
SELECT
    tc.table_schema,
    tc.table_name,
    kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY'
  AND tc.table_schema NOT IN ({excluded});
"""

_FK_SQL = """
SELECT
    tc.table_schema,
    tc.table_name,
    kcu.column_name AS source_column,
    ccu.table_name AS target_table,
    ccu.column_name AS target_column,
    ccu.table_schema AS target_schema
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
  AND tc.table_schema NOT IN ({excluded});
"""


class PostgreSQLExtractor(BaseMetadataExtractor):
    """Direct PostgreSQL metadata extractor using asyncpg."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._cfg = get_config()

    async def extract(
        self,
        server_name: str,
        database_name: str = "default",
        *,
        include_sample_values: bool = True,
        sample_limit: int = 5,
    ) -> DatabaseMetadata:
        try:
            import asyncpg  # noqa: F401
        except ImportError:
            raise ImportError("asyncpg is required for PostgreSQL extraction: pip install asyncpg")

        conn = await asyncpg.connect(self._dsn)
        try:
            excluded = ", ".join(
                f"'{s}'" for s in self._cfg.excluded_schemas
            )
            excluded_cols = {c.lower() for c in self._cfg.excluded_columns}

            # Fetch tables
            raw_tables = await conn.fetch(_TABLES_SQL.format(excluded=excluded))
            # Fetch columns
            raw_columns = await conn.fetch(_COLUMNS_SQL.format(excluded=excluded))
            # Fetch PKs
            raw_pks = await conn.fetch(_PK_SQL.format(excluded=excluded))
            # Fetch FKs
            raw_fks = await conn.fetch(_FK_SQL.format(excluded=excluded))

            # Build lookup maps
            col_map: Dict[str, List[ColumnMetadata]] = {}
            for row in raw_columns:
                key = f"{row['table_schema']}.{row['table_name']}"
                if row["column_name"].lower() in excluded_cols:
                    continue
                col = ColumnMetadata(
                    name=row["column_name"],
                    data_type=row["data_type"],
                    nullable=row["is_nullable"] == "YES",
                    description=row.get("column_comment"),
                    default_value=row.get("column_default"),
                )
                col_map.setdefault(key, []).append(col)

            pk_map: Dict[str, List[str]] = {}
            for row in raw_pks:
                key = f"{row['table_schema']}.{row['table_name']}"
                pk_map.setdefault(key, []).append(row["column_name"])

            fk_map: Dict[str, List[ForeignKeyMetadata]] = {}
            for row in raw_fks:
                key = f"{row['table_schema']}.{row['table_name']}"
                fk_map.setdefault(key, []).append(ForeignKeyMetadata(
                    source_column=row["source_column"],
                    target_table=row["target_table"],
                    target_column=row["target_column"],
                    target_schema=row.get("target_schema"),
                ))

            # Assemble tables
            tables: Dict[str, TableMetadata] = {}
            for row in raw_tables:
                fqn = f"{row['table_schema']}.{row['table_name']}"
                cols = col_map.get(fqn, [])
                pks = pk_map.get(fqn, [])
                fks = fk_map.get(fqn, [])

                # Mark PK/FK on columns
                pk_set = set(pks)
                fk_src_set = {fk.source_column for fk in fks}
                for c in cols:
                    c.is_primary_key = c.name in pk_set
                    c.is_foreign_key = c.name in fk_src_set

                table = TableMetadata(
                    schema_name=row["table_schema"],
                    table_name=row["table_name"],
                    columns=cols,
                    primary_keys=pks,
                    foreign_keys=fks,
                    row_count=row.get("row_count"),
                    table_comment=row.get("table_comment"),
                    description=row.get("table_comment"),
                )

                # Sample values (best-effort)
                if include_sample_values and cols:
                    try:
                        sample_cols = [c.name for c in cols[:5]]
                        sample_sql = (
                            f"SELECT DISTINCT {', '.join(sample_cols)} "
                            f"FROM {fqn} LIMIT {sample_limit}"
                        )
                        samples = await conn.fetch(sample_sql)
                        table.sample_values = {
                            col: [str(r.get(col, "")) for r in samples]
                            for col in sample_cols
                        }
                    except Exception:
                        pass

                tables[fqn] = table

            log.info("PostgreSQLExtractor: extracted %d tables", len(tables))
            return DatabaseMetadata(
                server_name=server_name,
                database_name=database_name,
                tables=tables,
            )
        finally:
            await conn.close()

    async def health_check(self) -> bool:
        try:
            import asyncpg
            conn = await asyncpg.connect(self._dsn)
            await conn.execute("SELECT 1")
            await conn.close()
            return True
        except Exception:
            return False
