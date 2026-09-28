"""MySQL metadata extractor via direct connection.

Queries ``information_schema`` for tables, columns, keys, and statistics.

Usage::

    extractor = MySQLExtractor(host="localhost", port=3306, user="root",
                               password="secret", database="mydb")
    meta = await extractor.extract("my-mysql-server")
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
    TableMetadata,
)

log = logging.getLogger(__name__)

_TABLES_SQL = """
SELECT
    TABLE_SCHEMA,
    TABLE_NAME,
    TABLE_COMMENT,
    TABLE_ROWS
FROM information_schema.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
  AND TABLE_SCHEMA NOT IN ({excluded})
ORDER BY TABLE_SCHEMA, TABLE_NAME;
"""

_COLUMNS_SQL = """
SELECT
    TABLE_SCHEMA,
    TABLE_NAME,
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE,
    COLUMN_DEFAULT,
    COLUMN_KEY,
    COLUMN_COMMENT
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA NOT IN ({excluded})
ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION;
"""

_FK_SQL = """
SELECT
    kcu.TABLE_SCHEMA,
    kcu.TABLE_NAME,
    kcu.COLUMN_NAME AS source_column,
    kcu.REFERENCED_TABLE_NAME AS target_table,
    kcu.REFERENCED_COLUMN_NAME AS target_column,
    kcu.REFERENCED_TABLE_SCHEMA AS target_schema
FROM information_schema.KEY_COLUMN_USAGE kcu
WHERE kcu.REFERENCED_TABLE_NAME IS NOT NULL
  AND kcu.TABLE_SCHEMA NOT IN ({excluded});
"""


class MySQLExtractor(BaseMetadataExtractor):
    """Direct MySQL metadata extractor using aiomysql."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "",
    ):
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
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
            import aiomysql  # noqa: F401
        except ImportError:
            raise ImportError("aiomysql is required for MySQL extraction: pip install aiomysql")

        conn = await aiomysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            db=self._database or database_name,
            charset="utf8mb4",
        )
        try:
            excluded = ", ".join(f"'{s}'" for s in self._cfg.excluded_schemas)
            excluded_cols = {c.lower() for c in self._cfg.excluded_columns}

            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Fetch tables
                await cur.execute(_TABLES_SQL.format(excluded=excluded))
                raw_tables = await cur.fetchall()

                # Fetch columns
                await cur.execute(_COLUMNS_SQL.format(excluded=excluded))
                raw_columns = await cur.fetchall()

                # Fetch FKs
                await cur.execute(_FK_SQL.format(excluded=excluded))
                raw_fks = await cur.fetchall()

            # Build lookup maps
            col_map: Dict[str, List[ColumnMetadata]] = {}
            pk_map: Dict[str, List[str]] = {}

            for row in raw_columns:
                key = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}"
                cname = row["COLUMN_NAME"]
                if cname.lower() in excluded_cols:
                    continue
                is_pk = row.get("COLUMN_KEY") == "PRI"
                col = ColumnMetadata(
                    name=cname,
                    data_type=row["DATA_TYPE"],
                    nullable=row["IS_NULLABLE"] == "YES",
                    description=row.get("COLUMN_COMMENT") or None,
                    default_value=row.get("COLUMN_DEFAULT"),
                    is_primary_key=is_pk,
                )
                col_map.setdefault(key, []).append(col)
                if is_pk:
                    pk_map.setdefault(key, []).append(cname)

            fk_map: Dict[str, List[ForeignKeyMetadata]] = {}
            for row in raw_fks:
                key = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}"
                fk_map.setdefault(key, []).append(ForeignKeyMetadata(
                    source_column=row["source_column"],
                    target_table=row["target_table"],
                    target_column=row["target_column"],
                    target_schema=row.get("target_schema"),
                ))

            # Assemble
            tables: Dict[str, TableMetadata] = {}
            for row in raw_tables:
                schema = row["TABLE_SCHEMA"]
                tname = row["TABLE_NAME"]
                fqn = f"{schema}.{tname}"
                cols = col_map.get(fqn, [])
                pks = pk_map.get(fqn, [])
                fks = fk_map.get(fqn, [])

                fk_src_set = {fk.source_column for fk in fks}
                for c in cols:
                    if c.name in fk_src_set:
                        c.is_foreign_key = True

                tables[fqn] = TableMetadata(
                    schema_name=schema,
                    table_name=tname,
                    columns=cols,
                    primary_keys=pks,
                    foreign_keys=fks,
                    row_count=row.get("TABLE_ROWS"),
                    table_comment=row.get("TABLE_COMMENT") or None,
                    description=row.get("TABLE_COMMENT") or None,
                )

            log.info("MySQLExtractor: extracted %d tables", len(tables))
            return DatabaseMetadata(
                server_name=server_name,
                database_name=database_name or self._database,
                tables=tables,
            )
        finally:
            conn.close()

    async def health_check(self) -> bool:
        try:
            import aiomysql
            conn = await aiomysql.connect(
                host=self._host, port=self._port,
                user=self._user, password=self._password,
            )
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
            conn.close()
            return True
        except Exception:
            return False
