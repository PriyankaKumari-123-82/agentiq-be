"""Service layer for the schema classifier: table categorization + sample
questions. Called from cold build (``schema_engine/orchestrator.py``) and
from the manual ``POST /agents/{agent_id}/classified_schema`` endpoint
(``api/v1/mcp.py``) — both funnel through this one implementation.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from ai_agent.models import MCPServerConfig
from ai_agent.schema_engine.models import TableCategory, DatabaseMetadata, SchemaChunk
from ai_agent.schema_engine.store.pg_models import SEDatabase
from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository

log = logging.getLogger(__name__)


class SchemaClassifierService:
    """Table categorization + sample questions for a set of MCP servers."""

    def __init__(self, servers: Optional[List[MCPServerConfig]] = None):
        self._servers = servers or []

    def get_schema(self, repo: SchemaEngineRepository) -> Dict[str, SEDatabase]:
        """Resolve each server's ``se_databases`` row, keyed by server_name.

        Servers whose schema hasn't been extracted yet (no ``se_databases``
        row) are skipped.
        """
        result: Dict[str, SEDatabase] = {}
        for srv in self._servers:
            db_row = repo.get_database(srv.name)
            if db_row:
                result[srv.name] = db_row
        return result

    def get_classified_schema(self, repo: SchemaEngineRepository) -> List[dict]:
        """Already-persisted categories + sample questions across all servers,
        sorted by priority. Used by ``GET /agents/{agent_id}/classified_schema``.
        """
        categories_out = []
        for server_name, db_row in self.get_schema(repo).items():
            for cat in repo.get_classifier_categories_for_database(db_row.id):
                categories_out.append({
                    "id": cat.id,
                    "server_name": server_name,
                    "category_name": cat.category_name,
                    "description": cat.description,
                    "priority": cat.priority,
                    "table_names": json.loads(cat.table_names_json or "[]"),
                    "questions": json.loads(cat.questions or "[]"),
                })

        categories_out.sort(key=lambda c: c["priority"])
        return categories_out

    async def save_classified_schema(
        self,
        repo: SchemaEngineRepository,
        server_name: str,
        db_meta: DatabaseMetadata,
        chunks: List[SchemaChunk],
        *,
        api_key: Optional[str] = None,
    ) -> List[TableCategory]:
        """Run the schema classifier and persist the result for one server.

        Used by both cold build and ``POST /agents/{agent_id}/classified_schema``.
        Does not swallow exceptions — callers decide how to handle failures
        (cold build treats this as best-effort; the manual endpoint lets
        failures surface to the caller).
        """
        from ai_agent.workflows.schema_classifier.schema_classifier import run as run_schema_classifier

        classifier_engine = run_schema_classifier()
        categories = await classifier_engine.build(db_meta, chunks, api_key=api_key)

        # Drop categories question generation failed for — a category with no
        # sample questions isn't useful to persist/surface.
        skipped = [c.category_name for c in categories if not c.questions]
        categories = [c for c in categories if c.questions]
        if skipped:
            log.warning(
                "SchemaClassifierService: skipping %d categories with no questions for '%s': %s",
                len(skipped), server_name, ", ".join(skipped),
            )

        db_row = repo.get_database(server_name)
        if not db_row:
            return []

        repo.replace_classifier_categories(db_row.id, [
            {
                "category_name": c.category_name,
                "description": c.description,
                "table_names": c.table_names,
                "questions": c.questions,
                "priority": c.priority,
            }
            for c in categories
        ])
        return categories
