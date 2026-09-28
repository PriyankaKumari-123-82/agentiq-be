"""Agent-facing tools exposed to the LangGraph runtime.

Each function is designed to be called from within an agent node as part
of the tool-selection loop.  They are thin wrappers around the schema
engine internals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ai_agent.schema_engine.models import RetrievalResult, SQLPlanResult, SQLValidationResult

log = logging.getLogger(__name__)


async def retrieve_schema_context(
    query: str,
    server_name: str,
    *,
    orchestrator: Any = None,
) -> Dict[str, Any]:
    """Retrieve relevant schema context for a natural language query.

    Returns::

        {
            "domains": [...],
            "tables": [...],
            "relationships": [...],
            "schema_context": "compact DDL string",
            "confidence": 0.91,
            "token_estimate": 1234,
        }
    """
    if orchestrator is None:
        raise ValueError("orchestrator is required")

    result: RetrievalResult = await orchestrator.retrieve(query)

    return {
        "domains": result.domains,
        "tables": result.table_names,
        "relationships": result.relationships,
        "schema_context": "\n".join(c.compact_ddl for c in result.chunks),
        "confidence": result.confidence,
        "token_estimate": result.token_estimate,
    }


async def discover_join_paths(
    source_table: str,
    target_table: str,
    *,
    orchestrator: Any = None,
) -> Dict[str, Any]:
    """Find FK join paths between two tables."""
    if orchestrator is None:
        raise ValueError("orchestrator is required")

    graph = orchestrator._graph
    if graph is None:
        return {"paths": [], "error": "Graph not available"}

    paths = graph.find_join_paths(source_table, target_table)
    return {
        "paths": [
            {
                "hops": p.hops,
                "tables": p.tables,
                "joins": p.to_sql_joins(),
            }
            for p in paths
        ],
    }


async def generate_sql_plan(
    query: str,
    server_name: str,
    *,
    orchestrator: Any = None,
) -> Dict[str, Any]:
    """Run the full two-stage pipeline: retrieve → identify → generate SQL.

    Returns the SQL plan including the query, validation status, and
    any warnings.
    """
    if orchestrator is None:
        raise ValueError("orchestrator is required")

    plan: SQLPlanResult = await orchestrator.plan_sql(query)
    return {
        "sql": plan.sql,
        "is_valid": plan.is_valid,
        "tables_used": plan.tables_used,
        "validation_errors": plan.validation_errors,
        "explanation": plan.explanation,
    }


async def validate_sql(
    sql: str,
    *,
    orchestrator: Any = None,
) -> Dict[str, Any]:
    """Validate SQL without executing it."""
    if orchestrator is None:
        raise ValueError("orchestrator is required")

    result: SQLValidationResult = orchestrator.validate(sql)
    return {
        "is_valid": result.is_valid,
        "cleaned_sql": result.cleaned_sql,
        "errors": result.errors,
        "warnings": result.warnings,
        "tables_referenced": result.tables_referenced,
        "has_limit": result.has_limit,
        "complexity": result.estimated_complexity,
    }
