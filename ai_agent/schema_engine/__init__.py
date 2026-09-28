"""Enterprise-grade schema retrieval and SQL planning subsystem.

Architecture
------------
User Query → Intent Extraction → Hybrid Schema Retrieval (BM25 + Vector + Graph)
→ Schema Pruning → Join Path Discovery → SQL Planning → SQL Generation
→ SQL Validation → Execution

Modules
-------
- **models**: Normalized metadata model (TableMetadata, ColumnMetadata, …)
- **config**: Central, overridable configuration for every component.
- **extractors**: Metadata extraction from PostgreSQL, MySQL, MCP servers.
- **store**: Persistent metadata + embedding storage.
- **indexing**: BM25 and embedding index builders.
- **retrieval**: Hybrid retrieval engine combining lexical, semantic, and graph.
- **graph**: Schema relationship graph + join-path discovery.
- **summarization**: LLM-driven table/column summarization.
- **planning**: Two-stage reasoning (schema identification → SQL generation).
- **validation**: SQL validation via sqlglot + custom guardrails.
- **tools**: Agent-facing tool wrappers for the runtime.
- **observability**: Metrics, structured logs, tracing hooks.
"""

__version__ = "0.1.0"
