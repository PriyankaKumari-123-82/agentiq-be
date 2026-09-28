# Schema Engine — Architecture

## Overview

The schema engine is an enterprise-grade subsystem that enables LLM-based agents
to answer natural-language questions against very large relational databases
**without sending full schemas to the LLM**.

## Pipeline

```
User Query
    │
    ▼
┌──────────────────────────┐
│  Metadata Extraction     │  MCP / PostgreSQL / MySQL
│  (extractors/)           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Schema Summarization    │  LLM-generated table summaries, domains, keywords
│  (summarization/)        │  (cached per content-hash)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Indexing                │  BM25 index + embedding vectors
│  (indexing/)             │  One SchemaChunk per table
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Hybrid Retrieval        │  BM25 (lexical) + Vector (semantic) + Graph (FK)
│  (retrieval/)            │  Fused scoring with configurable weights
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Schema Pruning          │  Token-budget enforcement
│  (planning/              │  Audit column compression
│   token_optimizer)       │  Top-k column filtering
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Stage 1: Identification │  LLM identifies tables, columns, joins
│  (planning/sql_planner)  │  NO SQL generated yet
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Join Path Discovery     │  BFS shortest-path on FK graph
│  (graph/schema_graph)    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Stage 2: SQL Generation │  LLM generates SQL using only pruned schema
│  (planning/sql_planner)  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  SQL Validation          │  Read-only check, injection detection,
│  (validation/)           │  table/column existence, LIMIT injection,
│                          │  sqlglot structural check
└────────────┬─────────────┘
             │
             ▼
         Execution
```

## Package Layout

```
ai_agent/schema_engine/
├── __init__.py              # Package docstring
├── config.py                # SchemaEngineConfig (SE_* env vars)
├── models.py                # Internal data models (not ORM)
├── orchestrator.py          # Facade wiring all components
│
├── extractors/
│   ├── base.py              # Abstract BaseMetadataExtractor
│   ├── mcp_extractor.py     # Via MCP server schema tool
│   ├── postgresql.py        # Direct asyncpg connection
│   └── mysql.py             # Direct aiomysql connection
│
├── summarization/
│   └── schema_summarizer.py # LLM batch summarization
│
├── indexing/
│   └── schema_indexer.py    # Chunk builder + BM25 + embeddings
│
├── graph/
│   └── schema_graph.py      # FK graph, BFS, join-path discovery
│
├── retrieval/
│   ├── hybrid_retriever.py  # BM25 + vector + graph fusion
│   └── domain_classifier.py # Hierarchical domain filtering
│
├── planning/
│   ├── sql_planner.py       # Two-stage reasoning pipeline
│   └── token_optimizer.py   # Token budget management
│
├── validation/
│   └── sql_validator.py     # Pre-execution SQL validation
│
├── store/
│   ├── metadata_store.py    # Redis + memory cache
│   └── query_log.py         # Query-log learning
│
├── observability/
│   └── metrics.py           # Structured metrics & tracing
│
├── tools/
│   └── agent_tools.py       # Agent-facing tool wrappers
│
└── tests/
    ├── synthetic_schema.py  # 50-table enterprise schema
    └── test_schema_engine.py # 29 unit tests
```

## Key Design Decisions

1. **Table-level chunking** — each chunk = one table with all metadata.
   No arbitrary token-based splitting.

2. **Hybrid retrieval** — BM25 (exact/acronym matches) + vector (semantic
   similarity) + graph (FK expansion).  Weights configurable.

3. **Two-stage reasoning** — Stage 1 identifies schema elements without SQL.
   Stage 2 generates SQL from only validated tables/columns.

4. **Token-budget enforcement at every layer** — indexer, retriever,
   optimizer, and runtime all enforce hard caps.

5. **Pluggable extractors** — MCP, PostgreSQL, MySQL.  Oracle/SQL Server
   ready via the abstract base class.

6. **No full schema ever reaches the LLM** — even the orchestrator's
   `build_context()` has a hard character cap.

## Configuration

All knobs are in `SchemaEngineConfig` and readable from `SE_*` environment
variables.  Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `SE_RETRIEVAL_TOP_K` | 15 | Max tables from retrieval |
| `SE_SCHEMA_CONTEXT_MAX_TOKENS` | 3000 | Token budget for LLM context |
| `SE_GRAPH_MAX_HOPS` | 2 | Max FK hops for join discovery |
| `SE_GRAPH_AUTO_EXPAND_HOPS` | 1 | Auto-expand neighbours |
| `SE_BM25_WEIGHT` | 0.35 | BM25 score weight |
| `SE_VECTOR_WEIGHT` | 0.45 | Vector score weight |
| `SE_GRAPH_WEIGHT` | 0.20 | Graph score weight |
| `SE_EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |
| `SE_SQL_MAX_ROWS` | 10000 | Default LIMIT |
| `SE_SQL_ALLOW_WRITE` | false | Read-only enforcement |

## Runtime Integration

`agent_runtime.py` calls the orchestrator in `_load_db_schema_context`:

```python
from ai_agent.schema_engine.orchestrator import get_orchestrator

orch = await get_orchestrator(server_name, mcp_manager, server_tools)
ctx = await orch.build_context(user_query)
# ctx is a token-budgeted compact schema string
state.schema_context = ctx
```

## Testing

```bash
pytest ai_agent/schema_engine/tests/ -v
```

29 tests cover models, BM25, graph, domain classifier, SQL validator,
and hybrid retrieval — all without LLM or Redis dependencies.
