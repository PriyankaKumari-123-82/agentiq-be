"""Persistent metadata and embedding storage.

Storage layers:
  - pg_models.py     : SQLModel table definitions for PostgreSQL
  - pg_repository.py : CRUD repository over PG models
  - vector_store.py  : Milvus embedding persistence
  - redis_cache.py   : Redis distributed cache + locks
  - metadata_store.py: Tiered store (PG → Redis → Memory)
  - query_log.py     : Durable query-learning log
"""
