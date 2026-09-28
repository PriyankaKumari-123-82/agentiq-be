"""Optimized schema management for SQL generation.

This module provides:
- Schema caching (Redis) to avoid fetching from DB repeatedly
- Schema summarization to reduce token usage
- Semantic search over schema to find relevant tables
- Schema versioning for cache invalidation
"""
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta

from ai_agent.core.config import settings

log = logging.getLogger(__name__)


@dataclass
class TableInfo:
    """Simplified table schema information."""
    name: str
    columns: List[Dict[str, Any]]
    description: Optional[str] = None
    sample_data: Optional[List[Dict]] = None
    foreign_keys: List[Dict] = None
    primary_key: Optional[str] = None
    
    def to_llm_context(self, include_samples: bool = False) -> str:
        """Convert to LLM-friendly string representation."""
        lines = [f"Table: {self.name}"]
        if self.description:
            lines.append(f"Description: {self.description}")
        
        lines.append("Columns:")
        for col in self.columns:
            col_desc = f"  - {col['name']}: {col.get('type', 'unknown')}"
            if col.get('description'):
                col_desc += f" ({col['description']})"
            if col.get('is_nullable') is False:
                col_desc += " NOT NULL"
            lines.append(col_desc)
        
        if self.primary_key:
            lines.append(f"Primary Key: {self.primary_key}")
        
        if self.foreign_keys:
            lines.append("Foreign Keys:")
            for fk in self.foreign_keys:
                lines.append(f"  - {fk.get('column')} -> {fk.get('references_table')}.{fk.get('references_column')}")
        
        if include_samples and self.sample_data:
            lines.append("Sample Data (first 3 rows):")
            for row in self.sample_data[:3]:
                lines.append(f"  {json.dumps(row)}")

        return "\n".join(lines)

    def to_compact_string(self) -> str:
        """Ultra-compact one-line representation for large schemas."""
        cols = [f"{c['name']}:{c.get('type', '?')}" for c in self.columns[:20]]
        if len(self.columns) > 20:
            cols.append(f"...(+{len(self.columns)-20} more)")
        pk = f" PK:{self.primary_key}" if self.primary_key else ""
        return f"{self.name}({','.join(cols)}){pk}"


@dataclass
class SchemaCacheEntry:
    """Cached schema with metadata."""
    server_name: str
    database_name: str
    tables: Dict[str, TableInfo]
    full_schema_hash: str
    created_at: datetime
    last_accessed: datetime
    version: int = 1
    
    def get_table_names(self) -> List[str]:
        return list(self.tables.keys())
    
    def to_summary(self) -> str:
        """Get a short summary of available tables."""
        lines = ["Available Tables:"]
        for name, info in self.tables.items():
            desc = f"  - {name}"
            if info.description:
                desc += f": {info.description[:60]}"
            lines.append(desc)
        return "\n".join(lines)


class SchemaCacheService:
    """Service for caching and retrieving database schemas."""
    
    def __init__(self):
        self._local_cache: Dict[str, SchemaCacheEntry] = {}
        self._redis_available = False
        self._try_redis()
    
    def _try_redis(self):
        """Attempt to connect to Redis for distributed caching."""
        try:
            import redis
            from ai_agent.core.config import settings
            
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._redis_available = True
            log.info("SchemaCache: Redis connected")
        except Exception as e:
            log.info(f"SchemaCache: Redis not available, using in-memory cache: {e}")
            self._redis = None
    
    def _get_cache_key(self, server_name: str, database_name: str) -> str:
        """Generate cache key for schema."""
        return f"schema:{server_name}:{database_name}"
    
    def _get_redis_key(self, server_name: str, database_name: str) -> str:
        """Generate Redis key."""
        return f"agentiq:schema:{server_name}:{database_name}"
    
    async def get_or_fetch_schema(
        self,
        server_name: str,
        database_name: str,
        fetch_func: callable,
        force_refresh: bool = False,
    ) -> SchemaCacheEntry:
        """Get schema from cache or fetch and cache it.
        
        Args:
            server_name: MCP server name
            database_name: Database name
            fetch_func: Async function to fetch schema from MCP
            force_refresh: Skip cache and re-fetch
            
        Returns:
            SchemaCacheEntry
        """
        cache_key = self._get_cache_key(server_name, database_name)
        
        if not force_refresh:
            # Try memory cache first
            if cache_key in self._local_cache:
                entry = self._local_cache[cache_key]
                # Check if cache is still fresh (1 hour for schema)
                if datetime.utcnow() - entry.created_at < timedelta(hours=1):
                    entry.last_accessed = datetime.utcnow()
                    log.info(f"SchemaCache: memory cache hit for {cache_key}")
                    return entry
            
            # Try Redis
            if self._redis_available:
                try:
                    redis_key = self._get_redis_key(server_name, database_name)
                    cached_data = self._redis.get(redis_key)
                    if cached_data:
                        data = json.loads(cached_data)
                        entry = self._deserialize_entry(data)
                        self._local_cache[cache_key] = entry
                        log.info(f"SchemaCache: Redis cache hit for {cache_key}")
                        return entry
                except Exception as e:
                    log.warning(f"SchemaCache: Redis read failed: {e}")
        
        # Fetch fresh schema
        log.info(f"SchemaCache: fetching fresh schema for {cache_key}")
        raw_schema = await fetch_func()
        entry = self._parse_schema(raw_schema, server_name, database_name)
        
        # Cache it
        self._local_cache[cache_key] = entry
        if self._redis_available:
            try:
                redis_key = self._get_redis_key(server_name, database_name)
                self._redis.setex(
                    redis_key,
                    timedelta(hours=2),  # Redis TTL
                    json.dumps(self._serialize_entry(entry)),
                )
            except Exception as e:
                log.warning(f"SchemaCache: Redis write failed: {e}")
        
        return entry
    
    def invalidate_cache(self, server_name: str, database_name: str):
        """Invalidate cached schema for a database."""
        cache_key = self._get_cache_key(server_name, database_name)
        self._local_cache.pop(cache_key, None)
        
        if self._redis_available:
            try:
                redis_key = self._get_redis_key(server_name, database_name)
                self._redis.delete(redis_key)
            except Exception as e:
                log.warning(f"SchemaCache: Redis delete failed: {e}")
    
    def _parse_schema(
        self,
        raw_schema: Any,
        server_name: str,
        database_name: str,
    ) -> SchemaCacheEntry:
        """Parse raw schema into structured format."""
        tables: Dict[str, TableInfo] = {}
        
        # Handle different schema formats
        if isinstance(raw_schema, dict):
            # Try to extract tables from various formats
            if "tables" in raw_schema:
                table_list = raw_schema["tables"]
            elif "schema" in raw_schema:
                table_list = raw_schema["schema"]
            else:
                # Assume keys are table names
                table_list = [
                    {"name": k, **v} for k, v in raw_schema.items()
                    if isinstance(v, (dict, list))
                ]
        elif isinstance(raw_schema, list):
            table_list = raw_schema
        else:
            # Try to parse as JSON string
            try:
                if isinstance(raw_schema, str):
                    parsed = json.loads(raw_schema)
                    return self._parse_schema(parsed, server_name, database_name)
            except json.JSONDecodeError:
                pass
            table_list = []
        
        for table_data in table_list:
            if isinstance(table_data, dict):
                name = table_data.get("name") or table_data.get("table_name")
                if not name:
                    continue
                
                columns = table_data.get("columns", [])
                if not isinstance(columns, list):
                    columns = []
                
                tables[name] = TableInfo(
                    name=name,
                    columns=columns,
                    description=table_data.get("description"),
                    primary_key=table_data.get("primary_key"),
                    foreign_keys=table_data.get("foreign_keys", []),
                    sample_data=table_data.get("sample_data"),
                )
        
        # Generate hash for versioning
        schema_str = json.dumps(raw_schema, sort_keys=True)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]
        
        return SchemaCacheEntry(
            server_name=server_name,
            database_name=database_name,
            tables=tables,
            full_schema_hash=schema_hash,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            version=1,
        )
    
    def _serialize_entry(self, entry: SchemaCacheEntry) -> Dict:
        """Serialize cache entry for Redis."""
        return {
            "server_name": entry.server_name,
            "database_name": entry.database_name,
            "tables": {
                name: {
                    "name": t.name,
                    "columns": t.columns,
                    "description": t.description,
                    "primary_key": t.primary_key,
                    "foreign_keys": t.foreign_keys,
                }
                for name, t in entry.tables.items()
            },
            "full_schema_hash": entry.full_schema_hash,
            "created_at": entry.created_at.isoformat(),
            "last_accessed": entry.last_accessed.isoformat(),
            "version": entry.version,
        }
    
    def _deserialize_entry(self, data: Dict) -> SchemaCacheEntry:
        """Deserialize cache entry from Redis."""
        tables = {
            name: TableInfo(
                name=t["name"],
                columns=t["columns"],
                description=t.get("description"),
                primary_key=t.get("primary_key"),
                foreign_keys=t.get("foreign_keys", []),
            )
            for name, t in data.get("tables", {}).items()
        }
        
        return SchemaCacheEntry(
            server_name=data["server_name"],
            database_name=data["database_name"],
            tables=tables,
            full_schema_hash=data["full_schema_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            version=data.get("version", 1),
        )


class SchemaSelector:
    """Intelligently select relevant tables from schema."""
    
    # Common table keywords mapping
    KEYWORD_MAPPINGS = {
        "user": ["users", "user", "accounts", "customers", "clients", "people", "persons"],
        "customer": ["customers", "clients", "users", "accounts"],
        "order": ["orders", "purchases", "transactions", "sales"],
        "product": ["products", "items", "goods", "inventory", "skus"],
        "payment": ["payments", "transactions", "invoices", "billing"],
        "sale": ["sales", "orders", "transactions", "revenue"],
        "employee": ["employees", "staff", "personnel", "users"],
        "category": ["categories", "types", "groups", "classes"],
        "location": ["locations", "addresses", "stores", "branches", "regions"],
        "date": ["dates", "created_at", "updated_at", "timestamp", "datetime"],
        "price": ["price", "cost", "amount", "total", "subtotal", "value"],
        "quantity": ["quantity", "qty", "count", "amount", "number"],
        "email": ["email", "email_address", "mail"],
        "phone": ["phone", "phone_number", "mobile", "contact_number"],
    }
    
    def __init__(self, schema_cache: SchemaCacheService):
        self.cache = schema_cache
    
    def select_relevant_tables(
        self,
        query: str,
        schema: SchemaCacheEntry,
        max_tables: int = 5,
    ) -> List[TableInfo]:
        """Select tables likely relevant to the query.
        
        Uses keyword matching and simple scoring. For production,
        this could use embeddings and semantic search.
        
        Args:
            query: User's natural language query
            schema: Full schema cache entry
            max_tables: Maximum tables to return
            
        Returns:
            List of relevant TableInfo objects
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Expand query words with keyword mappings
        expanded_keywords = set(query_words)
        for word in query_words:
            for concept, related in self.KEYWORD_MAPPINGS.items():
                if concept in word or word in concept:
                    expanded_keywords.update(related)
        
        # Score each table
        scored_tables: List[tuple[float, TableInfo]] = []
        
        for name, table in schema.tables.items():
            score = 0.0
            name_lower = name.lower()
            
            # Exact name match
            for word in query_words:
                if word in name_lower:
                    score += 10.0
                if name_lower in word:
                    score += 5.0
            
            # Keyword mapping match
            for keyword in expanded_keywords:
                if keyword in name_lower:
                    score += 3.0
            
            # Column name matches
            for col in table.columns:
                col_name = col.get("name", "").lower()
                for word in expanded_keywords:
                    if word in col_name:
                        score += 2.0
                    # Common join columns bonus
                    if col_name in ["id", "user_id", "order_id", "product_id"]:
                        if any(w in query_lower for w in ["join", "related", "with"]):
                            score += 1.0
            
            # Description match
            if table.description:
                desc_lower = table.description.lower()
                for word in query_words:
                    if word in desc_lower:
                        score += 3.0
            
            # Foreign key bonus (likely needed for joins)
            if table.foreign_keys:
                score += 0.5
            
            scored_tables.append((score, table))
        
        # Sort by score descending
        scored_tables.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N, filtering out zero scores unless we have nothing
        relevant = [t for s, t in scored_tables if s > 0][:max_tables]
        
        # If no relevant tables found, return the most central tables
        # (those with many foreign keys - likely fact tables)
        if not relevant:
            tables_with_fks = [
                (len(t.foreign_keys or []) + len(t.columns) / 10, t)
                for t in schema.tables.values()
            ]
            tables_with_fks.sort(reverse=True)
            relevant = [t for _, t in tables_with_fks[:max_tables]]
        
        log.info(f"SchemaSelector: selected {len(relevant)} tables for query: {[t.name for t in relevant]}")
        return relevant
    
    def build_context_for_query(
        self,
        query: str,
        schema: SchemaCacheEntry,
        max_tokens: int = 4000,
        include_samples: bool = False,
    ) -> str:
        """Build optimized schema context for LLM.

        Selects relevant tables and formats them concisely, enforcing max_tokens.

        Args:
            query: User's query
            schema: Full schema
            max_tokens: Approximate token limit for context (enforced via char count)
            include_samples: Include sample data (increases tokens)

        Returns:
            Formatted schema context string truncated to fit max_tokens
        """
        # Approximate 4 chars per token for character-based budget
        max_chars = max_tokens * 4

        # Select relevant tables (get more than we need, then truncate)
        relevant_tables = self.select_relevant_tables(query, schema, max_tables=10)

        # Build context iteratively to stay within budget
        parts = [
            f"Database: {schema.database_name}",
            f"Schema Version: {schema.full_schema_hash}",
            "",
            f"Selected tables from {len(schema.tables)} total:",
            "",
        ]

        current_chars = len("\n".join(parts))

        # Add tables one by one until we hit budget
        added_tables = 0
        for table in relevant_tables:
            # Try compact format first (much smaller)
            table_context = table.to_compact_string()
            table_chars = len(table_context) + 2  # +2 for newline

            if current_chars + table_chars > max_chars:
                # Switch to ultra-compact if still over budget
                if added_tables == 0:
                    # At least show table names in list format
                    table_names = ", ".join([t.name for t in relevant_tables[:5]])
                    parts.append(f"Tables: {table_names}...")
                    break
                break

            parts.append(table_context)
            current_chars += table_chars
            added_tables += 1

        # Add relationship hints if space allows
        if added_tables > 1:
            relationships = self._extract_relationships(relevant_tables[:added_tables])
            if relationships:
                rel_str = "\n".join(relationships)
                if current_chars + len(rel_str) + 20 < max_chars:
                    parts.append("")
                    parts.append("Relationships:")
                    parts.extend(relationships)

        result = "\n".join(parts)
        log.info(f"SchemaSelector: built context {len(result)} chars (~{len(result)//4} tokens), {added_tables} tables")
        return result
    
    def _extract_relationships(self, tables: List[TableInfo]) -> List[str]:
        """Extract relationship hints between selected tables."""
        relationships = []
        table_names = {t.name for t in tables}
        
        for table in tables:
            if table.foreign_keys:
                for fk in table.foreign_keys:
                    ref_table = fk.get("references_table")
                    if ref_table in table_names:
                        rel = f"  - {table.name}.{fk.get('column')} -> {ref_table}.{fk.get('references_column')}"
                        relationships.append(rel)
        
        return relationships


# Global instances
_schema_cache: Optional[SchemaCacheService] = None
_schema_selector: Optional[SchemaSelector] = None


def get_schema_cache() -> SchemaCacheService:
    """Get or create schema cache service."""
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = SchemaCacheService()
    return _schema_cache


def get_schema_selector() -> SchemaSelector:
    """Get or create schema selector."""
    global _schema_selector
    if _schema_selector is None:
        _schema_selector = SchemaSelector(get_schema_cache())
    return _schema_selector
