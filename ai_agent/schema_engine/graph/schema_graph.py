"""FK-based schema graph for join-path discovery and neighbour expansion.

The graph is built once from ``DatabaseMetadata`` and supports:
- Shortest-path join discovery between two tables (BFS).
- N-hop neighbour expansion for retrieved tables.
- Scored graph-expansion results for hybrid ranking.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.models import DatabaseMetadata, ForeignKeyMetadata

log = logging.getLogger(__name__)


@dataclass
class JoinEdge:
    """A single join between two tables."""
    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclass
class JoinPath:
    """An ordered sequence of joins connecting two tables."""
    edges: List[JoinEdge] = field(default_factory=list)

    @property
    def hops(self) -> int:
        return len(self.edges)

    @property
    def tables(self) -> List[str]:
        seen: List[str] = []
        for e in self.edges:
            if e.source_table not in seen:
                seen.append(e.source_table)
            if e.target_table not in seen:
                seen.append(e.target_table)
        return seen

    def to_sql_joins(self) -> List[str]:
        return [
            f"{e.source_table}.{e.source_column} = {e.target_table}.{e.target_column}"
            for e in self.edges
        ]


class SchemaGraph:
    """Bidirectional FK graph over a ``DatabaseMetadata`` instance."""

    def __init__(self, db: DatabaseMetadata):
        self._cfg = get_config()
        self._db = db
        # adjacency: table_name -> [(neighbour_table, JoinEdge)]
        self._adj: Dict[str, List[Tuple[str, JoinEdge]]] = defaultdict(list)
        self._build(db)

    def _build(self, db: DatabaseMetadata) -> None:
        for table in db.tables.values():
            src = table.table_name
            for fk in table.foreign_keys:
                tgt = fk.target_table
                edge = JoinEdge(
                    source_table=src,
                    source_column=fk.source_column,
                    target_table=tgt,
                    target_column=fk.target_column,
                )
                self._adj[src].append((tgt, edge))
                # Reverse edge for bidirectional traversal
                rev = JoinEdge(
                    source_table=tgt,
                    source_column=fk.target_column,
                    target_table=src,
                    target_column=fk.source_column,
                )
                self._adj[tgt].append((src, rev))

        log.info(
            "SchemaGraph: %d nodes, %d edges",
            len(self._adj), sum(len(v) for v in self._adj.values()) // 2,
        )

    # ── Neighbour expansion ──────────────────────────────────────────────

    def expand_neighbours(
        self, table_names: List[str], max_hops: int = None,
    ) -> Set[str]:
        """Return tables reachable within *max_hops* FK hops."""
        if max_hops is None:
            max_hops = self._cfg.graph_auto_expand_hops

        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque()
        for t in table_names:
            queue.append((t, 0))
            visited.add(t)

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbour, _ in self._adj.get(current, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, depth + 1))

        return visited

    # ── Shortest join path ───────────────────────────────────────────────

    def shortest_path(self, source: str, target: str) -> Optional[JoinPath]:
        """BFS shortest path between two table names."""
        if source == target:
            return JoinPath()
        if source not in self._adj or target not in self._adj:
            return None

        visited: Set[str] = {source}
        queue: deque[Tuple[str, List[JoinEdge]]] = deque([(source, [])])

        while queue:
            current, path = queue.popleft()
            for neighbour, edge in self._adj.get(current, []):
                if neighbour in visited:
                    continue
                new_path = path + [edge]
                if neighbour == target:
                    return JoinPath(edges=new_path)
                visited.add(neighbour)
                queue.append((neighbour, new_path))

        return None  # No path exists

    # ── All paths up to N hops ───────────────────────────────────────────

    def find_join_paths(
        self, source: str, target: str, max_hops: int = None,
    ) -> List[JoinPath]:
        """Return all join paths between source and target up to max_hops."""
        if max_hops is None:
            max_hops = self._cfg.graph_max_hops

        results: List[JoinPath] = []

        def dfs(current: str, visited: Set[str], path: List[JoinEdge]):
            if len(path) > max_hops:
                return
            if current == target and path:
                results.append(JoinPath(edges=list(path)))
                return
            for neighbour, edge in self._adj.get(current, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    path.append(edge)
                    dfs(neighbour, visited, path)
                    path.pop()
                    visited.discard(neighbour)

        dfs(source, {source}, [])
        results.sort(key=lambda p: p.hops)
        return results

    # ── Scoring for hybrid retrieval ─────────────────────────────────────

    def score_expansion(
        self, seed_tables: List[str], candidate: str,
    ) -> float:
        """Score a candidate table by its graph proximity to seed tables.

        Returns a value in [0, 1].  Direct FK neighbours score 1.0,
        2-hop neighbours 0.5, etc.
        """
        for seed in seed_tables:
            path = self.shortest_path(seed, candidate)
            if path is not None:
                hops = path.hops
                if hops == 0:
                    return 1.0
                return max(0.0, 1.0 - (hops - 1) * 0.5)
        return 0.0
