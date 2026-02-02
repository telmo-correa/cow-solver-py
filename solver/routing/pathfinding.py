"""Token graph and pathfinding for multi-hop routing.

This module provides graph-based pathfinding for discovering multi-hop
routes through available liquidity. It separates the graph structure
and pathfinding algorithms from pool storage.

Phase 4 will extend these primitives for:
- Split routing path enumeration
- Ring trade detection
- Joint optimization path selection
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Protocol

from solver.models.types import normalize_address

if TYPE_CHECKING:
    from solver.pools.registry import PoolRegistry


class PoolGraphSource(Protocol):
    """Protocol for sources that can provide graph edges.

    This allows PathFinder to work with any pool storage that
    provides the necessary token pair information.

    Implementation Note:
        This protocol accesses internal storage attributes of PoolRegistry.
        This is intentional - PathFinder is tightly coupled to PoolRegistry
        as an internal implementation detail. The protocol exists to:
        1. Document the required interface explicitly
        2. Enable testing with mock registries
        3. Allow future refactoring of the storage layer

        If PoolRegistry's internal structure changes, this protocol and
        TokenGraph._build_from_registry must be updated together.

        Value types use Any because we only iterate over keys for graph edges.
    """

    @property
    def _pools(self) -> dict[frozenset[str], Any]: ...
    @property
    def _v3_pools(self) -> dict[tuple[str, str, int], Any]: ...
    @property
    def _weighted_pools(self) -> dict[tuple[str, str], Any]: ...
    @property
    def _stable_pools(self) -> dict[tuple[str, str], Any]: ...
    @property
    def _limit_orders(self) -> dict[tuple[str, str], Any]: ...


class TokenGraph:
    """Graph of tokens connected by liquidity pools.

    Provides an adjacency list representation of the token graph
    where edges represent tradeable token pairs (via any pool type).

    This is a pure data structure with no caching - caching is handled
    by PathFinder which owns TokenGraph instances.
    """

    def __init__(self) -> None:
        """Initialize an empty token graph."""
        self._adjacency: dict[str, set[str]] = {}

    @classmethod
    def from_registry(cls, registry: PoolGraphSource) -> TokenGraph:
        """Build a TokenGraph from a PoolRegistry.

        Args:
            registry: Pool registry containing all pool types

        Returns:
            TokenGraph with edges for all tradeable token pairs
        """
        graph = cls()
        graph._build_from_registry(registry)
        return graph

    def _build_from_registry(self, registry: PoolGraphSource) -> None:
        """Build adjacency list from all pool types in registry."""
        # Add V2 pools
        for token_pair in registry._pools:
            tokens = list(token_pair)
            self._add_edge(tokens[0], tokens[1])

        # Add V3 pools
        for token_a, token_b, _fee in registry._v3_pools:
            self._add_edge(token_a, token_b)

        # Add weighted pools
        for (token_a, token_b), _ in registry._weighted_pools.items():
            self._add_edge(token_a, token_b)

        # Add stable pools
        for (token_a, token_b), _ in registry._stable_pools.items():
            self._add_edge(token_a, token_b)

        # Add limit orders (bidirectional for path discovery)
        # Note: Limit orders are unidirectional for trading, but we add
        # both directions for path finding. Routing validates actual tradability.
        for (taker_token, maker_token), _ in registry._limit_orders.items():
            self._add_edge(taker_token, maker_token)

    def _add_edge(self, token_a: str, token_b: str) -> None:
        """Add a bidirectional edge between two tokens."""
        if token_a not in self._adjacency:
            self._adjacency[token_a] = set()
        if token_b not in self._adjacency:
            self._adjacency[token_b] = set()
        self._adjacency[token_a].add(token_b)
        self._adjacency[token_b].add(token_a)

    def get_neighbors(self, token: str) -> set[str]:
        """Get all tokens directly tradeable with given token.

        Args:
            token: Token address (any case, will be normalized)

        Returns:
            Set of token addresses connected by at least one pool
        """
        return self._adjacency.get(normalize_address(token), set())

    def _get_neighbors_fast(self, token_normalized: str) -> set[str]:
        """Get neighbors for an already-normalized token address.

        Internal method for performance-critical paths where token
        is known to be normalized.
        """
        return self._adjacency.get(token_normalized, set())

    def has_token(self, token: str) -> bool:
        """Check if a token exists in the graph."""
        return normalize_address(token) in self._adjacency

    def _has_token_fast(self, token_normalized: str) -> bool:
        """Check if token exists (already-normalized address)."""
        return token_normalized in self._adjacency

    @property
    def token_count(self) -> int:
        """Number of unique tokens in the graph."""
        return len(self._adjacency)


class PathFinder:
    """Facade for pathfinding operations with caching.

    PathFinder owns a TokenGraph and provides cached pathfinding
    operations. It automatically invalidates the cache when pools
    change in the underlying registry.

    Usage:
        finder = PathFinder(registry)
        paths = finder.find_all_paths(token_in, token_out, max_hops=2)
    """

    def __init__(self, registry: PoolRegistry) -> None:
        """Initialize PathFinder for a pool registry.

        Args:
            registry: The pool registry to find paths through
        """
        self._registry = registry
        self._graph: TokenGraph | None = None
        # Cache for path queries: (token_in, token_out, max_hops) -> paths
        self._path_cache: dict[tuple[str, str, int], list[list[str]]] = {}
        # Cache for shortest path: (token_in, token_out, max_hops) -> path or None
        self._shortest_path_cache: dict[tuple[str, str, int], list[str] | None] = {}

    def invalidate(self) -> None:
        """Invalidate the cached graph and path caches.

        Call this after adding pools to the registry to ensure
        the graph is rebuilt on next pathfinding operation.
        """
        self._graph = None
        self._path_cache.clear()
        self._shortest_path_cache.clear()

    @property
    def graph(self) -> TokenGraph:
        """Get or build the token graph (lazy initialization)."""
        if self._graph is None:
            self._graph = TokenGraph.from_registry(self._registry)
        return self._graph

    def find_all_paths(
        self,
        token_in: str,
        token_out: str,
        max_hops: int = 3,
        max_paths: int = 20,
    ) -> list[list[str]]:
        """Find candidate paths from token_in to token_out.

        Uses BFS to enumerate paths up to max_hops. BFS naturally finds
        shorter paths first, so direct paths are returned before multi-hop.

        Path types by hops:
        - Direct (1 hop): [token_in, token_out]
        - 2-hop: [token_in, intermediate, token_out]
        - 3-hop: [token_in, int1, int2, token_out]

        Results are cached for repeated queries with the same parameters.

        Performance: For token pairs connected through high-degree hub tokens
        (like WETH with 600+ neighbors), there can be thousands of possible
        paths. The max_paths limit prevents exponential exploration while
        still finding good candidates (BFS finds shorter paths first).

        Args:
            token_in: Starting token address
            token_out: Target token address
            max_hops: Maximum number of swaps allowed (default 3)
            max_paths: Maximum number of paths to return (default 20)

        Returns:
            List of paths, each path is a list of token addresses.
            Empty list if no paths found.
        """
        token_in_norm = normalize_address(token_in)
        token_out_norm = normalize_address(token_out)

        if token_in_norm == token_out_norm:
            return []

        # Check cache first (cache key includes max_paths)
        cache_key = (token_in_norm, token_out_norm, max_hops)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        graph = self.graph
        if not graph._has_token_fast(token_in_norm) or not graph._has_token_fast(token_out_norm):
            self._path_cache[cache_key] = []
            return []

        candidates: list[list[str]] = []
        start_neighbors = graph._get_neighbors_fast(token_in_norm)
        end_neighbors = graph._get_neighbors_fast(token_out_norm)

        # Optimization 1: Check direct path first (very common case)
        if token_out_norm in start_neighbors:
            candidates.append([token_in_norm, token_out_norm])
            if len(candidates) >= max_paths or max_hops == 1:
                self._path_cache[cache_key] = candidates
                return candidates

        if max_hops < 2:
            self._path_cache[cache_key] = candidates
            return candidates

        # Optimization 2: For 2-hop paths, find common neighbors (intersection)
        # A 2-hop path exists if there's a common neighbor between start and end
        common_neighbors = start_neighbors & end_neighbors
        for intermediate in common_neighbors:
            if intermediate != token_in_norm and intermediate != token_out_norm:
                candidates.append([token_in_norm, intermediate, token_out_norm])
                if len(candidates) >= max_paths:
                    self._path_cache[cache_key] = candidates
                    return candidates

        if max_hops < 3:
            self._path_cache[cache_key] = candidates
            return candidates

        # For 3+ hops, use BFS but skip paths we've already found
        # Track visited to avoid cycles within a single path
        queue: deque[tuple[list[str], frozenset[str]]] = deque()
        # Start with 2-hop partial paths (not including destination)
        for neighbor in start_neighbors:
            if neighbor != token_out_norm:  # Skip direct paths (already handled)
                queue.append(([token_in_norm, neighbor], frozenset([token_in_norm, neighbor])))

        while queue and len(candidates) < max_paths:
            path, visited = queue.popleft()

            # Only explore if we have room for more hops
            if len(path) >= max_hops + 1:
                continue

            current = path[-1]
            current_neighbors = graph._get_neighbors_fast(current)

            # Check if destination is reachable
            if token_out_norm in current_neighbors:
                new_path = path + [token_out_norm]
                # Skip if this is a 2-hop path (already found above)
                if len(new_path) > 3 or current not in common_neighbors:
                    candidates.append(new_path)
                    if len(candidates) >= max_paths:
                        break

            # Continue exploring (only if we haven't reached hop limit)
            if len(path) < max_hops:
                for neighbor in current_neighbors:
                    if neighbor not in visited and neighbor != token_out_norm:
                        new_path = path + [neighbor]
                        new_visited = visited | frozenset([neighbor])
                        queue.append((new_path, new_visited))

        # Cache result before returning
        self._path_cache[cache_key] = candidates
        return candidates

    def find_shortest_path(
        self,
        token_in: str,
        token_out: str,
        max_hops: int = 3,
    ) -> list[str] | None:
        """Find the shortest path from token_in to token_out.

        Uses BFS to find the shortest path (by number of hops).
        More efficient than find_all_paths when only one path is needed.

        Results are cached for repeated queries with the same parameters.

        Args:
            token_in: Starting token address
            token_out: Target token address
            max_hops: Maximum number of swaps allowed (default 3)

        Returns:
            List of token addresses representing the path, or None if not found.
        """
        token_in_norm = normalize_address(token_in)
        token_out_norm = normalize_address(token_out)

        if token_in_norm == token_out_norm:
            return [token_in_norm]

        # Check cache first
        cache_key = (token_in_norm, token_out_norm, max_hops)
        if cache_key in self._shortest_path_cache:
            return self._shortest_path_cache[cache_key]

        graph = self.graph
        if not graph._has_token_fast(token_in_norm) or not graph._has_token_fast(token_out_norm):
            self._shortest_path_cache[cache_key] = None
            return None

        queue: deque[list[str]] = deque([[token_in_norm]])
        visited = {token_in_norm}

        while queue:
            path = queue.popleft()
            if len(path) > max_hops + 1:
                continue

            current = path[-1]
            # Use fast neighbor lookup since all tokens in BFS are already normalized
            for neighbor in graph._get_neighbors_fast(current):
                if neighbor == token_out_norm:
                    result = path + [neighbor]
                    self._shortest_path_cache[cache_key] = result
                    return result
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        self._shortest_path_cache[cache_key] = None
        return None


__all__ = ["TokenGraph", "PathFinder", "PoolGraphSource"]
