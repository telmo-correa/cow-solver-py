"""Graph data structures for order analysis.

This module provides reusable graph data structures:
- UnionFind: Efficient connected component detection
- OrderGraph: Directed graph of orders by token pair
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from solver.models.auction import Order
from solver.models.types import normalize_address


class UnionFind:
    """Union-Find data structure for efficient connected component detection.

    Uses path compression and union by rank for O(α(n)) amortized operations,
    where α is the inverse Ackermann function (effectively constant).
    """

    def __init__(self) -> None:
        """Initialize empty Union-Find structure."""
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        """Find the root of element x with path compression."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            return x

        # Path compression: make all nodes point directly to root
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        """Union the sets containing x and y using rank."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank: attach smaller tree under larger
        if self._rank[root_x] < self._rank[root_y]:
            self._parent[root_x] = root_y
        elif self._rank[root_x] > self._rank[root_y]:
            self._parent[root_y] = root_x
        else:
            self._parent[root_y] = root_x
            self._rank[root_x] += 1

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same component."""
        return self.find(x) == self.find(y)


@dataclass
class OrderGraph:
    """Directed graph of orders organized by token pair.

    Nodes are token addresses, edges are orders selling one token for another.
    graph[sell_token][buy_token] = list of orders on that edge.
    """

    # sell_token -> buy_token -> list of orders
    edges: dict[str, dict[str, list[Order]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    # Cached token set (computed lazily)
    _tokens_cache: set[str] | None = field(default=None, repr=False)

    @classmethod
    def from_orders(cls, orders: Iterable[Order]) -> OrderGraph:
        """Build graph from a collection of orders.

        Args:
            orders: Orders to add to the graph

        Returns:
            OrderGraph with all orders indexed by token pair
        """
        graph = cls()
        for order in orders:
            sell_token = normalize_address(order.sell_token)
            buy_token = normalize_address(order.buy_token)
            graph.edges[sell_token][buy_token].append(order)
        return graph

    @property
    def tokens(self) -> set[str]:
        """Get all unique tokens in the graph (cached)."""
        if self._tokens_cache is None:
            self._tokens_cache = set(self.edges.keys())
            for destinations in self.edges.values():
                self._tokens_cache.update(destinations.keys())
        return self._tokens_cache

    @property
    def edge_count(self) -> int:
        """Count directed edges (token pairs with at least one order)."""
        return sum(len(dests) for dests in self.edges.values())

    def get_orders(self, sell_token: str, buy_token: str) -> list[Order]:
        """Get orders selling sell_token for buy_token."""
        return self.edges.get(sell_token, {}).get(buy_token, [])

    def has_edge(self, sell_token: str, buy_token: str) -> bool:
        """Check if there's at least one order on this edge."""
        return len(self.get_orders(sell_token, buy_token)) > 0

    def find_3_cycles(self) -> list[tuple[str, str, str]]:
        """Find all 3-node cycles (A→B→C→A).

        Returns cycles as sorted tuples to deduplicate.
        The actual direction must be recovered when checking viability.

        Returns:
            List of 3-tuples of token addresses (sorted for deduplication)
        """
        seen: set[tuple[str, str, str]] = set()
        cycles: list[tuple[str, str, str]] = []
        tokens = list(self.edges.keys())

        for a in tokens:
            for b in self.edges[a]:
                if b == a:
                    continue
                for c in self.edges.get(b, {}):
                    if c in (a, b):
                        continue
                    # Check if C→A exists (closes the cycle)
                    if self.has_edge(c, a):
                        # Normalize to sorted tuple to avoid duplicates
                        sorted_tokens = sorted([a, b, c])
                        cycle = (sorted_tokens[0], sorted_tokens[1], sorted_tokens[2])
                        if cycle not in seen:
                            seen.add(cycle)
                            cycles.append(cycle)

        return cycles

    def find_4_cycles(self, limit: int = 50) -> list[tuple[str, str, str, str]]:
        """Find 4-node cycles (A→B→C→D→A).

        Limited to avoid combinatorial explosion on large graphs.

        Args:
            limit: Maximum number of cycles to find

        Returns:
            List of 4-tuples of token addresses (sorted for deduplication)
        """
        seen: set[tuple[str, str, str, str]] = set()
        cycles: list[tuple[str, str, str, str]] = []
        tokens = list(self.edges.keys())
        limit_reached = False

        for a in tokens:
            if limit_reached:
                break
            for b in self.edges[a]:
                if limit_reached:
                    break
                if b == a:
                    continue
                for c in self.edges.get(b, {}):
                    if limit_reached:
                        break
                    if c in (a, b):
                        continue
                    for d in self.edges.get(c, {}):
                        if d in (a, b, c):
                            continue
                        # Check if D→A exists (closes the cycle)
                        if self.has_edge(d, a):
                            # Normalize to sorted tuple to avoid duplicates
                            sorted_tokens = sorted([a, b, c, d])
                            cycle = (
                                sorted_tokens[0],
                                sorted_tokens[1],
                                sorted_tokens[2],
                                sorted_tokens[3],
                            )
                            if cycle not in seen:
                                seen.add(cycle)
                                cycles.append(cycle)
                                if len(cycles) >= limit:
                                    limit_reached = True
                                    break

        return cycles


def build_token_graph(orders: list[Order]) -> dict[str, set[str]]:
    """Build undirected token graph from orders.

    Args:
        orders: List of orders

    Returns:
        Adjacency list representation of token graph
    """
    graph: dict[str, set[str]] = defaultdict(set)
    for order in orders:
        sell_token = normalize_address(order.sell_token)
        buy_token = normalize_address(order.buy_token)
        graph[sell_token].add(buy_token)
        graph[buy_token].add(sell_token)
    return graph


def find_spanning_tree(graph: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Find a spanning tree of the token graph using BFS.

    Args:
        graph: Adjacency list of token graph

    Returns:
        List of edges (parent, child) in the spanning tree
    """
    if not graph:
        return []

    edges: list[tuple[str, str]] = []
    visited: set[str] = set()

    # Start from arbitrary node
    start = next(iter(graph))
    queue = [start]
    visited.add(start)

    while queue:
        current = queue.pop(0)
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                edges.append((current, neighbor))
                queue.append(neighbor)

    return edges


__all__ = [
    "UnionFind",
    "OrderGraph",
    "build_token_graph",
    "find_spanning_tree",
]
