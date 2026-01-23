"""Pool registry for managing liquidity across all pool types.

This module provides PoolRegistry for managing pools from multiple sources:
- UniswapV2 (constant product)
- UniswapV3 (concentrated liquidity)
- Balancer Weighted (weighted product)
- Balancer Stable (StableSwap)
- 0x Limit Orders (foreign limit orders)
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from solver.amm.balancer import BalancerStablePool, BalancerWeightedPool
from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.amm.uniswap_v3 import UniswapV3Pool
from solver.models.types import normalize_address
from solver.pools.limit_order import LimitOrderPool

if TYPE_CHECKING:
    from solver.models.auction import Liquidity

from .types import AnyPool


class PoolRegistry:
    """Registry of liquidity pools for routing.

    This class manages a collection of pools and provides methods for:
    - Looking up pools by token pair
    - Building token graphs for pathfinding
    - Finding multi-hop paths through available liquidity

    Supports UniswapV2, UniswapV3, Balancer Weighted, Balancer Stable, and 0x Limit Orders.
    """

    def __init__(self, pools: list[UniswapV2Pool] | None = None) -> None:
        """Initialize the registry with optional pools.

        Args:
            pools: Initial list of V2 pools. If None, starts empty.
        """
        self._pools: dict[frozenset[str], UniswapV2Pool] = {}
        # V3 pools keyed by (token0, token1, fee) for multiple fee tiers per pair
        self._v3_pools: dict[tuple[str, str, int], UniswapV3Pool] = {}
        # Balancer pools indexed by canonical token pair (for N-token pools, create N*(N-1)/2 entries)
        self._weighted_pools: dict[tuple[str, str], list[BalancerWeightedPool]] = {}
        self._stable_pools: dict[tuple[str, str], list[BalancerStablePool]] = {}
        # Limit orders indexed by (taker_token, maker_token) - unidirectional
        self._limit_orders: dict[tuple[str, str], list[LimitOrderPool]] = {}
        # Track unique pool IDs for O(1) count and duplicate detection
        self._weighted_pool_ids: set[str] = set()
        self._stable_pool_ids: set[str] = set()
        self._limit_order_ids: set[str] = set()
        self._graph: dict[str, set[str]] | None = None  # Cached graph

        if pools:
            for pool in pools:
                self.add_pool(pool)

    def add_pool(self, pool: UniswapV2Pool) -> None:
        """Add a V2 pool to the registry.

        Args:
            pool: The pool to add. If a pool for this token pair already exists,
                  it will be replaced.
        """
        token0_norm = normalize_address(pool.token0)
        token1_norm = normalize_address(pool.token1)
        pair_key = frozenset([token0_norm, token1_norm])
        self._pools[pair_key] = pool
        self._graph = None  # Invalidate cached graph

    def get_pool(self, token_a: str, token_b: str) -> UniswapV2Pool | None:
        """Get a V2 pool for a token pair (order independent).

        Args:
            token_a: First token address (any case)
            token_b: Second token address (any case)

        Returns:
            UniswapV2Pool if found, None otherwise
        """
        token_a_norm = normalize_address(token_a)
        token_b_norm = normalize_address(token_b)
        pair_key = frozenset([token_a_norm, token_b_norm])
        return self._pools.get(pair_key)

    def add_v3_pool(self, pool: UniswapV3Pool) -> None:
        """Add a V3 pool to the registry.

        Args:
            pool: The V3 pool to add. Unlike V2, multiple V3 pools can exist
                  for the same token pair with different fee tiers.
        """
        token0_norm = normalize_address(pool.token0)
        token1_norm = normalize_address(pool.token1)
        # Ensure canonical ordering (token0 < token1 by address)
        if token0_norm > token1_norm:
            token0_norm, token1_norm = token1_norm, token0_norm
        key = (token0_norm, token1_norm, pool.fee)
        self._v3_pools[key] = pool
        self._graph = None  # Invalidate cached graph

    def get_v3_pools(self, token_a: str, token_b: str) -> list[UniswapV3Pool]:
        """Get all V3 pools for a token pair (all fee tiers).

        Args:
            token_a: First token address (any case)
            token_b: Second token address (any case)

        Returns:
            List of UniswapV3Pool objects for this pair (may be empty)
        """
        token_a_norm = normalize_address(token_a)
        token_b_norm = normalize_address(token_b)
        # Ensure canonical ordering
        if token_a_norm > token_b_norm:
            token_a_norm, token_b_norm = token_b_norm, token_a_norm

        result = []
        for (t0, t1, _fee), pool in self._v3_pools.items():
            if t0 == token_a_norm and t1 == token_b_norm:
                result.append(pool)
        return result

    def add_weighted_pool(self, pool: BalancerWeightedPool) -> None:
        """Add a Balancer weighted pool to the registry.

        The pool is indexed by all token pairs it supports. For an N-token pool,
        this creates N*(N-1)/2 index entries.

        Duplicate pools (same ID) are ignored.

        Args:
            pool: The weighted pool to add
        """
        # Skip duplicate pools
        if pool.id in self._weighted_pool_ids:
            return

        self._weighted_pool_ids.add(pool.id)

        tokens = [r.token.lower() for r in pool.reserves]
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i + 1 :]:
                # Canonical pair ordering
                pair = (min(t1, t2), max(t1, t2))
                if pair not in self._weighted_pools:
                    self._weighted_pools[pair] = []
                self._weighted_pools[pair].append(pool)
        self._graph = None  # Invalidate cached graph

    def add_stable_pool(self, pool: BalancerStablePool) -> None:
        """Add a Balancer stable pool to the registry.

        The pool is indexed by all token pairs it supports. For an N-token pool,
        this creates N*(N-1)/2 index entries.

        Duplicate pools (same ID) are ignored.

        Args:
            pool: The stable pool to add
        """
        # Skip duplicate pools
        if pool.id in self._stable_pool_ids:
            return

        self._stable_pool_ids.add(pool.id)

        tokens = [r.token.lower() for r in pool.reserves]
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i + 1 :]:
                # Canonical pair ordering
                pair = (min(t1, t2), max(t1, t2))
                if pair not in self._stable_pools:
                    self._stable_pools[pair] = []
                self._stable_pools[pair].append(pool)
        self._graph = None  # Invalidate cached graph

    def get_weighted_pools(self, token_a: str, token_b: str) -> list[BalancerWeightedPool]:
        """Get all weighted pools for a token pair.

        Args:
            token_a: First token address (any case)
            token_b: Second token address (any case)

        Returns:
            List of BalancerWeightedPool objects for this pair (may be empty)
        """
        token_a_norm = normalize_address(token_a)
        token_b_norm = normalize_address(token_b)
        pair = (min(token_a_norm, token_b_norm), max(token_a_norm, token_b_norm))
        return self._weighted_pools.get(pair, [])

    def get_stable_pools(self, token_a: str, token_b: str) -> list[BalancerStablePool]:
        """Get all stable pools for a token pair.

        Args:
            token_a: First token address (any case)
            token_b: Second token address (any case)

        Returns:
            List of BalancerStablePool objects for this pair (may be empty)
        """
        token_a_norm = normalize_address(token_a)
        token_b_norm = normalize_address(token_b)
        pair = (min(token_a_norm, token_b_norm), max(token_a_norm, token_b_norm))
        return self._stable_pools.get(pair, [])

    def add_limit_order(self, order: LimitOrderPool) -> None:
        """Add a 0x limit order to the registry.

        Limit orders are unidirectional: taker_token -> maker_token only.
        Duplicate orders (same ID) are ignored.

        Args:
            order: The limit order to add
        """
        # Skip duplicate orders
        if order.id in self._limit_order_ids:
            return

        self._limit_order_ids.add(order.id)

        # Index by (taker_token, maker_token) - the only valid direction
        taker_norm = normalize_address(order.taker_token)
        maker_norm = normalize_address(order.maker_token)
        key = (taker_norm, maker_norm)

        if key not in self._limit_orders:
            self._limit_orders[key] = []
        self._limit_orders[key].append(order)
        self._graph = None  # Invalidate cached graph

    def get_limit_orders(self, token_in: str, token_out: str) -> list[LimitOrderPool]:
        """Get all limit orders for a token pair (directional).

        Unlike AMM pools, limit orders are unidirectional. This only returns
        orders where token_in is the taker token and token_out is the maker token.

        Args:
            token_in: Input token address (taker token)
            token_out: Output token address (maker token)

        Returns:
            List of LimitOrderPool objects for this direction (may be empty)
        """
        token_in_norm = normalize_address(token_in)
        token_out_norm = normalize_address(token_out)
        key = (token_in_norm, token_out_norm)
        return self._limit_orders.get(key, [])

    def get_pools_for_pair(self, token_a: str, token_b: str) -> list[AnyPool]:
        """Get all pools (V2 + V3 + Balancer + limit orders) for a token pair.

        For directional liquidity like limit orders, token_a is treated as
        the input token and token_b as the output token.

        Args:
            token_a: First token address / input token (any case)
            token_b: Second token address / output token (any case)

        Returns:
            List of all pools for this pair
        """
        pools: list[AnyPool] = []

        # Add V2 pool if exists (bidirectional)
        v2_pool = self.get_pool(token_a, token_b)
        if v2_pool is not None:
            pools.append(v2_pool)

        # Add all V3 pools (bidirectional)
        pools.extend(self.get_v3_pools(token_a, token_b))

        # Add all Balancer weighted pools (bidirectional)
        pools.extend(self.get_weighted_pools(token_a, token_b))

        # Add all Balancer stable pools (bidirectional)
        pools.extend(self.get_stable_pools(token_a, token_b))

        # Add all limit orders (unidirectional: token_a -> token_b only)
        pools.extend(self.get_limit_orders(token_a, token_b))

        return pools

    @property
    def v3_pool_count(self) -> int:
        """Return the number of V3 pools in the registry."""
        return len(self._v3_pools)

    @property
    def weighted_pool_count(self) -> int:
        """Return the number of unique weighted pools in the registry."""
        return len(self._weighted_pool_ids)

    @property
    def stable_pool_count(self) -> int:
        """Return the number of unique stable pools in the registry."""
        return len(self._stable_pool_ids)

    @property
    def limit_order_count(self) -> int:
        """Return the number of unique limit orders in the registry."""
        return len(self._limit_order_ids)

    def _build_graph(self) -> dict[str, set[str]]:
        """Build adjacency list of tokens connected by pools (all types)."""
        graph: dict[str, set[str]] = {}

        # Add V2 pools to graph
        for token_pair in self._pools:
            tokens = list(token_pair)
            token_a, token_b = tokens[0], tokens[1]

            if token_a not in graph:
                graph[token_a] = set()
            if token_b not in graph:
                graph[token_b] = set()

            graph[token_a].add(token_b)
            graph[token_b].add(token_a)

        # Add V3 pools to graph
        for token_a, token_b, _fee in self._v3_pools:
            if token_a not in graph:
                graph[token_a] = set()
            if token_b not in graph:
                graph[token_b] = set()

            graph[token_a].add(token_b)
            graph[token_b].add(token_a)

        # Add weighted pools to graph
        for (token_a, token_b), _ in self._weighted_pools.items():
            if token_a not in graph:
                graph[token_a] = set()
            if token_b not in graph:
                graph[token_b] = set()

            graph[token_a].add(token_b)
            graph[token_b].add(token_a)

        # Add stable pools to graph
        for (token_a, token_b), _ in self._stable_pools.items():
            if token_a not in graph:
                graph[token_a] = set()
            if token_b not in graph:
                graph[token_b] = set()

            graph[token_a].add(token_b)
            graph[token_b].add(token_a)

        # Add limit orders to graph (unidirectional: taker -> maker)
        for (taker_token, maker_token), _ in self._limit_orders.items():
            if taker_token not in graph:
                graph[taker_token] = set()
            if maker_token not in graph:
                graph[maker_token] = set()

            # Limit orders are unidirectional, but for path finding we add both
            # directions to discover paths. The actual routing will check validity.
            graph[taker_token].add(maker_token)
            graph[maker_token].add(taker_token)

        return graph

    def get_all_candidate_paths(
        self,
        token_in: str,
        token_out: str,
        max_hops: int = 2,
    ) -> list[list[str]]:
        """Generate all candidate paths from token_in to token_out.

        Similar to Rust's baseline solver path_candidates function.
        Generates all possible paths including:
        - Direct path: [token_in, token_out]
        - 1-hop paths: [token_in, intermediary, token_out]
        - 2-hop paths: [token_in, int1, int2, token_out] (if max_hops >= 2)

        Only includes paths where pools actually exist for each hop.

        Args:
            token_in: Starting token address
            token_out: Target token address
            max_hops: Maximum number of swaps allowed (default 2)

        Returns:
            List of paths, where each path is a list of token addresses.
            Empty list if no paths found.
        """
        token_in_norm = normalize_address(token_in)
        token_out_norm = normalize_address(token_out)

        if token_in_norm == token_out_norm:
            return []

        # Use cached graph or rebuild
        if self._graph is None:
            self._graph = self._build_graph()
        graph = self._graph

        if token_in_norm not in graph or token_out_norm not in graph:
            return []

        candidates: list[list[str]] = []

        # Use BFS to find ALL paths up to max_hops
        # Each queue entry is (path_so_far, visited_set)
        queue: deque[tuple[list[str], set[str]]] = deque()
        queue.append(([token_in_norm], {token_in_norm}))

        while queue:
            path, visited = queue.popleft()

            # Check if we've exceeded max hops
            # path has len nodes, so len-1 hops. We want at most max_hops.
            if len(path) > max_hops + 1:
                continue

            current = path[-1]

            for neighbor in graph.get(current, set()):
                if neighbor == token_out_norm:
                    # Found a valid path to destination
                    candidates.append(path + [neighbor])
                elif neighbor not in visited and len(path) < max_hops + 1:
                    # Continue exploring through this neighbor
                    new_visited = visited | {neighbor}
                    queue.append((path + [neighbor], new_visited))

        return candidates

    def find_path(
        self,
        token_in: str,
        token_out: str,
        max_hops: int = 2,
    ) -> list[str] | None:
        """Find a path from token_in to token_out through available pools.

        Uses BFS to find the shortest path (by number of hops).

        Args:
            token_in: Starting token address
            token_out: Target token address
            max_hops: Maximum number of swaps allowed (default 2)

        Returns:
            List of token addresses representing the path, or None if no path found.
        """
        token_in_norm = normalize_address(token_in)
        token_out_norm = normalize_address(token_out)

        if token_in_norm == token_out_norm:
            return [token_in_norm]

        # Use cached graph or rebuild
        if self._graph is None:
            self._graph = self._build_graph()
        graph = self._graph

        if token_in_norm not in graph or token_out_norm not in graph:
            return None

        queue: deque[list[str]] = deque([[token_in_norm]])
        visited = {token_in_norm}

        while queue:
            path = queue.popleft()
            if len(path) > max_hops + 1:
                continue

            current = path[-1]
            for neighbor in graph.get(current, set()):
                if neighbor == token_out_norm:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None

    def get_all_pools_on_path(self, path: list[str]) -> list[AnyPool]:
        """Get all pools needed to execute a multi-hop swap path.

        For each hop in the path, selects a pool from available options.
        Priority: V2 > V3 > Balancer Weighted > Balancer Stable

        Args:
            path: List of token addresses forming the swap path

        Returns:
            List of pools for each hop in the path

        Raises:
            ValueError: If any pool in the path is not found
        """
        pools: list[AnyPool] = []
        for i in range(len(path) - 1):
            all_pools = self.get_pools_for_pair(path[i], path[i + 1])
            if not all_pools:
                raise ValueError(f"No pool found for {path[i]} -> {path[i + 1]}")

            # Select best pool - prioritize V2 for simplicity in multi-hop
            selected_pool: AnyPool | None = None
            for pool in all_pools:
                if isinstance(pool, UniswapV2Pool):
                    selected_pool = pool
                    break
            if selected_pool is None:
                # Fall back to first available pool
                selected_pool = all_pools[0]

            pools.append(selected_pool)
        return pools

    @property
    def pool_count(self) -> int:
        """Return the number of V2 pools in the registry."""
        return len(self._pools)


def build_registry_from_liquidity(liquidity_list: list[Liquidity]) -> PoolRegistry:
    """Build a PoolRegistry from auction liquidity sources.

    Parses V2, V3, Balancer (weighted/stable), and 0x limit order pools.

    Args:
        liquidity_list: List of Liquidity objects from the auction

    Returns:
        PoolRegistry populated with parsed pools
    """
    from solver.amm.balancer import parse_stable_pool, parse_weighted_pool
    from solver.amm.uniswap_v2 import parse_liquidity_to_pool
    from solver.amm.uniswap_v3 import parse_v3_liquidity
    from solver.pools.limit_order import parse_limit_order

    registry = PoolRegistry()
    for liq in liquidity_list:
        # Try V2 first
        v2_pool = parse_liquidity_to_pool(liq)
        if v2_pool is not None:
            registry.add_pool(v2_pool)
            continue

        # Try V3
        v3_pool = parse_v3_liquidity(liq)
        if v3_pool is not None:
            registry.add_v3_pool(v3_pool)
            continue

        # Try Balancer weighted
        weighted_pool = parse_weighted_pool(liq)
        if weighted_pool is not None:
            registry.add_weighted_pool(weighted_pool)
            continue

        # Try Balancer stable
        stable_pool = parse_stable_pool(liq)
        if stable_pool is not None:
            registry.add_stable_pool(stable_pool)
            continue

        # Try limit order
        limit_order = parse_limit_order(liq)
        if limit_order is not None:
            registry.add_limit_order(limit_order)

    return registry
