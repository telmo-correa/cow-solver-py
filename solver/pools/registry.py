"""Pool registry for managing liquidity across all pool types.

This module provides PoolRegistry for managing pools from multiple sources:
- UniswapV2 (constant product)
- UniswapV3 (concentrated liquidity)
- Balancer Weighted (weighted product)
- Balancer Stable (StableSwap)
- 0x Limit Orders (foreign limit orders)

Pathfinding operations are delegated to PathFinder (solver.routing.pathfinding)
which provides efficient graph-based route discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from solver.amm.balancer import BalancerStablePool, BalancerWeightedPool
from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.amm.uniswap_v3 import UniswapV3Pool
from solver.models.types import normalize_address
from solver.pools.limit_order import LimitOrderPool
from solver.pools.types import AnyPool

logger = structlog.get_logger()

if TYPE_CHECKING:
    from solver.models.auction import Liquidity
    from solver.routing.pathfinding import PathFinder


class PoolRegistry:
    """Registry of liquidity pools for routing.

    This class manages pool storage and lookup. For pathfinding operations,
    use the `pathfinder` property which provides efficient graph-based routing.

    Supports UniswapV2, UniswapV3, Balancer Weighted, Balancer Stable, and 0x Limit Orders.
    """

    def __init__(self, pools: list[AnyPool] | None = None) -> None:
        """Initialize the registry with optional pools.

        Args:
            pools: Initial pools of any supported type. Dispatches to appropriate
                   add method based on pool type. If None, starts empty.
        """
        self._pools: dict[frozenset[str], UniswapV2Pool] = {}
        # V3 pools keyed by (token0, token1, fee) for multiple fee tiers per pair
        self._v3_pools: dict[tuple[str, str, int], UniswapV3Pool] = {}
        # Secondary index for V3: (token0, token1) -> list of pools (all fee tiers)
        # This avoids O(n) iteration in get_v3_pools
        self._v3_pools_by_pair: dict[tuple[str, str], list[UniswapV3Pool]] = {}
        # Balancer pools indexed by canonical token pair (for N-token pools, create N*(N-1)/2 entries)
        self._weighted_pools: dict[tuple[str, str], list[BalancerWeightedPool]] = {}
        self._stable_pools: dict[tuple[str, str], list[BalancerStablePool]] = {}
        # Limit orders indexed by (taker_token, maker_token) - unidirectional
        self._limit_orders: dict[tuple[str, str], list[LimitOrderPool]] = {}
        # Track unique pool IDs for O(1) count and duplicate detection
        self._weighted_pool_ids: set[str] = set()
        self._stable_pool_ids: set[str] = set()
        self._limit_order_ids: set[str] = set()
        # Lazy-initialized PathFinder for graph operations
        self._pathfinder: PathFinder | None = None
        # Cache for get_pools_for_pair (invalidated on pool add)
        self._pools_for_pair_cache: dict[tuple[str, str], list[AnyPool]] = {}

        if pools:
            for pool in pools:
                self.add_any_pool(pool)

    @property
    def pathfinder(self) -> PathFinder:
        """Get the PathFinder for this registry (lazy initialization).

        The PathFinder provides efficient graph-based pathfinding operations.
        It is automatically invalidated when pools are added.
        """
        if self._pathfinder is None:
            from solver.routing.pathfinding import PathFinder

            self._pathfinder = PathFinder(self)
        return self._pathfinder

    def _invalidate_pathfinder(self) -> None:
        """Invalidate the cached pathfinder and pools cache after pool changes."""
        if self._pathfinder is not None:
            self._pathfinder.invalidate()
        self._pools_for_pair_cache.clear()

    def add_any_pool(self, pool: AnyPool) -> None:
        """Add a pool of any supported type to the registry.

        Dispatches to the appropriate add method based on pool type.

        Args:
            pool: Pool to add (V2, V3, Balancer weighted/stable, or limit order)

        Raises:
            TypeError: If pool type is not supported
        """
        if isinstance(pool, UniswapV2Pool):
            self.add_pool(pool)
        elif isinstance(pool, UniswapV3Pool):
            self.add_v3_pool(pool)
        elif isinstance(pool, BalancerWeightedPool):
            self.add_weighted_pool(pool)
        elif isinstance(pool, BalancerStablePool):
            self.add_stable_pool(pool)
        elif isinstance(pool, LimitOrderPool):
            self.add_limit_order(pool)
        else:
            raise TypeError(f"Unknown pool type: {type(pool)}")

    def add_pool(self, pool: UniswapV2Pool) -> None:
        """Add a V2 pool to the registry.

        Args:
            pool: The pool to add. If a pool for this token pair already exists,
                  it will be replaced.
        """
        token0_norm = normalize_address(pool.token0)
        token1_norm = normalize_address(pool.token1)
        pair_key = frozenset([token0_norm, token1_norm])
        if pair_key in self._pools:
            logger.debug(
                "v2_pool_replaced",
                pool=pool.address[-8:],
                token0=token0_norm[-8:],
                token1=token1_norm[-8:],
            )
        self._pools[pair_key] = pool
        self._invalidate_pathfinder()

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
        pair_key = (token0_norm, token1_norm)

        # Check if this exact pool (same fee tier) already exists
        if key in self._v3_pools:
            # Replace existing pool (same behavior as before)
            self._v3_pools[key] = pool
            # Update secondary index - replace the old pool with new one
            if pair_key in self._v3_pools_by_pair:
                pools = self._v3_pools_by_pair[pair_key]
                for i, p in enumerate(pools):
                    if p.fee == pool.fee:
                        pools[i] = pool
                        break
        else:
            # New pool
            self._v3_pools[key] = pool
            # Update secondary index
            if pair_key not in self._v3_pools_by_pair:
                self._v3_pools_by_pair[pair_key] = []
            self._v3_pools_by_pair[pair_key].append(pool)

        self._invalidate_pathfinder()

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

        pair_key = (token_a_norm, token_b_norm)
        return self._v3_pools_by_pair.get(pair_key, [])

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

        tokens = [normalize_address(r.token) for r in pool.reserves]
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i + 1 :]:
                # Canonical pair ordering
                pair = (min(t1, t2), max(t1, t2))
                if pair not in self._weighted_pools:
                    self._weighted_pools[pair] = []
                self._weighted_pools[pair].append(pool)
        self._invalidate_pathfinder()

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

        tokens = [normalize_address(r.token) for r in pool.reserves]
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i + 1 :]:
                # Canonical pair ordering
                pair = (min(t1, t2), max(t1, t2))
                if pair not in self._stable_pools:
                    self._stable_pools[pair] = []
                self._stable_pools[pair].append(pool)
        self._invalidate_pathfinder()

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
        self._invalidate_pathfinder()

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
        # Normalize once, use for all lookups
        token_a_norm = normalize_address(token_a)
        token_b_norm = normalize_address(token_b)

        # Check cache first (directional key for limit orders)
        cache_key = (token_a_norm, token_b_norm)
        if cache_key in self._pools_for_pair_cache:
            return self._pools_for_pair_cache[cache_key]

        pools: list[AnyPool] = []

        # Add V2 pool if exists (bidirectional) - use frozenset for order-independent lookup
        v2_key = frozenset([token_a_norm, token_b_norm])
        v2_pool = self._pools.get(v2_key)
        if v2_pool is not None:
            pools.append(v2_pool)

        # Add all V3 pools (bidirectional) - canonical ordering for V3
        if token_a_norm > token_b_norm:
            v3_pair_key = (token_b_norm, token_a_norm)
        else:
            v3_pair_key = (token_a_norm, token_b_norm)
        pools.extend(self._v3_pools_by_pair.get(v3_pair_key, []))

        # Add all Balancer weighted pools (bidirectional) - canonical ordering
        balancer_pair = (min(token_a_norm, token_b_norm), max(token_a_norm, token_b_norm))
        pools.extend(self._weighted_pools.get(balancer_pair, []))

        # Add all Balancer stable pools (bidirectional)
        pools.extend(self._stable_pools.get(balancer_pair, []))

        # Add all limit orders (unidirectional: token_a -> token_b only)
        limit_key = (token_a_norm, token_b_norm)
        pools.extend(self._limit_orders.get(limit_key, []))

        # Cache the result
        self._pools_for_pair_cache[cache_key] = pools
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

    # Pathfinding methods (delegate to PathFinder)

    def get_all_candidate_paths(
        self,
        token_in: str,
        token_out: str,
        max_hops: int = 3,
    ) -> list[list[str]]:
        """Generate all candidate paths from token_in to token_out.

        Delegates to PathFinder.find_all_paths().

        Args:
            token_in: Starting token address
            token_out: Target token address
            max_hops: Maximum number of swaps allowed (default 2)

        Returns:
            List of paths, where each path is a list of token addresses.
        """
        return self.pathfinder.find_all_paths(token_in, token_out, max_hops)

    def find_path(
        self,
        token_in: str,
        token_out: str,
        max_hops: int = 3,
    ) -> list[str] | None:
        """Find a path from token_in to token_out through available pools.

        Delegates to PathFinder.find_shortest_path().

        Args:
            token_in: Starting token address
            token_out: Target token address
            max_hops: Maximum number of swaps allowed (default 2)

        Returns:
            List of token addresses representing the path, or None if no path found.
        """
        return self.pathfinder.find_shortest_path(token_in, token_out, max_hops)

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

    def get_all_token_pairs(self) -> set[tuple[str, str]]:
        """Get all unique token pairs with available liquidity.

        Returns bidirectional pairs for AMM pools and unidirectional pairs
        for limit orders. Each pair is (token_a, token_b) with canonical
        ordering (token_a < token_b) for bidirectional pools.

        Returns:
            Set of (token_a, token_b) tuples representing tradeable pairs
        """
        pairs: set[tuple[str, str]] = set()

        # V2 pools (bidirectional, canonical ordering)
        for token_pair in self._pools:
            tokens = sorted(token_pair)
            pairs.add((tokens[0], tokens[1]))

        # V3 pools (bidirectional, already canonical)
        for token_a, token_b, _fee in self._v3_pools:
            pairs.add((token_a, token_b))

        # Weighted pools (bidirectional, already canonical)
        for pair in self._weighted_pools:
            pairs.add(pair)

        # Stable pools (bidirectional, already canonical)
        for pair in self._stable_pools:
            pairs.add(pair)

        # Limit orders (unidirectional: taker -> maker)
        for pair in self._limit_orders:
            pairs.add(pair)

        return pairs

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
