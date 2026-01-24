"""Multi-pair price coordination strategy.

This strategy closes the gap between crossing orders and matched orders by
optimizing prices jointly across overlapping token pairs.

Key insight: When token X appears in pairs (X,Y) and (X,Z), we need a unified
price for X that works for both pairs. This module finds connected components
of overlapping pairs and uses LP optimization to find optimal fills.

Algorithm:
1. Find connected components of token pairs (Union-Find)
2. For each component, enumerate price candidates from order limits and AMM prices
3. Solve LP for each price combination to find max fill
4. Select best result per component

Time budget per component: ~50-100ms
- Component detection: <1ms
- Price enumeration: <5ms
- LP solve per price: 5-8ms (max 100 combinations)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import product
from typing import TYPE_CHECKING

import structlog

from solver.amm.uniswap_v2 import UniswapV2, uniswap_v2
from solver.fees.price_estimation import get_token_info
from solver.models.auction import AuctionInstance, Order
from solver.models.order_groups import OrderGroup, find_cow_opportunities
from solver.models.types import normalize_address
from solver.pools import PoolRegistry, build_registry_from_liquidity
from solver.routing.router import SingleOrderRouter
from solver.strategies.base import OrderFill, StrategyResult
from solver.strategies.double_auction import get_limit_price

if TYPE_CHECKING:
    from solver.amm.balancer import BalancerStableAMM, BalancerWeightedAMM
    from solver.amm.limit_order import LimitOrderAMM
    from solver.amm.uniswap_v3 import UniswapV3AMM

logger = structlog.get_logger()


# =============================================================================
# Phase 1: Connected Component Detection
# =============================================================================


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


def find_token_components(cow_groups: list[OrderGroup]) -> list[list[OrderGroup]]:
    """Find connected components of overlapping token pairs using Union-Find.

    Two pairs are connected if they share a token. For example:
    - (WETH, USDC) and (WETH, DAI) are connected via WETH
    - (USDC, DAI) would also connect to the above component

    Args:
        cow_groups: List of OrderGroups with CoW potential

    Returns:
        List of components, each component is a list of OrderGroups
    """
    if not cow_groups:
        return []

    # Map tokens to group indices
    token_to_groups: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(cow_groups):
        token_a = normalize_address(group.token_a)
        token_b = normalize_address(group.token_b)
        token_to_groups[token_a].append(i)
        token_to_groups[token_b].append(i)

    # Union groups that share tokens
    uf = UnionFind()
    for indices in token_to_groups.values():
        if len(indices) > 1:
            # All groups sharing this token should be in same component
            first = indices[0]
            for other in indices[1:]:
                uf.union(first, other)

    # Group by component root
    components: dict[int, list[OrderGroup]] = defaultdict(list)
    for i, group in enumerate(cow_groups):
        root = uf.find(i)
        components[root].append(group)

    # Sort components by total order count (largest first)
    result = list(components.values())
    result.sort(key=lambda c: sum(g.order_count for g in c), reverse=True)

    return result


# =============================================================================
# Phase 2: Price Enumeration
# =============================================================================


@dataclass
class PriceCandidates:
    """Price ratio candidates for token pairs within a component.

    For each edge in the token graph (spanning tree), we store candidate
    ratios derived from order limits and AMM spot prices.
    """

    # Map (token_a, token_b) -> list of price ratios (token_b / token_a)
    pair_ratios: dict[tuple[str, str], list[Decimal]] = field(default_factory=dict)

    def add_ratio(self, token_a: str, token_b: str, ratio: Decimal) -> None:
        """Add a price ratio for a token pair."""
        key = (token_a, token_b)
        if key not in self.pair_ratios:
            self.pair_ratios[key] = []
        if ratio not in self.pair_ratios[key]:
            self.pair_ratios[key].append(ratio)

    def get_ratios(self, token_a: str, token_b: str) -> list[Decimal]:
        """Get price ratios for a token pair."""
        return self.pair_ratios.get((token_a, token_b), [])


def build_price_candidates(
    component: list[OrderGroup],
    router: SingleOrderRouter | None,
    auction: AuctionInstance,
) -> PriceCandidates:
    """Build price candidates for a component from order limits and AMM prices.

    Sources:
    1. Order limit prices (via get_limit_price from double_auction.py)
    2. AMM spot prices (via router.get_reference_price)

    Args:
        component: List of OrderGroups in the component
        router: Router for AMM price queries (may be None)
        auction: Auction instance for token decimals

    Returns:
        PriceCandidates with ratios for each pair
    """
    candidates = PriceCandidates()

    for group in component:
        token_a = normalize_address(group.token_a)
        token_b = normalize_address(group.token_b)

        # Collect limit prices from sellers of A (price in B/A)
        for order in group.sellers_of_a:
            limit = get_limit_price(order, is_selling_a=True)
            if limit is not None and limit > 0:
                candidates.add_ratio(token_a, token_b, limit)

        # Collect limit prices from sellers of B (also in B/A)
        for order in group.sellers_of_b:
            limit = get_limit_price(order, is_selling_a=False)
            if limit is not None and limit > 0:
                candidates.add_ratio(token_a, token_b, limit)

        # Add AMM spot price if available
        if router is not None:
            token_a_info = get_token_info(auction, group.token_a)
            decimals = (
                18
                if token_a_info is None or token_a_info.decimals is None
                else token_a_info.decimals
            )
            amm_price = router.get_reference_price(
                group.token_a, group.token_b, token_in_decimals=decimals
            )
            if amm_price is not None and amm_price > 0:
                candidates.add_ratio(token_a, token_b, amm_price)

    return candidates


def build_token_graph(component: list[OrderGroup]) -> dict[str, set[str]]:
    """Build undirected token graph from component.

    Args:
        component: List of OrderGroups

    Returns:
        Adjacency list representation of token graph
    """
    graph: dict[str, set[str]] = defaultdict(set)
    for group in component:
        token_a = normalize_address(group.token_a)
        token_b = normalize_address(group.token_b)
        graph[token_a].add(token_b)
        graph[token_b].add(token_a)
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


def enumerate_price_combinations(
    candidates: PriceCandidates,
    spanning_tree: list[tuple[str, str]],
    max_combinations: int = 100,
) -> list[dict[str, Decimal]]:
    """Enumerate price combinations by propagating through spanning tree.

    For each combination of edge ratios, fix reference token price = 1
    and propagate prices through the tree.

    Args:
        candidates: Price candidates for each pair
        spanning_tree: Spanning tree edges
        max_combinations: Maximum combinations to enumerate

    Returns:
        List of price dictionaries (token -> price)
    """
    if not spanning_tree:
        return []

    # Get candidate ratios for each edge
    edge_ratios: list[list[Decimal]] = []
    for parent, child in spanning_tree:
        # Try both orderings to find the ratio
        ratios = candidates.get_ratios(parent, child)
        if not ratios:
            # Try reverse ordering and invert
            reverse_ratios = candidates.get_ratios(child, parent)
            if reverse_ratios:
                ratios = [Decimal(1) / r for r in reverse_ratios if r > 0]

        if not ratios:
            # No candidates - use a default ratio of 1
            ratios = [Decimal(1)]

        edge_ratios.append(ratios)

    # Limit total combinations
    total_combinations = 1
    for ratios in edge_ratios:
        total_combinations *= len(ratios)
        if total_combinations > max_combinations:
            break

    # If too many combinations, sample by taking first/last/middle from each edge
    if total_combinations > max_combinations:
        sampled_ratios: list[list[Decimal]] = []
        for ratios in edge_ratios:
            if len(ratios) <= 3:
                sampled_ratios.append(ratios)
            else:
                # Take first, last, and middle
                sorted_ratios = sorted(ratios)
                sampled_ratios.append(
                    [
                        sorted_ratios[0],
                        sorted_ratios[len(sorted_ratios) // 2],
                        sorted_ratios[-1],
                    ]
                )
        edge_ratios = sampled_ratios

    # Generate all combinations
    results: list[dict[str, Decimal]] = []
    reference_token = spanning_tree[0][0]

    for combo in product(*edge_ratios):
        prices: dict[str, Decimal] = {reference_token: Decimal(1)}

        # Propagate prices through tree
        for i, (parent, child) in enumerate(spanning_tree):
            ratio = combo[i]
            if parent in prices:
                # price[child] = price[parent] * ratio
                # (ratio is child/parent, i.e., B/A)
                prices[child] = prices[parent] * ratio
            elif child in prices and ratio > 0:
                # price[parent] = price[child] / ratio
                prices[parent] = prices[child] / ratio

        results.append(prices)
        if len(results) >= max_combinations:
            break

    return results


# =============================================================================
# Phase 3: LP Solver for Multi-Pair Fills
# =============================================================================


@dataclass
class LPResult:
    """Result of LP optimization for multi-pair fills.

    Attributes:
        fills: Order fills with amounts
        total_volume: Total matched volume (sum of fill amounts)
        prices: Token prices used for this solution
    """

    fills: list[OrderFill]
    total_volume: int
    prices: dict[str, Decimal]


def solve_fills_at_prices(
    component: list[OrderGroup],
    prices: dict[str, Decimal],
) -> LPResult | None:
    """Solve for optimal fills given fixed token prices using LP.

    LP Formulation:
    - Variables: x_i = fill ratio for each eligible order (0 to 1)
    - Objective: maximize sum(x_i * sell_amount_i)
    - Constraints:
      - Fill bounds: 0 <= x_i <= 1
      - Conservation: sum(sells of t) >= sum(buys of t) for each token t
        (we require conservation with tolerance for rounding)

    An order is eligible if its limit price is satisfied by the given prices.

    Args:
        component: List of OrderGroups to optimize
        prices: Token prices (token -> price)

    Returns:
        LPResult if solution found, None if no feasible solution
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        logger.warning("scipy not available, skipping LP optimization")
        return None

    # Collect all orders with their eligibility
    all_orders: list[tuple[Order, bool, str, str]] = []  # (order, is_selling_a, token_a, token_b)

    for group in component:
        token_a = normalize_address(group.token_a)
        token_b = normalize_address(group.token_b)

        for order in group.sellers_of_a:
            all_orders.append((order, True, token_a, token_b))
        for order in group.sellers_of_b:
            all_orders.append((order, False, token_a, token_b))

    if not all_orders:
        return None

    # Check which orders are eligible at given prices
    eligible_orders: list[tuple[Order, bool, str, str, Decimal]] = []  # + limit_price

    for order, is_selling_a, token_a, token_b in all_orders:
        limit = get_limit_price(order, is_selling_a)
        if limit is None:
            continue

        # Get prices for both tokens
        price_a = prices.get(token_a)
        price_b = prices.get(token_b)
        if price_a is None or price_b is None or price_a <= 0:
            continue

        # Current price ratio B/A at given prices
        current_ratio = price_b / price_a

        # Check if order's limit is satisfied
        # Sellers of A want at least `limit` B per A -> current_ratio >= limit
        # Sellers of B want to pay at most `limit` B per A -> current_ratio <= limit
        if is_selling_a:
            if current_ratio >= limit:
                eligible_orders.append((order, is_selling_a, token_a, token_b, limit))
        else:
            if current_ratio <= limit:
                eligible_orders.append((order, is_selling_a, token_a, token_b, limit))

    if not eligible_orders:
        return None

    n_orders = len(eligible_orders)

    # Get all tokens involved
    tokens: set[str] = set()
    for _, _, token_a, token_b, _ in eligible_orders:
        tokens.add(token_a)
        tokens.add(token_b)
    token_list = sorted(tokens)

    # Objective: maximize sum(x_i * sell_amount_i)
    # linprog minimizes, so we negate
    c = []
    for order, _, _, _, _ in eligible_orders:
        c.append(-order.sell_amount_int)

    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1) for _ in range(n_orders)]

    # Conservation constraints: for each token t,
    # sum(buys of t) - sum(sells of t) <= 0  (i.e., sells >= buys)
    # This allows surplus but not deficit
    # Use A_ub @ x <= b_ub format
    A_ub = []
    b_ub = []

    for token in token_list:
        row = [0.0] * n_orders
        for i, (order, _is_selling_a, _token_a, _token_b, _) in enumerate(eligible_orders):
            sell_token = normalize_address(order.sell_token)
            buy_token = normalize_address(order.buy_token)

            if sell_token == token:
                # This order sells `token` -> negative (we want sells >= buys)
                row[i] -= order.sell_amount_int
            if buy_token == token:
                # This order buys `token` -> positive
                row[i] += order.buy_amount_int

        # Constraint: buys - sells <= 0, i.e., sells >= buys
        A_ub.append(row)
        b_ub.append(0.0)

    # Solve LP
    try:
        result = linprog(
            c,
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            bounds=bounds,
            method="highs",
        )
    except Exception as e:
        logger.debug("lp_solve_error", error=str(e))
        return None

    if not result.success or result.x is None:
        return None

    # Convert solution to fills
    fills: list[OrderFill] = []
    total_volume = 0

    for i, (order, _is_selling_a, _token_a, _token_b, _) in enumerate(eligible_orders):
        fill_ratio = result.x[i]
        if fill_ratio < 0.001:  # Skip negligible fills
            continue

        # Calculate fill amounts
        sell_filled = int(order.sell_amount_int * fill_ratio)
        buy_filled = int(order.buy_amount_int * fill_ratio)

        if sell_filled <= 0 or buy_filled <= 0:
            continue

        # Check fill-or-kill constraint
        if not order.partially_fillable and fill_ratio < 0.999:
            continue

        fills.append(
            OrderFill(
                order=order,
                sell_filled=sell_filled,
                buy_filled=buy_filled,
            )
        )
        total_volume += sell_filled

    if not fills:
        return None

    return LPResult(
        fills=fills,
        total_volume=total_volume,
        prices=prices,
    )


# =============================================================================
# Phase 4: Strategy Integration
# =============================================================================


class MultiPairCowStrategy:
    """Strategy that coordinates prices across multiple overlapping token pairs.

    This replaces HybridCowStrategy with multi-pair optimization to unlock
    more CoW matches when tokens overlap across pairs.

    Algorithm:
    1. Find CoW opportunities (reuse find_cow_opportunities)
    2. Find connected components of overlapping pairs
    3. For each component:
       a. Enumerate price combinations (max 100)
       b. Solve LP for each price combination
       c. Keep best result (highest volume)
    4. Combine results from all components

    Args:
        amm: AMM implementation for swap math. Defaults to UniswapV2.
        router: Injected router for testing. If provided, used directly.
        v3_amm: UniswapV3 AMM for V3 pool routing. If None, V3 pools are skipped.
        weighted_amm: Balancer weighted AMM. If None, weighted pools are skipped.
        stable_amm: Balancer stable AMM. If None, stable pools are skipped.
        limit_order_amm: 0x limit order AMM. If None, limit orders are skipped.
        max_combinations: Maximum price combinations per component (default 100).
        max_components: Maximum components to process (default 10).
    """

    def __init__(
        self,
        amm: UniswapV2 | None = None,
        router: SingleOrderRouter | None = None,
        v3_amm: UniswapV3AMM | None = None,
        weighted_amm: BalancerWeightedAMM | None = None,
        stable_amm: BalancerStableAMM | None = None,
        limit_order_amm: LimitOrderAMM | None = None,
        max_combinations: int = 100,
        max_components: int = 10,
    ) -> None:
        """Initialize with optional AMM components for price queries."""
        self.amm = amm if amm is not None else uniswap_v2
        self._injected_router = router
        self.v3_amm = v3_amm
        self.weighted_amm = weighted_amm
        self.stable_amm = stable_amm
        self.limit_order_amm = limit_order_amm
        self.max_combinations = max_combinations
        self.max_components = max_components

    def _get_router(self, pool_registry: PoolRegistry) -> SingleOrderRouter:
        """Get the router to use for AMM price queries."""
        if self._injected_router is not None:
            return self._injected_router
        return SingleOrderRouter(
            amm=self.amm,
            pool_registry=pool_registry,
            v3_amm=self.v3_amm,
            weighted_amm=self.weighted_amm,
            stable_amm=self.stable_amm,
            limit_order_amm=self.limit_order_amm,
        )

    def try_solve(self, auction: AuctionInstance) -> StrategyResult | None:
        """Try to find CoW matches using multi-pair optimization.

        Args:
            auction: The auction to solve

        Returns:
            StrategyResult with CoW fills and remainders, or None if no matches
        """
        if auction.order_count < 2:
            return None

        # Find token pairs with CoW potential
        cow_groups = find_cow_opportunities(auction.orders)

        if not cow_groups:
            logger.debug(
                "multi_pair_no_opportunities",
                order_count=auction.order_count,
            )
            return None

        # Build pool registry for AMM price queries
        pool_registry = build_registry_from_liquidity(auction.liquidity)
        router = self._get_router(pool_registry)

        # Find connected components
        components = find_token_components(cow_groups)

        logger.debug(
            "multi_pair_components",
            total_groups=len(cow_groups),
            num_components=len(components),
            component_sizes=[len(c) for c in components[:5]],
        )

        # Process each component
        all_fills: list[OrderFill] = []
        all_prices: dict[str, str] = {}
        processed_uids: set[str] = set()

        for comp_idx, component in enumerate(components[: self.max_components]):
            result = self._solve_component(component, router, auction)

            if result is None:
                continue

            # Collect fills (avoiding duplicates)
            for fill in result.fills:
                if fill.order.uid not in processed_uids:
                    all_fills.append(fill)
                    processed_uids.add(fill.order.uid)

            # Collect prices (convert Decimal to str)
            for token, price in result.prices.items():
                all_prices[token] = str(int(price * Decimal(10**18)))

            logger.info(
                "multi_pair_component_solved",
                component_idx=comp_idx,
                groups=len(component),
                fills=len(result.fills),
                volume=result.total_volume,
            )

        if not all_fills:
            return None

        # Compute remainder orders
        filled_uids = {fill.order.uid for fill in all_fills}
        remainder_orders: list[Order] = []

        # Add fully unmatched orders
        for order in auction.orders:
            if order.uid not in filled_uids:
                remainder_orders.append(order)

        # Add partial remainders from fills
        for fill in all_fills:
            remainder = fill.get_remainder_order()
            if remainder:
                remainder_orders.append(remainder)

        # Normalize prices for solution format
        # The prices need to maintain token conservation
        normalized_prices = self._normalize_prices(all_fills)

        logger.info(
            "multi_pair_complete",
            total_fills=len(all_fills),
            total_remainders=len(remainder_orders),
            components_processed=min(len(components), self.max_components),
        )

        return StrategyResult(
            fills=all_fills,
            interactions=[],  # CoW matches have no AMM interactions
            prices=normalized_prices,
            gas=0,  # No on-chain swaps
            remainder_orders=remainder_orders,
        )

    def _solve_component(
        self,
        component: list[OrderGroup],
        router: SingleOrderRouter,
        auction: AuctionInstance,
    ) -> LPResult | None:
        """Solve a single connected component.

        For single-pair components, uses the proven double auction algorithm.
        For multi-pair components, tries multiple price combinations via LP.

        Args:
            component: List of OrderGroups in the component
            router: Router for AMM price queries
            auction: Auction instance

        Returns:
            Best LPResult or None if no valid solution
        """
        # For single-pair components, use the proven double auction algorithm
        if len(component) == 1:
            return self._solve_single_pair(component[0], router, auction)

        # For multi-pair components, use LP optimization
        # But limit complexity for large components
        total_orders = sum(g.order_count for g in component)
        if total_orders > 500:
            # Too many orders for full LP - but try the largest single pairs
            # Sort by order count and process top pairs individually
            logger.debug(
                "multi_pair_large_component_fallback",
                groups=len(component),
                orders=total_orders,
            )
            # Process top 10 largest pairs individually and combine results
            sorted_groups = sorted(component, key=lambda g: g.order_count, reverse=True)
            combined_fills: list[OrderFill] = []
            combined_volume = 0
            combined_prices: dict[str, Decimal] = {}
            matched_uids: set[str] = set()

            for group in sorted_groups[:10]:
                result = self._solve_single_pair(group, router, auction)
                if result is not None:
                    # Add fills that don't overlap with already matched orders
                    for fill in result.fills:
                        if fill.order.uid not in matched_uids:
                            combined_fills.append(fill)
                            matched_uids.add(fill.order.uid)
                            combined_volume += fill.sell_filled
                    # Merge prices
                    combined_prices.update(result.prices)

            if not combined_fills:
                return None

            return LPResult(
                fills=combined_fills,
                total_volume=combined_volume,
                prices=combined_prices,
            )

        # Build price candidates
        candidates = build_price_candidates(component, router, auction)

        # Build token graph and spanning tree
        token_graph = build_token_graph(component)
        spanning_tree = find_spanning_tree(token_graph)

        # Enumerate price combinations (limit for performance)
        max_combos = min(self.max_combinations, 20)  # Limit to 20 for speed
        price_combinations = enumerate_price_combinations(
            candidates,
            spanning_tree,
            max_combinations=max_combos,
        )

        if not price_combinations:
            logger.debug(
                "multi_pair_no_price_combinations",
                groups=len(component),
            )
            return None

        # Try each price combination
        best_result: LPResult | None = None

        for prices in price_combinations:
            result = solve_fills_at_prices(component, prices)

            if result is None:
                continue

            if best_result is None or result.total_volume > best_result.total_volume:
                best_result = result

            # Early termination if we found a good result for large components
            if best_result is not None and best_result.total_volume > 0 and len(component) > 5:
                break

        return best_result

    def _solve_single_pair(
        self,
        group: OrderGroup,
        router: SingleOrderRouter,
        auction: AuctionInstance,
    ) -> LPResult | None:
        """Solve a single token pair using double auction.

        This reuses the proven run_hybrid_auction algorithm.
        EBBO constraint: Only returns matches where clearing price >= AMM price.

        Args:
            group: Single OrderGroup to solve
            router: Router for AMM price queries
            auction: Auction instance

        Returns:
            LPResult if EBBO-compliant matches found, None otherwise
        """
        from solver.strategies.double_auction import run_hybrid_auction

        # Get token decimals for proper probe amount scaling
        token_a_info = get_token_info(auction, group.token_a)
        decimals = (
            18 if token_a_info is None or token_a_info.decimals is None else token_a_info.decimals
        )

        # Get AMM reference price (this is the EBBO benchmark)
        amm_price = router.get_reference_price(
            group.token_a, group.token_b, token_in_decimals=decimals
        )

        # Run hybrid auction
        result = run_hybrid_auction(group, amm_price=amm_price)

        if not result.cow_matches:
            return None

        # EBBO constraint: verify sellers receive at least AMM equivalent (zero tolerance)
        # For sellers of A: they receive total_cow_b tokens B
        # AMM would give them: int(total_cow_a * amm_price)
        # EBBO satisfied if: total_cow_b >= int(total_cow_a * amm_price)
        # This integer comparison properly accounts for rounding
        if amm_price is not None and result.total_cow_a > 0:
            amm_equivalent = int(Decimal(result.total_cow_a) * amm_price)
            if result.total_cow_b < amm_equivalent:
                clearing_rate = Decimal(result.total_cow_b) / Decimal(result.total_cow_a)
                logger.debug(
                    "multi_pair_ebbo_violation",
                    token_a=group.token_a[-8:],
                    token_b=group.token_b[-8:],
                    clearing_rate=float(clearing_rate),
                    amm_price=float(amm_price),
                    cow_b=result.total_cow_b,
                    amm_equivalent=amm_equivalent,
                    deficit=amm_equivalent - result.total_cow_b,
                )
                return None  # Reject this match - would violate EBBO

        # Convert matches to fills
        fills: list[OrderFill] = []
        total_volume = 0

        for match in result.cow_matches:
            # Seller (sells A, receives B)
            fill_seller = OrderFill(
                order=match.seller,
                sell_filled=match.amount_a,
                buy_filled=match.amount_b,
            )
            fills.append(fill_seller)
            total_volume += match.amount_a

            # Buyer (sells B, receives A)
            fill_buyer = OrderFill(
                order=match.buyer,
                sell_filled=match.amount_b,
                buy_filled=match.amount_a,
            )
            fills.append(fill_buyer)
            total_volume += match.amount_b

        # Build prices from the clearing result
        token_a = normalize_address(group.token_a)
        token_b = normalize_address(group.token_b)
        prices = {
            token_a: Decimal(str(result.total_cow_b)),
            token_b: Decimal(str(result.total_cow_a)),
        }

        return LPResult(
            fills=fills,
            total_volume=total_volume,
            prices=prices,
        )

    def _normalize_prices(self, fills: list[OrderFill]) -> dict[str, str]:
        """Normalize prices to maintain token conservation.

        For CoW matching, prices must satisfy:
        sum(price[sell_token] * sell_amount) = sum(price[buy_token] * buy_amount)

        This is achieved by setting prices based on actual fill amounts:
        price[token] = total amount of token received across all trades

        Args:
            fills: List of order fills

        Returns:
            Normalized price dictionary
        """
        # Track total sells and buys per token
        token_sells: dict[str, int] = defaultdict(int)
        token_buys: dict[str, int] = defaultdict(int)

        for fill in fills:
            sell_token = normalize_address(fill.order.sell_token)
            buy_token = normalize_address(fill.order.buy_token)
            token_sells[sell_token] += fill.sell_filled
            token_buys[buy_token] += fill.buy_filled

        # For balanced CoW matching:
        # What's sold of a token = what's bought of that token
        # price[token] = amount that flows through the settlement
        prices: dict[str, str] = {}

        for token in set(token_sells.keys()) | set(token_buys.keys()):
            # Use the buy amount as the price (what flows to buyers)
            # This ensures price[sell_token] * sell = price[buy_token] * buy
            prices[token] = str(max(token_sells.get(token, 0), token_buys.get(token, 0)))

        return prices


__all__ = [
    "UnionFind",
    "find_token_components",
    "PriceCandidates",
    "build_price_candidates",
    "build_token_graph",
    "find_spanning_tree",
    "enumerate_price_combinations",
    "LPResult",
    "solve_fills_at_prices",
    "MultiPairCowStrategy",
]
