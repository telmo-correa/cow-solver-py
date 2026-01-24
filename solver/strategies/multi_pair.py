"""Multi-pair price coordination strategy.

This strategy finds CoW matches by:
1. Processing bidirectional pairs via double auction
2. Finding cycles in the order graph and solving them

The strategy coordinates prices across overlapping token pairs and cycles
to maximize matched volume while maintaining EBBO compliance.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
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
from solver.strategies.components import find_token_components
from solver.strategies.graph import OrderGraph, find_spanning_tree
from solver.strategies.pricing import (
    LPResult,
    build_price_candidates,
    build_token_graph_from_groups,
    enumerate_price_combinations,
    solve_fills_at_prices,
)
from solver.strategies.settlement import solve_cycle

if TYPE_CHECKING:
    from solver.amm.balancer import BalancerStableAMM, BalancerWeightedAMM
    from solver.amm.limit_order import LimitOrderAMM
    from solver.amm.uniswap_v3 import UniswapV3AMM

logger = structlog.get_logger()


class MultiPairCowStrategy:
    """Strategy that coordinates prices across multiple overlapping token pairs.

    When use_generalized=True (default), this strategy:
    1. Processes bidirectional pairs via double auction (Phase 1)
    2. Finds cycles in the order graph and solves them (Phase 2)

    When use_generalized=False, only Phase 1 (bidirectional pairs) is used.

    Args:
        amm: AMM implementation for swap math. Defaults to UniswapV2.
        router: Injected router for testing. If provided, used directly.
        v3_amm: UniswapV3 AMM for V3 pool routing.
        weighted_amm: Balancer weighted AMM.
        stable_amm: Balancer stable AMM.
        limit_order_amm: 0x limit order AMM.
        max_combinations: Maximum price combinations per component.
        max_components: Maximum components to process.
        max_tokens: Maximum tokens per component for cycle detection.
        use_generalized: If True, also find cycles beyond bidirectional pairs.
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
        max_tokens: int = 4,
        use_generalized: bool = True,
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
        self.max_tokens = max_tokens
        self.use_generalized = use_generalized

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

        pool_registry = build_registry_from_liquidity(auction.liquidity)
        router = self._get_router(pool_registry)

        if self.use_generalized:
            return self._solve_with_cycles(auction, router)
        else:
            return self._solve_bidirectional_only(auction, router)

    def _solve_with_cycles(
        self,
        auction: AuctionInstance,
        router: SingleOrderRouter,
    ) -> StrategyResult | None:
        """Solve using both bidirectional pairs and cycle detection."""
        all_fills: list[OrderFill] = []
        all_prices: dict[str, str] = {}
        processed_uids: set[str] = set()

        # Phase 1: Process bidirectional pairs
        cow_groups = find_cow_opportunities(auction.orders)
        if cow_groups:
            components = find_token_components(cow_groups)
            for component in components[: self.max_components]:
                result = self._solve_component(component, router, auction)
                if result:
                    self._collect_fills(result, all_fills, all_prices, processed_uids)

        logger.debug("multi_pair_phase1_complete", fills=len(all_fills))

        # Phase 2: Find and solve cycles
        remaining_orders = [o for o in auction.orders if o.uid not in processed_uids]
        if remaining_orders and self.max_tokens >= 3:
            cycle_fills = self._solve_cycles(remaining_orders, router, auction, processed_uids)
            for fill in cycle_fills:
                if fill.order.uid not in processed_uids:
                    all_fills.append(fill)
                    processed_uids.add(fill.order.uid)
            logger.debug("multi_pair_phase2_complete", cycle_fills=len(cycle_fills))

        if not all_fills:
            return None

        return self._build_result(all_fills, auction)

    def _solve_bidirectional_only(
        self,
        auction: AuctionInstance,
        router: SingleOrderRouter,
    ) -> StrategyResult | None:
        """Solve using only bidirectional pairs."""
        cow_groups = find_cow_opportunities(auction.orders)
        if not cow_groups:
            return None

        all_fills: list[OrderFill] = []
        all_prices: dict[str, str] = {}
        processed_uids: set[str] = set()

        components = find_token_components(cow_groups)
        for component in components[: self.max_components]:
            result = self._solve_component(component, router, auction)
            if result:
                self._collect_fills(result, all_fills, all_prices, processed_uids)

        if not all_fills:
            return None

        return self._build_result(all_fills, auction)

    def _solve_component(
        self,
        component: list[OrderGroup],
        router: SingleOrderRouter,
        auction: AuctionInstance,
    ) -> LPResult | None:
        """Solve a single connected component of OrderGroups."""
        if len(component) == 1:
            return self._solve_single_pair(component[0], router, auction)

        # For multi-pair components, use LP optimization
        total_orders = sum(g.order_count for g in component)
        if total_orders > 500:
            return self._solve_large_component(component, router, auction)

        candidates = build_price_candidates(component, router, auction)
        token_graph = build_token_graph_from_groups(component)
        spanning_tree = find_spanning_tree(token_graph)

        price_combinations = enumerate_price_combinations(
            candidates, spanning_tree, max_combinations=min(self.max_combinations, 20)
        )

        if not price_combinations:
            return None

        best_result: LPResult | None = None
        for prices in price_combinations:
            result = solve_fills_at_prices(component, prices)
            if result and (best_result is None or result.total_volume > best_result.total_volume):
                best_result = result
                if len(component) > 5:
                    break  # Early termination for large components

        return best_result

    def _solve_large_component(
        self,
        component: list[OrderGroup],
        router: SingleOrderRouter,
        auction: AuctionInstance,
    ) -> LPResult | None:
        """Handle large components by processing top pairs individually."""
        sorted_groups = sorted(component, key=lambda g: g.order_count, reverse=True)
        combined_fills: list[OrderFill] = []
        combined_volume = 0
        combined_prices: dict[str, Decimal] = {}
        matched_uids: set[str] = set()

        for group in sorted_groups[:10]:
            result = self._solve_single_pair(group, router, auction)
            if result:
                for fill in result.fills:
                    if fill.order.uid not in matched_uids:
                        combined_fills.append(fill)
                        matched_uids.add(fill.order.uid)
                        combined_volume += fill.sell_filled
                combined_prices.update(result.prices)

        if not combined_fills:
            return None

        return LPResult(fills=combined_fills, total_volume=combined_volume, prices=combined_prices)

    def _solve_single_pair(
        self,
        group: OrderGroup,
        router: SingleOrderRouter,
        auction: AuctionInstance,
    ) -> LPResult | None:
        """Solve a single token pair using double auction with EBBO validation."""
        from solver.strategies.double_auction import run_hybrid_auction

        token_a_info = get_token_info(auction, group.token_a)
        decimals = (
            18 if token_a_info is None or token_a_info.decimals is None else token_a_info.decimals
        )

        amm_price = router.get_reference_price(
            group.token_a, group.token_b, token_in_decimals=decimals
        )
        result = run_hybrid_auction(group, amm_price=amm_price)

        if not result.cow_matches:
            return None

        # EBBO validation (zero tolerance)
        if amm_price is not None and result.total_cow_a > 0:
            amm_equivalent = int(Decimal(result.total_cow_a) * amm_price)
            if result.total_cow_b < amm_equivalent:
                return None

        # Convert matches to fills
        fills: list[OrderFill] = []
        total_volume = 0

        for match in result.cow_matches:
            fills.append(
                OrderFill(order=match.seller, sell_filled=match.amount_a, buy_filled=match.amount_b)
            )
            fills.append(
                OrderFill(order=match.buyer, sell_filled=match.amount_b, buy_filled=match.amount_a)
            )
            total_volume += match.amount_a + match.amount_b

        token_a = normalize_address(group.token_a)
        token_b = normalize_address(group.token_b)
        prices = {
            token_a: Decimal(str(result.total_cow_b)),
            token_b: Decimal(str(result.total_cow_a)),
        }

        return LPResult(fills=fills, total_volume=total_volume, prices=prices)

    def _solve_cycles(
        self,
        orders: list[Order],
        router: SingleOrderRouter,
        auction: AuctionInstance,
        already_matched: set[str],
    ) -> list[OrderFill]:
        """Find and solve cycles in the order graph."""
        graph = OrderGraph.from_orders(orders)

        cycles: list[tuple[str, ...]] = []
        if self.max_tokens >= 3:
            cycles.extend(graph.find_3_cycles())
        if self.max_tokens >= 4:
            cycles.extend(graph.find_4_cycles(limit=50))

        if not cycles:
            return []

        all_fills: list[OrderFill] = []
        matched_uids: set[str] = set(already_matched)

        for cycle_tokens in cycles:
            cycle_orders = self._get_cycle_orders(cycle_tokens, graph, matched_uids)
            if not cycle_orders:
                continue

            settlement = solve_cycle(cycle_orders)
            if not settlement:
                continue

            # EBBO validation
            if not self._verify_cycle_ebbo(
                settlement.fills, settlement.clearing_prices, router, auction
            ):
                continue

            for fill in settlement.fills:
                if fill.order.uid not in matched_uids:
                    all_fills.append(fill)
                    matched_uids.add(fill.order.uid)

        return all_fills

    def _get_cycle_orders(
        self,
        cycle_tokens: tuple[str, ...],
        graph: OrderGraph,
        matched_uids: set[str],
    ) -> list[Order]:
        """Get orders forming a cycle, trying all rotations and directions."""
        n = len(cycle_tokens)

        for start in range(n):
            for reverse in [False, True]:
                if reverse:
                    rotated = tuple(cycle_tokens[(start - i) % n] for i in range(n))
                else:
                    rotated = tuple(cycle_tokens[(start + i) % n] for i in range(n))

                cycle_orders: list[Order] = []
                valid = True

                for i in range(n):
                    from_token = rotated[i]
                    to_token = rotated[(i + 1) % n]

                    edge_orders = graph.get_orders(from_token, to_token)
                    available = [o for o in edge_orders if o.uid not in matched_uids]

                    if not available:
                        valid = False
                        break

                    best = min(
                        available,
                        key=lambda o: o.buy_amount_int / o.sell_amount_int
                        if o.sell_amount_int > 0
                        else float("inf"),
                    )
                    cycle_orders.append(best)

                if valid and len(cycle_orders) == n:
                    return cycle_orders

        return []

    def _verify_cycle_ebbo(
        self,
        fills: list[OrderFill],
        clearing_prices: dict[str, int],
        router: SingleOrderRouter,
        auction: AuctionInstance,
    ) -> bool:
        """Verify that cycle settlement satisfies EBBO constraints."""
        for fill in fills:
            order = fill.order
            sell_token = normalize_address(order.sell_token)
            buy_token = normalize_address(order.buy_token)

            sell_price = clearing_prices.get(sell_token, 0)
            buy_price = clearing_prices.get(buy_token, 0)

            if sell_price <= 0 or buy_price <= 0:
                continue

            clearing_rate = Decimal(buy_price) / Decimal(sell_price)

            token_info = auction.tokens.get(sell_token)
            decimals = token_info.decimals if token_info and token_info.decimals else 18

            amm_rate = router.get_reference_price(sell_token, buy_token, token_in_decimals=decimals)
            if amm_rate is None:
                continue

            if clearing_rate < amm_rate:
                return False

        return True

    def _collect_fills(
        self,
        result: LPResult,
        all_fills: list[OrderFill],
        all_prices: dict[str, str],
        processed_uids: set[str],
    ) -> None:
        """Collect fills from a result, avoiding duplicates."""
        for fill in result.fills:
            if fill.order.uid not in processed_uids:
                all_fills.append(fill)
                processed_uids.add(fill.order.uid)

        for token, price in result.prices.items():
            all_prices[token] = str(int(price * Decimal(10**18)))

    def _build_result(
        self,
        all_fills: list[OrderFill],
        auction: AuctionInstance,
    ) -> StrategyResult:
        """Build the final StrategyResult."""
        filled_uids = {fill.order.uid for fill in all_fills}
        remainder_orders: list[Order] = []

        for order in auction.orders:
            if order.uid not in filled_uids:
                remainder_orders.append(order)

        for fill in all_fills:
            remainder = fill.get_remainder_order()
            if remainder:
                remainder_orders.append(remainder)

        normalized_prices = self._normalize_prices(all_fills)

        return StrategyResult(
            fills=all_fills,
            interactions=[],
            prices=normalized_prices,
            gas=0,
            remainder_orders=remainder_orders,
        )

    def _normalize_prices(self, fills: list[OrderFill]) -> dict[str, str]:
        """Normalize prices to maintain token conservation."""
        token_sells: dict[str, int] = defaultdict(int)
        token_buys: dict[str, int] = defaultdict(int)

        for fill in fills:
            sell_token = normalize_address(fill.order.sell_token)
            buy_token = normalize_address(fill.order.buy_token)
            token_sells[sell_token] += fill.sell_filled
            token_buys[buy_token] += fill.buy_filled

        prices: dict[str, str] = {}
        for token in set(token_sells.keys()) | set(token_buys.keys()):
            prices[token] = str(max(token_sells.get(token, 0), token_buys.get(token, 0)))

        return prices


__all__ = ["MultiPairCowStrategy"]
