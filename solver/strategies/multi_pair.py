"""Multi-pair price coordination strategy.

This strategy finds CoW matches by:
1. Processing bidirectional pairs via double auction
2. Finding cycles in the order graph and solving them

The strategy coordinates prices across overlapping token pairs and cycles
to maximize matched volume while maintaining EBBO compliance.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, Overflow
from fractions import Fraction
from typing import TYPE_CHECKING

import structlog

from solver.models.auction import AuctionInstance, Order
from solver.models.order_groups import OrderGroup, find_cow_opportunities
from solver.models.types import normalize_address
from solver.pools import build_registry_from_liquidity
from solver.routing.router import SingleOrderRouter
from solver.strategies.base import OrderFill, StrategyResult
from solver.strategies.base_amm import AMMBackedStrategy
from solver.strategies.components import find_token_components
from solver.strategies.ebbo_bounds import get_ebbo_bounds, verify_fills_against_ebbo
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
    from solver.amm.uniswap_v2 import UniswapV2
    from solver.amm.uniswap_v3 import UniswapV3AMM

logger = structlog.get_logger()


class MultiPairCowStrategy(AMMBackedStrategy):
    """Strategy that coordinates prices across multiple overlapping token pairs.

    When use_generalized=True (default), this strategy:
    1. Processes bidirectional pairs via double auction (Phase 1)
    2. Finds cycles in the order graph and solves them (Phase 2)

    When use_generalized=False, only Phase 1 (bidirectional pairs) is used.

    **Constraint Enforcement:**

    1. Fill-or-Kill: Partially enforced by double auction algorithm. FOK orders
       that can't be fully matched are excluded. Cycle settlement may produce
       partial fills that violate FOK - such cases should be filtered downstream.

    2. Limit Price: Enforced via exact integer cross-multiplication in the double
       auction algorithm and cycle settlement. The clearing price must satisfy:
       buy_filled * sell_amount >= buy_amount * sell_filled.

    3. EBBO: Two-sided validation via get_ebbo_bounds() and run_hybrid_auction().
       - ebbo_min: Minimum clearing rate (protects sellers, from token_a→token_b)
       - ebbo_max: Maximum clearing rate (protects buyers, from token_b→token_a)
       Clearing rate must satisfy: ebbo_min <= rate <= ebbo_max. Zero tolerance.
       Cycle EBBO is verified via verify_fills_against_ebbo().

    4. Uniform Price: Enforced structurally. Each connected component uses
       consistent prices. Price conflicts between components are avoided by
       skipping results that would create inconsistent prices.

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
        super().__init__(
            amm=amm,
            router=router,
            v3_amm=v3_amm,
            weighted_amm=weighted_amm,
            stable_amm=stable_amm,
            limit_order_amm=limit_order_amm,
        )
        self.max_combinations = max_combinations
        self.max_components = max_components
        self.max_tokens = max_tokens
        self.use_generalized = use_generalized

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

        return self._build_result(all_fills, auction, all_prices)

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

        return self._build_result(all_fills, auction, all_prices)

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
        """Handle large components by processing non-overlapping pairs individually.

        For components with >500 orders, we can't do full LP optimization.
        Instead, we process the largest pairs that don't share tokens with
        already-processed pairs. This ensures clearing prices are consistent.
        """
        sorted_groups = sorted(component, key=lambda g: g.order_count, reverse=True)
        combined_fills: list[OrderFill] = []
        combined_volume = 0
        combined_prices: dict[str, Decimal] = {}
        matched_uids: set[str] = set()
        priced_tokens: set[str] = set()  # Tokens that already have prices

        for group in sorted_groups[:10]:
            # Skip pairs that would create inconsistent prices
            token_a = normalize_address(group.token_a)
            token_b = normalize_address(group.token_b)
            if token_a in priced_tokens or token_b in priced_tokens:
                logger.debug(
                    "multi_pair_skip_overlapping",
                    pair=f"{token_a[-8:]}/{token_b[-8:]}",
                    reason="tokens already have prices",
                )
                continue

            result = self._solve_single_pair(group, router, auction)
            if result:
                for fill in result.fills:
                    if fill.order.uid not in matched_uids:
                        combined_fills.append(fill)
                        matched_uids.add(fill.order.uid)
                        combined_volume += fill.sell_filled
                combined_prices.update(result.prices)
                # Mark these tokens as priced
                priced_tokens.add(token_a)
                priced_tokens.add(token_b)

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

        # Get two-sided EBBO bounds
        bounds = get_ebbo_bounds(group.token_a, group.token_b, router, auction)

        result = run_hybrid_auction(
            group,
            amm_price=bounds.amm_price,
            ebbo_min=bounds.ebbo_min,
            ebbo_max=bounds.ebbo_max,
        )

        if not result.cow_matches:
            return None

        # EBBO validation is now handled by run_hybrid_auction with both bounds

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

                    # Use Fraction for exact rational comparison (no float precision loss)
                    best = min(
                        available,
                        key=lambda o: Fraction(o.buy_amount_int, o.sell_amount_int)
                        if o.sell_amount_int > 0
                        else Fraction(10**100, 1),
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
        return verify_fills_against_ebbo(fills, clearing_prices, router, auction)

    def _collect_fills(
        self,
        result: LPResult,
        all_fills: list[OrderFill],
        all_prices: dict[str, str],
        processed_uids: set[str],
    ) -> None:
        """Collect fills from a result, avoiding duplicates and price conflicts.

        Only adds fills if their tokens don't already have prices set.
        This ensures clearing prices remain consistent across all fills.
        """
        # Check for price conflicts before adding fills
        for token in result.prices:
            if token in all_prices:
                # Token already has a price - skip this entire result to maintain consistency
                logger.debug(
                    "multi_pair_skip_conflicting_result",
                    token=token[-8:],
                    reason="price already set",
                )
                return

        for fill in result.fills:
            if fill.order.uid not in processed_uids:
                all_fills.append(fill)
                processed_uids.add(fill.order.uid)

        for token, price in result.prices.items():
            # Skip invalid prices (overflow protection)
            if not price.is_finite() or price <= 0:
                logger.warning(
                    "multi_pair_invalid_price",
                    token=token[-8:],
                    price=str(price),
                )
                continue
            try:
                # Use explicit rounding to nearest for price output
                scaled_price = (price * Decimal(10**18)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                all_prices[token] = str(int(scaled_price))
            except (InvalidOperation, Overflow):
                logger.warning(
                    "multi_pair_price_overflow",
                    token=token[-8:],
                    price=str(price),
                )
                continue

    def _build_result(
        self,
        all_fills: list[OrderFill],
        auction: AuctionInstance,
        prices: dict[str, str] | None = None,
    ) -> StrategyResult:
        """Build the final StrategyResult.

        Args:
            all_fills: All order fills
            auction: The auction instance
            prices: Pre-computed clearing prices. If None, falls back to _normalize_prices.
        """
        filled_uids = {fill.order.uid for fill in all_fills}
        remainder_orders: list[Order] = []

        for order in auction.orders:
            if order.uid not in filled_uids:
                remainder_orders.append(order)

        for fill in all_fills:
            remainder = fill.get_remainder_order()
            if remainder:
                remainder_orders.append(remainder)

        # Use provided prices if available, otherwise compute from fills
        final_prices = prices if prices else self._normalize_prices(all_fills)

        return StrategyResult(
            fills=all_fills,
            interactions=[],
            prices=final_prices,
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
