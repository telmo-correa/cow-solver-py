"""Ring trade detection and execution strategy.

Ring trades settle orders in cycles (A→B→C→A) without AMM interaction,
providing maximum gas savings when viable.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from solver.models.auction import AuctionInstance, Order
from solver.models.types import normalize_address
from solver.pools import build_registry_from_liquidity
from solver.routing.router import SingleOrderRouter
from solver.strategies.base import OrderFill, StrategyResult
from solver.strategies.graph import OrderGraph
from solver.strategies.settlement import (
    CycleViability,
    calculate_cycle_settlement,
    find_viable_cycle_direction,
)

logger = structlog.get_logger()


@dataclass
class RingTrade:
    """A viable ring trade ready for settlement.

    Attributes:
        cycle: Token addresses in execution order
        orders: One order per edge in the cycle
        fills: (sell_filled, buy_filled) for each order
        clearing_prices: Uniform prices for all tokens
        surplus: Scaled surplus ratio
    """

    cycle: tuple[str, ...]
    orders: list[Order]
    fills: list[tuple[int, int]]
    clearing_prices: dict[str, int]
    surplus: int = 0

    @property
    def order_uids(self) -> set[str]:
        """Get UIDs of all orders in this ring."""
        return {order.uid for order in self.orders}

    @property
    def token_set(self) -> set[str]:
        """Get all tokens involved in this ring."""
        return set(self.cycle)

    def to_strategy_result(self) -> StrategyResult:
        """Convert to StrategyResult for the solver."""
        order_fills = [
            OrderFill(order=order, sell_filled=sell_filled, buy_filled=buy_filled)
            for order, (sell_filled, buy_filled) in zip(self.orders, self.fills, strict=True)
        ]
        prices = {token: str(price) for token, price in self.clearing_prices.items()}

        return StrategyResult(
            fills=order_fills,
            interactions=[],
            prices=prices,
            gas=0,
            remainder_orders=[],
        )


class RingTradeStrategy:
    """Strategy that finds and executes ring trades.

    Ring trades match orders in cycles (A→B→C→A) without AMM interaction.

    Args:
        max_4_cycles: Maximum 4-node cycles to find
        router: Optional router for EBBO price verification
        enforce_ebbo: If True, reject rings that violate EBBO
    """

    def __init__(
        self,
        max_4_cycles: int = 50,
        router: SingleOrderRouter | None = None,
        enforce_ebbo: bool = True,
    ):
        """Initialize the strategy."""
        self.max_4_cycles = max_4_cycles
        self.router = router
        self.enforce_ebbo = enforce_ebbo
        self._near_viable_count = 0
        self._ebbo_violations = 0

    def try_solve(self, auction: AuctionInstance) -> StrategyResult | None:
        """Attempt to find ring trades in the auction."""
        self._near_viable_count = 0
        self._ebbo_violations = 0
        self._current_auction = auction

        orders = list(auction.orders)
        graph = OrderGraph.from_orders(orders)

        logger.debug(
            "ring_trade_graph_built",
            tokens=len(graph.tokens),
            edges=graph.edge_count,
            orders=len(orders),
        )

        cycles_3 = graph.find_3_cycles()
        cycles_4 = graph.find_4_cycles(limit=self.max_4_cycles)

        if not cycles_3 and not cycles_4:
            return None

        viable_rings: list[RingTrade] = []

        for cycle_3 in cycles_3:
            ring = self._try_build_ring(cycle_3, graph)
            if ring:
                viable_rings.append(ring)

        for cycle_4 in cycles_4:
            ring = self._try_build_ring(cycle_4, graph)
            if ring:
                viable_rings.append(ring)

        logger.info(
            "ring_trade_viability_checked",
            viable_rings=len(viable_rings),
            near_viable=self._near_viable_count,
            total_cycles=len(cycles_3) + len(cycles_4),
        )

        if not viable_rings:
            return None

        selected = self._select_rings(viable_rings)
        return self._combine_rings(selected)

    def _try_build_ring(
        self,
        cycle_tokens: tuple[str, ...],
        graph: OrderGraph,
    ) -> RingTrade | None:
        """Try to build a viable ring trade from a cycle."""
        result = find_viable_cycle_direction(cycle_tokens, graph.get_orders)

        if result is None:
            return None

        if result.near_viable and not result.viable:
            self._near_viable_count += 1
            return None

        if not result.viable:
            return None

        return self._calculate_settlement(result, graph)

    def _calculate_settlement(
        self,
        viability: CycleViability,
        graph: OrderGraph,  # noqa: ARG002 - kept for API consistency
    ) -> RingTrade | None:
        """Calculate settlement for a viable cycle."""
        settlement = calculate_cycle_settlement(viability)
        if not settlement:
            return None

        # Build cycle token order from orders
        cycle_tokens: list[str] = []
        for order in viability.orders:
            cycle_tokens.append(normalize_address(order.sell_token))

        # Convert fills to tuple format
        fills = [(f.sell_filled, f.buy_filled) for f in settlement.fills]

        ring = RingTrade(
            cycle=tuple(cycle_tokens),
            orders=viability.orders,
            fills=fills,
            clearing_prices=settlement.clearing_prices,
            surplus=settlement.surplus,
        )

        if not self._verify_ebbo(ring):
            return None

        return ring

    def _verify_ebbo(self, ring: RingTrade) -> bool:
        """Verify ring trade satisfies EBBO constraint."""
        auction = getattr(self, "_current_auction", None)
        if auction is None:
            return True

        router = self.router
        if router is None:
            pool_registry = build_registry_from_liquidity(auction.liquidity)
            if pool_registry.pool_count == 0:
                return True
            router = SingleOrderRouter(pool_registry=pool_registry)

        from decimal import Decimal

        for order in ring.orders:
            sell_token = normalize_address(order.sell_token)
            buy_token = normalize_address(order.buy_token)

            sell_price = ring.clearing_prices.get(sell_token, 0)
            buy_price = ring.clearing_prices.get(buy_token, 0)

            if sell_price == 0 or buy_price == 0:
                continue

            ring_rate = Decimal(sell_price) / Decimal(buy_price)

            token_info = auction.tokens.get(sell_token)
            token_decimals = token_info.decimals if token_info and token_info.decimals else 18

            amm_rate = router.get_reference_price(
                sell_token, buy_token, token_in_decimals=token_decimals
            )

            if amm_rate is None:
                continue

            if ring_rate < amm_rate:
                self._ebbo_violations += 1
                if self.enforce_ebbo:
                    return False

        return True

    def _select_rings(self, rings: list[RingTrade]) -> list[RingTrade]:
        """Select non-overlapping rings using greedy algorithm."""
        sorted_rings = sorted(rings, key=lambda r: r.surplus, reverse=True)

        selected: list[RingTrade] = []
        used_orders: set[str] = set()
        used_tokens: set[str] = set()

        for ring in sorted_rings:
            if ring.order_uids & used_orders:
                continue
            if ring.token_set & used_tokens:
                continue

            selected.append(ring)
            used_orders.update(ring.order_uids)
            used_tokens.update(ring.token_set)

        return selected

    def _combine_rings(self, rings: list[RingTrade]) -> StrategyResult:
        """Combine multiple ring trades into a single result."""
        all_fills: list[OrderFill] = []
        all_prices: dict[str, str] = {}

        for ring in rings:
            result = ring.to_strategy_result()
            all_fills.extend(result.fills)
            all_prices.update(result.prices)

        return StrategyResult(
            fills=all_fills,
            interactions=[],
            prices=all_prices,
            gas=0,
            remainder_orders=[],
        )


__all__ = [
    "OrderGraph",  # Re-export for backwards compatibility
    "RingTrade",
    "RingTradeStrategy",
]
