"""Ring trade detection and execution strategy.

Ring trades settle orders in cycles (A→B→C→A) without AMM interaction,
providing maximum gas savings when viable.

A ring is viable when the product of exchange rates around the cycle <= 1:
    rate_1 × rate_2 × rate_3 <= 1

Where rate_i = buy_amount_i / sell_amount_i (limit price).

When product < 1, there's surplus (traders are collectively generous).
When product = 1, everyone gets exactly their limit.
When product > 1, the ring is not viable (traders demand more than available).

This module implements:
- OrderGraph: Directed graph of orders by token pair
- Cycle detection: Find 3-node and 4-node cycles
- Viability check: Determine if cycles can settle profitably
- RingTradeStrategy: SolutionStrategy implementation

See docs/design/slice-4.4-ring-trade-detection.md for full design.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import structlog

from solver.models.auction import AuctionInstance, Order
from solver.models.types import normalize_address
from solver.strategies.base import OrderFill, StrategyResult

logger = structlog.get_logger()

# Threshold for "near-viable" cycles (for future AMM-assisted rings)
NEAR_VIABLE_THRESHOLD = 0.95

# Price precision (1e18) - standard for CoW Protocol clearing prices
PRICE_PRECISION = 10**18

# Tolerance for numerical comparisons (0.1% = 0.001)
SETTLEMENT_TOLERANCE = 0.001


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


@dataclass
class CycleViability:
    """Result of checking whether a cycle is viable.

    Attributes:
        viable: Whether the cycle can settle as a pure ring (product <= 1)
        surplus_ratio: 1 - product. >0 means surplus available, 0 means exact match.
        product: Product of exchange rates around the cycle
        orders: Best order to use on each edge (if viable or near-viable)
        near_viable: True if product is slightly > 1 (could be closed with AMM)
    """

    viable: bool
    surplus_ratio: float
    product: float
    orders: list[Order] = field(default_factory=list)

    @property
    def near_viable(self) -> bool:
        """Check if this cycle is near-viable (could be closed with AMM).

        Near-viable means product is slightly > 1, so an AMM could fill the gap.
        """
        return self.product <= 1.0 / NEAR_VIABLE_THRESHOLD and not self.viable


def check_cycle_viability(
    cycle: tuple[str, ...],
    graph: OrderGraph,
) -> CycleViability:
    """Check if a cycle is economically viable.

    A cycle is viable if the product of exchange rates <= 1.
    - product < 1: surplus exists (traders are generous)
    - product = 1: exact match (everyone gets exactly their limit)
    - product > 1: not viable (traders demand more than available)

    We select the LOWEST rate order on each edge (most generous) to maximize
    chances of viability and surplus.

    Args:
        cycle: Token addresses in cycle order (A, B, C, ...) where A→B→C→...→A
        graph: Order graph to look up orders

    Returns:
        CycleViability with viability status, surplus ratio, and orders to use
    """
    n = len(cycle)
    product = 1.0
    orders_used: list[Order] = []

    for i in range(n):
        from_token = cycle[i]
        to_token = cycle[(i + 1) % n]

        orders = graph.get_orders(from_token, to_token)
        if not orders:
            # No order on this edge - cycle not viable
            return CycleViability(
                viable=False,
                surplus_ratio=0.0,
                product=float("inf"),
                orders=[],
            )

        # Find order with LOWEST exchange rate on this edge (most generous)
        # Rate = buy_amount / sell_amount (limit price = minimum acceptable)
        # Lower rate = willing to accept less = more generous
        best_order: Order | None = None
        best_rate = float("inf")

        for order in orders:
            sell_amt = int(order.sell_amount) if order.sell_amount else 0
            buy_amt = int(order.buy_amount) if order.buy_amount else 0
            if sell_amt > 0 and buy_amt > 0:
                rate = buy_amt / sell_amt
                if rate < best_rate:
                    best_rate = rate
                    best_order = order

        if best_order is None or best_rate == float("inf"):
            return CycleViability(
                viable=False,
                surplus_ratio=0.0,
                product=float("inf"),
                orders=[],
            )

        product *= best_rate
        orders_used.append(best_order)

    # Calculate surplus ratio: how much surplus is available?
    # surplus_ratio > 0 means product < 1 (surplus exists)
    # surplus_ratio = 0 means product = 1 (exact match)
    # surplus_ratio < 0 means product > 1 (not viable)
    surplus_ratio = 1.0 - product

    return CycleViability(
        viable=product <= 1.0,
        surplus_ratio=surplus_ratio,
        product=product,
        orders=orders_used,
    )


def find_viable_cycle_direction(
    cycle_tokens: tuple[str, ...],
    graph: OrderGraph,
) -> CycleViability | None:
    """Find a viable direction for a cycle.

    Since cycles are stored as sorted tuples, we need to try all rotations
    and both directions to find one that's actually viable.

    Args:
        cycle_tokens: Sorted token addresses from cycle detection
        graph: Order graph

    Returns:
        CycleViability if a viable direction found, None otherwise.
        If no viable direction exists, returns the best near-viable result.
    """
    n = len(cycle_tokens)
    best_near_viable: CycleViability | None = None

    # Single pass: try all rotations and directions
    for start in range(n):
        # Forward direction
        rotated = tuple(cycle_tokens[(start + i) % n] for i in range(n))
        result = check_cycle_viability(rotated, graph)
        if result.viable:
            return result  # Early return on viable
        # For near-viable, prefer lower product (closer to 1 = easier to close with AMM)
        if result.near_viable and (
            best_near_viable is None or result.product < best_near_viable.product
        ):
            best_near_viable = result

        # Reverse direction
        reversed_cycle = tuple(reversed(rotated))
        result = check_cycle_viability(reversed_cycle, graph)
        if result.viable:
            return result  # Early return on viable
        if result.near_viable and (
            best_near_viable is None or result.product < best_near_viable.product
        ):
            best_near_viable = result

    return best_near_viable


@dataclass
class RingTrade:
    """A viable ring trade ready for settlement.

    Attributes:
        cycle: Token addresses in execution order
        orders: One order per edge in the cycle
        fills: (sell_filled, buy_filled) for each order
        clearing_prices: Uniform prices for all tokens
        surplus: Scaled surplus ratio (1 - product) * PRICE_PRECISION, used for ranking
    """

    cycle: tuple[str, ...]
    orders: list[Order]
    fills: list[tuple[int, int]]  # (sell_filled, buy_filled) per order
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
        """Convert to StrategyResult for the solver.

        Ring trades have:
        - Fills for each order
        - Clearing prices for tokens in the cycle
        - Zero interactions (no AMM calls)
        - Zero gas (peer-to-peer settlement)
        """
        order_fills = [
            OrderFill(
                order=order,
                sell_filled=sell_filled,
                buy_filled=buy_filled,
            )
            for order, (sell_filled, buy_filled) in zip(self.orders, self.fills, strict=True)
        ]

        # Convert int prices to string
        prices = {token: str(price) for token, price in self.clearing_prices.items()}

        return StrategyResult(
            fills=order_fills,
            interactions=[],  # No AMM interactions
            prices=prices,
            gas=0,  # Peer-to-peer, no gas
            remainder_orders=[],  # TODO: Handle partial fills
        )


class RingTradeStrategy:
    """Strategy that finds and executes ring trades.

    Ring trades match orders in cycles (A→B→C→A) without AMM interaction.
    This provides maximum gas savings when cycles are viable.

    The strategy:
    1. Builds an order graph from the auction
    2. Finds 3-node and 4-node cycles
    3. Checks each cycle for viability
    4. Selects non-overlapping rings (greedy by surplus)
    5. Returns combined result

    Note: This implementation handles "pure" rings only (product <= 1).
    AMM-assisted rings (product slightly > 1) are tracked but not executed.
    See docs/design/slice-4.4-ring-trade-detection.md for future plans.
    """

    def __init__(self, max_4_cycles: int = 50):
        """Initialize the strategy.

        Args:
            max_4_cycles: Maximum 4-node cycles to find (limits computation)
        """
        self.max_4_cycles = max_4_cycles
        self._near_viable_count = 0  # Track for metrics

    def try_solve(self, auction: AuctionInstance) -> StrategyResult | None:
        """Attempt to find ring trades in the auction.

        Args:
            auction: The auction to solve

        Returns:
            StrategyResult with ring trade fills, or None if no rings found
        """
        # Reset metrics
        self._near_viable_count = 0

        # Convert to list once for efficient access
        orders = list(auction.orders)

        # Build order graph
        graph = OrderGraph.from_orders(orders)

        logger.debug(
            "ring_trade_graph_built",
            tokens=len(graph.tokens),
            edges=graph.edge_count,
            orders=len(orders),
        )

        # Find cycles
        cycles_3 = graph.find_3_cycles()
        cycles_4 = graph.find_4_cycles(limit=self.max_4_cycles)

        logger.debug(
            "ring_trade_cycles_found",
            cycles_3=len(cycles_3),
            cycles_4=len(cycles_4),
        )

        if not cycles_3 and not cycles_4:
            return None

        # Check viability and build ring trades
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

        # Select non-overlapping rings (greedy by surplus)
        selected = self._select_rings(viable_rings)

        logger.info(
            "ring_trade_selected",
            selected_rings=len(selected),
            total_orders=sum(len(r.orders) for r in selected),
        )

        # Combine into single result
        return self._combine_rings(selected)

    def _try_build_ring(
        self,
        cycle_tokens: tuple[str, ...],
        graph: OrderGraph,
    ) -> RingTrade | None:
        """Try to build a viable ring trade from a cycle.

        Args:
            cycle_tokens: Sorted token addresses
            graph: Order graph

        Returns:
            RingTrade if viable, None otherwise
        """
        result = find_viable_cycle_direction(cycle_tokens, graph)

        if result is None:
            return None

        if result.near_viable and not result.viable:
            # Track for metrics but don't execute
            self._near_viable_count += 1
            logger.debug(
                "ring_trade_near_viable",
                cycle=cycle_tokens,
                product=result.product,
                surplus_ratio=result.surplus_ratio,
            )
            return None

        if not result.viable:
            return None

        # Calculate fill amounts and clearing prices
        return self._calculate_settlement(result)

    def _calculate_settlement(self, viability: CycleViability) -> RingTrade | None:
        """Calculate fill amounts and clearing prices for a viable cycle.

        The key constraint is token conservation: what flows out of each order
        must equal what flows into the next order in the cycle.

        We find the "bottleneck" - the order with the smallest normalized amount -
        and scale all fills proportionally.

        Args:
            viability: CycleViability with viable=True and orders populated

        Returns:
            RingTrade with calculated fills and prices, or None if infeasible
        """
        orders = viability.orders
        n = len(orders)

        if n == 0:
            return None

        # Parse amounts once (cache int conversions)
        sell_amounts: list[int] = []
        rates: list[float] = []

        for order in orders:
            sell_amt = int(order.sell_amount) if order.sell_amount else 0
            buy_amt = int(order.buy_amount) if order.buy_amount else 0
            if sell_amt == 0 or buy_amt == 0:
                return None
            sell_amounts.append(sell_amt)
            rates.append(buy_amt / sell_amt)

        # Calculate normalized amounts for each order
        # Normalize to the first token in the cycle as reference
        normalized_amounts: list[float] = []
        cumulative_rate = 1.0

        for i in range(n):
            # Normalize this order's sell amount to reference token
            normalized = sell_amounts[i] / cumulative_rate
            normalized_amounts.append(normalized)
            cumulative_rate *= rates[i]

        # Find bottleneck (minimum normalized amount)
        bottleneck_normalized = min(normalized_amounts)

        # Calculate sell_filled amounts first (determines the ring flow)
        sell_filled_amounts: list[int] = []
        cumulative_rate = 1.0

        for i in range(n):
            sell_filled = int(bottleneck_normalized * cumulative_rate)
            sell_filled_amounts.append(sell_filled)
            cumulative_rate *= rates[i]

        # Set buy_filled = sell_filled of next order (ensures conservation by construction)
        # Then verify limit prices are respected
        fills: list[tuple[int, int]] = []
        for i in range(n):
            sell_filled = sell_filled_amounts[i]
            next_i = (i + 1) % n
            buy_filled = sell_filled_amounts[next_i]  # Conservation by construction

            # Verify limit price: actual rate >= limit rate
            # actual_rate = buy_filled / sell_filled
            # limit_rate = rates[i] = buy_amount / sell_amount
            if sell_filled > 0:
                actual_rate = buy_filled / sell_filled
                limit_rate = rates[i]
                # Allow small tolerance for rounding
                if actual_rate < limit_rate * (1 - SETTLEMENT_TOLERANCE):
                    logger.warning(
                        "ring_trade_limit_price_violated",
                        order_idx=i,
                        actual_rate=actual_rate,
                        limit_rate=limit_rate,
                        deficit_pct=(limit_rate - actual_rate) / limit_rate * 100,
                    )
                    return None  # Reject - order wouldn't get their minimum

            fills.append((sell_filled, buy_filled))

        # Calculate clearing prices
        # Use the first token as reference (price = PRICE_PRECISION)
        # Then derive other prices from the exchange rates
        clearing_prices: dict[str, int] = {}
        price: float = PRICE_PRECISION  # Reference price

        cycle_tokens: list[str] = []
        for i, order in enumerate(orders):
            sell_token = normalize_address(order.sell_token)
            cycle_tokens.append(sell_token)
            clearing_prices[sell_token] = int(price)
            # Update price for next token
            price = price / rates[i]

        # Validate clearing price invariant: sell * sell_price ≈ buy * buy_price
        for i, order in enumerate(orders):
            sell_token = normalize_address(order.sell_token)
            buy_token = normalize_address(order.buy_token)
            sell_price = clearing_prices.get(sell_token, 0)
            buy_price = clearing_prices.get(buy_token, 0)
            if sell_price > 0 and buy_price > 0:
                sell_value = fills[i][0] * sell_price
                buy_value = fills[i][1] * buy_price
                if (
                    sell_value > 0
                    and abs(sell_value - buy_value) / sell_value > SETTLEMENT_TOLERANCE
                ):
                    logger.warning(
                        "ring_trade_price_invariant_error",
                        order_idx=i,
                        sell_value=sell_value,
                        buy_value=buy_value,
                        diff_pct=abs(sell_value - buy_value) / sell_value * 100,
                    )
                    return None  # Reject invalid settlement

        # Calculate surplus (scaled ratio representing available surplus)
        # When product < 1, surplus_ratio = 1 - product > 0
        # Note: This is a dimensionless scaled value, not actual surplus in tokens
        surplus = 0
        if viability.surplus_ratio > 0:
            surplus = int(viability.surplus_ratio * PRICE_PRECISION)

        return RingTrade(
            cycle=tuple(cycle_tokens),
            orders=orders,
            fills=fills,
            clearing_prices=clearing_prices,
            surplus=surplus,
        )

    def _select_rings(self, rings: list[RingTrade]) -> list[RingTrade]:
        """Select non-overlapping rings using greedy algorithm.

        Higher surplus rings are preferred. Rings cannot share orders OR tokens
        (to avoid clearing price conflicts when combining results).

        Note: Greedy selection by surplus is a heuristic. The optimal selection
        (maximum weighted independent set) is NP-hard. Greedy typically achieves
        good results in practice but may not maximize total surplus.

        Args:
            rings: List of viable ring trades

        Returns:
            List of selected non-overlapping rings
        """
        # Sort by surplus (descending)
        sorted_rings = sorted(rings, key=lambda r: r.surplus, reverse=True)

        selected: list[RingTrade] = []
        used_orders: set[str] = set()
        used_tokens: set[str] = set()

        for ring in sorted_rings:
            # Check if any order already used
            if ring.order_uids & used_orders:
                continue

            # Check if any token already used (prevents clearing price conflicts)
            if ring.token_set & used_tokens:
                continue

            selected.append(ring)
            used_orders.update(ring.order_uids)
            used_tokens.update(ring.token_set)

        return selected

    def _combine_rings(self, rings: list[RingTrade]) -> StrategyResult:
        """Combine multiple ring trades into a single result.

        Args:
            rings: List of selected ring trades

        Returns:
            Combined StrategyResult
        """
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
            remainder_orders=[],  # TODO: Handle partial fills
        )
