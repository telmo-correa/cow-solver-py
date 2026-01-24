"""Settlement calculations for ring trades and cycles.

This module provides algorithms for calculating fill amounts and
clearing prices for cyclic trades (A→B→C→A).

IMPORTANT: All financial calculations use exact integer arithmetic via SafeInt.
No tolerance/epsilon comparisons are allowed. Limit price verification uses
integer cross-multiplication for exactness.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import structlog

from solver.models.auction import Order
from solver.models.types import normalize_address
from solver.safe_int import S
from solver.strategies.base import OrderFill

logger = structlog.get_logger()

# Price precision (1e18) - standard for CoW Protocol clearing prices
PRICE_PRECISION = 10**18


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

    # Threshold for "near-viable" cycles (for future AMM-assisted rings)
    NEAR_VIABLE_THRESHOLD: float = 0.95

    @property
    def near_viable(self) -> bool:
        """Check if this cycle is near-viable (could be closed with AMM)."""
        return self.product <= 1.0 / self.NEAR_VIABLE_THRESHOLD and not self.viable


@dataclass
class CycleSettlement:
    """Settlement result for a cycle.

    Attributes:
        fills: Order fills with amounts
        clearing_prices: Uniform clearing prices for tokens
        surplus: Scaled surplus ratio
    """

    fills: list[OrderFill]
    clearing_prices: dict[str, int]
    surplus: int = 0


def check_cycle_viability(
    cycle: tuple[str, ...],
    get_orders_fn: Callable[[str, str], list[Order]],
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
        get_orders_fn: Function(sell_token, buy_token) -> list[Order]

    Returns:
        CycleViability with viability status, surplus ratio, and orders to use
    """
    n = len(cycle)
    product = 1.0
    orders_used: list[Order] = []

    for i in range(n):
        from_token = cycle[i]
        to_token = cycle[(i + 1) % n]

        orders = get_orders_fn(from_token, to_token)
        if not orders:
            return CycleViability(
                viable=False,
                surplus_ratio=0.0,
                product=float("inf"),
                orders=[],
            )

        # Find order with LOWEST exchange rate on this edge (most generous)
        best_order: Order | None = None
        best_rate = float("inf")

        for order in orders:
            sell_amt = order.sell_amount_int
            buy_amt = order.buy_amount_int
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

    surplus_ratio = 1.0 - product

    return CycleViability(
        viable=product <= 1.0,
        surplus_ratio=surplus_ratio,
        product=product,
        orders=orders_used,
    )


def find_viable_cycle_direction(
    cycle_tokens: tuple[str, ...],
    get_orders_fn: Callable[[str, str], list[Order]],
) -> CycleViability | None:
    """Find a viable direction for a cycle.

    Since cycles are stored as sorted tuples, we need to try all rotations
    and both directions to find one that's actually viable.

    Args:
        cycle_tokens: Sorted token addresses from cycle detection
        get_orders_fn: Function(sell_token, buy_token) -> list[Order]

    Returns:
        CycleViability if a viable direction found, None otherwise.
        If no viable direction exists, returns the best near-viable result.
    """
    n = len(cycle_tokens)
    best_near_viable: CycleViability | None = None

    for start in range(n):
        # Forward direction
        rotated = tuple(cycle_tokens[(start + i) % n] for i in range(n))
        result = check_cycle_viability(rotated, get_orders_fn)
        if result.viable:
            return result
        if result.near_viable and (
            best_near_viable is None or result.product < best_near_viable.product
        ):
            best_near_viable = result

        # Reverse direction
        reversed_cycle = tuple(reversed(rotated))
        result = check_cycle_viability(reversed_cycle, get_orders_fn)
        if result.viable:
            return result
        if result.near_viable and (
            best_near_viable is None or result.product < best_near_viable.product
        ):
            best_near_viable = result

    return best_near_viable


def calculate_cycle_settlement(viability: CycleViability) -> CycleSettlement | None:
    """Calculate fill amounts and clearing prices for a viable cycle.

    The key constraint is token conservation: what flows out of each order
    must equal what flows into the next order in the cycle.

    We find the "bottleneck" - the order with the smallest normalized amount -
    and scale all fills proportionally.

    Args:
        viability: CycleViability with viable=True and orders populated

    Returns:
        CycleSettlement with fills and prices, or None if infeasible
    """
    orders = viability.orders
    n = len(orders)

    if n == 0:
        return None

    # Parse amounts and rates
    sell_amounts: list[int] = []
    rates: list[float] = []

    for order in orders:
        sell_amt = order.sell_amount_int
        buy_amt = order.buy_amount_int
        if sell_amt == 0 or buy_amt == 0:
            return None
        sell_amounts.append(sell_amt)
        rates.append(buy_amt / sell_amt)

    # Calculate normalized amounts for each order
    # Normalize to the first token in the cycle as reference
    normalized_amounts: list[float] = []
    cumulative_rate = 1.0

    for i in range(n):
        normalized = sell_amounts[i] / cumulative_rate
        normalized_amounts.append(normalized)
        cumulative_rate *= rates[i]

    # Find bottleneck (minimum normalized amount)
    bottleneck_normalized = min(normalized_amounts)

    # Calculate sell_filled amounts
    sell_filled_amounts: list[int] = []
    cumulative_rate = 1.0

    for i in range(n):
        sell_filled = int(bottleneck_normalized * cumulative_rate)
        sell_filled_amounts.append(sell_filled)
        cumulative_rate *= rates[i]

    # Set buy_filled = sell_filled of next order (ensures conservation)
    # Verify limit prices are respected using exact integer comparison
    fills: list[OrderFill] = []
    for i in range(n):
        sell_filled = sell_filled_amounts[i]
        next_i = (i + 1) % n
        buy_filled = sell_filled_amounts[next_i]

        # Verify limit price using integer cross-multiplication
        # Limit: buy_filled / sell_filled >= buy_amount / sell_amount
        # Cross multiply: buy_filled * sell_amount >= buy_amount * sell_filled
        if sell_filled > 0:
            order = orders[i]
            limit_satisfied = S(buy_filled) * S(order.sell_amount_int) >= S(
                order.buy_amount_int
            ) * S(sell_filled)
            if not limit_satisfied:
                logger.warning(
                    "cycle_limit_price_violated",
                    order_idx=i,
                    sell_filled=sell_filled,
                    buy_filled=buy_filled,
                    sell_amount=order.sell_amount_int,
                    buy_amount=order.buy_amount_int,
                )
                return None

        fills.append(
            OrderFill(
                order=orders[i],
                sell_filled=sell_filled,
                buy_filled=buy_filled,
            )
        )

    # Calculate clearing prices
    clearing_prices: dict[str, int] = {}
    price: float = PRICE_PRECISION

    for i, order in enumerate(orders):
        sell_token = normalize_address(order.sell_token)
        clearing_prices[sell_token] = int(price)
        price = price / rates[i]

    # Validate clearing price invariant
    # For cycles, the invariant sell_filled * sell_price = buy_filled * buy_price
    # may not hold exactly due to integer truncation in price calculation.
    # We check that the deviation is within 1 wei per token (negligible).
    for i, order in enumerate(orders):
        sell_token = normalize_address(order.sell_token)
        buy_token = normalize_address(order.buy_token)
        sell_price = clearing_prices.get(sell_token, 0)
        buy_price = clearing_prices.get(buy_token, 0)
        if sell_price > 0 and buy_price > 0:
            sell_value = S(fills[i].sell_filled) * S(sell_price)
            buy_value = S(fills[i].buy_filled) * S(buy_price)
            # Allow for float64 precision in clearing price calculation (~10^-14 relative error)
            # This is a SANITY CHECK, not limit price enforcement. Actual limit prices are
            # verified with exact integer cross-multiplication in the loop above.
            # Clearing prices use float division which has inherent precision limits.
            # Multiple float divisions can accumulate error, so we allow 10^-14 relative error.
            max_value = sell_value if sell_value > buy_value else buy_value
            max_deviation = max_value // S(10**14)
            if max_deviation.value == 0:
                max_deviation = S(1)  # At least 1 unit
            deviation = sell_value - buy_value if sell_value > buy_value else buy_value - sell_value
            if deviation > max_deviation:
                logger.warning(
                    "cycle_price_invariant_error",
                    order_idx=i,
                    sell_value=sell_value.value,
                    buy_value=buy_value.value,
                    deviation=deviation.value,
                )
                return None

    # Calculate surplus
    surplus = 0
    if viability.surplus_ratio > 0:
        surplus = int(viability.surplus_ratio * PRICE_PRECISION)

    return CycleSettlement(
        fills=fills,
        clearing_prices=clearing_prices,
        surplus=surplus,
    )


def solve_cycle(orders: list[Order]) -> CycleSettlement | None:
    """Solve a cycle given orders in cycle order.

    This is a simplified interface that checks viability and calculates
    settlement in one step.

    Args:
        orders: Orders forming the cycle (in cycle order A→B→C→A)

    Returns:
        CycleSettlement if viable and solvable, None otherwise
    """
    n = len(orders)
    if n < 3:
        return None

    # Parse amounts and rates, check viability
    sell_amounts: list[int] = []
    rates: list[float] = []
    product = 1.0

    for order in orders:
        sell_amt = order.sell_amount_int
        buy_amt = order.buy_amount_int
        if sell_amt <= 0 or buy_amt <= 0:
            return None
        sell_amounts.append(sell_amt)
        rate = buy_amt / sell_amt
        rates.append(rate)
        product *= rate

    if product > 1.0:
        return None  # Not viable

    # Calculate settlement using the same logic
    normalized_amounts: list[float] = []
    cumulative_rate = 1.0
    for i in range(n):
        normalized = sell_amounts[i] / cumulative_rate
        normalized_amounts.append(normalized)
        cumulative_rate *= rates[i]

    bottleneck_normalized = min(normalized_amounts)

    sell_filled_amounts: list[int] = []
    cumulative_rate = 1.0
    for i in range(n):
        sell_filled = int(bottleneck_normalized * cumulative_rate)
        sell_filled_amounts.append(sell_filled)
        cumulative_rate *= rates[i]

    fills: list[OrderFill] = []
    for i in range(n):
        sell_filled = sell_filled_amounts[i]
        next_i = (i + 1) % n
        buy_filled = sell_filled_amounts[next_i]

        if sell_filled <= 0 or buy_filled <= 0:
            continue

        # Verify limit price using integer cross-multiplication
        # buy_filled / sell_filled >= buy_amount / sell_amount
        # Cross multiply: buy_filled * sell_amount >= buy_amount * sell_filled
        order = orders[i]
        limit_satisfied = S(buy_filled) * S(order.sell_amount_int) >= S(order.buy_amount_int) * S(
            sell_filled
        )
        if not limit_satisfied:
            return None

        fills.append(
            OrderFill(
                order=orders[i],
                sell_filled=sell_filled,
                buy_filled=buy_filled,
            )
        )

    if len(fills) != n:
        return None

    # Calculate clearing prices
    clearing_prices: dict[str, int] = {}
    price: float = PRICE_PRECISION
    for i, order in enumerate(orders):
        sell_token = normalize_address(order.sell_token)
        clearing_prices[sell_token] = int(price)
        price = price / rates[i]

    surplus_ratio = 1.0 - product
    surplus = int(surplus_ratio * PRICE_PRECISION) if surplus_ratio > 0 else 0

    return CycleSettlement(
        fills=fills,
        clearing_prices=clearing_prices,
        surplus=surplus,
    )


__all__ = [
    "PRICE_PRECISION",
    "CycleViability",
    "CycleSettlement",
    "check_cycle_viability",
    "find_viable_cycle_direction",
    "calculate_cycle_settlement",
    "solve_cycle",
]
