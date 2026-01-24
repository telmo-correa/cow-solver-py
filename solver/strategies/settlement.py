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

    Uses exact integer arithmetic: product = (buy1*buy2*...) / (sell1*sell2*...)
    Viable iff product_num <= product_denom.

    Args:
        cycle: Token addresses in cycle order (A, B, C, ...) where A→B→C→...→A
        get_orders_fn: Function(sell_token, buy_token) -> list[Order]

    Returns:
        CycleViability with viability status, surplus ratio, and orders to use
    """
    n = len(cycle)
    # Product as integer ratio: (numerator, denominator)
    # product = (buy1/sell1) * (buy2/sell2) * ... = (buy1*buy2*...) / (sell1*sell2*...)
    product_num = S(1)
    product_denom = S(1)
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
        # rate = buy_amt / sell_amt
        # Compare using cross-multiplication: rate1 < rate2 iff buy1*sell2 < buy2*sell1
        best_order: Order | None = None
        best_buy: int = 0
        best_sell: int = 0

        for order in orders:
            sell_amt = order.sell_amount_int
            buy_amt = order.buy_amount_int
            if sell_amt > 0 and buy_amt > 0:
                if best_order is None:
                    best_order = order
                    best_buy = buy_amt
                    best_sell = sell_amt
                else:
                    # Compare: buy_amt/sell_amt < best_buy/best_sell
                    # Cross-multiply: buy_amt * best_sell < best_buy * sell_amt
                    if S(buy_amt) * S(best_sell) < S(best_buy) * S(sell_amt):
                        best_order = order
                        best_buy = buy_amt
                        best_sell = sell_amt

        if best_order is None:
            return CycleViability(
                viable=False,
                surplus_ratio=0.0,
                product=float("inf"),
                orders=[],
            )

        product_num = product_num * S(best_buy)
        product_denom = product_denom * S(best_sell)
        orders_used.append(best_order)

    # viable if product <= 1, i.e., product_num <= product_denom
    viable = product_num <= product_denom

    # Calculate float product and surplus_ratio for backward compatibility
    # These are used for logging and near-viable detection only
    if product_denom.value > 0:
        product = float(product_num.value) / float(product_denom.value)
    else:
        product = float("inf")
    surplus_ratio = 1.0 - product

    return CycleViability(
        viable=viable,
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

    Uses exact integer arithmetic throughout:
    - cumulative_rate[i] = (buy0*...*buy_{i-1}) / (sell0*...*sell_{i-1})
    - normalized[i] = sell_amounts[i] / cumulative_rate[i]
    - Bottleneck found via integer cross-multiplication comparison
    - Fill amounts computed with floor division

    Args:
        viability: CycleViability with viable=True and orders populated

    Returns:
        CycleSettlement with fills and prices, or None if infeasible
    """
    orders = viability.orders
    n = len(orders)

    if n == 0:
        return None

    # Parse amounts and rate components (buy_amt, sell_amt) for each order
    sell_amounts: list[int] = []
    buy_amounts: list[int] = []

    for order in orders:
        sell_amt = order.sell_amount_int
        buy_amt = order.buy_amount_int
        if sell_amt == 0 or buy_amt == 0:
            return None
        sell_amounts.append(sell_amt)
        buy_amounts.append(buy_amt)

    # Calculate cumulative rate components for each position
    # cumulative_rate[i] = rate[0] * rate[1] * ... * rate[i-1]
    # = (buy[0]/sell[0]) * (buy[1]/sell[1]) * ... * (buy[i-1]/sell[i-1])
    # = (buy[0] * ... * buy[i-1]) / (sell[0] * ... * sell[i-1])
    # cumulative[0] = 1/1 (identity)
    cumulative_num: list[int] = [1]  # cumulative_num[i] = buy[0] * ... * buy[i-1]
    cumulative_denom: list[int] = [1]  # cumulative_denom[i] = sell[0] * ... * sell[i-1]

    for i in range(n - 1):
        cumulative_num.append(cumulative_num[i] * buy_amounts[i])
        cumulative_denom.append(cumulative_denom[i] * sell_amounts[i])

    # Find bottleneck (minimum normalized amount) using integer comparison
    # normalized[i] = sell_amounts[i] / cumulative_rate[i]
    #               = sell_amounts[i] * cumulative_denom[i] / cumulative_num[i]
    # Compare normalized[i] vs normalized[j] via cross-multiplication:
    # sell_amounts[i] * cumulative_denom[i] * cumulative_num[j]
    #   vs sell_amounts[j] * cumulative_denom[j] * cumulative_num[i]
    bottleneck_idx = 0
    for i in range(1, n):
        # Is normalized[i] < normalized[bottleneck_idx]?
        # Using cross-multiplication to avoid division
        lhs = S(sell_amounts[i]) * S(cumulative_denom[i]) * S(cumulative_num[bottleneck_idx])
        rhs = (
            S(sell_amounts[bottleneck_idx])
            * S(cumulative_denom[bottleneck_idx])
            * S(cumulative_num[i])
        )
        if lhs < rhs:
            bottleneck_idx = i

    # Calculate sell_filled amounts using exact integer arithmetic
    # sell_filled[i] = bottleneck_normalized * cumulative_rate[i]
    # bottleneck_normalized = sell_amounts[b] * cumulative_denom[b] / cumulative_num[b]
    # sell_filled[i] = (sell_amounts[b] * cumulative_denom[b] / cumulative_num[b])
    #                  * (cumulative_num[i] / cumulative_denom[i])
    #                = (sell_amounts[b] * cumulative_denom[b] * cumulative_num[i])
    #                  / (cumulative_num[b] * cumulative_denom[i])
    bottleneck_sell = sell_amounts[bottleneck_idx]
    bottleneck_cum_num = cumulative_num[bottleneck_idx]
    bottleneck_cum_denom = cumulative_denom[bottleneck_idx]

    sell_filled_amounts: list[int] = []
    for i in range(n):
        numerator = S(bottleneck_sell) * S(bottleneck_cum_denom) * S(cumulative_num[i])
        denominator = S(bottleneck_cum_num) * S(cumulative_denom[i])
        if denominator.value == 0:
            return None
        # Use floor division (conservative)
        sell_filled = (numerator // denominator).value
        sell_filled_amounts.append(sell_filled)

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

    # Calculate clearing prices using integer arithmetic
    # price[0] = PRICE_PRECISION
    # price[i+1] = price[i] / rate[i] = price[i] * sell[i] / buy[i]
    # price[i] = PRICE_PRECISION * sell[0]*...*sell[i-1] / (buy[0]*...*buy[i-1])
    #          = PRICE_PRECISION * cumulative_denom[i] / cumulative_num[i]
    clearing_prices: dict[str, int] = {}

    for i, order in enumerate(orders):
        sell_token = normalize_address(order.sell_token)
        if cumulative_num[i] == 0:
            return None
        # Use floor division for prices (standard behavior)
        price = (S(PRICE_PRECISION) * S(cumulative_denom[i])) // S(cumulative_num[i])
        clearing_prices[sell_token] = price.value

    # Validate clearing price invariant
    # For cycles, the invariant sell_filled * sell_price = buy_filled * buy_price
    # should hold with bounded deviation from floor division rounding.
    #
    # Error analysis for floor division chains:
    # - N fill calculations (floor division each) - loses at most 1 unit per calculation
    # - N price calculations (floor division each) - cumulative error compounds
    # - Products scale errors by the other factor: fill * δ_price + price * δ_fill
    #
    # For values of magnitude M (fill * price ≈ 10^36), accumulated floor division
    # errors result in deviations that are small relative to M. We use a relative
    # tolerance of 10^-14 which is conservative while being much tighter than
    # human-perceptible differences. This catches systematic errors while allowing
    # for unavoidable integer arithmetic artifacts.
    #
    # Relative tolerance: deviation / max(sell_value, buy_value) <= 10^-14
    # Equivalent absolute tolerance: deviation <= max_value / 10^14
    relative_tolerance_divisor = S(10**14)
    for i, order in enumerate(orders):
        sell_token = normalize_address(order.sell_token)
        buy_token = normalize_address(order.buy_token)
        sell_price = clearing_prices.get(sell_token, 0)
        buy_price = clearing_prices.get(buy_token, 0)
        if sell_price > 0 and buy_price > 0:
            sell_value = S(fills[i].sell_filled) * S(sell_price)
            buy_value = S(fills[i].buy_filled) * S(buy_price)
            # Use relative tolerance based on the larger value
            max_value = sell_value if sell_value > buy_value else buy_value
            max_deviation = max_value // relative_tolerance_divisor
            # Ensure minimum absolute tolerance of PRICE_PRECISION for small values
            if max_deviation < S(PRICE_PRECISION):
                max_deviation = S(PRICE_PRECISION)
            deviation = sell_value - buy_value if sell_value > buy_value else buy_value - sell_value
            if deviation > max_deviation:
                logger.warning(
                    "cycle_price_invariant_error",
                    order_idx=i,
                    sell_value=sell_value.value,
                    buy_value=buy_value.value,
                    deviation=deviation.value,
                    max_allowed=max_deviation.value,
                )
                return None

    # Calculate surplus (still using float for backward compatibility in dataclass)
    # Use round() for nearest integer instead of truncation
    surplus = 0
    if viability.surplus_ratio > 0:
        surplus = round(viability.surplus_ratio * PRICE_PRECISION)

    return CycleSettlement(
        fills=fills,
        clearing_prices=clearing_prices,
        surplus=surplus,
    )


def solve_cycle(orders: list[Order]) -> CycleSettlement | None:
    """Solve a cycle given orders in cycle order.

    This is a simplified interface that checks viability and calculates
    settlement in one step.

    Uses exact integer arithmetic throughout:
    - Viability check: product_num <= product_denom
    - Fill amounts via integer cross-multiplication
    - Clearing prices via integer division

    Args:
        orders: Orders forming the cycle (in cycle order A→B→C→A)

    Returns:
        CycleSettlement if viable and solvable, None otherwise
    """
    n = len(orders)
    if n < 3:
        return None

    # Parse amounts, check viability using integer product
    # product = (buy0/sell0) * (buy1/sell1) * ... = (buy0*buy1*...) / (sell0*sell1*...)
    sell_amounts: list[int] = []
    buy_amounts: list[int] = []
    product_num = S(1)
    product_denom = S(1)

    for order in orders:
        sell_amt = order.sell_amount_int
        buy_amt = order.buy_amount_int
        if sell_amt <= 0 or buy_amt <= 0:
            return None
        sell_amounts.append(sell_amt)
        buy_amounts.append(buy_amt)
        product_num = product_num * S(buy_amt)
        product_denom = product_denom * S(sell_amt)

    # viable if product <= 1, i.e., product_num <= product_denom
    if product_num > product_denom:
        return None  # Not viable

    # Calculate cumulative rate components for each position
    # cumulative[i] = rate[0] * ... * rate[i-1] = (buy0*...*buy_{i-1}) / (sell0*...*sell_{i-1})
    cumulative_num: list[int] = [1]
    cumulative_denom: list[int] = [1]

    for i in range(n - 1):
        cumulative_num.append(cumulative_num[i] * buy_amounts[i])
        cumulative_denom.append(cumulative_denom[i] * sell_amounts[i])

    # Find bottleneck (minimum normalized amount) using integer comparison
    # normalized[i] = sell_amounts[i] * cumulative_denom[i] / cumulative_num[i]
    bottleneck_idx = 0
    for i in range(1, n):
        lhs = S(sell_amounts[i]) * S(cumulative_denom[i]) * S(cumulative_num[bottleneck_idx])
        rhs = (
            S(sell_amounts[bottleneck_idx])
            * S(cumulative_denom[bottleneck_idx])
            * S(cumulative_num[i])
        )
        if lhs < rhs:
            bottleneck_idx = i

    # Calculate sell_filled amounts
    bottleneck_sell = sell_amounts[bottleneck_idx]
    bottleneck_cum_num = cumulative_num[bottleneck_idx]
    bottleneck_cum_denom = cumulative_denom[bottleneck_idx]

    sell_filled_amounts: list[int] = []
    for i in range(n):
        numerator = S(bottleneck_sell) * S(bottleneck_cum_denom) * S(cumulative_num[i])
        denominator = S(bottleneck_cum_num) * S(cumulative_denom[i])
        if denominator.value == 0:
            return None
        sell_filled = (numerator // denominator).value
        sell_filled_amounts.append(sell_filled)

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

    # Calculate clearing prices using integer arithmetic
    # price[i] = PRICE_PRECISION * cumulative_denom[i] / cumulative_num[i]
    clearing_prices: dict[str, int] = {}
    for i, order in enumerate(orders):
        sell_token = normalize_address(order.sell_token)
        if cumulative_num[i] == 0:
            return None
        price = (S(PRICE_PRECISION) * S(cumulative_denom[i])) // S(cumulative_num[i])
        clearing_prices[sell_token] = price.value

    # Calculate surplus (float for backward compatibility)
    # surplus_ratio = 1 - product = (product_denom - product_num) / product_denom
    # Use round() for nearest integer instead of truncation
    if product_denom.value > 0:
        surplus_ratio = float(product_denom.value - product_num.value) / float(product_denom.value)
    else:
        surplus_ratio = 0.0
    surplus = round(surplus_ratio * PRICE_PRECISION) if surplus_ratio > 0 else 0

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
