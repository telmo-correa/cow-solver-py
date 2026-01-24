"""Price enumeration and LP optimization for CoW matching.

This module provides:
- PriceCandidates: Collection of price ratio candidates
- Price enumeration from order limits and AMM prices
- LP solver for finding optimal fills at given prices
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from itertools import product
from typing import TYPE_CHECKING

import structlog

from solver.fees.price_estimation import get_token_info
from solver.models.auction import AuctionInstance, Order
from solver.models.order_groups import OrderGroup
from solver.models.types import normalize_address
from solver.strategies.base import OrderFill
from solver.strategies.double_auction import get_limit_price

if TYPE_CHECKING:
    from solver.routing.router import SingleOrderRouter

logger = structlog.get_logger()


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


def build_price_candidates_from_orders(
    orders: list[Order],
    router: SingleOrderRouter | None,
    auction: AuctionInstance | None,
) -> PriceCandidates:
    """Build price candidates from orders in a component.

    Sources:
    1. Order limit prices (buy_amount / sell_amount)
    2. AMM spot prices (via router.get_reference_price)

    Args:
        orders: Orders in the component
        router: Router for AMM price queries (may be None)
        auction: Auction instance for token decimals

    Returns:
        PriceCandidates with ratios for each token pair
    """
    candidates = PriceCandidates()

    # Track which pairs we've queried AMM prices for
    amm_queried: set[tuple[str, str]] = set()

    for order in orders:
        sell_token = normalize_address(order.sell_token)
        buy_token = normalize_address(order.buy_token)

        # Order limit price: buy_amount / sell_amount
        sell_amt = order.sell_amount_int
        buy_amt = order.buy_amount_int

        if sell_amt > 0 and buy_amt > 0:
            limit_price = Decimal(buy_amt) / Decimal(sell_amt)
            candidates.add_ratio(sell_token, buy_token, limit_price)

        # Add AMM spot price if available (only once per pair)
        if (
            router is not None
            and auction is not None
            and (sell_token, buy_token) not in amm_queried
        ):
            amm_queried.add((sell_token, buy_token))
            token_info = get_token_info(auction, sell_token)
            decimals = (
                18 if token_info is None or token_info.decimals is None else token_info.decimals
            )
            amm_price = router.get_reference_price(
                sell_token, buy_token, token_in_decimals=decimals
            )
            if amm_price is not None and amm_price > 0:
                candidates.add_ratio(sell_token, buy_token, amm_price)

    return candidates


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

        # Calculate fill amounts preserving the order's limit price
        # The user gets at least their requested amount (the limit price)
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


def build_token_graph_from_groups(component: list[OrderGroup]) -> dict[str, set[str]]:
    """Build undirected token graph from OrderGroups.

    Args:
        component: List of OrderGroups

    Returns:
        Adjacency list representation of token graph
    """
    from collections import defaultdict

    graph: dict[str, set[str]] = defaultdict(set)
    for group in component:
        token_a = normalize_address(group.token_a)
        token_b = normalize_address(group.token_b)
        graph[token_a].add(token_b)
        graph[token_b].add(token_a)
    return graph


def build_token_graph_from_orders(orders: list[Order]) -> dict[str, set[str]]:
    """Build undirected token graph from raw orders.

    Args:
        orders: List of orders

    Returns:
        Adjacency list representation of token graph
    """
    from collections import defaultdict

    graph: dict[str, set[str]] = defaultdict(set)
    for order in orders:
        sell_token = normalize_address(order.sell_token)
        buy_token = normalize_address(order.buy_token)
        graph[sell_token].add(buy_token)
        graph[buy_token].add(sell_token)
    return graph


def solve_fills_at_prices_v2(
    orders: list[Order],
    tokens: set[str],  # noqa: ARG001 - kept for API compatibility
    prices: dict[str, Decimal],
) -> LPResult | None:
    """Solve for optimal fills given fixed token prices using LP (for raw orders).

    This is a variant of solve_fills_at_prices that works with raw orders
    instead of OrderGroups. Used for generalized cycle detection.

    LP Formulation:
    - Variables: x_i = fill ratio for each eligible order (0 to 1)
    - Objective: maximize sum(x_i * sell_amount_i)
    - Constraints:
      - Fill bounds: 0 <= x_i <= 1
      - Conservation: sum(sells of t) >= sum(buys of t) for each token t

    An order is eligible if its limit price is satisfied by the given prices.

    Args:
        orders: List of orders to optimize
        tokens: Set of token addresses (unused, for API compatibility)
        prices: Token prices (token -> price)

    Returns:
        LPResult if solution found, None if no feasible solution
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        logger.warning("scipy not available, skipping LP optimization")
        return None

    if not orders:
        return None

    # Check which orders are eligible at given prices
    eligible_orders: list[tuple[Order, Decimal]] = []  # (order, limit_price)

    for order in orders:
        sell_token = normalize_address(order.sell_token)
        buy_token = normalize_address(order.buy_token)

        # Order limit price: buy_amount / sell_amount
        sell_amt = order.sell_amount_int
        buy_amt = order.buy_amount_int

        if sell_amt <= 0 or buy_amt <= 0:
            continue

        limit_price = Decimal(buy_amt) / Decimal(sell_amt)

        # Get prices for both tokens
        price_sell = prices.get(sell_token)
        price_buy = prices.get(buy_token)
        if price_sell is None or price_buy is None or price_sell <= 0:
            continue

        # Current price ratio: buy_token/sell_token at given prices
        current_ratio = price_buy / price_sell

        # Check if order's limit is satisfied (gets at least limit price)
        if current_ratio >= limit_price:
            eligible_orders.append((order, limit_price))

    if not eligible_orders:
        return None

    n_orders = len(eligible_orders)

    # Get all tokens involved
    all_tokens: set[str] = set()
    for order, _ in eligible_orders:
        all_tokens.add(normalize_address(order.sell_token))
        all_tokens.add(normalize_address(order.buy_token))
    token_list = sorted(all_tokens)

    # Objective: maximize sum(x_i * sell_amount_i)
    # linprog minimizes, so we negate
    c = [-order.sell_amount_int for order, _ in eligible_orders]

    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1) for _ in range(n_orders)]

    # Conservation constraints: for each token t,
    # sum(buys of t) - sum(sells of t) <= 0  (i.e., sells >= buys)
    A_ub = []
    b_ub = []

    for token in token_list:
        row = [0.0] * n_orders
        for i, (order, _) in enumerate(eligible_orders):
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

    for i, (order, _) in enumerate(eligible_orders):
        fill_ratio = result.x[i]
        if fill_ratio < 0.001:  # Skip negligible fills
            continue

        # Calculate fill amounts preserving the order's limit price
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


__all__ = [
    "PriceCandidates",
    "LPResult",
    "build_price_candidates",
    "build_price_candidates_from_orders",
    "enumerate_price_combinations",
    "solve_fills_at_prices",
    "build_token_graph_from_groups",
    "build_token_graph_from_orders",
    "solve_fills_at_prices_v2",
]
