"""Hybrid CoW+AMM auction algorithm.

This module extends the pure double auction by integrating AMM reference
prices for optimal clearing decisions.

IMPORTANT: All financial calculations use exact integer arithmetic via SafeInt.
No tolerance/epsilon comparisons are allowed.
"""

from __future__ import annotations

import decimal
from decimal import ROUND_HALF_UP, Decimal

import structlog

from solver.models.auction import Order
from solver.models.order_groups import OrderGroup
from solver.safe_int import S
from solver.strategies.decimal_utils import (
    DECIMAL_HIGH_PREC_CONTEXT,
    decimal_gt,
    decimal_le,
    decimal_lt,
)

from .core import (
    PriceRatio,
    _exact_price_key,
    _price_ratio_to_decimal,
    get_limit_price_ratio,
    run_double_auction,
)
from .types import (
    AMMRoute,
    DoubleAuctionMatch,
    HybridAuctionResult,
)

logger = structlog.get_logger()


# Scale factor for converting Decimal AMM prices to integer ratios
AMM_PRICE_SCALE = 10**18


def run_hybrid_auction(
    group: OrderGroup,
    amm_price: Decimal | None = None,
    respect_fill_or_kill: bool = True,
    ebbo_min: Decimal | None = None,
    ebbo_max: Decimal | None = None,
) -> HybridAuctionResult:
    """Run hybrid CoW+AMM auction on an order group.

    This extends the pure double auction by using the AMM reference price
    to determine which orders can be matched directly (CoW) vs which
    should route through AMM.

    The algorithm:
    1. If no AMM price, run pure double auction with EBBO bounds
    2. With AMM price:
       - Validate AMM price is within EBBO bounds
       - Sort asks ascending, bids descending by limit price
       - Match orders where: ask_limit <= AMM price <= bid_limit
       - Use AMM price as clearing price (fair reference)
       - Route remaining orders to AMM

    The key insight is that orders with overlapping limit prices
    relative to the AMM price can be matched directly, capturing
    the gas savings of CoW while ensuring fair execution.

    Args:
        group: OrderGroup with orders in both directions
        amm_price: Reference price from AMM (B per A). If None,
                   falls back to pure double auction behavior.
        respect_fill_or_kill: If True, skip non-partially-fillable orders
                              that would be only partially filled
        ebbo_min: Minimum clearing price (EBBO floor for sellers of A).
                  Derived from AMM rate for A→B direction.
        ebbo_max: Maximum clearing price (EBBO ceiling for buyers of A).
                  Derived from inverse of AMM rate for B→A direction.

    Returns:
        HybridAuctionResult with CoW matches and AMM routes
    """
    # Handle empty groups
    if not group.sellers_of_a and not group.sellers_of_b:
        return HybridAuctionResult(
            cow_matches=[],
            amm_routes=[],
            clearing_price=None,
            total_cow_a=0,
            total_cow_b=0,
        )

    # If no AMM price, use pure double auction with EBBO bounds
    if amm_price is None:
        pure_result = run_double_auction(
            group, respect_fill_or_kill, ebbo_min=ebbo_min, ebbo_max=ebbo_max
        )
        amm_routes = []

        # Convert unmatched orders to AMM routes
        for order, remaining in pure_result.unmatched_sellers:
            if remaining > 0:
                amm_routes.append(AMMRoute(order=order, amount=remaining, is_selling_a=True))
        for order, remaining in pure_result.unmatched_buyers:
            if remaining > 0:
                amm_routes.append(AMMRoute(order=order, amount=remaining, is_selling_a=False))

        return HybridAuctionResult(
            cow_matches=pure_result.matches,
            amm_routes=amm_routes,
            clearing_price=pure_result.clearing_price,
            total_cow_a=pure_result.total_a_matched,
            total_cow_b=pure_result.total_b_matched,
        )

    # Validate AMM price is positive
    # Use high-precision comparison for exactness
    if decimal_le(amm_price, Decimal(0)):
        logger.warning(
            "hybrid_auction_invalid_amm_price",
            amm_price=float(amm_price),
        )
        # Fall back to pure auction behavior with EBBO bounds
        return run_hybrid_auction(
            group,
            amm_price=None,
            respect_fill_or_kill=respect_fill_or_kill,
            ebbo_min=ebbo_min,
            ebbo_max=ebbo_max,
        )

    # Validate AMM price is within EBBO bounds
    # - ebbo_min: sellers of A must get at least this rate
    # - ebbo_max: buyers of A must pay at most this rate
    # Use high-precision comparison for exactness
    if ebbo_min is not None and decimal_lt(amm_price, ebbo_min):
        logger.debug(
            "hybrid_auction_amm_below_ebbo_min",
            amm_price=float(amm_price),
            ebbo_min=float(ebbo_min),
        )
        # AMM price doesn't satisfy seller EBBO - no compliant match possible
        return HybridAuctionResult(
            cow_matches=[],
            amm_routes=[
                AMMRoute(order=o, amount=o.sell_amount_int, is_selling_a=True)
                for o in group.sellers_of_a
            ]
            + [
                AMMRoute(order=o, amount=o.sell_amount_int, is_selling_a=False)
                for o in group.sellers_of_b
            ],
            clearing_price=None,
            total_cow_a=0,
            total_cow_b=0,
        )

    if ebbo_max is not None and decimal_gt(amm_price, ebbo_max):
        logger.debug(
            "hybrid_auction_amm_above_ebbo_max",
            amm_price=float(amm_price),
            ebbo_max=float(ebbo_max),
        )
        # AMM price doesn't satisfy buyer EBBO - no compliant match possible
        return HybridAuctionResult(
            cow_matches=[],
            amm_routes=[
                AMMRoute(order=o, amount=o.sell_amount_int, is_selling_a=True)
                for o in group.sellers_of_a
            ]
            + [
                AMMRoute(order=o, amount=o.sell_amount_int, is_selling_a=False)
                for o in group.sellers_of_b
            ],
            clearing_price=None,
            total_cow_a=0,
            total_cow_b=0,
        )

    # With AMM price: match orders that can clear at AMM price
    # Convert AMM price to integer ratio for exact calculations
    # Use round-to-nearest for reference price conversion
    # Use high-precision context to handle very large AMM prices (e.g., 10^11 * 10^18 = 10^29)
    with decimal.localcontext(DECIMAL_HIGH_PREC_CONTEXT):
        amm_price_scaled = (amm_price * AMM_PRICE_SCALE).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    amm_price_ratio: PriceRatio = (int(amm_price_scaled), AMM_PRICE_SCALE)
    amm_num, amm_denom = amm_price_ratio

    # Sort asks ascending (cheapest sellers first) - use integer ratios
    asks_raw = [
        (order, get_limit_price_ratio(order, is_selling_a=True), order.sell_amount_int)
        for order in group.sellers_of_a
    ]
    asks: list[tuple[Order, PriceRatio, int]] = [(o, p, a) for o, p, a in asks_raw if p is not None]
    invalid_asks = [o for o, p, _ in asks_raw if p is None]
    # Sort by price ratio ascending (cheapest first) using exact comparison
    asks.sort(key=lambda x: _exact_price_key(x[1]))

    # Sort bids descending (highest bidders first)
    bids_raw = [
        (order, get_limit_price_ratio(order, is_selling_a=False), order.sell_amount_int)
        for order in group.sellers_of_b
    ]
    bids: list[tuple[Order, PriceRatio, int]] = [(o, p, a) for o, p, a in bids_raw if p is not None]
    invalid_bids = [o for o, p, _ in bids_raw if p is None]
    # Sort by price ratio descending (highest bidders first) using exact comparison
    bids.sort(key=lambda x: _exact_price_key(x[1]), reverse=True)

    # Log invalid orders
    if invalid_asks or invalid_bids:
        logger.warning(
            "hybrid_auction_invalid_orders",
            invalid_asks=len(invalid_asks),
            invalid_bids=len(invalid_bids),
        )

    # Track remaining amounts
    ask_remaining = {order.uid: amount for order, _, amount in asks}
    bid_remaining = {order.uid: amount for order, _, amount in bids}

    matches: list[DoubleAuctionMatch] = []
    total_a_matched = 0
    total_b_matched = 0

    # Filter to orders that CAN trade at AMM price:
    # - Asks with limit <= AMM price (willing to sell at or below AMM)
    # - Bids with limit >= AMM price (willing to buy at or above AMM)
    # Use integer cross-multiplication: ask_num/ask_denom <= amm_num/amm_denom
    # iff ask_num * amm_denom <= amm_num * ask_denom
    matchable_asks = [
        (o, p, a) for o, p, a in asks if S(p[0]) * S(amm_denom) <= S(amm_num) * S(p[1])
    ]
    matchable_bids = [
        (o, p, a) for o, p, a in bids if S(p[0]) * S(amm_denom) >= S(amm_num) * S(p[1])
    ]

    # Match orders at AMM price
    ask_idx = 0
    bid_idx = 0

    while ask_idx < len(matchable_asks) and bid_idx < len(matchable_bids):
        ask_order, ask_limit, _ = matchable_asks[ask_idx]
        bid_order, bid_limit, _ = matchable_bids[bid_idx]

        ask_remaining_amount = ask_remaining[ask_order.uid]
        bid_remaining_amount = bid_remaining[bid_order.uid]

        if ask_remaining_amount <= 0:
            ask_idx += 1
            continue
        if bid_remaining_amount <= 0:
            bid_idx += 1
            continue

        # At AMM price, bid can buy this much A with remaining B
        # bid_can_buy_a = bid_remaining / (amm_num / amm_denom) = bid_remaining * amm_denom / amm_num
        if amm_num <= 0:
            bid_idx += 1
            continue
        bid_can_buy_a = (S(bid_remaining_amount) * S(amm_denom)) // S(amm_num)

        # Match amount
        match_a = min(ask_remaining_amount, bid_can_buy_a.value)

        if match_a <= 0:
            bid_idx += 1
            continue

        # Check fill-or-kill constraints
        if respect_fill_or_kill:
            if match_a < ask_remaining_amount and not ask_order.partially_fillable:
                # Ask can't be partially filled, skip to next bid
                bid_idx += 1
                continue
            # bid_fill_amount = match_a * amm_num / amm_denom (rounded up for EBBO)
            bid_fill_amount = (S(match_a) * S(amm_num) + S(amm_denom) - S(1)) // S(amm_denom)
            if bid_fill_amount.value < bid_remaining_amount and not bid_order.partially_fillable:
                # Bid can't be partially filled, skip to next ask
                ask_idx += 1
                continue

        # Calculate B amount at AMM price: match_b = match_a * amm_num / amm_denom
        # IMPORTANT: Round UP to ensure EBBO compliance (seller gets at least AMM rate)
        # ceil(a*b/c) = (a*b + c - 1) // c
        match_b = (S(match_a) * S(amm_num) + S(amm_denom) - S(1)) // S(amm_denom)

        if match_b.value <= 0:
            bid_idx += 1
            continue

        # Verify limit prices are satisfied using integer cross-multiplication
        # Ask limit: match_b / match_a >= ask_num / ask_denom
        # Cross multiply: match_b * ask_denom >= ask_num * match_a
        ask_num, ask_denom = ask_limit
        seller_satisfied = S(match_b.value) * S(ask_denom) >= S(ask_num) * S(match_a)
        if not seller_satisfied:
            logger.debug(
                "hybrid_auction_ask_limit_violation",
                match_a=match_a,
                match_b=match_b.value,
                limit=float(_price_ratio_to_decimal(ask_limit)),
            )
            ask_idx += 1
            continue

        # Bid limit: match_b / match_a <= bid_num / bid_denom
        # Cross multiply: match_b * bid_denom <= bid_num * match_a
        bid_num, bid_denom = bid_limit
        buyer_satisfied = S(match_b.value) * S(bid_denom) <= S(bid_num) * S(match_a)
        if not buyer_satisfied:
            logger.debug(
                "hybrid_auction_bid_limit_violation",
                match_a=match_a,
                match_b=match_b.value,
                limit=float(_price_ratio_to_decimal(bid_limit)),
            )
            bid_idx += 1
            continue

        # Record match
        match = DoubleAuctionMatch(
            seller=ask_order,
            buyer=bid_order,
            amount_a=match_a,
            amount_b=match_b.value,
            clearing_price=amm_price,
        )
        matches.append(match)

        # Update remaining
        ask_remaining[ask_order.uid] -= match_a
        bid_remaining[bid_order.uid] -= match_b.value
        total_a_matched += match_a
        total_b_matched += match_b.value

        # Advance indices
        if ask_remaining[ask_order.uid] <= 0:
            ask_idx += 1
        if bid_remaining[bid_order.uid] <= 0:
            bid_idx += 1

    # Collect orders for AMM routing
    amm_routes = []

    # Unmatched asks -> route to AMM
    for order, _, _ in asks:
        remaining = ask_remaining[order.uid]
        if remaining > 0:
            amm_routes.append(AMMRoute(order=order, amount=remaining, is_selling_a=True))

    # Invalid asks -> route to AMM (will likely fail but shouldn't be silently dropped)
    for order in invalid_asks:
        amm_routes.append(AMMRoute(order=order, amount=order.sell_amount_int, is_selling_a=True))

    # Unmatched bids -> route to AMM
    for order, _, _ in bids:
        remaining = bid_remaining[order.uid]
        if remaining > 0:
            amm_routes.append(AMMRoute(order=order, amount=remaining, is_selling_a=False))

    # Invalid bids -> route to AMM
    for order in invalid_bids:
        amm_routes.append(AMMRoute(order=order, amount=order.sell_amount_int, is_selling_a=False))

    logger.info(
        "hybrid_auction_complete",
        token_pair=f"{group.token_a[-8:]}/{group.token_b[-8:]}",
        amm_price=float(amm_price),
        cow_matches=len(matches),
        total_cow_a=total_a_matched,
        total_cow_b=total_b_matched,
        amm_routes=len(amm_routes),
    )

    return HybridAuctionResult(
        cow_matches=matches,
        amm_routes=amm_routes,
        clearing_price=amm_price if matches else None,
        total_cow_a=total_a_matched,
        total_cow_b=total_b_matched,
    )


__all__ = ["run_hybrid_auction"]
