"""Core double auction algorithm for multi-order CoW matching.

This module implements the double auction clearing mechanism for N-order
CoW matching on a single token pair. This is the core algorithm for
Phase 4 multi-order optimization.

Algorithm:
1. Sort sellers_of_a by limit price (ascending - cheapest sellers first)
2. Sort sellers_of_b by limit price (descending - highest bidders first)
3. Match orders from both sides until prices cross
4. Handle partial fills respecting fill-or-kill constraints

The double auction is optimal for single-pair matching because:
- It maximizes matched volume at the equilibrium price
- It ensures all matched orders get prices at least as good as their limits
- It can be computed in O(n log n) time
"""

from __future__ import annotations

from decimal import Decimal

import structlog

from solver.models.auction import Order
from solver.models.order_groups import OrderGroup

from .types import (
    DoubleAuctionMatch,
    DoubleAuctionResult,
    MatchingAtPriceResult,
)

logger = structlog.get_logger()


def get_limit_price(order: Order, is_selling_a: bool) -> Decimal | None:
    """Calculate the limit price for an order in terms of B/A.

    For sellers of A (selling A to get B):
        min acceptable price = buy_amount / sell_amount (B per A they need)

    For sellers of B (selling B to get A):
        max acceptable price = sell_amount / buy_amount (B per A they'll pay)

    Args:
        order: The order to calculate limit price for
        is_selling_a: True if order is selling token A

    Returns:
        Limit price as Decimal (B per A), or None if invalid order
    """
    sell = Decimal(order.sell_amount_int)
    buy = Decimal(order.buy_amount_int)

    # Guard against division by zero
    if sell <= 0 or buy <= 0:
        return None

    if is_selling_a:
        # Selling A, wanting B: limit = min B/A acceptable
        # They sell `sell` A and want at least `buy` B
        # Price = buy / sell (B per A they need to receive)
        return buy / sell
    else:
        # Selling B, wanting A: limit = max B/A they'll pay
        # They sell `sell` B and want at least `buy` A
        # Price = sell / buy (B per A they're willing to pay)
        return sell / buy


def _execute_matches_at_price(
    clearing_price: Decimal,
    asks: list[tuple[Order, Decimal, int]],
    bids: list[tuple[Order, Decimal, int]],
    respect_fill_or_kill: bool,
) -> MatchingAtPriceResult:
    """Execute matching at a given clearing price.

    This is a helper function for run_double_auction that tries to match
    orders at a specific price.

    Args:
        clearing_price: The price to use for matching (B per A)
        asks: Sorted list of (order, limit_price, amount) for sellers of A
        bids: Sorted list of (order, limit_price, amount) for sellers of B
        respect_fill_or_kill: If True, skip orders that would be partially filled

    Returns:
        MatchingAtPriceResult with matches and remaining amounts
    """
    ask_remaining = {order.uid: amount for order, _, amount in asks}
    bid_remaining = {order.uid: amount for order, _, amount in bids}

    matches: list[DoubleAuctionMatch] = []
    total_a_matched = 0
    total_b_matched = 0

    # Only match orders whose limits are satisfied by the clearing price
    matchable_asks = [(o, p, a) for o, p, a in asks if p <= clearing_price]
    matchable_bids = [(o, p, a) for o, p, a in bids if p >= clearing_price]

    ask_idx = 0
    bid_idx = 0

    while ask_idx < len(matchable_asks) and bid_idx < len(matchable_bids):
        ask_order, ask_price, _ = matchable_asks[ask_idx]
        bid_order, bid_price, _ = matchable_bids[bid_idx]

        # Get remaining amounts
        ask_remaining_amount = ask_remaining[ask_order.uid]
        bid_remaining_amount = bid_remaining[bid_order.uid]

        if ask_remaining_amount <= 0:
            ask_idx += 1
            continue

        if bid_remaining_amount <= 0:
            bid_idx += 1
            continue

        # Amount of A the bid can buy with remaining B at clearing price
        bid_can_buy_a = int(Decimal(bid_remaining_amount) / clearing_price)

        # Match is minimum of what ask has and what bid can buy
        match_a = min(ask_remaining_amount, bid_can_buy_a)

        if match_a <= 0:
            bid_idx += 1
            continue

        # Check fill-or-kill constraints
        if respect_fill_or_kill:
            # Would ask be partially filled?
            if match_a < ask_remaining_amount and not ask_order.partially_fillable:
                # This ask can't be partially filled, skip to next bid
                bid_idx += 1
                continue

            # Would bid be partially filled?
            bid_fill_amount = int(Decimal(match_a) * clearing_price)
            if bid_fill_amount < bid_remaining_amount and not bid_order.partially_fillable:
                # This bid can't be partially filled, skip to next ask
                ask_idx += 1
                continue

        # Calculate B amount for this match at uniform clearing price
        match_b = int(Decimal(match_a) * clearing_price)

        if match_b <= 0:
            bid_idx += 1
            continue

        # Verify limit prices are satisfied after integer truncation
        actual_price = Decimal(match_b) / Decimal(match_a)
        if actual_price < ask_price:
            logger.debug(
                "execute_matches_ask_limit_violation",
                clearing_price=float(clearing_price),
                actual_price=float(actual_price),
                ask_limit=float(ask_price),
            )
            ask_idx += 1
            continue

        if actual_price > bid_price:
            logger.debug(
                "execute_matches_bid_limit_violation",
                clearing_price=float(clearing_price),
                actual_price=float(actual_price),
                bid_limit=float(bid_price),
            )
            bid_idx += 1
            continue

        # Record match
        match = DoubleAuctionMatch(
            seller=ask_order,
            buyer=bid_order,
            amount_a=match_a,
            amount_b=match_b,
            clearing_price=clearing_price,
        )
        matches.append(match)

        # Update remaining amounts
        ask_remaining[ask_order.uid] -= match_a
        bid_remaining[bid_order.uid] -= match_b
        total_a_matched += match_a
        total_b_matched += match_b

        # Move to next order if current is exhausted
        if ask_remaining[ask_order.uid] <= 0:
            ask_idx += 1
        if bid_remaining[bid_order.uid] <= 0:
            bid_idx += 1

    return MatchingAtPriceResult(
        matches=matches,
        ask_remaining=ask_remaining,
        bid_remaining=bid_remaining,
        total_a_matched=total_a_matched,
        total_b_matched=total_b_matched,
    )


def run_double_auction(
    group: OrderGroup,
    respect_fill_or_kill: bool = True,
    ebbo_min: Decimal | None = None,
    ebbo_max: Decimal | None = None,
) -> DoubleAuctionResult:
    """Run double auction clearing on an order group.

    Implements a two-pass double auction algorithm:
    1. First pass: Find the valid clearing price range from matchable orders
    2. Intersect with EBBO bounds (if provided)
    3. Try candidate prices: midpoint, then boundary prices if needed
    4. Select the price that maximizes matched volume

    Using a uniform clearing price is essential for valid CoW Protocol settlement.
    Each trade must satisfy: prices[sell_token] * sell = prices[buy_token] * buy,
    which requires all matches to execute at the same price.

    For fill-or-kill orders, the midpoint price may not allow complete fills.
    In this case, boundary prices (max_ask_limit or min_bid_limit) are tried,
    which may allow complete fills by favoring one side.

    Args:
        group: OrderGroup with orders in both directions
        respect_fill_or_kill: If True, skip non-partially-fillable orders
                              that would be only partially filled
        ebbo_min: Minimum valid clearing price (EBBO floor for sellers).
                  If provided, clearing_price must be >= ebbo_min.
        ebbo_max: Maximum valid clearing price (EBBO ceiling for buyers).
                  If provided, clearing_price must be <= ebbo_max.

    Returns:
        DoubleAuctionResult with all matches and remainders
    """
    if not group.has_cow_potential:
        return DoubleAuctionResult(
            matches=[],
            total_a_matched=0,
            total_b_matched=0,
            clearing_price=None,
            unmatched_sellers=[(o, o.sell_amount_int) for o in group.sellers_of_a],
            unmatched_buyers=[(o, o.sell_amount_int) for o in group.sellers_of_b],
        )

    # Calculate limit prices and sort
    # Sellers of A: sorted ascending (cheapest first)
    asks_raw = [
        (order, get_limit_price(order, is_selling_a=True), order.sell_amount_int)
        for order in group.sellers_of_a
    ]
    # Separate valid and invalid orders
    asks = [(o, p, a) for o, p, a in asks_raw if p is not None]
    invalid_asks = [(o, a) for o, p, a in asks_raw if p is None]
    asks.sort(key=lambda x: x[1])

    # Sellers of B: sorted descending (highest bidders first)
    bids_raw = [
        (order, get_limit_price(order, is_selling_a=False), order.sell_amount_int)
        for order in group.sellers_of_b
    ]
    # Separate valid and invalid orders
    bids = [(o, p, a) for o, p, a in bids_raw if p is not None]
    invalid_bids = [(o, a) for o, p, a in bids_raw if p is None]
    bids.sort(key=lambda x: x[1], reverse=True)

    # Log invalid orders (zero/negative amounts)
    if invalid_asks or invalid_bids:
        logger.warning(
            "double_auction_invalid_orders",
            invalid_asks=len(invalid_asks),
            invalid_bids=len(invalid_bids),
        )

    logger.debug(
        "double_auction_start",
        token_a=group.token_a[-8:],
        token_b=group.token_b[-8:],
        num_asks=len(asks),
        num_bids=len(bids),
        best_ask=float(asks[0][1]) if asks else None,
        best_bid=float(bids[0][1]) if bids else None,
    )

    # ========================================================================
    # FIRST PASS: Determine the uniform clearing price
    # ========================================================================
    # Find all matchable pairs (ask_limit <= bid_limit) and compute valid price range.
    # The clearing price must satisfy ALL participants:
    # - All matched asks need: clearing_price >= their limit (they get enough B per A)
    # - All matched bids need: clearing_price <= their limit (they don't pay too much)
    #
    # Valid range: [max(matched_ask_limits), min(matched_bid_limits)]

    # Find orders that could potentially match (have overlapping prices)
    matchable_ask_limits: list[Decimal] = []
    matchable_bid_limits: list[Decimal] = []

    # Simulate matching to find which orders participate
    temp_ask_remaining = {order.uid: amount for order, _, amount in asks}
    temp_bid_remaining = {order.uid: amount for order, _, amount in bids}
    ask_idx = 0
    bid_idx = 0

    while ask_idx < len(asks) and bid_idx < len(bids):
        ask_order, ask_price, _ = asks[ask_idx]
        bid_order, bid_price, _ = bids[bid_idx]

        # Prices crossed - no more matches
        if ask_price > bid_price:
            break

        ask_amt = temp_ask_remaining[ask_order.uid]
        bid_amt = temp_bid_remaining[bid_order.uid]

        if ask_amt <= 0:
            ask_idx += 1
            continue
        if bid_amt <= 0:
            bid_idx += 1
            continue

        # Use midpoint temporarily to estimate fill amounts
        temp_price = (ask_price + bid_price) / 2
        bid_can_buy_a = int(Decimal(bid_amt) / temp_price)
        match_a = min(ask_amt, bid_can_buy_a)

        if match_a <= 0:
            bid_idx += 1
            continue

        match_b = int(Decimal(match_a) * temp_price)

        if match_b <= 0:
            # Price too small for meaningful B amount, try next bid
            bid_idx += 1
            continue

        # Only record limits AFTER confirming match is valid (match_a > 0 and match_b > 0)
        # This ensures the clearing price is only influenced by orders that actually match
        matchable_ask_limits.append(ask_price)
        matchable_bid_limits.append(bid_price)

        temp_ask_remaining[ask_order.uid] -= match_a
        temp_bid_remaining[bid_order.uid] -= match_b

        # Advance indices when orders are exhausted
        if temp_ask_remaining[ask_order.uid] <= 0:
            ask_idx += 1
        if temp_bid_remaining[bid_order.uid] <= 0:
            bid_idx += 1

    # Compute uniform clearing price from valid range
    if not matchable_ask_limits or not matchable_bid_limits:
        # No matches possible
        return DoubleAuctionResult(
            matches=[],
            total_a_matched=0,
            total_b_matched=0,
            clearing_price=None,
            unmatched_sellers=[(o, a) for o, _, a in asks] + list(invalid_asks),
            unmatched_buyers=[(o, a) for o, _, a in bids] + list(invalid_bids),
        )

    # Valid clearing price range: [max_ask_limit, min_bid_limit]
    max_ask_limit = max(matchable_ask_limits)
    min_bid_limit = min(matchable_bid_limits)

    if max_ask_limit > min_bid_limit:
        # This shouldn't happen if we found matchable pairs, but handle defensively
        logger.warning(
            "double_auction_invalid_price_range",
            max_ask=float(max_ask_limit),
            min_bid=float(min_bid_limit),
        )
        return DoubleAuctionResult(
            matches=[],
            total_a_matched=0,
            total_b_matched=0,
            clearing_price=None,
            unmatched_sellers=[(o, a) for o, _, a in asks] + list(invalid_asks),
            unmatched_buyers=[(o, a) for o, _, a in bids] + list(invalid_bids),
        )

    # ========================================================================
    # INTERSECT WITH EBBO BOUNDS
    # ========================================================================
    # If EBBO bounds are provided, constrain the valid price range upfront.
    # This avoids wasted computation on matches that would fail EBBO validation.
    #
    # - ebbo_min: Sellers must get at least this rate (price >= ebbo_min)
    # - ebbo_max: Buyers must pay at most this rate (price <= ebbo_max)

    price_floor = max_ask_limit
    price_ceiling = min_bid_limit

    if ebbo_min is not None:
        price_floor = max(price_floor, ebbo_min)
    if ebbo_max is not None:
        price_ceiling = min(price_ceiling, ebbo_max)

    if price_floor > price_ceiling:
        # No EBBO-compliant match possible
        logger.debug(
            "double_auction_ebbo_no_valid_range",
            price_floor=float(price_floor),
            price_ceiling=float(price_ceiling),
            ebbo_min=float(ebbo_min) if ebbo_min else None,
            ebbo_max=float(ebbo_max) if ebbo_max else None,
        )
        return DoubleAuctionResult(
            matches=[],
            total_a_matched=0,
            total_b_matched=0,
            clearing_price=None,
            unmatched_sellers=[(o, a) for o, _, a in asks] + list(invalid_asks),
            unmatched_buyers=[(o, a) for o, _, a in bids] + list(invalid_bids),
        )

    # ========================================================================
    # SECOND PASS: Try candidate prices and select the best one
    # ========================================================================
    # For fill-or-kill orders, the midpoint price may cause partial fills that
    # violate constraints. We try multiple candidate prices and select the one
    # that maximizes matched volume.
    #
    # Candidate prices (within EBBO-constrained range):
    # 1. Midpoint (fair to both sides)
    # 2. price_floor (favors sellers / EBBO minimum)
    # 3. price_ceiling (favors buyers / EBBO maximum)

    midpoint_price = (price_floor + price_ceiling) / 2
    candidate_prices = [midpoint_price, price_floor, price_ceiling]

    # Remove duplicates while preserving order (midpoint first)
    seen: set[Decimal] = set()
    unique_prices: list[Decimal] = []
    for p in candidate_prices:
        if p not in seen:
            seen.add(p)
            unique_prices.append(p)

    logger.debug(
        "double_auction_candidate_prices",
        max_ask_limit=float(max_ask_limit),
        min_bid_limit=float(min_bid_limit),
        midpoint=float(midpoint_price),
        num_candidates=len(unique_prices),
    )

    # Try each candidate price and track the best result
    # Initialize with original amounts (used when no matches happen)
    best_result = MatchingAtPriceResult(
        matches=[],
        ask_remaining={order.uid: amount for order, _, amount in asks},
        bid_remaining={order.uid: amount for order, _, amount in bids},
        total_a_matched=0,
        total_b_matched=0,
    )
    best_price: Decimal | None = None

    for candidate_price in unique_prices:
        result = _execute_matches_at_price(
            clearing_price=candidate_price,
            asks=asks,
            bids=bids,
            respect_fill_or_kill=respect_fill_or_kill,
        )

        # Select this price if it gives better volume (measured by token A matched)
        if result.total_a_matched > best_result.total_a_matched:
            best_result = result
            best_price = candidate_price

            logger.debug(
                "double_auction_new_best_price",
                price=float(candidate_price),
                total_a_matched=result.total_a_matched,
                num_matches=len(result.matches),
            )

    # Use the best result
    matches = best_result.matches
    ask_remaining = best_result.ask_remaining
    bid_remaining = best_result.bid_remaining
    total_a_matched = best_result.total_a_matched
    total_b_matched = best_result.total_b_matched
    clearing_price = best_price if best_price is not None else midpoint_price

    # Log individual matches for debugging
    for match in matches:
        logger.debug(
            "double_auction_match",
            seller=match.seller.uid[:12] + "...",
            buyer=match.buyer.uid[:12] + "...",
            amount_a=match.amount_a,
            amount_b=match.amount_b,
            price=float(match.clearing_price),
        )

    # Collect unmatched orders (including invalid orders which couldn't participate)
    unmatched_sellers = [
        (order, ask_remaining[order.uid]) for order, _, _ in asks if ask_remaining[order.uid] > 0
    ]
    # Add invalid asks to unmatched (they have invalid amounts but should still be tracked)
    unmatched_sellers.extend(invalid_asks)

    unmatched_buyers = [
        (order, bid_remaining[order.uid]) for order, _, _ in bids if bid_remaining[order.uid] > 0
    ]
    # Add invalid bids to unmatched
    unmatched_buyers.extend(invalid_bids)

    logger.info(
        "double_auction_complete",
        token_pair=f"{group.token_a[-8:]}/{group.token_b[-8:]}",
        num_matches=len(matches),
        total_a_matched=total_a_matched,
        total_b_matched=total_b_matched,
        clearing_price=float(clearing_price) if matches else None,
        unmatched_sellers=len(unmatched_sellers),
        unmatched_buyers=len(unmatched_buyers),
    )

    return DoubleAuctionResult(
        matches=matches,
        total_a_matched=total_a_matched,
        total_b_matched=total_b_matched,
        clearing_price=clearing_price if matches else None,
        unmatched_sellers=unmatched_sellers,
        unmatched_buyers=unmatched_buyers,
    )


def calculate_surplus(result: DoubleAuctionResult) -> int:
    """Calculate total surplus generated by the double auction.

    Surplus = sum of (what users got - what they minimally required)

    For each match:
    - Seller surplus: amount_b received - (amount_a * ask_limit_price)
    - Buyer surplus: (bid_limit_price * amount_a) - amount_b paid

    Returns:
        Total surplus in token B units (can be converted to USD later)
    """
    total_surplus = 0

    for match in result.matches:
        # Seller wanted at least ask_price B per A
        ask_price = get_limit_price(match.seller, is_selling_a=True)
        if ask_price is None:
            continue
        seller_min_b = int(Decimal(match.amount_a) * ask_price)
        seller_surplus = match.amount_b - seller_min_b

        # Buyer was willing to pay up to bid_price B per A
        bid_price = get_limit_price(match.buyer, is_selling_a=False)
        if bid_price is None:
            continue
        buyer_max_b = int(Decimal(match.amount_a) * bid_price)
        buyer_surplus = buyer_max_b - match.amount_b

        total_surplus += seller_surplus + buyer_surplus

    return total_surplus


__all__ = [
    "get_limit_price",
    "run_double_auction",
    "calculate_surplus",
]
