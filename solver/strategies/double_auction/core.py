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

IMPORTANT: All financial calculations use exact integer arithmetic via SafeInt.
No tolerance/epsilon comparisons are allowed. Price ratios are represented as
(numerator, denominator) integer pairs to avoid floating-point precision issues.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_UP, Decimal

import structlog

from solver.models.auction import Order
from solver.models.order_groups import OrderGroup
from solver.safe_int import S

from .types import (
    DoubleAuctionMatch,
    DoubleAuctionResult,
    _MatchingAtPriceResult,
)

logger = structlog.get_logger()

# Type alias for price ratios: (numerator, denominator) representing num/denom
# For asks (sellers of A): price = buy_amount / sell_amount (B per A they want)
# For bids (sellers of B): price = sell_amount / buy_amount (B per A they'll pay)
PriceRatio = tuple[int, int]


def _price_ratio_to_decimal(ratio: PriceRatio) -> Decimal:
    """Convert a price ratio to Decimal for logging/storage."""
    num, denom = ratio
    if denom == 0:
        return Decimal("0")
    return Decimal(num) / Decimal(denom)


def _compare_prices(p1: PriceRatio, p2: PriceRatio) -> int:
    """Compare two price ratios using integer cross-multiplication.

    Returns:
        -1 if p1 < p2
         0 if p1 == p2
         1 if p1 > p2
    """
    # p1 = n1/d1, p2 = n2/d2
    # p1 < p2 iff n1*d2 < n2*d1
    n1, d1 = p1
    n2, d2 = p2
    lhs = S(n1) * S(d2)
    rhs = S(n2) * S(d1)
    if lhs < rhs:
        return -1
    elif lhs > rhs:
        return 1
    return 0


def _midpoint_price(p1: PriceRatio, p2: PriceRatio) -> PriceRatio:
    """Compute midpoint of two price ratios.

    midpoint = (p1 + p2) / 2 = (n1/d1 + n2/d2) / 2 = (n1*d2 + n2*d1) / (2*d1*d2)
    """
    n1, d1 = p1
    n2, d2 = p2
    num = S(n1) * S(d2) + S(n2) * S(d1)
    denom = S(2) * S(d1) * S(d2)
    return (num.value, denom.value)


def _price_ratio_sort_key(ratio: PriceRatio) -> int:
    """Convert price ratio to integer sort key for list sorting.

    Uses 10^18 scaling which is sufficient for token amounts (max 18 decimals).
    """
    num, denom = ratio
    if denom <= 0:
        return 0
    return num * 10**18 // denom


def get_limit_price_ratio(order: Order, is_selling_a: bool) -> PriceRatio | None:
    """Calculate the limit price for an order as an integer ratio (B/A).

    For sellers of A (selling A to get B):
        min acceptable price = buy_amount / sell_amount (B per A they need)

    For sellers of B (selling B to get A):
        max acceptable price = sell_amount / buy_amount (B per A they'll pay)

    Args:
        order: The order to calculate limit price for
        is_selling_a: True if order is selling token A

    Returns:
        Price ratio as (numerator, denominator), or None if invalid order
    """
    sell = order.sell_amount_int
    buy = order.buy_amount_int

    # Guard against division by zero
    if sell <= 0 or buy <= 0:
        return None

    if is_selling_a:
        # Selling A, wanting B: limit = min B/A acceptable
        # They sell `sell` A and want at least `buy` B
        # Price = buy / sell (B per A they need to receive)
        return (buy, sell)
    else:
        # Selling B, wanting A: limit = max B/A they'll pay
        # They sell `sell` B and want at least `buy` A
        # Price = sell / buy (B per A they're willing to pay)
        return (sell, buy)


def get_limit_price(order: Order, is_selling_a: bool) -> Decimal | None:
    """Calculate the limit price for an order in terms of B/A.

    DEPRECATED: Use get_limit_price_ratio for exact integer arithmetic.
    This function is kept for backward compatibility with calculate_surplus.

    Returns:
        Limit price as Decimal (B per A), or None if invalid order
    """
    ratio = get_limit_price_ratio(order, is_selling_a)
    if ratio is None:
        return None
    return _price_ratio_to_decimal(ratio)


def _execute_matches_at_price(
    clearing_price: PriceRatio,
    asks: list[tuple[Order, PriceRatio, int]],
    bids: list[tuple[Order, PriceRatio, int]],
    respect_fill_or_kill: bool,
) -> _MatchingAtPriceResult:
    """Execute matching at a given clearing price using exact integer arithmetic.

    This is a helper function for run_double_auction that tries to match
    orders at a specific price.

    IMPORTANT: All calculations use SafeInt and integer cross-multiplication.
    No floating-point or tolerance comparisons.

    Args:
        clearing_price: Price ratio (num, denom) where price = num/denom (B per A)
        asks: Sorted list of (order, limit_ratio, amount) for sellers of A
        bids: Sorted list of (order, limit_ratio, amount) for sellers of B
        respect_fill_or_kill: If True, skip orders that would be partially filled

    Returns:
        _MatchingAtPriceResult with matches and remaining amounts
    """
    price_num, price_denom = clearing_price

    ask_remaining = {order.uid: amount for order, _, amount in asks}
    bid_remaining = {order.uid: amount for order, _, amount in bids}

    matches: list[DoubleAuctionMatch] = []
    total_a_matched = 0
    total_b_matched = 0

    # Only match orders whose limits are satisfied by the clearing price
    # Ask limit satisfied: clearing_price >= ask_limit (seller gets enough)
    # Bid limit satisfied: clearing_price <= bid_limit (buyer doesn't overpay)
    matchable_asks = [(o, p, a) for o, p, a in asks if _compare_prices(clearing_price, p) >= 0]
    matchable_bids = [(o, p, a) for o, p, a in bids if _compare_prices(clearing_price, p) <= 0]

    ask_idx = 0
    bid_idx = 0

    while ask_idx < len(matchable_asks) and bid_idx < len(matchable_bids):
        ask_order, ask_limit, _ = matchable_asks[ask_idx]
        bid_order, bid_limit, _ = matchable_bids[bid_idx]

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
        # bid_can_buy_a = bid_remaining_amount / (price_num / price_denom)
        #               = bid_remaining_amount * price_denom / price_num
        # Use floor division (buyer can buy at most this much)
        if price_num <= 0:
            bid_idx += 1
            continue
        bid_can_buy_a = (S(bid_remaining_amount) * S(price_denom)) // S(price_num)

        # Match is minimum of what ask has and what bid can buy
        match_a = min(ask_remaining_amount, bid_can_buy_a.value)

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
            # bid_fill_amount = match_a * price_num / price_denom (floor)
            bid_fill_amount = (S(match_a) * S(price_num)) // S(price_denom)
            if bid_fill_amount.value < bid_remaining_amount and not bid_order.partially_fillable:
                # This bid can't be partially filled, skip to next ask
                ask_idx += 1
                continue

        # Calculate B amount for this match at uniform clearing price
        # match_b = match_a * price_num / price_denom
        # Use floor division (conservative for buyer, but we'll verify seller limit)
        match_b = (S(match_a) * S(price_num)) // S(price_denom)

        if match_b.value <= 0:
            bid_idx += 1
            continue

        # Verify limit prices are satisfied using integer cross-multiplication
        # Seller (ask) limit: match_b / match_a >= ask_num / ask_denom
        # Cross multiply: match_b * ask_denom >= ask_num * match_a
        ask_num, ask_denom = ask_limit
        seller_satisfied = S(match_b.value) * S(ask_denom) >= S(ask_num) * S(match_a)
        if not seller_satisfied:
            logger.debug(
                "execute_matches_ask_limit_violation",
                clearing_price=float(_price_ratio_to_decimal(clearing_price)),
                match_a=match_a,
                match_b=match_b.value,
                ask_limit=float(_price_ratio_to_decimal(ask_limit)),
            )
            ask_idx += 1
            continue

        # Buyer (bid) limit: match_b / match_a <= bid_num / bid_denom
        # Cross multiply: match_b * bid_denom <= bid_num * match_a
        bid_num, bid_denom = bid_limit
        buyer_satisfied = S(match_b.value) * S(bid_denom) <= S(bid_num) * S(match_a)
        if not buyer_satisfied:
            logger.debug(
                "execute_matches_bid_limit_violation",
                clearing_price=float(_price_ratio_to_decimal(clearing_price)),
                match_a=match_a,
                match_b=match_b.value,
                bid_limit=float(_price_ratio_to_decimal(bid_limit)),
            )
            bid_idx += 1
            continue

        # Record match (convert price ratio to Decimal for storage)
        match = DoubleAuctionMatch(
            seller=ask_order,
            buyer=bid_order,
            amount_a=match_a,
            amount_b=match_b.value,
            clearing_price=_price_ratio_to_decimal(clearing_price),
        )
        matches.append(match)

        # Update remaining amounts
        ask_remaining[ask_order.uid] -= match_a
        bid_remaining[bid_order.uid] -= match_b.value
        total_a_matched += match_a
        total_b_matched += match_b.value

        # Move to next order if current is exhausted
        if ask_remaining[ask_order.uid] <= 0:
            ask_idx += 1
        if bid_remaining[bid_order.uid] <= 0:
            bid_idx += 1

    return _MatchingAtPriceResult(
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

    IMPORTANT: All internal calculations use exact integer arithmetic via SafeInt.
    EBBO bounds are converted to integer ratios for exact comparison.

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

    # Calculate limit prices as integer ratios and sort
    # Sellers of A: sorted ascending (cheapest first)
    asks_raw = [
        (order, get_limit_price_ratio(order, is_selling_a=True), order.sell_amount_int)
        for order in group.sellers_of_a
    ]
    # Separate valid and invalid orders
    asks: list[tuple[Order, PriceRatio, int]] = [(o, p, a) for o, p, a in asks_raw if p is not None]
    invalid_asks = [(o, a) for o, p, a in asks_raw if p is None]
    # Sort by price ratio ascending (cheapest first)
    asks.sort(key=lambda x: _price_ratio_sort_key(x[1]))

    # Sellers of B: sorted descending (highest bidders first)
    bids_raw = [
        (order, get_limit_price_ratio(order, is_selling_a=False), order.sell_amount_int)
        for order in group.sellers_of_b
    ]
    # Separate valid and invalid orders
    bids: list[tuple[Order, PriceRatio, int]] = [(o, p, a) for o, p, a in bids_raw if p is not None]
    invalid_bids = [(o, a) for o, p, a in bids_raw if p is None]
    # Sort by price ratio descending (highest bidders first)
    bids.sort(key=lambda x: _price_ratio_sort_key(x[1]), reverse=True)

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
        best_ask=float(_price_ratio_to_decimal(asks[0][1])) if asks else None,
        best_bid=float(_price_ratio_to_decimal(bids[0][1])) if bids else None,
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
    # All comparisons use integer cross-multiplication (no floating point).

    # Find orders that could potentially match (have overlapping prices)
    matchable_ask_limits: list[PriceRatio] = []
    matchable_bid_limits: list[PriceRatio] = []

    # Simulate matching to find which orders participate
    temp_ask_remaining = {order.uid: amount for order, _, amount in asks}
    temp_bid_remaining = {order.uid: amount for order, _, amount in bids}
    ask_idx = 0
    bid_idx = 0

    while ask_idx < len(asks) and bid_idx < len(bids):
        ask_order, ask_limit, _ = asks[ask_idx]
        bid_order, bid_limit, _ = bids[bid_idx]

        # Prices crossed - no more matches (ask_limit > bid_limit)
        if _compare_prices(ask_limit, bid_limit) > 0:
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
        temp_price = _midpoint_price(ask_limit, bid_limit)
        temp_num, temp_denom = temp_price
        if temp_num <= 0:
            bid_idx += 1
            continue

        # bid_can_buy_a = bid_amt / (temp_num / temp_denom) = bid_amt * temp_denom / temp_num
        bid_can_buy_a = (S(bid_amt) * S(temp_denom)) // S(temp_num)
        match_a = min(ask_amt, bid_can_buy_a.value)

        if match_a <= 0:
            bid_idx += 1
            continue

        # match_b = match_a * temp_num / temp_denom
        match_b = (S(match_a) * S(temp_num)) // S(temp_denom)

        if match_b.value <= 0:
            # Price too small for meaningful B amount, try next bid
            bid_idx += 1
            continue

        # Only record limits AFTER confirming match is valid (match_a > 0 and match_b > 0)
        # This ensures the clearing price is only influenced by orders that actually match
        matchable_ask_limits.append(ask_limit)
        matchable_bid_limits.append(bid_limit)

        temp_ask_remaining[ask_order.uid] -= match_a
        temp_bid_remaining[bid_order.uid] -= match_b.value

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
    # Find max ask limit (highest price sellers need)
    max_ask_limit = matchable_ask_limits[0]
    for limit in matchable_ask_limits[1:]:
        if _compare_prices(limit, max_ask_limit) > 0:
            max_ask_limit = limit

    # Find min bid limit (lowest price buyers accept)
    min_bid_limit = matchable_bid_limits[0]
    for limit in matchable_bid_limits[1:]:
        if _compare_prices(limit, min_bid_limit) < 0:
            min_bid_limit = limit

    if _compare_prices(max_ask_limit, min_bid_limit) > 0:
        # This shouldn't happen if we found matchable pairs, but handle defensively
        logger.warning(
            "double_auction_invalid_price_range",
            max_ask=float(_price_ratio_to_decimal(max_ask_limit)),
            min_bid=float(_price_ratio_to_decimal(min_bid_limit)),
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
    #
    # Convert Decimal EBBO bounds to integer ratios for exact comparison.
    # We scale by 10^18 to preserve precision.

    EBBO_SCALE = 10**18

    price_floor: PriceRatio = max_ask_limit
    price_ceiling: PriceRatio = min_bid_limit

    if ebbo_min is not None and ebbo_min > 0:
        # Convert Decimal to ratio: ebbo_min ≈ (ebbo_min * SCALE) / SCALE
        # For EBBO min (seller protection), round UP to be conservative
        # Sellers must get at least this rate, so we round up to not under-count
        ebbo_min_scaled = (ebbo_min * EBBO_SCALE).quantize(Decimal("1"), rounding=ROUND_UP)
        ebbo_min_ratio: PriceRatio = (int(ebbo_min_scaled), EBBO_SCALE)
        if _compare_prices(ebbo_min_ratio, price_floor) > 0:
            price_floor = ebbo_min_ratio

    if ebbo_max is not None and ebbo_max > 0:
        # For EBBO max (buyer protection), round DOWN to be conservative
        # Buyers must pay at most this rate, so we round down to not over-count
        ebbo_max_scaled = (ebbo_max * EBBO_SCALE).quantize(Decimal("1"), rounding=ROUND_DOWN)
        ebbo_max_ratio: PriceRatio = (int(ebbo_max_scaled), EBBO_SCALE)
        if _compare_prices(ebbo_max_ratio, price_ceiling) < 0:
            price_ceiling = ebbo_max_ratio

    if _compare_prices(price_floor, price_ceiling) > 0:
        # No EBBO-compliant match possible
        logger.debug(
            "double_auction_ebbo_no_valid_range",
            price_floor=float(_price_ratio_to_decimal(price_floor)),
            price_ceiling=float(_price_ratio_to_decimal(price_ceiling)),
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

    midpoint_price = _midpoint_price(price_floor, price_ceiling)
    candidate_prices: list[PriceRatio] = [midpoint_price, price_floor, price_ceiling]

    # Remove duplicates while preserving order (midpoint first)
    # Two ratios are equal if their cross-products are equal
    unique_prices: list[PriceRatio] = []
    for p in candidate_prices:
        is_dup = False
        for existing in unique_prices:
            if _compare_prices(p, existing) == 0:
                is_dup = True
                break
        if not is_dup:
            unique_prices.append(p)

    logger.debug(
        "double_auction_candidate_prices",
        max_ask_limit=float(_price_ratio_to_decimal(max_ask_limit)),
        min_bid_limit=float(_price_ratio_to_decimal(min_bid_limit)),
        midpoint=float(_price_ratio_to_decimal(midpoint_price)),
        num_candidates=len(unique_prices),
    )

    # Try each candidate price and track the best result
    # Initialize with original amounts (used when no matches happen)
    best_result = _MatchingAtPriceResult(
        matches=[],
        ask_remaining={order.uid: amount for order, _, amount in asks},
        bid_remaining={order.uid: amount for order, _, amount in bids},
        total_a_matched=0,
        total_b_matched=0,
    )
    best_price: PriceRatio | None = None

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
                price=float(_price_ratio_to_decimal(candidate_price)),
                total_a_matched=result.total_a_matched,
                num_matches=len(result.matches),
            )

    # Use the best result
    matches = best_result.matches
    ask_remaining = best_result.ask_remaining
    bid_remaining = best_result.bid_remaining
    total_a_matched = best_result.total_a_matched
    total_b_matched = best_result.total_b_matched
    # Convert price ratio to Decimal for result
    final_price_ratio = best_price if best_price is not None else midpoint_price
    clearing_price = _price_ratio_to_decimal(final_price_ratio)

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

    Uses exact integer arithmetic with PriceRatio to avoid truncation issues.
    Ceiling division is used for seller_min_b (conservative for seller surplus)
    and floor division for buyer_max_b (conservative for buyer surplus).

    Returns:
        Total surplus in token B units (can be converted to USD later)
    """
    total_surplus = 0

    for match in result.matches:
        # Seller wanted at least ask_price B per A
        # ask_price = (buy_amount, sell_amount) where price = buy_amount / sell_amount
        ask_ratio = get_limit_price_ratio(match.seller, is_selling_a=True)
        if ask_ratio is None:
            continue
        ask_num, ask_denom = ask_ratio
        # seller_min_b = amount_a * ask_num / ask_denom
        # Use ceiling division (seller wanted AT LEAST this, round up for conservative surplus)
        seller_min_b = (S(match.amount_a) * S(ask_num)).ceiling_div(S(ask_denom))
        seller_surplus = match.amount_b - seller_min_b.value

        # Buyer was willing to pay up to bid_price B per A
        # bid_price = (sell_amount, buy_amount) where price = sell_amount / buy_amount
        bid_ratio = get_limit_price_ratio(match.buyer, is_selling_a=False)
        if bid_ratio is None:
            continue
        bid_num, bid_denom = bid_ratio
        # buyer_max_b = amount_a * bid_num / bid_denom
        # Use floor division (buyer would pay AT MOST this, round down for conservative surplus)
        buyer_max_b = (S(match.amount_a) * S(bid_num)) // S(bid_denom)
        buyer_surplus = buyer_max_b.value - match.amount_b

        total_surplus += seller_surplus + buyer_surplus

    return total_surplus


__all__ = [
    "get_limit_price",
    "get_limit_price_ratio",
    "run_double_auction",
    "calculate_surplus",
    "PriceRatio",
    "_price_ratio_sort_key",
    "_price_ratio_to_decimal",
]
