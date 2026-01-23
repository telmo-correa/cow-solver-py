"""Double auction algorithm for multi-order CoW matching.

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

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from solver.models.auction import Order
from solver.models.order_groups import OrderGroup

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


@dataclass
class DoubleAuctionMatch:
    """A single match between two orders in a double auction.

    Represents a partial or complete fill of two counter-party orders.

    Attributes:
        seller: Order selling token A (the "ask" side)
        buyer: Order selling token B to buy token A (the "bid" side)
        amount_a: Amount of token A being exchanged
        amount_b: Amount of token B being exchanged
        clearing_price: The price at which this match clears (B per A)
    """

    seller: Order  # Sells A, wants B
    buyer: Order  # Sells B, wants A
    amount_a: int  # Amount of A exchanged
    amount_b: int  # Amount of B exchanged
    clearing_price: Decimal  # Price in B/A


@dataclass
class DoubleAuctionResult:
    """Result of running double auction on an OrderGroup.

    Attributes:
        matches: List of individual matches
        total_a_matched: Total amount of token A matched
        total_b_matched: Total amount of token B matched
        clearing_price: Uniform clearing price (B per A)
        unmatched_sellers: Sellers of A with remaining amounts
        unmatched_buyers: Sellers of B with remaining amounts
    """

    matches: list[DoubleAuctionMatch]
    total_a_matched: int
    total_b_matched: int
    clearing_price: Decimal | None
    unmatched_sellers: list[tuple[Order, int]]  # (order, remaining_amount)
    unmatched_buyers: list[tuple[Order, int]]  # (order, remaining_amount)

    @property
    def order_count(self) -> int:
        """Number of orders involved in matches."""
        orders = set()
        for m in self.matches:
            orders.add(m.seller.uid)
            orders.add(m.buyer.uid)
        return len(orders)


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


def run_double_auction(
    group: OrderGroup,
    respect_fill_or_kill: bool = True,
) -> DoubleAuctionResult:
    """Run double auction clearing on an order group.

    Implements the classic double auction algorithm:
    1. Sort asks (sellers of A) by price ascending
    2. Sort bids (sellers of B) by price descending
    3. Match until prices cross

    Args:
        group: OrderGroup with orders in both directions
        respect_fill_or_kill: If True, skip non-partially-fillable orders
                              that would be only partially filled

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
    # Filter out invalid orders (None prices)
    asks = [(o, p, a) for o, p, a in asks_raw if p is not None]
    asks.sort(key=lambda x: x[1])

    # Sellers of B: sorted descending (highest bidders first)
    bids_raw = [
        (order, get_limit_price(order, is_selling_a=False), order.sell_amount_int)
        for order in group.sellers_of_b
    ]
    # Filter out invalid orders (None prices)
    bids = [(o, p, a) for o, p, a in bids_raw if p is not None]
    bids.sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        "double_auction_start",
        token_a=group.token_a[-8:],
        token_b=group.token_b[-8:],
        num_asks=len(asks),
        num_bids=len(bids),
        best_ask=float(asks[0][1]) if asks else None,
        best_bid=float(bids[0][1]) if bids else None,
    )

    # Track remaining amounts
    ask_remaining = {order.uid: amount for order, _, amount in asks}
    bid_remaining = {order.uid: amount for order, _, amount in bids}

    matches: list[DoubleAuctionMatch] = []
    total_a_matched = 0
    total_b_matched = 0
    last_clearing_price: Decimal | None = None

    # Match orders
    ask_idx = 0
    bid_idx = 0

    while ask_idx < len(asks) and bid_idx < len(bids):
        ask_order, ask_price, _ = asks[ask_idx]
        bid_order, bid_price, _ = bids[bid_idx]

        # Check if prices cross (no more matches possible)
        if ask_price > bid_price:
            logger.debug(
                "double_auction_prices_crossed",
                ask_price=float(ask_price),
                bid_price=float(bid_price),
            )
            break

        # Get remaining amounts
        ask_remaining_amount = ask_remaining[ask_order.uid]
        bid_remaining_amount = bid_remaining[bid_order.uid]

        if ask_remaining_amount <= 0:
            ask_idx += 1
            continue

        if bid_remaining_amount <= 0:
            bid_idx += 1
            continue

        # Calculate match amount
        # The ask sells A, the bid buys A
        # We need to convert bid's B amount to equivalent A amount

        # Use midpoint price for clearing (fair to both sides)
        clearing_price = (ask_price + bid_price) / 2

        # Amount of A the bid can buy with remaining B
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

        # Calculate B amount for this match
        match_b = int(Decimal(match_a) * clearing_price)

        if match_b <= 0:
            bid_idx += 1
            continue

        # Verify limit prices are satisfied
        # Ask limit: needs at least ask_price B per A
        actual_ask_price = Decimal(match_b) / Decimal(match_a)
        if actual_ask_price < ask_price:
            logger.warning(
                "double_auction_ask_limit_violation",
                actual=float(actual_ask_price),
                limit=float(ask_price),
            )
            ask_idx += 1
            continue

        # Bid limit: pays at most bid_price B per A
        if actual_ask_price > bid_price:
            logger.warning(
                "double_auction_bid_limit_violation",
                actual=float(actual_ask_price),
                limit=float(bid_price),
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
        last_clearing_price = clearing_price

        logger.debug(
            "double_auction_match",
            seller=ask_order.uid[:12] + "...",
            buyer=bid_order.uid[:12] + "...",
            amount_a=match_a,
            amount_b=match_b,
            price=float(clearing_price),
        )

        # Move to next order if current is exhausted
        if ask_remaining[ask_order.uid] <= 0:
            ask_idx += 1
        if bid_remaining[bid_order.uid] <= 0:
            bid_idx += 1

    # Collect unmatched orders
    unmatched_sellers = [
        (order, ask_remaining[order.uid]) for order, _, _ in asks if ask_remaining[order.uid] > 0
    ]
    unmatched_buyers = [
        (order, bid_remaining[order.uid]) for order, _, _ in bids if bid_remaining[order.uid] > 0
    ]

    logger.info(
        "double_auction_complete",
        token_pair=f"{group.token_a[-8:]}/{group.token_b[-8:]}",
        num_matches=len(matches),
        total_a_matched=total_a_matched,
        total_b_matched=total_b_matched,
        clearing_price=float(last_clearing_price) if last_clearing_price else None,
        unmatched_sellers=len(unmatched_sellers),
        unmatched_buyers=len(unmatched_buyers),
    )

    return DoubleAuctionResult(
        matches=matches,
        total_a_matched=total_a_matched,
        total_b_matched=total_b_matched,
        clearing_price=last_clearing_price,
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


@dataclass
class AMMRoute:
    """An order that should be routed through AMM.

    Attributes:
        order: The order to route
        amount: Amount to route (may be partial)
        is_selling_a: True if selling token A (matches OrderGroup convention)
    """

    order: Order
    amount: int  # Amount of the sell token
    is_selling_a: bool


@dataclass
class HybridAuctionResult:
    """Result of hybrid CoW+AMM auction.

    Combines direct CoW matches with AMM routing decisions.

    Attributes:
        cow_matches: Orders matched directly via CoW
        amm_routes: Orders that should route through AMM
        clearing_price: Price used for CoW matches (B per A)
        total_cow_a: Total token A matched via CoW
        total_cow_b: Total token B matched via CoW
    """

    cow_matches: list[DoubleAuctionMatch]
    amm_routes: list[AMMRoute]
    clearing_price: Decimal | None
    total_cow_a: int
    total_cow_b: int


def run_hybrid_auction(
    group: OrderGroup,
    amm_price: Decimal | None = None,
    respect_fill_or_kill: bool = True,
) -> HybridAuctionResult:
    """Run hybrid CoW+AMM auction on an order group.

    This extends the pure double auction by using the AMM reference price
    to determine which orders can be matched directly (CoW) vs which
    should route through AMM.

    The algorithm:
    1. If no AMM price, run pure double auction
    2. With AMM price:
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

    # If no AMM price, use pure double auction
    if amm_price is None:
        pure_result = run_double_auction(group, respect_fill_or_kill)
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

    # With AMM price: match orders that can clear at AMM price
    # Sort asks ascending (cheapest sellers first)
    asks_raw = [
        (order, get_limit_price(order, is_selling_a=True), order.sell_amount_int)
        for order in group.sellers_of_a
    ]
    asks = [(o, p, a) for o, p, a in asks_raw if p is not None]
    asks.sort(key=lambda x: x[1])

    # Sort bids descending (highest bidders first)
    bids_raw = [
        (order, get_limit_price(order, is_selling_a=False), order.sell_amount_int)
        for order in group.sellers_of_b
    ]
    bids = [(o, p, a) for o, p, a in bids_raw if p is not None]
    bids.sort(key=lambda x: x[1], reverse=True)

    # Track remaining amounts
    ask_remaining = {order.uid: amount for order, _, amount in asks}
    bid_remaining = {order.uid: amount for order, _, amount in bids}

    matches: list[DoubleAuctionMatch] = []
    total_a_matched = 0
    total_b_matched = 0

    # Filter to orders that CAN trade at AMM price:
    # - Asks with limit <= AMM price (willing to sell at or below AMM)
    # - Bids with limit >= AMM price (willing to buy at or above AMM)
    matchable_asks = [(o, p, a) for o, p, a in asks if p <= amm_price]
    matchable_bids = [(o, p, a) for o, p, a in bids if p >= amm_price]

    # Match orders at AMM price
    ask_idx = 0
    bid_idx = 0

    while ask_idx < len(matchable_asks) and bid_idx < len(matchable_bids):
        ask_order, _, _ = matchable_asks[ask_idx]
        bid_order, _, _ = matchable_bids[bid_idx]

        ask_remaining_amount = ask_remaining[ask_order.uid]
        bid_remaining_amount = bid_remaining[bid_order.uid]

        if ask_remaining_amount <= 0:
            ask_idx += 1
            continue
        if bid_remaining_amount <= 0:
            bid_idx += 1
            continue

        # At AMM price, bid can buy this much A with remaining B
        bid_can_buy_a = int(Decimal(bid_remaining_amount) / amm_price)

        # Match amount
        match_a = min(ask_remaining_amount, bid_can_buy_a)

        if match_a <= 0:
            bid_idx += 1
            continue

        # Check fill-or-kill constraints
        if respect_fill_or_kill:
            if match_a < ask_remaining_amount and not ask_order.partially_fillable:
                # Ask can't be partially filled, skip to next bid
                bid_idx += 1
                continue
            bid_fill_amount = int(Decimal(match_a) * amm_price)
            if bid_fill_amount < bid_remaining_amount and not bid_order.partially_fillable:
                # Bid can't be partially filled, skip to next ask
                ask_idx += 1
                continue

        # Calculate B amount at AMM price
        match_b = int(Decimal(match_a) * amm_price)

        if match_b <= 0:
            bid_idx += 1
            continue

        # Record match
        match = DoubleAuctionMatch(
            seller=ask_order,
            buyer=bid_order,
            amount_a=match_a,
            amount_b=match_b,
            clearing_price=amm_price,
        )
        matches.append(match)

        # Update remaining
        ask_remaining[ask_order.uid] -= match_a
        bid_remaining[bid_order.uid] -= match_b
        total_a_matched += match_a
        total_b_matched += match_b

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

    # Unmatched bids -> route to AMM
    for order, _, _ in bids:
        remaining = bid_remaining[order.uid]
        if remaining > 0:
            amm_routes.append(AMMRoute(order=order, amount=remaining, is_selling_a=False))

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


__all__ = [
    "DoubleAuctionMatch",
    "DoubleAuctionResult",
    "AMMRoute",
    "HybridAuctionResult",
    "get_limit_price",
    "run_double_auction",
    "run_hybrid_auction",
    "calculate_surplus",
]
