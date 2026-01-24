"""Data types for double auction matching.

This module contains the data classes used by the double auction algorithms
for representing matches, results, and routing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from solver.models.auction import Order


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


@dataclass
class MatchingAtPriceResult:
    """Result of executing matches at a specific clearing price.

    Internal dataclass used by _execute_matches_at_price helper.

    Attributes:
        matches: List of matches executed at this price
        ask_remaining: Remaining amounts for each ask order (by UID)
        bid_remaining: Remaining amounts for each bid order (by UID)
        total_a_matched: Total amount of token A matched
        total_b_matched: Total amount of token B matched
    """

    matches: list[DoubleAuctionMatch]
    ask_remaining: dict[str, int]
    bid_remaining: dict[str, int]
    total_a_matched: int
    total_b_matched: int


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


__all__ = [
    "DoubleAuctionMatch",
    "DoubleAuctionResult",
    "MatchingAtPriceResult",
    "AMMRoute",
    "HybridAuctionResult",
]
