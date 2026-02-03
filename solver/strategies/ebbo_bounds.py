"""EBBO bounds calculation for CoW matching.

This module provides utilities for computing two-sided EBBO (Ethereum Best
Bid/Offer) bounds from AMM reference prices.

EBBO ensures users get execution at least as good as available AMMs:
- ebbo_min: Floor price for sellers (they must get at least this rate)
- ebbo_max: Ceiling price for buyers (they must pay at most this rate)

IMPORTANT: All financial calculations use exact integer arithmetic.
Decimal comparisons are converted to integer arithmetic for exactness.
"""

from __future__ import annotations

import decimal
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from solver.fees.price_estimation import get_token_info
from solver.models.types import normalize_address

# Use context with high precision for Decimal operations to ensure exactness
_DECIMAL_HIGH_PREC_CONTEXT = decimal.Context(prec=78)


def _decimal_lt(a: Decimal, b: Decimal) -> bool:
    """Compare a < b with high precision for exactness."""
    with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
        diff = a - b
        return diff < 0


if TYPE_CHECKING:
    from solver.models.auction import AuctionInstance
    from solver.routing.router import SingleOrderRouter
    from solver.strategies.base import OrderFill


@dataclass
class EBBOBounds:
    """Two-sided EBBO bounds for a token pair.

    For a pair A/B where price is expressed as "B per A":
    - ebbo_min: Minimum rate sellers of A must receive (from A→B AMM quote)
    - ebbo_max: Maximum rate buyers of A can pay (from inverse of B→A AMM quote)

    Attributes:
        ebbo_min: Floor price for sellers (B per A). None if no AMM liquidity.
        ebbo_max: Ceiling price for buyers (B per A). None if no AMM liquidity.
        amm_price: Reference price to use for clearing (typically ebbo_min).
    """

    ebbo_min: Decimal | None
    ebbo_max: Decimal | None
    amm_price: Decimal | None


def get_ebbo_bounds(
    token_a: str,
    token_b: str,
    router: SingleOrderRouter,
    auction: AuctionInstance,
) -> EBBOBounds:
    """Compute two-sided EBBO bounds for a token pair.

    Gets AMM reference prices for both directions and derives:
    - ebbo_min: A→B rate (what sellers of A must receive)
    - ebbo_max: 1 / (B→A rate) (what buyers of A can pay)

    Args:
        token_a: Address of token A (the "base" token)
        token_b: Address of token B (the "quote" token)
        router: Router for AMM price queries
        auction: Auction context for token decimals

    Returns:
        EBBOBounds with ebbo_min, ebbo_max, and amm_price
    """
    # Get token decimals with fallback to 18
    token_a_info = get_token_info(auction, token_a)
    token_b_info = get_token_info(auction, token_b)

    decimals_a = (
        18 if token_a_info is None or token_a_info.decimals is None else token_a_info.decimals
    )
    decimals_b = (
        18 if token_b_info is None or token_b_info.decimals is None else token_b_info.decimals
    )

    # ebbo_min: rate sellers of A must get (A→B direction)
    ebbo_min = router.get_reference_price(token_a, token_b, token_in_decimals=decimals_a)

    # ebbo_max: rate buyers of A can pay (derived from B→A direction)
    # If B→A rate is R, then max A→B rate is 1/R
    ebbo_b_to_a = router.get_reference_price(token_b, token_a, token_in_decimals=decimals_b)
    # Use high-precision context for exact inversion
    if ebbo_b_to_a and ebbo_b_to_a > 0:
        with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
            ebbo_max = Decimal(1) / ebbo_b_to_a
    else:
        ebbo_max = None

    # Use ebbo_min as the clearing price (standard approach)
    amm_price = ebbo_min

    return EBBOBounds(ebbo_min=ebbo_min, ebbo_max=ebbo_max, amm_price=amm_price)


def verify_fills_against_ebbo(
    fills: list[OrderFill],
    clearing_prices: dict[str, int],
    router: SingleOrderRouter,
    auction: AuctionInstance,
) -> bool:
    """Verify that all fills satisfy EBBO constraints.

    This is a shared utility for verifying EBBO compliance across strategies.
    Used by MultiPairCowStrategy, UnifiedCowStrategy, and RingTradeStrategy
    for cycle/ring verification.

    For each fill, checks that the clearing rate (buy_price / sell_price)
    is at least as good as the AMM reference rate for that order's token pair.

    Args:
        fills: List of order fills to verify
        clearing_prices: Token prices from the settlement (token -> price)
        router: Router for AMM price queries
        auction: Auction context for token decimals

    Returns:
        True if all fills satisfy EBBO, False if any violation found
    """
    for fill in fills:
        order = fill.order
        sell_token = normalize_address(order.sell_token)
        buy_token = normalize_address(order.buy_token)

        sell_price = clearing_prices.get(sell_token, 0)
        buy_price = clearing_prices.get(buy_token, 0)

        # Skip orders with missing prices - can't validate EBBO without both prices.
        # This is consistent with ebbo.py:185-186 which also skips missing prices.
        if sell_price <= 0 or buy_price <= 0:
            continue

        # Get sell token decimals for probe amount scaling in price query
        sell_info = auction.tokens.get(sell_token)
        sell_decimals = sell_info.decimals if sell_info and sell_info.decimals else 18

        # Get AMM reference rate as integer ratio for exact comparison
        # Try the new ratio method first, with fallback to the Decimal method
        try:
            amm_ratio = router.get_reference_price_ratio(
                sell_token, buy_token, token_in_decimals=sell_decimals
            )
        except AttributeError:
            # Router doesn't have get_reference_price_ratio method
            amm_ratio = None

        # Validate that we got a proper tuple (not a Mock or other non-tuple)
        if not isinstance(amm_ratio, tuple) or len(amm_ratio) != 2:
            # Fall back to the Decimal method
            amm_rate = router.get_reference_price(
                sell_token, buy_token, token_in_decimals=sell_decimals
            )
            if amm_rate is None:
                continue

            # Clearing rate = sell_price / buy_price (both in raw amounts)
            # No decimal scaling needed - both prices and AMM rate are in raw units
            with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
                clearing_rate = Decimal(sell_price) / Decimal(buy_price)

            if _decimal_lt(clearing_rate, amm_rate):
                return False
            continue

        amm_out, amm_in = amm_ratio

        # EBBO check using cross-multiplication (exact integer arithmetic):
        # clearing_rate = sell_price / buy_price (both in raw amounts)
        # amm_rate = amm_out / amm_in (also in raw units)
        #
        # No decimal scaling needed - both prices and AMM rate are in raw units.
        #
        # Check: clearing_rate >= amm_rate
        # sell_price / buy_price >= amm_out / amm_in
        # sell_price * amm_in >= buy_price * amm_out
        if sell_price * amm_in < buy_price * amm_out:
            return False

    return True


__all__ = ["EBBOBounds", "get_ebbo_bounds", "verify_fills_against_ebbo"]
