"""EBBO bounds calculation for CoW matching.

This module provides utilities for computing two-sided EBBO (Ethereum Best
Bid/Offer) bounds from AMM reference prices.

EBBO ensures users get execution at least as good as available AMMs:
- ebbo_min: Floor price for sellers (they must get at least this rate)
- ebbo_max: Ceiling price for buyers (they must pay at most this rate)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from solver.fees.price_estimation import get_token_info

if TYPE_CHECKING:
    from solver.models.auction import AuctionInstance
    from solver.routing.router import SingleOrderRouter


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
    ebbo_max = Decimal(1) / ebbo_b_to_a if ebbo_b_to_a and ebbo_b_to_a > 0 else None

    # Use ebbo_min as the clearing price (standard approach)
    amm_price = ebbo_min

    return EBBOBounds(ebbo_min=ebbo_min, ebbo_max=ebbo_max, amm_price=amm_price)


__all__ = ["EBBOBounds", "get_ebbo_bounds"]
