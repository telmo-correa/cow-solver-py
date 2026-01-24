"""Double auction algorithms for multi-order CoW matching.

This package provides algorithms for matching multiple orders on a single
token pair using double auction clearing.

Modules:
    types: Data classes for matches, results, and routing decisions
    core: Pure double auction algorithm
    hybrid: CoW+AMM hybrid auction with AMM reference prices

Usage:
    from solver.strategies.double_auction import (
        run_double_auction,
        run_hybrid_auction,
        DoubleAuctionMatch,
        DoubleAuctionResult,
        HybridAuctionResult,
    )
"""

from solver.strategies.double_auction.core import (
    calculate_surplus,
    get_limit_price,
    run_double_auction,
)
from solver.strategies.double_auction.hybrid import run_hybrid_auction
from solver.strategies.double_auction.types import (
    AMMRoute,
    DoubleAuctionMatch,
    DoubleAuctionResult,
    HybridAuctionResult,
    MatchingAtPriceResult,
)

__all__ = [
    # Types
    "DoubleAuctionMatch",
    "DoubleAuctionResult",
    "MatchingAtPriceResult",
    "AMMRoute",
    "HybridAuctionResult",
    # Core functions
    "get_limit_price",
    "run_double_auction",
    "calculate_surplus",
    # Hybrid auction
    "run_hybrid_auction",
]
