"""Solution strategies for the CoW Protocol solver.

Strategies are tried in order of preference. Each strategy returns a StrategyResult
if it can handle the auction (even partially), or None to pass to the next strategy.

StrategyResults can be composed - e.g., CoW matching fills part of an order,
then AMM routing fills the remainder.
"""

from solver.strategies.amm_routing import AmmRoutingStrategy
from solver.strategies.base import (
    OrderFill,
    PriceWorsened,
    SolutionStrategy,
    StrategyResult,
)
from solver.strategies.cow_match import CowMatchStrategy
from solver.strategies.double_auction import (
    AMMRoute,
    DoubleAuctionMatch,
    DoubleAuctionResult,
    HybridAuctionResult,
    calculate_surplus,
    run_double_auction,
    run_hybrid_auction,
)
from solver.strategies.hybrid_cow import HybridCowStrategy

__all__ = [
    "SolutionStrategy",
    "StrategyResult",
    "OrderFill",
    "PriceWorsened",
    "CowMatchStrategy",
    "HybridCowStrategy",
    "AmmRoutingStrategy",
    # Double auction (Phase 4)
    "DoubleAuctionMatch",
    "DoubleAuctionResult",
    "AMMRoute",
    "HybridAuctionResult",
    "run_double_auction",
    "run_hybrid_auction",
    "calculate_surplus",
]
