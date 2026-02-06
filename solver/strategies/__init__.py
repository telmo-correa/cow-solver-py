"""Solution strategies for the CoW Protocol solver.

Strategies are tried in order by the Solver. Each strategy returns a StrategyResult
if it can handle the auction (even partially), or None to pass to the next strategy.

**Default Production Chain:**
    1. MultiPairCowStrategy - N-order joint optimization across overlapping pairs
    2. AmmRoutingStrategy - AMM routing fallback

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
from solver.strategies.base_amm import AMMBackedStrategy
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
from solver.strategies.multi_pair import MultiPairCowStrategy

__all__ = [
    # === Base Protocol ===
    "SolutionStrategy",
    "StrategyResult",
    "OrderFill",
    "PriceWorsened",
    "AMMBackedStrategy",
    # === Production Strategies ===
    "CowMatchStrategy",
    "MultiPairCowStrategy",
    "AmmRoutingStrategy",
    # === Double Auction (public API) ===
    "DoubleAuctionMatch",
    "DoubleAuctionResult",
    "AMMRoute",
    "HybridAuctionResult",
    "run_double_auction",
    "run_hybrid_auction",
    "calculate_surplus",
]
