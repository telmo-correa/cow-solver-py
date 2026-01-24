"""Solution strategies for the CoW Protocol solver.

Strategies are tried in order by the Solver. Each strategy returns a StrategyResult
if it can handle the auction (even partially), or None to pass to the next strategy.

**Default Production Chain:**
    1. CowMatchStrategy - 2-order direct peer-to-peer matching
    2. MultiPairCowStrategy - N-order joint optimization across overlapping pairs
    3. AmmRoutingStrategy - AMM routing fallback

**Research/Experimental (not in default chain):**
    - HybridCowStrategy - Superseded by MultiPairCowStrategy
    - RingTradeStrategy - Low ROI (0.12% match rate)

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

# Import from new modular files
from solver.strategies.components import find_token_components
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
from solver.strategies.graph import UnionFind, build_token_graph, find_spanning_tree
from solver.strategies.hybrid_cow import HybridCowStrategy
from solver.strategies.multi_pair import MultiPairCowStrategy
from solver.strategies.pricing import (
    LPResult,
    PriceCandidates,
    build_price_candidates,
    enumerate_price_combinations,
    solve_fills_at_prices,
)
from solver.strategies.ring_trade import (
    OrderGraph,
    RingTrade,
    RingTradeStrategy,
)
from solver.strategies.settlement import CycleViability

__all__ = [
    # === Base Protocol ===
    "SolutionStrategy",
    "StrategyResult",
    "OrderFill",
    "PriceWorsened",
    "AMMBackedStrategy",  # Base class for AMM-backed strategies
    # === Production Strategies ===
    "CowMatchStrategy",  # 2-order matching
    "MultiPairCowStrategy",  # N-order joint optimization (Slice 4.6)
    "AmmRoutingStrategy",  # AMM routing fallback
    # === Research/Experimental Strategies ===
    "HybridCowStrategy",  # DEPRECATED: superseded by MultiPairCowStrategy
    "RingTradeStrategy",  # Research: cyclic trades (Slice 4.4)
    # === Double Auction Utilities ===
    "DoubleAuctionMatch",
    "DoubleAuctionResult",
    "AMMRoute",
    "HybridAuctionResult",
    "run_double_auction",
    "run_hybrid_auction",
    "calculate_surplus",
    # === Multi-Pair Utilities (Slice 4.6) ===
    "UnionFind",
    "find_token_components",
    "PriceCandidates",
    "build_price_candidates",
    "build_token_graph",
    "find_spanning_tree",
    "enumerate_price_combinations",
    "LPResult",
    "solve_fills_at_prices",
    # === Ring Trade Utilities (Slice 4.4) ===
    "OrderGraph",
    "CycleViability",
    "RingTrade",
]
