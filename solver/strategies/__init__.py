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
from solver.strategies.multi_pair import (
    LPResult,
    MultiPairCowStrategy,
    PriceCandidates,
    UnionFind,
    build_price_candidates,
    build_token_graph,
    enumerate_price_combinations,
    find_spanning_tree,
    find_token_components,
    solve_fills_at_prices,
)
from solver.strategies.ring_trade import (
    CycleViability,
    OrderGraph,
    RingTrade,
    RingTradeStrategy,
)

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
    # Multi-pair coordination (Slice 4.6)
    "MultiPairCowStrategy",
    "UnionFind",
    "find_token_components",
    "PriceCandidates",
    "build_price_candidates",
    "build_token_graph",
    "find_spanning_tree",
    "enumerate_price_combinations",
    "LPResult",
    "solve_fills_at_prices",
    # Ring trades (Slice 4.4)
    "OrderGraph",
    "CycleViability",
    "RingTrade",
    "RingTradeStrategy",
]
