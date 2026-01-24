"""Research and experimental strategies.

This package contains strategies that are not part of the default production
chain but are useful for research, benchmarking, and experimentation.

Strategies:
    HybridCowStrategy: N-order CoW with AMM reference price (superseded by MultiPairCowStrategy)
    RingTradeStrategy: Cyclic trades A→B→C→A (low ROI at 0.12% match rate)
"""

from solver.strategies.research.hybrid_cow import HybridCowStrategy
from solver.strategies.research.ring_trade import (
    OrderGraph,
    RingTrade,
    RingTradeStrategy,
)

__all__ = [
    "HybridCowStrategy",
    "RingTradeStrategy",
    "OrderGraph",
    "RingTrade",
]
