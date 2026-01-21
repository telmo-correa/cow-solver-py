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

__all__ = [
    "SolutionStrategy",
    "StrategyResult",
    "OrderFill",
    "PriceWorsened",
    "CowMatchStrategy",
    "AmmRoutingStrategy",
]
