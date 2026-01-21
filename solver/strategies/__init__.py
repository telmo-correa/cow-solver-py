"""Solution strategies for the CoW Protocol solver.

Strategies are tried in order of preference. Each strategy returns a Solution
if it can handle the auction, or None to pass to the next strategy.
"""

from solver.strategies.amm_routing import AmmRoutingStrategy
from solver.strategies.base import SolutionStrategy
from solver.strategies.cow_match import CowMatchStrategy

__all__ = [
    "SolutionStrategy",
    "CowMatchStrategy",
    "AmmRoutingStrategy",
]
