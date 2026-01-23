"""Order routing logic.

This package handles routing orders through AMM pools and building solutions
for the CoW Protocol settlement.

Module structure:
- router.py: SingleOrderRouter facade class
- types.py: HopResult and RoutingResult dataclasses
- handlers/: Pool-specific routing handlers (V2, V3, Balancer)
- multihop.py: Multi-hop routing through multiple pools
- registry.py: HandlerRegistry for centralized pool dispatch
- pathfinding.py: TokenGraph and PathFinder for route discovery
- solution.py: Solution building from routing results
"""

from solver.routing.pathfinding import PathFinder, TokenGraph
from solver.routing.registry import HandlerRegistry
from solver.routing.router import SingleOrderRouter
from solver.routing.types import HopResult, RoutingResult

__all__ = [
    "HandlerRegistry",
    "HopResult",
    "PathFinder",
    "RoutingResult",
    "SingleOrderRouter",
    "TokenGraph",
]
