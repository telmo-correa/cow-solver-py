"""Pool-specific routing handlers.

Each handler implements routing logic for a specific pool type:
- UniswapV2Handler: Constant product pools with fee deduction
- UniswapV3Handler: Concentrated liquidity pools (quoter-based)
- BalancerHandler: Weighted and stable pools (singledispatch)

The PoolHandler protocol defines the common interface for all handlers.
"""

from solver.routing.handlers.balancer import BalancerHandler
from solver.routing.handlers.base import PoolHandler
from solver.routing.handlers.v2 import UniswapV2Handler
from solver.routing.handlers.v3 import UniswapV3Handler

__all__ = [
    "PoolHandler",
    "UniswapV2Handler",
    "UniswapV3Handler",
    "BalancerHandler",
]
