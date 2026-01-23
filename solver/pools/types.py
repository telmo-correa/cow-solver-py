"""Pool type definitions.

Provides the AnyPool union type for use throughout the codebase.
"""

from typing import TypeAlias

from solver.amm.balancer import BalancerStablePool, BalancerWeightedPool
from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.amm.uniswap_v3 import UniswapV3Pool

# Union type for all pool types
AnyPool: TypeAlias = UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool

__all__ = [
    "AnyPool",
    "UniswapV2Pool",
    "UniswapV3Pool",
    "BalancerWeightedPool",
    "BalancerStablePool",
]
