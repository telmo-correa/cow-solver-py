"""AMM (Automated Market Maker) implementations."""

from solver.amm.base import AMM, SwapResult
from solver.amm.limit_order import LimitOrderAMM, limit_order_amm
from solver.amm.uniswap_v2 import (
    UniswapV2,
    UniswapV2Pool,
    parse_liquidity_to_pool,
    uniswap_v2,
)
from solver.pools import PoolRegistry, build_registry_from_liquidity

__all__ = [
    # Base classes
    "AMM",
    "SwapResult",
    # UniswapV2
    "UniswapV2",
    "UniswapV2Pool",
    "uniswap_v2",
    "parse_liquidity_to_pool",
    # Limit Orders
    "LimitOrderAMM",
    "limit_order_amm",
    # Pool registry (from solver.pools)
    "PoolRegistry",
    "build_registry_from_liquidity",
]
