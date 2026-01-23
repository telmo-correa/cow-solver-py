"""UniswapV3 AMM implementation package.

This package provides UniswapV3 concentrated liquidity pool support:
- Pool dataclass (UniswapV3Pool)
- Quoter implementations (Mock and Web3-based)
- Swap encoding for SwapRouterV2
- AMM class for simulation and encoding
- Liquidity parsing from auction data

All exports are maintained for backward compatibility with:
    from solver.amm.uniswap_v3 import UniswapV3Pool, UniswapV3AMM
"""

from .amm import UniswapV3AMM
from .constants import (
    QUOTER_V2_ADDRESS,
    SWAP_ROUTER_V2_ADDRESS,
    V3_FEE_HIGH,
    V3_FEE_LOW,
    V3_FEE_LOWEST,
    V3_FEE_MEDIUM,
    V3_FEE_TIERS,
    V3_SWAP_GAS_COST,
    V3_TICK_SPACING,
)
from .encoding import (
    EXACT_INPUT_SINGLE_SELECTOR,
    EXACT_OUTPUT_SINGLE_SELECTOR,
    SWAP_ROUTER_V2_ABI,
    encode_exact_input_single,
    encode_exact_output_single,
)
from .parsing import parse_v3_liquidity
from .pool import UniswapV3Pool
from .quoter import (
    QUOTER_V2_ABI,
    MockUniswapV3Quoter,
    QuoteKey,
    UniswapV3Quoter,
    Web3UniswapV3Quoter,
)

__all__ = [
    # Constants
    "V3_FEE_LOWEST",
    "V3_FEE_LOW",
    "V3_FEE_MEDIUM",
    "V3_FEE_HIGH",
    "V3_FEE_TIERS",
    "V3_TICK_SPACING",
    "V3_SWAP_GAS_COST",
    "QUOTER_V2_ADDRESS",
    "SWAP_ROUTER_V2_ADDRESS",
    # Pool
    "UniswapV3Pool",
    # Quoter
    "UniswapV3Quoter",
    "QuoteKey",
    "MockUniswapV3Quoter",
    "Web3UniswapV3Quoter",
    "QUOTER_V2_ABI",
    # Encoding
    "EXACT_INPUT_SINGLE_SELECTOR",
    "EXACT_OUTPUT_SINGLE_SELECTOR",
    "SWAP_ROUTER_V2_ABI",
    "encode_exact_input_single",
    "encode_exact_output_single",
    # AMM
    "UniswapV3AMM",
    # Parsing
    "parse_v3_liquidity",
]
