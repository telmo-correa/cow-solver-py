"""Protocol constants for CoW Protocol solver.

Centralizes well-known addresses and protocol parameters.
"""

from solver.models.types import is_valid_address

# CoW Protocol Settlement contract address (mainnet)
COW_SETTLEMENT = "0x9008d19f58aabd9ed0d60971565aa8510560ab41"

# Reference price scaling factor (1e18 for precision)
# Used to express clearing prices as integers
PRICE_SCALE = 10**18


def _validate_token_address(name: str, address: str) -> str:
    """Validate and return a token address.

    Args:
        name: Name of the token (for error messages)
        address: The address to validate

    Returns:
        The validated address

    Raises:
        ValueError: If the address is invalid
    """
    if not is_valid_address(address):
        raise ValueError(f"Invalid {name} address: {address} (must be 0x + 40 hex chars)")
    return address


# Gas estimation constants (matching Rust solver baseline)
# Per-swap gas cost for UniswapV2-style pools
POOL_SWAP_GAS_COST = 60_000

# Settlement overhead components (from shared/src/price_estimation/gas.rs)
# SETTLEMENT = 7365 (isSolver check)
# TRADE = 35000 + 2*3000 + 3000 = 44000 (computeTradeExecutions + transfer overhead + interaction)
# ERC20_TRANSFER = 27513 (per transfer, x2 for in/out)
# SETTLEMENT_OVERHEAD = SETTLEMENT + TRADE + 2 * ERC20_TRANSFER
SETTLEMENT_OVERHEAD = 7365 + 44_000 + 2 * 27_513  # = 106_391


# Well-known token addresses on mainnet (lowercase for consistency)
# All addresses are validated at import time to catch typos early
WETH = _validate_token_address("WETH", "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2")
USDC = _validate_token_address("USDC", "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")
USDT = _validate_token_address("USDT", "0xdac17f958d2ee523a2206206994597c13d831ec7")
DAI = _validate_token_address("DAI", "0x6b175474e89094c44da98b954eedeac495271d0f")
