"""Shared token constants for tests.

All addresses are lowercase for consistency with normalize_address().

Usage:
    from tests.helpers import WETH, USDC
    # or
    from tests.helpers.constants import WETH, USDC
"""

# =============================================================================
# Mainnet tokens (most commonly used)
# =============================================================================

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"  # Wrapped Ether (18 decimals)
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"  # USD Coin (6 decimals)
DAI = "0x6b175474e89094c44da98b954eedeac495271d0f"  # Dai Stablecoin (18 decimals)
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"  # Tether USD (6 decimals)
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"  # Wrapped Bitcoin (8 decimals)

# =============================================================================
# Additional tokens (used in ring trade tests)
# =============================================================================

UNI = "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"  # Uniswap Token (18 decimals)
GNO = "0x6810e776880c02933d47db1b9fc05908e5386b96"  # Gnosis Token (18 decimals)
COW = "0xdef1ca1fb7fbcdc777520aa7f396b4e015f497ab"  # CoW Protocol Token (18 decimals)

# =============================================================================
# Cross-chain tokens
# =============================================================================

WXDAI_GNOSIS = "0xe91d153e0b41518a2ce8dd3d7944fa863463a97d"  # Wrapped xDAI on Gnosis


# =============================================================================
# Token decimals lookup (for tests that need it)
# =============================================================================

TOKEN_DECIMALS = {
    WETH: 18,
    USDC: 6,
    DAI: 18,
    USDT: 6,
    WBTC: 8,
    UNI: 18,
    GNO: 18,
    COW: 18,
    WXDAI_GNOSIS: 18,
}


__all__ = [
    "WETH",
    "USDC",
    "DAI",
    "USDT",
    "WBTC",
    "UNI",
    "GNO",
    "COW",
    "WXDAI_GNOSIS",
    "TOKEN_DECIMALS",
]
