"""Balancer pool dataclasses.

Data structures for weighted and stable pools.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True)
class WeightedTokenReserve:
    """Reserve information for a token in a weighted pool.

    Attributes:
        token: Token address (case-insensitive comparison supported)
        balance: Raw balance from auction (in token's native decimals)
        weight: Normalized weight (sum of all weights in pool = 1.0)
        scaling_factor: From auction data. For 6-decimal tokens like USDC,
            this is 10^12 to normalize to 18 decimals.
    """

    token: str
    balance: int
    weight: Decimal
    scaling_factor: int


@dataclass(frozen=True)
class BalancerWeightedPool:
    """Balancer V2 weighted pool.

    Attributes:
        id: Liquidity ID from auction (used for Interaction)
        address: Pool contract address
        pool_id: balancerPoolId (32-byte hex string for settlement encoding)
        reserves: Token reserves, sorted by token address
        fee: Swap fee as decimal (e.g., 0.003 for 0.3%)
        version: Pool version - affects power function rounding
        gas_estimate: Gas cost estimate from auction data
    """

    id: str
    address: str
    pool_id: str
    reserves: tuple[WeightedTokenReserve, ...]
    fee: Decimal
    version: Literal["v0", "v3Plus"]
    gas_estimate: int

    def get_reserve(self, token: str) -> WeightedTokenReserve | None:
        """Get reserve for a specific token."""
        token_lower = token.lower()
        for reserve in self.reserves:
            if reserve.token.lower() == token_lower:
                return reserve
        return None

    @property
    def liquidity_id(self) -> str:
        """Alias for id field, for compatibility with LiquidityInteraction."""
        return self.id


@dataclass(frozen=True)
class StableTokenReserve:
    """Reserve information for a token in a stable pool.

    Attributes:
        token: Token address (case-insensitive comparison supported)
        balance: Raw balance from auction (in token's native decimals)
        scaling_factor: From auction data. For 6-decimal tokens like USDC,
            this is 10^12 to normalize to 18 decimals.
    """

    token: str
    balance: int
    scaling_factor: int


@dataclass(frozen=True)
class BalancerStablePool:
    """Balancer V2 stable pool (StableSwap / Curve-style).

    Attributes:
        id: Liquidity ID from auction (used for Interaction)
        address: Pool contract address
        pool_id: balancerPoolId (32-byte hex string for settlement encoding)
        reserves: Token reserves, sorted by token address
        amplification_parameter: Raw A parameter from auction JSON (e.g., 5000.0).
            NOTE: This is the unscaled value. The AMM internally multiplies
            by AMP_PRECISION (1000) before passing to the stable math functions,
            so A=5000 becomes 5,000,000 in calculations.
        fee: Swap fee as decimal (e.g., 0.0001 for 0.01%)
        gas_estimate: Gas cost estimate from auction data
    """

    id: str
    address: str
    pool_id: str
    reserves: tuple[StableTokenReserve, ...]
    amplification_parameter: Decimal
    fee: Decimal
    gas_estimate: int

    def get_reserve(self, token: str) -> StableTokenReserve | None:
        """Get reserve for a specific token."""
        token_lower = token.lower()
        for reserve in self.reserves:
            if reserve.token.lower() == token_lower:
                return reserve
        return None

    @property
    def liquidity_id(self) -> str:
        """Alias for id field, for compatibility with LiquidityInteraction."""
        return self.id
