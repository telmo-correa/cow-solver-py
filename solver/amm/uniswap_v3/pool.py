"""UniswapV3Pool dataclass for concentrated liquidity pools."""

from __future__ import annotations

from dataclasses import dataclass, field

from solver.models.types import normalize_address

from .constants import SWAP_ROUTER_V2_ADDRESS, V3_SWAP_GAS_COST, V3_TICK_SPACING


@dataclass
class UniswapV3Pool:
    """Represents a UniswapV3 concentrated liquidity pool.

    UniswapV3 pools have liquidity concentrated in price ranges (ticks).
    The pool state includes:
    - Current price (as sqrtPriceX96)
    - Current tick
    - Active liquidity at current tick
    - Net liquidity changes at each initialized tick

    Note: We store the full state for completeness, but swap simulation
    is delegated to the QuoterV2 contract rather than done locally.
    """

    address: str
    token0: str
    token1: str
    fee: int  # Fee in Uniswap units (e.g., 3000 for 0.3%)
    sqrt_price_x96: int  # Current sqrt(price) * 2^96
    liquidity: int  # Current active liquidity
    tick: int  # Current tick index
    liquidity_net: dict[int, int] = field(default_factory=dict)  # tick -> net liquidity
    router: str = SWAP_ROUTER_V2_ADDRESS
    liquidity_id: str | None = None  # ID from auction for LiquidityInteraction
    gas_estimate: int = V3_SWAP_GAS_COST

    @property
    def fee_percent(self) -> float:
        """Fee as percentage (e.g., 0.3 for 0.3%)."""
        return self.fee / 10000

    @property
    def fee_decimal(self) -> float:
        """Fee as decimal (e.g., 0.003 for 0.3%)."""
        return self.fee / 1_000_000

    @property
    def tick_spacing(self) -> int:
        """Get tick spacing for this pool's fee tier."""
        return V3_TICK_SPACING.get(self.fee, 60)  # Default to medium

    def get_token_out(self, token_in: str) -> str:
        """Get the output token for a given input token."""
        token_in_norm = normalize_address(token_in)
        if token_in_norm == normalize_address(self.token0):
            return self.token1
        elif token_in_norm == normalize_address(self.token1):
            return self.token0
        else:
            raise ValueError(f"Token {token_in} not in pool")

    def is_token0(self, token: str) -> bool:
        """Check if token is token0 (determines swap direction)."""
        return normalize_address(token) == normalize_address(self.token0)


__all__ = ["UniswapV3Pool"]
