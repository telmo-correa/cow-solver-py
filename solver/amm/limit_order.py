"""0x Protocol limit order AMM implementation.

Provides swap simulation for 0x foreign limit orders. Unlike AMM pools,
limit orders have a fixed exchange rate (no slippage curve).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from solver.amm.base import SwapResult
from solver.models.types import normalize_address

if TYPE_CHECKING:
    from solver.pools.limit_order import LimitOrderPool


@dataclass
class LimitOrderAMM:
    """Swap calculator for 0x limit orders.

    Limit orders have linear pricing (no slippage):
    - Sell: amount_out = amount_in * maker_amount / taker_amount
    - Buy: amount_in = amount_out * taker_amount / maker_amount

    This follows 0x's LibMathV06 implementation.
    """

    def simulate_swap(
        self,
        pool: LimitOrderPool,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> SwapResult | None:
        """Simulate a swap through a limit order (exact input).

        Args:
            pool: The limit order to swap through
            token_in: Input token address (must be taker_token)
            token_out: Output token address (must be maker_token)
            amount_in: Amount of input token

        Returns:
            SwapResult if valid, None if tokens don't match or amount exceeds limit
        """
        # Validate token pair (limit orders are unidirectional)
        if not pool.supports_pair(token_in, token_out):
            return None

        # Cap input at pool's maximum capacity for partial fills
        actual_amount_in = min(amount_in, pool.taker_amount)

        # Calculate output: amount_in * maker_amount / taker_amount
        # This is linear scaling - no slippage curve
        amount_out = (actual_amount_in * pool.maker_amount) // pool.taker_amount

        # Cap output at maker amount (sanity check)
        amount_out = min(amount_out, pool.maker_amount)

        return SwapResult(
            amount_in=actual_amount_in,
            amount_out=amount_out,
            pool_address=pool.address,
            token_in=normalize_address(token_in),
            token_out=normalize_address(token_out),
            gas_estimate=pool.gas_estimate,
        )

    def simulate_swap_exact_output(
        self,
        pool: LimitOrderPool,
        token_in: str,
        token_out: str,
        amount_out: int,
    ) -> SwapResult | None:
        """Simulate a swap to get exact output (buy order).

        Args:
            pool: The limit order to swap through
            token_in: Input token address (must be taker_token)
            token_out: Output token address (must be maker_token)
            amount_out: Desired output amount

        Returns:
            SwapResult if valid, None if tokens don't match or amount exceeds limit
        """
        # Validate token pair (limit orders are unidirectional)
        if not pool.supports_pair(token_in, token_out):
            return None

        # Check output doesn't exceed order limit
        if amount_out > pool.maker_amount:
            return None

        # Calculate input: amount_out * taker_amount / maker_amount
        # Round up for buy orders to ensure we get at least amount_out
        amount_in = (amount_out * pool.taker_amount + pool.maker_amount - 1) // pool.maker_amount

        # Check input doesn't exceed taker amount
        if amount_in > pool.taker_amount:
            return None

        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            pool_address=pool.address,
            token_in=normalize_address(token_in),
            token_out=normalize_address(token_out),
            gas_estimate=pool.gas_estimate,
        )


# Singleton instance for convenience
limit_order_amm = LimitOrderAMM()

__all__ = ["LimitOrderAMM", "limit_order_amm"]
