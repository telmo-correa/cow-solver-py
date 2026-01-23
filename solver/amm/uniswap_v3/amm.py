"""UniswapV3 AMM class for swap simulation and encoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from solver.models.types import normalize_address

from .encoding import encode_exact_input_single, encode_exact_output_single
from .pool import UniswapV3Pool
from .quoter import UniswapV3Quoter

if TYPE_CHECKING:
    from solver.amm.base import SwapResult

logger = structlog.get_logger()


class UniswapV3AMM:
    """UniswapV3 AMM using QuoterV2 for swap simulation.

    Unlike V2 which uses local math, V3 quotes are obtained from the
    QuoterV2 contract (via RPC or mock for testing).
    """

    def __init__(self, quoter: UniswapV3Quoter | None = None):
        """Initialize V3 AMM.

        Args:
            quoter: Quoter implementation for getting swap quotes.
                   If None, all quote methods return None (V3 disabled).
        """
        self.quoter = quoter

    def simulate_swap(
        self,
        pool: UniswapV3Pool,
        token_in: str,
        amount_in: int,
    ) -> SwapResult | None:
        """Simulate a swap through a V3 pool (exact input).

        Args:
            pool: The V3 liquidity pool
            token_in: Input token address
            amount_in: Amount of input token

        Returns:
            SwapResult with amounts and pool info, or None if quote fails
        """
        from solver.amm.base import SwapResult

        if self.quoter is None:
            logger.debug("v3_amm_no_quoter", pool=pool.address)
            return None

        token_out = pool.get_token_out(token_in)
        amount_out = self.quoter.quote_exact_input(
            token_in=token_in,
            token_out=token_out,
            fee=pool.fee,
            amount_in=amount_in,
        )

        if amount_out is None:
            logger.debug(
                "v3_amm_quote_failed",
                pool=pool.address,
                token_in=token_in,
                amount_in=amount_in,
            )
            return None

        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            pool_address=pool.address,
            token_in=normalize_address(token_in),
            token_out=normalize_address(token_out),
            gas_estimate=pool.gas_estimate,
        )

    def simulate_swap_exact_output(
        self,
        pool: UniswapV3Pool,
        token_in: str,
        amount_out: int,
    ) -> SwapResult | None:
        """Simulate a swap to get exact output amount.

        Args:
            pool: The V3 liquidity pool
            token_in: Input token address
            amount_out: Desired output amount

        Returns:
            SwapResult with required input and desired output, or None if quote fails
        """
        from solver.amm.base import SwapResult

        if self.quoter is None:
            logger.debug("v3_amm_no_quoter", pool=pool.address)
            return None

        token_out = pool.get_token_out(token_in)
        amount_in = self.quoter.quote_exact_output(
            token_in=token_in,
            token_out=token_out,
            fee=pool.fee,
            amount_out=amount_out,
        )

        if amount_in is None:
            logger.debug(
                "v3_amm_quote_exact_output_failed",
                pool=pool.address,
                token_in=token_in,
                amount_out=amount_out,
            )
            return None

        # Forward verification: compute actual output from selling amount_in
        # The quoter gives accurate results, but we do forward verification
        # to return the actual output (which may be >= requested due to rounding)
        actual_output = self.quoter.quote_exact_input(
            token_in=token_in,
            token_out=token_out,
            fee=pool.fee,
            amount_in=amount_in,
        )

        # If forward quote fails, fall back to requested output
        if actual_output is None:
            actual_output = amount_out

        return SwapResult(
            amount_in=amount_in,
            amount_out=actual_output,  # Use actual forward-simulated output
            pool_address=pool.address,
            token_in=normalize_address(token_in),
            token_out=normalize_address(token_out),
            gas_estimate=pool.gas_estimate,
        )

    def encode_swap(
        self,
        pool: UniswapV3Pool,
        token_in: str,
        amount_in: int,
        amount_out_min: int,
        recipient: str,
    ) -> tuple[str, str]:
        """Encode a V3 swap as calldata for SwapRouterV2.

        Args:
            pool: The V3 liquidity pool
            token_in: Input token address
            amount_in: Amount of input token
            amount_out_min: Minimum output (slippage protection)
            recipient: Address to receive output tokens

        Returns:
            Tuple of (router_address, calldata_hex)
        """
        token_out = pool.get_token_out(token_in)
        return encode_exact_input_single(
            token_in=token_in,
            token_out=token_out,
            fee=pool.fee,
            recipient=recipient,
            amount_in=amount_in,
            amount_out_minimum=amount_out_min,
        )

    def encode_swap_exact_output(
        self,
        pool: UniswapV3Pool,
        token_in: str,
        amount_out: int,
        amount_in_max: int,
        recipient: str,
    ) -> tuple[str, str]:
        """Encode a V3 swap for exact output as calldata for SwapRouterV2.

        Args:
            pool: The V3 liquidity pool
            token_in: Input token address
            amount_out: Exact amount of output token desired
            amount_in_max: Maximum input amount (slippage protection)
            recipient: Address to receive output tokens

        Returns:
            Tuple of (router_address, calldata_hex)
        """
        token_out = pool.get_token_out(token_in)
        return encode_exact_output_single(
            token_in=token_in,
            token_out=token_out,
            fee=pool.fee,
            recipient=recipient,
            amount_out=amount_out,
            amount_in_maximum=amount_in_max,
        )

    def max_fill_sell_order(
        self,
        pool: UniswapV3Pool,
        token_in: str,
        _token_out: str,
        sell_amount: int,
        buy_amount: int,
    ) -> int:
        """Calculate maximum input for a sell order that satisfies the limit price.

        Uses binary search since V3 quotes are obtained from the quoter contract.

        Args:
            pool: The V3 liquidity pool
            token_in: Input token address
            _token_out: Output token address (unused, derived from pool)
            sell_amount: Order's sell amount (search range upper bound)
            buy_amount: Order's minimum buy amount (for limit check)

        Returns:
            Maximum input amount that satisfies the limit, or 0 if impossible
        """
        if sell_amount <= 0 or buy_amount <= 0:
            return 0

        # Binary search for maximum fill
        lo, hi = 0, sell_amount

        while lo < hi:
            mid = (lo + hi + 1) // 2
            result = self.simulate_swap(pool, token_in, mid)

            if result is None:
                hi = mid - 1
                continue

            # Check limit: output/input >= buy_amount/sell_amount
            if result.amount_out * sell_amount >= buy_amount * mid:
                lo = mid
            else:
                hi = mid - 1

        # Verify the final result
        if lo > 0:
            result = self.simulate_swap(pool, token_in, lo)
            if result is None or result.amount_out * sell_amount < buy_amount * lo:
                return 0

        return lo

    def max_fill_buy_order(
        self,
        pool: UniswapV3Pool,
        token_in: str,
        _token_out: str,
        sell_amount: int,
        buy_amount: int,
    ) -> int:
        """Calculate maximum output for a buy order that satisfies the limit price.

        Uses binary search since V3 quotes are obtained from the quoter contract.

        Args:
            pool: The V3 liquidity pool
            token_in: Input token address
            _token_out: Output token address (unused, derived from pool)
            sell_amount: Order's maximum sell amount (for limit check)
            buy_amount: Order's desired buy amount (search range upper bound)

        Returns:
            Maximum output amount that satisfies the limit, or 0 if impossible
        """
        if sell_amount <= 0 or buy_amount <= 0:
            return 0

        # Binary search for maximum fill
        lo, hi = 0, buy_amount

        while lo < hi:
            mid = (lo + hi + 1) // 2
            result = self.simulate_swap_exact_output(pool, token_in, mid)

            if result is None:
                hi = mid - 1
                continue

            # Check limit: input/output <= sell_amount/buy_amount
            if result.amount_in * buy_amount <= sell_amount * mid:
                lo = mid
            else:
                hi = mid - 1

        # Verify the final result
        if lo > 0:
            result = self.simulate_swap_exact_output(pool, token_in, lo)
            if result is None or result.amount_in * buy_amount > sell_amount * lo:
                return 0

        return lo


__all__ = ["UniswapV3AMM"]
