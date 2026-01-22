"""Balancer AMM classes.

High-level AMM classes for swap simulation through Balancer pools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from solver.math.fixed_point import AMP_PRECISION, Bfp
from solver.models.types import normalize_address

from .errors import BalancerError
from .pools import (
    BalancerStablePool,
    BalancerWeightedPool,
    StableTokenReserve,
    WeightedTokenReserve,
)
from .scaling import (
    add_swap_fee_amount,
    scale_down_down,
    scale_down_up,
    scale_up,
    subtract_swap_fee_amount,
)
from .stable_math import stable_calc_in_given_out, stable_calc_out_given_in
from .weighted_math import calc_in_given_out, calc_out_given_in

if TYPE_CHECKING:
    from solver.amm.base import SwapResult

logger = structlog.get_logger()


# =============================================================================
# Reserve helpers
# =============================================================================


def _get_weighted_reserves(
    pool: BalancerWeightedPool,
    token_in: str,
    token_out: str,
) -> tuple[WeightedTokenReserve, WeightedTokenReserve] | None:
    """Get reserves for a token pair in a weighted pool.

    Args:
        pool: The weighted pool
        token_in: Input token address
        token_out: Output token address

    Returns:
        Tuple of (reserve_in, reserve_out), or None if tokens not found or same
    """
    # Reject self-swaps
    if token_in.lower() == token_out.lower():
        logger.debug(
            "weighted_amm_self_swap",
            pool_id=pool.id,
            token=token_in,
        )
        return None

    reserve_in = pool.get_reserve(token_in)
    reserve_out = pool.get_reserve(token_out)

    if reserve_in is None:
        logger.debug(
            "weighted_amm_token_not_found",
            pool_id=pool.id,
            token=token_in,
            role="input",
        )
        return None

    if reserve_out is None:
        logger.debug(
            "weighted_amm_token_not_found",
            pool_id=pool.id,
            token=token_out,
            role="output",
        )
        return None

    return reserve_in, reserve_out


def _get_stable_reserves(
    pool: BalancerStablePool,
    token_in: str,
    token_out: str,
) -> tuple[StableTokenReserve, int, StableTokenReserve, int] | None:
    """Get reserves and indices for a token pair in a stable pool.

    Args:
        pool: The stable pool
        token_in: Input token address
        token_out: Output token address

    Returns:
        Tuple of (reserve_in, index_in, reserve_out, index_out), or None if not found
    """
    reserve_in = None
    index_in = None
    reserve_out = None
    index_out = None

    token_in_lower = token_in.lower()
    token_out_lower = token_out.lower()

    for i, reserve in enumerate(pool.reserves):
        token_lower = reserve.token.lower()
        if token_lower == token_in_lower:
            reserve_in = reserve
            index_in = i
        elif token_lower == token_out_lower:
            reserve_out = reserve
            index_out = i

    if reserve_in is None or index_in is None:
        logger.debug(
            "stable_amm_token_not_found",
            pool_id=pool.id,
            token=token_in,
            role="input",
        )
        return None

    if reserve_out is None or index_out is None:
        logger.debug(
            "stable_amm_token_not_found",
            pool_id=pool.id,
            token=token_out,
            role="output",
        )
        return None

    return reserve_in, index_in, reserve_out, index_out


# =============================================================================
# Partial Fill Binary Search Helpers
# =============================================================================


def _binary_search_max_sell_fill(
    simulate_fn: Any,
    sell_amount: int,
    buy_amount: int,
) -> int:
    """Binary search for maximum sell order fill that satisfies limit price.

    Args:
        simulate_fn: Callable that takes amount_in and returns SwapResult or None
        sell_amount: Order's sell amount (search range upper bound)
        buy_amount: Order's minimum buy amount (for limit check)

    Returns:
        Maximum input amount that satisfies the limit, or 0 if impossible
    """
    if sell_amount <= 0 or buy_amount <= 0:
        return 0

    lo, hi = 0, sell_amount

    while lo < hi:
        mid = (lo + hi + 1) // 2
        result = simulate_fn(mid)

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
        result = simulate_fn(lo)
        if result is None or result.amount_out * sell_amount < buy_amount * lo:
            return 0

    return lo


def _binary_search_max_buy_fill(
    simulate_fn: Any,
    sell_amount: int,
    buy_amount: int,
) -> int:
    """Binary search for maximum buy order fill that satisfies limit price.

    Args:
        simulate_fn: Callable that takes amount_out and returns SwapResult or None
        sell_amount: Order's maximum sell amount (for limit check)
        buy_amount: Order's desired buy amount (search range upper bound)

    Returns:
        Maximum output amount that satisfies the limit, or 0 if impossible
    """
    if sell_amount <= 0 or buy_amount <= 0:
        return 0

    lo, hi = 0, buy_amount

    while lo < hi:
        mid = (lo + hi + 1) // 2
        result = simulate_fn(mid)

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
        result = simulate_fn(lo)
        if result is None or result.amount_in * buy_amount > sell_amount * lo:
            return 0

    return lo


# =============================================================================
# AMM Classes
# =============================================================================


class BalancerWeightedAMM:
    """Balancer V2 weighted pool AMM for swap simulation.

    Provides simulate_swap and simulate_swap_exact_output methods
    compatible with the router's pool selection logic.
    """

    def simulate_swap(
        self,
        pool: BalancerWeightedPool,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> SwapResult | None:
        """Simulate a swap through a weighted pool (exact input).

        Args:
            pool: The weighted pool
            token_in: Input token address
            token_out: Output token address
            amount_in: Amount of input token

        Returns:
            SwapResult with amounts and pool info, or None if swap fails
        """
        from solver.amm.base import SwapResult

        reserves = _get_weighted_reserves(pool, token_in, token_out)
        if reserves is None:
            return None
        reserve_in, reserve_out = reserves

        try:
            # Scale up balances and amount
            balance_in = scale_up(reserve_in.balance, reserve_in.scaling_factor)
            balance_out = scale_up(reserve_out.balance, reserve_out.scaling_factor)
            amount_in_scaled = scale_up(amount_in, reserve_in.scaling_factor)

            # Convert weights to Bfp
            weight_in = Bfp.from_decimal(reserve_in.weight)
            weight_out = Bfp.from_decimal(reserve_out.weight)

            # Subtract fee from input
            amount_in_after_fee = subtract_swap_fee_amount(amount_in_scaled.value, pool.fee)

            # Calculate output
            amount_out_scaled = calc_out_given_in(
                balance_in=balance_in,
                weight_in=weight_in,
                balance_out=balance_out,
                weight_out=weight_out,
                amount_in=Bfp(amount_in_after_fee),
                _version=pool.version,
            )

            # Scale down output
            amount_out_result = scale_down_down(amount_out_scaled, reserve_out.scaling_factor)

            return SwapResult(
                amount_in=amount_in,
                amount_out=amount_out_result,
                pool_address=pool.address,
                token_in=normalize_address(token_in),
                token_out=normalize_address(token_out),
                gas_estimate=pool.gas_estimate,
            )

        except BalancerError as e:
            logger.debug(
                "weighted_amm_swap_failed",
                pool_id=pool.id,
                token_in=token_in,
                token_out=token_out,
                amount_in=amount_in,
                error=str(e),
            )
            return None

    def simulate_swap_exact_output(
        self,
        pool: BalancerWeightedPool,
        token_in: str,
        token_out: str,
        amount_out: int,
    ) -> SwapResult | None:
        """Simulate a swap to get exact output amount (buy order).

        Args:
            pool: The weighted pool
            token_in: Input token address
            token_out: Output token address
            amount_out: Desired output amount

        Returns:
            SwapResult with required input and desired output, or None if swap fails
        """
        from solver.amm.base import SwapResult

        reserves = _get_weighted_reserves(pool, token_in, token_out)
        if reserves is None:
            return None
        reserve_in, reserve_out = reserves

        try:
            # Scale up balances and amount
            balance_in = scale_up(reserve_in.balance, reserve_in.scaling_factor)
            balance_out = scale_up(reserve_out.balance, reserve_out.scaling_factor)
            amount_out_scaled = scale_up(amount_out, reserve_out.scaling_factor)

            # Convert weights to Bfp
            weight_in = Bfp.from_decimal(reserve_in.weight)
            weight_out = Bfp.from_decimal(reserve_out.weight)

            # Calculate required input (before fee)
            amount_in_raw = calc_in_given_out(
                balance_in=balance_in,
                weight_in=weight_in,
                balance_out=balance_out,
                weight_out=weight_out,
                amount_out=amount_out_scaled,
                _version=pool.version,
            )

            # Add fee to input
            amount_in_with_fee = add_swap_fee_amount(amount_in_raw.value, pool.fee)

            # Scale down input (round up for buy orders)
            amount_in_result = scale_down_up(Bfp(amount_in_with_fee), reserve_in.scaling_factor)

            return SwapResult(
                amount_in=amount_in_result,
                amount_out=amount_out,
                pool_address=pool.address,
                token_in=normalize_address(token_in),
                token_out=normalize_address(token_out),
                gas_estimate=pool.gas_estimate,
            )

        except BalancerError as e:
            logger.debug(
                "weighted_amm_exact_output_failed",
                pool_id=pool.id,
                token_in=token_in,
                token_out=token_out,
                amount_out=amount_out,
                error=str(e),
            )
            return None

    def max_fill_sell_order(
        self,
        pool: BalancerWeightedPool,
        token_in: str,
        token_out: str,
        sell_amount: int,
        buy_amount: int,
    ) -> int:
        """Calculate maximum input for a sell order that satisfies the limit price.

        Uses binary search since weighted pool math doesn't yield a closed-form solution.
        """
        return _binary_search_max_sell_fill(
            simulate_fn=lambda amt: self.simulate_swap(pool, token_in, token_out, amt),
            sell_amount=sell_amount,
            buy_amount=buy_amount,
        )

    def max_fill_buy_order(
        self,
        pool: BalancerWeightedPool,
        token_in: str,
        token_out: str,
        sell_amount: int,
        buy_amount: int,
    ) -> int:
        """Calculate maximum output for a buy order that satisfies the limit price.

        Uses binary search since weighted pool math doesn't yield a closed-form solution.
        """
        return _binary_search_max_buy_fill(
            simulate_fn=lambda amt: self.simulate_swap_exact_output(pool, token_in, token_out, amt),
            sell_amount=sell_amount,
            buy_amount=buy_amount,
        )


class BalancerStableAMM:
    """Balancer V2 stable pool AMM for swap simulation.

    Provides simulate_swap and simulate_swap_exact_output methods
    compatible with the router's pool selection logic.
    """

    def simulate_swap(
        self,
        pool: BalancerStablePool,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> SwapResult | None:
        """Simulate a swap through a stable pool (exact input).

        Args:
            pool: The stable pool
            token_in: Input token address
            token_out: Output token address
            amount_in: Amount of input token

        Returns:
            SwapResult with amounts and pool info, or None if swap fails
        """
        from solver.amm.base import SwapResult

        reserves = _get_stable_reserves(pool, token_in, token_out)
        if reserves is None:
            return None
        reserve_in, index_in, reserve_out, index_out = reserves

        try:
            # Build scaled balances list
            balances = [scale_up(r.balance, r.scaling_factor) for r in pool.reserves]

            # Scale input amount and subtract fee
            amount_in_scaled = scale_up(amount_in, reserve_in.scaling_factor)
            amount_in_after_fee = subtract_swap_fee_amount(amount_in_scaled.value, pool.fee)

            # Calculate output
            # Note: amp must be scaled by AMP_PRECISION (JSON has raw A value)
            amount_out_scaled = stable_calc_out_given_in(
                amp=int(pool.amplification_parameter * AMP_PRECISION),
                balances=balances,
                token_index_in=index_in,
                token_index_out=index_out,
                amount_in=Bfp(amount_in_after_fee),
            )

            # Scale down output
            amount_out_result = scale_down_down(amount_out_scaled, reserve_out.scaling_factor)

            return SwapResult(
                amount_in=amount_in,
                amount_out=amount_out_result,
                pool_address=pool.address,
                token_in=normalize_address(token_in),
                token_out=normalize_address(token_out),
                gas_estimate=pool.gas_estimate,
            )

        except BalancerError as e:
            logger.debug(
                "stable_amm_swap_failed",
                pool_id=pool.id,
                token_in=token_in,
                token_out=token_out,
                amount_in=amount_in,
                error=str(e),
            )
            return None

    def simulate_swap_exact_output(
        self,
        pool: BalancerStablePool,
        token_in: str,
        token_out: str,
        amount_out: int,
    ) -> SwapResult | None:
        """Simulate a swap to get exact output amount (buy order).

        Args:
            pool: The stable pool
            token_in: Input token address
            token_out: Output token address
            amount_out: Desired output amount

        Returns:
            SwapResult with required input and desired output, or None if swap fails
        """
        from solver.amm.base import SwapResult

        reserves = _get_stable_reserves(pool, token_in, token_out)
        if reserves is None:
            return None
        reserve_in, index_in, reserve_out, index_out = reserves

        try:
            # Build scaled balances list
            balances = [scale_up(r.balance, r.scaling_factor) for r in pool.reserves]

            # Scale output amount
            amount_out_scaled = scale_up(amount_out, reserve_out.scaling_factor)

            # Calculate required input (before fee)
            # Note: amp must be scaled by AMP_PRECISION (JSON has raw A value)
            amount_in_raw = stable_calc_in_given_out(
                amp=int(pool.amplification_parameter * AMP_PRECISION),
                balances=balances,
                token_index_in=index_in,
                token_index_out=index_out,
                amount_out=amount_out_scaled,
            )

            # Add fee to input and scale down (round up for buy orders)
            amount_in_with_fee = add_swap_fee_amount(amount_in_raw.value, pool.fee)
            amount_in_result = scale_down_up(Bfp(amount_in_with_fee), reserve_in.scaling_factor)

            return SwapResult(
                amount_in=amount_in_result,
                amount_out=amount_out,
                pool_address=pool.address,
                token_in=normalize_address(token_in),
                token_out=normalize_address(token_out),
                gas_estimate=pool.gas_estimate,
            )

        except BalancerError as e:
            logger.debug(
                "stable_amm_exact_output_failed",
                pool_id=pool.id,
                token_in=token_in,
                token_out=token_out,
                amount_out=amount_out,
                error=str(e),
            )
            return None

    def max_fill_sell_order(
        self,
        pool: BalancerStablePool,
        token_in: str,
        token_out: str,
        sell_amount: int,
        buy_amount: int,
    ) -> int:
        """Calculate maximum input for a sell order that satisfies the limit price.

        Uses binary search since stable pool math doesn't yield a closed-form solution.
        """
        return _binary_search_max_sell_fill(
            simulate_fn=lambda amt: self.simulate_swap(pool, token_in, token_out, amt),
            sell_amount=sell_amount,
            buy_amount=buy_amount,
        )

    def max_fill_buy_order(
        self,
        pool: BalancerStablePool,
        token_in: str,
        token_out: str,
        sell_amount: int,
        buy_amount: int,
    ) -> int:
        """Calculate maximum output for a buy order that satisfies the limit price.

        Uses binary search since stable pool math doesn't yield a closed-form solution.
        """
        return _binary_search_max_buy_fill(
            simulate_fn=lambda amt: self.simulate_swap_exact_output(pool, token_in, token_out, amt),
            sell_amount=sell_amount,
            buy_amount=buy_amount,
        )
