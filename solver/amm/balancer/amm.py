"""Balancer AMM classes.

High-level AMM classes for swap simulation through Balancer pools.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any

import structlog

from solver.math.fixed_point import AMP_PRECISION, Bfp
from solver.models.types import normalize_address
from solver.safe_int import UINT256_MAX, S

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
    from collections.abc import Callable

    from solver.amm.base import SwapResult

logger = structlog.get_logger()


# =============================================================================
# Forward Verification (matches Rust's converge_in_amount exactly)
# =============================================================================


def converge_in_amount(
    in_amount: int,
    exact_out_amount: int,
    get_amount_out: Callable[[int], int | None],
) -> tuple[int, int] | None:
    """Verify and bump input amount until forward simulation yields sufficient output.

    Balancer V2 pools are "unstable": computing an input amount large enough to
    buy X tokens, then selling that computed amount in the same pool state, may
    yield X-δ tokens. This function bumps the input until forward simulation
    gives >= requested output.

    This is a direct translation of Rust's converge_in_amount from:
    crates/shared/src/sources/balancer_v2/swap/mod.rs:335-364

    Args:
        in_amount: Initial input amount (after fee, in token decimals)
        exact_out_amount: Required output amount
        get_amount_out: Function that simulates selling in_amount, returns output or None

    Returns:
        Tuple of (converged_input, actual_output), or None if convergence fails.
        The actual_output may be >= exact_out_amount due to rounding.
    """
    out_amount = get_amount_out(in_amount)
    if out_amount is None:
        return None
    if out_amount >= exact_out_amount:
        return (in_amount, out_amount)

    # Approximate the output deficit in input tokens at current trading price,
    # then multiply by 10 for each iteration.
    # bump = ceil((exact_out - out) * in_amount / max(out, 1))
    deficit = exact_out_amount - out_amount
    divisor = max(out_amount, 1)
    bump = max(1, (deficit * in_amount + divisor - 1) // divisor)  # ceil_div

    for _ in range(6):
        bumped_in_amount = in_amount + bump
        # Overflow protection: if bumped amount exceeds uint256, give up
        if bumped_in_amount > UINT256_MAX:
            logger.debug(
                "converge_in_amount_overflow",
                in_amount=in_amount,
                bump=bump,
                message="Bump would overflow uint256",
            )
            return None
        out_amount = get_amount_out(bumped_in_amount)
        if out_amount is None:
            return None
        if out_amount >= exact_out_amount:
            return (bumped_in_amount, out_amount)
        # Overflow protection: cap bump at uint256 max to prevent unbounded growth
        if bump > UINT256_MAX // 10:
            logger.debug(
                "converge_in_amount_bump_overflow",
                bump=bump,
                message="Bump multiplication would overflow uint256",
            )
            return None
        bump *= 10

    return None


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
    if normalize_address(token_in) == normalize_address(token_out):
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

    # Zero-balance check to prevent division by zero in swap calculations
    if reserve_in.balance == 0:
        logger.debug(
            "weighted_amm_zero_balance",
            pool_id=pool.id,
            token=token_in,
            role="input",
        )
        return None

    if reserve_out.balance == 0:
        logger.debug(
            "weighted_amm_zero_balance",
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

    token_in_norm = normalize_address(token_in)
    token_out_norm = normalize_address(token_out)

    for i, reserve in enumerate(pool.reserves):
        token_norm = normalize_address(reserve.token)
        if token_norm == token_in_norm:
            reserve_in = reserve
            index_in = i
        elif token_norm == token_out_norm:
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

    # Zero-balance check to prevent division by zero in swap calculations
    if reserve_in.balance == 0:
        logger.debug(
            "stable_amm_zero_balance",
            pool_id=pool.id,
            token=token_in,
            role="input",
        )
        return None

    if reserve_out.balance == 0:
        logger.debug(
            "stable_amm_zero_balance",
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
        # Use SafeInt for overflow protection on large amounts
        if S(result.amount_out) * S(sell_amount) >= S(buy_amount) * S(mid):
            lo = mid
        else:
            hi = mid - 1

    # Verify the final result
    if lo > 0:
        result = simulate_fn(lo)
        if result is None or S(result.amount_out) * S(sell_amount) < S(buy_amount) * S(lo):
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
        # Use SafeInt for overflow protection on large amounts
        if S(result.amount_in) * S(buy_amount) <= S(sell_amount) * S(mid):
            lo = mid
        else:
            hi = mid - 1

    # Verify the final result
    if lo > 0:
        result = simulate_fn(lo)
        if result is None or S(result.amount_in) * S(buy_amount) > S(sell_amount) * S(lo):
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

            # Order matches Rust: downscale_up first, then add_swap_fee_amount
            amount_in_before_fee = scale_down_up(amount_in_raw, reserve_in.scaling_factor)
            in_amount = add_swap_fee_amount(amount_in_before_fee, pool.fee)

            # Forward verification: bump input until forward sim gives >= requested output
            def get_amount_out_weighted(amt_in: int) -> int | None:
                sim_result = self.simulate_swap(pool, token_in, token_out, amt_in)
                if sim_result is None:
                    return None
                return sim_result.amount_out

            converged = converge_in_amount(in_amount, amount_out, get_amount_out_weighted)
            if converged is None:
                return None

            converged_input, actual_output = converged
            return SwapResult(
                amount_in=converged_input,
                amount_out=actual_output,  # Use actual forward-simulated output
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
            # Use explicit rounding for Decimal-to-int conversion
            amp_scaled = (pool.amplification_parameter * AMP_PRECISION).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            amount_out_scaled = stable_calc_out_given_in(
                amp=int(amp_scaled),
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
            # Use explicit rounding for Decimal-to-int conversion
            amp_scaled = (pool.amplification_parameter * AMP_PRECISION).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            amount_in_raw = stable_calc_in_given_out(
                amp=int(amp_scaled),
                balances=balances,
                token_index_in=index_in,
                token_index_out=index_out,
                amount_out=amount_out_scaled,
            )

            # Order matches Rust: downscale_up first, then add_swap_fee_amount
            amount_in_before_fee = scale_down_up(amount_in_raw, reserve_in.scaling_factor)
            in_amount = add_swap_fee_amount(amount_in_before_fee, pool.fee)

            # Forward verification: bump input until forward sim gives >= requested output
            def get_amount_out_stable(amt_in: int) -> int | None:
                sim_result = self.simulate_swap(pool, token_in, token_out, amt_in)
                if sim_result is None:
                    return None
                return sim_result.amount_out

            converged = converge_in_amount(in_amount, amount_out, get_amount_out_stable)
            if converged is None:
                return None

            converged_input, actual_output = converged
            return SwapResult(
                amount_in=converged_input,
                amount_out=actual_output,  # Use actual forward-simulated output
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
