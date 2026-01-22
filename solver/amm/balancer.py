"""Balancer weighted and stable pool implementations.

This module provides swap calculations for Balancer V2 pools, matching the
Rust baseline solver's implementation exactly.

Pool types supported:
- Weighted Product (V0 and V3Plus)
- Stable pools (StableSwap / Curve-style)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Literal

import structlog

from solver.math.fixed_point import (
    AMP_PRECISION,
    MAX_IN_RATIO,
    MAX_OUT_RATIO,
    ONE_18,
    Bfp,
)
from solver.models.types import normalize_address

if TYPE_CHECKING:
    from solver.amm.base import SwapResult
    from solver.models.auction import Liquidity, TokenBalance

logger = structlog.get_logger()

__all__ = [
    # Weighted pool dataclasses
    "WeightedTokenReserve",
    "BalancerWeightedPool",
    # Stable pool dataclasses
    "StableTokenReserve",
    "BalancerStablePool",
    # AMM classes
    "BalancerWeightedAMM",
    "BalancerStableAMM",
    # Weighted pool math functions
    "calc_out_given_in",
    "calc_in_given_out",
    # Stable pool math functions
    "calculate_invariant",
    "get_token_balance_given_invariant_and_all_other_balances",
    "stable_calc_out_given_in",
    "stable_calc_in_given_out",
    # Stable pool helpers
    "filter_bpt_token",
    # Pool parsing
    "parse_weighted_pool",
    "parse_stable_pool",
    # Scaling helpers
    "scale_up",
    "scale_down_down",
    "scale_down_up",
    # Fee helpers
    "subtract_swap_fee_amount",
    "add_swap_fee_amount",
    # Errors
    "BalancerError",
    "MaxInRatioError",
    "MaxOutRatioError",
    "InvalidFeeError",
    "InvalidScalingFactorError",
    "ZeroWeightError",
    "ZeroBalanceError",
    "StableInvariantDidNotConverge",
    "StableGetBalanceDidNotConverge",
]


# =============================================================================
# Errors
# =============================================================================


class BalancerError(Exception):
    """Base error for Balancer operations."""

    pass


class MaxInRatioError(BalancerError):
    """Error 304: Input amount exceeds 30% of balance_in."""

    pass


class MaxOutRatioError(BalancerError):
    """Error 305: Output amount exceeds 30% of balance_out."""

    pass


class InvalidFeeError(BalancerError):
    """Swap fee must be in range [0, 1)."""

    pass


class InvalidScalingFactorError(BalancerError):
    """Scaling factor must be positive."""

    pass


class ZeroWeightError(BalancerError):
    """Token weight must be positive."""

    pass


class ZeroBalanceError(BalancerError):
    """Token balance must be positive for swaps."""

    pass


class StableInvariantDidNotConverge(BalancerError):
    """Newton-Raphson iteration for stable invariant D did not converge."""

    pass


class StableGetBalanceDidNotConverge(BalancerError):
    """Newton-Raphson iteration for stable balance Y did not converge."""

    pass


# =============================================================================
# Data classes
# =============================================================================


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


# =============================================================================
# Scaling helpers
# =============================================================================


def scale_up(amount: int, scaling_factor: int) -> Bfp:
    """Scale token amount to 18 decimals for internal math.

    Args:
        amount: Amount in token's native decimals
        scaling_factor: Factor to scale by (e.g., 10^12 for 6-decimal tokens)

    Returns:
        Amount as Bfp (18-decimal fixed-point)

    Raises:
        InvalidScalingFactorError: If scaling_factor <= 0
    """
    if scaling_factor <= 0:
        raise InvalidScalingFactorError(f"Scaling factor must be positive, got {scaling_factor}")
    return Bfp.from_wei(amount * scaling_factor)


def scale_down_down(bfp: Bfp, scaling_factor: int) -> int:
    """Scale 18-decimal result back to token decimals, rounding down.

    Args:
        bfp: Amount in 18-decimal fixed-point
        scaling_factor: Factor to scale by

    Returns:
        Amount in token's native decimals

    Raises:
        InvalidScalingFactorError: If scaling_factor <= 0
    """
    if scaling_factor <= 0:
        raise InvalidScalingFactorError(f"Scaling factor must be positive, got {scaling_factor}")
    return bfp.value // scaling_factor


def scale_down_up(bfp: Bfp, scaling_factor: int) -> int:
    """Scale 18-decimal result back to token decimals, rounding up.

    Args:
        bfp: Amount in 18-decimal fixed-point
        scaling_factor: Factor to scale by

    Returns:
        Amount in token's native decimals

    Raises:
        InvalidScalingFactorError: If scaling_factor <= 0
    """
    if scaling_factor <= 0:
        raise InvalidScalingFactorError(f"Scaling factor must be positive, got {scaling_factor}")
    if bfp.value == 0:
        return 0
    return (bfp.value - 1) // scaling_factor + 1


# =============================================================================
# Fee application
# =============================================================================


def subtract_swap_fee_amount(amount: int, swap_fee: Decimal) -> int:
    """Subtract swap fee from input amount.

    Used for sell orders (exact input): fee is deducted before the swap.

    Args:
        amount: Input amount before fee
        swap_fee: Fee as decimal (e.g., 0.003 for 0.3%), must be in [0, 1)

    Returns:
        Amount after fee deduction

    Raises:
        InvalidFeeError: If swap_fee is not in range [0, 1)
    """
    if swap_fee < 0 or swap_fee >= 1:
        raise InvalidFeeError(f"Swap fee must be in range [0, 1), got {swap_fee}")
    amount_bfp = Bfp.from_wei(amount)
    fee_bfp = Bfp.from_decimal(swap_fee)
    fee_amount = amount_bfp.mul_up(fee_bfp)
    amount_without_fees = amount_bfp.sub(fee_amount)
    return amount_without_fees.value


def add_swap_fee_amount(amount: int, swap_fee: Decimal) -> int:
    """Add swap fee to the calculated input amount.

    Used for buy orders (exact output): after calculating the raw input
    needed via calc_in_given_out, this function adds the fee on top.

    Formula: amount_with_fee = amount / (1 - fee)

    Args:
        amount: Raw input amount from calc_in_given_out (before fee consideration)
        swap_fee: Fee as decimal (e.g., 0.003 for 0.3%), must be in [0, 1)

    Returns:
        Final input amount including fee

    Raises:
        InvalidFeeError: If swap_fee is not in range [0, 1)
    """
    if swap_fee < 0 or swap_fee >= 1:
        raise InvalidFeeError(f"Swap fee must be in range [0, 1), got {swap_fee}")
    fee_bfp = Bfp.from_decimal(swap_fee)
    complement = fee_bfp.complement()  # 1 - fee
    amount_bfp = Bfp.from_wei(amount)
    amount_with_fees = amount_bfp.div_up(complement)
    return amount_with_fees.value


# =============================================================================
# Weighted pool math
# =============================================================================


def calc_out_given_in(
    balance_in: Bfp,
    weight_in: Bfp,
    balance_out: Bfp,
    weight_out: Bfp,
    amount_in: Bfp,
    *,
    _version: Literal["v0", "v3Plus"] = "v0",
) -> Bfp:
    """Calculate output amount for a given input (sell order).

    This is the core weighted pool formula. Fee should be subtracted from
    amount_in BEFORE calling this function.

    Formula:
        amount_out = balance_out * (1 - (balance_in / (balance_in + amount_in))^(weight_in / weight_out))

    Args:
        balance_in: Scaled balance of input token (must be positive)
        weight_in: Weight of input token (must be positive)
        balance_out: Scaled balance of output token (must be positive)
        weight_out: Weight of output token (must be positive)
        amount_in: Scaled input amount (after fee subtraction)
        _version: Pool version for power function variant (V3Plus optimization TODO)

    Returns:
        Scaled output amount

    Raises:
        MaxInRatioError: If amount_in > balance_in * 0.3 (30% limit)
        ZeroWeightError: If weight_in or weight_out is zero
        ZeroBalanceError: If balance_in or balance_out is zero
    """
    # Validate weights
    if weight_in.value <= 0:
        raise ZeroWeightError("weight_in must be positive")
    if weight_out.value <= 0:
        raise ZeroWeightError("weight_out must be positive")

    # Validate balances
    if balance_in.value <= 0:
        raise ZeroBalanceError("balance_in must be positive")
    if balance_out.value <= 0:
        raise ZeroBalanceError("balance_out must be positive")

    # Check ratio limit: amount_in must not exceed 30% of balance_in
    max_amount_in = balance_in.mul_down(MAX_IN_RATIO)
    if amount_in.value > max_amount_in.value:
        raise MaxInRatioError(f"Input {amount_in.value} exceeds 30% of balance {balance_in.value}")

    # denominator = balance_in + amount_in
    denominator = balance_in.add(amount_in)

    # base = balance_in / denominator (rounded up for conservative estimate)
    base = balance_in.div_up(denominator)

    # exponent = weight_in / weight_out (rounded down)
    exponent = weight_in.div_down(weight_out)

    # power = base ^ exponent (rounded up)
    # V3Plus uses optimized pow_up_v3 for common exponents (1, 2, 4)
    power = base.pow_up_v3(exponent) if _version == "v3Plus" else base.pow_up(exponent)

    # amount_out = balance_out * (1 - power) (rounded down)
    return balance_out.mul_down(power.complement())


def calc_in_given_out(
    balance_in: Bfp,
    weight_in: Bfp,
    balance_out: Bfp,
    weight_out: Bfp,
    amount_out: Bfp,
    *,
    _version: Literal["v0", "v3Plus"] = "v0",
) -> Bfp:
    """Calculate input amount for a given output (buy order).

    This is the core weighted pool formula. Fee should be added to the result
    AFTER calling this function.

    Formula:
        amount_in = balance_in * ((balance_out / (balance_out - amount_out))^(weight_out / weight_in) - 1)

    Args:
        balance_in: Scaled balance of input token (must be positive)
        weight_in: Weight of input token (must be positive)
        balance_out: Scaled balance of output token (must be positive)
        weight_out: Weight of output token (must be positive)
        amount_out: Scaled output amount
        _version: Pool version for power function variant (V3Plus optimization TODO)

    Returns:
        Scaled input amount (before fee addition)

    Raises:
        MaxOutRatioError: If amount_out > balance_out * 0.3 (30% limit)
        ZeroWeightError: If weight_in or weight_out is zero
        ZeroBalanceError: If balance_in or balance_out is zero or if amount_out >= balance_out
    """
    # Validate weights
    if weight_in.value <= 0:
        raise ZeroWeightError("weight_in must be positive")
    if weight_out.value <= 0:
        raise ZeroWeightError("weight_out must be positive")

    # Validate balances
    if balance_in.value <= 0:
        raise ZeroBalanceError("balance_in must be positive")
    if balance_out.value <= 0:
        raise ZeroBalanceError("balance_out must be positive")

    # Check ratio limit: amount_out must not exceed 30% of balance_out
    max_amount_out = balance_out.mul_down(MAX_OUT_RATIO)
    if amount_out.value > max_amount_out.value:
        raise MaxOutRatioError(
            f"Output {amount_out.value} exceeds 30% of balance {balance_out.value}"
        )

    # denominator = balance_out - amount_out
    # Check for zero denominator (would cause division by zero)
    if amount_out.value >= balance_out.value:
        raise ZeroBalanceError("amount_out must be less than balance_out")
    denominator = balance_out.sub(amount_out)

    # base = balance_out / denominator (rounded up)
    base = balance_out.div_up(denominator)

    # exponent = weight_out / weight_in (rounded UP for buy orders - differs from calc_out!)
    exponent = weight_out.div_up(weight_in)

    # power = base ^ exponent (rounded up)
    # V3Plus uses optimized pow_up_v3 for common exponents (1, 2, 4)
    power = base.pow_up_v3(exponent) if _version == "v3Plus" else base.pow_up(exponent)

    # ratio = power - 1
    one = Bfp(ONE_18)
    ratio = power.sub(one)

    # amount_in = balance_in * ratio (rounded up)
    return balance_in.mul_up(ratio)


# =============================================================================
# Stable pool math
# =============================================================================

# Maximum iterations for Newton-Raphson convergence
_STABLE_MAX_ITERATIONS = 255


def calculate_invariant(amp: int, balances: list[Bfp]) -> Bfp:
    """Calculate StableSwap invariant D using Newton-Raphson iteration.

    Uses Balancer's parameterization where the Newton-Raphson formula uses
    A*n (not A*n^n). The n^n factor is incorporated through the iterative
    d_p calculation.

    Algorithm:
        1. Initial guess: D = sum(balances)
        2. Iterate using Newton-Raphson until |D_new - D_old| <= 1 wei
        3. Max iterations: 255

    Args:
        amp: Amplification parameter (scaled by AMP_PRECISION=1000)
        balances: List of token balances (already scaled to 18 decimals)

    Returns:
        The calculated invariant D as Bfp

    Raises:
        StableInvariantDidNotConverge: If iteration doesn't converge
        ZeroBalanceError: If any balance is zero
    """
    n_coins = len(balances)
    if n_coins == 0:
        return Bfp(0)

    # Check for zero balances
    for i, bal in enumerate(balances):
        if bal.value <= 0:
            raise ZeroBalanceError(f"Balance at index {i} must be positive")

    # Calculate sum of balances
    sum_balances = sum(b.value for b in balances)

    # Initial guess: D = sum(balances)
    d_prev = sum_balances

    # amp_times_n = A * n (Balancer convention, NOT A * n^n)
    # Note: amp already includes AMP_PRECISION factor
    amp_times_n = amp * n_coins

    for _ in range(_STABLE_MAX_ITERATIONS):
        # d_p = D^(n+1) / (n^n * prod(balances))
        # Computed iteratively: d_p = D, then d_p = d_p * D / (n * balance_i) for each i
        d_p = d_prev
        for bal in balances:
            # d_p = d_p * D / (n * balance)
            # Using integer division that rounds down
            d_p = (d_p * d_prev) // (n_coins * bal.value)

        # Newton-Raphson numerator:
        # From Rust: ((ann * sum / AMP_PRECISION + d_p * n_coins) * d)
        term1 = (amp_times_n * sum_balances) // AMP_PRECISION
        numerator = (term1 + d_p * n_coins) * d_prev

        # Newton-Raphson denominator:
        # From Rust: ((ann - AMP_PRECISION) * d / AMP_PRECISION + (n_coins + 1) * d_p)
        term2 = ((amp_times_n - AMP_PRECISION) * d_prev) // AMP_PRECISION
        denominator = term2 + (n_coins + 1) * d_p

        # d_new = numerator / denominator
        d_new = numerator // denominator

        # Check convergence: |d_new - d_prev| <= 1
        if d_new > d_prev:
            if d_new - d_prev <= 1:
                return Bfp(d_new)
        else:
            if d_prev - d_new <= 1:
                return Bfp(d_new)

        d_prev = d_new

    raise StableInvariantDidNotConverge(
        f"Stable invariant did not converge after {_STABLE_MAX_ITERATIONS} iterations"
    )


def get_token_balance_given_invariant_and_all_other_balances(
    amp: int,
    balances: list[Bfp],
    invariant: Bfp,
    token_index: int,
) -> Bfp:
    """Solve for balance[token_index] given D and all other balances.

    Uses Newton-Raphson iteration to find y (the unknown balance) such that
    the StableSwap invariant is preserved.

    This implementation exactly matches Balancer's StableMath.sol:
    https://github.com/balancer-labs/balancer-v2-monorepo/blob/stable-deployment/pkg/pool-stable/contracts/StableMath.sol#L465-L516

    Args:
        amp: Amplification parameter (scaled by AMP_PRECISION=1000)
        balances: List of token balances (the value at token_index will be used in c calculation)
        invariant: The invariant D to preserve
        token_index: Index of the token whose balance we're solving for

    Returns:
        The calculated balance as Bfp

    Raises:
        StableGetBalanceDidNotConverge: If iteration doesn't converge
        IndexError: If token_index is out of range
    """
    n_coins = len(balances)
    if token_index < 0 or token_index >= n_coins:
        raise IndexError(f"token_index {token_index} out of range for {n_coins} tokens")

    d = invariant.value
    amp_times_total = amp * n_coins

    # Calculate P_D and sum
    # P_D starts as balance[0] * n
    # For each subsequent balance j: P_D = P_D * balance[j] * n / invariant
    sum_balances = balances[0].value
    p_d = balances[0].value * n_coins

    for j in range(1, n_coins):
        # P_D = Math.divDown(Math.mul(Math.mul(P_D, balances[j]), balances.length), invariant)
        p_d = (p_d * balances[j].value * n_coins) // d
        sum_balances += balances[j].value

    # sum = sum - balances[token_index]
    sum_others = sum_balances - balances[token_index].value

    inv2 = d * d

    # c = inv2 / (ampTimesTotal * P_D) * AMP_PRECISION * balances[tokenIndex]
    # Using div_up for the first division
    amp_times_p_d = amp_times_total * p_d
    # div_up: (inv2 + amp_times_p_d - 1) // amp_times_p_d
    if amp_times_p_d == 0:
        raise StableGetBalanceDidNotConverge("amp_times_p_d is zero")
    c_step1 = (inv2 + amp_times_p_d - 1) // amp_times_p_d  # div_up
    c = c_step1 * AMP_PRECISION * balances[token_index].value

    # b = sum_others + invariant / ampTimesTotal * AMP_PRECISION
    # Using div_down for the division
    b = sum_others + (d // amp_times_total) * AMP_PRECISION

    # Initial guess: tokenBalance = (inv2 + c) / (invariant + b)
    # Using div_up
    numerator_init = inv2 + c
    denominator_init = d + b
    token_balance = (numerator_init + denominator_init - 1) // denominator_init  # div_up

    for _ in range(_STABLE_MAX_ITERATIONS):
        prev_token_balance = token_balance

        # tokenBalance = (tokenBalance² + c) / (2*tokenBalance + b - invariant)
        # Using div_up
        numerator = token_balance * token_balance + c
        denominator = 2 * token_balance + b - d

        if denominator <= 0:
            raise StableGetBalanceDidNotConverge("Denominator became non-positive")

        token_balance = (numerator + denominator - 1) // denominator  # div_up

        # Check convergence: |token_balance - prev_token_balance| <= 1
        if token_balance > prev_token_balance:
            if token_balance - prev_token_balance <= 1:
                return Bfp(token_balance)
        else:
            if prev_token_balance - token_balance <= 1:
                return Bfp(token_balance)

    raise StableGetBalanceDidNotConverge(
        f"Stable get_balance did not converge after {_STABLE_MAX_ITERATIONS} iterations"
    )


def stable_calc_out_given_in(
    amp: int,
    balances: list[Bfp],
    token_index_in: int,
    token_index_out: int,
    amount_in: Bfp,
) -> Bfp:
    """Calculate output amount for a given input in a stable pool.

    Fee should be subtracted from amount_in BEFORE calling this function.
    Unlike weighted pools, stable pools do not enforce ratio limits.

    Algorithm:
        1. Calculate current invariant D
        2. Add amount_in to balances[token_index_in]
        3. Solve for new balances[token_index_out] given D
        4. Return: old_balance_out - new_balance_out - 1 (1 wei rounding protection)

    Args:
        amp: Amplification parameter (scaled by AMP_PRECISION=1000)
        balances: List of scaled token balances (18 decimals)
        token_index_in: Index of input token
        token_index_out: Index of output token
        amount_in: Scaled input amount (after fee subtraction)

    Returns:
        Scaled output amount

    Raises:
        StableInvariantDidNotConverge: If invariant calculation doesn't converge
        StableGetBalanceDidNotConverge: If balance calculation doesn't converge
        ValueError: If token_index_in == token_index_out
        IndexError: If token indices are out of range
    """
    # Validate indices
    n_coins = len(balances)
    if token_index_in < 0 or token_index_in >= n_coins:
        raise IndexError(f"token_index_in {token_index_in} out of range for {n_coins} tokens")
    if token_index_out < 0 or token_index_out >= n_coins:
        raise IndexError(f"token_index_out {token_index_out} out of range for {n_coins} tokens")
    if token_index_in == token_index_out:
        raise ValueError("Cannot swap token with itself")

    # Calculate current invariant
    invariant = calculate_invariant(amp, balances)

    # Create new balances list with updated input balance
    new_balances = list(balances)
    new_balances[token_index_in] = Bfp(balances[token_index_in].value + amount_in.value)

    # Calculate new output balance that preserves invariant
    new_balance_out = get_token_balance_given_invariant_and_all_other_balances(
        amp, new_balances, invariant, token_index_out
    )

    # Output = old_balance_out - new_balance_out - 1 (rounding protection)
    old_balance_out = balances[token_index_out].value
    new_balance_out_val = new_balance_out.value

    # Ensure we don't underflow
    if new_balance_out_val >= old_balance_out:
        return Bfp(0)

    amount_out = old_balance_out - new_balance_out_val - 1
    return Bfp(amount_out)


def stable_calc_in_given_out(
    amp: int,
    balances: list[Bfp],
    token_index_in: int,
    token_index_out: int,
    amount_out: Bfp,
) -> Bfp:
    """Calculate input amount for a given output in a stable pool.

    Fee should be added to the result AFTER calling this function.
    Unlike weighted pools, stable pools do not enforce ratio limits.

    Algorithm:
        1. Calculate current invariant D
        2. Subtract amount_out from balances[token_index_out]
        3. Solve for new balances[token_index_in] given D
        4. Return: new_balance_in - old_balance_in + 1 (1 wei rounding protection)

    Args:
        amp: Amplification parameter (scaled by AMP_PRECISION=1000)
        balances: List of scaled token balances (18 decimals)
        token_index_in: Index of input token
        token_index_out: Index of output token
        amount_out: Scaled output amount

    Returns:
        Scaled input amount (before fee addition)

    Raises:
        StableInvariantDidNotConverge: If invariant calculation doesn't converge
        StableGetBalanceDidNotConverge: If balance calculation doesn't converge
        ZeroBalanceError: If amount_out >= balance_out
        ValueError: If token_index_in == token_index_out
        IndexError: If token indices are out of range
    """
    # Validate indices
    n_coins = len(balances)
    if token_index_in < 0 or token_index_in >= n_coins:
        raise IndexError(f"token_index_in {token_index_in} out of range for {n_coins} tokens")
    if token_index_out < 0 or token_index_out >= n_coins:
        raise IndexError(f"token_index_out {token_index_out} out of range for {n_coins} tokens")
    if token_index_in == token_index_out:
        raise ValueError("Cannot swap token with itself")

    # Check that we're not requesting more than available
    if amount_out.value >= balances[token_index_out].value:
        raise ZeroBalanceError("amount_out must be less than balance_out")

    # Calculate current invariant
    invariant = calculate_invariant(amp, balances)

    # Create new balances list with updated output balance
    new_balances = list(balances)
    new_balances[token_index_out] = Bfp(balances[token_index_out].value - amount_out.value)

    # Calculate new input balance that preserves invariant
    new_balance_in = get_token_balance_given_invariant_and_all_other_balances(
        amp, new_balances, invariant, token_index_in
    )

    # Input = new_balance_in - old_balance_in + 1 (rounding protection)
    old_balance_in = balances[token_index_in].value
    new_balance_in_val = new_balance_in.value

    amount_in = new_balance_in_val - old_balance_in + 1
    return Bfp(amount_in)


def filter_bpt_token(pool: BalancerStablePool) -> BalancerStablePool:
    """Filter out BPT token from composable stable pool reserves.

    For composable stable pools, the pool's own BPT token is included
    in the reserves but must be filtered out before calculations.

    Detection: BPT token address == pool address

    Args:
        pool: The stable pool to filter

    Returns:
        A new pool with BPT token removed from reserves (if present)
    """
    pool_address_lower = pool.address.lower()
    filtered_reserves = tuple(r for r in pool.reserves if r.token.lower() != pool_address_lower)

    # If nothing was filtered, return original pool
    if len(filtered_reserves) == len(pool.reserves):
        return pool

    # Return new pool with filtered reserves
    return BalancerStablePool(
        id=pool.id,
        address=pool.address,
        pool_id=pool.pool_id,
        reserves=filtered_reserves,
        amplification_parameter=pool.amplification_parameter,
        fee=pool.fee,
        gas_estimate=pool.gas_estimate,
    )


# =============================================================================
# Pool Parsing
# =============================================================================


def _get_liquidity_extra(liquidity: Liquidity, key: str, default: Any = None) -> Any:
    """Get extra field from Liquidity model.

    Pydantic v2 stores extra fields in model_extra (dict).
    """
    # Try model_extra first (Pydantic v2)
    if hasattr(liquidity, "model_extra") and liquidity.model_extra and key in liquidity.model_extra:
        return liquidity.model_extra[key]

    # Fall back to direct attribute access
    return getattr(liquidity, key, default)


def _normalize_dict_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Normalize dictionary keys to lowercase for case-insensitive lookup."""
    return {k.lower(): v for k, v in d.items()}


def _parse_balance(
    token_data: TokenBalance,
    liquidity_id: str,
    token_addr: str,
    pool_type: str,
) -> int | None:
    """Parse and validate balance from token data.

    Returns None if balance is invalid or non-positive.
    """
    balance_raw = token_data.get("balance", "0")
    try:
        balance = int(balance_raw)
    except (ValueError, TypeError):
        logger.warning(
            f"{pool_type}_invalid_balance",
            liquidity_id=liquidity_id,
            token=token_addr,
            raw_balance=balance_raw,
        )
        return None

    if balance <= 0:
        logger.debug(
            f"{pool_type}_zero_balance",
            liquidity_id=liquidity_id,
            token=token_addr,
            balance=balance,
        )
        return None

    return balance


def _parse_scaling_factor(
    scaling_factors_dict: dict[str, Any],
    token_addr: str,
    liquidity_id: str,
    pool_type: str,
    token_data: TokenBalance | None = None,
) -> int:
    """Parse scaling factor with fallback to default of 1.

    Looks up scaling factor in two places:
    1. Top-level scaling_factors_dict (Python auction format)
    2. Per-token data with "scalingFactor" key (Rust auction format)

    Falls back to 1 if not found or invalid.
    """
    token_addr_lower = token_addr.lower()
    scaling_raw: Any = scaling_factors_dict.get(token_addr_lower)
    if scaling_raw is None and token_data is not None:
        # Try per-token data (Rust auction format uses "scalingFactor")
        scaling_raw = token_data.get("scalingFactor", "1")
    if scaling_raw is None:
        scaling_raw = "1"
    try:
        return int(scaling_raw)
    except (ValueError, TypeError):
        logger.warning(
            f"{pool_type}_invalid_scaling",
            liquidity_id=liquidity_id,
            token=token_addr,
            raw_scaling=scaling_raw,
        )
        return 1


def _parse_fee(
    liquidity_fee: str | None,
    default_fee: str,
    liquidity_id: str,
    pool_type: str,
) -> Decimal:
    """Parse fee with fallback to default."""
    if liquidity_fee:
        try:
            return Decimal(str(liquidity_fee))
        except (InvalidOperation, TypeError):
            logger.warning(
                f"{pool_type}_invalid_fee",
                liquidity_id=liquidity_id,
                raw_fee=liquidity_fee,
                using_default=default_fee,
            )
    return Decimal(default_fee)


def _parse_gas_estimate(gas_estimate_raw: str | None, default: int) -> int:
    """Parse gas estimate with fallback to default."""
    if gas_estimate_raw:
        with contextlib.suppress(ValueError, TypeError):
            return int(gas_estimate_raw)
    return default


def parse_weighted_pool(liquidity: Liquidity) -> BalancerWeightedPool | None:
    """Parse weightedProduct liquidity into BalancerWeightedPool.

    Args:
        liquidity: Liquidity source from the auction

    Returns:
        BalancerWeightedPool if liquidity is a weighted pool, None otherwise
    """
    if liquidity.kind != "weightedProduct":
        return None

    if liquidity.address is None:
        logger.debug("weighted_pool_missing_address", liquidity_id=liquidity.id)
        return None

    # tokens must be a dict with balance info
    if not isinstance(liquidity.tokens, dict):
        logger.debug("weighted_pool_invalid_tokens", liquidity_id=liquidity.id)
        return None

    # Get balancerPoolId from extra fields
    pool_id = _get_liquidity_extra(liquidity, "balancerPoolId")
    if pool_id is None:
        logger.debug("weighted_pool_missing_pool_id", liquidity_id=liquidity.id)
        return None

    # Try to get weights from top-level dict first (Python format)
    weights_dict_raw = _get_liquidity_extra(liquidity, "weights")
    weights_dict: dict[str, str] = {}
    if weights_dict_raw is not None and isinstance(weights_dict_raw, dict):
        weights_dict = _normalize_dict_keys(weights_dict_raw)

    # Get scaling factors dict (may be top-level or per-token)
    scaling_factors_raw = _get_liquidity_extra(liquidity, "scalingFactors", {})
    scaling_factors_dict = (
        _normalize_dict_keys(scaling_factors_raw) if isinstance(scaling_factors_raw, dict) else {}
    )

    # Get version (defaults to v0)
    version_raw = _get_liquidity_extra(liquidity, "version", "v0")
    version: Literal["v0", "v3Plus"] = "v3Plus" if version_raw == "v3Plus" else "v0"

    # Parse token reserves
    reserves: list[WeightedTokenReserve] = []
    for token_addr, token_data in liquidity.tokens.items():
        if not isinstance(token_data, dict):
            logger.debug(
                "weighted_pool_invalid_token_data",
                liquidity_id=liquidity.id,
                token=token_addr,
            )
            return None

        token_addr_lower = token_addr.lower()

        # Get weight: first try top-level dict, then try per-token data (Rust format)
        weight_raw: str | None = weights_dict.get(token_addr_lower)
        if weight_raw is None:
            # Try to get weight from token data (Rust auction format)
            weight_from_data = token_data.get("weight")
            weight_raw = str(weight_from_data) if weight_from_data is not None else None
        if weight_raw is None:
            logger.debug(
                "weighted_pool_missing_weight",
                liquidity_id=liquidity.id,
                token=token_addr,
            )
            return None

        try:
            weight = Decimal(str(weight_raw))
        except (InvalidOperation, TypeError):
            logger.warning(
                "weighted_pool_invalid_weight",
                liquidity_id=liquidity.id,
                token=token_addr,
                raw_weight=weight_raw,
            )
            return None

        # Validate weight is positive
        if weight <= 0:
            logger.warning(
                "weighted_pool_invalid_weight_value",
                liquidity_id=liquidity.id,
                token=token_addr,
                weight=str(weight),
            )
            return None

        # Parse and validate balance
        balance = _parse_balance(token_data, liquidity.id, token_addr, "weighted_pool")
        if balance is None:
            return None

        # Get scaling factor (supports both top-level dict and per-token data)
        scaling_factor = _parse_scaling_factor(
            scaling_factors_dict, token_addr, liquidity.id, "weighted_pool", token_data
        )

        reserves.append(
            WeightedTokenReserve(
                token=token_addr,
                balance=balance,
                weight=weight,
                scaling_factor=scaling_factor,
            )
        )

    if len(reserves) < 2:
        logger.debug(
            "weighted_pool_insufficient_tokens",
            liquidity_id=liquidity.id,
            token_count=len(reserves),
        )
        return None

    # Validate weight sum is approximately 1.0 (allow small tolerance for rounding)
    total_weight = sum(r.weight for r in reserves)
    if not (Decimal("0.99") <= total_weight <= Decimal("1.01")):
        logger.warning(
            "weighted_pool_invalid_weight_sum",
            liquidity_id=liquidity.id,
            total_weight=str(total_weight),
        )
        return None

    # Sort reserves by token address (required for Balancer)
    reserves.sort(key=lambda r: r.token.lower())

    # Parse fee (default 0.3%)
    fee = _parse_fee(liquidity.fee, "0.003", liquidity.id, "weighted_pool")

    # Get gas estimate
    gas_estimate = _parse_gas_estimate(liquidity.gas_estimate, 88892)

    return BalancerWeightedPool(
        id=liquidity.id,
        address=liquidity.address,
        pool_id=pool_id,
        reserves=tuple(reserves),
        fee=fee,
        version=version,
        gas_estimate=gas_estimate,
    )


def parse_stable_pool(liquidity: Liquidity) -> BalancerStablePool | None:
    """Parse stable liquidity into BalancerStablePool.

    For composable stable pools, automatically filters out the BPT token
    (token address == pool address) during parsing.

    Args:
        liquidity: Liquidity source from the auction

    Returns:
        BalancerStablePool if liquidity is a stable pool, None otherwise
    """
    if liquidity.kind != "stable":
        return None

    if liquidity.address is None:
        logger.debug("stable_pool_missing_address", liquidity_id=liquidity.id)
        return None

    # tokens must be a dict with balance info
    if not isinstance(liquidity.tokens, dict):
        logger.debug("stable_pool_invalid_tokens", liquidity_id=liquidity.id)
        return None

    # Get balancerPoolId from extra fields
    pool_id = _get_liquidity_extra(liquidity, "balancerPoolId")
    if pool_id is None:
        logger.debug("stable_pool_missing_pool_id", liquidity_id=liquidity.id)
        return None

    # Get amplification parameter (required for stable pools)
    amp_raw = _get_liquidity_extra(liquidity, "amplificationParameter")
    if amp_raw is None:
        logger.debug("stable_pool_missing_amp", liquidity_id=liquidity.id)
        return None

    try:
        amplification_parameter = Decimal(str(amp_raw))
    except (InvalidOperation, TypeError):
        logger.warning(
            "stable_pool_invalid_amp",
            liquidity_id=liquidity.id,
            raw_amp=amp_raw,
        )
        return None

    # Validate amplification parameter is positive
    if amplification_parameter <= 0:
        logger.warning(
            "stable_pool_invalid_amp_value",
            liquidity_id=liquidity.id,
            amp=str(amplification_parameter),
        )
        return None

    # Get scaling factors dict (normalized for case-insensitive lookup)
    scaling_factors_raw = _get_liquidity_extra(liquidity, "scalingFactors", {})
    scaling_factors_dict = (
        _normalize_dict_keys(scaling_factors_raw) if isinstance(scaling_factors_raw, dict) else {}
    )

    # Pool address (lowercase) for BPT filtering
    pool_address_lower = liquidity.address.lower()

    # Parse token reserves
    reserves: list[StableTokenReserve] = []
    for token_addr, token_data in liquidity.tokens.items():
        # Skip BPT token (pool's own token in composable stable pools)
        if token_addr.lower() == pool_address_lower:
            continue

        if not isinstance(token_data, dict):
            logger.debug(
                "stable_pool_invalid_token_data",
                liquidity_id=liquidity.id,
                token=token_addr,
            )
            return None

        # Parse and validate balance
        balance = _parse_balance(token_data, liquidity.id, token_addr, "stable_pool")
        if balance is None:
            return None

        # Get scaling factor (supports both top-level dict and per-token data)
        scaling_factor = _parse_scaling_factor(
            scaling_factors_dict, token_addr, liquidity.id, "stable_pool", token_data
        )

        reserves.append(
            StableTokenReserve(
                token=token_addr,
                balance=balance,
                scaling_factor=scaling_factor,
            )
        )

    if len(reserves) < 2:
        logger.debug(
            "stable_pool_insufficient_tokens",
            liquidity_id=liquidity.id,
            token_count=len(reserves),
        )
        return None

    # Sort reserves by token address (required for Balancer)
    reserves.sort(key=lambda r: r.token.lower())

    # Parse fee (default 0.01% for stable pools)
    fee = _parse_fee(liquidity.fee, "0.0001", liquidity.id, "stable_pool")

    # Get gas estimate
    gas_estimate = _parse_gas_estimate(liquidity.gas_estimate, 183520)

    return BalancerStablePool(
        id=liquidity.id,
        address=liquidity.address,
        pool_id=pool_id,
        reserves=tuple(reserves),
        amplification_parameter=amplification_parameter,
        fee=fee,
        gas_estimate=gas_estimate,
    )


# =============================================================================
# AMM Classes
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


# -----------------------------------------------------------------------------
# Partial Fill Binary Search Helpers
# -----------------------------------------------------------------------------


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
