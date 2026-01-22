"""Balancer weighted and stable pool implementations.

This module provides swap calculations for Balancer V2 pools, matching the
Rust baseline solver's implementation exactly.

Pool types supported:
- Weighted Product (V0 and V3Plus)
- Stable pools (StableSwap / Curve-style)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from solver.math.fixed_point import (
    AMP_PRECISION,
    MAX_IN_RATIO,
    MAX_OUT_RATIO,
    ONE_18,
    Bfp,
)

__all__ = [
    # Weighted pool dataclasses
    "WeightedTokenReserve",
    "BalancerWeightedPool",
    # Stable pool dataclasses
    "StableTokenReserve",
    "BalancerStablePool",
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
        amplification_parameter: A parameter (scaled by AMP_PRECISION=1000)
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

    def get_token_index(self, token: str) -> int | None:
        """Get index of token in reserves list."""
        token_lower = token.lower()
        for i, reserve in enumerate(self.reserves):
            if reserve.token.lower() == token_lower:
                return i
        return None


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
    # V0 and V3Plus use the same pow_up (V3Plus optimization not implemented yet)
    power = base.pow_up(exponent)

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
    power = base.pow_up(exponent)

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
