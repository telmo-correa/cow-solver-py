"""Balancer weighted pool math.

Core math functions for weighted product pools.
"""

from typing import Literal

from solver.math.fixed_point import MAX_IN_RATIO, MAX_OUT_RATIO, ONE_18, Bfp

from .errors import MaxInRatioError, MaxOutRatioError, ZeroBalanceError, ZeroWeightError


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
