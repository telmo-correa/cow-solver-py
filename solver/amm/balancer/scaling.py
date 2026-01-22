"""Balancer scaling and fee helpers.

Functions for scaling token amounts between native decimals and 18-decimal
fixed-point, and for applying swap fees.
"""

from decimal import Decimal

from solver.math.fixed_point import Bfp

from .errors import InvalidFeeError, InvalidScalingFactorError


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
