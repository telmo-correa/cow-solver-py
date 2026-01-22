"""Balancer weighted and stable pool implementations.

This module provides swap calculations for Balancer V2 pools, matching the
Rust baseline solver's implementation exactly.

Pool types supported:
- Weighted Product (V0 and V3Plus)
- Stable pools (to be added in Slice 3.2.3)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from solver.math.fixed_point import (
    MAX_IN_RATIO,
    MAX_OUT_RATIO,
    ONE_18,
    Bfp,
)

__all__ = [
    # Dataclasses
    "WeightedTokenReserve",
    "BalancerWeightedPool",
    # Math functions
    "calc_out_given_in",
    "calc_in_given_out",
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
    """
    return Bfp.from_wei(amount * scaling_factor)


def scale_down_down(bfp: Bfp, scaling_factor: int) -> int:
    """Scale 18-decimal result back to token decimals, rounding down.

    Args:
        bfp: Amount in 18-decimal fixed-point
        scaling_factor: Factor to scale by

    Returns:
        Amount in token's native decimals
    """
    return bfp.value // scaling_factor


def scale_down_up(bfp: Bfp, scaling_factor: int) -> int:
    """Scale 18-decimal result back to token decimals, rounding up.

    Args:
        bfp: Amount in 18-decimal fixed-point
        scaling_factor: Factor to scale by

    Returns:
        Amount in token's native decimals
    """
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
        swap_fee: Fee as decimal (e.g., 0.003 for 0.3%)

    Returns:
        Amount after fee deduction
    """
    amount_bfp = Bfp.from_wei(amount)
    fee_bfp = Bfp.from_decimal(swap_fee)
    fee_amount = amount_bfp.mul_up(fee_bfp)
    amount_without_fees = amount_bfp.sub(fee_amount)
    return amount_without_fees.value


def add_swap_fee_amount(amount: int, swap_fee: Decimal) -> int:
    """Add swap fee to output amount.

    Used for buy orders (exact output): fee is added after calculating
    the required input.

    Formula: amount_with_fee = amount / (1 - fee)

    Args:
        amount: Input amount before fee
        swap_fee: Fee as decimal (e.g., 0.003 for 0.3%)

    Returns:
        Amount including fee
    """
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
        balance_in: Scaled balance of input token
        weight_in: Weight of input token
        balance_out: Scaled balance of output token
        weight_out: Weight of output token
        amount_in: Scaled input amount (after fee subtraction)
        _version: Pool version for power function variant (V3Plus optimization TODO)

    Returns:
        Scaled output amount

    Raises:
        MaxInRatioError: If amount_in > balance_in * 0.3 (30% limit)
    """
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
        balance_in: Scaled balance of input token
        weight_in: Weight of input token
        balance_out: Scaled balance of output token
        weight_out: Weight of output token
        amount_out: Scaled output amount
        _version: Pool version for power function variant (V3Plus optimization TODO)

    Returns:
        Scaled input amount (before fee addition)

    Raises:
        MaxOutRatioError: If amount_out > balance_out * 0.3 (30% limit)
    """
    # Check ratio limit: amount_out must not exceed 30% of balance_out
    max_amount_out = balance_out.mul_down(MAX_OUT_RATIO)
    if amount_out.value > max_amount_out.value:
        raise MaxOutRatioError(
            f"Output {amount_out.value} exceeds 30% of balance {balance_out.value}"
        )

    # denominator = balance_out - amount_out
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
