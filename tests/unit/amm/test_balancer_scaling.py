"""Tests for Balancer scaling and fee helpers.

This module tests:
- Bfp (Balancer Fixed Point) basics
- Decimal scaling (scale_up, scale_down_down, scale_down_up)
- Fee application (subtract_swap_fee_amount, add_swap_fee_amount)
"""

from decimal import Decimal

import pytest

from solver.amm.balancer import (
    InvalidFeeError,
    InvalidScalingFactorError,
    add_swap_fee_amount,
    scale_down_down,
    scale_down_up,
    scale_up,
    subtract_swap_fee_amount,
)
from solver.math.fixed_point import ONE_18, Bfp


class TestBfpBasics:
    """Basic sanity tests for Bfp operations in weighted math context."""

    def test_bfp_mul_down(self) -> None:
        """Multiplication rounds down."""
        a = Bfp.from_wei(ONE_18 + 1)  # 1.000...001
        b = Bfp.from_wei(ONE_18 + 1)  # 1.000...001
        result = a.mul_down(b)
        # (10^18 + 1)^2 / 10^18 = 10^18 + 2 + 1/10^18
        # Floor division gives 10^18 + 2
        assert result.value == ONE_18 + 2

    def test_bfp_mul_up(self) -> None:
        """Multiplication rounds up."""
        a = Bfp.from_wei(ONE_18 + 1)
        b = Bfp.from_wei(ONE_18 + 1)
        result = a.mul_up(b)
        # Ceiling gives 10^18 + 3
        assert result.value == ONE_18 + 3

    def test_bfp_complement(self) -> None:
        """Complement calculates 1 - x."""
        x = Bfp.from_wei(3 * 10**17)  # 0.3
        comp = x.complement()
        assert comp.value == 7 * 10**17  # 0.7


class TestScaling:
    """Tests for decimal scaling helpers."""

    def test_scale_up_18_decimals(self) -> None:
        """Token with 18 decimals needs no scaling."""
        amount = 1_000_000_000_000_000_000  # 1 token
        scaling_factor = 1
        result = scale_up(amount, scaling_factor)
        assert result.value == amount

    def test_scale_up_6_decimals(self) -> None:
        """Token with 6 decimals (like USDC) needs 10^12 scaling."""
        amount = 1_000_000  # 1 USDC
        scaling_factor = 10**12
        result = scale_up(amount, scaling_factor)
        assert result.value == 1_000_000_000_000_000_000  # 1 * 10^18

    def test_scale_down_down(self) -> None:
        """Downscale rounds down."""
        bfp = Bfp.from_wei(1_500_000_000_000_000_000)  # 1.5 * 10^18
        scaling_factor = 10**12
        result = scale_down_down(bfp, scaling_factor)
        assert result == 1_500_000  # 1.5 * 10^6

    def test_scale_down_up(self) -> None:
        """Downscale rounds up."""
        bfp = Bfp.from_wei(1_500_000_000_000_000_001)  # Just over 1.5
        scaling_factor = 10**12
        result = scale_down_up(bfp, scaling_factor)
        assert result == 1_500_001  # Rounds up

    def test_scale_down_up_exact(self) -> None:
        """Downscale exact value stays same."""
        bfp = Bfp.from_wei(1_500_000_000_000_000_000)
        scaling_factor = 10**12
        result = scale_down_up(bfp, scaling_factor)
        assert result == 1_500_000

    def test_scale_up_zero_scaling_factor_raises(self) -> None:
        """Zero scaling factor raises error."""
        with pytest.raises(InvalidScalingFactorError):
            scale_up(1000, 0)

    def test_scale_up_negative_scaling_factor_raises(self) -> None:
        """Negative scaling factor raises error."""
        with pytest.raises(InvalidScalingFactorError):
            scale_up(1000, -1)

    def test_scale_down_down_zero_scaling_factor_raises(self) -> None:
        """Zero scaling factor raises error."""
        with pytest.raises(InvalidScalingFactorError):
            scale_down_down(Bfp.from_wei(1000), 0)

    def test_scale_down_up_zero_scaling_factor_raises(self) -> None:
        """Zero scaling factor raises error."""
        with pytest.raises(InvalidScalingFactorError):
            scale_down_up(Bfp.from_wei(1000), 0)


class TestFeeApplication:
    """Tests for fee subtraction and addition."""

    def test_subtract_fee_zero(self) -> None:
        """Zero fee returns original amount."""
        amount = 1_000_000_000_000_000_000
        result = subtract_swap_fee_amount(amount, Decimal("0"))
        assert result == amount

    def test_subtract_fee_0_3_percent(self) -> None:
        """0.3% fee subtraction."""
        amount = 1_000_000_000_000_000_000  # 1 * 10^18
        result = subtract_swap_fee_amount(amount, Decimal("0.003"))
        # 1 - (1 * 0.003) = 0.997
        # mul_up rounds the fee up, so result is slightly less
        expected = 997_000_000_000_000_000  # 0.997 * 10^18
        # Allow for rounding
        assert result <= expected
        assert result >= expected - 1

    def test_add_fee_zero(self) -> None:
        """Zero fee returns original amount."""
        amount = 1_000_000_000_000_000_000
        result = add_swap_fee_amount(amount, Decimal("0"))
        assert result == amount

    def test_add_fee_0_3_percent(self) -> None:
        """0.3% fee addition."""
        amount = 997_000_000_000_000_000  # 0.997 * 10^18
        result = add_swap_fee_amount(amount, Decimal("0.003"))
        # 0.997 / (1 - 0.003) = 0.997 / 0.997 = 1.0
        # div_up rounds up, so slightly more
        expected = 1_000_000_000_000_000_000  # 1 * 10^18
        assert result >= expected
        assert result <= expected + 2

    def test_subtract_fee_negative_raises(self) -> None:
        """Negative fee raises error."""
        with pytest.raises(InvalidFeeError):
            subtract_swap_fee_amount(1_000_000_000_000_000_000, Decimal("-0.01"))

    def test_subtract_fee_one_raises(self) -> None:
        """Fee of exactly 1.0 raises error."""
        with pytest.raises(InvalidFeeError):
            subtract_swap_fee_amount(1_000_000_000_000_000_000, Decimal("1.0"))

    def test_subtract_fee_greater_than_one_raises(self) -> None:
        """Fee > 1.0 raises error."""
        with pytest.raises(InvalidFeeError):
            subtract_swap_fee_amount(1_000_000_000_000_000_000, Decimal("1.5"))

    def test_add_fee_negative_raises(self) -> None:
        """Negative fee raises error."""
        with pytest.raises(InvalidFeeError):
            add_swap_fee_amount(1_000_000_000_000_000_000, Decimal("-0.01"))

    def test_add_fee_one_raises(self) -> None:
        """Fee of exactly 1.0 raises error (would cause division by zero)."""
        with pytest.raises(InvalidFeeError):
            add_swap_fee_amount(1_000_000_000_000_000_000, Decimal("1.0"))
