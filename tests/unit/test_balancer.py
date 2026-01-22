"""Tests for Balancer weighted pool math.

Test vectors are taken from the Rust baseline solver:
cow-services/crates/shared/src/sources/balancer_v2/swap/weighted_math.rs
"""

from decimal import Decimal

import pytest

from solver.amm.balancer import (
    BalancerWeightedPool,
    MaxInRatioError,
    MaxOutRatioError,
    WeightedTokenReserve,
    add_swap_fee_amount,
    calc_in_given_out,
    calc_out_given_in,
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


class TestCalcOutGivenIn:
    """Tests for calc_out_given_in (sell order math)."""

    def test_basic_50_50_pool(self) -> None:
        """Simple 50/50 pool swap."""
        # Pool: 100 TOKEN_A / 100 TOKEN_B, weights 0.5 / 0.5
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)  # 0.5
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)  # 0.5
        amount_in = Bfp.from_wei(10 * ONE_18)  # 10 tokens

        result = calc_out_given_in(balance_in, weight_in, balance_out, weight_out, amount_in)

        # For 50/50 pool with equal weights, this is like constant product
        # x * y = k, (100 + 10) * (100 - out) = 100 * 100
        # out = 100 - 10000/110 = 100 - 90.909... = 9.09...
        # Due to power function approximation, result will be close
        assert 9 * ONE_18 < result.value < 10 * ONE_18

    def test_rust_test_vector_1(self) -> None:
        """Test vector from Rust weighted_math.rs line 146-168."""
        # Input values from Rust test
        balance_in = Bfp.from_wei(100_000_000_000_000_000_000_000)  # 100k * 10^18
        weight_in = Bfp.from_wei(300_000_000_000_000)  # 0.0003 * 10^18
        balance_out = Bfp.from_wei(10_000_000_000_000_000_000)  # 10 * 10^18
        weight_out = Bfp.from_wei(700_000_000_000_000)  # 0.0007 * 10^18
        amount_in = Bfp.from_wei(10_000_000_000_000_000)  # 0.01 * 10^18

        result = calc_out_given_in(balance_in, weight_in, balance_out, weight_out, amount_in)

        # Expected output from Rust: 428_571_297_950
        # Allow some tolerance due to power function precision
        expected = 428_571_297_950
        tolerance = expected // 1000  # 0.1% tolerance
        assert abs(result.value - expected) <= tolerance, (
            f"Expected ~{expected}, got {result.value}"
        )

    def test_max_in_ratio_exactly_30_percent(self) -> None:
        """30% of balance is the maximum allowed input."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        # 30% of 100 = 30, but mul_down rounds down
        # MAX_IN_RATIO = 0.3, so max_amount_in = 100 * 0.3 = 30 (with rounding)
        amount_in = Bfp.from_wei(30 * ONE_18)

        # Should work (exactly at limit, allowing for rounding)
        result = calc_out_given_in(balance_in, weight_in, balance_out, weight_out, amount_in)
        assert result.value > 0

    def test_max_in_ratio_exceeded(self) -> None:
        """Input exceeding 30% of balance raises error."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_in = Bfp.from_wei(31 * ONE_18)  # 31% > 30%

        with pytest.raises(MaxInRatioError):
            calc_out_given_in(balance_in, weight_in, balance_out, weight_out, amount_in)


class TestCalcInGivenOut:
    """Tests for calc_in_given_out (buy order math)."""

    def test_basic_50_50_pool(self) -> None:
        """Simple 50/50 pool buy order."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_out = Bfp.from_wei(10 * ONE_18)  # Want 10 tokens

        result = calc_in_given_out(balance_in, weight_in, balance_out, weight_out, amount_out)

        # For 50/50 pool, this is like constant product
        # To get 10 out of 100, need to put in: 100 * (100/90 - 1) = 100 * 10/90 = 11.11...
        assert 11 * ONE_18 < result.value < 12 * ONE_18

    def test_rust_test_vector_2(self) -> None:
        """Test vector from Rust weighted_math.rs line 170-192."""
        # Input values from Rust test
        balance_in = Bfp.from_wei(100_000_000_000_000_000_000_000)  # 100k * 10^18
        weight_in = Bfp.from_wei(300_000_000_000_000)  # 0.0003 * 10^18
        balance_out = Bfp.from_wei(10_000_000_000_000_000_000)  # 10 * 10^18
        weight_out = Bfp.from_wei(700_000_000_000_000)  # 0.0007 * 10^18
        amount_out = Bfp.from_wei(10_000_000_000_000_000)  # 0.01 * 10^18

        result = calc_in_given_out(balance_in, weight_in, balance_out, weight_out, amount_out)

        # Expected input from Rust: 233_722_784_701_541_000_000
        expected = 233_722_784_701_541_000_000
        tolerance = expected // 1000  # 0.1% tolerance
        assert abs(result.value - expected) <= tolerance, (
            f"Expected ~{expected}, got {result.value}"
        )

    def test_max_out_ratio_exactly_30_percent(self) -> None:
        """30% of balance is the maximum allowed output."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_out = Bfp.from_wei(30 * ONE_18)  # 30%

        # Should work (exactly at limit)
        result = calc_in_given_out(balance_in, weight_in, balance_out, weight_out, amount_out)
        assert result.value > 0

    def test_max_out_ratio_exceeded(self) -> None:
        """Output exceeding 30% of balance raises error."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_out = Bfp.from_wei(31 * ONE_18)  # 31% > 30%

        with pytest.raises(MaxOutRatioError):
            calc_in_given_out(balance_in, weight_in, balance_out, weight_out, amount_out)


class TestWeightedPoolDataclass:
    """Tests for BalancerWeightedPool dataclass."""

    def test_get_reserve_found(self) -> None:
        """Get reserve for existing token."""
        pool = BalancerWeightedPool(
            id="test-pool",
            address="0x1234",
            pool_id="0xabcd",
            reserves=(
                WeightedTokenReserve(
                    token="0xAAAA",
                    balance=1000,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
                WeightedTokenReserve(
                    token="0xBBBB",
                    balance=2000,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=100000,
        )

        # Case-insensitive lookup
        reserve = pool.get_reserve("0xaaaa")
        assert reserve is not None
        assert reserve.balance == 1000

    def test_get_reserve_not_found(self) -> None:
        """Get reserve for non-existing token returns None."""
        pool = BalancerWeightedPool(
            id="test-pool",
            address="0x1234",
            pool_id="0xabcd",
            reserves=(
                WeightedTokenReserve(
                    token="0xAAAA",
                    balance=1000,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=100000,
        )

        reserve = pool.get_reserve("0xCCCC")
        assert reserve is None


class TestWeightedPoolIntegration:
    """Integration tests combining all components."""

    def test_sell_order_full_flow(self) -> None:
        """Full flow: scale, subtract fee, calculate, scale back."""
        # Pool with 100 TOKEN_A (18 dec) / 100 TOKEN_B (18 dec), 0.3% fee
        balance_in_raw = 100 * ONE_18
        balance_out_raw = 100 * ONE_18
        scaling_factor = 1  # 18-decimal tokens
        weight = Bfp.from_wei(ONE_18 // 2)  # 0.5
        fee = Decimal("0.003")

        # Sell 10 TOKEN_A
        amount_in_raw = 10 * ONE_18

        # Step 1: Subtract fee
        amount_in_after_fee = subtract_swap_fee_amount(amount_in_raw, fee)

        # Step 2: Scale up (no-op for 18 decimal)
        balance_in = scale_up(balance_in_raw, scaling_factor)
        balance_out = scale_up(balance_out_raw, scaling_factor)
        amount_in = scale_up(amount_in_after_fee, scaling_factor)

        # Step 3: Calculate
        amount_out_scaled = calc_out_given_in(balance_in, weight, balance_out, weight, amount_in)

        # Step 4: Scale down
        amount_out = scale_down_down(amount_out_scaled, scaling_factor)

        # Should get slightly less than ~9.09 due to fee
        assert 8 * ONE_18 < amount_out < 10 * ONE_18

    def test_buy_order_full_flow(self) -> None:
        """Full flow for buy order."""
        # Pool with 100 TOKEN_A (18 dec) / 100 TOKEN_B (18 dec), 0.3% fee
        balance_in_raw = 100 * ONE_18
        balance_out_raw = 100 * ONE_18
        scaling_factor = 1
        weight = Bfp.from_wei(ONE_18 // 2)
        fee = Decimal("0.003")

        # Want 10 TOKEN_B
        amount_out_raw = 10 * ONE_18

        # Step 1: Scale up
        balance_in = scale_up(balance_in_raw, scaling_factor)
        balance_out = scale_up(balance_out_raw, scaling_factor)
        amount_out = scale_up(amount_out_raw, scaling_factor)

        # Step 2: Calculate
        amount_in_scaled = calc_in_given_out(balance_in, weight, balance_out, weight, amount_out)

        # Step 3: Scale down (round up for input)
        amount_in_before_fee = scale_down_up(amount_in_scaled, scaling_factor)

        # Step 4: Add fee
        amount_in = add_swap_fee_amount(amount_in_before_fee, fee)

        # Should need slightly more than ~11.11 due to fee
        assert 11 * ONE_18 < amount_in < 13 * ONE_18
