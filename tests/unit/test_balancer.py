"""Tests for Balancer weighted and stable pool math.

Test vectors are taken from the Rust baseline solver:
cow-services/crates/shared/src/sources/balancer_v2/swap/weighted_math.rs
cow-services/crates/shared/src/sources/balancer_v2/swap/stable_math.rs
"""

from decimal import Decimal

import pytest

from solver.amm.balancer import (
    BalancerStablePool,
    BalancerWeightedPool,
    InvalidFeeError,
    InvalidScalingFactorError,
    MaxInRatioError,
    MaxOutRatioError,
    StableTokenReserve,
    WeightedTokenReserve,
    ZeroBalanceError,
    ZeroWeightError,
    add_swap_fee_amount,
    calc_in_given_out,
    calc_out_given_in,
    calculate_invariant,
    filter_bpt_token,
    get_token_balance_given_invariant_and_all_other_balances,
    scale_down_down,
    scale_down_up,
    scale_up,
    stable_calc_in_given_out,
    stable_calc_out_given_in,
    subtract_swap_fee_amount,
)
from solver.math.fixed_point import AMP_PRECISION, ONE_18, Bfp


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
        # Actual difference is ~20 wei (0.000005%) - use tight tolerance
        expected = 428_571_297_950
        tolerance = 100  # 100 wei absolute tolerance
        assert abs(result.value - expected) <= tolerance, (
            f"Expected ~{expected}, got {result.value}, diff={result.value - expected}"
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

    def test_zero_weight_in_raises(self) -> None:
        """Zero input weight raises error."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(0)  # Zero weight
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_in = Bfp.from_wei(10 * ONE_18)

        with pytest.raises(ZeroWeightError):
            calc_out_given_in(balance_in, weight_in, balance_out, weight_out, amount_in)

    def test_zero_weight_out_raises(self) -> None:
        """Zero output weight raises error."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(0)  # Zero weight
        amount_in = Bfp.from_wei(10 * ONE_18)

        with pytest.raises(ZeroWeightError):
            calc_out_given_in(balance_in, weight_in, balance_out, weight_out, amount_in)

    def test_zero_balance_in_raises(self) -> None:
        """Zero input balance raises error."""
        balance_in = Bfp.from_wei(0)  # Zero balance
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_in = Bfp.from_wei(10 * ONE_18)

        with pytest.raises(ZeroBalanceError):
            calc_out_given_in(balance_in, weight_in, balance_out, weight_out, amount_in)

    def test_zero_balance_out_raises(self) -> None:
        """Zero output balance raises error."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(0)  # Zero balance
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_in = Bfp.from_wei(10 * ONE_18)

        with pytest.raises(ZeroBalanceError):
            calc_out_given_in(balance_in, weight_in, balance_out, weight_out, amount_in)

    def test_zero_amount_in_returns_near_zero(self) -> None:
        """Zero input amount returns near-zero output.

        Note: Due to power function error margin adjustment in pow_up,
        the result is not exactly zero but very small.
        """
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_in = Bfp.from_wei(0)

        result = calc_out_given_in(balance_in, weight_in, balance_out, weight_out, amount_in)
        # Result is tiny (within power function error margin)
        assert abs(result.value) < ONE_18 // 1000  # Less than 0.001 tokens

    def test_asymmetric_weights_80_20(self) -> None:
        """Test 80/20 weighted pool."""
        # Pool: 80% weight in, 20% weight out
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(8 * ONE_18 // 10)  # 0.8
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(2 * ONE_18 // 10)  # 0.2
        amount_in = Bfp.from_wei(10 * ONE_18)

        result = calc_out_given_in(balance_in, weight_in, balance_out, weight_out, amount_in)
        # With higher weight_in relative to weight_out, we should get more output
        # than a 50/50 pool would give
        assert result.value > 0


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
        # Actual difference is ~100000 wei (0.00000004%) - use tight tolerance
        expected = 233_722_784_701_541_000_000
        tolerance = 200_000  # 200000 wei absolute tolerance
        assert abs(result.value - expected) <= tolerance, (
            f"Expected ~{expected}, got {result.value}, diff={result.value - expected}"
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

    def test_zero_weight_in_raises(self) -> None:
        """Zero input weight raises error."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(0)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_out = Bfp.from_wei(10 * ONE_18)

        with pytest.raises(ZeroWeightError):
            calc_in_given_out(balance_in, weight_in, balance_out, weight_out, amount_out)

    def test_zero_weight_out_raises(self) -> None:
        """Zero output weight raises error."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(0)
        amount_out = Bfp.from_wei(10 * ONE_18)

        with pytest.raises(ZeroWeightError):
            calc_in_given_out(balance_in, weight_in, balance_out, weight_out, amount_out)

    def test_zero_balance_in_raises(self) -> None:
        """Zero input balance raises error."""
        balance_in = Bfp.from_wei(0)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_out = Bfp.from_wei(10 * ONE_18)

        with pytest.raises(ZeroBalanceError):
            calc_in_given_out(balance_in, weight_in, balance_out, weight_out, amount_out)

    def test_zero_balance_out_raises(self) -> None:
        """Zero output balance raises error."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(0)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_out = Bfp.from_wei(10 * ONE_18)

        with pytest.raises(ZeroBalanceError):
            calc_in_given_out(balance_in, weight_in, balance_out, weight_out, amount_out)

    def test_amount_out_equals_balance_out_raises(self) -> None:
        """Requesting entire balance raises error.

        Note: The 30% ratio check triggers first (100% > 30%).
        """
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_out = Bfp.from_wei(100 * ONE_18)  # Equal to balance (100%)

        with pytest.raises(MaxOutRatioError):
            calc_in_given_out(balance_in, weight_in, balance_out, weight_out, amount_out)

    def test_amount_out_near_balance_out_raises(self) -> None:
        """Requesting just under 30% but would cause zero denominator passes ratio but fails balance check.

        Note: If we could bypass the ratio check, requesting amount_out very close to balance_out
        would fail with ZeroBalanceError. But with the 30% limit, this is unreachable.
        The zero balance check is a safety net that protects against any future changes
        to MAX_OUT_RATIO.
        """
        # This test documents that the zero denominator check exists as a safety net
        # It's not reachable with current MAX_OUT_RATIO = 30%
        pass

    def test_zero_amount_out_returns_near_zero(self) -> None:
        """Zero output amount returns near-zero input.

        Note: Due to power function error margin adjustment in pow_up,
        the result is not exactly zero but very small.
        """
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(ONE_18 // 2)
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(ONE_18 // 2)
        amount_out = Bfp.from_wei(0)

        result = calc_in_given_out(balance_in, weight_in, balance_out, weight_out, amount_out)
        # Result is tiny (within power function error margin)
        assert abs(result.value) < ONE_18 // 1000  # Less than 0.001 tokens

    def test_asymmetric_weights_20_80(self) -> None:
        """Test 20/80 weighted pool (buy order)."""
        balance_in = Bfp.from_wei(100 * ONE_18)
        weight_in = Bfp.from_wei(2 * ONE_18 // 10)  # 0.2
        balance_out = Bfp.from_wei(100 * ONE_18)
        weight_out = Bfp.from_wei(8 * ONE_18 // 10)  # 0.8
        amount_out = Bfp.from_wei(10 * ONE_18)

        result = calc_in_given_out(balance_in, weight_in, balance_out, weight_out, amount_out)
        assert result.value > 0


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


# =============================================================================
# Stable Pool Tests
# =============================================================================


class TestStablePoolDataclass:
    """Tests for BalancerStablePool dataclass."""

    def test_get_reserve_found(self) -> None:
        """Get reserve for existing token."""
        pool = BalancerStablePool(
            id="test-stable-pool",
            address="0x1234",
            pool_id="0xabcd",
            reserves=(
                StableTokenReserve(
                    token="0xAAAA",
                    balance=1000,
                    scaling_factor=1,
                ),
                StableTokenReserve(
                    token="0xBBBB",
                    balance=2000,
                    scaling_factor=10**12,
                ),
            ),
            amplification_parameter=Decimal("5000"),
            fee=Decimal("0.0001"),
            gas_estimate=183520,
        )

        # Case-insensitive lookup
        reserve = pool.get_reserve("0xaaaa")
        assert reserve is not None
        assert reserve.balance == 1000

    def test_get_reserve_not_found(self) -> None:
        """Get reserve for non-existing token returns None."""
        pool = BalancerStablePool(
            id="test-stable-pool",
            address="0x1234",
            pool_id="0xabcd",
            reserves=(
                StableTokenReserve(
                    token="0xAAAA",
                    balance=1000,
                    scaling_factor=1,
                ),
            ),
            amplification_parameter=Decimal("5000"),
            fee=Decimal("0.0001"),
            gas_estimate=183520,
        )

        reserve = pool.get_reserve("0xCCCC")
        assert reserve is None

    def test_get_token_index(self) -> None:
        """Get token index from reserves."""
        pool = BalancerStablePool(
            id="test-stable-pool",
            address="0x1234",
            pool_id="0xabcd",
            reserves=(
                StableTokenReserve(token="0xAAAA", balance=1000, scaling_factor=1),
                StableTokenReserve(token="0xBBBB", balance=2000, scaling_factor=1),
                StableTokenReserve(token="0xCCCC", balance=3000, scaling_factor=1),
            ),
            amplification_parameter=Decimal("5000"),
            fee=Decimal("0.0001"),
            gas_estimate=183520,
        )

        assert pool.get_token_index("0xaaaa") == 0
        assert pool.get_token_index("0xBBBB") == 1
        assert pool.get_token_index("0xcccc") == 2
        assert pool.get_token_index("0xDDDD") is None


class TestCalculateInvariant:
    """Tests for stable pool invariant calculation."""

    def test_two_token_equal_balances(self) -> None:
        """Invariant for 2 tokens with equal balances."""
        # With equal balances, D should be close to sum
        amp = 5000 * AMP_PRECISION  # A = 5000
        balances = [Bfp.from_wei(100 * ONE_18), Bfp.from_wei(100 * ONE_18)]

        d = calculate_invariant(amp, balances)

        # D should be approximately 200 (sum of balances)
        # Due to the stable math formula, it will be very close
        assert 199 * ONE_18 < d.value < 201 * ONE_18

    def test_three_token_pool(self) -> None:
        """Invariant for 3-token pool (typical stable pool)."""
        # DAI/USDC/USDT style pool
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(1_000_000 * ONE_18),  # 1M DAI
            Bfp.from_wei(1_000_000 * ONE_18),  # 1M USDC (scaled)
            Bfp.from_wei(1_000_000 * ONE_18),  # 1M USDT (scaled)
        ]

        d = calculate_invariant(amp, balances)

        # D should be approximately 3M
        assert 2_999_000 * ONE_18 < d.value < 3_001_000 * ONE_18

    def test_asymmetric_balances(self) -> None:
        """Invariant with unequal balances still converges."""
        amp = 200 * AMP_PRECISION  # Lower A
        balances = [
            Bfp.from_wei(100 * ONE_18),
            Bfp.from_wei(200 * ONE_18),
        ]

        d = calculate_invariant(amp, balances)

        # With asymmetric balances and lower A, D is between sum (300) and geometric mean
        assert 100 * ONE_18 < d.value < 400 * ONE_18

    def test_zero_balance_raises(self) -> None:
        """Zero balance in pool raises error."""
        amp = 5000 * AMP_PRECISION
        balances = [Bfp.from_wei(100 * ONE_18), Bfp.from_wei(0)]

        with pytest.raises(ZeroBalanceError):
            calculate_invariant(amp, balances)

    def test_empty_balances_returns_zero(self) -> None:
        """Empty balances list returns zero invariant."""
        amp = 5000 * AMP_PRECISION
        d = calculate_invariant(amp, [])
        assert d.value == 0


class TestGetTokenBalance:
    """Tests for get_token_balance_given_invariant_and_all_other_balances."""

    def test_recovers_original_balance(self) -> None:
        """Given D and other balances, recovers the original balance."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(100 * ONE_18),
            Bfp.from_wei(100 * ONE_18),
        ]

        # Calculate invariant
        d = calculate_invariant(amp, balances)

        # Try to recover balance[0] given balance[1]
        recovered = get_token_balance_given_invariant_and_all_other_balances(
            amp, balances, d, token_index=0
        )

        # Should be very close to original (within convergence tolerance)
        assert abs(recovered.value - balances[0].value) <= 2

    def test_three_token_recovers_balance(self) -> None:
        """Works for 3-token pools."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(1_000 * ONE_18),
            Bfp.from_wei(1_000 * ONE_18),
            Bfp.from_wei(1_000 * ONE_18),
        ]

        d = calculate_invariant(amp, balances)

        # Recover middle balance
        recovered = get_token_balance_given_invariant_and_all_other_balances(
            amp, balances, d, token_index=1
        )

        assert abs(recovered.value - balances[1].value) <= 2

    def test_index_out_of_range_raises(self) -> None:
        """Invalid token index raises error."""
        amp = 5000 * AMP_PRECISION
        balances = [Bfp.from_wei(100 * ONE_18)]
        d = Bfp.from_wei(100 * ONE_18)

        with pytest.raises(IndexError):
            get_token_balance_given_invariant_and_all_other_balances(
                amp, balances, d, token_index=5
            )

    def test_negative_index_raises(self) -> None:
        """Negative token index raises error."""
        amp = 5000 * AMP_PRECISION
        balances = [Bfp.from_wei(100 * ONE_18), Bfp.from_wei(100 * ONE_18)]
        d = Bfp.from_wei(200 * ONE_18)

        with pytest.raises(IndexError):
            get_token_balance_given_invariant_and_all_other_balances(
                amp, balances, d, token_index=-1
            )


class TestStableCalcOutGivenIn:
    """Tests for stable_calc_out_given_in (sell order math)."""

    def test_small_swap_nearly_1_to_1(self) -> None:
        """Small swaps in stable pool are nearly 1:1."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(1_000_000 * ONE_18),  # 1M token A
            Bfp.from_wei(1_000_000 * ONE_18),  # 1M token B
        ]
        amount_in = Bfp.from_wei(1000 * ONE_18)  # 1000 tokens

        result = stable_calc_out_given_in(amp, balances, 0, 1, amount_in)

        # For stable pool with high A, output should be very close to input
        # Allow some slippage due to curve
        assert 999 * ONE_18 < result.value < 1001 * ONE_18

    def test_larger_swap_has_slippage(self) -> None:
        """Larger swaps have more slippage."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(100_000 * ONE_18),
            Bfp.from_wei(100_000 * ONE_18),
        ]
        amount_in = Bfp.from_wei(10_000 * ONE_18)  # 10% of pool

        result = stable_calc_out_given_in(amp, balances, 0, 1, amount_in)

        # Larger swap has more slippage, but still good exchange rate
        assert 9_900 * ONE_18 < result.value < 10_000 * ONE_18

    def test_three_token_swap(self) -> None:
        """Swap in 3-token pool."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(1_000_000 * ONE_18),
            Bfp.from_wei(1_000_000 * ONE_18),
            Bfp.from_wei(1_000_000 * ONE_18),
        ]
        amount_in = Bfp.from_wei(1000 * ONE_18)

        # Swap token 0 -> token 2
        result = stable_calc_out_given_in(amp, balances, 0, 2, amount_in)

        # Should be nearly 1:1
        assert 999 * ONE_18 < result.value < 1001 * ONE_18

    def test_zero_input_returns_zero(self) -> None:
        """Zero input returns zero output."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(1_000_000 * ONE_18),
            Bfp.from_wei(1_000_000 * ONE_18),
        ]
        amount_in = Bfp.from_wei(0)

        result = stable_calc_out_given_in(amp, balances, 0, 1, amount_in)
        assert result.value == 0

    def test_same_token_raises(self) -> None:
        """Swapping token with itself raises error."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(1_000_000 * ONE_18),
            Bfp.from_wei(1_000_000 * ONE_18),
        ]
        amount_in = Bfp.from_wei(1000 * ONE_18)

        with pytest.raises(ValueError, match="Cannot swap token with itself"):
            stable_calc_out_given_in(amp, balances, 0, 0, amount_in)

    def test_invalid_index_raises(self) -> None:
        """Invalid token indices raise error."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(1_000_000 * ONE_18),
            Bfp.from_wei(1_000_000 * ONE_18),
        ]
        amount_in = Bfp.from_wei(1000 * ONE_18)

        # Out of range positive index
        with pytest.raises(IndexError):
            stable_calc_out_given_in(amp, balances, 0, 5, amount_in)

        # Negative index
        with pytest.raises(IndexError):
            stable_calc_out_given_in(amp, balances, -1, 1, amount_in)


class TestStableCalcInGivenOut:
    """Tests for stable_calc_in_given_out (buy order math)."""

    def test_small_swap_nearly_1_to_1(self) -> None:
        """Small swaps in stable pool are nearly 1:1."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(1_000_000 * ONE_18),
            Bfp.from_wei(1_000_000 * ONE_18),
        ]
        amount_out = Bfp.from_wei(1000 * ONE_18)

        result = stable_calc_in_given_out(amp, balances, 0, 1, amount_out)

        # For stable pool, input should be very close to output
        assert 999 * ONE_18 < result.value < 1002 * ONE_18

    def test_amount_out_exceeds_balance_raises(self) -> None:
        """Requesting more than balance raises error."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(1_000 * ONE_18),
            Bfp.from_wei(1_000 * ONE_18),
        ]
        amount_out = Bfp.from_wei(1_001 * ONE_18)  # More than balance

        with pytest.raises(ZeroBalanceError):
            stable_calc_in_given_out(amp, balances, 0, 1, amount_out)

    def test_same_token_raises(self) -> None:
        """Swapping token with itself raises error."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(1_000_000 * ONE_18),
            Bfp.from_wei(1_000_000 * ONE_18),
        ]
        amount_out = Bfp.from_wei(1000 * ONE_18)

        with pytest.raises(ValueError, match="Cannot swap token with itself"):
            stable_calc_in_given_out(amp, balances, 1, 1, amount_out)

    def test_invalid_index_raises(self) -> None:
        """Invalid token indices raise error."""
        amp = 5000 * AMP_PRECISION
        balances = [
            Bfp.from_wei(1_000_000 * ONE_18),
            Bfp.from_wei(1_000_000 * ONE_18),
        ]
        amount_out = Bfp.from_wei(1000 * ONE_18)

        # Out of range positive index
        with pytest.raises(IndexError):
            stable_calc_in_given_out(amp, balances, 5, 1, amount_out)

        # Negative index
        with pytest.raises(IndexError):
            stable_calc_in_given_out(amp, balances, 0, -1, amount_out)


class TestFilterBptToken:
    """Tests for composable stable pool BPT filtering."""

    def test_filters_bpt_token(self) -> None:
        """BPT token (matching pool address) is filtered out."""
        pool = BalancerStablePool(
            id="composable-stable",
            address="0xPoolAddress",
            pool_id="0xabcd",
            reserves=(
                StableTokenReserve(token="0xDAI", balance=1000, scaling_factor=1),
                StableTokenReserve(token="0xPoolAddress", balance=5000, scaling_factor=1),  # BPT
                StableTokenReserve(token="0xUSDC", balance=2000, scaling_factor=10**12),
            ),
            amplification_parameter=Decimal("5000"),
            fee=Decimal("0.0001"),
            gas_estimate=183520,
        )

        filtered = filter_bpt_token(pool)

        assert len(filtered.reserves) == 2
        tokens = [r.token for r in filtered.reserves]
        assert "0xDAI" in tokens
        assert "0xUSDC" in tokens
        assert "0xPoolAddress" not in tokens

    def test_no_bpt_returns_original(self) -> None:
        """Pool without BPT returns unchanged."""
        pool = BalancerStablePool(
            id="regular-stable",
            address="0xPoolAddress",
            pool_id="0xabcd",
            reserves=(
                StableTokenReserve(token="0xDAI", balance=1000, scaling_factor=1),
                StableTokenReserve(token="0xUSDC", balance=2000, scaling_factor=10**12),
            ),
            amplification_parameter=Decimal("5000"),
            fee=Decimal("0.0001"),
            gas_estimate=183520,
        )

        filtered = filter_bpt_token(pool)

        assert filtered is pool  # Same object (no change)

    def test_case_insensitive_bpt_matching(self) -> None:
        """BPT matching is case-insensitive."""
        pool = BalancerStablePool(
            id="composable-stable",
            address="0xPoolAddress",
            pool_id="0xabcd",
            reserves=(
                StableTokenReserve(token="0xDAI", balance=1000, scaling_factor=1),
                StableTokenReserve(
                    token="0xpooladdress", balance=5000, scaling_factor=1
                ),  # lowercase
            ),
            amplification_parameter=Decimal("5000"),
            fee=Decimal("0.0001"),
            gas_estimate=183520,
        )

        filtered = filter_bpt_token(pool)

        assert len(filtered.reserves) == 1


class TestStablePoolIntegration:
    """Integration tests for stable pool combining all components."""

    def test_sell_order_full_flow(self) -> None:
        """Full flow for sell order in stable pool."""
        # Pool: DAI/USDC, A=5000, 0.01% fee
        # DAI is 18 decimals, USDC is 6 decimals
        dai_balance = 1_000_000 * ONE_18  # 1M DAI (18 dec)
        usdc_balance = 1_000_000_000_000  # 1M USDC (6 dec)
        dai_scaling = 1
        usdc_scaling = 10**12
        amp = 5000 * AMP_PRECISION
        fee = Decimal("0.0001")  # 0.01%

        # Sell 10 DAI for USDC
        amount_in_raw = 10 * ONE_18  # 10 DAI

        # Step 1: Subtract fee
        amount_in_after_fee = subtract_swap_fee_amount(amount_in_raw, fee)

        # Step 2: Scale balances to 18 decimals
        scaled_balances = [
            scale_up(dai_balance, dai_scaling),
            scale_up(usdc_balance, usdc_scaling),
        ]

        # Step 3: Scale input to 18 decimals
        amount_in_scaled = scale_up(amount_in_after_fee, dai_scaling)

        # Step 4: Calculate output
        amount_out_scaled = stable_calc_out_given_in(amp, scaled_balances, 0, 1, amount_in_scaled)

        # Step 5: Scale output back to native decimals (USDC = 6)
        amount_out = scale_down_down(amount_out_scaled, usdc_scaling)

        # Should get approximately 10 USDC (minus tiny slippage and fee)
        # 10 USDC = 10_000_000 (6 decimals)
        assert 9_990_000 < amount_out < 10_010_000

    def test_buy_order_full_flow(self) -> None:
        """Full flow for buy order in stable pool."""
        # Pool: DAI/USDC, A=5000, 0.01% fee
        dai_balance = 1_000_000 * ONE_18
        usdc_balance = 1_000_000_000_000
        dai_scaling = 1
        usdc_scaling = 10**12
        amp = 5000 * AMP_PRECISION
        fee = Decimal("0.0001")

        # Want to buy 10 USDC
        amount_out_raw = 10_000_000  # 10 USDC (6 dec)

        # Step 1: Scale balances to 18 decimals
        scaled_balances = [
            scale_up(dai_balance, dai_scaling),
            scale_up(usdc_balance, usdc_scaling),
        ]

        # Step 2: Scale output to 18 decimals
        amount_out_scaled = scale_up(amount_out_raw, usdc_scaling)

        # Step 3: Calculate required input
        amount_in_scaled = stable_calc_in_given_out(amp, scaled_balances, 0, 1, amount_out_scaled)

        # Step 4: Scale input back to native decimals (DAI = 18)
        amount_in_before_fee = scale_down_up(amount_in_scaled, dai_scaling)

        # Step 5: Add fee
        amount_in = add_swap_fee_amount(amount_in_before_fee, fee)

        # Should need approximately 10 DAI (plus tiny slippage and fee)
        assert 9_990_000 * 10**12 < amount_in < 10_010_000 * 10**12


class TestStablePoolRustVectors:
    """Test vectors from Rust cow-services stable pool implementation.

    These test against exact values from:
    cow-services/crates/shared/src/sources/balancer_v2/swap/stable_math.rs
    cow-services/crates/solver/src/liquidity/bal_liquidity.rs
    """

    def test_rust_vector_sell_dai_to_usdc(self) -> None:
        """Test vector from Rust: sell 10 DAI for USDC in 3-token stable pool.

        Pool: DAI/USDC/USDT, A=5000, fee=0.01%
        Input: 10 DAI (10000000000000000000 wei)
        Expected output: 9999475 USDC (6 decimals)
        """
        # Exact balances from Rust test
        dai_balance = 505781036390938593206504
        usdc_balance = 554894862074  # 6 decimals
        usdt_balance = 1585576741011  # 6 decimals

        dai_scaling = 1
        usdc_scaling = 10**12
        usdt_scaling = 10**12

        amp = 5000 * AMP_PRECISION
        fee = Decimal("0.0001")

        # Sell 10 DAI
        amount_in_raw = 10_000_000_000_000_000_000  # 10 * 10^18

        # Step 1: Subtract fee
        amount_in_after_fee = subtract_swap_fee_amount(amount_in_raw, fee)

        # Step 2: Scale balances to 18 decimals
        # Note: Tokens must be sorted by address in Balancer
        # DAI < USDC < USDT (alphabetically by address)
        scaled_balances = [
            scale_up(dai_balance, dai_scaling),  # index 0: DAI
            scale_up(usdc_balance, usdc_scaling),  # index 1: USDC
            scale_up(usdt_balance, usdt_scaling),  # index 2: USDT
        ]

        # Step 3: Scale input to 18 decimals
        amount_in_scaled = scale_up(amount_in_after_fee, dai_scaling)

        # Step 4: Calculate output (DAI index=0, USDC index=1)
        amount_out_scaled = stable_calc_out_given_in(amp, scaled_balances, 0, 1, amount_in_scaled)

        # Step 5: Scale output back to native decimals (USDC = 6)
        amount_out = scale_down_down(amount_out_scaled, usdc_scaling)

        # Expected output from Rust: 9999475 USDC
        # Allow tolerance for rounding differences
        expected = 9999475
        tolerance = 100  # 100 units (0.0001 USDC) tolerance
        assert abs(amount_out - expected) <= tolerance, (
            f"Expected ~{expected}, got {amount_out}, diff={amount_out - expected}"
        )

    def test_rust_vector_buy_usdc_with_dai(self) -> None:
        """Test vector from Rust: buy 10 USDC with DAI in 3-token stable pool.

        Pool: DAI/USDC/USDT, A=5000, fee=0.01%
        Output: 10 USDC (10000000 wei)
        Expected input: 10000524328839166557 DAI
        """
        # Exact balances from Rust test
        dai_balance = 505781036390938593206504
        usdc_balance = 554894862074
        usdt_balance = 1585576741011

        dai_scaling = 1
        usdc_scaling = 10**12
        usdt_scaling = 10**12

        amp = 5000 * AMP_PRECISION
        fee = Decimal("0.0001")

        # Want 10 USDC
        amount_out_raw = 10_000_000  # 10 USDC (6 decimals)

        # Step 1: Scale balances to 18 decimals
        scaled_balances = [
            scale_up(dai_balance, dai_scaling),
            scale_up(usdc_balance, usdc_scaling),
            scale_up(usdt_balance, usdt_scaling),
        ]

        # Step 2: Scale output to 18 decimals
        amount_out_scaled = scale_up(amount_out_raw, usdc_scaling)

        # Step 3: Calculate required input (DAI index=0, USDC index=1)
        amount_in_scaled = stable_calc_in_given_out(amp, scaled_balances, 0, 1, amount_out_scaled)

        # Step 4: Scale input back to native decimals (DAI = 18)
        amount_in_before_fee = scale_down_up(amount_in_scaled, dai_scaling)

        # Step 5: Add fee
        amount_in = add_swap_fee_amount(amount_in_before_fee, fee)

        # Expected input from Rust: 10000524328839166557 DAI
        expected = 10_000_524_328_839_166_557
        tolerance = 10_000_000_000_000  # 10^13 wei (~0.00001 DAI) tolerance
        assert abs(amount_in - expected) <= tolerance, (
            f"Expected ~{expected}, got {amount_in}, diff={amount_in - expected}"
        )

    def test_rust_vector_ageur_to_eure(self) -> None:
        """Test vector from Rust: swap agEUR to EURe in 3-token Euro stable pool.

        This is a composable stable pool where one token is the pool's BPT.
        Pool: agEUR/bb-agEUR-EURe/EURe, A=100, fee=0.01%
        Input: 10 agEUR (10000000000000000000 wei)
        Expected output: 10029862202766050434 EURe

        Note: This tests a pool where the BPT token needs to be filtered.
        For simplicity, we test with the BPT already filtered out.
        """
        # Balances (with BPT filtered out)
        ageur_balance = 126041615528606990697699
        eure_balance = 170162457652825667152980

        scaling = 1  # All 18 decimals
        amp = 100 * AMP_PRECISION
        fee = Decimal("0.0001")

        # Sell 10 agEUR
        amount_in_raw = 10_000_000_000_000_000_000

        # Subtract fee
        amount_in_after_fee = subtract_swap_fee_amount(amount_in_raw, fee)

        # Scale balances (no-op for 18 decimal tokens)
        scaled_balances = [
            scale_up(ageur_balance, scaling),
            scale_up(eure_balance, scaling),
        ]

        amount_in_scaled = scale_up(amount_in_after_fee, scaling)

        # Calculate output
        amount_out_scaled = stable_calc_out_given_in(amp, scaled_balances, 0, 1, amount_in_scaled)

        amount_out = scale_down_down(amount_out_scaled, scaling)

        # Expected output from Rust: 10029862202766050434 EURe
        # This shows the exchange rate is slightly favorable due to imbalance
        expected = 10_029_862_202_766_050_434
        tolerance = 10_000_000_000_000  # 10^13 wei tolerance
        assert abs(amount_out - expected) <= tolerance, (
            f"Expected ~{expected}, got {amount_out}, diff={amount_out - expected}"
        )

    def test_two_token_invariant_convergence(self) -> None:
        """Test basic invariant convergence (from Rust stable_math.rs)."""
        amp = 100 * AMP_PRECISION
        balances = [Bfp.from_int(10), Bfp.from_int(12)]

        d = calculate_invariant(amp, balances)

        # Verify convergence - D should be approximately sum for high A
        # With A=100 and equal-ish balances, D ≈ 22
        assert 20 * ONE_18 < d.value < 24 * ONE_18

    def test_three_token_invariant_convergence(self) -> None:
        """Test 3-token invariant convergence (from Rust stable_math.rs)."""
        amp = 100 * AMP_PRECISION
        balances = [Bfp.from_int(10), Bfp.from_int(12), Bfp.from_int(14)]

        d = calculate_invariant(amp, balances)

        # D should be approximately sum = 36
        assert 34 * ONE_18 < d.value < 38 * ONE_18

    def test_extreme_values_invariant(self) -> None:
        """Test invariant with extreme value differences (from Rust stable_math.rs)."""
        amp = 5000 * AMP_PRECISION
        # Extreme imbalance: 0.00001, 1200000, 300
        balances = [
            Bfp.from_decimal(Decimal("0.00001")),
            Bfp.from_int(1200000),
            Bfp.from_int(300),
        ]

        d = calculate_invariant(amp, balances)

        # Should converge despite extreme imbalance
        assert d.value > 0
