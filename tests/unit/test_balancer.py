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
    parse_stable_pool,
    parse_weighted_pool,
    scale_down_down,
    scale_down_up,
    scale_up,
    stable_calc_in_given_out,
    stable_calc_out_given_in,
    subtract_swap_fee_amount,
)
from solver.math.fixed_point import AMP_PRECISION, ONE_18, Bfp
from solver.models.auction import Liquidity


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


class TestParseWeightedPool:
    """Tests for parse_weighted_pool function."""

    # Test addresses (proper 40-char hex)
    TOKEN_A = "0x" + "a" * 40
    TOKEN_B = "0x" + "b" * 40
    TOKEN_C = "0x" + "c" * 40
    POOL_ADDR = "0x" + "1" * 40
    # 64-char hex for balancerPoolId
    POOL_ID = "0x" + "1" * 64

    def test_parse_weighted_pool_basic(self) -> None:
        """Parse a basic 2-token weighted pool."""
        liq = Liquidity(
            id="pool-1",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "1000000000000000000000"},
                self.TOKEN_B: {"balance": "2000000000000000000000"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1000000000000",
            },
            weights={
                self.TOKEN_A: "0.5",
                self.TOKEN_B: "0.5",
            },
        )

        pool = parse_weighted_pool(liq)

        assert pool is not None
        assert pool.id == "pool-1"  # id is the liquidity id
        assert pool.pool_id == self.POOL_ID  # pool_id is the balancerPoolId
        assert pool.address == self.POOL_ADDR
        assert len(pool.reserves) == 2  # field is 'reserves', not 'tokens'
        assert pool.fee == Decimal("0.003")

        # Check reserves
        token_a = next(r for r in pool.reserves if r.token == self.TOKEN_A)
        token_b = next(r for r in pool.reserves if r.token == self.TOKEN_B)

        assert token_a.balance == 1000000000000000000000
        assert token_a.weight == Decimal("0.5")
        assert token_a.scaling_factor == 1

        assert token_b.balance == 2000000000000000000000
        assert token_b.weight == Decimal("0.5")
        assert token_b.scaling_factor == 1000000000000

    def test_parse_weighted_pool_three_tokens(self) -> None:
        """Parse a 3-token weighted pool."""
        liq = Liquidity(
            id="pool-3tok",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
                self.TOKEN_C: {"balance": "300"},
            },
            address=self.POOL_ADDR,
            fee="0.01",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
                self.TOKEN_C: "1",
            },
            weights={
                self.TOKEN_A: "0.33",
                self.TOKEN_B: "0.33",
                self.TOKEN_C: "0.34",
            },
        )

        pool = parse_weighted_pool(liq)

        assert pool is not None
        assert len(pool.reserves) == 3

    def test_parse_weighted_pool_wrong_kind(self) -> None:
        """Return None for non-weighted pool."""
        liq = Liquidity(
            id="pool-v2",
            kind="constantProduct",
            tokens=[self.TOKEN_A, self.TOKEN_B],
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_missing_weights(self) -> None:
        """Return None if weights are missing."""
        liq = Liquidity(
            id="pool-no-weights",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
            },
            # No weights field
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_default_fee(self) -> None:
        """Use default fee when not provided."""
        liq = Liquidity(
            id="pool-no-fee",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            # No fee field
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
            },
            weights={
                self.TOKEN_A: "0.5",
                self.TOKEN_B: "0.5",
            },
        )

        pool = parse_weighted_pool(liq)

        assert pool is not None
        assert pool.fee == Decimal("0.003")  # Default fee

    def test_parse_weighted_pool_default_scaling_factors(self) -> None:
        """Use default scaling factors when not provided."""
        liq = Liquidity(
            id="pool-no-scaling",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            # No scalingFactors field
            weights={
                self.TOKEN_A: "0.5",
                self.TOKEN_B: "0.5",
            },
        )

        pool = parse_weighted_pool(liq)

        assert pool is not None
        # Default scaling factor is 1 - field is 'reserves' not 'tokens'
        for reserve in pool.reserves:
            assert reserve.scaling_factor == 1

    def test_parse_weighted_pool_zero_balance(self) -> None:
        """Return None if any token has zero balance."""
        liq = Liquidity(
            id="pool-zero-balance",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "0"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.5", self.TOKEN_B: "0.5"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_negative_balance(self) -> None:
        """Return None if any token has negative balance."""
        liq = Liquidity(
            id="pool-negative-balance",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "-100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.5", self.TOKEN_B: "0.5"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_invalid_weight_sum(self) -> None:
        """Return None if weight sum is not approximately 1.0."""
        liq = Liquidity(
            id="pool-bad-weight-sum",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.3", self.TOKEN_B: "0.3"},  # Sum = 0.6, not 1.0
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_zero_weight(self) -> None:
        """Return None if any token has zero weight."""
        liq = Liquidity(
            id="pool-zero-weight",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0", self.TOKEN_B: "1.0"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_case_insensitive_lookup(self) -> None:
        """Weights and scalingFactors should be looked up case-insensitively."""
        # Use mixed case in tokens keys
        token_a_mixed = "0x" + "A" * 40  # Uppercase
        token_b_mixed = "0x" + "B" * 40  # Uppercase

        liq = Liquidity(
            id="pool-case-insensitive",
            kind="weightedProduct",
            tokens={
                token_a_mixed: {"balance": "100"},
                token_b_mixed: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            # Use lowercase in weights/scalingFactors
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.5", self.TOKEN_B: "0.5"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is not None
        assert len(pool.reserves) == 2

    def test_parse_weighted_pool_reserves_sorted(self) -> None:
        """Reserves should be sorted by token address (lowercase)."""
        # TOKEN_C < TOKEN_A < TOKEN_B when sorted by lowercase
        # 0xcc... < 0xaa... is False, 0xaa... < 0xbb... < 0xcc... alphabetically
        liq = Liquidity(
            id="pool-sorting",
            kind="weightedProduct",
            tokens={
                self.TOKEN_C: {"balance": "300"},  # 0xcc...
                self.TOKEN_A: {"balance": "100"},  # 0xaa...
                self.TOKEN_B: {"balance": "200"},  # 0xbb...
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
                self.TOKEN_C: "1",
            },
            weights={
                self.TOKEN_A: "0.33",
                self.TOKEN_B: "0.33",
                self.TOKEN_C: "0.34",
            },
        )

        pool = parse_weighted_pool(liq)
        assert pool is not None
        # Should be sorted: TOKEN_A < TOKEN_B < TOKEN_C
        assert pool.reserves[0].token == self.TOKEN_A
        assert pool.reserves[1].token == self.TOKEN_B
        assert pool.reserves[2].token == self.TOKEN_C

    def test_parse_weighted_pool_invalid_token_data_type(self) -> None:
        """Return None if token_data is not a dict (e.g., tokens is a list)."""
        # This simulates the simplified format where tokens is a list of addresses
        liq = Liquidity(
            id="weighted-invalid-token-data",
            kind="weightedProduct",
            tokens=[self.TOKEN_A, self.TOKEN_B],  # List instead of dict
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.5", self.TOKEN_B: "0.5"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_single_token(self) -> None:
        """Return None if pool has only one valid token."""
        liq = Liquidity(
            id="weighted-single-token",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                # TOKEN_B has zero balance - will be filtered out
                self.TOKEN_B: {"balance": "0"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.5", self.TOKEN_B: "0.5"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is None  # Insufficient tokens (need at least 2)


class TestParseStablePool:
    """Tests for parse_stable_pool function."""

    # Test addresses (proper 40-char hex)
    TOKEN_A = "0x" + "a" * 40
    TOKEN_B = "0x" + "b" * 40
    TOKEN_C = "0x" + "c" * 40
    POOL_ADDR = "0x" + "2" * 40
    DAI_ADDR = "0x" + "d" * 40
    USDC_ADDR = "0x" + "e" * 40
    USDT_ADDR = "0x" + "f" * 40
    # 64-char hex for balancerPoolId
    POOL_ID = "0x" + "2" * 64

    def test_parse_stable_pool_basic(self) -> None:
        """Parse a basic 2-token stable pool."""
        liq = Liquidity(
            id="stable-1",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000000000000000000000"},
                self.TOKEN_B: {"balance": "1000000000000000000000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1000000000000",
            },
            amplificationParameter="200",
        )

        pool = parse_stable_pool(liq)

        assert pool is not None
        assert pool.pool_id == self.POOL_ID  # pool_id is balancerPoolId, not id
        assert pool.address == self.POOL_ADDR
        assert len(pool.reserves) == 2  # Field is 'reserves' not 'tokens'
        assert pool.fee == Decimal("0.0004")
        assert pool.amplification_parameter == 200

    def test_parse_stable_pool_three_tokens(self) -> None:
        """Parse a 3-token stable pool."""
        liq = Liquidity(
            id="stable-3",
            kind="stable",
            tokens={
                self.DAI_ADDR: {"balance": "1000000000000000000000000"},
                self.USDC_ADDR: {"balance": "1000000000000"},
                self.USDT_ADDR: {"balance": "1000000000000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.DAI_ADDR: "1",
                self.USDC_ADDR: "1000000000000",
                self.USDT_ADDR: "1000000000000",
            },
            amplificationParameter="2000",
        )

        pool = parse_stable_pool(liq)

        assert pool is not None
        assert len(pool.reserves) == 3
        assert pool.amplification_parameter == 2000

    def test_parse_stable_pool_filters_bpt(self) -> None:
        """BPT token should be filtered out (composable stable pools)."""
        pool_address = self.POOL_ADDR
        liq = Liquidity(
            id="composable-stable",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                self.TOKEN_B: {"balance": "2000"},
                pool_address: {"balance": "1000000"},  # BPT token
            },
            address=pool_address,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
                pool_address: "1",
            },
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)

        assert pool is not None
        # BPT should be filtered out
        assert len(pool.reserves) == 2
        assert all(r.token != pool_address for r in pool.reserves)

    def test_parse_stable_pool_wrong_kind(self) -> None:
        """Return None for non-stable pool."""
        liq = Liquidity(
            id="pool-v2",
            kind="constantProduct",
            tokens=[self.TOKEN_A, self.TOKEN_B],
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_missing_amp(self) -> None:
        """Return None if amplification parameter is missing."""
        liq = Liquidity(
            id="stable-no-amp",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
            },
            # No amplificationParameter field
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_default_fee(self) -> None:
        """Use default fee when not provided."""
        liq = Liquidity(
            id="stable-no-fee",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            # No fee field
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
            },
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)

        assert pool is not None
        assert pool.fee == Decimal("0.0001")  # Default stable fee is 0.01%

    def test_parse_stable_pool_zero_balance(self) -> None:
        """Return None if any token has zero balance."""
        liq = Liquidity(
            id="stable-zero-balance",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "0"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_negative_balance(self) -> None:
        """Return None if any token has negative balance."""
        liq = Liquidity(
            id="stable-negative-balance",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "-1000"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_zero_amp(self) -> None:
        """Return None if amplification parameter is zero."""
        liq = Liquidity(
            id="stable-zero-amp",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="0",
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_negative_amp(self) -> None:
        """Return None if amplification parameter is negative."""
        liq = Liquidity(
            id="stable-negative-amp",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="-100",
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_case_insensitive_lookup(self) -> None:
        """scalingFactors should be looked up case-insensitively."""
        # Use mixed case in tokens keys
        token_a_mixed = "0x" + "A" * 40  # Uppercase
        token_b_mixed = "0x" + "B" * 40  # Uppercase

        liq = Liquidity(
            id="stable-case-insensitive",
            kind="stable",
            tokens={
                token_a_mixed: {"balance": "1000"},
                token_b_mixed: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            # Use lowercase in scalingFactors
            scalingFactors={self.TOKEN_A: "1000000000000", self.TOKEN_B: "1"},
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is not None
        assert len(pool.reserves) == 2

    def test_parse_stable_pool_reserves_sorted(self) -> None:
        """Reserves should be sorted by token address (lowercase)."""
        liq = Liquidity(
            id="stable-sorting",
            kind="stable",
            tokens={
                self.TOKEN_C: {"balance": "3000"},  # 0xcc...
                self.TOKEN_A: {"balance": "1000"},  # 0xaa...
                self.TOKEN_B: {"balance": "2000"},  # 0xbb...
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
                self.TOKEN_C: "1",
            },
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is not None
        # Should be sorted: TOKEN_A < TOKEN_B < TOKEN_C
        assert pool.reserves[0].token == self.TOKEN_A
        assert pool.reserves[1].token == self.TOKEN_B
        assert pool.reserves[2].token == self.TOKEN_C

    def test_parse_stable_pool_invalid_token_data_type(self) -> None:
        """Return None if token_data is not a dict (e.g., tokens is a list)."""
        liq = Liquidity(
            id="stable-invalid-token-data",
            kind="stable",
            tokens=[self.TOKEN_A, self.TOKEN_B],  # List instead of dict
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_single_token(self) -> None:
        """Return None if pool has only one valid token after filtering."""
        liq = Liquidity(
            id="stable-single-token",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                # TOKEN_B has zero balance - will be filtered out
                self.TOKEN_B: {"balance": "0"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is None  # Insufficient tokens (need at least 2)


class TestPoolRegistryBalancer:
    """Tests for PoolRegistry Balancer pool support."""

    # Test addresses (proper 40-char hex)
    TOKEN_A = "0x" + "a" * 40
    TOKEN_B = "0x" + "b" * 40
    TOKEN_C = "0x" + "c" * 40
    TOKEN_D = "0x" + "d" * 40
    POOL_1 = "0x" + "1" * 40
    POOL_2 = "0x" + "2" * 40
    POOL_3 = "0x" + "3" * 40
    # 64-char hex for balancerPoolId
    POOL_ID_1 = "0x" + "1" * 64
    POOL_ID_2 = "0x" + "2" * 64

    def test_add_and_get_weighted_pool(self) -> None:
        """Add and retrieve weighted pool."""
        from solver.amm.uniswap_v2 import PoolRegistry

        registry = PoolRegistry()
        pool = BalancerWeightedPool(
            id="weighted-1",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A,
                    balance=1000,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B,
                    balance=2000,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=88892,
        )

        registry.add_weighted_pool(pool)

        # Should find by token pair
        pools = registry.get_weighted_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1
        assert pools[0].id == "weighted-1"

        # Also find with reversed order
        pools = registry.get_weighted_pools(self.TOKEN_B, self.TOKEN_A)
        assert len(pools) == 1

    def test_add_and_get_stable_pool(self) -> None:
        """Add and retrieve stable pool."""
        from solver.amm.uniswap_v2 import PoolRegistry

        registry = PoolRegistry()
        pool = BalancerStablePool(
            id="stable-1",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                StableTokenReserve(
                    token=self.TOKEN_A,
                    balance=1000000000000000000000,
                    scaling_factor=1,
                ),
                StableTokenReserve(
                    token=self.TOKEN_B,
                    balance=1000000000,
                    scaling_factor=1000000000000,
                ),
            ),
            fee=Decimal("0.0004"),
            amplification_parameter=Decimal("200"),
            gas_estimate=183520,
        )

        registry.add_stable_pool(pool)

        # Should find by token pair
        pools = registry.get_stable_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1
        assert pools[0].id == "stable-1"

    def test_weighted_pool_multi_token_indexing(self) -> None:
        """3-token weighted pool indexed by all pairs."""
        from solver.amm.uniswap_v2 import PoolRegistry

        registry = PoolRegistry()
        pool = BalancerWeightedPool(
            id="weighted-3tok",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A, balance=100, weight=Decimal("0.33"), scaling_factor=1
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B, balance=200, weight=Decimal("0.33"), scaling_factor=1
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_C, balance=300, weight=Decimal("0.34"), scaling_factor=1
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=88892,
        )

        registry.add_weighted_pool(pool)

        # Should find by all 3 token pairs
        assert len(registry.get_weighted_pools(self.TOKEN_A, self.TOKEN_B)) == 1
        assert len(registry.get_weighted_pools(self.TOKEN_A, self.TOKEN_C)) == 1
        assert len(registry.get_weighted_pools(self.TOKEN_B, self.TOKEN_C)) == 1

    def test_get_pools_for_pair_includes_balancer(self) -> None:
        """get_pools_for_pair returns all pool types."""
        from solver.amm.uniswap_v2 import PoolRegistry, UniswapV2Pool

        registry = PoolRegistry()

        # Add a V2 pool
        v2_pool = UniswapV2Pool(
            address=self.POOL_1,
            token0=self.TOKEN_A,
            token1=self.TOKEN_B,
            reserve0=1000,
            reserve1=2000,
            fee_bps=30,  # 0.3%
        )
        registry.add_pool(v2_pool)

        # Add a weighted pool
        weighted_pool = BalancerWeightedPool(
            id="weighted-1",
            address=self.POOL_2,
            pool_id=self.POOL_ID_1,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A, balance=500, weight=Decimal("0.5"), scaling_factor=1
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B, balance=1000, weight=Decimal("0.5"), scaling_factor=1
                ),
            ),
            fee=Decimal("0.002"),
            version="v0",
            gas_estimate=88892,
        )
        registry.add_weighted_pool(weighted_pool)

        # get_pools_for_pair should return both
        pools = registry.get_pools_for_pair(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 2

    def test_pool_counts(self) -> None:
        """Pool count properties work correctly."""
        from solver.amm.uniswap_v2 import PoolRegistry

        registry = PoolRegistry()

        weighted = BalancerWeightedPool(
            id="w1",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A, balance=100, weight=Decimal("0.5"), scaling_factor=1
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B, balance=100, weight=Decimal("0.5"), scaling_factor=1
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=88892,
        )
        stable = BalancerStablePool(
            id="s1",
            address=self.POOL_2,
            pool_id=self.POOL_ID_2,
            reserves=(
                StableTokenReserve(token=self.TOKEN_C, balance=100, scaling_factor=1),
                StableTokenReserve(token=self.TOKEN_D, balance=100, scaling_factor=1),
            ),
            fee=Decimal("0.0004"),
            amplification_parameter=Decimal("100"),
            gas_estimate=183520,
        )

        assert registry.weighted_pool_count == 0
        assert registry.stable_pool_count == 0

        registry.add_weighted_pool(weighted)
        assert registry.weighted_pool_count == 1

        registry.add_stable_pool(stable)
        assert registry.stable_pool_count == 1

    def test_duplicate_weighted_pool_ignored(self) -> None:
        """Adding the same weighted pool twice is a no-op."""
        from solver.amm.uniswap_v2 import PoolRegistry

        registry = PoolRegistry()

        weighted = BalancerWeightedPool(
            id="w1",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A, balance=100, weight=Decimal("0.5"), scaling_factor=1
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B, balance=100, weight=Decimal("0.5"), scaling_factor=1
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=88892,
        )

        registry.add_weighted_pool(weighted)
        assert registry.weighted_pool_count == 1

        # Adding same pool again should be ignored
        registry.add_weighted_pool(weighted)
        assert registry.weighted_pool_count == 1

        # get_weighted_pools should return only one
        pools = registry.get_weighted_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1

    def test_duplicate_stable_pool_ignored(self) -> None:
        """Adding the same stable pool twice is a no-op."""
        from solver.amm.uniswap_v2 import PoolRegistry

        registry = PoolRegistry()

        stable = BalancerStablePool(
            id="s1",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                StableTokenReserve(token=self.TOKEN_A, balance=100, scaling_factor=1),
                StableTokenReserve(token=self.TOKEN_B, balance=100, scaling_factor=1),
            ),
            fee=Decimal("0.0004"),
            amplification_parameter=Decimal("100"),
            gas_estimate=183520,
        )

        registry.add_stable_pool(stable)
        assert registry.stable_pool_count == 1

        # Adding same pool again should be ignored
        registry.add_stable_pool(stable)
        assert registry.stable_pool_count == 1

        # get_stable_pools should return only one
        pools = registry.get_stable_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1


class TestBuildRegistryWithBalancer:
    """Tests for build_registry_from_liquidity with Balancer pools."""

    # Test addresses (proper 40-char hex)
    TOKEN_A = "0x" + "a" * 40
    TOKEN_B = "0x" + "b" * 40
    TOKEN_C = "0x" + "c" * 40
    TOKEN_D = "0x" + "d" * 40
    TOKEN_E = "0x" + "e" * 40
    TOKEN_F = "0x" + "f" * 40
    POOL_1 = "0x" + "1" * 40
    POOL_2 = "0x" + "2" * 40
    POOL_3 = "0x" + "3" * 40
    # 64-char hex for balancerPoolId
    POOL_ID_1 = "0x" + "1" * 64
    POOL_ID_2 = "0x" + "2" * 64
    POOL_ID_3 = "0x" + "3" * 64

    def test_parse_weighted_pool_from_liquidity(self) -> None:
        """build_registry_from_liquidity parses weighted pools."""
        from solver.amm.uniswap_v2 import build_registry_from_liquidity

        liquidity_list = [
            Liquidity(
                id="weighted-pool",
                kind="weightedProduct",
                tokens={
                    self.TOKEN_A: {"balance": "10000000000000000000"},
                    self.TOKEN_B: {"balance": "100000000000000000000"},
                },
                address=self.POOL_1,
                fee="0.003",
                balancerPoolId=self.POOL_ID_1,
                scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
                weights={self.TOKEN_A: "0.8", self.TOKEN_B: "0.2"},
            )
        ]

        registry = build_registry_from_liquidity(liquidity_list)

        assert registry.weighted_pool_count == 1
        pools = registry.get_weighted_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1
        assert pools[0].id == "weighted-pool"  # id is the liquidity id

    def test_parse_stable_pool_from_liquidity(self) -> None:
        """build_registry_from_liquidity parses stable pools."""
        from solver.amm.uniswap_v2 import build_registry_from_liquidity

        liquidity_list = [
            Liquidity(
                id="stable-pool",
                kind="stable",
                tokens={
                    self.TOKEN_A: {"balance": "1000000000000000000000000"},
                    self.TOKEN_B: {"balance": "1000000000000"},
                },
                address=self.POOL_1,
                fee="0.0004",
                balancerPoolId=self.POOL_ID_1,
                scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1000000000000"},
                amplificationParameter="200",
            )
        ]

        registry = build_registry_from_liquidity(liquidity_list)

        assert registry.stable_pool_count == 1
        pools = registry.get_stable_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1
        assert pools[0].id == "stable-pool"  # id is the liquidity id

    def test_mixed_pool_types(self) -> None:
        """build_registry_from_liquidity handles mixed pool types."""
        from solver.amm.uniswap_v2 import build_registry_from_liquidity

        liquidity_list = [
            # V2 pool
            Liquidity(
                id="v2-pool",
                kind="constantProduct",
                tokens={
                    self.TOKEN_A: {"balance": "1000"},
                    self.TOKEN_B: {"balance": "2000"},
                },
                address=self.POOL_1,
                fee="0.003",
            ),
            # Weighted pool
            Liquidity(
                id="weighted-pool",
                kind="weightedProduct",
                tokens={
                    self.TOKEN_C: {"balance": "500"},
                    self.TOKEN_D: {"balance": "1500"},
                },
                address=self.POOL_2,
                fee="0.002",
                balancerPoolId=self.POOL_ID_2,
                scalingFactors={self.TOKEN_C: "1", self.TOKEN_D: "1"},
                weights={self.TOKEN_C: "0.5", self.TOKEN_D: "0.5"},
            ),
            # Stable pool
            Liquidity(
                id="stable-pool",
                kind="stable",
                tokens={
                    self.TOKEN_E: {"balance": "1000"},
                    self.TOKEN_F: {"balance": "1000"},
                },
                address=self.POOL_3,
                fee="0.0004",
                balancerPoolId=self.POOL_ID_3,
                scalingFactors={self.TOKEN_E: "1", self.TOKEN_F: "1"},
                amplificationParameter="100",
            ),
        ]

        registry = build_registry_from_liquidity(liquidity_list)

        assert registry.pool_count == 1  # V2
        assert registry.weighted_pool_count == 1
        assert registry.stable_pool_count == 1


class TestBalancerWeightedAMM:
    """Tests for BalancerWeightedAMM class."""

    # Test addresses
    TOKEN_A = "0x" + "a" * 40
    TOKEN_B = "0x" + "b" * 40
    POOL_ADDR = "0x" + "1" * 40
    POOL_ID = "0x" + "1" * 64

    def _make_weighted_pool(
        self,
        balance_a: int = 1_000_000_000_000_000_000_000,  # 1000 * 10^18
        balance_b: int = 1_000_000_000_000_000_000_000,
        weight_a: str = "0.5",
        weight_b: str = "0.5",
        fee: str = "0.003",
        scaling_factor_a: int = 1,
        scaling_factor_b: int = 1,
    ) -> BalancerWeightedPool:
        """Create a weighted pool for testing."""
        return BalancerWeightedPool(
            id="weighted-test",
            address=self.POOL_ADDR,
            pool_id=self.POOL_ID,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A,
                    balance=balance_a,
                    weight=Decimal(weight_a),
                    scaling_factor=scaling_factor_a,
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B,
                    balance=balance_b,
                    weight=Decimal(weight_b),
                    scaling_factor=scaling_factor_b,
                ),
            ),
            fee=Decimal(fee),
            version="v0",
            gas_estimate=88892,
        )

    def test_simulate_swap_basic(self) -> None:
        """Basic sell order simulation."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        # Swap 10 tokens (10 * 10^18)
        amount_in = 10_000_000_000_000_000_000
        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, amount_in)

        assert result is not None
        assert result.amount_in == amount_in
        assert result.amount_out > 0
        assert result.amount_out < amount_in  # Output less due to fee
        assert result.pool_address == self.POOL_ADDR
        assert result.token_in.lower() == self.TOKEN_A.lower()
        assert result.token_out.lower() == self.TOKEN_B.lower()

    def test_simulate_swap_invalid_token_in(self) -> None:
        """Returns None for invalid input token."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        invalid_token = "0x" + "c" * 40
        result = amm.simulate_swap(pool, invalid_token, self.TOKEN_B, 1000)

        assert result is None

    def test_simulate_swap_invalid_token_out(self) -> None:
        """Returns None for invalid output token."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        invalid_token = "0x" + "c" * 40
        result = amm.simulate_swap(pool, self.TOKEN_A, invalid_token, 1000)

        assert result is None

    def test_simulate_swap_self_swap_rejected(self) -> None:
        """Returns None when token_in == token_out."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_A, 1000)

        assert result is None

    def test_simulate_swap_exact_output_self_swap_rejected(self) -> None:
        """Returns None when token_in == token_out for exact output."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        result = amm.simulate_swap_exact_output(pool, self.TOKEN_A, self.TOKEN_A, 1000)

        assert result is None

    def test_simulate_swap_exceeds_max_in_ratio(self) -> None:
        """Returns None when swap exceeds 30% of reserves."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool(balance_a=1000)

        # Try to swap more than 30% of balance
        amount_in = 500  # 50% of balance
        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, amount_in)

        assert result is None

    def test_simulate_swap_exact_output_basic(self) -> None:
        """Basic buy order simulation."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        # Want to receive 10 tokens
        amount_out = 10_000_000_000_000_000_000
        result = amm.simulate_swap_exact_output(pool, self.TOKEN_A, self.TOKEN_B, amount_out)

        assert result is not None
        assert result.amount_out == amount_out
        assert result.amount_in > amount_out  # Input more due to fee
        assert result.pool_address == self.POOL_ADDR

    def test_simulate_swap_exact_output_invalid_token(self) -> None:
        """Returns None for invalid input token."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        invalid_token = "0x" + "c" * 40
        result = amm.simulate_swap_exact_output(pool, invalid_token, self.TOKEN_B, 1000)

        assert result is None

    def test_simulate_swap_asymmetric_weights(self) -> None:
        """Swap with asymmetric weights (80/20 pool)."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool(weight_a="0.8", weight_b="0.2")

        amount_in = 10_000_000_000_000_000_000
        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, amount_in)

        assert result is not None
        # With 80/20 weights, swapping from heavier token gives less output
        assert result.amount_out > 0

    def test_liquidity_id_property(self) -> None:
        """liquidity_id property returns id field."""
        pool = self._make_weighted_pool()
        assert pool.liquidity_id == "weighted-test"
        assert pool.liquidity_id == pool.id


class TestBalancerStableAMM:
    """Tests for BalancerStableAMM class."""

    # Test addresses
    TOKEN_A = "0x" + "a" * 40
    TOKEN_B = "0x" + "b" * 40
    TOKEN_C = "0x" + "c" * 40
    POOL_ADDR = "0x" + "2" * 40
    POOL_ID = "0x" + "2" * 64

    def _make_stable_pool(
        self,
        balance_a: int = 1_000_000_000_000_000_000_000_000,  # 1M * 10^18
        balance_b: int = 1_000_000_000_000_000_000_000_000,
        scaling_factor_a: int = 1,
        scaling_factor_b: int = 1,
        amp: str = "200",
        fee: str = "0.0004",
    ) -> BalancerStablePool:
        """Create a stable pool for testing."""
        return BalancerStablePool(
            id="stable-test",
            address=self.POOL_ADDR,
            pool_id=self.POOL_ID,
            reserves=(
                StableTokenReserve(
                    token=self.TOKEN_A,
                    balance=balance_a,
                    scaling_factor=scaling_factor_a,
                ),
                StableTokenReserve(
                    token=self.TOKEN_B,
                    balance=balance_b,
                    scaling_factor=scaling_factor_b,
                ),
            ),
            amplification_parameter=Decimal(amp),
            fee=Decimal(fee),
            gas_estimate=183520,
        )

    def test_simulate_swap_basic(self) -> None:
        """Basic sell order simulation - stable pools give nearly 1:1."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_stable_pool()

        # Swap 1000 tokens
        amount_in = 1_000_000_000_000_000_000_000  # 1000 * 10^18
        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, amount_in)

        assert result is not None
        assert result.amount_in == amount_in
        assert result.amount_out > 0
        # Stable pool output should be close to input (minus small fee and slippage)
        assert result.amount_out > amount_in * 99 // 100  # Within 1%
        assert result.pool_address == self.POOL_ADDR
        assert result.token_in.lower() == self.TOKEN_A.lower()
        assert result.token_out.lower() == self.TOKEN_B.lower()

    def test_simulate_swap_invalid_token_in(self) -> None:
        """Returns None for invalid input token."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_stable_pool()

        invalid_token = "0x" + "d" * 40
        result = amm.simulate_swap(pool, invalid_token, self.TOKEN_B, 1000)

        assert result is None

    def test_simulate_swap_invalid_token_out(self) -> None:
        """Returns None for invalid output token."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_stable_pool()

        invalid_token = "0x" + "d" * 40
        result = amm.simulate_swap(pool, self.TOKEN_A, invalid_token, 1000)

        assert result is None

    def test_simulate_swap_exact_output_basic(self) -> None:
        """Basic buy order simulation."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_stable_pool()

        # Want to receive 1000 tokens
        amount_out = 1_000_000_000_000_000_000_000
        result = amm.simulate_swap_exact_output(pool, self.TOKEN_A, self.TOKEN_B, amount_out)

        assert result is not None
        assert result.amount_out == amount_out
        # Input should be close to output (plus small fee)
        assert result.amount_in < amount_out * 102 // 100  # Within 2%
        assert result.pool_address == self.POOL_ADDR

    def test_simulate_swap_exact_output_invalid_token(self) -> None:
        """Returns None for invalid input token."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_stable_pool()

        invalid_token = "0x" + "d" * 40
        result = amm.simulate_swap_exact_output(pool, invalid_token, self.TOKEN_B, 1000)

        assert result is None

    def test_simulate_swap_different_scaling_factors(self) -> None:
        """Swap with different scaling factors (like USDC with 6 decimals)."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        # Simulate USDC (6 decimals) vs DAI (18 decimals)
        # USDC needs scaling_factor = 10^12 to normalize to 18 decimals
        pool = self._make_stable_pool(
            balance_a=1_000_000_000_000_000_000_000_000,  # 1M DAI (18 dec)
            balance_b=1_000_000_000_000,  # 1M USDC (6 dec)
            scaling_factor_a=1,
            scaling_factor_b=1_000_000_000_000,  # 10^12
        )

        # Swap 1000 DAI (18 decimals) for USDC
        amount_in = 1_000_000_000_000_000_000_000
        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, amount_in)

        assert result is not None
        assert result.amount_out > 0
        # Output should be in USDC's decimals (6)
        # 1000 DAI ≈ 1000 USDC = 1_000_000_000 (6 decimals)
        # Allow for some slippage and fee
        assert result.amount_out > 990_000_000  # At least 990 USDC

    def test_liquidity_id_property(self) -> None:
        """liquidity_id property returns id field."""
        pool = self._make_stable_pool()
        assert pool.liquidity_id == "stable-test"
        assert pool.liquidity_id == pool.id


class TestBalancerAMMIntegration:
    """Integration tests comparing AMM outputs to math functions."""

    TOKEN_A = "0x" + "a" * 40
    TOKEN_B = "0x" + "b" * 40
    POOL_ADDR = "0x" + "1" * 40
    POOL_ID = "0x" + "1" * 64

    def test_weighted_amm_matches_math_functions(self) -> None:
        """AMM simulate_swap matches direct math function calls."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Create pool with known values
        pool = BalancerWeightedPool(
            id="test",
            address=self.POOL_ADDR,
            pool_id=self.POOL_ID,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A,
                    balance=1_000_000_000_000_000_000_000,  # 1000 * 10^18
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B,
                    balance=1_000_000_000_000_000_000_000,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=88892,
        )

        amm = BalancerWeightedAMM()
        amount_in = 10_000_000_000_000_000_000  # 10 * 10^18

        # Get AMM result
        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, amount_in)
        assert result is not None

        # Calculate manually
        balance_in = scale_up(1_000_000_000_000_000_000_000, 1)
        balance_out = scale_up(1_000_000_000_000_000_000_000, 1)
        amount_in_scaled = scale_up(amount_in, 1)

        amount_in_after_fee = subtract_swap_fee_amount(amount_in_scaled.value, Decimal("0.003"))
        amount_out_scaled = calc_out_given_in(
            balance_in=balance_in,
            weight_in=Bfp.from_decimal(Decimal("0.5")),
            balance_out=balance_out,
            weight_out=Bfp.from_decimal(Decimal("0.5")),
            amount_in=Bfp(amount_in_after_fee),
        )
        expected_output = scale_down_down(amount_out_scaled, 1)

        assert result.amount_out == expected_output

    def test_stable_amm_matches_math_functions(self) -> None:
        """AMM simulate_swap matches direct math function calls."""
        from solver.amm.balancer import BalancerStableAMM

        # Create pool with known values
        pool = BalancerStablePool(
            id="test",
            address=self.POOL_ADDR,
            pool_id=self.POOL_ID,
            reserves=(
                StableTokenReserve(
                    token=self.TOKEN_A,
                    balance=1_000_000_000_000_000_000_000_000,  # 1M * 10^18
                    scaling_factor=1,
                ),
                StableTokenReserve(
                    token=self.TOKEN_B,
                    balance=1_000_000_000_000_000_000_000_000,
                    scaling_factor=1,
                ),
            ),
            amplification_parameter=Decimal("200"),
            fee=Decimal("0.0004"),
            gas_estimate=183520,
        )

        amm = BalancerStableAMM()
        amount_in = 1_000_000_000_000_000_000_000  # 1000 * 10^18

        # Get AMM result
        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, amount_in)
        assert result is not None

        # Calculate manually
        balances = [
            scale_up(1_000_000_000_000_000_000_000_000, 1),
            scale_up(1_000_000_000_000_000_000_000_000, 1),
        ]
        amount_in_scaled = scale_up(amount_in, 1)
        amount_in_after_fee = subtract_swap_fee_amount(amount_in_scaled.value, Decimal("0.0004"))

        # Note: AMM multiplies pool.amplification_parameter by AMP_PRECISION (1000)
        # so we need to do the same for the manual calculation to match
        amount_out_scaled = stable_calc_out_given_in(
            amp=200 * 1000,  # scaled by AMP_PRECISION
            balances=balances,
            token_index_in=0,
            token_index_out=1,
            amount_in=Bfp(amount_in_after_fee),
        )
        expected_output = scale_down_down(amount_out_scaled, 1)

        assert result.amount_out == expected_output


class TestBalancerMultiTokenPools:
    """Tests for pools with 3+ tokens (e.g., DAI/USDC/USDT)."""

    TOKEN_A = "0x" + "a" * 40  # DAI
    TOKEN_B = "0x" + "b" * 40  # USDC
    TOKEN_C = "0x" + "c" * 40  # USDT
    POOL_ADDR = "0x" + "1" * 40
    POOL_ID = "0x" + "1" * 64

    def _make_three_token_weighted_pool(self) -> BalancerWeightedPool:
        """Create a 3-token weighted pool."""
        return BalancerWeightedPool(
            id="weighted-3token",
            address=self.POOL_ADDR,
            pool_id=self.POOL_ID,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A,
                    balance=1_000_000_000_000_000_000_000,
                    weight=Decimal("0.33"),
                    scaling_factor=1,
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B,
                    balance=1_000_000_000_000_000_000_000,
                    weight=Decimal("0.34"),
                    scaling_factor=1,
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_C,
                    balance=1_000_000_000_000_000_000_000,
                    weight=Decimal("0.33"),
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=88892,
        )

    def _make_three_token_stable_pool(self) -> BalancerStablePool:
        """Create a 3-token stable pool (like 3pool)."""
        return BalancerStablePool(
            id="stable-3token",
            address=self.POOL_ADDR,
            pool_id=self.POOL_ID,
            reserves=(
                StableTokenReserve(
                    token=self.TOKEN_A,
                    balance=1_000_000_000_000_000_000_000_000,
                    scaling_factor=1,
                ),
                StableTokenReserve(
                    token=self.TOKEN_B,
                    balance=1_000_000_000_000_000_000_000_000,
                    scaling_factor=1,
                ),
                StableTokenReserve(
                    token=self.TOKEN_C,
                    balance=1_000_000_000_000_000_000_000_000,
                    scaling_factor=1,
                ),
            ),
            amplification_parameter=Decimal("200"),
            fee=Decimal("0.0004"),
            gas_estimate=183520,
        )

    def test_weighted_swap_a_to_b(self) -> None:
        """Swap token A -> B in 3-token weighted pool."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_three_token_weighted_pool()

        amount_in = 10_000_000_000_000_000_000
        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, amount_in)

        assert result is not None
        assert result.amount_out > 0
        assert result.pool_address == self.POOL_ADDR

    def test_weighted_swap_a_to_c(self) -> None:
        """Swap token A -> C in 3-token weighted pool."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_three_token_weighted_pool()

        amount_in = 10_000_000_000_000_000_000
        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_C, amount_in)

        assert result is not None
        assert result.amount_out > 0
        assert result.pool_address == self.POOL_ADDR

    def test_weighted_swap_b_to_c(self) -> None:
        """Swap token B -> C in 3-token weighted pool."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_three_token_weighted_pool()

        amount_in = 10_000_000_000_000_000_000
        result = amm.simulate_swap(pool, self.TOKEN_B, self.TOKEN_C, amount_in)

        assert result is not None
        assert result.amount_out > 0

    def test_weighted_exact_output_a_to_c(self) -> None:
        """Buy order A -> C in 3-token weighted pool."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_three_token_weighted_pool()

        amount_out = 10_000_000_000_000_000_000
        result = amm.simulate_swap_exact_output(pool, self.TOKEN_A, self.TOKEN_C, amount_out)

        assert result is not None
        assert result.amount_out == amount_out
        assert result.amount_in > 0

    def test_stable_swap_a_to_b(self) -> None:
        """Swap token A -> B in 3-token stable pool."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_three_token_stable_pool()

        amount_in = 1_000_000_000_000_000_000_000
        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, amount_in)

        assert result is not None
        assert result.amount_out > 0
        assert result.pool_address == self.POOL_ADDR

    def test_stable_swap_a_to_c(self) -> None:
        """Swap token A -> C in 3-token stable pool."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_three_token_stable_pool()

        amount_in = 1_000_000_000_000_000_000_000
        result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_C, amount_in)

        assert result is not None
        assert result.amount_out > 0

    def test_stable_swap_b_to_c(self) -> None:
        """Swap token B -> C in 3-token stable pool."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_three_token_stable_pool()

        amount_in = 1_000_000_000_000_000_000_000
        result = amm.simulate_swap(pool, self.TOKEN_B, self.TOKEN_C, amount_in)

        assert result is not None
        assert result.amount_out > 0

    def test_stable_exact_output_a_to_c(self) -> None:
        """Buy order A -> C in 3-token stable pool."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_three_token_stable_pool()

        amount_out = 1_000_000_000_000_000_000_000
        result = amm.simulate_swap_exact_output(pool, self.TOKEN_A, self.TOKEN_C, amount_out)

        assert result is not None
        assert result.amount_out == amount_out
        assert result.amount_in > 0

    def test_different_outputs_for_different_targets(self) -> None:
        """Swapping A->B should give different result than A->C."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_three_token_weighted_pool()

        amount_in = 10_000_000_000_000_000_000

        result_ab = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, amount_in)
        result_ac = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_C, amount_in)

        assert result_ab is not None
        assert result_ac is not None
        # Results should differ because B has weight 0.34 while C has 0.33
        assert result_ab.amount_out != result_ac.amount_out


class TestWeightedMaxFill:
    """Tests for max_fill_sell_order and max_fill_buy_order on weighted pools."""

    TOKEN_A = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    TOKEN_B = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    def _make_weighted_pool(self) -> BalancerWeightedPool:
        """Create a 50/50 weighted pool for testing."""
        return BalancerWeightedPool(
            id="test-weighted-pool",
            address="0x1111111111111111111111111111111111111111",
            pool_id="0x" + "11" * 32,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A,
                    balance=100_000_000_000_000_000_000_000,  # 100K tokens
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B,
                    balance=100_000_000_000_000_000_000_000,  # 100K tokens
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.003"),
            version="v3Plus",
            gas_estimate=120_000,
        )

    def test_max_fill_sell_order_full_fill_possible(self) -> None:
        """When full fill satisfies limit, returns full amount."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        # Selling 1000 tokens with a very generous limit price
        sell_amount = 1000_000_000_000_000_000_000  # 1000 tokens
        buy_amount = 100_000_000_000_000_000_000  # 100 tokens min (10:1 rate)

        max_fill = amm.max_fill_sell_order(
            pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount
        )

        # Should be able to fill entire sell amount
        assert max_fill == sell_amount

    def test_max_fill_sell_order_partial_fill(self) -> None:
        """When full fill doesn't satisfy limit, returns partial fill."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        # Selling 10000 tokens requiring 1:1 rate (impossible to fill fully due to slippage)
        sell_amount = 10_000_000_000_000_000_000_000  # 10K tokens
        buy_amount = 10_000_000_000_000_000_000_000  # 10K tokens min (1:1 rate)

        max_fill = amm.max_fill_sell_order(
            pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount
        )

        # Should return some partial amount less than sell_amount
        assert max_fill < sell_amount
        assert max_fill >= 0

    def test_max_fill_sell_order_impossible(self) -> None:
        """When limit price is impossible, returns 0."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        # Asking for way more output than possible (100:1 rate in a 50/50 pool)
        sell_amount = 1000_000_000_000_000_000_000  # 1000 tokens
        buy_amount = 100_000_000_000_000_000_000_000  # 100K tokens min (impossible)

        max_fill = amm.max_fill_sell_order(
            pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount
        )

        assert max_fill == 0

    def test_max_fill_sell_order_verifies_limit(self) -> None:
        """Max fill result actually satisfies the limit price."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        sell_amount = 5_000_000_000_000_000_000_000  # 5K tokens
        buy_amount = 4_000_000_000_000_000_000_000  # 4K tokens min (reasonable limit)

        max_fill = amm.max_fill_sell_order(
            pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount
        )

        if max_fill > 0:
            result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, max_fill)
            assert result is not None
            # Verify: output/input >= buy_amount/sell_amount
            assert result.amount_out * sell_amount >= buy_amount * max_fill

    def test_max_fill_buy_order_full_fill_possible(self) -> None:
        """When full fill satisfies limit, returns full buy amount."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        # Buying 1000 tokens with generous sell budget
        sell_amount = 10_000_000_000_000_000_000_000  # 10K tokens max
        buy_amount = 1000_000_000_000_000_000_000  # 1000 tokens desired

        max_fill = amm.max_fill_buy_order(pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount)

        # Should be able to buy the full amount
        assert max_fill == buy_amount

    def test_max_fill_buy_order_partial_fill(self) -> None:
        """When full fill doesn't satisfy limit, returns partial fill."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        # Very tight budget that won't allow full buy
        sell_amount = 500_000_000_000_000_000_000  # 500 tokens max
        buy_amount = 1000_000_000_000_000_000_000  # 1000 tokens desired (impossible at 1:1)

        max_fill = amm.max_fill_buy_order(pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount)

        # Should return some partial amount less than buy_amount
        assert max_fill < buy_amount
        assert max_fill >= 0

    def test_max_fill_buy_order_verifies_limit(self) -> None:
        """Max fill result actually satisfies the limit price."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        sell_amount = 2_000_000_000_000_000_000_000  # 2K tokens max
        buy_amount = 1500_000_000_000_000_000_000  # 1.5K tokens desired

        max_fill = amm.max_fill_buy_order(pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount)

        if max_fill > 0:
            result = amm.simulate_swap_exact_output(pool, self.TOKEN_A, self.TOKEN_B, max_fill)
            assert result is not None
            # Verify: input/output <= sell_amount/buy_amount
            assert result.amount_in * buy_amount <= sell_amount * max_fill

    def test_max_fill_zero_amounts(self) -> None:
        """Zero amounts return 0."""
        from solver.amm.balancer import BalancerWeightedAMM

        amm = BalancerWeightedAMM()
        pool = self._make_weighted_pool()

        assert amm.max_fill_sell_order(pool, self.TOKEN_A, self.TOKEN_B, 0, 100) == 0
        assert amm.max_fill_sell_order(pool, self.TOKEN_A, self.TOKEN_B, 100, 0) == 0
        assert amm.max_fill_buy_order(pool, self.TOKEN_A, self.TOKEN_B, 0, 100) == 0
        assert amm.max_fill_buy_order(pool, self.TOKEN_A, self.TOKEN_B, 100, 0) == 0


class TestStableMaxFill:
    """Tests for max_fill_sell_order and max_fill_buy_order on stable pools."""

    TOKEN_A = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    TOKEN_B = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    def _make_stable_pool(self) -> BalancerStablePool:
        """Create a stable pool for testing."""
        return BalancerStablePool(
            id="test-stable-pool",
            address="0x2222222222222222222222222222222222222222",
            pool_id="0x" + "22" * 32,
            reserves=(
                StableTokenReserve(
                    token=self.TOKEN_A,
                    balance=100_000_000_000_000_000_000_000_000,  # 100M tokens
                    scaling_factor=1,
                ),
                StableTokenReserve(
                    token=self.TOKEN_B,
                    balance=100_000_000_000_000_000_000_000_000,  # 100M tokens
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.0001"),  # 0.01%
            amplification_parameter=Decimal("5000"),
            gas_estimate=100_000,
        )

    def test_max_fill_sell_order_full_fill_possible(self) -> None:
        """When full fill satisfies limit, returns full amount."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_stable_pool()

        # Stable pools should be near 1:1, so generous 0.9:1 limit should work
        sell_amount = 1000_000_000_000_000_000_000  # 1000 tokens
        buy_amount = 900_000_000_000_000_000_000  # 900 tokens min (0.9:1)

        max_fill = amm.max_fill_sell_order(
            pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount
        )

        # Should be able to fill entire sell amount
        assert max_fill == sell_amount

    def test_max_fill_sell_order_partial_fill(self) -> None:
        """When full fill doesn't satisfy limit, returns partial fill."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_stable_pool()

        # Require strict 1:1 rate with a large amount (slippage will prevent full fill)
        sell_amount = 10_000_000_000_000_000_000_000_000  # 10M tokens
        buy_amount = 10_000_000_000_000_000_000_000_000  # 10M tokens min

        max_fill = amm.max_fill_sell_order(
            pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount
        )

        # Might get partial fill or 0 if limit too strict
        assert max_fill <= sell_amount
        assert max_fill >= 0

    def test_max_fill_sell_order_verifies_limit(self) -> None:
        """Max fill result actually satisfies the limit price."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_stable_pool()

        sell_amount = 5_000_000_000_000_000_000_000  # 5K tokens
        buy_amount = 4_900_000_000_000_000_000_000  # 4.9K tokens min (0.98:1)

        max_fill = amm.max_fill_sell_order(
            pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount
        )

        if max_fill > 0:
            result = amm.simulate_swap(pool, self.TOKEN_A, self.TOKEN_B, max_fill)
            assert result is not None
            # Verify: output/input >= buy_amount/sell_amount
            assert result.amount_out * sell_amount >= buy_amount * max_fill

    def test_max_fill_buy_order_full_fill_possible(self) -> None:
        """When full fill satisfies limit, returns full buy amount."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_stable_pool()

        # Generous 1.1:1 rate should allow full fill
        sell_amount = 1100_000_000_000_000_000_000  # 1.1K tokens max
        buy_amount = 1000_000_000_000_000_000_000  # 1K tokens desired

        max_fill = amm.max_fill_buy_order(pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount)

        # Should be able to buy the full amount
        assert max_fill == buy_amount

    def test_max_fill_buy_order_verifies_limit(self) -> None:
        """Max fill result actually satisfies the limit price."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_stable_pool()

        sell_amount = 1050_000_000_000_000_000_000  # 1050 tokens max
        buy_amount = 1000_000_000_000_000_000_000  # 1000 tokens desired

        max_fill = amm.max_fill_buy_order(pool, self.TOKEN_A, self.TOKEN_B, sell_amount, buy_amount)

        if max_fill > 0:
            result = amm.simulate_swap_exact_output(pool, self.TOKEN_A, self.TOKEN_B, max_fill)
            assert result is not None
            # Verify: input/output <= sell_amount/buy_amount
            assert result.amount_in * buy_amount <= sell_amount * max_fill

    def test_max_fill_zero_amounts(self) -> None:
        """Zero amounts return 0."""
        from solver.amm.balancer import BalancerStableAMM

        amm = BalancerStableAMM()
        pool = self._make_stable_pool()

        assert amm.max_fill_sell_order(pool, self.TOKEN_A, self.TOKEN_B, 0, 100) == 0
        assert amm.max_fill_sell_order(pool, self.TOKEN_A, self.TOKEN_B, 100, 0) == 0
        assert amm.max_fill_buy_order(pool, self.TOKEN_A, self.TOKEN_B, 0, 100) == 0
        assert amm.max_fill_buy_order(pool, self.TOKEN_A, self.TOKEN_B, 100, 0) == 0
