"""Tests for Balancer AMM classes.

This module tests:
- BalancerWeightedAMM (simulate_swap, simulate_swap_exact_output, max_fill)
- BalancerStableAMM (simulate_swap, simulate_swap_exact_output, max_fill)
- AMM integration tests comparing outputs to math functions
- Multi-token pool tests (3+ tokens)
- Max fill tests for partial orders
"""

from decimal import Decimal

from solver.amm.balancer import (
    BalancerStablePool,
    BalancerWeightedPool,
    StableTokenReserve,
    WeightedTokenReserve,
    calc_out_given_in,
    scale_down_down,
    scale_up,
    subtract_swap_fee_amount,
)
from solver.math.fixed_point import Bfp


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
        # Output is >= requested due to forward simulation rounding
        assert result.amount_out >= amount_out
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
        from solver.amm.balancer import BalancerStableAMM, stable_calc_out_given_in

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
        # Output is >= requested due to forward simulation rounding
        assert result.amount_out >= amount_out
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
