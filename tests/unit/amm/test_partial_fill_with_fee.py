"""Tests for partial fill calculation with solver fee included in limit price check.

When calculating partial fills for limit orders, the solver fee must be included
in the limit price validation. The formula is:

    output / (input + fee) >= limit_price

Not the naive:
    output / input >= limit_price

This ensures that after the fee is deducted from the executed amount,
the user still receives at least the limit price.
"""

import pytest

from solver.amm.uniswap_v2 import UniswapV2, UniswapV2Pool


class TestPartialFillWithSolverFee:
    """Test that partial fill calculations account for solver fee in limit price."""

    @pytest.fixture
    def amm(self) -> UniswapV2:
        return UniswapV2()

    @pytest.fixture
    def pool(self) -> UniswapV2Pool:
        """Pool with WETH/COW liquidity.

        Using the same values as the partial_fill Rust test fixture:
        - reserve_weth: 3828187314911751990 (~3.83 WETH)
        - reserve_cow: 179617892578796375604692 (~179617 COW)
        - Pool rate: ~46,918 COW per WETH
        """
        return UniswapV2Pool(
            address="0x97b744df0b59d93a866304f97431d8efad29a08d",
            token0="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
            token1="0xdef1ca1fb7fbcdc777520aa7f396b4e015f497ab",  # COW
            reserve0=3828187314911751990,  # WETH reserve
            reserve1=179617892578796375604692,  # COW reserve
            fee_bps=30,
        )

    def test_partial_fill_without_fee_is_larger(self, amm: UniswapV2, pool: UniswapV2Pool):
        """Without fee consideration, partial fill is larger than with fee."""
        # Order: sell 1 WETH for at least 40000 COW (limit rate: 40000)
        sell_amount = 1_000_000_000_000_000_000  # 1 WETH
        buy_amount = 40_000_000_000_000_000_000_000  # 40000 COW
        solver_fee = 2_495_865_000_000_000  # ~0.0025 WETH (from Rust expected)

        # Calculate max fill without fee
        max_input_no_fee = amm.max_fill_sell_order(
            reserve_in=pool.reserve0,
            reserve_out=pool.reserve1,
            sell_amount=sell_amount,
            buy_amount=buy_amount,
            fee_multiplier=pool.fee_multiplier,
        )

        # Calculate max fill with fee
        max_input_with_fee = amm.max_fill_sell_order(
            reserve_in=pool.reserve0,
            reserve_out=pool.reserve1,
            sell_amount=sell_amount,
            buy_amount=buy_amount,
            fee_multiplier=pool.fee_multiplier,
            solver_fee=solver_fee,
        )

        # With fee, max fill should be smaller
        assert max_input_with_fee < max_input_no_fee

        # Verify the fill ratios are reasonable
        # Python without fee: ~0.6507 WETH (65% fill)
        # Python with fee: should be less (around 63%)
        assert max_input_no_fee > 600_000_000_000_000_000  # > 0.6 WETH
        assert max_input_with_fee < max_input_no_fee  # strictly less with fee
        # The reduction should be meaningful (fee is ~0.25% of sell_amount)
        reduction = max_input_no_fee - max_input_with_fee
        assert reduction > 0  # definitely reduced

    def test_partial_fill_with_fee_satisfies_limit_price(self, amm: UniswapV2, pool: UniswapV2Pool):
        """Partial fill with fee must satisfy: output * sell >= buy * (input + fee)."""
        sell_amount = 1_000_000_000_000_000_000
        buy_amount = 40_000_000_000_000_000_000_000
        solver_fee = 2_495_865_000_000_000

        max_input = amm.max_fill_sell_order(
            reserve_in=pool.reserve0,
            reserve_out=pool.reserve1,
            sell_amount=sell_amount,
            buy_amount=buy_amount,
            fee_multiplier=pool.fee_multiplier,
            solver_fee=solver_fee,
        )

        # Get actual output
        actual_output = amm.get_amount_out(
            max_input, pool.reserve0, pool.reserve1, pool.fee_multiplier
        )

        # Verify limit price is satisfied WITH fee in denominator
        # output / (input + fee) >= buy_amount / sell_amount
        # Rearranged: output * sell_amount >= buy_amount * (input + fee)
        assert actual_output * sell_amount >= buy_amount * (max_input + solver_fee)

    def test_partial_fill_without_fee_violates_limit_when_fee_applied(
        self, amm: UniswapV2, pool: UniswapV2Pool
    ):
        """The naive fill (no fee) violates limit price when fee is considered."""
        sell_amount = 1_000_000_000_000_000_000
        buy_amount = 40_000_000_000_000_000_000_000
        solver_fee = 2_495_865_000_000_000

        # Get naive fill (ignoring fee)
        naive_fill = amm.max_fill_sell_order(
            reserve_in=pool.reserve0,
            reserve_out=pool.reserve1,
            sell_amount=sell_amount,
            buy_amount=buy_amount,
            fee_multiplier=pool.fee_multiplier,
            solver_fee=0,  # No fee
        )

        actual_output = amm.get_amount_out(
            naive_fill, pool.reserve0, pool.reserve1, pool.fee_multiplier
        )

        # The naive fill PASSES without fee
        assert actual_output * sell_amount >= buy_amount * naive_fill

        # But FAILS when fee is included
        assert actual_output * sell_amount < buy_amount * (naive_fill + solver_fee)

    def test_zero_fee_gives_same_result_as_no_fee_param(self, amm: UniswapV2, pool: UniswapV2Pool):
        """Passing solver_fee=0 should give the same result as the default."""
        sell_amount = 1_000_000_000_000_000_000
        buy_amount = 40_000_000_000_000_000_000_000

        max_input_default = amm.max_fill_sell_order(
            reserve_in=pool.reserve0,
            reserve_out=pool.reserve1,
            sell_amount=sell_amount,
            buy_amount=buy_amount,
            fee_multiplier=pool.fee_multiplier,
        )

        max_input_zero_fee = amm.max_fill_sell_order(
            reserve_in=pool.reserve0,
            reserve_out=pool.reserve1,
            sell_amount=sell_amount,
            buy_amount=buy_amount,
            fee_multiplier=pool.fee_multiplier,
            solver_fee=0,
        )

        assert max_input_default == max_input_zero_fee


class TestPartialBuyFillWithSolverFee:
    """Test that buy order partial fills account for solver fee."""

    @pytest.fixture
    def amm(self) -> UniswapV2:
        return UniswapV2()

    @pytest.fixture
    def pool(self) -> UniswapV2Pool:
        """Same pool as sell tests."""
        return UniswapV2Pool(
            address="0x97b744df0b59d93a866304f97431d8efad29a08d",
            token0="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
            token1="0xdef1ca1fb7fbcdc777520aa7f396b4e015f497ab",  # COW
            reserve0=3828187314911751990,
            reserve1=179617892578796375604692,
            fee_bps=30,
        )

    def test_buy_order_partial_fill_with_fee(self, amm: UniswapV2, pool: UniswapV2Pool):
        """Buy order partial fill accounts for fee on input side."""
        # Buy order: want 40000 COW, willing to pay up to 1 WETH
        sell_amount = 1_000_000_000_000_000_000  # max 1 WETH input
        buy_amount = 40_000_000_000_000_000_000_000  # want 40000 COW
        solver_fee = 2_495_865_000_000_000  # ~0.0025 WETH

        # For buy orders, limit check is: (input + fee) / output <= sell_amount / buy_amount
        # Rearranged: (input + fee) * buy_amount <= sell_amount * output
        max_output_with_fee = amm.max_fill_buy_order(
            reserve_in=pool.reserve0,
            reserve_out=pool.reserve1,
            sell_amount=sell_amount,
            buy_amount=buy_amount,
            fee_multiplier=pool.fee_multiplier,
            solver_fee=solver_fee,
        )

        max_output_no_fee = amm.max_fill_buy_order(
            reserve_in=pool.reserve0,
            reserve_out=pool.reserve1,
            sell_amount=sell_amount,
            buy_amount=buy_amount,
            fee_multiplier=pool.fee_multiplier,
            solver_fee=0,
        )

        # With fee, max output should be smaller (we can buy less)
        assert max_output_with_fee < max_output_no_fee

        # Verify limit is satisfied with fee
        if max_output_with_fee > 0:
            actual_input = amm.get_amount_in(
                max_output_with_fee, pool.reserve0, pool.reserve1, pool.fee_multiplier
            )
            # (input + fee) * buy_amount <= sell_amount * output
            assert (actual_input + solver_fee) * buy_amount <= sell_amount * max_output_with_fee
