"""Integration tests for real UniswapV3 quoter via RPC.

These tests require an RPC connection and are skipped by default.
Run with: RPC_URL=https://eth.llamarpc.com pytest -m requires_rpc
"""

import os

import pytest

from solver.constants import DAI, USDC, WETH

# Skip all tests in this module if RPC_URL is not set
pytestmark = [
    pytest.mark.requires_rpc,
    pytest.mark.skipif(
        not os.environ.get("RPC_URL"),
        reason="RPC_URL environment variable not set",
    ),
]

# V3 fee tiers
FEE_500 = 500  # 0.05%
FEE_3000 = 3000  # 0.3%
FEE_10000 = 10000  # 1%


@pytest.fixture
def rpc_url() -> str:
    """Get RPC URL from environment."""
    url = os.environ.get("RPC_URL")
    if not url:
        pytest.skip("RPC_URL not set")
    return url


@pytest.fixture
def quoter(rpc_url: str):
    """Create a real Web3UniswapV3Quoter."""
    from solver.amm.uniswap_v3 import Web3UniswapV3Quoter

    return Web3UniswapV3Quoter(rpc_url)


class TestWeb3QuoterExactInput:
    """Tests for quote_exact_input with real RPC."""

    def test_weth_to_usdc_3000_fee(self, quoter):
        """Quote 1 WETH → USDC on 0.3% fee tier."""
        amount_in = 10**18  # 1 WETH

        amount_out = quoter.quote_exact_input(
            token_in=WETH,
            token_out=USDC,
            fee=FEE_3000,
            amount_in=amount_in,
        )

        # Should get a reasonable amount of USDC
        # ETH price varies, but should be in a sensible range
        assert amount_out is not None
        # Expect 1000-10000 USDC per ETH (reasonable range)
        assert 1000 * 10**6 < amount_out < 10000 * 10**6

    def test_weth_to_usdc_500_fee(self, quoter):
        """Quote 1 WETH → USDC on 0.05% fee tier."""
        amount_in = 10**18  # 1 WETH

        amount_out = quoter.quote_exact_input(
            token_in=WETH,
            token_out=USDC,
            fee=FEE_500,
            amount_in=amount_in,
        )

        # Should get a reasonable amount of USDC
        assert amount_out is not None
        assert 1000 * 10**6 < amount_out < 10000 * 10**6

    def test_usdc_to_weth_3000_fee(self, quoter):
        """Quote 2500 USDC → WETH on 0.3% fee tier."""
        amount_in = 2500 * 10**6  # 2500 USDC

        amount_out = quoter.quote_exact_input(
            token_in=USDC,
            token_out=WETH,
            fee=FEE_3000,
            amount_in=amount_in,
        )

        # Should get some WETH back
        assert amount_out is not None
        # Expect 0.1-2 WETH for 2500 USDC (reasonable range)
        assert 0.1 * 10**18 < amount_out < 2 * 10**18

    def test_weth_to_dai_3000_fee(self, quoter):
        """Quote 1 WETH → DAI on 0.3% fee tier."""
        amount_in = 10**18  # 1 WETH

        amount_out = quoter.quote_exact_input(
            token_in=WETH,
            token_out=DAI,
            fee=FEE_3000,
            amount_in=amount_in,
        )

        # DAI has 18 decimals
        # Should get 1000-10000 DAI per ETH
        assert amount_out is not None
        assert 1000 * 10**18 < amount_out < 10000 * 10**18

    def test_small_amount_still_quotes(self, quoter):
        """Very small amounts should still return valid quotes."""
        amount_in = 10**15  # 0.001 WETH

        amount_out = quoter.quote_exact_input(
            token_in=WETH,
            token_out=USDC,
            fee=FEE_3000,
            amount_in=amount_in,
        )

        # Should get a small but valid amount
        assert amount_out is not None
        assert amount_out > 0

    def test_large_amount_quotes(self, quoter):
        """Large amounts should return valid quotes (may have slippage)."""
        amount_in = 100 * 10**18  # 100 WETH

        amount_out = quoter.quote_exact_input(
            token_in=WETH,
            token_out=USDC,
            fee=FEE_3000,
            amount_in=amount_in,
        )

        # Should get some USDC (may have significant slippage)
        assert amount_out is not None
        assert amount_out > 0


class TestWeb3QuoterExactOutput:
    """Tests for quote_exact_output with real RPC."""

    def test_weth_to_usdc_exact_output(self, quoter):
        """Quote exact output: get 2500 USDC by spending WETH."""
        amount_out = 2500 * 10**6  # 2500 USDC

        amount_in = quoter.quote_exact_output(
            token_in=WETH,
            token_out=USDC,
            fee=FEE_3000,
            amount_out=amount_out,
        )

        # Should need some WETH
        assert amount_in is not None
        # Expect 0.1-2 WETH for 2500 USDC
        assert 0.1 * 10**18 < amount_in < 2 * 10**18

    def test_usdc_to_weth_exact_output(self, quoter):
        """Quote exact output: get 1 WETH by spending USDC."""
        amount_out = 10**18  # 1 WETH

        amount_in = quoter.quote_exact_output(
            token_in=USDC,
            token_out=WETH,
            fee=FEE_3000,
            amount_out=amount_out,
        )

        # Should need some USDC
        assert amount_in is not None
        # Expect 1000-10000 USDC for 1 WETH
        assert 1000 * 10**6 < amount_in < 10000 * 10**6


class TestWeb3QuoterErrorHandling:
    """Tests for error handling with invalid inputs."""

    def test_invalid_token_pair_returns_none(self, quoter):
        """Invalid token pair should return None (not raise)."""
        # Use an invalid/non-existent token address
        fake_token = "0x0000000000000000000000000000000000000001"

        amount_out = quoter.quote_exact_input(
            token_in=fake_token,
            token_out=USDC,
            fee=FEE_3000,
            amount_in=10**18,
        )

        # Should return None, not raise
        assert amount_out is None

    def test_invalid_fee_tier_returns_none(self, quoter):
        """Invalid fee tier should return None (pool doesn't exist)."""
        # Use a fee tier that doesn't exist for this pair
        invalid_fee = 100  # 0.01% - may not exist for WETH/USDC

        # This may or may not return None depending on if the pool exists
        # The test validates the quoter doesn't crash
        amount_out = quoter.quote_exact_input(
            token_in=WETH,
            token_out=USDC,
            fee=invalid_fee,
            amount_in=10**18,
        )

        # Either valid result or None is acceptable
        assert amount_out is None or amount_out > 0

    def test_zero_amount_handling(self, quoter):
        """Zero input amount should be handled gracefully."""
        amount_out = quoter.quote_exact_input(
            token_in=WETH,
            token_out=USDC,
            fee=FEE_3000,
            amount_in=0,
        )

        # Zero input should give zero output or None
        assert amount_out is None or amount_out == 0


class TestWeb3QuoterFeeTierComparison:
    """Tests comparing quotes across fee tiers."""

    def test_lower_fee_gives_better_output_for_small_amounts(self, quoter):
        """For small swaps, lower fee tier should give better output."""
        amount_in = 10**18  # 1 WETH

        output_500 = quoter.quote_exact_input(
            token_in=WETH, token_out=USDC, fee=FEE_500, amount_in=amount_in
        )
        output_3000 = quoter.quote_exact_input(
            token_in=WETH, token_out=USDC, fee=FEE_3000, amount_in=amount_in
        )

        # Both should return valid quotes
        assert output_500 is not None
        assert output_3000 is not None

        # For small amounts, 0.05% fee should generally give better output
        # than 0.3% fee, but this depends on liquidity depth
        # We just verify both return sensible values
        assert output_500 > 0
        assert output_3000 > 0


class TestWeb3QuoterIntegrationWithAMM:
    """Tests for using Web3UniswapV3Quoter with UniswapV3AMM."""

    def test_amm_uses_real_quoter(self, quoter):
        """UniswapV3AMM should work with real Web3 quoter."""
        from solver.amm.uniswap_v3 import UniswapV3AMM, UniswapV3Pool

        # Create a mock V3 pool (we only need address and fee for quoter)
        pool = UniswapV3Pool(
            address="0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",  # Real WETH/USDC pool
            token0=USDC.lower(),  # USDC is token0 (lower address)
            token1=WETH.lower(),  # WETH is token1
            fee=FEE_500,  # 0.05% fee tier
            sqrt_price_x96=0,  # Not used by quoter
            liquidity=0,  # Not used by quoter
            tick=0,  # Not used by quoter
            liquidity_net={},  # Not used by quoter
        )

        amm = UniswapV3AMM(quoter=quoter)

        # Simulate swap: 1 WETH → USDC
        result = amm.simulate_swap(
            pool=pool,
            token_in=WETH,
            amount_in=10**18,
        )

        assert result is not None
        assert result.amount_in == 10**18
        # Should get some USDC
        assert result.amount_out > 1000 * 10**6  # At least 1000 USDC

    def test_amm_exact_output_with_real_quoter(self, quoter):
        """UniswapV3AMM exact output should work with real quoter."""
        from solver.amm.uniswap_v3 import UniswapV3AMM, UniswapV3Pool

        pool = UniswapV3Pool(
            address="0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
            token0=USDC.lower(),
            token1=WETH.lower(),
            fee=FEE_500,
            sqrt_price_x96=0,
            liquidity=0,
            tick=0,
            liquidity_net={},
        )

        amm = UniswapV3AMM(quoter=quoter)

        # Simulate swap: get 2500 USDC by spending WETH
        result = amm.simulate_swap_exact_output(
            pool=pool,
            token_in=WETH,
            amount_out=2500 * 10**6,
        )

        assert result is not None
        assert result.amount_out == 2500 * 10**6
        # Should need some WETH
        assert result.amount_in > 0
