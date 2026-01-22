"""Unit tests for UniswapV3 AMM implementation."""

import json
from pathlib import Path

import pytest

from solver.amm.uniswap_v3 import (
    QUOTER_V2_ADDRESS,
    SWAP_ROUTER_V2_ADDRESS,
    V3_FEE_LOW,
    V3_FEE_MEDIUM,
    V3_SWAP_GAS_COST,
    UniswapV3Pool,
    parse_v3_liquidity,
)
from solver.models.auction import Liquidity

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "liquidity"


def load_liquidity_fixture(name: str) -> Liquidity:
    """Load a liquidity fixture and parse as Liquidity model."""
    fixture_path = FIXTURES_DIR / name
    with open(fixture_path) as f:
        data = json.load(f)
    return Liquidity(**data)


class TestUniswapV3Pool:
    """Tests for UniswapV3Pool dataclass."""

    def test_pool_creation(self):
        """Test basic pool creation."""
        pool = UniswapV3Pool(
            address="0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
            token0="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            token1="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            fee=3000,
            sqrt_price_x96=1234567890,
            liquidity=1000000000,
            tick=-202000,
        )

        assert pool.fee == 3000
        assert pool.fee_percent == 0.3
        assert pool.fee_decimal == 0.003
        assert pool.tick_spacing == 60

    def test_fee_properties(self):
        """Test fee conversion properties."""
        # 0.3% fee
        pool = UniswapV3Pool(
            address="0x1234",
            token0="0xAAAA",
            token1="0xBBBB",
            fee=3000,
            sqrt_price_x96=0,
            liquidity=0,
            tick=0,
        )
        assert pool.fee_percent == 0.3
        assert pool.fee_decimal == 0.003

        # 0.05% fee
        pool.fee = 500
        assert pool.fee_percent == 0.05
        assert pool.fee_decimal == 0.0005

        # 1% fee
        pool.fee = 10000
        assert pool.fee_percent == 1.0
        assert pool.fee_decimal == 0.01

    def test_tick_spacing(self):
        """Test tick spacing for different fee tiers."""
        pool = UniswapV3Pool(
            address="0x1234",
            token0="0xAAAA",
            token1="0xBBBB",
            fee=100,
            sqrt_price_x96=0,
            liquidity=0,
            tick=0,
        )
        assert pool.tick_spacing == 1

        pool.fee = 500
        assert pool.tick_spacing == 10

        pool.fee = 3000
        assert pool.tick_spacing == 60

        pool.fee = 10000
        assert pool.tick_spacing == 200

    def test_get_token_out(self):
        """Test getting output token."""
        pool = UniswapV3Pool(
            address="0x1234",
            token0="0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            token1="0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            fee=3000,
            sqrt_price_x96=0,
            liquidity=0,
            tick=0,
        )

        # token0 in -> token1 out
        assert (
            pool.get_token_out("0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA").lower()
            == pool.token1.lower()
        )
        # Case insensitive
        assert (
            pool.get_token_out("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").lower()
            == pool.token1.lower()
        )

        # token1 in -> token0 out
        assert (
            pool.get_token_out("0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB").lower()
            == pool.token0.lower()
        )

    def test_get_token_out_invalid_token(self):
        """Test error on invalid token."""
        pool = UniswapV3Pool(
            address="0x1234",
            token0="0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            token1="0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            fee=3000,
            sqrt_price_x96=0,
            liquidity=0,
            tick=0,
        )

        with pytest.raises(ValueError, match="not in pool"):
            pool.get_token_out("0xCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")

    def test_is_token0(self):
        """Test token0 check."""
        pool = UniswapV3Pool(
            address="0x1234",
            token0="0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            token1="0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            fee=3000,
            sqrt_price_x96=0,
            liquidity=0,
            tick=0,
        )

        assert pool.is_token0("0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA") is True
        assert pool.is_token0("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") is True
        assert pool.is_token0("0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB") is False

    def test_default_values(self):
        """Test default values."""
        pool = UniswapV3Pool(
            address="0x1234",
            token0="0xAAAA",
            token1="0xBBBB",
            fee=3000,
            sqrt_price_x96=0,
            liquidity=0,
            tick=0,
        )

        assert pool.router == SWAP_ROUTER_V2_ADDRESS
        assert pool.liquidity_id is None
        assert pool.gas_estimate == V3_SWAP_GAS_COST
        assert pool.liquidity_net == {}


class TestParseV3Liquidity:
    """Tests for parse_v3_liquidity function."""

    def test_parse_basic_pool(self):
        """Test parsing a basic V3 pool from fixture."""
        liquidity = load_liquidity_fixture("v3_pool_basic.json")
        pool = parse_v3_liquidity(liquidity)

        assert pool is not None
        assert pool.address == "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        assert pool.fee == 3000  # 0.003 * 1_000_000
        assert pool.sqrt_price_x96 == 1234567890123456789012345678
        assert pool.liquidity == 12345678901234567890
        assert pool.tick == -202000
        assert pool.liquidity_id == "v3-pool-weth-usdc-3000"
        assert pool.router == "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45"
        assert pool.gas_estimate == 106000

    def test_parse_liquidity_net(self):
        """Test parsing liquidityNet from fixture."""
        liquidity = load_liquidity_fixture("v3_pool_basic.json")
        pool = parse_v3_liquidity(liquidity)

        assert pool is not None
        assert len(pool.liquidity_net) == 4
        assert pool.liquidity_net[-887220] == 1000000000000000000
        assert pool.liquidity_net[-202200] == 500000000000000000
        assert pool.liquidity_net[-201800] == -500000000000000000
        assert pool.liquidity_net[887220] == -1000000000000000000

    def test_parse_low_fee_pool(self):
        """Test parsing pool with 0.05% fee."""
        liquidity = load_liquidity_fixture("v3_pool_low_fee.json")
        pool = parse_v3_liquidity(liquidity)

        assert pool is not None
        assert pool.fee == 500  # 0.0005 * 1_000_000
        assert pool.fee_percent == 0.05
        assert pool.liquidity_net == {}  # Empty in fixture

    def test_parse_fee_decimal_format(self):
        """Test parsing fee from decimal string."""
        liquidity = Liquidity(
            id="test",
            kind="concentratedLiquidity",
            tokens=[
                "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            ],
            address="0x1234567890123456789012345678901234567890",
            fee="0.003",  # Decimal format
        )
        pool = parse_v3_liquidity(liquidity)

        assert pool is not None
        assert pool.fee == 3000

    def test_parse_fee_integer_format(self):
        """Test parsing fee from integer string."""
        liquidity = Liquidity(
            id="test",
            kind="concentratedLiquidity",
            tokens=[
                "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            ],
            address="0x1234567890123456789012345678901234567890",
            fee="3000",  # Integer format (already in Uniswap units)
        )
        pool = parse_v3_liquidity(liquidity)

        assert pool is not None
        assert pool.fee == 3000

    def test_token_ordering(self):
        """Test that tokens are ordered correctly (lower address first)."""
        # Create liquidity with tokens in "wrong" order
        # 0xBB... < 0xCC... in bytes, so should be reordered
        liquidity = Liquidity(
            id="test",
            kind="concentratedLiquidity",
            tokens=[
                "0xCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",  # Higher address first
                "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",  # Lower address second
            ],
            address="0x1234567890123456789012345678901234567890",
            fee="0.003",
        )
        pool = parse_v3_liquidity(liquidity)

        assert pool is not None
        # Token0 should be the lower address
        assert pool.token0.lower() == "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        assert pool.token1.lower() == "0xcccccccccccccccccccccccccccccccccccccccc"

    def test_ignore_non_v3_liquidity(self):
        """Test that non-V3 liquidity is ignored."""
        liquidity = Liquidity(
            id="test",
            kind="constantProduct",  # V2 pool
            tokens=[
                "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            ],
            address="0x1234567890123456789012345678901234567890",
        )
        pool = parse_v3_liquidity(liquidity)
        assert pool is None

    def test_ignore_missing_address(self):
        """Test that liquidity without address is ignored."""
        liquidity = Liquidity(
            id="test",
            kind="concentratedLiquidity",
            tokens=[
                "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            ],
            # No address
        )
        pool = parse_v3_liquidity(liquidity)
        assert pool is None

    def test_ignore_wrong_token_count(self):
        """Test that liquidity with wrong token count is ignored."""
        # 3 tokens
        liquidity = Liquidity(
            id="test",
            kind="concentratedLiquidity",
            tokens=[
                "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
                "0xCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            ],
            address="0x1234567890123456789012345678901234567890",
        )
        pool = parse_v3_liquidity(liquidity)
        assert pool is None

        # 1 token
        liquidity = Liquidity(
            id="test",
            kind="concentratedLiquidity",
            tokens=["0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"],
            address="0x1234567890123456789012345678901234567890",
        )
        pool = parse_v3_liquidity(liquidity)
        assert pool is None

    def test_default_fee_when_missing(self):
        """Test that default fee is used when not provided."""
        liquidity = Liquidity(
            id="test",
            kind="concentratedLiquidity",
            tokens=[
                "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            ],
            address="0x1234567890123456789012345678901234567890",
            # No fee
        )
        pool = parse_v3_liquidity(liquidity)

        assert pool is not None
        assert pool.fee == V3_FEE_MEDIUM  # Default 0.3%

    def test_default_router_when_missing(self):
        """Test that default router is used when not provided."""
        liquidity = Liquidity(
            id="test",
            kind="concentratedLiquidity",
            tokens=[
                "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            ],
            address="0x1234567890123456789012345678901234567890",
            # No router
        )
        pool = parse_v3_liquidity(liquidity)

        assert pool is not None
        assert pool.router == SWAP_ROUTER_V2_ADDRESS

    def test_default_values_for_v3_fields(self):
        """Test default values for V3-specific fields."""
        liquidity = Liquidity(
            id="test",
            kind="concentratedLiquidity",
            tokens=[
                "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            ],
            address="0x1234567890123456789012345678901234567890",
            # No sqrtPrice, liquidity, tick, liquidityNet
        )
        pool = parse_v3_liquidity(liquidity)

        assert pool is not None
        assert pool.sqrt_price_x96 == 0
        assert pool.liquidity == 0
        assert pool.tick == 0
        assert pool.liquidity_net == {}


class TestV3Constants:
    """Tests for V3 constants."""

    def test_fee_tiers(self):
        """Test fee tier constants are correct."""
        from solver.amm.uniswap_v3 import (
            V3_FEE_HIGH,
            V3_FEE_LOWEST,
            V3_FEE_MEDIUM,
            V3_FEE_TIERS,
        )

        assert V3_FEE_LOWEST == 100
        assert V3_FEE_LOW == 500
        assert V3_FEE_MEDIUM == 3000
        assert V3_FEE_HIGH == 10000
        assert V3_FEE_TIERS == [100, 500, 3000, 10000]

    def test_contract_addresses(self):
        """Test contract addresses are correct."""
        assert QUOTER_V2_ADDRESS == "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
        assert SWAP_ROUTER_V2_ADDRESS == "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"

    def test_gas_cost(self):
        """Test gas cost constant."""
        assert V3_SWAP_GAS_COST == 106_000


class TestMockUniswapV3Quoter:
    """Tests for MockUniswapV3Quoter."""

    def test_returns_configured_exact_input_quote(self):
        """Test mock quoter returns configured value for exact input."""
        from solver.amm.uniswap_v3 import MockUniswapV3Quoter, QuoteKey

        token_in = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # WETH
        token_out = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # USDC
        fee = 3000
        amount_in = 10**18  # 1 WETH

        key = QuoteKey(token_in, token_out, fee, amount_in, is_exact_input=True)
        expected_out = 3000 * 10**6  # 3000 USDC

        quoter = MockUniswapV3Quoter(quotes={key: expected_out})
        result = quoter.quote_exact_input(token_in, token_out, fee, amount_in)

        assert result == expected_out

    def test_returns_configured_exact_output_quote(self):
        """Test mock quoter returns configured value for exact output."""
        from solver.amm.uniswap_v3 import MockUniswapV3Quoter, QuoteKey

        token_in = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # WETH
        token_out = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # USDC
        fee = 3000
        amount_out = 3000 * 10**6  # 3000 USDC

        key = QuoteKey(token_in, token_out, fee, amount_out, is_exact_input=False)
        expected_in = 10**18  # 1 WETH

        quoter = MockUniswapV3Quoter(quotes={key: expected_in})
        result = quoter.quote_exact_output(token_in, token_out, fee, amount_out)

        assert result == expected_in

    def test_tracks_calls(self):
        """Test mock quoter tracks all calls."""
        from solver.amm.uniswap_v3 import MockUniswapV3Quoter

        quoter = MockUniswapV3Quoter(default_rate=3000.0)

        token_in = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        token_out = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"

        quoter.quote_exact_input(token_in, token_out, 3000, 10**18)
        quoter.quote_exact_output(token_in, token_out, 500, 2000 * 10**6)

        assert len(quoter.calls) == 2
        assert quoter.calls[0] == ("exact_input", token_in, token_out, 3000, 10**18)
        assert quoter.calls[1] == ("exact_output", token_in, token_out, 500, 2000 * 10**6)

    def test_returns_none_for_unknown_quote(self):
        """Test mock quoter returns None for unconfigured quotes."""
        from solver.amm.uniswap_v3 import MockUniswapV3Quoter

        quoter = MockUniswapV3Quoter()  # No quotes configured, no default_rate

        result = quoter.quote_exact_input("0xAAAA", "0xBBBB", 3000, 10**18)

        assert result is None

    def test_uses_default_rate_for_exact_input(self):
        """Test mock quoter uses default_rate for unconfigured exact input quotes."""
        from solver.amm.uniswap_v3 import MockUniswapV3Quoter

        quoter = MockUniswapV3Quoter(default_rate=2500.0)  # 1 token_in = 2500 token_out

        amount_in = 2 * 10**18
        result = quoter.quote_exact_input("0xAAAA", "0xBBBB", 3000, amount_in)

        assert result == int(amount_in * 2500.0)

    def test_uses_default_rate_for_exact_output(self):
        """Test mock quoter uses default_rate for unconfigured exact output quotes."""
        from solver.amm.uniswap_v3 import MockUniswapV3Quoter

        quoter = MockUniswapV3Quoter(default_rate=2500.0)  # 1 token_in = 2500 token_out

        amount_out = 5000 * 10**6
        result = quoter.quote_exact_output("0xAAAA", "0xBBBB", 3000, amount_out)

        # amount_in = amount_out / rate
        assert result == int(amount_out / 2500.0)

    def test_quote_key_address_normalization(self):
        """Test QuoteKey normalizes addresses for comparison."""
        from solver.amm.uniswap_v3 import MockUniswapV3Quoter, QuoteKey

        # Use uppercase address in config
        key = QuoteKey(
            "0xC02AAA39B223FE8D0A0E5C4F27EAD9083C756CC2",
            "0xA0B86991C6218B36C1D19D4A2E9EB0CE3606EB48",
            3000,
            10**18,
            is_exact_input=True,
        )
        quoter = MockUniswapV3Quoter(quotes={key: 3000 * 10**6})

        # Query with lowercase address
        result = quoter.quote_exact_input(
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            3000,
            10**18,
        )

        assert result == 3000 * 10**6


class TestQuoterV2ABI:
    """Tests for QuoterV2 ABI."""

    def test_abi_has_required_functions(self):
        """Test that ABI includes required functions."""
        from solver.amm.uniswap_v3 import QUOTER_V2_ABI

        function_names = {f["name"] for f in QUOTER_V2_ABI}
        assert "quoteExactInputSingle" in function_names
        assert "quoteExactOutputSingle" in function_names

    def test_exact_input_single_params(self):
        """Test quoteExactInputSingle has correct parameters."""
        from solver.amm.uniswap_v3 import QUOTER_V2_ABI

        func = next(f for f in QUOTER_V2_ABI if f["name"] == "quoteExactInputSingle")

        # Should have tuple input with correct components
        assert len(func["inputs"]) == 1
        assert func["inputs"][0]["type"] == "tuple"

        component_names = {c["name"] for c in func["inputs"][0]["components"]}
        assert "tokenIn" in component_names
        assert "tokenOut" in component_names
        assert "amountIn" in component_names
        assert "fee" in component_names
        assert "sqrtPriceLimitX96" in component_names

    def test_exact_output_single_params(self):
        """Test quoteExactOutputSingle has correct parameters."""
        from solver.amm.uniswap_v3 import QUOTER_V2_ABI

        func = next(f for f in QUOTER_V2_ABI if f["name"] == "quoteExactOutputSingle")

        # Should have tuple input with correct components
        assert len(func["inputs"]) == 1
        assert func["inputs"][0]["type"] == "tuple"

        component_names = {c["name"] for c in func["inputs"][0]["components"]}
        assert "tokenIn" in component_names
        assert "tokenOut" in component_names
        assert "amount" in component_names  # Note: "amount" not "amountOut"
        assert "fee" in component_names
        assert "sqrtPriceLimitX96" in component_names


class TestSwapRouterEncoding:
    """Tests for SwapRouterV2 calldata encoding."""

    def test_exact_input_single_selector(self):
        """Test exactInputSingle function selector is correct."""
        from solver.amm.uniswap_v3 import EXACT_INPUT_SINGLE_SELECTOR

        # Selector for exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))
        assert bytes.fromhex("04e45aaf") == EXACT_INPUT_SINGLE_SELECTOR

    def test_exact_output_single_selector(self):
        """Test exactOutputSingle function selector is correct."""
        from solver.amm.uniswap_v3 import EXACT_OUTPUT_SINGLE_SELECTOR

        # Selector for exactOutputSingle((address,address,uint24,address,uint256,uint256,uint160))
        assert bytes.fromhex("5023b4df") == EXACT_OUTPUT_SINGLE_SELECTOR

    def test_encode_exact_input_single_returns_router_address(self):
        """Test encode_exact_input_single returns correct router address."""
        from solver.amm.uniswap_v3 import (
            SWAP_ROUTER_V2_ADDRESS,
            encode_exact_input_single,
        )

        router, calldata = encode_exact_input_single(
            token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            fee=3000,
            recipient="0x9008D19f58AAbD9eD0D60971565AA8510560ab41",
            amount_in=10**18,
            amount_out_minimum=2900 * 10**6,
        )

        assert router == SWAP_ROUTER_V2_ADDRESS

    def test_encode_exact_input_single_calldata_starts_with_selector(self):
        """Test encode_exact_input_single calldata starts with correct selector."""
        from solver.amm.uniswap_v3 import encode_exact_input_single

        _, calldata = encode_exact_input_single(
            token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            fee=3000,
            recipient="0x9008D19f58AAbD9eD0D60971565AA8510560ab41",
            amount_in=10**18,
            amount_out_minimum=2900 * 10**6,
        )

        # Calldata should start with 0x + selector
        assert calldata.startswith("0x04e45aaf")

    def test_encode_exact_input_single_calldata_length(self):
        """Test encode_exact_input_single produces correct calldata length."""
        from solver.amm.uniswap_v3 import encode_exact_input_single

        _, calldata = encode_exact_input_single(
            token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            fee=3000,
            recipient="0x9008D19f58AAbD9eD0D60971565AA8510560ab41",
            amount_in=10**18,
            amount_out_minimum=2900 * 10**6,
        )

        # 4 bytes selector + 7 * 32 bytes for tuple params = 228 bytes = 456 hex chars + 2 for "0x"
        assert len(calldata) == 2 + 4 * 2 + 7 * 32 * 2  # 458

    def test_encode_exact_output_single_returns_router_address(self):
        """Test encode_exact_output_single returns correct router address."""
        from solver.amm.uniswap_v3 import (
            SWAP_ROUTER_V2_ADDRESS,
            encode_exact_output_single,
        )

        router, calldata = encode_exact_output_single(
            token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            fee=3000,
            recipient="0x9008D19f58AAbD9eD0D60971565AA8510560ab41",
            amount_out=3000 * 10**6,
            amount_in_maximum=11 * 10**17,  # 1.1 WETH max
        )

        assert router == SWAP_ROUTER_V2_ADDRESS

    def test_encode_exact_output_single_calldata_starts_with_selector(self):
        """Test encode_exact_output_single calldata starts with correct selector."""
        from solver.amm.uniswap_v3 import encode_exact_output_single

        _, calldata = encode_exact_output_single(
            token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            fee=3000,
            recipient="0x9008D19f58AAbD9eD0D60971565AA8510560ab41",
            amount_out=3000 * 10**6,
            amount_in_maximum=11 * 10**17,
        )

        # Calldata should start with 0x + selector
        assert calldata.startswith("0x5023b4df")

    def test_encode_exact_input_single_address_normalization(self):
        """Test that addresses are normalized in encoding."""
        from solver.amm.uniswap_v3 import encode_exact_input_single

        # Use lowercase addresses
        _, calldata1 = encode_exact_input_single(
            token_in="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            token_out="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            fee=3000,
            recipient="0x9008d19f58aabd9ed0d60971565aa8510560ab41",
            amount_in=10**18,
            amount_out_minimum=2900 * 10**6,
        )

        # Use uppercase addresses
        _, calldata2 = encode_exact_input_single(
            token_in="0xC02AAA39B223FE8D0A0E5C4F27EAD9083C756CC2",
            token_out="0xA0B86991C6218B36C1D19D4A2E9EB0CE3606EB48",
            fee=3000,
            recipient="0x9008D19F58AABD9ED0D60971565AA8510560AB41",
            amount_in=10**18,
            amount_out_minimum=2900 * 10**6,
        )

        # Calldata should be identical (addresses normalized)
        assert calldata1 == calldata2

    def test_encode_exact_input_single_with_price_limit(self):
        """Test encoding with non-zero sqrtPriceLimitX96."""
        from solver.amm.uniswap_v3 import encode_exact_input_single

        _, calldata = encode_exact_input_single(
            token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            fee=3000,
            recipient="0x9008D19f58AAbD9eD0D60971565AA8510560ab41",
            amount_in=10**18,
            amount_out_minimum=2900 * 10**6,
            sqrt_price_limit_x96=79228162514264337593543950336,  # Example price limit
        )

        # Should still be valid calldata
        assert calldata.startswith("0x04e45aaf")
        assert len(calldata) == 458


class TestSwapRouterABI:
    """Tests for SwapRouterV2 ABI."""

    def test_abi_has_required_functions(self):
        """Test that ABI includes required functions."""
        from solver.amm.uniswap_v3 import SWAP_ROUTER_V2_ABI

        function_names = {f["name"] for f in SWAP_ROUTER_V2_ABI}
        assert "exactInputSingle" in function_names
        assert "exactOutputSingle" in function_names

    def test_exact_input_single_abi_params(self):
        """Test exactInputSingle has correct parameters."""
        from solver.amm.uniswap_v3 import SWAP_ROUTER_V2_ABI

        func = next(f for f in SWAP_ROUTER_V2_ABI if f["name"] == "exactInputSingle")

        # Should have tuple input with correct components
        assert len(func["inputs"]) == 1
        assert func["inputs"][0]["type"] == "tuple"

        component_names = {c["name"] for c in func["inputs"][0]["components"]}
        assert "tokenIn" in component_names
        assert "tokenOut" in component_names
        assert "fee" in component_names
        assert "recipient" in component_names
        assert "amountIn" in component_names
        assert "amountOutMinimum" in component_names
        assert "sqrtPriceLimitX96" in component_names

    def test_exact_output_single_abi_params(self):
        """Test exactOutputSingle has correct parameters."""
        from solver.amm.uniswap_v3 import SWAP_ROUTER_V2_ABI

        func = next(f for f in SWAP_ROUTER_V2_ABI if f["name"] == "exactOutputSingle")

        # Should have tuple input with correct components
        assert len(func["inputs"]) == 1
        assert func["inputs"][0]["type"] == "tuple"

        component_names = {c["name"] for c in func["inputs"][0]["components"]}
        assert "tokenIn" in component_names
        assert "tokenOut" in component_names
        assert "fee" in component_names
        assert "recipient" in component_names
        assert "amountOut" in component_names
        assert "amountInMaximum" in component_names
        assert "sqrtPriceLimitX96" in component_names
