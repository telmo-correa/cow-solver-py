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
