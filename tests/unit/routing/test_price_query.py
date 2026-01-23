"""Tests for AMM reference price queries."""

from decimal import Decimal

from solver.amm.uniswap_v2 import UniswapV2, UniswapV2Pool
from solver.pools import PoolRegistry
from solver.routing.router import SingleOrderRouter

# Test token addresses
TOKEN_A = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
TOKEN_B = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"


class TestGetReferencePrice:
    """Tests for get_reference_price method."""

    def test_single_v2_pool(self) -> None:
        """Single V2 pool returns correct price."""
        # Pool with 1000 A and 2000 B -> price = 2 B/A
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=1000 * 10**18,  # 1000 A
            reserve1=2000 * 10**18,  # 2000 B
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price = router.get_reference_price(TOKEN_A, TOKEN_B)

        # Price should be close to 2.0 (reserve_B / reserve_A)
        # Slightly less due to fee: 1 A -> ~1.994 B at 0.3% fee
        assert price is not None
        assert Decimal("1.9") < price < Decimal("2.1")

    def test_reverse_direction(self) -> None:
        """Price query works in both directions."""
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=1000 * 10**18,
            reserve1=2000 * 10**18,
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)

        # A -> B price (~2 B/A)
        price_ab = router.get_reference_price(TOKEN_A, TOKEN_B)
        # B -> A price (~0.5 A/B)
        price_ba = router.get_reference_price(TOKEN_B, TOKEN_A)

        assert price_ab is not None
        assert price_ba is not None
        # They should be approximate inverses
        assert Decimal("0.9") < price_ab * price_ba < Decimal("1.1")

    def test_no_pool_returns_none(self) -> None:
        """Returns None when no pool exists for the pair."""
        registry = PoolRegistry()
        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)

        price = router.get_reference_price(TOKEN_A, TOKEN_B)
        assert price is None

    def test_multiple_pools_returns_best(self) -> None:
        """With multiple pools, returns the best price."""
        # Pool 1: price ~2 B/A
        pool1 = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=1000 * 10**18,
            reserve1=2000 * 10**18,
        )
        # Pool 2: price ~3 B/A (better for selling A)
        pool2 = UniswapV2Pool(
            address="0x2222222222222222222222222222222222222222",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=1000 * 10**18,
            reserve1=3000 * 10**18,
        )
        registry = PoolRegistry()
        registry.add_pool(pool1)
        registry.add_pool(pool2)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price = router.get_reference_price(TOKEN_A, TOKEN_B)

        # Should return the better price (higher output for selling A)
        assert price is not None
        assert Decimal("2.5") < price < Decimal("3.5")

    def test_zero_liquidity_pool_handled(self) -> None:
        """Pools with zero reserves are handled gracefully."""
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=0,  # No liquidity
            reserve1=0,
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price = router.get_reference_price(TOKEN_A, TOKEN_B)

        # Should return None (no usable liquidity)
        assert price is None

    def test_small_probe_amount(self) -> None:
        """Price is accurate for small probe amounts."""
        # Very liquid pool
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=1_000_000 * 10**18,  # 1M A
            reserve1=2_000_000 * 10**18,  # 2M B
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price = router.get_reference_price(TOKEN_A, TOKEN_B)

        # With large liquidity, price should be very close to 2.0 even after fee
        assert price is not None
        # 0.997 * 2 = 1.994 (fee-adjusted)
        assert Decimal("1.99") < price < Decimal("2.01")


class TestGetReferencePriceDecimal:
    """Tests for Decimal precision in price queries."""

    def test_returns_decimal(self) -> None:
        """Price is returned as Decimal for precision."""
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=1000 * 10**18,
            reserve1=2000 * 10**18,
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price = router.get_reference_price(TOKEN_A, TOKEN_B)

        assert price is not None
        assert isinstance(price, Decimal)

    def test_precision_maintained(self) -> None:
        """High precision is maintained in price calculation."""
        # Pool with awkward ratio
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=333 * 10**18,
            reserve1=777 * 10**18,
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price = router.get_reference_price(TOKEN_A, TOKEN_B)

        assert price is not None
        # 777/333 = 2.333... but after fee slightly less
        # The exact value should be preserved in Decimal
        assert Decimal("2.3") < price < Decimal("2.4")


# Token addresses for decimal tests
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # 6 decimals
USDT = "0xdAC17F958D2ee523a2206206994597C13D831ec7"  # 6 decimals
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # 18 decimals
WBTC = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"  # 8 decimals


class TestGetReferencePriceNon18Decimals:
    """Tests for get_reference_price with non-18-decimal tokens.

    These tests verify that price queries work correctly for tokens like
    USDC (6 decimals) and WBTC (8 decimals), not just 18-decimal tokens.

    The key issue: probe_amount must be scaled to the input token's decimals,
    otherwise a 6-decimal token would use probe_amount=10^15 which equals
    1 billion tokens (not 0.001 tokens as intended).
    """

    def test_6_decimal_to_18_decimal(self) -> None:
        """USDC (6 decimals) to WETH (18 decimals) returns correct price.

        Pool: 2,500,000 USDC / 1,000 WETH (price = 2500 USDC/WETH)

        The price is returned in RAW units (raw_WETH / raw_USDC) for consistency
        with order limit prices which are also in raw units. This allows direct
        comparison in CoW matching.

        Human-readable: 1 USDC = 0.0004 WETH
        Raw: 1 raw_USDC = 4e8 raw_WETH (because 0.0004 * 10^(18-6) = 4e8)
        """
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=USDC,
            token1=WETH,
            reserve0=2_500_000 * 10**6,  # 2.5M USDC (6 decimals)
            reserve1=1_000 * 10**18,  # 1000 WETH (18 decimals)
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price = router.get_reference_price(USDC, WETH, token_in_decimals=6)

        # Raw price = human_price * 10^(out_decimals - in_decimals)
        # = 0.0004 * 10^12 = 4e8 (with fee: ~3.988e8)
        assert price is not None, "Price should not be None for valid pool"
        assert Decimal("3e8") < price < Decimal("5e8"), f"Expected ~4e8, got {price}"

    def test_18_decimal_to_6_decimal(self) -> None:
        """WETH (18 decimals) to USDC (6 decimals) returns correct price.

        Pool: 2,500,000 USDC / 1,000 WETH (price = 2500 USDC/WETH)

        Human-readable: 1 WETH = 2500 USDC
        Raw: 1 raw_WETH = 2.5e-9 raw_USDC (because 2500 * 10^(6-18) = 2.5e-9)
        """
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=USDC,
            token1=WETH,
            reserve0=2_500_000 * 10**6,  # 2.5M USDC (6 decimals)
            reserve1=1_000 * 10**18,  # 1000 WETH (18 decimals)
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price = router.get_reference_price(WETH, USDC, token_in_decimals=18)

        # Raw price = human_price * 10^(out_decimals - in_decimals)
        # = 2500 * 10^-12 = 2.5e-9 (with fee: ~2.49e-9)
        assert price is not None, "Price should not be None for valid pool"
        assert Decimal("2e-9") < price < Decimal("3e-9"), f"Expected ~2.5e-9, got {price}"

    def test_6_decimal_to_6_decimal(self) -> None:
        """Both tokens 6 decimals (e.g., USDC/USDT stablecoin pair).

        Pool: 1,000,000 USDC / 1,000,000 USDT (price = 1.0)
        """
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=USDC,
            token1=USDT,
            reserve0=1_000_000 * 10**6,  # 1M USDC (6 decimals)
            reserve1=1_000_000 * 10**6,  # 1M USDT (6 decimals)
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price = router.get_reference_price(USDC, USDT, token_in_decimals=6)

        # Price should be ~1.0 (with fee: ~0.997)
        assert price is not None, "Price should not be None for valid pool"
        assert Decimal("0.99") < price < Decimal("1.01"), f"Expected ~1.0, got {price}"

    def test_8_decimal_to_18_decimal(self) -> None:
        """WBTC (8 decimals) to WETH (18 decimals).

        Pool: 100 WBTC / 1,500 WETH (price = 15 WETH/WBTC)

        Human-readable: 1 WBTC = 15 WETH
        Raw: 1 raw_WBTC = 15e10 raw_WETH (because 15 * 10^(18-8) = 1.5e11)
        """
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=WBTC,
            token1=WETH,
            reserve0=100 * 10**8,  # 100 WBTC (8 decimals)
            reserve1=1_500 * 10**18,  # 1500 WETH (18 decimals)
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price = router.get_reference_price(WBTC, WETH, token_in_decimals=8)

        # Raw price = human_price * 10^(out_decimals - in_decimals)
        # = 15 * 10^10 = 1.5e11 (with fee: ~1.495e11)
        assert price is not None, "Price should not be None for valid pool"
        assert Decimal("1.4e11") < price < Decimal("1.6e11"), f"Expected ~1.5e11, got {price}"

    def test_reverse_direction_mixed_decimals(self) -> None:
        """Prices in both directions should be approximate inverses."""
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=USDC,
            token1=WETH,
            reserve0=2_500_000 * 10**6,  # 2.5M USDC (6 decimals)
            reserve1=1_000 * 10**18,  # 1000 WETH (18 decimals)
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price_usdc_weth = router.get_reference_price(USDC, WETH, token_in_decimals=6)
        price_weth_usdc = router.get_reference_price(WETH, USDC, token_in_decimals=18)

        assert price_usdc_weth is not None, "USDC->WETH price should not be None"
        assert price_weth_usdc is not None, "WETH->USDC price should not be None"

        # Product of prices should be close to 1 (allowing for fees in both directions)
        # Each direction has ~0.3% fee, so product is ~0.994
        product = price_usdc_weth * price_weth_usdc
        assert Decimal("0.98") < product < Decimal("1.02"), f"Expected ~1.0, got {product}"

    def test_small_liquidity_6_decimal(self) -> None:
        """Small pool with 6-decimal tokens should still return valid price.

        This tests that probe amount doesn't exceed pool reserves.
        Pool: 10,000 USDC / 4 WETH (small pool, same ratio as large pool)

        Human-readable: 1 USDC = 0.0004 WETH
        Raw: 1 raw_USDC = 4e8 raw_WETH
        """
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=USDC,
            token1=WETH,
            reserve0=10_000 * 10**6,  # 10K USDC (6 decimals)
            reserve1=4 * 10**18,  # 4 WETH (18 decimals)
        )
        registry = PoolRegistry()
        registry.add_pool(pool)

        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)
        price = router.get_reference_price(USDC, WETH, token_in_decimals=6)

        # Raw price = human_price * 10^(out_decimals - in_decimals)
        # = 0.0004 * 10^12 = 4e8 (with fee: ~3.988e8)
        # With token_in_decimals=6, probe_amount=10^3=0.001 USDC (not 1 billion)
        assert price is not None, "Price should not be None even for small pools"
        assert Decimal("3e8") < price < Decimal("5e8"), f"Expected ~4e8, got {price}"
