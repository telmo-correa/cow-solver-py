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
