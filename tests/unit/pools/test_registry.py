"""Tests for PoolRegistry."""

from decimal import Decimal

import pytest

from solver.amm.balancer import (
    BalancerStablePool,
    BalancerWeightedPool,
    StableTokenReserve,
    WeightedTokenReserve,
)
from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.amm.uniswap_v3 import UniswapV3Pool
from solver.pools import PoolRegistry
from solver.pools.limit_order import LimitOrderPool


# Test fixtures
@pytest.fixture
def token_a() -> str:
    return "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1"


@pytest.fixture
def token_b() -> str:
    return "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2"


@pytest.fixture
def v2_pool(token_a: str, token_b: str) -> UniswapV2Pool:
    return UniswapV2Pool(
        address="0x1111111111111111111111111111111111111111",
        token0=token_a,
        token1=token_b,
        reserve0=1000 * 10**18,
        reserve1=2000 * 10**18,
        gas_estimate=100000,
    )


@pytest.fixture
def v3_pool(token_a: str, token_b: str) -> UniswapV3Pool:
    return UniswapV3Pool(
        address="0x2222222222222222222222222222222222222222",
        token0=token_a,
        token1=token_b,
        fee=3000,
        sqrt_price_x96=2**96,
        liquidity=1000000,
        tick=0,
        gas_estimate=150000,
    )


@pytest.fixture
def weighted_pool(token_a: str, token_b: str) -> BalancerWeightedPool:
    return BalancerWeightedPool(
        id="0x3333333333333333333333333333333333333333",
        address="0x3333333333333333333333333333333333333333",
        pool_id="0x3333333333333333333333333333333333333333000000000000000000000001",
        reserves=(
            WeightedTokenReserve(
                token=token_a,
                balance=1000 * 10**18,
                weight=Decimal("0.5"),
                scaling_factor=10**18,
            ),
            WeightedTokenReserve(
                token=token_b,
                balance=2000 * 10**18,
                weight=Decimal("0.5"),
                scaling_factor=10**18,
            ),
        ),
        fee=Decimal("0.003"),
        version="v3Plus",
        gas_estimate=120000,
    )


@pytest.fixture
def stable_pool(token_a: str, token_b: str) -> BalancerStablePool:
    return BalancerStablePool(
        id="0x4444444444444444444444444444444444444444",
        address="0x4444444444444444444444444444444444444444",
        pool_id="0x4444444444444444444444444444444444444444000000000000000000000002",
        reserves=(
            StableTokenReserve(
                token=token_a,
                balance=1000 * 10**18,
                scaling_factor=10**18,
            ),
            StableTokenReserve(
                token=token_b,
                balance=1000 * 10**18,
                scaling_factor=10**18,
            ),
        ),
        amplification_parameter=Decimal("200"),
        fee=Decimal("0.0004"),
        gas_estimate=130000,
    )


@pytest.fixture
def limit_order(token_a: str, token_b: str) -> LimitOrderPool:
    return LimitOrderPool(
        id="0x5555555555555555555555555555555555555555",
        address="0x6666666666666666666666666666666666666666",
        maker_token=token_b,
        taker_token=token_a,
        maker_amount=1000 * 10**18,
        taker_amount=500 * 10**18,
        taker_token_fee_amount=0,
        gas_estimate=80000,
    )


class TestAddAnyPool:
    """Tests for add_any_pool method."""

    def test_add_v2_pool(self, v2_pool: UniswapV2Pool, token_a: str, token_b: str) -> None:
        registry = PoolRegistry()
        registry.add_any_pool(v2_pool)

        assert registry.pool_count == 1
        assert registry.get_pool(token_a, token_b) == v2_pool

    def test_add_v3_pool(self, v3_pool: UniswapV3Pool, token_a: str, token_b: str) -> None:
        registry = PoolRegistry()
        registry.add_any_pool(v3_pool)

        assert registry.v3_pool_count == 1
        pools = registry.get_v3_pools(token_a, token_b)
        assert len(pools) == 1
        assert pools[0] == v3_pool

    def test_add_weighted_pool(
        self, weighted_pool: BalancerWeightedPool, token_a: str, token_b: str
    ) -> None:
        registry = PoolRegistry()
        registry.add_any_pool(weighted_pool)

        assert registry.weighted_pool_count == 1
        pools = registry.get_weighted_pools(token_a, token_b)
        assert len(pools) == 1
        assert pools[0] == weighted_pool

    def test_add_stable_pool(
        self, stable_pool: BalancerStablePool, token_a: str, token_b: str
    ) -> None:
        registry = PoolRegistry()
        registry.add_any_pool(stable_pool)

        assert registry.stable_pool_count == 1
        pools = registry.get_stable_pools(token_a, token_b)
        assert len(pools) == 1
        assert pools[0] == stable_pool

    def test_add_limit_order(self, limit_order: LimitOrderPool, token_a: str, token_b: str) -> None:
        registry = PoolRegistry()
        registry.add_any_pool(limit_order)

        assert registry.limit_order_count == 1
        # Limit orders are directional: taker_token -> maker_token
        orders = registry.get_limit_orders(token_a, token_b)
        assert len(orders) == 1
        assert orders[0] == limit_order

    def test_add_unknown_type_raises(self) -> None:
        registry = PoolRegistry()

        with pytest.raises(TypeError, match="Unknown pool type"):
            registry.add_any_pool("not a pool")  # type: ignore[arg-type]


class TestConstructorWithMixedPools:
    """Tests for constructor accepting mixed pool types."""

    def test_empty_constructor(self) -> None:
        registry = PoolRegistry()
        assert registry.pool_count == 0
        assert registry.v3_pool_count == 0
        assert registry.weighted_pool_count == 0
        assert registry.stable_pool_count == 0
        assert registry.limit_order_count == 0

    def test_constructor_with_v2_pools(self, v2_pool: UniswapV2Pool) -> None:
        registry = PoolRegistry(pools=[v2_pool])
        assert registry.pool_count == 1

    def test_constructor_with_mixed_pools(
        self,
        v2_pool: UniswapV2Pool,
        v3_pool: UniswapV3Pool,
        weighted_pool: BalancerWeightedPool,
        stable_pool: BalancerStablePool,
        limit_order: LimitOrderPool,
    ) -> None:
        registry = PoolRegistry(pools=[v2_pool, v3_pool, weighted_pool, stable_pool, limit_order])

        assert registry.pool_count == 1
        assert registry.v3_pool_count == 1
        assert registry.weighted_pool_count == 1
        assert registry.stable_pool_count == 1
        assert registry.limit_order_count == 1

    def test_constructor_with_multiple_v3_fee_tiers(self, token_a: str, token_b: str) -> None:
        pools = [
            UniswapV3Pool(
                address=f"0x{i}111111111111111111111111111111111111111",
                token0=token_a,
                token1=token_b,
                fee=fee,
                sqrt_price_x96=2**96,
                liquidity=1000000,
                tick=0,
                gas_estimate=150000,
            )
            for i, fee in enumerate([500, 3000, 10000])
        ]

        registry = PoolRegistry(pools=pools)
        assert registry.v3_pool_count == 3


class TestGetPoolsForPair:
    """Tests for get_pools_for_pair returning all pool types."""

    def test_returns_all_pool_types(
        self,
        v2_pool: UniswapV2Pool,
        v3_pool: UniswapV3Pool,
        weighted_pool: BalancerWeightedPool,
        stable_pool: BalancerStablePool,
        limit_order: LimitOrderPool,
        token_a: str,
        token_b: str,
    ) -> None:
        registry = PoolRegistry(pools=[v2_pool, v3_pool, weighted_pool, stable_pool, limit_order])

        pools = registry.get_pools_for_pair(token_a, token_b)

        # Should include all 5 pools
        assert len(pools) == 5
        assert v2_pool in pools
        assert v3_pool in pools
        assert weighted_pool in pools
        assert stable_pool in pools
        assert limit_order in pools
