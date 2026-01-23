"""Unit tests for mixed pool type routing (V2 + V3 + Balancer combinations)."""

from decimal import Decimal

from solver.amm.balancer import (
    BalancerStablePool,
    BalancerWeightedPool,
    StableTokenReserve,
    WeightedTokenReserve,
)
from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.amm.uniswap_v3 import (
    MockUniswapV3Quoter,
    UniswapV3AMM,
    UniswapV3Pool,
)
from solver.models.auction import Order, OrderClass, OrderKind
from solver.pools import PoolRegistry
from solver.routing.router import SingleOrderRouter


def make_order(
    sell_token: str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    buy_token: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    sell_amount: str = "1000000000000000000",
    buy_amount: str = "2000000000",
    kind: OrderKind = OrderKind.SELL,
) -> Order:
    """Create a test order with sensible defaults."""
    return Order(
        uid="0x" + "01" * 56,
        sellToken=sell_token,
        buyToken=buy_token,
        sellAmount=sell_amount,
        buyAmount=buy_amount,
        kind=kind,
        **{"class": OrderClass.LIMIT},
    )


class TestMixedMultiHopRouting:
    """Tests for multi-hop routing through mixed pool types."""

    # Token addresses for tests
    WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
    WBTC = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"

    def _make_v2_pool(
        self, token0: str, token1: str, reserve0: int, reserve1: int
    ) -> UniswapV2Pool:
        """Create a V2 pool for testing."""
        # Ensure canonical ordering
        if token0.lower() > token1.lower():
            token0, token1 = token1, token0
            reserve0, reserve1 = reserve1, reserve0
        return UniswapV2Pool(
            address=f"0x{token0[-6:]}{token1[-6:]}1111111111111111111111",
            token0=token0.lower(),
            token1=token1.lower(),
            reserve0=reserve0,
            reserve1=reserve1,
            fee_bps=30,
            liquidity_id=f"v2-{token0[-4:]}-{token1[-4:]}",
        )

    def _make_weighted_pool(self, token0: str, token1: str, balance0: int, balance1: int):
        """Create a Balancer weighted pool for testing."""
        return BalancerWeightedPool(
            id=f"weighted-{token0[-4:]}-{token1[-4:]}",
            address=f"0x{token0[-6:]}{token1[-6:]}2222222222222222222222",
            pool_id="0x" + "33" * 32,
            reserves=(
                WeightedTokenReserve(
                    token=token0.lower(),
                    balance=balance0,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
                WeightedTokenReserve(
                    token=token1.lower(),
                    balance=balance1,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.003"),
            version="v3Plus",
            gas_estimate=120_000,
        )

    def _make_stable_pool(self, token0: str, token1: str, balance0: int, balance1: int):
        """Create a Balancer stable pool for testing."""
        return BalancerStablePool(
            id=f"stable-{token0[-4:]}-{token1[-4:]}",
            address=f"0x{token0[-6:]}{token1[-6:]}4444444444444444444444",
            pool_id="0x" + "44" * 32,
            reserves=(
                StableTokenReserve(
                    token=token0.lower(),
                    balance=balance0,
                    scaling_factor=1,
                ),
                StableTokenReserve(
                    token=token1.lower(),
                    balance=balance1,
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.0001"),  # 0.01%
            amplification_parameter=Decimal("5000"),
            gas_estimate=100_000,
        )

    def test_multihop_v2_then_weighted(self):
        """Multi-hop route: V2 pool for hop 1, weighted pool for hop 2."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Create pools: WETH -> DAI (V2), DAI -> USDC (weighted)
        v2_pool = self._make_v2_pool(
            self.WETH,
            self.DAI,
            100_000_000_000_000_000_000_000,  # 100K WETH
            200_000_000_000_000_000_000_000_000,  # 200M DAI
        )
        weighted_pool = self._make_weighted_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI
            100_000_000_000_000,  # 100M USDC (6 decimals)
        )

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_weighted_pool(weighted_pool)

        weighted_amm = BalancerWeightedAMM()
        router = SingleOrderRouter(pool_registry=registry, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",  # Very low min
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # First hop through V2
        assert isinstance(result.hops[0].pool, UniswapV2Pool)
        # Second hop could be weighted (if it gives better quote) or V2
        assert result.amount_out > 0

    def test_multihop_weighted_then_v2(self):
        """Multi-hop route: weighted pool for hop 1, V2 pool for hop 2."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Create pools: WETH -> DAI (weighted), DAI -> USDC (V2)
        weighted_pool = self._make_weighted_pool(
            self.WETH,
            self.DAI,
            100_000_000_000_000_000_000_000,  # 100K WETH
            200_000_000_000_000_000_000_000_000,  # 200M DAI
        )
        v2_pool = self._make_v2_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI (18 decimals)
            100_000_000_000_000,  # 100M USDC (6 decimals)
        )

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_weighted_pool(weighted_pool)

        weighted_amm = BalancerWeightedAMM()
        router = SingleOrderRouter(pool_registry=registry, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        assert result.amount_out > 0

    def test_multihop_selects_best_pool_per_hop(self):
        """Router selects the pool with best quote for each hop."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Create two pools for WETH -> DAI with different rates
        # V2 pool: 1 WETH = 2000 DAI
        v2_pool = self._make_v2_pool(
            self.WETH,
            self.DAI,
            1_000_000_000_000_000_000_000,  # 1K WETH
            2_000_000_000_000_000_000_000_000,  # 2M DAI
        )
        # Weighted pool: 1 WETH = 2500 DAI (better rate)
        weighted_pool = self._make_weighted_pool(
            self.WETH,
            self.DAI,
            1_000_000_000_000_000_000_000,  # 1K WETH
            2_500_000_000_000_000_000_000_000,  # 2.5M DAI
        )
        # Second hop: DAI -> USDC (only V2 available)
        v2_pool_2 = self._make_v2_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI
            100_000_000_000_000,  # 100M USDC
        )

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_pool(v2_pool_2)
        registry.add_weighted_pool(weighted_pool)

        weighted_amm = BalancerWeightedAMM()
        router = SingleOrderRouter(pool_registry=registry, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2

        # First hop should use weighted pool (better rate)
        assert isinstance(result.hops[0].pool, BalancerWeightedPool)
        # Second hop uses V2 (only option)
        assert isinstance(result.hops[1].pool, UniswapV2Pool)

    def test_multihop_three_hops_mixed(self):
        """Multi-hop route with 3 hops through mixed pool types."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Path: WBTC -> WETH (V2) -> DAI (weighted) -> USDC (V2)
        pool1 = self._make_v2_pool(
            self.WBTC,
            self.WETH,
            10_000_000_000,  # 100 WBTC (8 decimals)
            1_500_000_000_000_000_000_000,  # 1.5K WETH
        )
        pool2 = self._make_weighted_pool(
            self.WETH,
            self.DAI,
            50_000_000_000_000_000_000_000,  # 50K WETH
            100_000_000_000_000_000_000_000_000,  # 100M DAI
        )
        pool3 = self._make_v2_pool(
            self.DAI,
            self.USDC,
            50_000_000_000_000_000_000_000_000,  # 50M DAI
            50_000_000_000_000,  # 50M USDC
        )

        registry = PoolRegistry()
        registry.add_pool(pool1)
        registry.add_pool(pool3)
        registry.add_weighted_pool(pool2)

        weighted_amm = BalancerWeightedAMM()
        router = SingleOrderRouter(pool_registry=registry, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WBTC,
            buy_token=self.USDC,
            sell_amount="10000000",  # 0.1 WBTC
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 3
        assert result.amount_out > 0

    def test_multihop_build_solution_mixed_pools(self):
        """build_solution works correctly with mixed pool types in multi-hop."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Create a two-hop path with mixed pools
        v2_pool = self._make_v2_pool(
            self.WETH,
            self.DAI,
            100_000_000_000_000_000_000_000,
            200_000_000_000_000_000_000_000_000,
        )
        weighted_pool = self._make_weighted_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,
            100_000_000_000_000,
        )

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_weighted_pool(weighted_pool)

        weighted_amm = BalancerWeightedAMM()
        router = SingleOrderRouter(pool_registry=registry, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",
            buy_amount="1",
        )
        result = router.route_order(order)
        assert result.success is True

        solution = router.build_solution(result)
        assert solution is not None
        assert len(solution.trades) == 1
        # Should have interactions for each hop
        assert len(solution.interactions) >= 1

    def test_multihop_buy_order_uses_registry_fallback(self):
        """Buy order multi-hop uses registry's default pool selection (fallback)."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Create pools: WETH -> DAI (both V2 and weighted), DAI -> USDC (V2)
        v2_pool_1 = self._make_v2_pool(
            self.WETH,
            self.DAI,
            100_000_000_000_000_000_000_000,  # 100K WETH
            200_000_000_000_000_000_000_000_000,  # 200M DAI
        )
        weighted_pool = self._make_weighted_pool(
            self.WETH,
            self.DAI,
            100_000_000_000_000_000_000_000,  # 100K WETH
            250_000_000_000_000_000_000_000_000,  # 250M DAI (better rate)
        )
        v2_pool_2 = self._make_v2_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI
            100_000_000_000_000,  # 100M USDC
        )

        registry = PoolRegistry()
        registry.add_pool(v2_pool_1)
        registry.add_pool(v2_pool_2)
        registry.add_weighted_pool(weighted_pool)

        weighted_amm = BalancerWeightedAMM()
        router = SingleOrderRouter(pool_registry=registry, weighted_amm=weighted_amm)

        # Buy order: want to receive USDC, willing to spend WETH
        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="10000000000000000000",  # 10 WETH max
            buy_amount="1000000000",  # Want 1000 USDC
            kind=OrderKind.BUY,
        )
        result = router.route_order(order)

        # Buy order multi-hop should succeed using registry's fallback selection
        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # Registry fallback prioritizes V2 pools
        assert isinstance(result.hops[0].pool, UniswapV2Pool)
        assert isinstance(result.hops[1].pool, UniswapV2Pool)

    def test_multihop_v2_then_stable(self):
        """Multi-hop route: V2 pool for hop 1, stable pool for hop 2."""
        from solver.amm.balancer import BalancerStableAMM

        # Create pools: WETH -> DAI (V2), DAI -> USDC (stable)
        v2_pool = self._make_v2_pool(
            self.WETH,
            self.DAI,
            100_000_000_000_000_000_000_000,  # 100K WETH
            200_000_000_000_000_000_000_000_000,  # 200M DAI
        )
        stable_pool = self._make_stable_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI (18 decimals)
            100_000_000_000_000_000_000_000_000,  # 100M USDC (18 decimals for stable math)
        )

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_stable_pool(stable_pool)

        stable_amm = BalancerStableAMM()
        router = SingleOrderRouter(pool_registry=registry, stable_amm=stable_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",  # Very low min
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # First hop through V2
        assert isinstance(result.hops[0].pool, UniswapV2Pool)
        # Second hop through stable
        assert isinstance(result.hops[1].pool, BalancerStablePool)
        assert result.amount_out > 0

    def test_multihop_stable_then_v2(self):
        """Multi-hop route: stable pool for hop 1, V2 pool for hop 2."""
        from solver.amm.balancer import BalancerStableAMM

        # Create pools: DAI -> USDC (stable), USDC -> WETH (V2)
        stable_pool = self._make_stable_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI
            100_000_000_000_000_000_000_000_000,  # 100M USDC (18 decimals for stable math)
        )
        v2_pool = self._make_v2_pool(
            self.USDC,
            self.WETH,
            100_000_000_000_000_000_000_000_000,  # 100M USDC (scaled to 18 decimals)
            50_000_000_000_000_000_000_000,  # 50K WETH
        )

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_stable_pool(stable_pool)

        stable_amm = BalancerStableAMM()
        router = SingleOrderRouter(pool_registry=registry, stable_amm=stable_amm)

        order = make_order(
            sell_token=self.DAI,
            buy_token=self.WETH,
            sell_amount="1000000000000000000000",  # 1000 DAI
            buy_amount="1",  # Very low min
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # First hop should be through stable pool
        assert isinstance(result.hops[0].pool, BalancerStablePool)
        # Second hop through V2
        assert isinstance(result.hops[1].pool, UniswapV2Pool)
        assert result.amount_out > 0

    def test_multihop_mixed_weighted_stable(self):
        """Multi-hop through weighted then stable pool."""
        from solver.amm.balancer import (
            BalancerStableAMM,
            BalancerWeightedAMM,
        )

        # Path: WETH -> DAI (weighted) -> USDC (stable)
        weighted_pool = self._make_weighted_pool(
            self.WETH,
            self.DAI,
            50_000_000_000_000_000_000_000,  # 50K WETH
            100_000_000_000_000_000_000_000_000,  # 100M DAI
        )
        stable_pool = self._make_stable_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI
            100_000_000_000_000_000_000_000_000,  # 100M USDC
        )

        registry = PoolRegistry()
        registry.add_weighted_pool(weighted_pool)
        registry.add_stable_pool(stable_pool)

        weighted_amm = BalancerWeightedAMM()
        stable_amm = BalancerStableAMM()
        router = SingleOrderRouter(
            pool_registry=registry, weighted_amm=weighted_amm, stable_amm=stable_amm
        )

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # First hop through weighted
        assert isinstance(result.hops[0].pool, BalancerWeightedPool)
        # Second hop through stable
        assert isinstance(result.hops[1].pool, BalancerStablePool)
        assert result.amount_out > 0

    def test_multihop_buy_order_through_stable(self):
        """Buy order multi-hop through stable pool."""
        from solver.amm.balancer import BalancerStableAMM

        # Path: WETH -> DAI (V2) -> USDC (stable)
        v2_pool = self._make_v2_pool(
            self.WETH,
            self.DAI,
            100_000_000_000_000_000_000_000,  # 100K WETH
            200_000_000_000_000_000_000_000_000,  # 200M DAI
        )
        stable_pool = self._make_stable_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI
            100_000_000_000_000_000_000_000_000,  # 100M USDC
        )

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_stable_pool(stable_pool)

        stable_amm = BalancerStableAMM()
        router = SingleOrderRouter(pool_registry=registry, stable_amm=stable_amm)

        # Buy order: want USDC, spend WETH
        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="10000000000000000000",  # 10 WETH max
            buy_amount="1000000000000000000000",  # Want 1000 USDC (18 decimals)
            kind=OrderKind.BUY,
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # Buy order should work backwards through the path
        assert result.amount_out == 1000000000000000000000  # Exact buy amount
        assert result.amount_in > 0
        assert result.amount_in <= 10000000000000000000


class TestV3MixedMultiHopRouting:
    """Tests for multi-hop routing that includes UniswapV3 pools."""

    # Token addresses for tests
    WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
    WBTC = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"

    def _make_v2_pool(
        self, token0: str, token1: str, reserve0: int, reserve1: int
    ) -> UniswapV2Pool:
        """Create a V2 pool for testing."""
        if token0.lower() > token1.lower():
            token0, token1 = token1, token0
            reserve0, reserve1 = reserve1, reserve0
        return UniswapV2Pool(
            address=f"0x{token0[-6:]}{token1[-6:]}1111111111111111111111",
            token0=token0.lower(),
            token1=token1.lower(),
            reserve0=reserve0,
            reserve1=reserve1,
            fee_bps=30,
            liquidity_id=f"v2-{token0[-4:]}-{token1[-4:]}",
        )

    def _make_v3_pool(self, token0: str, token1: str, fee: int = 3000) -> UniswapV3Pool:
        """Create a V3 pool for testing."""
        if token0.lower() > token1.lower():
            token0, token1 = token1, token0
        return UniswapV3Pool(
            address=f"0x{token0[-6:]}{token1[-6:]}3333333333333333333333",
            token0=token0.lower(),
            token1=token1.lower(),
            fee=fee,
            sqrt_price_x96=0,
            liquidity=0,
            tick=0,
            liquidity_id=f"v3-{token0[-4:]}-{token1[-4:]}",
            gas_estimate=150_000,
        )

    def _make_weighted_pool(self, token0: str, token1: str, balance0: int, balance1: int):
        """Create a Balancer weighted pool for testing."""
        return BalancerWeightedPool(
            id=f"weighted-{token0[-4:]}-{token1[-4:]}",
            address=f"0x{token0[-6:]}{token1[-6:]}2222222222222222222222",
            pool_id="0x" + "33" * 32,
            reserves=(
                WeightedTokenReserve(
                    token=token0.lower(),
                    balance=balance0,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
                WeightedTokenReserve(
                    token=token1.lower(),
                    balance=balance1,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.003"),
            version="v3Plus",
            gas_estimate=120_000,
        )

    def _make_stable_pool(self, token0: str, token1: str, balance0: int, balance1: int):
        """Create a Balancer stable pool for testing."""
        return BalancerStablePool(
            id=f"stable-{token0[-4:]}-{token1[-4:]}",
            address=f"0x{token0[-6:]}{token1[-6:]}4444444444444444444444",
            pool_id="0x" + "55" * 32,
            reserves=(
                StableTokenReserve(
                    token=token0.lower(),
                    balance=balance0,
                    scaling_factor=1,
                ),
                StableTokenReserve(
                    token=token1.lower(),
                    balance=balance1,
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.0001"),
            amplification_parameter=Decimal("5000"),
            gas_estimate=100_000,
        )

    def _make_v3_quoter(self, rate: float) -> MockUniswapV3Quoter:
        """Create a mock V3 quoter with a fixed rate."""
        return MockUniswapV3Quoter(default_rate=rate)

    def test_multihop_v2_then_v3(self):
        """Multi-hop route: V2 pool for hop 1, V3 pool for hop 2."""
        # Path: WETH -> DAI (V2) -> USDC (V3)
        v2_pool = self._make_v2_pool(
            self.WETH,
            self.DAI,
            100_000_000_000_000_000_000_000,  # 100K WETH
            200_000_000_000_000_000_000_000_000,  # 200M DAI
        )
        v3_pool = self._make_v3_pool(self.DAI, self.USDC)

        # V3 quoter: 1 DAI = 1 USDC (scaled for decimals)
        quoter = self._make_v3_quoter(1e6 / 1e18)  # DAI (18) -> USDC (6)
        v3_amm = UniswapV3AMM(quoter=quoter)

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_v3_pool(v3_pool)

        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # First hop through V2
        assert isinstance(result.hops[0].pool, UniswapV2Pool)
        # Second hop through V3
        assert isinstance(result.hops[1].pool, UniswapV3Pool)
        assert result.amount_out > 0

    def test_multihop_v3_then_v2(self):
        """Multi-hop route: V3 pool for hop 1, V2 pool for hop 2."""
        # Path: WETH -> DAI (V3) -> USDC (V2)
        v3_pool = self._make_v3_pool(self.WETH, self.DAI)
        v2_pool = self._make_v2_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI
            100_000_000_000_000,  # 100M USDC (6 decimals)
        )

        # V3 quoter: 1 WETH = 2000 DAI
        quoter = self._make_v3_quoter(2000e18 / 1e18)  # WETH -> DAI
        v3_amm = UniswapV3AMM(quoter=quoter)

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_v3_pool(v3_pool)

        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # First hop through V3
        assert isinstance(result.hops[0].pool, UniswapV3Pool)
        # Second hop through V2
        assert isinstance(result.hops[1].pool, UniswapV2Pool)
        assert result.amount_out > 0

    def test_multihop_v3_then_weighted(self):
        """Multi-hop route: V3 pool for hop 1, weighted pool for hop 2."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Path: WETH -> DAI (V3) -> USDC (weighted)
        v3_pool = self._make_v3_pool(self.WETH, self.DAI)
        weighted_pool = self._make_weighted_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI
            100_000_000_000_000,  # 100M USDC (6 decimals)
        )

        # V3 quoter: 1 WETH = 2000 DAI
        quoter = self._make_v3_quoter(2000e18 / 1e18)
        v3_amm = UniswapV3AMM(quoter=quoter)
        weighted_amm = BalancerWeightedAMM()

        registry = PoolRegistry()
        registry.add_v3_pool(v3_pool)
        registry.add_weighted_pool(weighted_pool)

        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # First hop through V3
        assert isinstance(result.hops[0].pool, UniswapV3Pool)
        # Second hop through weighted
        assert isinstance(result.hops[1].pool, BalancerWeightedPool)
        assert result.amount_out > 0

    def test_multihop_v3_then_stable(self):
        """Multi-hop route: V3 pool for hop 1, stable pool for hop 2."""
        from solver.amm.balancer import BalancerStableAMM

        # Path: WETH -> DAI (V3) -> USDC (stable)
        v3_pool = self._make_v3_pool(self.WETH, self.DAI)
        stable_pool = self._make_stable_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI
            100_000_000_000_000_000_000_000_000,  # 100M USDC (18 decimals for stable)
        )

        # V3 quoter: 1 WETH = 2000 DAI
        quoter = self._make_v3_quoter(2000e18 / 1e18)
        v3_amm = UniswapV3AMM(quoter=quoter)
        stable_amm = BalancerStableAMM()

        registry = PoolRegistry()
        registry.add_v3_pool(v3_pool)
        registry.add_stable_pool(stable_pool)

        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm, stable_amm=stable_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # First hop through V3
        assert isinstance(result.hops[0].pool, UniswapV3Pool)
        # Second hop through stable
        assert isinstance(result.hops[1].pool, BalancerStablePool)
        assert result.amount_out > 0

    def test_multihop_weighted_then_v3(self):
        """Multi-hop route: weighted pool for hop 1, V3 pool for hop 2."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Path: WETH -> DAI (weighted) -> USDC (V3)
        weighted_pool = self._make_weighted_pool(
            self.WETH,
            self.DAI,
            100_000_000_000_000_000_000_000,  # 100K WETH
            200_000_000_000_000_000_000_000_000,  # 200M DAI
        )
        v3_pool = self._make_v3_pool(self.DAI, self.USDC)

        # V3 quoter: 1 DAI = 1 USDC (scaled for decimals)
        quoter = self._make_v3_quoter(1e6 / 1e18)
        v3_amm = UniswapV3AMM(quoter=quoter)
        weighted_amm = BalancerWeightedAMM()

        registry = PoolRegistry()
        registry.add_weighted_pool(weighted_pool)
        registry.add_v3_pool(v3_pool)

        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # First hop through weighted
        assert isinstance(result.hops[0].pool, BalancerWeightedPool)
        # Second hop through V3
        assert isinstance(result.hops[1].pool, UniswapV3Pool)
        assert result.amount_out > 0

    def test_multihop_v3_weighted_v3(self):
        """Multi-hop route: V3 -> weighted -> V3 (3 hops)."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Path: WBTC -> WETH (V3) -> DAI (weighted) -> USDC (V3)
        v3_pool_1 = self._make_v3_pool(self.WBTC, self.WETH)
        weighted_pool = self._make_weighted_pool(
            self.WETH,
            self.DAI,
            50_000_000_000_000_000_000_000,  # 50K WETH
            100_000_000_000_000_000_000_000_000,  # 100M DAI
        )
        v3_pool_2 = self._make_v3_pool(self.DAI, self.USDC)

        # V3 quoter: 1 WBTC = 15 WETH, 1 DAI = 1 USDC
        quoter = self._make_v3_quoter(15e18 / 1e8)  # WBTC->WETH: 15 WETH per WBTC (8 decimals)
        v3_amm = UniswapV3AMM(quoter=quoter)
        weighted_amm = BalancerWeightedAMM()

        registry = PoolRegistry()
        registry.add_v3_pool(v3_pool_1)
        registry.add_v3_pool(v3_pool_2)
        registry.add_weighted_pool(weighted_pool)

        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WBTC,
            buy_token=self.USDC,
            sell_amount="10000000",  # 0.1 WBTC
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 3
        assert result.amount_out > 0

    def test_multihop_selects_best_v3_vs_weighted(self):
        """Multi-hop selects best pool when V3 and weighted available for same hop."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Create two pools for WETH -> DAI: V3 and weighted
        # V3: 1 WETH = 2500 DAI (better)
        v3_pool = self._make_v3_pool(self.WETH, self.DAI)
        # Weighted: 1 WETH = 2000 DAI
        weighted_pool = self._make_weighted_pool(
            self.WETH,
            self.DAI,
            1_000_000_000_000_000_000_000,  # 1K WETH
            2_000_000_000_000_000_000_000_000,  # 2M DAI
        )
        # Second hop only V2 available
        v2_pool = self._make_v2_pool(
            self.DAI,
            self.USDC,
            100_000_000_000_000_000_000_000_000,  # 100M DAI
            100_000_000_000_000,  # 100M USDC
        )

        quoter = self._make_v3_quoter(2500e18 / 1e18)  # V3 gives 2500 DAI per WETH
        v3_amm = UniswapV3AMM(quoter=quoter)
        weighted_amm = BalancerWeightedAMM()

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_v3_pool(v3_pool)
        registry.add_weighted_pool(weighted_pool)

        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.hops is not None
        assert len(result.hops) == 2
        # First hop should select V3 (better quote: 2500 vs ~1994 after fees)
        assert isinstance(result.hops[0].pool, UniswapV3Pool)


class TestAllPoolTypesSelection:
    """Tests for best-pool selection when all 4 pool types are available."""

    WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"

    def _make_v2_pool(self, reserve_weth: int, reserve_usdc: int) -> UniswapV2Pool:
        """Create a WETH/USDC V2 pool."""
        return UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=self.USDC.lower(),  # USDC < WETH lexicographically
            token1=self.WETH.lower(),
            reserve0=reserve_usdc,
            reserve1=reserve_weth,
            fee_bps=30,
            liquidity_id="v2-weth-usdc",
        )

    def _make_v3_pool(self) -> UniswapV3Pool:
        """Create a WETH/USDC V3 pool."""
        return UniswapV3Pool(
            address="0x2222222222222222222222222222222222222222",
            token0=self.USDC.lower(),
            token1=self.WETH.lower(),
            fee=3000,
            sqrt_price_x96=0,
            liquidity=0,
            tick=0,
            liquidity_id="v3-weth-usdc",
            gas_estimate=150_000,
        )

    def _make_weighted_pool(self, balance_weth: int, balance_usdc: int):
        """Create a WETH/USDC weighted pool.

        Args:
            balance_weth: WETH balance in 18 decimals (native)
            balance_usdc: USDC balance in 6 decimals (native)
        """
        return BalancerWeightedPool(
            id="weighted-weth-usdc",
            address="0x3333333333333333333333333333333333333333",
            pool_id="0x" + "44" * 32,
            reserves=(
                WeightedTokenReserve(
                    token=self.WETH.lower(),
                    balance=balance_weth,
                    weight=Decimal("0.5"),
                    scaling_factor=1,  # 18 decimals, no scaling needed
                ),
                WeightedTokenReserve(
                    token=self.USDC.lower(),
                    balance=balance_usdc,
                    weight=Decimal("0.5"),
                    scaling_factor=10**12,  # 6 decimals -> 18 decimals
                ),
            ),
            fee=Decimal("0.003"),
            version="v3Plus",
            gas_estimate=120_000,
        )

    def _make_stable_pool(self, balance_weth: int, balance_usdc: int):
        """Create a WETH/USDC stable pool.

        Note: WETH/USDC is unrealistic for a stable pool, but this tests
        the router's ability to compare across all pool types correctly.

        Args:
            balance_weth: WETH balance in 18 decimals (native)
            balance_usdc: USDC balance in 6 decimals (native)
        """
        return BalancerStablePool(
            id="stable-weth-usdc",
            address="0x4444444444444444444444444444444444444444",
            pool_id="0x" + "55" * 32,
            reserves=(
                StableTokenReserve(
                    token=self.WETH.lower(),
                    balance=balance_weth,
                    scaling_factor=1,  # 18 decimals, no scaling needed
                ),
                StableTokenReserve(
                    token=self.USDC.lower(),
                    balance=balance_usdc,
                    scaling_factor=10**12,  # 6 decimals -> 18 decimals
                ),
            ),
            fee=Decimal("0.0001"),
            amplification_parameter=Decimal("5000"),
            gas_estimate=100_000,
        )

    def test_selects_best_among_all_four_pool_types(self):
        """Router selects the pool with best quote across V2, V3, weighted, stable."""
        from solver.amm.balancer import (
            BalancerStableAMM,
            BalancerWeightedAMM,
        )

        # V2: 1 WETH = 2000 USDC (reserve ratio)
        v2_pool = self._make_v2_pool(
            reserve_weth=100_000_000_000_000_000_000_000,  # 100K WETH (18 dec)
            reserve_usdc=200_000_000_000_000,  # 200M USDC (6 dec)
        )
        # V3: 1 WETH = 2200 USDC (mock)
        v3_pool = self._make_v3_pool()
        quoter = MockUniswapV3Quoter(default_rate=2200e6 / 1e18)
        v3_amm = UniswapV3AMM(quoter=quoter)

        # Weighted: 1 WETH = 2500 USDC (best rate)
        weighted_pool = self._make_weighted_pool(
            balance_weth=100_000_000_000_000_000_000_000,  # 100K WETH (18 dec)
            balance_usdc=250_000_000_000_000,  # 250M USDC (6 dec)
        )
        weighted_amm = BalancerWeightedAMM()

        # Stable: 1 WETH = 1 USDC (stable pools give ~1:1, poor for volatile pairs)
        # With proper scaling, this gives ~1 USDC per WETH (very poor rate)
        stable_pool = self._make_stable_pool(
            balance_weth=100_000_000_000_000_000_000_000,  # 100K WETH (18 dec)
            balance_usdc=100_000_000_000,  # 100K USDC (6 dec)
        )
        stable_amm = BalancerStableAMM()

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_v3_pool(v3_pool)
        registry.add_weighted_pool(weighted_pool)
        registry.add_stable_pool(stable_pool)

        router = SingleOrderRouter(
            pool_registry=registry,
            v3_amm=v3_amm,
            weighted_amm=weighted_amm,
            stable_amm=stable_amm,
        )

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        # Should select weighted pool (best rate: ~2500 USDC after fees)
        # V2 gives ~1994 USDC, V3 gives 2200 USDC, stable gives ~1 USDC
        assert isinstance(result.pool, BalancerWeightedPool)

    def test_selects_v3_when_best_among_all_four(self):
        """Router selects V3 when it has the best quote among all 4 pool types."""
        from solver.amm.balancer import BalancerStableAMM, BalancerWeightedAMM

        # V2: 1 WETH = 2000 USDC
        v2_pool = self._make_v2_pool(
            reserve_weth=100_000_000_000_000_000_000_000,  # 100K WETH (18 dec)
            reserve_usdc=200_000_000_000_000,  # 200M USDC (6 dec)
        )
        # V3: 1 WETH = 3000 USDC (best)
        v3_pool = self._make_v3_pool()
        quoter = MockUniswapV3Quoter(default_rate=3000e6 / 1e18)
        v3_amm = UniswapV3AMM(quoter=quoter)

        # Weighted: 1 WETH = 2500 USDC
        weighted_pool = self._make_weighted_pool(
            balance_weth=100_000_000_000_000_000_000_000,  # 100K WETH (18 dec)
            balance_usdc=250_000_000_000_000,  # 250M USDC (6 dec)
        )
        weighted_amm = BalancerWeightedAMM()

        # Stable: poor rate for volatile pair
        stable_pool = self._make_stable_pool(
            balance_weth=100_000_000_000_000_000_000_000,  # 100K WETH (18 dec)
            balance_usdc=100_000_000_000,  # 100K USDC (6 dec)
        )
        stable_amm = BalancerStableAMM()

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_v3_pool(v3_pool)
        registry.add_weighted_pool(weighted_pool)
        registry.add_stable_pool(stable_pool)

        router = SingleOrderRouter(
            pool_registry=registry,
            v3_amm=v3_amm,
            weighted_amm=weighted_amm,
            stable_amm=stable_amm,
        )

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        # Should select V3 pool (best rate: 3000 USDC)
        assert isinstance(result.pool, UniswapV3Pool)

    def test_v3_vs_weighted_direct_comparison(self):
        """Direct comparison between V3 and weighted pool for same pair."""
        from solver.amm.balancer import BalancerWeightedAMM

        # V3: 1 WETH = 2000 USDC
        v3_pool = self._make_v3_pool()
        quoter = MockUniswapV3Quoter(default_rate=2000e6 / 1e18)
        v3_amm = UniswapV3AMM(quoter=quoter)

        # Weighted: 1 WETH = 2500 USDC (better)
        weighted_pool = self._make_weighted_pool(
            balance_weth=100_000_000_000_000_000_000_000,
            balance_usdc=250_000_000_000_000,
        )
        weighted_amm = BalancerWeightedAMM()

        registry = PoolRegistry()
        registry.add_v3_pool(v3_pool)
        registry.add_weighted_pool(weighted_pool)

        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        # Weighted should win
        assert isinstance(result.pool, BalancerWeightedPool)

    def test_v3_vs_stable_direct_comparison(self):
        """Direct comparison between V3 and stable pool for same pair."""
        from solver.amm.balancer import BalancerStableAMM

        # For stablecoins: DAI/USDC
        dai = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
        usdc = self.USDC

        # V3: slightly worse rate for stablecoins
        v3_pool = UniswapV3Pool(
            address="0x5555555555555555555555555555555555555555",
            token0=dai.lower(),
            token1=usdc.lower(),
            fee=100,  # 0.01% fee tier
            sqrt_price_x96=0,
            liquidity=0,
            tick=0,
            liquidity_id="v3-dai-usdc",
            gas_estimate=150_000,
        )
        # V3 gives 0.998 USDC per DAI
        quoter = MockUniswapV3Quoter(default_rate=0.998e6 / 1e18)
        v3_amm = UniswapV3AMM(quoter=quoter)

        # Stable: near 1:1 (better for stablecoins)
        stable_pool = BalancerStablePool(
            id="stable-dai-usdc",
            address="0x6666666666666666666666666666666666666666",
            pool_id="0x" + "77" * 32,
            reserves=(
                StableTokenReserve(
                    token=dai.lower(),
                    balance=100_000_000_000_000_000_000_000_000,  # 100M DAI
                    scaling_factor=1,
                ),
                StableTokenReserve(
                    token=usdc.lower(),
                    balance=100_000_000_000_000_000_000_000_000,  # 100M USDC (18 dec)
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.0001"),
            amplification_parameter=Decimal("5000"),
            gas_estimate=100_000,
        )
        stable_amm = BalancerStableAMM()

        registry = PoolRegistry()
        registry.add_v3_pool(v3_pool)
        registry.add_stable_pool(stable_pool)

        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm, stable_amm=stable_amm)

        order = make_order(
            sell_token=dai,
            buy_token=usdc,
            sell_amount="1000000000000000000000",  # 1000 DAI
            buy_amount="1",
        )
        result = router.route_order(order)

        assert result.success is True
        # Stable should win for stablecoin pair
        assert isinstance(result.pool, BalancerStablePool)
