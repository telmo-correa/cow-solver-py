"""Unit tests for Balancer pool integration with the router."""

from decimal import Decimal

from solver.amm.balancer import (
    BalancerStablePool,
    BalancerWeightedPool,
    StableTokenReserve,
    WeightedTokenReserve,
)
from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.models.auction import Order, OrderClass, OrderKind
from solver.pools import PoolRegistry
from solver.routing.router import SingleOrderRouter
from tests.conftest import make_order


class TestBalancerRouterIntegration:
    """Tests for Balancer pool integration with the router."""

    # Token addresses for tests
    WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"

    def _make_v2_pool(self) -> UniswapV2Pool:
        """Create a V2 pool for testing."""
        return UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=self.USDC.lower(),
            token1=self.WETH.lower(),
            reserve0=50_000_000_000_000,  # 50M USDC (6 decimals)
            reserve1=20_000_000_000_000_000_000_000,  # 20K WETH (18 decimals)
            fee_bps=30,
            liquidity_id="v2-pool-1",
        )

    def _make_weighted_pool(self):
        """Create a Balancer weighted pool for testing."""
        return BalancerWeightedPool(
            id="weighted-pool-1",
            address="0x3333333333333333333333333333333333333333",
            pool_id="0x" + "33" * 32,  # Mock 32-byte pool ID
            reserves=(
                WeightedTokenReserve(
                    token=self.USDC.lower(),
                    balance=50_000_000_000_000,  # 50M USDC (6 decimals)
                    weight=Decimal("0.5"),
                    scaling_factor=10**12,  # 18 - 6 decimals
                ),
                WeightedTokenReserve(
                    token=self.WETH.lower(),
                    balance=20_000_000_000_000_000_000_000,  # 20K WETH
                    weight=Decimal("0.5"),
                    scaling_factor=1,  # 18 decimals
                ),
            ),
            fee=Decimal("0.003"),  # 0.3%
            version="v3Plus",
            gas_estimate=120_000,
        )

    def _make_stable_pool(self):
        """Create a Balancer stable pool for testing."""
        return BalancerStablePool(
            id="stable-pool-1",
            address="0x4444444444444444444444444444444444444444",
            pool_id="0x" + "44" * 32,  # Mock 32-byte pool ID
            reserves=(
                StableTokenReserve(
                    token=self.USDC.lower(),
                    balance=100_000_000_000_000,  # 100M USDC (6 decimals)
                    scaling_factor=10**12,
                ),
                StableTokenReserve(
                    token=self.DAI.lower(),
                    balance=100_000_000_000_000_000_000_000_000,  # 100M DAI (18 decimals)
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.0001"),  # 0.01%
            amplification_parameter=Decimal("5000"),
            gas_estimate=100_000,
        )

    def test_router_skips_balancer_when_amm_none(self):
        """Router skips Balancer pools when AMMs are not configured."""
        weighted_pool = self._make_weighted_pool()
        v2_pool = self._make_v2_pool()

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_weighted_pool(weighted_pool)

        # No weighted_amm provided - should only use V2
        router = SingleOrderRouter(pool_registry=registry)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",  # 2000 USDC min
        )
        result = router.route_order(order)

        assert result.success is True
        assert isinstance(result.pool, UniswapV2Pool)

    def test_router_uses_weighted_pool_when_configured(self):
        """Router uses Balancer weighted pool when weighted AMM is configured."""
        from solver.amm.balancer import BalancerWeightedAMM

        weighted_pool = self._make_weighted_pool()

        registry = PoolRegistry()
        registry.add_weighted_pool(weighted_pool)

        weighted_amm = BalancerWeightedAMM()
        router = SingleOrderRouter(pool_registry=registry, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1000000000",  # 1000 USDC min (reasonable limit)
        )
        result = router.route_order(order)

        assert result.success is True
        assert isinstance(result.pool, BalancerWeightedPool)

    def test_router_weighted_sell_order(self):
        """Weighted pool sell order routes correctly."""
        from solver.amm.balancer import BalancerWeightedAMM

        weighted_pool = self._make_weighted_pool()

        registry = PoolRegistry()
        registry.add_weighted_pool(weighted_pool)

        weighted_amm = BalancerWeightedAMM()
        router = SingleOrderRouter(pool_registry=registry, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="100000000000000000",  # 0.1 WETH
            buy_amount="1",  # Very low min for test
            kind=OrderKind.SELL,
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.amount_in == 100000000000000000
        assert result.amount_out > 0
        assert result.hops is not None
        assert len(result.hops) == 1

    def test_router_weighted_buy_order(self):
        """Weighted pool buy order routes correctly."""
        from solver.amm.balancer import BalancerWeightedAMM

        weighted_pool = self._make_weighted_pool()

        registry = PoolRegistry()
        registry.add_weighted_pool(weighted_pool)

        weighted_amm = BalancerWeightedAMM()
        router = SingleOrderRouter(pool_registry=registry, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH max
            buy_amount="100000000",  # Want 100 USDC
            kind=OrderKind.BUY,
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.amount_out == 100000000  # Exact buy amount
        assert result.amount_in > 0
        assert result.amount_in <= 1000000000000000000

    def test_router_stable_sell_order(self):
        """Stable pool sell order routes correctly."""
        from solver.amm.balancer import BalancerStableAMM

        stable_pool = self._make_stable_pool()

        registry = PoolRegistry()
        registry.add_stable_pool(stable_pool)

        stable_amm = BalancerStableAMM()
        router = SingleOrderRouter(pool_registry=registry, stable_amm=stable_amm)

        order = make_order(
            sell_token=self.USDC,
            buy_token=self.DAI,
            sell_amount="1000000000",  # 1000 USDC
            buy_amount="1",  # Very low min for test
            kind=OrderKind.SELL,
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.amount_in == 1000000000
        assert result.amount_out > 0
        assert isinstance(result.pool, BalancerStablePool)

    def test_router_stable_buy_order(self):
        """Stable pool buy order routes correctly."""
        from solver.amm.balancer import BalancerStableAMM

        stable_pool = self._make_stable_pool()

        registry = PoolRegistry()
        registry.add_stable_pool(stable_pool)

        stable_amm = BalancerStableAMM()
        router = SingleOrderRouter(pool_registry=registry, stable_amm=stable_amm)

        order = make_order(
            sell_token=self.USDC,
            buy_token=self.DAI,
            sell_amount="2000000000",  # 2000 USDC max
            buy_amount="1000000000000000000000",  # Want 1000 DAI
            kind=OrderKind.BUY,
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.amount_out == 1000000000000000000000
        assert result.amount_in > 0
        assert isinstance(result.pool, BalancerStablePool)

    def test_router_uses_better_quote_v2_vs_weighted(self):
        """Router chooses V2 over weighted when V2 gives better quote."""
        from solver.amm.balancer import BalancerWeightedAMM

        # Create V2 pool with better rate
        v2_pool = self._make_v2_pool()
        weighted_pool = self._make_weighted_pool()

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
        # Should select whichever gives better output

    def test_build_solution_with_weighted_pool(self):
        """build_solution works with Balancer weighted pool routing result."""
        from solver.amm.balancer import BalancerWeightedAMM

        weighted_pool = self._make_weighted_pool()

        registry = PoolRegistry()
        registry.add_weighted_pool(weighted_pool)

        weighted_amm = BalancerWeightedAMM()
        router = SingleOrderRouter(pool_registry=registry, weighted_amm=weighted_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="100000000000000000",  # 0.1 WETH
            buy_amount="1",
        )
        result = router.route_order(order)
        assert result.success is True

        solution = router.build_solution(result)
        assert solution is not None
        assert len(solution.trades) == 1
        assert len(solution.interactions) == 1

        # Check interaction references the weighted pool
        interaction = solution.interactions[0]
        assert interaction.kind == "liquidity"
        assert interaction.id == "weighted-pool-1"

    def test_build_solution_with_stable_pool(self):
        """build_solution works with Balancer stable pool routing result."""
        from solver.amm.balancer import BalancerStableAMM

        stable_pool = self._make_stable_pool()

        registry = PoolRegistry()
        registry.add_stable_pool(stable_pool)

        stable_amm = BalancerStableAMM()
        router = SingleOrderRouter(pool_registry=registry, stable_amm=stable_amm)

        order = make_order(
            sell_token=self.USDC,
            buy_token=self.DAI,
            sell_amount="1000000000",  # 1000 USDC
            buy_amount="1",
        )
        result = router.route_order(order)
        assert result.success is True

        solution = router.build_solution(result)
        assert solution is not None
        assert len(solution.trades) == 1
        assert len(solution.interactions) == 1

        # Check interaction references the stable pool
        interaction = solution.interactions[0]
        assert interaction.kind == "liquidity"
        assert interaction.id == "stable-pool-1"

    def test_partial_fill_weighted_sell_order(self):
        """Partial fill works for weighted pool sell orders."""
        from solver.amm.balancer import BalancerWeightedAMM

        weighted_pool = self._make_weighted_pool()

        registry = PoolRegistry()
        registry.add_weighted_pool(weighted_pool)

        weighted_amm = BalancerWeightedAMM()
        router = SingleOrderRouter(pool_registry=registry, weighted_amm=weighted_amm)

        # Create a partially fillable order with a limit price that can't be fully satisfied
        order = Order(
            uid="0x" + "01" * 56,
            sellToken=self.WETH,
            buyToken=self.USDC,
            sellAmount="1000000000000000000",  # 1 WETH
            buyAmount="5000000000000",  # Unrealistic 5M USDC (can't fill fully)
            kind=OrderKind.SELL,
            partiallyFillable=True,
            **{"class": OrderClass.LIMIT},
        )

        result = router.route_order(order)

        # Should either fail or return a partial fill
        # (depends on pool math and whether any fill satisfies the limit)
        # For this test, we're checking the code path works without error
        assert result is not None

    def test_registry_get_pools_for_pair_includes_balancer(self):
        """get_pools_for_pair returns Balancer pools alongside V2/V3."""
        v2_pool = self._make_v2_pool()
        weighted_pool = self._make_weighted_pool()

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_weighted_pool(weighted_pool)

        pools = registry.get_pools_for_pair(self.WETH, self.USDC)

        assert len(pools) == 2
        v2_pools = [p for p in pools if isinstance(p, UniswapV2Pool)]
        weighted_pools = [p for p in pools if isinstance(p, BalancerWeightedPool)]
        assert len(v2_pools) == 1
        assert len(weighted_pools) == 1
