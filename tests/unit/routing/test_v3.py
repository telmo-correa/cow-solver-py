"""Unit tests for UniswapV3 integration with the router."""

from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.amm.uniswap_v3 import (
    MockUniswapV3Quoter,
    QuoteKey,
    UniswapV3AMM,
    UniswapV3Pool,
)
from solver.models.auction import OrderKind
from solver.pools import PoolRegistry
from solver.routing.router import SingleOrderRouter
from tests.conftest import make_order


class TestV3RouterIntegration:
    """Tests for UniswapV3 integration with the router."""

    # Token addresses for tests
    WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"

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

    def _make_v3_pool(self, fee: int = 3000) -> UniswapV3Pool:
        """Create a V3 pool for testing."""
        return UniswapV3Pool(
            address="0x2222222222222222222222222222222222222222",
            token0=self.USDC.lower(),
            token1=self.WETH.lower(),
            fee=fee,
            sqrt_price_x96=0,
            liquidity=0,
            tick=0,
            liquidity_id="v3-pool-1",
        )

    def test_router_uses_v2_when_no_v3_amm(self):
        """Router uses V2 pool when V3 AMM is not configured."""
        v2_pool = self._make_v2_pool()
        v3_pool = self._make_v3_pool()

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_v3_pool(v3_pool)

        # No V3 AMM provided
        router = SingleOrderRouter(pool_registry=registry, v3_amm=None)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",  # 2000 USDC min
        )
        result = router.route_order(order)

        assert result.success is True
        # Should use V2 pool
        assert result.pool == v2_pool
        assert result.pool.address == v2_pool.address

    def test_router_uses_v3_when_better_quote(self):
        """Router uses V3 pool when it gives better output."""
        v2_pool = self._make_v2_pool()
        v3_pool = self._make_v3_pool()

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_v3_pool(v3_pool)

        # Configure mock quoter to give better quote than V2
        # V2 pool at current reserves gives ~2492 USDC for 1 WETH
        # Configure V3 to give 3000 USDC (better)
        v3_output = 3000_000_000  # 3000 USDC (6 decimals)
        quote_key = QuoteKey(
            token_in=self.WETH,
            token_out=self.USDC,
            fee=3000,
            amount=1000000000000000000,  # 1 WETH
            is_exact_input=True,
        )
        mock_quoter = MockUniswapV3Quoter(quotes={quote_key: v3_output})

        v3_amm = UniswapV3AMM(quoter=mock_quoter)
        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",  # 2000 USDC min
        )
        result = router.route_order(order)

        assert result.success is True
        # Should use V3 pool (better quote)
        assert isinstance(result.pool, UniswapV3Pool)
        assert result.pool.address == v3_pool.address
        assert result.amount_out == v3_output

    def test_router_uses_v2_when_better_quote(self):
        """Router uses V2 pool when it gives better output than V3."""
        v2_pool = self._make_v2_pool()
        v3_pool = self._make_v3_pool()

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_v3_pool(v3_pool)

        # Configure mock quoter to give worse quote than V2
        # V2 pool gives ~2492 USDC for 1 WETH
        # Configure V3 to give only 2000 USDC (worse)
        v3_output = 2000_000_000  # 2000 USDC (6 decimals)
        quote_key = QuoteKey(
            token_in=self.WETH,
            token_out=self.USDC,
            fee=3000,
            amount=1000000000000000000,  # 1 WETH
            is_exact_input=True,
        )
        mock_quoter = MockUniswapV3Quoter(quotes={quote_key: v3_output})

        v3_amm = UniswapV3AMM(quoter=mock_quoter)
        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="1000000000",  # 1000 USDC min
        )
        result = router.route_order(order)

        assert result.success is True
        # Should use V2 pool (better quote: ~2492 USDC vs 2000 USDC)
        assert isinstance(result.pool, UniswapV2Pool)
        assert result.pool.address == v2_pool.address

    def test_router_skips_v3_when_quoter_fails(self):
        """Router falls back to V2 when V3 quoter returns None."""
        v2_pool = self._make_v2_pool()
        v3_pool = self._make_v3_pool()

        registry = PoolRegistry()
        registry.add_pool(v2_pool)
        registry.add_v3_pool(v3_pool)

        # Mock quoter with no configured quotes (returns None for all)
        mock_quoter = MockUniswapV3Quoter()

        v3_amm = UniswapV3AMM(quoter=mock_quoter)
        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",
            buy_amount="2000000000",
        )
        result = router.route_order(order)

        assert result.success is True
        # Should fall back to V2
        assert isinstance(result.pool, UniswapV2Pool)
        assert result.pool.address == v2_pool.address

    def test_router_v3_only_auction(self):
        """Router handles auction with only V3 liquidity."""
        v3_pool = self._make_v3_pool()

        registry = PoolRegistry()
        registry.add_v3_pool(v3_pool)

        mock_quoter = MockUniswapV3Quoter(default_rate=(2_500_000_000, 10**18))
        v3_amm = UniswapV3AMM(quoter=mock_quoter)
        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",
            buy_amount="2000000000",
        )
        result = router.route_order(order)

        assert result.success is True
        assert isinstance(result.pool, UniswapV3Pool)

    def test_router_v3_buy_order(self):
        """Router handles buy orders with V3 pools."""
        v3_pool = self._make_v3_pool()

        registry = PoolRegistry()
        registry.add_v3_pool(v3_pool)

        # For buy orders, mock quoter needs to return input amount
        # For exact output of 2000 USDC, need ~0.8 WETH input
        mock_quoter = MockUniswapV3Quoter(default_rate=(2_500_000_000, 10**18))
        v3_amm = UniswapV3AMM(quoter=mock_quoter)
        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",  # 1 WETH max
            buy_amount="2000000000",  # Want exactly 2000 USDC
            kind=OrderKind.BUY,
        )
        result = router.route_order(order)

        assert result.success is True
        assert isinstance(result.pool, UniswapV3Pool)
        assert result.amount_out == 2000000000  # Exact buy amount

    def test_build_solution_with_v3_pool(self):
        """build_solution works with V3 pool routing result."""
        v3_pool = self._make_v3_pool()

        registry = PoolRegistry()
        registry.add_v3_pool(v3_pool)

        mock_quoter = MockUniswapV3Quoter(default_rate=(2_500_000_000, 10**18))
        v3_amm = UniswapV3AMM(quoter=mock_quoter)
        router = SingleOrderRouter(pool_registry=registry, v3_amm=v3_amm)

        order = make_order(
            sell_token=self.WETH,
            buy_token=self.USDC,
            sell_amount="1000000000000000000",
            buy_amount="2000000000",
        )
        result = router.route_order(order)
        assert result.success is True

        solution = router.build_solution(result)
        assert solution is not None
        assert len(solution.trades) == 1
        assert len(solution.interactions) == 1

        # Check interaction references V3 pool
        interaction = solution.interactions[0]
        assert interaction.kind == "liquidity"
        assert interaction.id == "v3-pool-1"

    def test_registry_v3_pool_count(self):
        """PoolRegistry tracks V3 pool count separately."""
        registry = PoolRegistry()

        # Add V2 pool
        v2_pool = self._make_v2_pool()
        registry.add_pool(v2_pool)

        # Add V3 pools with different fees
        for fee in [500, 3000, 10000]:
            v3_pool = self._make_v3_pool(fee)
            v3_pool.liquidity_id = f"v3-pool-{fee}"
            registry.add_v3_pool(v3_pool)

        assert registry.pool_count == 1  # V2 pools
        assert registry.v3_pool_count == 3  # V3 pools

    def test_registry_get_pools_for_pair(self):
        """get_pools_for_pair returns both V2 and V3 pools."""
        registry = PoolRegistry()

        v2_pool = self._make_v2_pool()
        registry.add_pool(v2_pool)

        v3_pool_low = self._make_v3_pool(500)
        v3_pool_low.liquidity_id = "v3-pool-500"
        v3_pool_med = self._make_v3_pool(3000)
        v3_pool_med.liquidity_id = "v3-pool-3000"
        registry.add_v3_pool(v3_pool_low)
        registry.add_v3_pool(v3_pool_med)

        pools = registry.get_pools_for_pair(self.WETH, self.USDC)

        assert len(pools) == 3
        v2_pools = [p for p in pools if isinstance(p, UniswapV2Pool)]
        v3_pools = [p for p in pools if isinstance(p, UniswapV3Pool)]
        assert len(v2_pools) == 1
        assert len(v3_pools) == 2
