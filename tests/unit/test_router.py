"""Unit tests for the order router."""

from solver.amm.base import SwapResult
from solver.amm.uniswap_v2 import PoolRegistry, UniswapV2Pool
from solver.amm.uniswap_v3 import (
    MockUniswapV3Quoter,
    QuoteKey,
    UniswapV3AMM,
    UniswapV3Pool,
)
from solver.models.auction import Order, OrderClass, OrderKind
from solver.routing.router import SingleOrderRouter
from solver.solver import Solver

# Note: The `router` fixture is defined in conftest.py with test pools


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


class TestRouterErrorHandling:
    """Tests for router error handling paths."""

    def test_invalid_sell_amount_format(self, router):
        """Non-numeric sell amount returns failure."""
        order = make_order(sell_amount="not-a-number")
        result = router.route_order(order)

        assert result.success is False
        assert "Invalid amount format" in result.error

    def test_invalid_buy_amount_format(self, router):
        """Non-numeric buy amount returns failure."""
        order = make_order(buy_amount="invalid")
        result = router.route_order(order)

        assert result.success is False
        assert "Invalid amount format" in result.error

    def test_zero_sell_amount(self, router):
        """Zero sell amount returns failure."""
        order = make_order(sell_amount="0")
        result = router.route_order(order)

        assert result.success is False
        assert "Sell amount must be positive" in result.error

    def test_negative_sell_amount(self, router):
        """Negative sell amount returns failure."""
        order = make_order(sell_amount="-1000")
        result = router.route_order(order)

        assert result.success is False
        assert "Sell amount must be positive" in result.error

    def test_zero_buy_amount(self, router):
        """Zero buy amount returns failure."""
        order = make_order(buy_amount="0")
        result = router.route_order(order)

        assert result.success is False
        assert "Buy amount must be positive" in result.error

    def test_negative_buy_amount(self, router):
        """Negative buy amount returns failure."""
        order = make_order(buy_amount="-500")
        result = router.route_order(order)

        assert result.success is False
        assert "Buy amount must be positive" in result.error

    def test_no_pool_for_pair(self, router):
        """Unknown token pair returns failure."""
        order = make_order(
            sell_token="0x1111111111111111111111111111111111111111",
            buy_token="0x2222222222222222222222222222222222222222",
        )
        result = router.route_order(order)

        assert result.success is False
        assert "No route found" in result.error

    def test_limit_price_not_met(self, router):
        """Order with unrealistic limit price returns failure."""
        # Asking for way more USDC than the pool can provide at this rate
        order = make_order(
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="10000000000000",  # 10M USDC (unrealistic)
        )
        result = router.route_order(order)

        assert result.success is False
        assert "below minimum" in result.error

    def test_successful_routing(self, router):
        """Valid order returns success."""
        order = make_order(
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",  # 2000 USDC (realistic limit)
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.error is None
        assert result.amount_in == 1000000000000000000
        assert result.amount_out > 2000000000  # Should get more than minimum


class TestBuyOrderRouting:
    """Tests for buy order routing (exact output)."""

    def test_buy_order_success(self, router):
        """Buy order with valid limit price succeeds."""
        order = make_order(
            sell_amount="1000000000000000000",  # 1 WETH max to pay
            buy_amount="2000000000",  # Want exactly 2000 USDC
            kind=OrderKind.BUY,
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.error is None
        assert result.amount_out == 2000000000  # Exact output
        assert result.amount_in <= 1000000000000000000  # Within max

    def test_buy_order_exceeds_max_input(self, router):
        """Buy order requiring more than max input fails."""
        order = make_order(
            sell_amount="100000000000000",  # 0.0001 WETH (way too little)
            buy_amount="2000000000",  # Want 2000 USDC
            kind=OrderKind.BUY,
        )
        result = router.route_order(order)

        assert result.success is False
        assert "exceeds maximum" in result.error

    def test_buy_order_with_realistic_amounts(self, router):
        """Buy order with realistic WETH/USDC amounts."""
        # Want to buy 2400 USDC, willing to pay up to 1 WETH
        # Pool has 50M USDC and 20K WETH, so rate is ~2500 USDC/WETH
        # With 0.3% fee, buying 2400 USDC should require ~0.97 WETH
        order = make_order(
            sell_amount="1000000000000000000",  # 1 WETH max
            buy_amount="2400000000",  # 2400 USDC (6 decimals)
            kind=OrderKind.BUY,
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.amount_out == 2400000000
        assert result.amount_in < 1000000000000000000  # Less than max

    def test_buy_order_vs_sell_order_amounts(self, router):
        """Compare buy and sell order for same trade."""
        # Sell 1 WETH, see how much USDC we get
        sell_order = make_order(
            sell_amount="1000000000000000000",
            buy_amount="1",  # Low min
            kind=OrderKind.SELL,
        )
        sell_result = router.route_order(sell_order)

        # Now place a buy order for that USDC amount
        buy_order = make_order(
            sell_amount="1100000000000000000",  # More WETH than needed
            buy_amount=str(sell_result.amount_out),  # Exact USDC from sell
            kind=OrderKind.BUY,
        )
        buy_result = router.route_order(buy_order)

        assert buy_result.success is True
        # Required input should be approximately 1 WETH
        # Due to rounding, it's slightly higher
        assert abs(buy_result.amount_in - sell_result.amount_in) < 10**15  # Within 0.001 WETH


class TestBuildSolution:
    """Tests for solution building."""

    def test_build_solution_returns_none_for_failed_routing(self, router):
        """build_solution returns None when routing failed."""
        order = make_order(sell_amount="0")  # Will fail
        routing_result = router.route_order(order)

        solution = router.build_solution(routing_result)
        assert solution is None

    def test_build_solution_normalizes_addresses(self, router):
        """build_solution uses normalized (lowercase) addresses in prices."""
        # Use mixed-case addresses in order
        order = make_order(
            sell_token="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # Mixed case
            buy_token="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # Mixed case
        )
        routing_result = router.route_order(order)
        assert routing_result.success

        solution = router.build_solution(routing_result)
        assert solution is not None

        # Prices should use lowercase addresses
        expected_sell = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
        expected_buy = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

        assert expected_sell in solution.prices
        assert expected_buy in solution.prices

    def test_build_solution_sell_order_executed_amount(self, router):
        """Sell order solution uses amount_in as executedAmount."""
        order = make_order(
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",
            kind=OrderKind.SELL,
        )
        routing_result = router.route_order(order)
        assert routing_result.success

        solution = router.build_solution(routing_result)
        assert solution is not None
        assert len(solution.trades) == 1
        # For sell orders, executed_amount = amount sold
        assert solution.trades[0].executed_amount == str(routing_result.amount_in)

    def test_build_solution_buy_order_executed_amount(self, router):
        """Buy order solution uses amount_out as executedAmount."""
        order = make_order(
            sell_amount="1000000000000000000",  # 1 WETH max
            buy_amount="2000000000",  # Want 2000 USDC
            kind=OrderKind.BUY,
        )
        routing_result = router.route_order(order)
        assert routing_result.success

        solution = router.build_solution(routing_result)
        assert solution is not None
        assert len(solution.trades) == 1
        # For buy orders, executed_amount = amount bought
        assert solution.trades[0].executed_amount == str(routing_result.amount_out)

    def test_build_solution_buy_order_uses_exact_output_encoding(self, router):
        """Buy order solution uses swapTokensForExactTokens encoding."""
        order = make_order(
            sell_amount="1000000000000000000",
            buy_amount="2000000000",
            kind=OrderKind.BUY,
        )
        routing_result = router.route_order(order)
        solution = router.build_solution(routing_result)

        assert solution is not None
        assert len(solution.interactions) == 1
        # Should be a LiquidityInteraction with correct structure
        interaction = solution.interactions[0]
        assert interaction.kind == "liquidity"
        assert interaction.internalize is True
        # For buy orders, the output amount should be the requested buy amount
        assert interaction.output_amount == "2000000000"


class TestMultiHopRouting:
    """Tests for multi-hop routing (A→B→C)."""

    def test_multihop_sell_order_usdc_to_dai(self, router):
        """Multi-hop sell order: USDC → WETH → DAI (no direct pool)."""
        # USDC and DAI have no direct pool, must go via WETH
        from solver.constants import DAI, USDC

        order = make_order(
            sell_token=USDC,
            buy_token=DAI,
            sell_amount="1000000000",  # 1000 USDC (6 decimals)
            buy_amount="900000000000000000000",  # 900 DAI min (18 decimals)
            kind=OrderKind.SELL,
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.is_multihop is True
        assert result.path is not None
        assert len(result.path) == 3  # USDC → WETH → DAI
        assert result.pools is not None
        assert len(result.pools) == 2
        # 2 swaps * 60k gas per swap = 120k (before settlement overhead)
        assert result.gas_estimate == 120_000

    def test_multihop_buy_order_usdc_to_dai(self, router):
        """Multi-hop buy order: USDC → WETH → DAI."""
        from solver.constants import DAI, USDC

        order = make_order(
            sell_token=USDC,
            buy_token=DAI,
            sell_amount="2000000000",  # 2000 USDC max
            buy_amount="1000000000000000000000",  # Want exactly 1000 DAI
            kind=OrderKind.BUY,
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.is_multihop is True
        assert result.amount_out == 1000000000000000000000  # Exact output

    def test_multihop_direct_pool_preferred(self, router):
        """Direct pool is preferred over multi-hop when available."""
        # WETH/USDC has a direct pool
        order = make_order(
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",  # 2000 USDC min
            kind=OrderKind.SELL,
        )
        result = router.route_order(order)

        assert result.success is True
        # Should use direct routing, not multi-hop
        assert result.is_multihop is False
        assert result.path is None  # No path for direct routing

    def test_multihop_solution_has_correct_gas(self, router):
        """Multi-hop solution includes gas for all hops."""
        from solver.constants import DAI, USDC

        order = make_order(
            sell_token=USDC,
            buy_token=DAI,
            sell_amount="1000000000",
            buy_amount="1",  # Very low min for test
            kind=OrderKind.SELL,
        )
        result = router.route_order(order)
        solution = router.build_solution(result)

        assert solution is not None
        # Gas = 2 swaps * 60k + 106391 settlement overhead = 226391
        assert solution.gas == 226_391

    def test_multihop_solution_encodes_full_path(self, router):
        """Multi-hop solution encodes the full path in calldata."""
        from solver.constants import DAI, USDC, WETH

        order = make_order(
            sell_token=USDC,
            buy_token=DAI,
            sell_amount="1000000000",
            buy_amount="1",
            kind=OrderKind.SELL,
        )
        result = router.route_order(order)
        solution = router.build_solution(result)

        assert solution is not None
        # Multi-hop routes have one LiquidityInteraction per hop
        assert len(solution.interactions) == 2

        # First hop: USDC → WETH
        hop1 = solution.interactions[0]
        assert hop1.kind == "liquidity"
        assert hop1.input_token.lower() == USDC.lower()
        assert hop1.output_token.lower() == WETH.lower()

        # Second hop: WETH → DAI
        hop2 = solution.interactions[1]
        assert hop2.kind == "liquidity"
        assert hop2.input_token.lower() == WETH.lower()
        assert hop2.output_token.lower() == DAI.lower()

        # Amounts should chain correctly: hop1.output = hop2.input
        assert hop1.output_amount == hop2.input_amount


class TestDependencyInjection:
    """Tests demonstrating dependency injection capabilities."""

    def test_router_with_custom_pool_finder(self):
        """Router can use a custom pool finder function."""

        # Create a pool finder that always returns None (no pools available)
        def no_pools_finder(_token_a: str, _token_b: str) -> UniswapV2Pool | None:
            return None

        router = SingleOrderRouter(pool_finder=no_pools_finder)
        # Use tokens that are NOT in the global pool registry to ensure
        # multi-hop routing also fails (no path exists)
        order = make_order(
            sell_token="0x1111111111111111111111111111111111111111",
            buy_token="0x2222222222222222222222222222222222222222",
        )
        result = router.route_order(order)

        assert result.success is False
        assert "No route found" in result.error

    def test_router_with_mock_amm(self):
        """Router can use a mock AMM for controlled testing."""

        class MockAMM:
            """Mock AMM that returns predictable swap results."""

            SWAP_GAS = 100_000

            def simulate_swap(
                self, pool: UniswapV2Pool, token_in: str, amount_in: int
            ) -> SwapResult:
                # Return a fixed amount regardless of input
                return SwapResult(
                    amount_in=amount_in,
                    amount_out=5000_000_000,
                    pool_address=pool.address,
                    token_in=token_in,
                    token_out=pool.token1 if token_in == pool.token0 else pool.token0,
                    gas_estimate=100_000,
                )

            def encode_swap(
                self,
                _token_in: str,
                _token_out: str,
                _amount_in: int,
                _amount_out_min: int,
                _recipient: str,
            ) -> tuple[str, str]:
                return ("0xmock_target", "0xmock_calldata")

        # Create a registry with a dummy pool
        dummy_pool = UniswapV2Pool(
            address="0x1234567890123456789012345678901234567890",
            token0="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            token1="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            reserve0=1000 * 10**18,
            reserve1=2500000 * 10**6,
            fee_bps=30,
        )

        registry = PoolRegistry()
        registry.add_pool(dummy_pool)

        router = SingleOrderRouter(amm=MockAMM(), pool_registry=registry)
        order = make_order(
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",  # 2000 USDC minimum
        )
        result = router.route_order(order)

        # Mock AMM returns 5000 USDC, which exceeds minimum
        assert result.success is True
        assert result.amount_out == 5000_000_000

    def test_solver_with_custom_router(self):
        """Solver can use a custom router for testing."""
        from solver.models.auction import AuctionInstance, Token
        from solver.routing.router import RoutingResult

        class MockRouter:
            """Mock router that always fails routing."""

            def route_order(self, order):
                return RoutingResult(
                    order=order,
                    amount_in=0,
                    amount_out=0,
                    pool=None,
                    success=False,
                    error="Mock failure",
                )

        solver = Solver(router=MockRouter())

        # Create a minimal auction
        auction = AuctionInstance(
            id="test",
            tokens={
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": Token(
                    decimals=18,
                    symbol="WETH",
                    referencePrice="1000000000000000000",
                    availableBalance="1000000000000000000",
                    trusted=True,
                ),
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": Token(
                    decimals=6,
                    symbol="USDC",
                    referencePrice="400000000000000",
                    availableBalance="100000000000",
                    trusted=True,
                ),
            },
            orders=[make_order()],
            liquidity=[],
            effectiveGasPrice="30000000000",
            deadline="2106-01-01T00:00:00.000Z",
            surplusCapturingJitOrderOwners=[],
        )

        response = solver.solve(auction)

        # Mock router always fails, so no solutions
        assert len(response.solutions) == 0


class TestBuildSolutionErrorPaths:
    """Tests for build_solution error handling."""

    def test_build_solution_returns_none_for_no_hops(self):
        """build_solution returns None when routing result has no hops."""
        from solver.routing.router import RoutingResult, SingleOrderRouter

        router = SingleOrderRouter()
        order = make_order()

        # Create a routing result with success=True but no hops
        # This is an inconsistent state that should be handled gracefully
        result = RoutingResult(
            order=order,
            amount_in=1000,
            amount_out=2000,
            pool=None,
            success=True,
            hops=None,  # No hops
        )

        solution = router.build_solution(result)
        assert solution is None

    def test_build_solution_returns_none_for_empty_hops(self):
        """build_solution returns None when routing result has empty hops list."""
        from solver.routing.router import RoutingResult, SingleOrderRouter

        router = SingleOrderRouter()
        order = make_order()

        result = RoutingResult(
            order=order,
            amount_in=1000,
            amount_out=2000,
            pool=None,
            success=True,
            hops=[],  # Empty hops
        )

        solution = router.build_solution(result)
        assert solution is None


class TestNetworkConfiguration:
    """Tests for network configuration."""

    def test_supported_networks_default(self):
        """Default supported networks includes mainnet."""
        from solver.api.endpoints import SUPPORTED_NETWORKS

        assert "mainnet" in SUPPORTED_NETWORKS

    def test_supported_networks_from_environment(self, monkeypatch):
        """SUPPORTED_NETWORKS can be configured via environment."""
        # Set the environment variable
        monkeypatch.setenv("COW_SUPPORTED_NETWORKS", "mainnet,arbitrum-one,gnosis")

        # Re-import to pick up the new environment variable
        # (Note: In practice, this would require restarting the app)
        import importlib

        import solver.api.endpoints

        importlib.reload(solver.api.endpoints)

        from solver.api.endpoints import SUPPORTED_NETWORKS

        assert "mainnet" in SUPPORTED_NETWORKS
        assert "arbitrum-one" in SUPPORTED_NETWORKS
        assert "gnosis" in SUPPORTED_NETWORKS

        # Reset to default
        monkeypatch.delenv("COW_SUPPORTED_NETWORKS", raising=False)
        importlib.reload(solver.api.endpoints)


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

        mock_quoter = MockUniswapV3Quoter(default_rate=2500_000_000)
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
        mock_quoter = MockUniswapV3Quoter(default_rate=2500_000_000)
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

        mock_quoter = MockUniswapV3Quoter(default_rate=2500_000_000)
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
