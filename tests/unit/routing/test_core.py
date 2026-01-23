"""Unit tests for router core functionality - error handling, buy orders, build solution."""

from solver.models.auction import Order, OrderClass, OrderKind
from solver.routing.router import RoutingResult, SingleOrderRouter

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
        # Router returns "No route found" when all paths fail
        assert result.error is not None

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
        # Router returns "No route found" when all paths fail (including max input exceeded)
        assert result.error is not None

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


class TestBuildSolutionErrorPaths:
    """Tests for build_solution error handling."""

    def test_build_solution_returns_none_for_no_hops(self):
        """build_solution returns None when routing result has no hops."""
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
