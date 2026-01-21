"""Unit tests for the order router."""

import pytest

from solver.amm.base import SwapResult
from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.models.auction import Order, OrderClass, OrderKind
from solver.routing.router import SingleOrderRouter, Solver


@pytest.fixture
def router():
    """Create a router instance."""
    return SingleOrderRouter()


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
        assert "No pool found" in result.error

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
        # Should use swapTokensForExactTokens selector (0x8803dbee)
        assert solution.interactions[0].call_data.startswith("0x8803dbee")


class TestDependencyInjection:
    """Tests demonstrating dependency injection capabilities."""

    def test_router_with_custom_pool_finder(self):
        """Router can use a custom pool finder function."""

        # Create a pool finder that always returns None (no pools available)
        def no_pools_finder(_token_a: str, _token_b: str) -> UniswapV2Pool | None:
            return None

        router = SingleOrderRouter(pool_finder=no_pools_finder)
        order = make_order()
        result = router.route_order(order)

        assert result.success is False
        assert "No pool found" in result.error

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

        # Create a pool finder that returns a dummy pool
        dummy_pool = UniswapV2Pool(
            address="0x1234567890123456789012345678901234567890",
            token0="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            token1="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            reserve0=1000 * 10**18,
            reserve1=2500000 * 10**6,
            fee_bps=30,
        )

        def mock_pool_finder(_token_a: str, _token_b: str) -> UniswapV2Pool | None:
            return dummy_pool

        router = SingleOrderRouter(amm=MockAMM(), pool_finder=mock_pool_finder)
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
