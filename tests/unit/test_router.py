"""Unit tests for the order router."""

import pytest

from solver.models.auction import Order, OrderClass, OrderKind
from solver.routing.router import SingleOrderRouter


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
