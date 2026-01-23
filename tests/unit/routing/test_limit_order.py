"""Tests for 0x limit order routing."""

import pytest

from solver.amm.limit_order import LimitOrderAMM
from solver.models.auction import Order
from solver.pools.limit_order import LimitOrderPool
from solver.pools.registry import PoolRegistry
from solver.routing.handlers.limit_order import LimitOrderHandler
from solver.routing.router import SingleOrderRouter


class TestLimitOrderHandler:
    """Tests for LimitOrderHandler."""

    WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

    @pytest.fixture
    def amm(self):
        """Create a LimitOrderAMM instance."""
        return LimitOrderAMM()

    @pytest.fixture
    def handler(self, amm):
        """Create a LimitOrderHandler instance."""
        return LimitOrderHandler(amm)

    @pytest.fixture
    def pool(self):
        """Create a test limit order pool.

        Offers 2500 USDC for 1 WETH (rate: 2500 USDC/WETH)
        """
        return LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token=self.USDC,
            taker_token=self.WETH,
            maker_amount=2500_000_000,  # 2500 USDC
            taker_amount=1_000_000_000_000_000_000,  # 1 WETH
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

    def _make_sell_order(self, sell_amount: int, buy_amount: int) -> Order:
        """Create a sell order (WETH -> USDC)."""
        return Order(
            uid="0x" + "a" * 112,
            sellToken=self.WETH,
            buyToken=self.USDC,
            sellAmount=str(sell_amount),
            buyAmount=str(buy_amount),
            fullSellAmount=str(sell_amount),
            fullBuyAmount=str(buy_amount),
            kind="sell",
            partiallyFillable=False,
            class_="market",
            sellTokenSource="erc20",
            buyTokenDestination="erc20",
            preInteractions=[],
            postInteractions=[],
            validTo=0,
            appData="0x" + "0" * 64,
            signingScheme="presign",
            signature="0x",
            owner="0x" + "1" * 40,
            feePolicies=[],
        )

    def _make_buy_order(self, sell_amount: int, buy_amount: int) -> Order:
        """Create a buy order (WETH -> USDC)."""
        return Order(
            uid="0x" + "b" * 112,
            sellToken=self.WETH,
            buyToken=self.USDC,
            sellAmount=str(sell_amount),
            buyAmount=str(buy_amount),
            fullSellAmount=str(sell_amount),
            fullBuyAmount=str(buy_amount),
            kind="buy",
            partiallyFillable=False,
            class_="market",
            sellTokenSource="erc20",
            buyTokenDestination="erc20",
            preInteractions=[],
            postInteractions=[],
            validTo=0,
            appData="0x" + "0" * 64,
            signingScheme="presign",
            signature="0x",
            owner="0x" + "1" * 40,
            feePolicies=[],
        )

    def test_route_sell_order_success(self, handler, pool):
        """Test routing a sell order through a limit order."""
        order = self._make_sell_order(
            sell_amount=500_000_000_000_000_000,  # 0.5 WETH
            buy_amount=1000_000_000,  # Min 1000 USDC
        )

        result = handler.route(order, pool, order.sell_amount_int, order.buy_amount_int)

        assert result.success
        assert result.amount_in == 500_000_000_000_000_000
        assert result.amount_out == 1250_000_000  # 1250 USDC (linear pricing)
        assert result.gas_estimate == 66_358

    def test_route_sell_order_fails_limit_price(self, handler, pool):
        """Test that sell order fails when output below limit."""
        order = self._make_sell_order(
            sell_amount=500_000_000_000_000_000,  # 0.5 WETH
            buy_amount=2000_000_000,  # Min 2000 USDC (too high)
        )

        result = handler.route(order, pool, order.sell_amount_int, order.buy_amount_int)

        assert not result.success
        assert "below minimum" in result.error

    def test_route_buy_order_success(self, handler, pool):
        """Test routing a buy order through a limit order."""
        order = self._make_buy_order(
            sell_amount=1_000_000_000_000_000_000,  # Max 1 WETH
            buy_amount=1250_000_000,  # Want 1250 USDC
        )

        result = handler.route(order, pool, order.sell_amount_int, order.buy_amount_int)

        assert result.success
        assert result.amount_out == 1250_000_000
        assert result.amount_in == 500_000_000_000_000_000  # 0.5 WETH

    def test_route_buy_order_fails_limit_price(self, handler, pool):
        """Test that buy order fails when input exceeds limit."""
        order = self._make_buy_order(
            sell_amount=100_000_000_000_000_000,  # Max 0.1 WETH
            buy_amount=1000_000_000,  # Want 1000 USDC (needs 0.4 WETH)
        )

        result = handler.route(order, pool, order.sell_amount_int, order.buy_amount_int)

        assert not result.success
        assert "exceeds maximum" in result.error

    def test_handler_without_amm(self, pool):
        """Test that handler without AMM returns error."""
        handler = LimitOrderHandler(None)
        order = self._make_sell_order(sell_amount=100, buy_amount=100)

        result = handler.route(order, pool, 100, 100)

        assert not result.success
        assert "not configured" in result.error


class TestLimitOrderRouting:
    """Tests for limit order routing through SingleOrderRouter."""

    WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

    def _make_sell_order(self, sell_amount: int, buy_amount: int) -> Order:
        """Create a sell order (WETH -> USDC)."""
        return Order(
            uid="0x" + "a" * 112,
            sellToken=self.WETH,
            buyToken=self.USDC,
            sellAmount=str(sell_amount),
            buyAmount=str(buy_amount),
            fullSellAmount=str(sell_amount),
            fullBuyAmount=str(buy_amount),
            kind="sell",
            partiallyFillable=False,
            class_="market",
            sellTokenSource="erc20",
            buyTokenDestination="erc20",
            preInteractions=[],
            postInteractions=[],
            validTo=0,
            appData="0x" + "0" * 64,
            signingScheme="presign",
            signature="0x",
            owner="0x" + "1" * 40,
            feePolicies=[],
        )

    def test_router_routes_through_limit_order(self):
        """Test that the router can route through limit orders."""
        # Create registry with limit order
        registry = PoolRegistry()
        limit_order = LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token=self.USDC,
            taker_token=self.WETH,
            maker_amount=2500_000_000,
            taker_amount=1_000_000_000_000_000_000,
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )
        registry.add_limit_order(limit_order)

        # Create router with limit order AMM
        router = SingleOrderRouter(
            pool_registry=registry,
            limit_order_amm=LimitOrderAMM(),
        )

        # Create order
        order = self._make_sell_order(
            sell_amount=500_000_000_000_000_000,  # 0.5 WETH
            buy_amount=1000_000_000,  # Min 1000 USDC
        )

        # Route order
        result = router.route_order(order)

        assert result.success
        assert result.amount_in == 500_000_000_000_000_000
        assert result.amount_out == 1250_000_000

    def test_router_builds_solution_for_limit_order(self):
        """Test that the router can build a solution from limit order routing."""
        # Create registry with limit order
        registry = PoolRegistry()
        limit_order = LimitOrderPool(
            id="limit-order-123",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token=self.USDC,
            taker_token=self.WETH,
            maker_amount=2500_000_000,
            taker_amount=1_000_000_000_000_000_000,
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )
        registry.add_limit_order(limit_order)

        # Create router with limit order AMM
        router = SingleOrderRouter(
            pool_registry=registry,
            limit_order_amm=LimitOrderAMM(),
        )

        # Create order
        order = self._make_sell_order(
            sell_amount=500_000_000_000_000_000,
            buy_amount=1000_000_000,
        )

        # Route and build solution
        result = router.route_order(order)
        assert result.success

        solution = router.build_solution(result)
        assert solution is not None
        assert len(solution.trades) == 1
        assert len(solution.interactions) == 1

        # Check interaction references the limit order
        interaction = solution.interactions[0]
        assert interaction.id == "limit-order-123"
