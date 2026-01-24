"""Tests for the AmmRoutingStrategy.

These tests verify that AmmRoutingStrategy properly:
- Routes orders through AMM pools
- Respects limit price constraints
- Handles partial fills
"""

from unittest.mock import Mock

from solver.models.auction import AuctionInstance, Order, Token
from solver.routing.types import RoutingResult
from solver.strategies.amm_routing import AmmRoutingStrategy

# Token addresses for tests
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"


def make_order(
    uid: str,
    sell_token: str = WETH,
    buy_token: str = USDC,
    sell_amount: str = "1000000000000000000",  # 1 WETH
    buy_amount: str = "2500000000",  # 2500 USDC
    kind: str = "sell",
    partially_fillable: bool = False,
    order_class: str = "market",
) -> Order:
    """Create a minimal Order for testing."""
    return Order(
        uid=uid,
        sell_token=sell_token,
        buy_token=buy_token,
        sell_amount=sell_amount,
        buy_amount=buy_amount,
        kind=kind,
        class_=order_class,
        partially_fillable=partially_fillable,
    )


def make_auction(orders: list[Order]) -> AuctionInstance:
    """Create a minimal AuctionInstance for testing."""
    weth_lower = WETH.lower()
    usdc_lower = USDC.lower()
    return AuctionInstance(
        id="test",
        orders=orders,
        tokens={
            weth_lower: Token(decimals=18, available_balance="0"),
            usdc_lower: Token(decimals=6, available_balance="0"),
        },
    )


class TestAmmRoutingLimitPrice:
    """Tests for limit price validation in AmmRoutingStrategy.

    The router validates limit prices at the handler level. These tests
    verify that violations properly propagate to return None from the strategy.
    """

    def test_sell_order_limit_satisfied(self):
        """Sell order with limit price satisfied returns a result."""
        order = make_order(
            uid="0x" + "01" * 56,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2500000000",  # 2500 USDC (limit: 2500 USDC/WETH)
            kind="sell",
        )

        # Mock router that returns successful routing (3000 USDC > 2500 limit)
        mock_router = Mock()
        mock_solution = Mock()
        mock_solution.prices = {WETH.lower(): "3000000000", USDC.lower(): "1000000000000000000"}
        mock_solution.interactions = []
        mock_solution.gas = 100000

        mock_router.route_order.return_value = RoutingResult(
            order=order,
            amount_in=1000000000000000000,
            amount_out=3000000000,  # 3000 USDC - exceeds limit
            pool=None,
            success=True,
        )
        mock_router.build_solution.return_value = mock_solution

        strategy = AmmRoutingStrategy()
        strategy._get_router = Mock(return_value=mock_router)
        strategy._create_fee_calculator = Mock(return_value=None)

        auction = make_auction([order])
        result = strategy.try_solve(auction)

        # Should succeed - limit is satisfied
        assert result is not None
        assert len(result.fills) == 1
        assert result.fills[0].buy_filled == 3000000000

    def test_sell_order_limit_violated(self):
        """Sell order with limit price violated returns None."""
        order = make_order(
            uid="0x" + "01" * 56,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="3000000000",  # 3000 USDC (limit: 3000 USDC/WETH)
            kind="sell",
        )

        # Mock router that returns failure (AMM only offers 2500 < 3000 limit)
        mock_router = Mock()
        mock_router.route_order.return_value = RoutingResult(
            order=order,
            amount_in=1000000000000000000,
            amount_out=2500000000,  # 2500 USDC - below limit
            pool=None,
            success=False,
            error="Output 2500000000 below minimum 3000000000",
        )

        strategy = AmmRoutingStrategy()
        strategy._get_router = Mock(return_value=mock_router)

        auction = make_auction([order])
        result = strategy.try_solve(auction)

        # Should fail - limit violated
        assert result is None

    def test_buy_order_limit_satisfied(self):
        """Buy order with limit price satisfied returns a result."""
        order = make_order(
            uid="0x" + "01" * 56,
            sell_amount="3000000000",  # max 3000 USDC
            buy_amount="1000000000000000000",  # wants 1 WETH
            sell_token=USDC,
            buy_token=WETH,
            kind="buy",
        )

        # Mock router - AMM requires 2500 USDC (< 3000 max)
        mock_router = Mock()
        mock_solution = Mock()
        mock_solution.prices = {WETH.lower(): "2500000000", USDC.lower(): "1000000000000000000"}
        mock_solution.interactions = []
        mock_solution.gas = 100000

        mock_router.route_order.return_value = RoutingResult(
            order=order,
            amount_in=2500000000,  # 2500 USDC - within limit
            amount_out=1000000000000000000,  # 1 WETH
            pool=None,
            success=True,
        )
        mock_router.build_solution.return_value = mock_solution

        strategy = AmmRoutingStrategy()
        strategy._get_router = Mock(return_value=mock_router)
        strategy._create_fee_calculator = Mock(return_value=None)

        auction = make_auction([order])
        result = strategy.try_solve(auction)

        # Should succeed - limit is satisfied
        assert result is not None
        assert len(result.fills) == 1
        assert result.fills[0].sell_filled == 2500000000

    def test_buy_order_limit_violated(self):
        """Buy order with limit price violated returns None."""
        order = make_order(
            uid="0x" + "01" * 56,
            sell_amount="2000000000",  # max 2000 USDC
            buy_amount="1000000000000000000",  # wants 1 WETH
            sell_token=USDC,
            buy_token=WETH,
            kind="buy",
        )

        # Mock router - AMM requires 2500 USDC (> 2000 max)
        mock_router = Mock()
        mock_router.route_order.return_value = RoutingResult(
            order=order,
            amount_in=2500000000,  # 2500 USDC - exceeds limit
            amount_out=1000000000000000000,  # 1 WETH
            pool=None,
            success=False,
            error="Required input 2500000000 exceeds maximum 2000000000",
        )

        strategy = AmmRoutingStrategy()
        strategy._get_router = Mock(return_value=mock_router)

        auction = make_auction([order])
        result = strategy.try_solve(auction)

        # Should fail - limit violated
        assert result is None

    def test_sell_order_exactly_at_limit(self):
        """Sell order getting exactly the limit amount succeeds."""
        order = make_order(
            uid="0x" + "01" * 56,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2500000000",  # 2500 USDC (limit: 2500 USDC/WETH)
            kind="sell",
        )

        # Mock router - AMM gives exactly 2500 USDC (at limit)
        mock_router = Mock()
        mock_solution = Mock()
        mock_solution.prices = {WETH.lower(): "2500000000", USDC.lower(): "1000000000000000000"}
        mock_solution.interactions = []
        mock_solution.gas = 100000

        mock_router.route_order.return_value = RoutingResult(
            order=order,
            amount_in=1000000000000000000,
            amount_out=2500000000,  # Exactly at limit
            pool=None,
            success=True,
        )
        mock_router.build_solution.return_value = mock_solution

        strategy = AmmRoutingStrategy()
        strategy._get_router = Mock(return_value=mock_router)
        strategy._create_fee_calculator = Mock(return_value=None)

        auction = make_auction([order])
        result = strategy.try_solve(auction)

        # Should succeed - exactly at limit is OK
        assert result is not None
        assert len(result.fills) == 1
        assert result.fills[0].buy_filled == 2500000000

    def test_buy_order_exactly_at_limit(self):
        """Buy order paying exactly the max amount succeeds."""
        order = make_order(
            uid="0x" + "01" * 56,
            sell_amount="2500000000",  # max 2500 USDC
            buy_amount="1000000000000000000",  # wants 1 WETH
            sell_token=USDC,
            buy_token=WETH,
            kind="buy",
        )

        # Mock router - AMM requires exactly 2500 USDC (at limit)
        mock_router = Mock()
        mock_solution = Mock()
        mock_solution.prices = {WETH.lower(): "2500000000", USDC.lower(): "1000000000000000000"}
        mock_solution.interactions = []
        mock_solution.gas = 100000

        mock_router.route_order.return_value = RoutingResult(
            order=order,
            amount_in=2500000000,  # Exactly at limit
            amount_out=1000000000000000000,
            pool=None,
            success=True,
        )
        mock_router.build_solution.return_value = mock_solution

        strategy = AmmRoutingStrategy()
        strategy._get_router = Mock(return_value=mock_router)
        strategy._create_fee_calculator = Mock(return_value=None)

        auction = make_auction([order])
        result = strategy.try_solve(auction)

        # Should succeed - exactly at limit is OK
        assert result is not None
        assert len(result.fills) == 1
        assert result.fills[0].sell_filled == 2500000000

    def test_sell_order_tiny_violation(self):
        """Sell order with tiny limit violation returns None."""
        order = make_order(
            uid="0x" + "01" * 56,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2500000001",  # 2500.000001 USDC (very tight limit)
            kind="sell",
        )

        # Mock router - AMM gives 2500.000000 USDC (1 wei short!)
        mock_router = Mock()
        mock_router.route_order.return_value = RoutingResult(
            order=order,
            amount_in=1000000000000000000,
            amount_out=2500000000,  # 1 wei below limit
            pool=None,
            success=False,
            error="Output 2500000000 below minimum 2500000001",
        )

        strategy = AmmRoutingStrategy()
        strategy._get_router = Mock(return_value=mock_router)

        auction = make_auction([order])
        result = strategy.try_solve(auction)

        # Should fail - even tiny violation is rejected
        assert result is None

    def test_buy_order_tiny_violation(self):
        """Buy order with tiny limit violation returns None."""
        order = make_order(
            uid="0x" + "01" * 56,
            sell_amount="2500000000",  # max 2500 USDC
            buy_amount="1000000000000000000",  # wants 1 WETH
            sell_token=USDC,
            buy_token=WETH,
            kind="buy",
        )

        # Mock router - AMM requires 2500.000001 USDC (1 wei over!)
        mock_router = Mock()
        mock_router.route_order.return_value = RoutingResult(
            order=order,
            amount_in=2500000001,  # 1 wei over limit
            amount_out=1000000000000000000,
            pool=None,
            success=False,
            error="Required input 2500000001 exceeds maximum 2500000000",
        )

        strategy = AmmRoutingStrategy()
        strategy._get_router = Mock(return_value=mock_router)

        auction = make_auction([order])
        result = strategy.try_solve(auction)

        # Should fail - even tiny violation is rejected
        assert result is None


class TestAmmRoutingMultipleOrders:
    """Tests for multiple order handling in AmmRoutingStrategy."""

    def test_multiple_orders_one_violates_limit(self):
        """When one order violates limit, only the valid one is filled."""
        # Order A: valid limit
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",  # 2000 USDC (satisfied by AMM)
            kind="sell",
        )
        # Order B: limit too high
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="5000000000",  # 5000 USDC (not satisfied by AMM)
            kind="sell",
        )

        # Mock router
        mock_router = Mock()
        mock_solution = Mock()
        mock_solution.prices = {WETH.lower(): "2500000000", USDC.lower(): "1000000000000000000"}
        mock_solution.interactions = []
        mock_solution.gas = 100000

        def route_order_side_effect(order):
            if order.uid == order_a.uid:
                return RoutingResult(
                    order=order,
                    amount_in=1000000000000000000,
                    amount_out=2500000000,  # Satisfies 2000 limit
                    pool=None,
                    success=True,
                )
            else:
                return RoutingResult(
                    order=order,
                    amount_in=1000000000000000000,
                    amount_out=2500000000,  # Doesn't satisfy 5000 limit
                    pool=None,
                    success=False,
                    error="Output 2500000000 below minimum 5000000000",
                )

        mock_router.route_order.side_effect = route_order_side_effect
        mock_router.build_solution.return_value = mock_solution

        strategy = AmmRoutingStrategy()
        strategy._get_router = Mock(return_value=mock_router)
        strategy._create_fee_calculator = Mock(return_value=None)

        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Should succeed with just order_a
        assert result is not None
        assert len(result.fills) == 1
        assert result.fills[0].order.uid == order_a.uid
