"""Unit tests for the Solver class."""

from solver.models.auction import AuctionInstance, Order, OrderClass, OrderKind, Token
from solver.solver import Solver
from solver.strategies.base import OrderFill, StrategyResult


def make_order(
    uid: str = "0x" + "01" * 56,
    sell_token: str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    buy_token: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    sell_amount: str = "1000000000000000000",  # 1 WETH
    buy_amount: str = "2000000000",  # 2000 USDC
    kind: OrderKind = OrderKind.SELL,
    order_class: OrderClass = OrderClass.MARKET,
    partially_fillable: bool = False,
) -> Order:
    """Create a test order with sensible defaults."""
    return Order(
        uid=uid,
        sellToken=sell_token,
        buyToken=buy_token,
        sellAmount=sell_amount,
        buyAmount=buy_amount,
        kind=kind,
        **{"class": order_class},
        partiallyFillable=partially_fillable,
    )


def make_auction(orders: list[Order]) -> AuctionInstance:
    """Create a test auction with minimal setup."""
    tokens = {}
    for order in orders:
        if order.sell_token not in tokens:
            tokens[order.sell_token] = Token(
                decimals=18,
                symbol="TKN",
                referencePrice="1000000000000000000",
                availableBalance="1000000000000000000000",
            )
        if order.buy_token not in tokens:
            tokens[order.buy_token] = Token(
                decimals=18,
                symbol="TKN",
                referencePrice="1000000000000000000",
                availableBalance="1000000000000000000000",
            )
    return AuctionInstance(id="test-auction", orders=orders, tokens=tokens)


class MockStrategy:
    """Mock strategy for testing Solver behavior."""

    def __init__(
        self,
        result: StrategyResult | None = None,
        name: str = "MockStrategy",
    ):
        self._result = result
        self._name = name
        self.calls: list[AuctionInstance] = []

    def try_solve(self, auction: AuctionInstance) -> StrategyResult | None:
        self.calls.append(auction)
        return self._result


class TestSolverEmptyAuction:
    """Tests for empty auction handling."""

    def test_empty_auction_returns_empty_response(self):
        """Solver returns empty response for empty auction."""
        solver = Solver(strategies=[])
        auction = AuctionInstance(id="empty", orders=[])

        response = solver.solve(auction)

        assert response.solutions == []

    def test_no_strategies_returns_empty_response(self):
        """Solver with no strategies returns empty response."""
        solver = Solver(strategies=[])
        auction = make_auction([make_order()])

        response = solver.solve(auction)

        assert response.solutions == []


class TestSolverStrategyComposition:
    """Tests for strategy composition behavior."""

    def test_first_strategy_called_with_full_auction(self):
        """First strategy receives the full auction."""
        mock_strategy = MockStrategy(result=None)
        solver = Solver(strategies=[mock_strategy])
        order = make_order()
        auction = make_auction([order])

        solver.solve(auction)

        assert len(mock_strategy.calls) == 1
        assert mock_strategy.calls[0].order_count == 1
        assert mock_strategy.calls[0].orders[0].uid == order.uid

    def test_strategy_stops_when_all_orders_filled(self):
        """Solver stops trying strategies when all orders are filled."""
        order = make_order()
        fill = OrderFill(order, sell_filled=1000, buy_filled=2000)
        result = StrategyResult(
            fills=[fill],
            interactions=[],
            prices={},
            gas=0,
            remainder_orders=[],
        )

        first_strategy = MockStrategy(result=result, name="first")
        second_strategy = MockStrategy(result=None, name="second")
        solver = Solver(strategies=[first_strategy, second_strategy])
        auction = make_auction([order])

        solver.solve(auction)

        # First strategy should be called
        assert len(first_strategy.calls) == 1
        # Second strategy should NOT be called (no remainders)
        assert len(second_strategy.calls) == 0

    def test_remainder_orders_passed_to_next_strategy(self):
        """Remainder orders are passed to subsequent strategies."""
        order1 = make_order(uid="0x" + "01" * 56)
        order2 = make_order(uid="0x" + "02" * 56)

        # First strategy fills order1, leaves order2 as remainder
        fill1 = OrderFill(order1, sell_filled=1000, buy_filled=2000)
        result1 = StrategyResult(
            fills=[fill1],
            interactions=[],
            prices={},
            gas=0,
            remainder_orders=[order2],
        )

        first_strategy = MockStrategy(result=result1, name="first")
        second_strategy = MockStrategy(result=None, name="second")
        solver = Solver(strategies=[first_strategy, second_strategy])
        auction = make_auction([order1, order2])

        solver.solve(auction)

        # Second strategy should be called with remainder
        assert len(second_strategy.calls) == 1
        assert second_strategy.calls[0].order_count == 1
        assert second_strategy.calls[0].orders[0].uid == order2.uid

    def test_none_result_continues_to_next_strategy(self):
        """Strategies returning None are skipped."""
        first_strategy = MockStrategy(result=None, name="first")
        second_strategy = MockStrategy(result=None, name="second")
        solver = Solver(strategies=[first_strategy, second_strategy])
        auction = make_auction([make_order()])

        solver.solve(auction)

        # Both strategies should be called
        assert len(first_strategy.calls) == 1
        assert len(second_strategy.calls) == 1


class TestSolverSolutionBuilding:
    """Tests for solution building behavior."""

    def test_single_fill_combined_into_one_solution(self):
        """Single fill results in one solution."""
        order = make_order()
        fill = OrderFill(order, sell_filled=1000, buy_filled=2000)
        result = StrategyResult(
            fills=[fill],
            interactions=[],
            prices={
                order.sell_token.lower(): str(fill.buy_filled),
                order.buy_token.lower(): str(fill.sell_filled),
            },
            gas=0,
            remainder_orders=[],
        )

        mock_strategy = MockStrategy(result=result)
        solver = Solver(strategies=[mock_strategy])
        auction = make_auction([order])

        response = solver.solve(auction)

        assert len(response.solutions) == 1

    def test_cow_match_fills_combined(self):
        """CoW match (multiple fills, no interactions) combined into one solution."""
        order1 = make_order(
            uid="0x" + "01" * 56,
            sell_token="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            buy_token="0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
        order2 = make_order(
            uid="0x" + "02" * 56,
            sell_token="0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            buy_token="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )

        fill1 = OrderFill(order1, sell_filled=1000, buy_filled=2000)
        fill2 = OrderFill(order2, sell_filled=2000, buy_filled=1000)

        # CoW match: no interactions (direct peer-to-peer)
        result = StrategyResult(
            fills=[fill1, fill2],
            interactions=[],  # No AMM interactions
            prices={
                "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": "2000",
                "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": "1000",
            },
            gas=0,
            remainder_orders=[],
        )

        mock_strategy = MockStrategy(result=result)
        solver = Solver(strategies=[mock_strategy])
        auction = make_auction([order1, order2])

        response = solver.solve(auction)

        # Should be combined into one solution
        assert len(response.solutions) == 1
        assert len(response.solutions[0].trades) == 2


class TestSolverDefaultStrategies:
    """Tests for default strategy initialization."""

    def test_default_solver_has_strategies(self):
        """Default solver initializes with strategies."""
        solver = Solver()

        assert len(solver.strategies) > 0

    def test_default_solver_strategy_order(self):
        """Default solver has CowMatch, MultiPairCow, AmmRouting in order."""
        solver = Solver()

        strategy_names = [type(s).__name__ for s in solver.strategies]
        assert "CowMatchStrategy" in strategy_names
        assert "MultiPairCowStrategy" in strategy_names
        assert "AmmRoutingStrategy" in strategy_names


class TestSolverSubAuction:
    """Tests for sub-auction creation."""

    def test_sub_auction_preserves_tokens(self):
        """Sub-auction preserves token metadata from original."""
        order1 = make_order(uid="0x" + "01" * 56)
        order2 = make_order(uid="0x" + "02" * 56)

        auction = make_auction([order1, order2])
        # Add extra token metadata
        auction = auction.model_copy(
            update={
                "tokens": {
                    order1.sell_token: Token(
                        decimals=18,
                        symbol="WETH",
                        referencePrice="1000000000000000000",
                        availableBalance="999999",
                    ),
                    order1.buy_token: Token(
                        decimals=6,
                        symbol="USDC",
                        referencePrice="500000000000000000000000000",
                        availableBalance="888888",
                    ),
                }
            }
        )

        solver = Solver(strategies=[])
        sub_auction = solver._create_sub_auction(auction, [order2])

        # Tokens should be preserved
        assert sub_auction.tokens == auction.tokens
        assert sub_auction.tokens[order1.sell_token].symbol == "WETH"
        assert sub_auction.tokens[order1.buy_token].decimals == 6

    def test_sub_auction_contains_only_specified_orders(self):
        """Sub-auction contains only the specified orders."""
        order1 = make_order(uid="0x" + "01" * 56)
        order2 = make_order(uid="0x" + "02" * 56)
        order3 = make_order(uid="0x" + "03" * 56)

        auction = make_auction([order1, order2, order3])
        solver = Solver(strategies=[])

        sub_auction = solver._create_sub_auction(auction, [order2, order3])

        assert sub_auction.order_count == 2
        order_uids = [o.uid for o in sub_auction.orders]
        assert order1.uid not in order_uids
        assert order2.uid in order_uids
        assert order3.uid in order_uids
