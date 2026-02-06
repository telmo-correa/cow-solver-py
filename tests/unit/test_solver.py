"""Unit tests for the Solver class."""

from decimal import Decimal
from unittest.mock import MagicMock

from solver.ebbo import EBBOValidator, EBBOViolation
from solver.models.auction import AuctionInstance, Order, OrderClass, OrderKind, Token
from solver.models.solution import LiquidityInteraction
from solver.solver import Solver, get_default_solver
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
        """Default solver has MultiPairCow, AmmRouting in order."""
        solver = Solver()

        strategy_names = [type(s).__name__ for s in solver.strategies]
        assert strategy_names == ["MultiPairCowStrategy", "AmmRoutingStrategy"]


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


class TestSolverEBBOFiltering:
    """Tests for mandatory EBBO filtering in the solver."""

    def test_ebbo_validator_filters_violations(self):
        """EBBO validator should filter out fills that violate EBBO."""
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

        # Create mock validator that reports a violation
        mock_validator = MagicMock(spec=EBBOValidator)
        mock_validator.check_clearing_prices.return_value = [
            EBBOViolation(
                sell_token=order.sell_token,
                buy_token=order.buy_token,
                clearing_rate=Decimal("0.5"),
                ebbo_rate=Decimal("1.0"),
                deficit_pct=50.0,
            )
        ]

        mock_strategy = MockStrategy(result=result)
        solver = Solver(strategies=[mock_strategy])
        solver.ebbo_validator = mock_validator

        auction = make_auction([order])
        solver.solve(auction)

        # Fill should be rejected due to EBBO violation
        # Either no solutions or solution without this fill
        # The solver should call check_clearing_prices
        mock_validator.check_clearing_prices.assert_called()

    def test_ebbo_filtering_passes_valid_fills(self):
        """EBBO validator should pass fills that don't violate EBBO."""
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

        # Create mock validator that reports no violations
        mock_validator = MagicMock(spec=EBBOValidator)
        mock_validator.check_clearing_prices.return_value = []

        mock_strategy = MockStrategy(result=result)
        solver = Solver(strategies=[mock_strategy])
        solver.ebbo_validator = mock_validator

        auction = make_auction([order])
        response = solver.solve(auction)

        # Should have one solution (fill passed validation)
        assert len(response.solutions) == 1

    def test_ebbo_filtering_rejects_all_fills(self):
        """When all fills violate EBBO, result should be filtered out."""
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

        # Create mock validator that reports a violation
        mock_validator = MagicMock(spec=EBBOValidator)
        mock_validator.check_clearing_prices.return_value = [
            EBBOViolation(
                sell_token=order.sell_token,
                buy_token=order.buy_token,
                clearing_rate=Decimal("0.5"),
                ebbo_rate=Decimal("1.0"),
                deficit_pct=50.0,
            )
        ]

        mock_strategy = MockStrategy(result=result)
        solver = Solver(strategies=[mock_strategy])
        solver.ebbo_validator = mock_validator

        auction = make_auction([order])
        response = solver.solve(auction)

        # All fills rejected, no solutions
        assert len(response.solutions) == 0

    def test_ebbo_filtering_partial_rejection(self):
        """EBBO filtering should keep valid fills while rejecting violators."""
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

        result = StrategyResult(
            fills=[fill1, fill2],
            interactions=[],
            prices={
                "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": "2000",
                "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": "1000",
            },
            gas=0,
            remainder_orders=[],
        )

        # Create mock validator that only rejects order1
        def check_prices(_prices, orders, _auction):
            if orders[0].uid == order1.uid:
                return [
                    EBBOViolation(
                        sell_token=order1.sell_token,
                        buy_token=order1.buy_token,
                        clearing_rate=Decimal("0.5"),
                        ebbo_rate=Decimal("1.0"),
                        deficit_pct=50.0,
                    )
                ]
            return []

        mock_validator = MagicMock(spec=EBBOValidator)
        mock_validator.check_clearing_prices.side_effect = check_prices

        mock_strategy = MockStrategy(result=result)
        solver = Solver(strategies=[mock_strategy])
        solver.ebbo_validator = mock_validator

        auction = make_auction([order1, order2])
        response = solver.solve(auction)

        # Only fill2 should pass, so one solution with one trade
        assert len(response.solutions) == 1
        assert len(response.solutions[0].trades) == 1

    def test_no_ebbo_validator_allows_all_fills(self):
        """When no EBBO validator is available, all fills should pass."""
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
        # Solver without ebbo_validator or liquidity
        solver = Solver(strategies=[mock_strategy])
        solver.ebbo_validator = None

        # Auction without liquidity (can't build validator)
        auction = make_auction([order])
        response = solver.solve(auction)

        # Should have one solution (no validator to filter)
        assert len(response.solutions) == 1


class TestSolverEBBOInteractionFiltering:
    """C1: EBBO filter should only keep interactions for valid fills."""

    def test_rejected_fill_drops_its_interaction(self):
        """When a fill is rejected by EBBO, its interaction should also be dropped."""
        # Use valid checksum addresses
        token_a = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
        token_b = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"
        token_c = "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"
        token_d = "0xdDdDDDDDDdDdDdDDddDDDDddDDDddDDdDdDDDDDd"

        order1 = make_order(
            uid="0x" + "01" * 56,
            sell_token=token_a,
            buy_token=token_b,
        )
        order2 = make_order(
            uid="0x" + "02" * 56,
            sell_token=token_c,
            buy_token=token_d,
        )

        fill1 = OrderFill(order1, sell_filled=1000, buy_filled=2000)
        fill2 = OrderFill(order2, sell_filled=3000, buy_filled=4000)

        # Create interactions for both fills
        interaction1 = LiquidityInteraction(
            kind="liquidity",
            id="pool1",
            input_token=token_a,
            output_token=token_b,
            input_amount="1000",
            output_amount="2000",
        )
        interaction2 = LiquidityInteraction(
            kind="liquidity",
            id="pool2",
            input_token=token_c,
            output_token=token_d,
            input_amount="3000",
            output_amount="4000",
        )

        result = StrategyResult(
            fills=[fill1, fill2],
            interactions=[interaction1, interaction2],
            prices={
                token_a.lower(): "2000",
                token_b.lower(): "1000",
                token_c.lower(): "4000",
                token_d.lower(): "3000",
            },
            gas=0,
            remainder_orders=[],
        )

        # Reject order1 only
        def check_prices(_prices, orders, _auction):
            if orders[0].uid == order1.uid:
                return [
                    EBBOViolation(
                        sell_token=order1.sell_token,
                        buy_token=order1.buy_token,
                        clearing_rate=Decimal("0.5"),
                        ebbo_rate=Decimal("1.0"),
                        deficit_pct=50.0,
                    )
                ]
            return []

        mock_validator = MagicMock(spec=EBBOValidator)
        mock_validator.check_clearing_prices.side_effect = check_prices

        mock_strategy = MockStrategy(result=result)
        solver = Solver(strategies=[mock_strategy])
        solver.ebbo_validator = mock_validator

        auction = make_auction([order1, order2])
        response = solver.solve(auction)

        # Only fill2 should survive
        assert len(response.solutions) == 1
        # The interaction for order1 should be dropped
        solution = response.solutions[0]
        for interaction in solution.interactions:
            if hasattr(interaction, "input_token"):
                # Should not have the rejected fill's interaction tokens
                assert interaction.input_token.lower() != order1.sell_token.lower()

    def test_cow_match_no_interactions_unaffected(self):
        """CoW matches with no interactions should work fine through EBBO filter."""
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

        result = StrategyResult(
            fills=[fill1, fill2],
            interactions=[],  # CoW match - no interactions
            prices={
                "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": "2000",
                "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": "1000",
            },
            gas=0,
            remainder_orders=[],
        )

        # All fills pass EBBO
        mock_validator = MagicMock(spec=EBBOValidator)
        mock_validator.check_clearing_prices.return_value = []

        mock_strategy = MockStrategy(result=result)
        solver = Solver(strategies=[mock_strategy])
        solver.ebbo_validator = mock_validator

        auction = make_auction([order1, order2])
        response = solver.solve(auction)

        assert len(response.solutions) == 1
        assert len(response.solutions[0].trades) == 2
        assert len(response.solutions[0].interactions) == 0


class TestLazySingleton:
    """H11: Lazy singleton for default solver."""

    def test_get_default_solver_returns_solver(self):
        """get_default_solver returns a Solver instance."""
        solver = get_default_solver()
        assert isinstance(solver, Solver)

    def test_get_default_solver_returns_same_instance(self):
        """get_default_solver returns the same instance on repeated calls."""
        solver1 = get_default_solver()
        solver2 = get_default_solver()
        assert solver1 is solver2
