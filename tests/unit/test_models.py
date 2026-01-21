"""Tests for Pydantic models."""

import json
from pathlib import Path

import pytest

from solver.models.auction import AuctionInstance, Order, OrderKind, Token
from solver.models.solution import Solution, SolverResponse


class TestToken:
    """Tests for Token model."""

    def test_parse_minimal(self):
        """Token can be parsed with minimal fields."""
        data = {"availableBalance": "1000000"}
        token = Token.model_validate(data)
        assert token.available_balance == "1000000"
        assert token.decimals is None
        assert token.trusted is False

    def test_parse_full(self):
        """Token can be parsed with all fields."""
        data = {
            "decimals": 18,
            "symbol": "WETH",
            "referencePrice": "1000000000000000000",
            "availableBalance": "5000000000000000000",
            "trusted": True,
        }
        token = Token.model_validate(data)
        assert token.decimals == 18
        assert token.symbol == "WETH"
        assert token.reference_price == "1000000000000000000"
        assert token.trusted is True


class TestOrder:
    """Tests for Order model."""

    def test_parse_sell_order(self):
        """Sell order can be parsed correctly."""
        data = {
            "uid": "0x" + "01" * 56,
            "sellToken": "0x" + "aa" * 20,
            "buyToken": "0x" + "bb" * 20,
            "sellAmount": "1000000000000000000",
            "buyAmount": "2500000000",
            "kind": "sell",
            "class": "limit",
        }
        order = Order.model_validate(data)
        assert order.is_sell_order
        assert not order.is_buy_order
        assert order.kind == OrderKind.SELL
        assert not order.partially_fillable

    def test_parse_buy_order(self):
        """Buy order can be parsed correctly."""
        data = {
            "uid": "0x" + "02" * 56,
            "sellToken": "0x" + "aa" * 20,
            "buyToken": "0x" + "bb" * 20,
            "sellAmount": "1000000000000000000",
            "buyAmount": "2500000000",
            "kind": "buy",
            "class": "market",
            "partiallyFillable": True,
        }
        order = Order.model_validate(data)
        assert order.is_buy_order
        assert not order.is_sell_order
        assert order.partially_fillable

    def test_limit_price(self):
        """Limit price is calculated correctly."""
        data = {
            "uid": "0x" + "03" * 56,
            "sellToken": "0x" + "aa" * 20,
            "buyToken": "0x" + "bb" * 20,
            "sellAmount": "1000000000000000000",  # 1e18
            "buyAmount": "2500000000",  # 2.5e9
            "kind": "sell",
            "class": "limit",
        }
        order = Order.model_validate(data)
        # limit_price = buy / sell = 2.5e9 / 1e18 = 2.5e-9
        assert order.limit_price == pytest.approx(2.5e-9)


class TestAuctionInstance:
    """Tests for AuctionInstance model."""

    def test_load_from_fixture(self):
        """AuctionInstance can be loaded from fixture file."""
        fixture_path = (
            Path(__file__).parent.parent / "fixtures/auctions/single_order/basic_sell.json"
        )
        if not fixture_path.exists():
            pytest.skip("Fixture not found")

        with open(fixture_path) as f:
            data = json.load(f)

        auction = AuctionInstance.model_validate(data)
        assert auction.id == "test_single_sell_001"
        assert auction.order_count == 1
        assert len(auction.tokens) == 2

    def test_token_pairs(self):
        """token_pairs property returns unique pairs."""
        data = {
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xAAAA" + "00" * 18,
                    "buyToken": "0xBBBB" + "00" * 18,
                    "sellAmount": "100",
                    "buyAmount": "200",
                    "kind": "sell",
                    "class": "limit",
                },
                {
                    "uid": "0x" + "02" * 56,
                    "sellToken": "0xBBBB" + "00" * 18,
                    "buyToken": "0xAAAA" + "00" * 18,
                    "sellAmount": "200",
                    "buyAmount": "100",
                    "kind": "sell",
                    "class": "limit",
                },
            ]
        }
        auction = AuctionInstance.model_validate(data)
        # Both orders trade between same pair (just opposite directions)
        assert len(auction.token_pairs) == 1

    def test_orders_for_pair(self):
        """orders_for_pair returns correct orders."""
        token_a = "0xAAAA" + "00" * 18
        token_b = "0xBBBB" + "00" * 18
        token_c = "0xCCCC" + "00" * 18

        data = {
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": token_a,
                    "buyToken": token_b,
                    "sellAmount": "100",
                    "buyAmount": "200",
                    "kind": "sell",
                    "class": "limit",
                },
                {
                    "uid": "0x" + "02" * 56,
                    "sellToken": token_a,
                    "buyToken": token_c,
                    "sellAmount": "100",
                    "buyAmount": "300",
                    "kind": "sell",
                    "class": "limit",
                },
            ]
        }
        auction = AuctionInstance.model_validate(data)

        ab_orders = auction.orders_for_pair(token_a, token_b)
        assert len(ab_orders) == 1
        assert ab_orders[0].buy_token.lower() == token_b.lower()


class TestSolution:
    """Tests for Solution model."""

    def test_empty_solution(self):
        """Empty solution can be created."""
        solution = Solution.empty()
        assert solution.id == 0
        assert len(solution.prices) == 0
        assert len(solution.trades) == 0
        assert len(solution.interactions) == 0

    def test_solution_with_trade(self):
        """Solution with trade can be created."""
        data = {
            "id": 1,
            "prices": {
                "0x" + "aa" * 20: "1000000000000000000",
                "0x" + "bb" * 20: "2500000000",
            },
            "trades": [
                {
                    "order": "0x" + "01" * 56,
                    "executedAmount": "1000000000000000000",
                }
            ],
        }
        solution = Solution.model_validate(data)
        assert solution.id == 1
        assert len(solution.prices) == 2
        assert len(solution.trades) == 1


class TestSolverResponse:
    """Tests for SolverResponse model."""

    def test_empty_response(self):
        """Empty response can be created."""
        response = SolverResponse.empty()
        assert len(response.solutions) == 0

    def test_response_with_empty_solution(self):
        """Response with empty solution can be created."""
        response = SolverResponse.with_empty_solution()
        assert len(response.solutions) == 1
        assert response.solutions[0].id == 0

    def test_serialize_response(self):
        """Response can be serialized to JSON."""
        response = SolverResponse(
            solutions=[
                Solution(
                    id=0,
                    prices={"0x" + "aa" * 20: "1000000"},
                    trades=[],
                )
            ]
        )
        json_str = response.model_dump_json(by_alias=True)
        data = json.loads(json_str)
        assert "solutions" in data
        assert len(data["solutions"]) == 1
