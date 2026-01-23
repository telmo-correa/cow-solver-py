"""Integration tests for 0x limit order routing through the API."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from solver.api.main import app
from solver.models.solution import SolverResponse

# Standard token addresses
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"


def make_limit_order_liquidity(
    maker_amount: int = 2500_000_000,  # 2500 USDC
    taker_amount: int = 1_000_000_000_000_000_000,  # 1 WETH
) -> dict:
    """Create a limit order liquidity entry.

    Default: Offers 2500 USDC for 1 WETH (rate: 2500 USDC/WETH)
    """
    return {
        "kind": "limitOrder",
        "id": "limit-order-0",
        "address": "0xdef1c0ded9bec7f1a1670819833240f027b25eff",
        "makerToken": USDC,  # What the maker provides (output for us)
        "takerToken": WETH,  # What the maker wants (input for us)
        "makerAmount": str(maker_amount),
        "takerAmount": str(taker_amount),
        "takerTokenFeeAmount": "0",
        "gasEstimate": "66358",
    }


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create a test client for the API."""
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestLimitOrderSellRouting:
    """Tests for routing sell orders through limit orders."""

    def test_routes_weth_to_usdc_through_limit_order(self, client):
        """A WETH->USDC sell order routes through a matching limit order."""
        auction = {
            "id": "limit_sell_test",
            "tokens": {
                WETH: {
                    "decimals": 18,
                    "symbol": "WETH",
                    "availableBalance": "1000000000000000000",
                    "trusted": True,
                },
                USDC: {
                    "decimals": 6,
                    "symbol": "USDC",
                    "availableBalance": "0",
                    "trusted": True,
                },
            },
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "500000000000000000",  # 0.5 WETH
                    "buyAmount": "1000000000",  # Min 1000 USDC
                    "kind": "sell",
                    "class": "market",
                }
            ],
            "liquidity": [make_limit_order_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have a solution
        assert len(solver_response.solutions) >= 1

        solution = solver_response.solutions[0]

        # Should have one trade
        assert len(solution.trades) == 1
        trade = solution.trades[0]
        assert trade.order == "0x" + "01" * 56
        # Sell order: executed amount is the sell amount
        assert int(trade.executed_amount) == 500_000_000_000_000_000

        # Should have one interaction (limit order)
        assert len(solution.interactions) == 1
        interaction = solution.interactions[0]
        assert interaction.kind == "liquidity"
        assert interaction.id == "limit-order-0"

        # Should have prices for both tokens
        assert len(solution.prices) == 2

    def test_partial_fill_against_limit_order(self, client):
        """Order can partially fill against limit order when liquidity is limited."""
        auction = {
            "id": "limit_partial_fill",
            "tokens": {
                WETH: {
                    "decimals": 18,
                    "symbol": "WETH",
                    "availableBalance": "2000000000000000000",
                    "trusted": True,
                },
                USDC: {
                    "decimals": 6,
                    "symbol": "USDC",
                    "availableBalance": "0",
                    "trusted": True,
                },
            },
            "orders": [
                {
                    "uid": "0x" + "02" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "2000000000000000000",  # 2 WETH (> limit order's 1 WETH capacity)
                    "buyAmount": "4000000000",  # Min 4000 USDC
                    "kind": "sell",
                    "class": "market",
                    "partiallyFillable": True,
                }
            ],
            # Limit order only accepts 1 WETH
            "liquidity": [make_limit_order_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have a solution (partial fill)
        assert len(solver_response.solutions) >= 1

        solution = solver_response.solutions[0]
        assert len(solution.trades) == 1

        trade = solution.trades[0]
        # Only 1 WETH filled (limit order capacity)
        assert int(trade.executed_amount) == 1_000_000_000_000_000_000

    def test_limit_price_not_satisfied_no_solution(self, client):
        """No solution when limit price cannot be satisfied."""
        auction = {
            "id": "limit_price_fail",
            "tokens": {
                WETH: {
                    "decimals": 18,
                    "symbol": "WETH",
                    "availableBalance": "1000000000000000000",
                    "trusted": True,
                },
                USDC: {
                    "decimals": 6,
                    "symbol": "USDC",
                    "availableBalance": "0",
                    "trusted": True,
                },
            },
            "orders": [
                {
                    "uid": "0x" + "03" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    # Limit order offers 2500 USDC/WETH, but order wants 3000
                    "buyAmount": "3000000000",  # Min 3000 USDC (too high)
                    "kind": "sell",
                    "class": "market",
                }
            ],
            "liquidity": [make_limit_order_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # No solution (limit price not satisfiable)
        assert len(solver_response.solutions) == 0


class TestLimitOrderBuyRouting:
    """Tests for routing buy orders through limit orders."""

    def test_routes_buy_order_through_limit_order(self, client):
        """A buy order routes through a matching limit order."""
        auction = {
            "id": "limit_buy_test",
            "tokens": {
                WETH: {
                    "decimals": 18,
                    "symbol": "WETH",
                    "availableBalance": "1000000000000000000",
                    "trusted": True,
                },
                USDC: {
                    "decimals": 6,
                    "symbol": "USDC",
                    "availableBalance": "0",
                    "trusted": True,
                },
            },
            "orders": [
                {
                    "uid": "0x" + "04" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",  # Max 1 WETH
                    "buyAmount": "1250000000",  # Want exactly 1250 USDC
                    "kind": "buy",
                    "class": "market",
                }
            ],
            "liquidity": [make_limit_order_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have a solution
        assert len(solver_response.solutions) >= 1

        solution = solver_response.solutions[0]

        # Should have one trade
        assert len(solution.trades) == 1
        trade = solution.trades[0]
        assert trade.order == "0x" + "04" * 56
        # Buy order: executed amount is the buy amount
        assert int(trade.executed_amount) == 1250_000_000

        # Should have one interaction
        assert len(solution.interactions) == 1

    def test_buy_order_fails_when_input_exceeds_limit(self, client):
        """Buy order fails when required input exceeds sell limit."""
        auction = {
            "id": "limit_buy_too_expensive",
            "tokens": {
                WETH: {
                    "decimals": 18,
                    "symbol": "WETH",
                    "availableBalance": "1000000000000000000",
                    "trusted": True,
                },
                USDC: {
                    "decimals": 6,
                    "symbol": "USDC",
                    "availableBalance": "0",
                    "trusted": True,
                },
            },
            "orders": [
                {
                    "uid": "0x" + "05" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "100000000000000000",  # Max 0.1 WETH
                    "buyAmount": "1000000000",  # Want 1000 USDC (requires 0.4 WETH)
                    "kind": "buy",
                    "class": "market",
                }
            ],
            "liquidity": [make_limit_order_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # No solution (would require 0.4 WETH but only 0.1 WETH available)
        assert len(solver_response.solutions) == 0


class TestLimitOrderDirectionality:
    """Tests for limit order directionality (unidirectional)."""

    def test_reverse_direction_no_route(self, client):
        """Limit orders are unidirectional - reverse direction has no route."""
        # Limit order: accepts WETH, provides USDC
        # Order: wants to sell USDC for WETH (opposite direction)
        auction = {
            "id": "limit_reverse_direction",
            "tokens": {
                WETH: {
                    "decimals": 18,
                    "symbol": "WETH",
                    "availableBalance": "0",
                    "trusted": True,
                },
                USDC: {
                    "decimals": 6,
                    "symbol": "USDC",
                    "availableBalance": "3000000000",
                    "trusted": True,
                },
            },
            "orders": [
                {
                    "uid": "0x" + "06" * 56,
                    "sellToken": USDC,  # Selling USDC
                    "buyToken": WETH,  # Buying WETH
                    "sellAmount": "2500000000",  # 2500 USDC
                    "buyAmount": "500000000000000000",  # Want 0.5 WETH
                    "kind": "sell",
                    "class": "market",
                }
            ],
            # Limit order accepts WETH->USDC, NOT USDC->WETH
            "liquidity": [make_limit_order_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # No solution (wrong direction)
        assert len(solver_response.solutions) == 0


class TestLimitOrderWithOtherLiquidity:
    """Tests for limit orders combined with AMM liquidity."""

    def test_limit_order_beats_amm_when_better(self, client):
        """Limit order is chosen when it offers better price than AMM."""
        auction = {
            "id": "limit_vs_amm",
            "tokens": {
                WETH: {
                    "decimals": 18,
                    "symbol": "WETH",
                    "availableBalance": "1000000000000000000",
                    "trusted": True,
                },
                USDC: {
                    "decimals": 6,
                    "symbol": "USDC",
                    "availableBalance": "0",
                    "trusted": True,
                },
            },
            "orders": [
                {
                    "uid": "0x" + "07" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "500000000000000000",  # 0.5 WETH
                    "buyAmount": "1000000000",  # Min 1000 USDC
                    "kind": "sell",
                    "class": "market",
                }
            ],
            "liquidity": [
                # Limit order: 2500 USDC/WETH (better)
                make_limit_order_liquidity(),
                # AMM pool: ~2000 USDC/WETH after slippage
                {
                    "kind": "constantProduct",
                    "id": "v2-pool-0",
                    "address": "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
                    "router": "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
                    "gasEstimate": "110000",
                    "tokens": {
                        USDC: {"balance": "10000000000"},  # 10,000 USDC
                        WETH: {"balance": "5000000000000000000"},  # 5 WETH
                    },
                    "fee": "0.003",
                },
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        assert len(solver_response.solutions) >= 1
        solution = solver_response.solutions[0]

        # Should choose limit order (better rate)
        assert len(solution.interactions) == 1
        assert solution.interactions[0].id == "limit-order-0"


class TestBenchmarkFixtures:
    """Tests using the benchmark fixtures."""

    def test_limit_order_sell_fixture(self, client):
        """Test the limit_order_sell.json benchmark fixture."""
        import json
        from pathlib import Path

        fixture_path = (
            Path(__file__).parent.parent / "fixtures/auctions/benchmark/limit_order_sell.json"
        )
        with open(fixture_path) as f:
            auction = json.load(f)

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        assert len(data["solutions"]) >= 1

        solution = data["solutions"][0]
        assert len(solution["trades"]) == 1
        # 0.5 WETH sell at 2500 USDC/WETH = 1250 USDC output
        # Verify the trade executed
        trade = solution["trades"][0]
        assert int(trade["executedAmount"]) == 500_000_000_000_000_000

    def test_limit_order_buy_fixture(self, client):
        """Test the limit_order_buy.json benchmark fixture."""
        import json
        from pathlib import Path

        fixture_path = (
            Path(__file__).parent.parent / "fixtures/auctions/benchmark/limit_order_buy.json"
        )
        with open(fixture_path) as f:
            auction = json.load(f)

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        assert len(data["solutions"]) >= 1

        solution = data["solutions"][0]
        assert len(solution["trades"]) == 1
        # Buy 1250 USDC at 2500 USDC/WETH rate = 0.5 WETH input
        trade = solution["trades"][0]
        # For buy orders, executed_amount is the buy amount
        assert int(trade["executedAmount"]) == 1250_000_000
