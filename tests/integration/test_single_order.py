"""Integration tests for single order routing."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from solver.api.endpoints import get_solver
from solver.api.main import app
from solver.models.solution import SolverResponse
from solver.routing.router import SingleOrderRouter, Solver
from tests.conftest import MockAMM, MockPoolFinder, MockSwapConfig


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create a test client for the API."""
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestSingleSellOrderRouting:
    """Tests for routing single sell orders through UniswapV2."""

    def test_routes_weth_to_usdc_sell_order(self, client):
        """A WETH→USDC sell order should produce a solution with trades."""
        # WETH/USDC is a known pool in our registry
        auction = {
            "id": "test_weth_usdc",
            "tokens": {
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {
                    "decimals": 18,
                    "symbol": "WETH",
                    "availableBalance": "1000000000000000000",
                    "trusted": True,
                },
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {
                    "decimals": 6,
                    "symbol": "USDC",
                    "availableBalance": "0",
                    "trusted": True,
                },
            },
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2000000000",  # Min 2000 USDC (reasonable limit price)
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have at least one solution
        assert len(solver_response.solutions) >= 1

        solution = solver_response.solutions[0]

        # Should have exactly one trade
        assert len(solution.trades) == 1
        trade = solution.trades[0]
        assert trade.order == "0x" + "01" * 56
        assert int(trade.executed_amount) == 1000000000000000000  # Full sell amount

        # Should have at least one interaction (the AMM swap)
        assert len(solution.interactions) >= 1
        interaction = solution.interactions[0]
        assert interaction.target.startswith("0x")  # Valid address
        assert interaction.call_data.startswith("0x")  # Valid calldata

        # Should have prices for both tokens
        assert len(solution.prices) == 2

    def test_returns_empty_for_unknown_token_pair(self, client):
        """Unknown token pairs should return empty (no route found)."""
        auction = {
            "id": "test_unknown_pair",
            "orders": [
                {
                    "uid": "0x" + "02" * 56,
                    "sellToken": "0x1111111111111111111111111111111111111111",  # Unknown
                    "buyToken": "0x2222222222222222222222222222222222222222",  # Unknown
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "1000000000000000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # No solution (no pool for this pair)
        assert len(solver_response.solutions) == 0

    def test_returns_empty_when_price_too_high(self, client):
        """If limit price can't be satisfied, return empty."""
        auction = {
            "id": "test_bad_price",
            "orders": [
                {
                    "uid": "0x" + "03" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "10000000000000",  # 10M USDC (unrealistic)
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # No solution (can't meet limit price)
        assert len(solver_response.solutions) == 0

    def test_handles_reverse_direction(self, client):
        """USDC→WETH should also work (reverse of common pair)."""
        auction = {
            "id": "test_usdc_weth",
            "orders": [
                {
                    "uid": "0x" + "04" * 56,
                    "sellToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                    "buyToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                    "sellAmount": "2500000000",  # 2500 USDC
                    "buyAmount": "500000000000000000",  # Min 0.5 WETH
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should find a solution
        assert len(solver_response.solutions) >= 1
        assert len(solver_response.solutions[0].trades) == 1


class TestClearingPrices:
    """Tests for clearing price calculation."""

    def test_clearing_prices_are_valid(self, client):
        """Clearing prices should satisfy the CoW Protocol constraint."""
        auction = {
            "id": "test_prices",
            "orders": [
                {
                    "uid": "0x" + "07" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2000000000",  # Min 2000 USDC
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        assert len(data["solutions"]) == 1

        solution = data["solutions"][0]
        prices = solution["prices"]

        # Get the prices (addresses are normalized to lowercase in the solution)
        weth_addr = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
        usdc_addr = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

        assert weth_addr in prices
        assert usdc_addr in prices

        price_sell = int(prices[weth_addr])
        price_buy = int(prices[usdc_addr])

        # CoW Protocol constraint: executed_sell * price_sell >= executed_buy * price_buy
        # For this order: 1 WETH sold, getting ~2500 USDC
        # The prices should encode the exchange rate correctly

        # The sell token (WETH) should have the reference price (1e18)
        assert price_sell == 10**18

        # The buy token price is: (amount_in * PRICE_SCALE) / amount_out
        # With amount_in = 1e18 (1 WETH) and amount_out ~= 2.5e9 (2500 USDC with 6 decimals)
        # price_buy ≈ (1e18 * 1e18) / 2.5e9 ≈ 4e26
        # This large number is correct - it encodes how to balance the clearing equation
        assert price_buy > 0

        # Verify the CoW Protocol constraint holds
        # executed_sell * price_sell >= executed_buy * price_buy
        # The trade executes full sell amount (1 WETH) for the AMM output
        sell_amount = 1000000000000000000  # 1 WETH

        # For a sell order, executed_amount is the sell amount
        # The buy amount comes from the AMM calculation
        # We verify the constraint: sell * price_sell == buy * price_buy (approximately, due to rounding)

        # The clearing prices should satisfy: sell_amount * price_sell / price_buy ≈ buy_amount
        implied_buy = (sell_amount * price_sell) // price_buy
        # The implied buy amount should be close to what the AMM produces (~2492 USDC)
        assert implied_buy > 2000 * 10**6  # At least 2000 USDC (our limit price)
        assert implied_buy < 3000 * 10**6  # Less than 3000 USDC (reasonable upper bound)

    def test_solution_has_gas_estimate(self, client):
        """Solutions should include gas estimates."""
        auction = {
            "id": "test_gas",
            "orders": [
                {
                    "uid": "0x" + "08" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        data = response.json()

        assert len(data["solutions"]) == 1
        solution = data["solutions"][0]

        # Should have gas estimate
        assert "gas" in solution
        assert solution["gas"] is not None
        assert solution["gas"] > 0


class TestSolverBehavior:
    """Tests for general solver behavior."""

    def test_empty_auction_returns_empty(self, client):
        """Empty auction returns empty response."""
        auction = {"id": "empty", "orders": []}

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        assert data["solutions"] == []

    def test_multi_order_auction_returns_empty_for_now(self, client):
        """Multi-order auctions not yet supported."""
        auction = {
            "id": "multi_order",
            "orders": [
                {
                    "uid": "0x" + "05" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "limit",
                },
                {
                    "uid": "0x" + "06" * 56,
                    "sellToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "buyToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "sellAmount": "3000000000",
                    "buyAmount": "1000000000000000000",
                    "kind": "sell",
                    "class": "limit",
                },
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # Multi-order not yet implemented
        assert len(data["solutions"]) == 0


class TestRoutingWithMocks:
    """Tests using dependency injection for controlled, deterministic behavior."""

    def test_exact_output_with_mock_amm(self, mock_weth_usdc_pool):
        """Mock AMM allows testing with exact, predictable outputs."""
        # Configure mock to return exactly 3000 USDC for any swap
        mock_amm = MockAMM(MockSwapConfig(fixed_output=3000_000_000))
        mock_finder = MockPoolFinder(default_pool=mock_weth_usdc_pool)

        router = SingleOrderRouter(amm=mock_amm, pool_finder=mock_finder)
        solver = Solver(router=router)
        app.dependency_overrides[get_solver] = lambda: solver

        client = TestClient(app)
        auction = {
            "id": "test_exact_output",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2500000000",  # Min 2500 USDC
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        data = response.json()

        # Should succeed since 3000 USDC > 2500 USDC minimum
        assert len(data["solutions"]) == 1

        # The solution should reflect the exact mock output
        solution = data["solutions"][0]
        assert len(solution["trades"]) == 1

        app.dependency_overrides.clear()

    def test_limit_price_rejection_with_mock(self, mock_weth_usdc_pool):
        """Test limit price rejection with controlled mock output."""
        # Configure mock to return only 1000 USDC (below limit)
        mock_amm = MockAMM(MockSwapConfig(fixed_output=1000_000_000))
        mock_finder = MockPoolFinder(default_pool=mock_weth_usdc_pool)

        router = SingleOrderRouter(amm=mock_amm, pool_finder=mock_finder)
        solver = Solver(router=router)
        app.dependency_overrides[get_solver] = lambda: solver

        client = TestClient(app)
        auction = {
            "id": "test_limit_rejection",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2000000000",  # Min 2000 USDC
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        data = response.json()

        # Should fail since 1000 USDC < 2000 USDC minimum
        assert len(data["solutions"]) == 0

        app.dependency_overrides.clear()

    def test_no_pool_scenario_with_mock(self):
        """Test behavior when no pool is found."""
        # Pool finder that returns None for all pairs
        mock_finder = MockPoolFinder(default_pool=None)
        mock_amm = MockAMM()

        router = SingleOrderRouter(amm=mock_amm, pool_finder=mock_finder)
        solver = Solver(router=router)
        app.dependency_overrides[get_solver] = lambda: solver

        client = TestClient(app)
        auction = {
            "id": "test_no_pool",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "1",  # Very low limit to isolate pool finding
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        data = response.json()

        # Should return empty - no pool found
        assert len(data["solutions"]) == 0

        # Verify pool finder was called but AMM was not
        assert len(mock_finder.calls) == 1
        assert len(mock_amm.swap_calls) == 0

        app.dependency_overrides.clear()

    def test_swap_call_tracking(self, mock_weth_usdc_pool):
        """Verify we can inspect mock calls for detailed assertions."""
        mock_amm = MockAMM(MockSwapConfig(fixed_output=3000_000_000))
        mock_finder = MockPoolFinder(default_pool=mock_weth_usdc_pool)

        router = SingleOrderRouter(amm=mock_amm, pool_finder=mock_finder)
        solver = Solver(router=router)
        app.dependency_overrides[get_solver] = lambda: solver

        client = TestClient(app)
        sell_amount = 2_500_000_000_000_000_000  # 2.5 WETH
        auction = {
            "id": "test_tracking",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": str(sell_amount),
                    "buyAmount": "1",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        # Verify the exact parameters passed to the mock
        assert len(mock_amm.swap_calls) == 1
        call = mock_amm.swap_calls[0]
        assert call["amount_in"] == sell_amount
        assert call["pool"] == mock_weth_usdc_pool.address

        # Verify pool finder received the correct tokens
        assert len(mock_finder.calls) == 1
        token_a, token_b = mock_finder.calls[0]
        assert "c02aaa39b223fe8d0a0e5c4f27ead9083c756cc2" in token_a.lower()
        assert "a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48" in token_b.lower()

        app.dependency_overrides.clear()
