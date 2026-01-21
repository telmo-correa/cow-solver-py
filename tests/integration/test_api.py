"""Integration tests for the solver API."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from solver.api.endpoints import get_solver
from solver.api.main import app
from solver.models.solution import SolverResponse
from solver.solver import Solver

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "auctions"

# Token addresses
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"


def make_weth_usdc_pool_liquidity():
    """Create liquidity data for WETH/USDC pool."""
    return {
        "kind": "constantProduct",
        "id": "uniswap-v2-weth-usdc",
        "address": "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
        "router": "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
        "gasEstimate": "110000",
        "tokens": {
            USDC: {"balance": "50000000000000"},  # 50M USDC
            WETH: {"balance": "20000000000000000000000"},  # 20K WETH
        },
        "fee": "0.003",
    }


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create a test client for the API."""
    # Ensure dependency overrides are cleared after test
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_with_mock_solver(mock_router_success) -> Iterator[TestClient]:
    """Create a test client with an injected mock solver."""
    mock_solver = Solver(router=mock_router_success)
    app.dependency_overrides[get_solver] = lambda: mock_solver
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestSolveEndpoint:
    """Tests for the POST /{environment}/{network} endpoint."""

    def test_health_check(self, client):
        """Health endpoint returns ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_accepts_empty_auction(self, client):
        """Endpoint accepts an auction with no orders."""
        auction = {"id": "test_empty", "orders": [], "tokens": {}}
        response = client.post("/staging/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        assert "solutions" in data

    def test_accepts_single_order_auction(self, client):
        """Endpoint accepts a single-order auction from fixture."""
        fixture_path = FIXTURES_DIR / "single_order" / "basic_sell.json"
        with open(fixture_path) as f:
            auction = json.load(f)

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)
        # Currently returns empty, which is valid
        assert isinstance(solver_response.solutions, list)

    def test_accepts_cow_pair_auction(self, client):
        """Endpoint accepts a CoW pair auction from fixture."""
        fixture_path = FIXTURES_DIR / "cow_pairs" / "basic_cow.json"
        with open(fixture_path) as f:
            auction = json.load(f)

        response = client.post("/shadow/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)
        assert isinstance(solver_response.solutions, list)

    def test_returns_valid_response_format(self, client):
        """Response matches the expected SolverResponse schema."""
        auction = {
            "id": "test_format",
            "tokens": {
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {
                    "availableBalance": "1000000000000000000",
                    "trusted": True,
                }
            },
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2500000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)
        assert response.status_code == 200

        # Validate response can be parsed
        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Response should have solutions list (even if empty)
        assert hasattr(solver_response, "solutions")

    def test_unsupported_network_returns_empty(self, client):
        """Unsupported networks should return empty solutions."""
        auction = {
            "id": "test_unsupported_network",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2500000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        # Test various unsupported networks
        for network in ["arbitrum-one", "gnosis", "sepolia", "polygon"]:
            response = client.post(f"/production/{network}", json=auction)
            assert response.status_code == 200

            data = response.json()
            solver_response = SolverResponse.model_validate(data)

            # Should return empty solutions for unsupported networks
            assert len(solver_response.solutions) == 0, f"Expected empty for {network}"

    def test_mainnet_is_supported(self, client):
        """Mainnet should be a supported network."""
        auction = {
            "id": "test_mainnet_supported",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",  # Reasonable limit price
                    "kind": "sell",
                    "class": "limit",
                }
            ],
            "liquidity": [make_weth_usdc_pool_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Mainnet should return solutions (if route found)
        assert len(solver_response.solutions) >= 1


class TestDependencyInjection:
    """Tests demonstrating FastAPI dependency injection for the solver."""

    def test_can_override_solver_dependency(self, injected_solver):
        """Verify solver can be overridden via FastAPI dependency_overrides."""
        app.dependency_overrides[get_solver] = lambda: injected_solver

        client = TestClient(app)
        auction = {
            "id": "test_di",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
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
        assert response.status_code == 200

        # Clean up
        app.dependency_overrides.clear()

    def test_mock_solver_controls_response(self, mock_router_failure):
        """Mock solver can force specific behavior (e.g., always fail)."""
        failing_solver = Solver(router=mock_router_failure)
        app.dependency_overrides[get_solver] = lambda: failing_solver

        client = TestClient(app)
        auction = {
            "id": "test_fail",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
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
        assert response.status_code == 200

        data = response.json()
        # Mock router always fails, so no solutions
        assert len(data["solutions"]) == 0

        # Verify the mock was called
        assert len(mock_router_failure.route_calls) == 1

        # Clean up
        app.dependency_overrides.clear()

    def test_injected_solver_uses_mock_amm(self, mock_amm, mock_pool_finder):
        """Verify mock AMM is called when using injected solver."""
        from solver.routing.router import SingleOrderRouter

        router = SingleOrderRouter(amm=mock_amm, pool_finder=mock_pool_finder)
        solver = Solver(router=router)
        app.dependency_overrides[get_solver] = lambda: solver

        client = TestClient(app)
        auction = {
            "id": "test_mock_amm",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2000000000",  # 2000 USDC
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        # Verify mock AMM was called
        assert len(mock_amm.swap_calls) == 1
        assert mock_amm.swap_calls[0]["amount_in"] == 1000000000000000000

        # Verify mock pool finder was called
        assert len(mock_pool_finder.calls) == 1

        data = response.json()
        # Should have a solution since mock AMM returns realistic values
        assert len(data["solutions"]) == 1

        # Clean up
        app.dependency_overrides.clear()
