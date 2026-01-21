"""Integration tests for the solver API."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from solver.api.main import app
from solver.models.solution import SolverResponse

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "auctions"


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


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
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",  # Reasonable limit price
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Mainnet should return solutions (if route found)
        assert len(solver_response.solutions) >= 1
