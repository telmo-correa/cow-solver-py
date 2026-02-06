"""Tests for solver endpoint timeout (H10).

Verifies that the solve endpoint respects the auction deadline
and returns empty response when the deadline has passed.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from solver.api.main import app
from solver.models.solution import SolverResponse
from solver.solver import Solver


def make_order() -> dict:
    """Create a minimal order dict for the API."""
    return {
        "uid": "0x" + "01" * 56,
        "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "sellAmount": "1000000000000000000",
        "buyAmount": "2000000000",
        "kind": "sell",
        "class": "market",
        "partiallyFillable": False,
    }


def make_auction_payload(deadline: str | None = None) -> dict:
    """Create a minimal auction payload dict."""
    payload = {
        "id": "test-auction",
        "orders": [make_order()],
        "tokens": {
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": {
                "decimals": 18,
                "symbol": "WETH",
                "referencePrice": "1000000000000000000",
                "availableBalance": "1000000000000000000000",
            },
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": {
                "decimals": 6,
                "symbol": "USDC",
                "referencePrice": "500000000000000000000000000",
                "availableBalance": "1000000000000000000000",
            },
        },
    }
    if deadline is not None:
        payload["deadline"] = deadline
    return payload


class TestEndpointTimeout:
    """H10: Solver should respect auction deadline."""

    def test_past_deadline_returns_empty(self) -> None:
        """Request with a deadline in the past returns empty response immediately."""
        # Deadline 10 seconds ago
        past = datetime.now(UTC) - timedelta(seconds=10)
        deadline_str = past.isoformat()

        mock_solver = MagicMock(spec=Solver)
        mock_solver.solve.return_value = SolverResponse.empty()

        from solver.api.endpoints import get_solver

        app.dependency_overrides[get_solver] = lambda: mock_solver

        try:
            client = TestClient(app)
            response = client.post(
                "/production/mainnet",
                json=make_auction_payload(deadline=deadline_str),
            )

            assert response.status_code == 200
            data = response.json()
            assert data["solutions"] == []

            # Solver.solve should NOT have been called (deadline already passed)
            mock_solver.solve.assert_not_called()
        finally:
            app.dependency_overrides.clear()

    def test_no_deadline_proceeds_normally(self) -> None:
        """Request without deadline calls solver normally."""
        mock_solver = MagicMock(spec=Solver)
        mock_solver.solve.return_value = SolverResponse.empty()

        from solver.api.endpoints import get_solver

        app.dependency_overrides[get_solver] = lambda: mock_solver

        try:
            client = TestClient(app)
            response = client.post(
                "/production/mainnet",
                json=make_auction_payload(deadline=None),
            )

            assert response.status_code == 200
            # Solver.solve should have been called
            mock_solver.solve.assert_called_once()
        finally:
            app.dependency_overrides.clear()

    def test_future_deadline_proceeds_normally(self) -> None:
        """Request with future deadline calls solver normally."""
        future = datetime.now(UTC) + timedelta(seconds=30)
        deadline_str = future.isoformat()

        mock_solver = MagicMock(spec=Solver)
        mock_solver.solve.return_value = SolverResponse.empty()

        from solver.api.endpoints import get_solver

        app.dependency_overrides[get_solver] = lambda: mock_solver

        try:
            client = TestClient(app)
            response = client.post(
                "/production/mainnet",
                json=make_auction_payload(deadline=deadline_str),
            )

            assert response.status_code == 200
            mock_solver.solve.assert_called_once()
        finally:
            app.dependency_overrides.clear()
