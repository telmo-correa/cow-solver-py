"""Unit tests for API error handling."""

import pytest
from fastapi.testclient import TestClient

from solver.api.endpoints import get_solver
from solver.api.main import app
from solver.models.auction import AuctionInstance
from solver.models.solution import SolverResponse


@pytest.fixture
def client():
    """Create a test client for the API."""
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestSolverExceptionHandling:
    """Tests for solver exception handling."""

    def test_solver_exception_returns_empty_response(self, client):
        """Solver raising exception returns empty solutions, not 500."""

        class ExplodingSolver:
            """Mock solver that always raises."""

            def solve(self, _auction: AuctionInstance) -> SolverResponse:
                raise RuntimeError("Boom! This should be caught.")

        app.dependency_overrides[get_solver] = lambda: ExplodingSolver()

        try:
            auction = {
                "id": "test",
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

            response = client.post("/staging/mainnet", json=auction)

            # Should return 200 with empty solutions, not 500
            assert response.status_code == 200
            data = response.json()
            assert data["solutions"] == []
        finally:
            app.dependency_overrides.clear()


class TestInvalidJsonSchema:
    """Tests for invalid JSON schema handling."""

    def test_missing_orders_field_defaults_to_empty(self, client):
        """Request missing 'orders' field defaults to empty list (200 OK)."""
        auction = {"id": "test"}  # Missing 'orders' - defaults to []

        response = client.post("/staging/mainnet", json=auction)

        # Orders defaults to [], so request succeeds with empty auction
        assert response.status_code == 200
        data = response.json()
        assert data["solutions"] == []

    def test_orders_not_a_list(self, client):
        """Request with orders as string returns 422."""
        auction = {"id": "test", "orders": "not-a-list"}

        response = client.post("/staging/mainnet", json=auction)

        assert response.status_code == 422

    def test_order_missing_required_fields(self, client):
        """Order missing required fields returns 422."""
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    # Missing sellToken, buyToken, sellAmount, buyAmount, kind, class
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        assert response.status_code == 422
        data = response.json()
        # Should indicate which fields are missing
        assert "detail" in data


class TestInvalidTokenAddresses:
    """Tests for invalid token address handling."""

    def test_invalid_hex_address_format(self, client):
        """Token address not starting with 0x returns 422."""
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "not-a-hex-address",  # Invalid
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        assert response.status_code == 422

    def test_address_wrong_length(self, client):
        """Token address with wrong length returns 422."""
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD908",  # Too short
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        assert response.status_code == 422


class TestInvalidOrderAmounts:
    """Tests for invalid order amount handling."""

    def test_negative_sell_amount(self, client):
        """Negative sell amount returns 422."""
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "-1000000000000000000",  # Negative
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_non_numeric_sell_amount(self, client):
        """Non-numeric sell amount returns 422."""
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "not-a-number",  # Invalid
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        assert response.status_code == 422

    def test_negative_buy_amount(self, client):
        """Negative buy amount returns 422."""
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "-500",  # Negative
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        assert response.status_code == 422


class TestInvalidOrderKind:
    """Tests for invalid order kind handling."""

    def test_invalid_kind_value(self, client):
        """Invalid order kind returns 422."""
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",
                    "kind": "invalid-kind",  # Invalid
                    "class": "limit",
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        assert response.status_code == 422


class TestInvalidOrderClass:
    """Tests for invalid order class handling."""

    def test_invalid_class_value(self, client):
        """Invalid order class returns 422."""
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "invalid-class",  # Invalid
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        assert response.status_code == 422


class TestInvalidUid:
    """Tests for invalid order UID handling."""

    def test_uid_wrong_length(self, client):
        """UID with wrong length returns 422."""
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x0123456789",  # Too short (should be 0x + 112 hex chars)
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        assert response.status_code == 422


class TestMalformedJson:
    """Tests for malformed JSON handling."""

    def test_invalid_json_syntax(self, client):
        """Invalid JSON syntax returns 422."""
        response = client.post(
            "/staging/mainnet",
            content="{invalid json}",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_empty_body(self, client):
        """Empty request body returns 422."""
        response = client.post(
            "/staging/mainnet",
            content="",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


class TestValidEdgeCases:
    """Tests for edge cases that should be accepted."""

    def test_accepts_zero_sell_amount(self, client):
        """Zero sell amount is accepted (validation at solve time)."""
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "0",  # Zero is valid uint256
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        # Request accepted, but solver may return empty (business logic)
        assert response.status_code == 200

    def test_accepts_very_large_amounts(self, client):
        """Very large amounts are accepted if valid uint256."""
        max_uint256 = str(2**256 - 1)
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": max_uint256,  # Max uint256
                    "buyAmount": "1",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        # Request accepted (solver handles business logic)
        assert response.status_code == 200

    def test_accepts_lowercase_addresses(self, client):
        """Lowercase addresses are accepted."""
        auction = {
            "id": "test",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # Lowercase
                    "buyToken": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # Lowercase
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/staging/mainnet", json=auction)

        assert response.status_code == 200
