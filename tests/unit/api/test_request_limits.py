"""Tests for API request size limits."""

from fastapi.testclient import TestClient

from solver.api.main import app


class TestRequestSizeLimits:
    """Fix 12: Request body size limit."""

    def test_oversized_request_returns_413(self):
        """Request with Content-Length exceeding limit returns 413."""
        client = TestClient(app)
        # Send request with large Content-Length header
        response = client.post(
            "/staging/mainnet",
            json={"id": "test", "orders": []},
            headers={"Content-Length": str(20 * 1024 * 1024)},  # 20 MB
        )
        assert response.status_code == 413
        assert response.json()["detail"] == "Request too large"

    def test_normal_request_accepted(self):
        """Normal-sized request is accepted."""
        client = TestClient(app)
        response = client.post(
            "/staging/mainnet",
            json={"id": "test", "orders": []},
        )
        assert response.status_code == 200


class TestHealthEndpoint:
    """Fix 28: Health endpoint improvement."""

    def test_health_returns_ok(self):
        """Health endpoint returns ok status."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "scipy_available" in data

    def test_health_shows_scipy_status(self):
        """Health endpoint reports scipy availability."""
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["scipy_available"], bool)
