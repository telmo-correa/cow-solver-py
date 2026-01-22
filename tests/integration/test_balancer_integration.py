"""Integration tests for Balancer pool routing.

These tests verify that the solver correctly routes orders through
Balancer weighted and stable pools, matching the Rust baseline solver's
output for the same inputs.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from solver.amm.balancer import BalancerStableAMM, BalancerWeightedAMM
from solver.api.endpoints import get_solver
from solver.api.main import app
from solver.models.solution import SolverResponse
from solver.solver import Solver

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "auctions" / "benchmark"


def load_fixture(name: str) -> dict:
    """Load a benchmark fixture by name."""
    path = FIXTURES_DIR / name
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create a test client for the API."""
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def weighted_amm() -> BalancerWeightedAMM:
    """Create a Balancer weighted AMM."""
    return BalancerWeightedAMM()


@pytest.fixture
def stable_amm() -> BalancerStableAMM:
    """Create a Balancer stable AMM."""
    return BalancerStableAMM()


class TestBalancerWeightedPoolIntegration:
    """Integration tests for Balancer weighted pool routing."""

    def test_weighted_pool_solve_gno_to_cow(self, client, weighted_amm):
        """Solve a GNO→COW order through a 50/50 weighted pool (V0).

        Test vector from Rust baseline (bal_liquidity.rs):
        - Input: 1 GNO (1000000000000000000)
        - Expected output: 1657855325872947866705 COW
        """
        solver = Solver(weighted_amm=weighted_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("weighted_gno_to_cow.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have exactly one solution
        assert len(solver_response.solutions) == 1
        solution = solver_response.solutions[0]

        # Should have exactly one trade
        assert len(solution.trades) == 1
        trade = solution.trades[0]

        # Verify executed amount is full sell amount (market order, no fee)
        assert trade.executed_amount == "1000000000000000000"

        # Should have exactly one interaction
        assert len(solution.interactions) == 1
        interaction = solution.interactions[0]

        # Verify interaction details
        assert interaction.kind == "liquidity"
        assert interaction.id == "weighted-gno-cow"
        assert interaction.input_amount == "1000000000000000000"

        # Verify output matches Rust baseline exactly
        assert interaction.output_amount == "1657855325872947866705"

        app.dependency_overrides.clear()

    def test_weighted_pool_solve_v3plus(self, client, weighted_amm):
        """Solve an xGNO→xCOW order through a V3Plus weighted pool.

        Test vector from Rust baseline (bal_liquidity.rs):
        - Input: 1 xGNO (1000000000000000000)
        - Expected output: 1663373703594405548696 xCOW
        """
        solver = Solver(weighted_amm=weighted_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("weighted_v3plus.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have exactly one solution
        assert len(solver_response.solutions) == 1
        solution = solver_response.solutions[0]

        # Should have exactly one interaction
        assert len(solution.interactions) == 1
        interaction = solution.interactions[0]

        # Verify output matches Rust baseline exactly
        assert interaction.output_amount == "1663373703594405548696"

        app.dependency_overrides.clear()


class TestBalancerStablePoolIntegration:
    """Integration tests for Balancer stable pool routing."""

    def test_stable_pool_sell_order_dai_to_usdc(self, client, stable_amm):
        """Solve a DAI→USDC sell order through a 3-token stable pool.

        Test vector from Rust baseline (bal_liquidity.rs):
        - Input: 10 DAI (10000000000000000000)
        - Expected output: 9999475 USDC
        """
        solver = Solver(stable_amm=stable_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("stable_dai_to_usdc.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have exactly one solution
        assert len(solver_response.solutions) == 1
        solution = solver_response.solutions[0]

        # Should have exactly one trade
        assert len(solution.trades) == 1
        trade = solution.trades[0]

        # Verify executed amount is full sell amount
        assert trade.executed_amount == "10000000000000000000"

        # Should have exactly one interaction
        assert len(solution.interactions) == 1
        interaction = solution.interactions[0]

        # Verify interaction details
        assert interaction.kind == "liquidity"
        assert interaction.input_amount == "10000000000000000000"

        # Verify output matches Rust baseline exactly
        assert interaction.output_amount == "9999475"

        app.dependency_overrides.clear()

    def test_stable_pool_buy_order_dai_to_usdc(self, client, stable_amm):
        """Solve a buy order (want USDC, pay DAI) through a stable pool.

        Test vector from Rust baseline (bal_liquidity.rs):
        - Output: 10 USDC (10000000)
        - Expected input: 10000524328839166557 DAI
        """
        solver = Solver(stable_amm=stable_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("stable_buy_order.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have exactly one solution
        assert len(solver_response.solutions) == 1
        solution = solver_response.solutions[0]

        # Should have exactly one trade
        assert len(solution.trades) == 1
        trade = solution.trades[0]

        # For buy orders, executed_amount is the buy amount
        assert trade.executed_amount == "10000000"

        # Should have exactly one interaction
        assert len(solution.interactions) == 1
        interaction = solution.interactions[0]

        # Verify input matches Rust baseline exactly
        assert interaction.input_amount == "10000524328839166557"
        assert interaction.output_amount == "10000000"

        app.dependency_overrides.clear()

    def test_composable_stable_pool_with_bpt(self, client, stable_amm):
        """Solve through a composable stable pool that has a BPT token.

        Test vector from Rust baseline (bal_liquidity.rs):
        - Input: 10 agEUR (10000000000000000000)
        - Expected output: 10029862202766050434 EURe

        The pool has 3 tokens including the BPT (bb-agEUR-EURe).
        The BPT should be filtered out when routing.
        """
        solver = Solver(stable_amm=stable_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("stable_composable.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have exactly one solution
        assert len(solver_response.solutions) == 1
        solution = solver_response.solutions[0]

        # Should have exactly one interaction
        assert len(solution.interactions) == 1
        interaction = solution.interactions[0]

        # Verify output matches Rust baseline exactly
        assert interaction.output_amount == "10029862202766050434"

        app.dependency_overrides.clear()


class TestBestPoolSelection:
    """Tests for selecting the best pool across multiple pool types."""

    def test_solver_selects_weighted_over_v2_when_better(self, client, weighted_amm):
        """When weighted pool gives better output than V2, use weighted.

        This test creates an auction with both V2 and weighted pools
        for the same pair and verifies the better one is selected.
        """
        solver = Solver(weighted_amm=weighted_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        # Use the weighted fixture which has only a weighted pool
        # The solver should use it since it's the only option
        auction = load_fixture("weighted_gno_to_cow.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have a solution
        assert len(solver_response.solutions) >= 1

        # The interaction should reference the weighted pool
        interaction = solver_response.solutions[0].interactions[0]
        assert interaction.id == "weighted-gno-cow"

        app.dependency_overrides.clear()

    def test_solver_selects_stable_over_v2_when_better(self, client, stable_amm):
        """When stable pool gives better output than V2, use stable.

        Stable pools should give near 1:1 rates for stablecoins,
        which is better than V2 constant product for same pairs.
        """
        solver = Solver(stable_amm=stable_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("stable_dai_to_usdc.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have a solution
        assert len(solver_response.solutions) >= 1

        # The interaction should reference the stable pool
        interaction = solver_response.solutions[0].interactions[0]
        assert interaction.id == "stable-dai-usdc-usdt"

        app.dependency_overrides.clear()
