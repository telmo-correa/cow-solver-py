"""Integration tests for UniswapV3 routing."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from solver.amm.uniswap_v3 import MockUniswapV3Quoter, UniswapV3AMM
from solver.api.endpoints import get_solver
from solver.api.main import app
from solver.models.solution import SolverResponse
from solver.solver import Solver

# Standard token addresses for tests
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "auctions" / "v3"


def load_fixture(name: str) -> dict:
    """Load a V3 fixture by name."""
    path = FIXTURES_DIR / name
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create a test client for the API."""
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def mock_v3_quoter() -> MockUniswapV3Quoter:
    """Create a mock V3 quoter with a reasonable default rate.

    Uses 2500 USDC per WETH rate (common during testing).
    For USDC→WETH: amount_out = amount_in * 2500
    For WETH→USDC: uses 1/2500 but with proper decimal handling.
    """
    # Rate of 2500 means: 1 WETH (1e18 wei) → 2500 USDC (2500e6)
    # Since WETH has 18 decimals and USDC has 6, the rate is:
    # amount_out_usdc = amount_in_weth * 2500 * 1e6 / 1e18 = amount_in_weth * 2.5e-9
    # But our mock uses a simple multiplier, so we need to adjust.
    #
    # For simplicity: use a ratio that makes 1e18 WETH → 2500e6 USDC
    # amount_out = amount_in * numerator // denominator
    # numerator = 2500e6 = 2_500_000_000
    # denominator = 1e18
    # So for 1 WETH (1e18): output = 1e18 * 2_500_000_000 // 1e18 = 2_500_000_000 USDC
    return MockUniswapV3Quoter(default_rate=(2_500_000_000, 10**18))


@pytest.fixture
def mock_v3_amm(mock_v3_quoter: MockUniswapV3Quoter) -> UniswapV3AMM:
    """Create a V3 AMM with mock quoter."""
    return UniswapV3AMM(quoter=mock_v3_quoter)


class TestV3SingleOrderRouting:
    """Tests for routing single orders through UniswapV3 pools."""

    def test_v3_only_sell_order(self, client, mock_v3_amm):
        """A WETH→USDC sell order should route through V3 pool.

        When only V3 liquidity is available, the router should use V3.
        """
        # Override solver with V3 AMM
        solver = Solver(v3_amm=mock_v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("v3_single_order.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have a solution
        assert len(solver_response.solutions) >= 1

        solution = solver_response.solutions[0]

        # Should have exactly one trade
        assert len(solution.trades) == 1
        trade = solution.trades[0]

        # For limit orders, executed_amount + fee = sell_amount
        # The fee is deducted from the executed amount
        sell_amount = 1000000000000000000  # 1 WETH
        executed = int(trade.executed_amount)
        fee = int(trade.fee) if trade.fee else 0
        assert executed + fee == sell_amount, (
            f"executed ({executed}) + fee ({fee}) should equal sell_amount ({sell_amount})"
        )

        # Should have interaction (LiquidityInteraction for V3)
        assert len(solution.interactions) >= 1

        # Should have prices for both tokens
        assert len(solution.prices) == 2

        app.dependency_overrides.clear()

    def test_v3_buy_order(self, client):
        """A buy order should route through V3 pool.

        Buy orders use exact output quotes from the quoter.
        """
        # Create quoter with explicit quote for this test
        from solver.amm.uniswap_v3 import QuoteKey

        quoter = MockUniswapV3Quoter()
        # For exact output of 1 WETH, return 2600 USDC needed (within limit of 3000)
        quoter.quotes[
            QuoteKey(
                token_in=USDC.lower(),
                token_out=WETH.lower(),
                fee=3000,  # 0.3% fee tier
                amount=1000000000000000000,  # 1 WETH output
                is_exact_input=False,
            )
        ] = 2600_000_000  # 2600 USDC

        v3_amm = UniswapV3AMM(quoter=quoter)
        solver = Solver(v3_amm=v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("v3_buy_order.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        solver_response = SolverResponse.model_validate(data)

        # Should have a solution
        assert len(solver_response.solutions) >= 1

        solution = solver_response.solutions[0]

        # Should have exactly one trade
        assert len(solution.trades) == 1
        trade = solution.trades[0]
        # For buy order, executed_amount is the buy amount (WETH)
        assert int(trade.executed_amount) == 1000000000000000000  # Full 1 WETH

        app.dependency_overrides.clear()

    def test_v3_order_respects_limit_price(self, client):
        """V3 order should fail if quote doesn't meet limit price."""
        # Configure quoter to return low output (below limit)
        quoter = MockUniswapV3Quoter(
            default_rate=(1_000_000_000, 10**18)  # Only 1000 USDC per WETH
        )
        v3_amm = UniswapV3AMM(quoter=quoter)
        solver = Solver(v3_amm=v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("v3_single_order.json")
        # Fixture has buyAmount of 2400 USDC, but quoter returns only 1000
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # No solution because 1000 USDC < 2400 USDC limit
        assert len(data["solutions"]) == 0

        app.dependency_overrides.clear()


class TestV2V3PoolSelection:
    """Tests for best-quote selection between V2 and V3 pools."""

    def test_selects_v3_when_better_quote(self, client):
        """Router should select V3 pool when it gives better quote.

        V3 pool with 2500 USDC/WETH vs V2 pool with 2500 USDC/WETH
        V3 has lower fee (0.05%) vs V2 (0.3%), so V3 should win.
        """
        # Configure mock V3 to return better rate than V2
        quoter = MockUniswapV3Quoter(
            default_rate=(2_600_000_000, 10**18)  # 2600 USDC per WETH (better than V2's ~2500)
        )
        v3_amm = UniswapV3AMM(quoter=quoter)
        solver = Solver(v3_amm=v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("v2_v3_mixed.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        assert len(data["solutions"]) >= 1

        solution = data["solutions"][0]
        # Should have a trade
        assert len(solution["trades"]) == 1

        # The interaction should reference V3 pool (fee 500 or 3000)
        # V3 gives 2600 USDC vs V2's ~2497 USDC
        interaction = solution["interactions"][0]
        # The liquidity ID tells us which pool was used
        # For mixed fixtures, V3 pool IDs start with "v3-"
        assert interaction["id"].startswith("v3-")

        app.dependency_overrides.clear()

    def test_selects_v2_when_better_quote(self, client):
        """Router should select V2 pool when it gives better quote."""
        # Configure V3 to return worse rate than V2
        quoter = MockUniswapV3Quoter(
            default_rate=(2_000_000_000, 10**18)  # 2000 USDC per WETH (worse than V2's ~2500)
        )
        v3_amm = UniswapV3AMM(quoter=quoter)
        solver = Solver(v3_amm=v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("v2_v3_mixed.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        assert len(data["solutions"]) >= 1

        solution = data["solutions"][0]
        # Should have a trade
        assert len(solution["trades"]) == 1

        # The interaction should reference V2 pool
        interaction = solution["interactions"][0]
        # V2 pool IDs start with "v2-"
        assert interaction["id"].startswith("v2-")

        app.dependency_overrides.clear()

    def test_falls_back_to_v2_when_v3_quoter_fails(self, client):
        """Router should fall back to V2 when V3 quoter returns None."""
        # Configure V3 quoter to return None (failure)
        quoter = MockUniswapV3Quoter()  # No quotes, no default_rate → returns None
        v3_amm = UniswapV3AMM(quoter=quoter)
        solver = Solver(v3_amm=v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("v2_v3_mixed.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        assert len(data["solutions"]) >= 1

        solution = data["solutions"][0]
        # Should still have a solution via V2
        assert len(solution["trades"]) == 1

        # The interaction should reference V2 pool (V3 failed)
        interaction = solution["interactions"][0]
        assert interaction["id"].startswith("v2-")

        app.dependency_overrides.clear()

    def test_v3_only_with_no_v2_liquidity(self, client, mock_v3_amm):
        """Router should use V3 when no V2 liquidity exists."""
        solver = Solver(v3_amm=mock_v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        # V3 single order fixture has only V3 liquidity
        auction = load_fixture("v3_single_order.json")
        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        assert len(data["solutions"]) >= 1

        solution = data["solutions"][0]
        assert len(solution["trades"]) == 1
        assert len(solution["interactions"]) >= 1

        # Interaction should be V3 pool
        interaction = solution["interactions"][0]
        assert interaction["id"].startswith("v3-")

        app.dependency_overrides.clear()


class TestV3ApiEndpoint:
    """Tests for V3 support through the API endpoint."""

    def test_api_returns_v3_solution(self, client, mock_v3_amm):
        """API endpoint should return valid solution for V3-only auction."""
        solver = Solver(v3_amm=mock_v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("v3_single_order.json")
        response = client.post("/production/mainnet", json=auction)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "solutions" in data
        assert len(data["solutions"]) >= 1

        solution = data["solutions"][0]
        assert "trades" in solution
        assert "interactions" in solution
        assert "prices" in solution
        assert "gas" in solution

        # Gas should be non-zero for V3 swap
        assert solution["gas"] > 0

        app.dependency_overrides.clear()

    def test_api_handles_v3_parsing(self, client, mock_v3_amm):
        """API should correctly parse V3 liquidity from auction."""
        solver = Solver(v3_amm=mock_v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        # Send raw auction JSON (tests parsing)
        auction = {
            "id": "v3_parsing_test",
            "tokens": {
                WETH: {
                    "decimals": 18,
                    "symbol": "WETH",
                    "trusted": True,
                    "availableBalance": "1000000000000000000",
                },
                USDC: {
                    "decimals": 6,
                    "symbol": "USDC",
                    "trusted": True,
                    "availableBalance": "10000000000",
                },
            },
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",  # 2000 USDC
                    "kind": "sell",
                    "class": "limit",
                }
            ],
            "liquidity": [
                {
                    "id": "v3-weth-usdc-500",
                    "kind": "concentratedLiquidity",
                    "address": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
                    "tokens": [WETH, USDC],
                    "fee": "0.0005",  # 0.05% = 500 bps
                    "sqrtPrice": "1887339785326389816925594",
                    "liquidity": "12000000000000000000",
                    "tick": 201390,
                    "liquidityNet": {},
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        assert len(data["solutions"]) >= 1

        solution = data["solutions"][0]
        assert len(solution["trades"]) == 1

        # Interaction should reference the V3 pool we provided
        interaction = solution["interactions"][0]
        assert interaction["id"] == "v3-weth-usdc-500"

        app.dependency_overrides.clear()

    def test_api_clearing_prices_with_v3(self, client, mock_v3_amm):
        """API should return valid clearing prices for V3 solution."""
        solver = Solver(v3_amm=mock_v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("v3_single_order.json")
        response = client.post("/production/mainnet", json=auction)

        data = response.json()
        assert len(data["solutions"]) >= 1

        solution = data["solutions"][0]
        prices = solution["prices"]

        # Normalize addresses to lowercase for comparison
        weth_price = prices.get(WETH.lower())
        usdc_price = prices.get(USDC.lower())

        assert weth_price is not None
        assert usdc_price is not None

        # Prices should be positive
        assert int(weth_price) > 0
        assert int(usdc_price) > 0

        app.dependency_overrides.clear()


class TestV3GasEstimates:
    """Tests for V3 gas estimates in solutions."""

    def test_v3_solution_has_gas_estimate(self, client, mock_v3_amm):
        """V3 solutions should include gas estimates."""
        solver = Solver(v3_amm=mock_v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        auction = load_fixture("v3_single_order.json")
        response = client.post("/production/mainnet", json=auction)

        data = response.json()
        assert len(data["solutions"]) >= 1

        solution = data["solutions"][0]
        # V3 swaps have a default gas estimate of 150,000
        assert solution["gas"] >= 100000  # At least 100k gas

        app.dependency_overrides.clear()

    def test_v3_gas_from_pool_config(self, client):
        """V3 gas estimate should come from pool configuration if provided."""
        quoter = MockUniswapV3Quoter(default_rate=(2_500_000_000, 10**18))
        v3_amm = UniswapV3AMM(quoter=quoter)
        solver = Solver(v3_amm=v3_amm)
        app.dependency_overrides[get_solver] = lambda: solver

        # Auction with custom gas estimate in V3 pool
        auction = {
            "id": "v3_custom_gas",
            "tokens": {
                WETH: {
                    "decimals": 18,
                    "symbol": "WETH",
                    "trusted": True,
                    "availableBalance": "1000000000000000000",
                },
                USDC: {
                    "decimals": 6,
                    "symbol": "USDC",
                    "trusted": True,
                    "availableBalance": "10000000000",
                },
            },
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
            "liquidity": [
                {
                    "id": "v3-weth-usdc-3000",
                    "kind": "concentratedLiquidity",
                    "address": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
                    "tokens": [WETH, USDC],
                    "fee": "0.003",
                    "sqrtPrice": "1887339785326389816925594",
                    "liquidity": "12000000000000000000",
                    "tick": 201390,
                    "liquidityNet": {},
                    "gasEstimate": "200000",  # Custom gas estimate
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        data = response.json()

        assert len(data["solutions"]) >= 1
        solution = data["solutions"][0]
        # Gas = pool gas estimate (200k) + settlement overhead (106,391)
        # The settlement overhead is: 7365 + 44000 + 2*27513 = 106391
        assert solution["gas"] == 200000 + 106391  # 306391

        app.dependency_overrides.clear()
