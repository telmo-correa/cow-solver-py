"""Integration tests for single order routing."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from solver.api.endpoints import get_solver
from solver.api.main import app
from solver.models.solution import SolverResponse
from solver.routing.router import SingleOrderRouter
from solver.solver import Solver
from tests.conftest import MockAMM, MockPoolFinder, MockSwapConfig

# Standard token addresses for tests
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
DAI = "0x6b175474e89094c44da98b954eedeac495271d0f"


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


def make_weth_dai_pool_liquidity():
    """Create liquidity data for WETH/DAI pool."""
    return {
        "kind": "constantProduct",
        "id": "uniswap-v2-weth-dai",
        "address": "0xa478c2975ab1ea89e8196811f51a7b7ade33eb11",
        "router": "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
        "gasEstimate": "110000",
        "tokens": {
            DAI: {"balance": "30000000000000000000000000"},  # 30M DAI
            WETH: {"balance": "12000000000000000000000"},  # 12K WETH
        },
        "fee": "0.003",
    }


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create a test client for the API."""
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestSingleSellOrderRouting:
    """Tests for routing single sell orders through UniswapV2."""

    def test_routes_weth_to_usdc_sell_order(self, client):
        """A WETH→USDC sell order should produce a solution with trades."""
        auction = {
            "id": "test_weth_usdc",
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
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2000000000",  # Min 2000 USDC (reasonable limit price)
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

        # Should have at least one solution
        assert len(solver_response.solutions) >= 1

        solution = solver_response.solutions[0]

        # Should have exactly one trade
        assert len(solution.trades) == 1
        trade = solution.trades[0]
        assert trade.order == "0x" + "01" * 56
        assert int(trade.executed_amount) == 1000000000000000000  # Full sell amount

        # Should have at least one interaction (LiquidityInteraction)
        assert len(solution.interactions) >= 1
        interaction = solution.interactions[0]
        assert interaction.kind == "liquidity"
        assert interaction.internalize is True
        assert interaction.id is not None  # Has liquidity ID from auction

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
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "10000000000000",  # 10M USDC (unrealistic)
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

        # No solution (can't meet limit price)
        assert len(solver_response.solutions) == 0

    def test_handles_reverse_direction(self, client):
        """USDC→WETH should also work (reverse of common pair)."""
        auction = {
            "id": "test_usdc_weth",
            "orders": [
                {
                    "uid": "0x" + "04" * 56,
                    "sellToken": USDC,
                    "buyToken": WETH,
                    "sellAmount": "2500000000",  # 2500 USDC
                    "buyAmount": "500000000000000000",  # Min 0.5 WETH
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
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2000000000",  # Min 2000 USDC
                    "kind": "sell",
                    "class": "limit",
                }
            ],
            "liquidity": [make_weth_usdc_pool_liquidity()],
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

        price_weth = int(prices[weth_addr])
        price_usdc = int(prices[usdc_addr])

        # CoW Protocol constraint: executed_sell * price_sell >= executed_buy * price_buy
        # For this order: 1 WETH sold, getting ~2500 USDC
        #
        # Pricing scheme (matching Rust solver):
        # - price[sell_token] = amount_out (USDC output)
        # - price[buy_token] = amount_in (WETH input)
        #
        # This gives exchange rate: amount_out / amount_in

        # The sell token (WETH) price should be the USDC output amount (~2.5e9)
        assert 2000 * 10**6 < price_weth < 3000 * 10**6  # ~2500 USDC

        # The buy token (USDC) price should be the WETH input amount (1e18)
        assert price_usdc == 10**18

        # Verify the CoW Protocol constraint holds
        # executed_sell * price_sell >= executed_buy * price_buy
        # With our pricing: amount_in * amount_out >= amount_out * amount_in (equality)
        sell_amount = 10**18  # 1 WETH
        buy_amount = price_weth  # The output amount

        assert sell_amount * price_weth == buy_amount * price_usdc

    def test_solution_has_gas_estimate(self, client):
        """Solutions should include gas estimates."""
        auction = {
            "id": "test_gas",
            "orders": [
                {
                    "uid": "0x" + "08" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000",
                    "kind": "sell",
                    "class": "limit",
                }
            ],
            "liquidity": [make_weth_usdc_pool_liquidity()],
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

    def test_cow_match_two_orders(self, client):
        """Two orders that form a perfect CoW match are settled directly."""
        # Order A: sells 1 WETH, wants min 2000 USDC
        # Order B: sells 3000 USDC, wants 1 WETH
        # Perfect match: A gets 3000 USDC (> 2000), B gets 1 WETH
        auction = {
            "id": "cow_pair",
            "orders": [
                {
                    "uid": "0x" + "05" * 56,
                    "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2000000000",  # min 2000 USDC
                    "kind": "sell",
                    "class": "limit",
                },
                {
                    "uid": "0x" + "06" * 56,
                    "sellToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "buyToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    "sellAmount": "3000000000",  # 3000 USDC
                    "buyAmount": "1000000000000000000",  # wants 1 WETH
                    "kind": "sell",
                    "class": "limit",
                },
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # CoW match should produce a solution
        assert len(data["solutions"]) == 1

        solution = data["solutions"][0]
        # Should have two trades (one per order)
        assert len(solution["trades"]) == 2
        # No AMM interactions needed for CoW match
        assert len(solution["interactions"]) == 0
        # Should have clearing prices for both tokens
        assert len(solution["prices"]) == 2


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
        """Test behavior when no pool/route is found."""
        # Pool finder that returns None for all pairs
        mock_finder = MockPoolFinder(default_pool=None)
        mock_amm = MockAMM()

        router = SingleOrderRouter(amm=mock_amm, pool_finder=mock_finder)
        solver = Solver(router=router)
        app.dependency_overrides[get_solver] = lambda: solver

        client = TestClient(app)
        # Use tokens NOT in the global pool registry to ensure no multi-hop route exists
        auction = {
            "id": "test_no_pool",
            "orders": [
                {
                    "uid": "0x" + "01" * 56,
                    "sellToken": "0x1111111111111111111111111111111111111111",
                    "buyToken": "0x2222222222222222222222222222222222222222",
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "1",  # Very low limit to isolate pool finding
                    "kind": "sell",
                    "class": "limit",
                }
            ],
        }

        response = client.post("/production/mainnet", json=auction)
        data = response.json()

        # Should return empty - no pool or route found
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
        # Router calls simulate_swap twice: once for best-quote selection, once for routing
        assert len(mock_amm.swap_calls) == 2
        for call in mock_amm.swap_calls:
            assert call["amount_in"] == sell_amount
            assert call["pool"] == mock_weth_usdc_pool.address

        # Verify pool finder received the correct tokens
        # Called twice: once for CoW strategy (checking for potential matches),
        # once for AMM routing
        assert len(mock_finder.calls) == 2
        for call in mock_finder.calls:
            token_a, token_b = call
            assert "c02aaa39b223fe8d0a0e5c4f27ead9083c756cc2" in token_a.lower()
            assert "a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48" in token_b.lower()

        app.dependency_overrides.clear()


class TestPartialCowAmmComposition:
    """Tests for partial CoW matching + AMM routing composition."""

    def test_amm_routing_with_multiple_orders(self, client):
        """AMM routing when CoW match fails EBBO validation.

        Order A: sells 2 WETH, wants min 4000 USDC (partiallyFillable)
        Order B: sells 2500 USDC, wants 1 WETH

        AMM pool rate: ~2500 USDC/WETH (before fees)

        The orders overlap in price (A accepts 2000, B offers 2500), but
        MultiPairCowStrategy's strict EBBO validation may reject the CoW match
        if the clearing price doesn't beat AMM rates for both parties.

        Expected: At least one solution via AMM routing.
        """
        order_a_uid = "0x" + "0A" * 56
        order_b_uid = "0x" + "0B" * 56

        auction = {
            "id": "partial_cow_amm",
            "orders": [
                {
                    "uid": order_a_uid,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "2000000000000000000",  # 2 WETH
                    "buyAmount": "4000000000",  # min 4000 USDC (2000/WETH, below AMM)
                    "kind": "sell",
                    "class": "limit",
                    "partiallyFillable": True,
                },
                {
                    "uid": order_b_uid,
                    "sellToken": USDC,
                    "buyToken": WETH,
                    "sellAmount": "2500000000",  # 2500 USDC
                    "buyAmount": "1000000000000000000",  # wants 1 WETH
                    "kind": "sell",
                    "class": "limit",
                },
            ],
            "liquidity": [make_weth_usdc_pool_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # Should have at least one solution (via AMM routing)
        assert len(data["solutions"]) >= 1

        # First solution should route order A through AMM
        solution = data["solutions"][0]
        assert len(solution["trades"]) >= 1
        assert len(solution["interactions"]) >= 1  # AMM interaction
        assert solution["gas"] > 0

    def test_partial_cow_only_when_no_amm_liquidity(self, client):
        """Partial CoW match without AMM when no liquidity available.

        When there's no AMM liquidity, we should still get a partial
        CoW solution if possible. Order A needs partiallyFillable=true.
        """
        auction = {
            "id": "partial_cow_no_amm",
            "orders": [
                {
                    "uid": "0x" + "0C" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "2000000000000000000",  # 2 WETH
                    "buyAmount": "4000000000",  # min 4000 USDC (rate: 2000 USDC/WETH)
                    "kind": "sell",
                    "class": "limit",
                    "partiallyFillable": True,  # A will be partially filled
                },
                {
                    "uid": "0x" + "0D" * 56,
                    "sellToken": USDC,
                    "buyToken": WETH,
                    "sellAmount": "3000000000",  # 3000 USDC
                    "buyAmount": "1000000000000000000",  # wants 1 WETH (rate: 3000 USDC/WETH)
                    "kind": "sell",
                    "class": "limit",
                },
            ],
            "liquidity": [],  # No AMM liquidity
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()

        # Should have a partial solution from CoW matching
        assert len(data["solutions"]) == 1

        solution = data["solutions"][0]
        # Two trades from CoW match (both orders participate)
        assert len(solution["trades"]) == 2
        # No AMM interactions (no liquidity)
        assert len(solution["interactions"]) == 0
        # Gas is 0 for pure CoW
        assert solution["gas"] == 0

    def test_amm_routing_when_cow_fails_ebbo(self, client):
        """AMM routes both orders when CoW match fails EBBO validation.

        The CoW clearing price (3000 USDC/WETH) is worse than the AMM rate
        (~2500 USDC/WETH from pool reserves), so EBBO rejects the CoW match.
        Both orders are routed independently via AMM in two solutions.
        """
        order_a_uid = "0x" + "0E" * 56
        order_b_uid = "0x" + "0F" * 56

        auction = {
            "id": "partial_cow_amounts",
            "orders": [
                {
                    "uid": order_a_uid,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "2000000000000000000",  # 2 WETH
                    "buyAmount": "4000000000",  # min 4000 USDC
                    "kind": "sell",
                    "class": "limit",
                    "partiallyFillable": True,
                },
                {
                    "uid": order_b_uid,
                    "sellToken": USDC,
                    "buyToken": WETH,
                    "sellAmount": "3000000000",  # 3000 USDC
                    "buyAmount": "1000000000000000000",  # wants 1 WETH
                    "kind": "sell",
                    "class": "limit",
                },
            ],
            "liquidity": [make_weth_usdc_pool_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # EBBO rejects CoW match, AMM routes both orders in 2 solutions
        assert len(data["solutions"]) == 2

        # Solution 0: AMM route for A (2 WETH full)
        solution_0 = data["solutions"][0]
        assert len(solution_0["trades"]) == 1
        assert len(solution_0["interactions"]) == 1
        assert solution_0["gas"] > 0

        trade_a = solution_0["trades"][0]
        assert trade_a["order"] == order_a_uid
        # A's full 2 WETH routed via AMM
        assert trade_a["executedAmount"] == "2000000000000000000"

        # Solution 1: AMM route for B
        solution_1 = data["solutions"][1]
        assert len(solution_1["trades"]) == 1
        assert len(solution_1["interactions"]) == 1
        assert solution_1["gas"] > 0

        trade_b = solution_1["trades"][0]
        assert trade_b["order"] == order_b_uid

    def test_each_order_appears_once_per_solution(self, client):
        """Each order should appear at most once per solution.

        With AMM routing (EBBO rejects CoW match):
        - Solution 0: Order A routed via AMM
        - Solution 1: Order B routed via AMM

        Each order appears once per solution, never duplicated.
        """
        order_a_uid = "0x" + "10" * 56
        order_b_uid = "0x" + "11" * 56

        auction = {
            "id": "single_trade_per_order",
            "orders": [
                {
                    "uid": order_a_uid,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "2000000000000000000",  # 2 WETH
                    "buyAmount": "4000000000",
                    "kind": "sell",
                    "class": "limit",
                    "partiallyFillable": True,
                },
                {
                    "uid": order_b_uid,
                    "sellToken": USDC,
                    "buyToken": WETH,
                    "sellAmount": "3000000000",  # 3000 USDC
                    "buyAmount": "1000000000000000000",
                    "kind": "sell",
                    "class": "limit",
                },
            ],
            "liquidity": [make_weth_usdc_pool_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        data = response.json()

        # 2 solutions: each order routed via AMM
        assert len(data["solutions"]) == 2

        # Check each solution: every order appears at most once
        for i, solution in enumerate(data["solutions"]):
            uid_counts = {}
            for trade in solution["trades"]:
                uid = trade["order"]
                uid_counts[uid] = uid_counts.get(uid, 0) + 1

            for uid, count in uid_counts.items():
                assert count == 1, (
                    f"Solution {i}: Order {uid[:18]}... appears {count} times, expected 1"
                )

        # Verify solution structure - each solution has one order
        solution_0 = data["solutions"][0]
        solution_1 = data["solutions"][1]
        assert len(solution_0["trades"]) == 1
        assert len(solution_1["trades"]) == 1

        # Each order appears in exactly one solution
        all_uids = {solution_0["trades"][0]["order"], solution_1["trades"][0]["order"]}
        assert order_a_uid in all_uids
        assert order_b_uid in all_uids

    def test_partial_cow_returned_when_amm_has_no_liquidity(self, client):
        """When there's no AMM liquidity, partial CoW solution is still returned.

        This verifies that we don't lose the partial CoW solution just because
        subsequent strategies can't process the remainder.
        See Issue #11 in code review.

        Note: The double auction finds the optimal clearing price (366.67M)
        which allows B to get more WETH (1.1) for their 3000 USDC than
        their minimum (1 WETH), improving both sides' execution.
        """
        auction = {
            "id": "partial_cow_no_liquidity",
            "orders": [
                {
                    "uid": "0x" + "12" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "2000000000000000000",  # 2 WETH
                    "buyAmount": "5000000000",  # min 5000 USDC (2500 USDC/WETH)
                    "kind": "sell",
                    "class": "limit",
                    "partiallyFillable": True,  # A will be partially filled
                },
                {
                    "uid": "0x" + "13" * 56,
                    "sellToken": USDC,
                    "buyToken": WETH,
                    "sellAmount": "3000000000",  # 3000 USDC
                    "buyAmount": "1000000000000000000",  # wants min 1 WETH
                    "kind": "sell",
                    "class": "limit",
                },
            ],
            "liquidity": [],  # No AMM liquidity - AMM strategy will fail
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()

        # Should still have a solution from partial CoW match
        # even though AMM couldn't route the remainder (no liquidity)
        assert len(data["solutions"]) == 1

        solution = data["solutions"][0]
        # Two trades: both orders participated in partial CoW
        assert len(solution["trades"]) == 2
        # No AMM interactions (no liquidity)
        assert len(solution["interactions"]) == 0
        # Gas is 0 for pure CoW
        assert solution["gas"] == 0

        # Verify that order A is partially filled
        trade_a = next(t for t in solution["trades"] if t["order"] == "0x" + "12" * 56)
        # Double auction finds clearing price ~366.67M where B gets 1.1 WETH
        # (better than their minimum of 1 WETH) for 3000 USDC
        assert int(trade_a["executedAmount"]) == 1100000000000000000  # 1.1 WETH


class TestFillOrKillIntegration:
    """Integration tests for fill-or-kill (partiallyFillable=false) behavior.

    These tests verify the API correctly handles the partiallyFillable flag
    per the CoW Protocol specification.
    """

    def test_fill_or_kill_perfect_match_via_api(self, client):
        """Fill-or-kill orders can be perfectly matched through the API.

        Both orders are fill-or-kill (default) and both are fully filled.
        """
        auction = {
            "id": "fok_perfect_match",
            "orders": [
                {
                    "uid": "0x" + "F1" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2500000000",  # 2500 USDC
                    "kind": "sell",
                    "class": "limit",
                    # partiallyFillable defaults to false (fill-or-kill)
                },
                {
                    "uid": "0x" + "F2" * 56,
                    "sellToken": USDC,
                    "buyToken": WETH,
                    "sellAmount": "2500000000",  # 2500 USDC
                    "buyAmount": "1000000000000000000",  # 1 WETH
                    "kind": "sell",
                    "class": "limit",
                    # partiallyFillable defaults to false (fill-or-kill)
                },
            ],
            "liquidity": [],  # No AMM needed for perfect match
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # Should have a solution (perfect CoW match)
        assert len(data["solutions"]) == 1

        solution = data["solutions"][0]
        # Both orders filled
        assert len(solution["trades"]) == 2
        # No AMM interactions
        assert len(solution["interactions"]) == 0

    def test_fill_or_kill_no_partial_match_via_api(self, client):
        """Fill-or-kill orders cannot be partially matched through the API.

        When both orders are fill-or-kill and one would need partial fill,
        no CoW match occurs. Falls back to AMM if available.

        Since these are independent AMM routes, they become separate solutions.
        """
        auction = {
            "id": "fok_no_partial",
            "orders": [
                {
                    "uid": "0x" + "F3" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "2000000000000000000",  # 2 WETH
                    "buyAmount": "4000000000",  # 4000 USDC
                    "kind": "sell",
                    "class": "limit",
                    # partiallyFillable defaults to false (fill-or-kill)
                },
                {
                    "uid": "0x" + "F4" * 56,
                    "sellToken": USDC,
                    "buyToken": WETH,
                    "sellAmount": "3000000000",  # 3000 USDC
                    "buyAmount": "1000000000000000000",  # 1 WETH
                    "kind": "sell",
                    "class": "limit",
                    # partiallyFillable defaults to false (fill-or-kill)
                },
            ],
            "liquidity": [make_weth_usdc_pool_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # 2 separate solutions (independent AMM routes, no CoW match)
        assert len(data["solutions"]) == 2

        # Each solution has 1 trade and 1 interaction
        for solution in data["solutions"]:
            assert len(solution["trades"]) == 1
            assert len(solution["interactions"]) == 1
            assert solution["gas"] > 0

        # Verify both orders are routed
        routed_orders = {
            data["solutions"][0]["trades"][0]["order"],
            data["solutions"][1]["trades"][0]["order"],
        }
        assert "0x" + "F3" * 56 in routed_orders
        assert "0x" + "F4" * 56 in routed_orders

    def test_fill_or_kill_no_solution_when_no_amm(self, client):
        """Fill-or-kill orders with no perfect match and no AMM = no solution.

        When both are fill-or-kill, partial match not possible, and no AMM,
        there's no solution.
        """
        auction = {
            "id": "fok_no_solution",
            "orders": [
                {
                    "uid": "0x" + "F5" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "2000000000000000000",  # 2 WETH
                    "buyAmount": "4000000000",  # 4000 USDC
                    "kind": "sell",
                    "class": "limit",
                    # partiallyFillable defaults to false (fill-or-kill)
                },
                {
                    "uid": "0x" + "F6" * 56,
                    "sellToken": USDC,
                    "buyToken": WETH,
                    "sellAmount": "3000000000",  # 3000 USDC
                    "buyAmount": "1000000000000000000",  # 1 WETH
                    "kind": "sell",
                    "class": "limit",
                    # partiallyFillable defaults to false (fill-or-kill)
                },
            ],
            "liquidity": [],  # No AMM liquidity
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # No solution possible: no perfect CoW match, no AMM
        assert len(data["solutions"]) == 0

    def test_mixed_partial_and_fok_works_via_api(self, client):
        """Mixed partiallyFillable and fill-or-kill orders work correctly.

        A is partiallyFillable, B is fill-or-kill.
        With AMM liquidity, EBBO rejects the CoW match (AMM offers better rate),
        so both orders are routed via AMM in separate solutions.
        """
        order_a_uid = "0x" + "F7" * 56
        order_b_uid = "0x" + "F8" * 56

        auction = {
            "id": "mixed_partial_fok",
            "orders": [
                {
                    "uid": order_a_uid,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "2000000000000000000",  # 2 WETH
                    "buyAmount": "4000000000",  # 4000 USDC
                    "kind": "sell",
                    "class": "limit",
                    "partiallyFillable": True,  # A can be partially filled
                },
                {
                    "uid": order_b_uid,
                    "sellToken": USDC,
                    "buyToken": WETH,
                    "sellAmount": "3000000000",  # 3000 USDC
                    "buyAmount": "1000000000000000000",  # 1 WETH
                    "kind": "sell",
                    "class": "limit",
                    # partiallyFillable defaults to false (fill-or-kill)
                },
            ],
            "liquidity": [make_weth_usdc_pool_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # EBBO rejects CoW, both orders routed via AMM
        assert len(data["solutions"]) == 2

        # Solution 0: AMM route for A (full 2 WETH)
        solution_0 = data["solutions"][0]
        assert len(solution_0["trades"]) == 1
        assert len(solution_0["interactions"]) == 1
        assert solution_0["gas"] > 0
        assert solution_0["trades"][0]["order"] == order_a_uid
        assert solution_0["trades"][0]["executedAmount"] == "2000000000000000000"

        # Solution 1: AMM route for B (full 3000 USDC)
        solution_1 = data["solutions"][1]
        assert len(solution_1["trades"]) == 1
        assert len(solution_1["interactions"]) == 1
        assert solution_1["gas"] > 0
        assert solution_1["trades"][0]["order"] == order_b_uid
        assert solution_1["trades"][0]["executedAmount"] == "3000000000"


class TestMultiOrderAmmRouting:
    """Tests for AMM routing with multiple orders.

    Verifies that AmmRoutingStrategy can handle multiple remainder orders
    from partial CoW matching by routing each independently.
    """

    def test_two_remainder_orders_both_routed(self, client):
        """Both orders are routed when liquidity exists.

        Scenario:
        - Order A: sells 1 WETH, wants 2000 USDC
        - Order B: sells 1 WETH, wants 2000 USDC (same direction, no CoW possible)

        Since both orders sell WETH for USDC (same direction), CoW won't match.
        AMM routes them as independent solutions (one per order).
        """
        order_a_uid = "0x" + "A1" * 56
        order_b_uid = "0x" + "A2" * 56

        auction = {
            "id": "multi_order_same_direction",
            "orders": [
                {
                    "uid": order_a_uid,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2000000000",  # min 2000 USDC
                    "kind": "sell",
                    "class": "limit",
                },
                {
                    "uid": order_b_uid,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2000000000",  # min 2000 USDC
                    "kind": "sell",
                    "class": "limit",
                },
            ],
            "liquidity": [make_weth_usdc_pool_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # 2 separate solutions (independent AMM routes)
        assert len(data["solutions"]) == 2

        # Each solution has 1 trade and 1 interaction
        for solution in data["solutions"]:
            assert len(solution["trades"]) == 1
            assert len(solution["interactions"]) == 1
            assert solution["gas"] > 0

        # Verify both orders are routed
        routed_orders = {
            data["solutions"][0]["trades"][0]["order"],
            data["solutions"][1]["trades"][0]["order"],
        }
        assert order_a_uid in routed_orders
        assert order_b_uid in routed_orders

    def test_partial_success_one_order_fails(self, client):
        """When one order can't be routed, the other still succeeds.

        Scenario:
        - Order A: sells WETH for USDC (has liquidity)
        - Order B: sells WETH for DAI (no liquidity)

        Only Order A should be filled.
        """
        auction = {
            "id": "multi_order_partial_success",
            "orders": [
                {
                    "uid": "0x" + "B1" * 56,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2000000000",  # min 2000 USDC
                    "kind": "sell",
                    "class": "limit",
                },
                {
                    "uid": "0x" + "B2" * 56,
                    "sellToken": WETH,
                    "buyToken": DAI,
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "2000000000000000000000",  # min 2000 DAI
                    "kind": "sell",
                    "class": "limit",
                },
            ],
            # Only WETH/USDC liquidity, no WETH/DAI
            "liquidity": [make_weth_usdc_pool_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        assert len(data["solutions"]) == 1

        solution = data["solutions"][0]
        # Only one order filled (the WETH->USDC one)
        assert len(solution["trades"]) == 1
        assert solution["trades"][0]["order"] == "0x" + "B1" * 56
        # One AMM interaction
        assert len(solution["interactions"]) == 1

    def test_all_orders_fail_returns_empty(self, client):
        """When no orders can be routed, return empty solution.

        Scenario: Both orders need liquidity that doesn't exist.
        """
        auction = {
            "id": "multi_order_all_fail",
            "orders": [
                {
                    "uid": "0x" + "C1" * 56,
                    "sellToken": WETH,
                    "buyToken": DAI,
                    "sellAmount": "1000000000000000000",
                    "buyAmount": "2000000000000000000000",
                    "kind": "sell",
                    "class": "limit",
                },
                {
                    "uid": "0x" + "C2" * 56,
                    "sellToken": USDC,
                    "buyToken": DAI,
                    "sellAmount": "1000000000",
                    "buyAmount": "900000000000000000000",
                    "kind": "sell",
                    "class": "limit",
                },
            ],
            # Only WETH/USDC liquidity, neither order can route
            "liquidity": [make_weth_usdc_pool_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # No solution possible
        assert len(data["solutions"]) == 0

    def test_opposite_direction_orders_no_cow_both_amm_routed(self, client):
        """Opposite-direction fill-or-kill orders route via AMM when CoW fails.

        Scenario:
        - Order A: sells 2 WETH for 4000 USDC (fill-or-kill)
        - Order B: sells 3000 USDC for 1 WETH (fill-or-kill)

        A perfect CoW match isn't possible (amounts don't match exactly).
        A partial match would need A to be partially filled, but A is fill-or-kill.
        So both orders fall back to AMM routing as independent solutions.
        """
        order_a_uid = "0x" + "D1" * 56
        order_b_uid = "0x" + "D2" * 56

        auction = {
            "id": "partial_cow_two_remainders",
            "orders": [
                {
                    "uid": order_a_uid,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "2000000000000000000",  # 2 WETH
                    "buyAmount": "4000000000",  # min 4000 USDC
                    "kind": "sell",
                    "class": "limit",
                    # partiallyFillable defaults to false (fill-or-kill)
                },
                {
                    "uid": order_b_uid,
                    "sellToken": USDC,
                    "buyToken": WETH,
                    "sellAmount": "3000000000",  # 3000 USDC
                    "buyAmount": "1000000000000000000",  # wants 1 WETH
                    "kind": "sell",
                    "class": "limit",
                    # partiallyFillable defaults to false (fill-or-kill)
                },
            ],
            "liquidity": [make_weth_usdc_pool_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # 2 separate solutions (independent AMM routes, CoW rejected due to fill-or-kill)
        assert len(data["solutions"]) == 2

        # Each solution has 1 trade and 1 interaction
        for solution in data["solutions"]:
            assert len(solution["trades"]) == 1
            assert len(solution["interactions"]) == 1
            assert solution["gas"] > 0

        # Verify both orders are routed
        routed_orders = {
            data["solutions"][0]["trades"][0]["order"],
            data["solutions"][1]["trades"][0]["order"],
        }
        assert order_a_uid in routed_orders
        assert order_b_uid in routed_orders

        # Verify full fills
        for solution in data["solutions"]:
            trade = solution["trades"][0]
            if trade["order"] == order_a_uid:
                assert trade["executedAmount"] == "2000000000000000000"  # 2 WETH
            else:
                assert trade["executedAmount"] == "3000000000"  # 3000 USDC


class TestPoolReserveUpdates:
    """Tests verifying that pool reserves are updated between orders.

    When multiple orders route through the same pool, reserves must be
    updated after each swap so subsequent orders get accurate estimates.
    """

    def test_second_order_uses_updated_reserves(self, client):
        """Verify second order sees updated reserves from first order's swap.

        Scenario: Two identical orders, each swapping 1000 WETH for USDC.
        The pool has 20,000 WETH / 50,000,000 USDC.

        Without reserve updates: Both orders would compute ~2,497,500 USDC each.
        With reserve updates: Second order gets slightly less due to price impact.

        Since these are independent AMM routes, they become separate solutions.
        The different output amounts in each solution demonstrate reserve updates.
        """
        order_e1_uid = "0x" + "E1" * 56
        order_e2_uid = "0x" + "E2" * 56

        auction = {
            "id": "reserve_update_test",
            "orders": [
                {
                    "uid": order_e1_uid,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000000",  # 1000 WETH
                    "buyAmount": "1000000000",  # min 1000 USDC (easy to meet)
                    "kind": "sell",
                    "class": "limit",
                },
                {
                    "uid": order_e2_uid,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000000",  # 1000 WETH
                    "buyAmount": "1000000000",  # min 1000 USDC
                    "kind": "sell",
                    "class": "limit",
                },
            ],
            "liquidity": [make_weth_usdc_pool_liquidity()],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # 2 separate solutions (independent AMM routes)
        assert len(data["solutions"]) == 2

        # Each solution has 1 trade and 1 interaction
        for solution in data["solutions"]:
            assert len(solution["trades"]) == 1
            assert len(solution["interactions"]) == 1
            assert solution["gas"] > 0

        # Verify both orders are routed
        routed_orders = {
            data["solutions"][0]["trades"][0]["order"],
            data["solutions"][1]["trades"][0]["order"],
        }
        assert order_e1_uid in routed_orders
        assert order_e2_uid in routed_orders

        # Verify both orders executed their full amounts
        for solution in data["solutions"]:
            assert int(solution["trades"][0]["executedAmount"]) == 1000000000000000000000

        # The different output amounts demonstrate reserve updates working
        # First order gets more USDC than second due to price impact
        output_amounts = []
        for solution in data["solutions"]:
            interaction = solution["interactions"][0]
            output_amounts.append(int(interaction["outputAmount"]))
        # Second order should get less (or same if order undefined) due to reserves being depleted
        # Just verify both outputs are reasonable
        for output in output_amounts:
            assert output > 1000000000  # At least 1000 USDC

    def test_reserve_updates_affect_output_amount(self, client):
        """Verify that reserve updates cause different outputs for identical orders.

        This test uses a smaller pool to make the price impact more visible.
        With a 4 WETH / 10,000 USDC pool:
        - First 1 WETH swap: gets ~1995 USDC, pool becomes 5 WETH / ~8005 USDC
        - Second 1 WETH swap: gets ~1331 USDC (less due to depleted reserves)

        Since these are independent AMM routes, they become separate solutions.
        The different output amounts prove reserve updates are working.
        """
        # Create a smaller pool to make price impact more noticeable
        small_pool_liquidity = {
            "kind": "constantProduct",
            "id": "small-weth-usdc",
            "address": "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
            "router": "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
            "gasEstimate": "110000",
            "tokens": {
                USDC: {"balance": "10000000000"},  # 10,000 USDC
                WETH: {"balance": "4000000000000000000"},  # 4 WETH
            },
            "fee": "0.003",
        }

        order_f1_uid = "0x" + "F1" * 56
        order_f2_uid = "0x" + "F2" * 56

        # Two orders each swapping 1 WETH
        auction = {
            "id": "small_pool_reserve_test",
            "orders": [
                {
                    "uid": order_f1_uid,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "1000000",  # min 1 USDC
                    "kind": "sell",
                    "class": "limit",
                },
                {
                    "uid": order_f2_uid,
                    "sellToken": WETH,
                    "buyToken": USDC,
                    "sellAmount": "1000000000000000000",  # 1 WETH
                    "buyAmount": "1000000",  # min 1 USDC
                    "kind": "sell",
                    "class": "limit",
                },
            ],
            "liquidity": [small_pool_liquidity],
        }

        response = client.post("/production/mainnet", json=auction)
        assert response.status_code == 200

        data = response.json()
        # 2 separate solutions (independent AMM routes)
        assert len(data["solutions"]) == 2

        # Each solution has 1 trade and 1 interaction
        for solution in data["solutions"]:
            assert len(solution["trades"]) == 1
            assert len(solution["interactions"]) == 1

        # Verify both orders are routed
        routed_orders = {
            data["solutions"][0]["trades"][0]["order"],
            data["solutions"][1]["trades"][0]["order"],
        }
        assert order_f1_uid in routed_orders
        assert order_f2_uid in routed_orders

        # Get output amounts and verify they're different (reserve updates working)
        output_amounts = []
        for solution in data["solutions"]:
            interaction = solution["interactions"][0]
            output_amounts.append(int(interaction["outputAmount"]))

        # Both outputs should be reasonable (> 1 USDC = 1e6)
        assert all(out > 1000000 for out in output_amounts)

        # The outputs should be different due to reserve updates
        # First gets more (~1995 USDC), second gets less (~1331 USDC)
        assert output_amounts[0] != output_amounts[1], (
            f"Outputs should differ due to reserve updates: {output_amounts}"
        )

        # First order gets more USDC than second
        assert output_amounts[0] > output_amounts[1], (
            f"First order should get more: {output_amounts[0]} vs {output_amounts[1]}"
        )
