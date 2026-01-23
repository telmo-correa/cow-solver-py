"""Unit tests for router dependency injection capabilities."""

from solver.amm.base import SwapResult
from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.pools import PoolRegistry
from solver.routing.router import SingleOrderRouter
from solver.solver import Solver
from tests.conftest import make_order


class TestDependencyInjection:
    """Tests demonstrating dependency injection capabilities."""

    def test_router_with_custom_pool_finder(self):
        """Router can use a custom pool finder function."""

        # Create a pool finder that always returns None (no pools available)
        def no_pools_finder(_token_a: str, _token_b: str) -> UniswapV2Pool | None:
            return None

        router = SingleOrderRouter(pool_finder=no_pools_finder)
        # Use tokens that are NOT in the global pool registry to ensure
        # multi-hop routing also fails (no path exists)
        order = make_order(
            sell_token="0x1111111111111111111111111111111111111111",
            buy_token="0x2222222222222222222222222222222222222222",
        )
        result = router.route_order(order)

        assert result.success is False
        assert "No route found" in result.error

    def test_router_with_mock_amm(self):
        """Router can use a mock AMM for controlled testing."""

        class MockAMM:
            """Mock AMM that returns predictable swap results."""

            SWAP_GAS = 100_000

            def simulate_swap(
                self, pool: UniswapV2Pool, token_in: str, amount_in: int
            ) -> SwapResult:
                # Return a fixed amount regardless of input
                return SwapResult(
                    amount_in=amount_in,
                    amount_out=5000_000_000,
                    pool_address=pool.address,
                    token_in=token_in,
                    token_out=pool.token1 if token_in == pool.token0 else pool.token0,
                    gas_estimate=100_000,
                )

            def encode_swap(
                self,
                _token_in: str,
                _token_out: str,
                _amount_in: int,
                _amount_out_min: int,
                _recipient: str,
            ) -> tuple[str, str]:
                return ("0xmock_target", "0xmock_calldata")

        # Create a registry with a dummy pool
        dummy_pool = UniswapV2Pool(
            address="0x1234567890123456789012345678901234567890",
            token0="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            token1="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            reserve0=1000 * 10**18,
            reserve1=2500000 * 10**6,
            fee_bps=30,
        )

        registry = PoolRegistry()
        registry.add_pool(dummy_pool)

        router = SingleOrderRouter(amm=MockAMM(), pool_registry=registry)
        order = make_order(
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",  # 2000 USDC minimum
        )
        result = router.route_order(order)

        # Mock AMM returns 5000 USDC, which exceeds minimum
        assert result.success is True
        assert result.amount_out == 5000_000_000

    def test_solver_with_custom_router(self):
        """Solver can use a custom router for testing."""
        from solver.models.auction import AuctionInstance, Token
        from solver.routing.router import RoutingResult

        class MockRouter:
            """Mock router that always fails routing."""

            def route_order(self, order):
                return RoutingResult(
                    order=order,
                    amount_in=0,
                    amount_out=0,
                    pool=None,
                    success=False,
                    error="Mock failure",
                )

        solver = Solver(router=MockRouter())

        # Create a minimal auction
        auction = AuctionInstance(
            id="test",
            tokens={
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": Token(
                    decimals=18,
                    symbol="WETH",
                    referencePrice="1000000000000000000",
                    availableBalance="1000000000000000000",
                    trusted=True,
                ),
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": Token(
                    decimals=6,
                    symbol="USDC",
                    referencePrice="400000000000000",
                    availableBalance="100000000000",
                    trusted=True,
                ),
            },
            orders=[make_order()],
            liquidity=[],
            effectiveGasPrice="30000000000",
            deadline="2106-01-01T00:00:00.000Z",
            surplusCapturingJitOrderOwners=[],
        )

        response = solver.solve(auction)

        # Mock router always fails, so no solutions
        assert len(response.solutions) == 0
