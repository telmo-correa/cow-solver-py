"""Pytest configuration and fixtures."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from solver.amm.base import SwapResult
from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.models import AuctionInstance
from solver.routing.router import RoutingResult, SingleOrderRouter, Solver

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUCTIONS_DIR = FIXTURES_DIR / "auctions"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the fixtures directory path."""
    return FIXTURES_DIR


@pytest.fixture
def auctions_dir() -> Path:
    """Return the auctions fixtures directory path."""
    return AUCTIONS_DIR


def load_auction_fixture(name: str) -> AuctionInstance:
    """Load an auction fixture by name.

    Args:
        name: Fixture name (e.g., "single_order/basic_sell")

    Returns:
        Parsed AuctionInstance
    """
    path = AUCTIONS_DIR / f"{name}.json"
    with open(path) as f:
        data = json.load(f)
    return AuctionInstance.model_validate(data)


def iter_auction_fixtures(category: str | None = None) -> Iterator[tuple[str, AuctionInstance]]:
    """Iterate over auction fixtures, optionally filtered by category.

    Args:
        category: Optional subdirectory to filter (e.g., "single_order")

    Yields:
        Tuples of (fixture_name, AuctionInstance)
    """
    search_dir = AUCTIONS_DIR / category if category else AUCTIONS_DIR
    if not search_dir.exists():
        return

    for path in search_dir.rglob("*.json"):
        rel_path = path.relative_to(AUCTIONS_DIR)
        name = str(rel_path.with_suffix(""))
        yield name, load_auction_fixture(name)


@pytest.fixture
def single_order_auctions() -> list[tuple[str, AuctionInstance]]:
    """Load all single-order auction fixtures."""
    return list(iter_auction_fixtures("single_order"))


@pytest.fixture
def cow_pair_auctions() -> list[tuple[str, AuctionInstance]]:
    """Load all CoW pair auction fixtures."""
    return list(iter_auction_fixtures("cow_pairs"))


@pytest.fixture
def multi_hop_auctions() -> list[tuple[str, AuctionInstance]]:
    """Load all multi-hop auction fixtures."""
    return list(iter_auction_fixtures("multi_hop"))


# =============================================================================
# Mock classes for dependency injection
# =============================================================================


@dataclass
class MockSwapConfig:
    """Configuration for mock AMM swap behavior."""

    # Fixed output amount (if set, ignores input)
    fixed_output: int | None = None
    # Output multiplier (output = input * multiplier)
    output_multiplier: float | None = None
    # Default: realistic WETH/USDC rate (~2500 USDC per WETH)
    default_rate: float = 2500.0


class MockAMM:
    """Mock AMM for testing with configurable swap behavior.

    Usage:
        # Fixed output regardless of input
        mock_amm = MockAMM(MockSwapConfig(fixed_output=5000_000_000))

        # Output as multiplier of input
        mock_amm = MockAMM(MockSwapConfig(output_multiplier=2500.0))

        # Use default realistic rate
        mock_amm = MockAMM()
    """

    SWAP_GAS = 100_000

    def __init__(self, config: MockSwapConfig | None = None) -> None:
        self.config = config or MockSwapConfig()
        self.swap_calls: list[dict] = []  # Track calls for assertions

    def simulate_swap(self, pool: UniswapV2Pool, token_in: str, amount_in: int) -> SwapResult:
        """Simulate a swap with configurable output."""
        self.swap_calls.append(
            {
                "pool": pool.address,
                "token_in": token_in,
                "amount_in": amount_in,
            }
        )

        if self.config.fixed_output is not None:
            amount_out = self.config.fixed_output
        elif self.config.output_multiplier is not None:
            amount_out = int(amount_in * self.config.output_multiplier)
        else:
            # Default: assume WETH (18 decimals) to USDC (6 decimals)
            # 1 WETH = ~2500 USDC
            amount_out = int(amount_in * self.config.default_rate / 10**12)

        token_out = pool.token1 if token_in.lower() == pool.token0.lower() else pool.token0

        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            pool_address=pool.address,
            token_in=token_in,
            token_out=token_out,
            gas_estimate=self.SWAP_GAS,
        )

    def encode_swap(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        amount_out_min: int,
        recipient: str,
    ) -> tuple[str, str]:
        """Return mock calldata (valid hex format)."""
        # Mark interface-required params as intentionally unused
        _ = (token_in, token_out, recipient)
        # Encode amount_in and amount_out_min as hex for valid calldata
        calldata = f"0x{amount_in:064x}{amount_out_min:064x}"
        return (
            "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # UniswapV2Router02
            calldata,
        )


class MockPoolFinder:
    """Mock pool finder for testing.

    Usage:
        # Return a specific pool for any pair
        finder = MockPoolFinder(default_pool=my_pool)

        # Return different pools for different pairs
        finder = MockPoolFinder(pools={("0xtoken1", "0xtoken2"): pool1})

        # Return no pools (simulate unknown pair)
        finder = MockPoolFinder(default_pool=None)
    """

    def __init__(
        self,
        default_pool: UniswapV2Pool | None = None,
        pools: dict[tuple[str, str], UniswapV2Pool] | None = None,
    ) -> None:
        self.default_pool = default_pool
        self.pools = pools or {}
        self.calls: list[tuple[str, str]] = []  # Track calls for assertions

    def __call__(self, token_a: str, token_b: str) -> UniswapV2Pool | None:
        """Find a pool for the token pair."""
        self.calls.append((token_a, token_b))

        # Check both orderings
        key1 = (token_a.lower(), token_b.lower())
        key2 = (token_b.lower(), token_a.lower())

        if key1 in self.pools:
            return self.pools[key1]
        if key2 in self.pools:
            return self.pools[key2]

        return self.default_pool


class MockRouter:
    """Mock router for testing solver behavior.

    Usage:
        # Always succeed with fixed amounts
        router = MockRouter(success=True, amount_out=5000_000_000)

        # Always fail with error
        router = MockRouter(success=False, error="No liquidity")
    """

    def __init__(
        self,
        success: bool = True,
        amount_in: int = 1000000000000000000,
        amount_out: int = 2500_000_000,
        error: str | None = None,
    ) -> None:
        self.success = success
        self.amount_in = amount_in
        self.amount_out = amount_out
        self.error = error
        self.route_calls: list = []  # Track calls for assertions

    def route_order(self, order) -> RoutingResult:
        """Route an order with mock result."""
        self.route_calls.append(order)

        return RoutingResult(
            order=order,
            amount_in=self.amount_in if self.success else 0,
            amount_out=self.amount_out if self.success else 0,
            pool=None,
            success=self.success,
            error=self.error if not self.success else None,
        )

    def build_solution(self, routing_result: RoutingResult, solution_id: int = 0):
        """Delegate to real router for solution building."""
        _ = solution_id  # Interface-required param
        if not routing_result.success:
            return None
        # For simplicity, return None - mock router is for testing solve() logic
        return None


# =============================================================================
# Pytest fixtures for mocks
# =============================================================================


@pytest.fixture
def mock_weth_usdc_pool() -> UniswapV2Pool:
    """A mock WETH/USDC pool with reasonable reserves."""
    return UniswapV2Pool(
        address="0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
        token0="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
        token1="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        reserve0=10_000 * 10**18,  # 10,000 WETH
        reserve1=25_000_000 * 10**6,  # 25M USDC
        fee_bps=30,
    )


@pytest.fixture
def mock_amm() -> MockAMM:
    """A mock AMM with default realistic behavior."""
    return MockAMM()


@pytest.fixture
def mock_pool_finder(mock_weth_usdc_pool: UniswapV2Pool) -> MockPoolFinder:
    """A mock pool finder that returns the WETH/USDC pool."""
    return MockPoolFinder(default_pool=mock_weth_usdc_pool)


@pytest.fixture
def mock_router_success() -> MockRouter:
    """A mock router that always succeeds."""
    return MockRouter(success=True)


@pytest.fixture
def mock_router_failure() -> MockRouter:
    """A mock router that always fails."""
    return MockRouter(success=False, error="Mock routing failure")


@pytest.fixture
def injected_router(mock_amm: MockAMM, mock_pool_finder: MockPoolFinder) -> SingleOrderRouter:
    """A router with injected mock dependencies."""
    return SingleOrderRouter(amm=mock_amm, pool_finder=mock_pool_finder)


@pytest.fixture
def injected_solver(injected_router: SingleOrderRouter) -> Solver:
    """A solver with injected mock router."""
    return Solver(router=injected_router)
