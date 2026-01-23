"""Tests for pathfinding module."""

from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.pools import PoolRegistry
from solver.routing.pathfinding import PathFinder, TokenGraph

# Token addresses for testing
TOKEN_A = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
TOKEN_B = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
TOKEN_C = "0xcccccccccccccccccccccccccccccccccccccccc"
TOKEN_D = "0xdddddddddddddddddddddddddddddddddddddddd"


def make_v2_pool(token0: str, token1: str, address_suffix: str = "01") -> UniswapV2Pool:
    """Create a test V2 pool."""
    return UniswapV2Pool(
        address=f"0x{address_suffix}{'11' * 19}",
        token0=token0,
        token1=token1,
        reserve0=1000 * 10**18,
        reserve1=1000 * 10**18,
        gas_estimate=100000,
    )


class TestTokenGraph:
    """Tests for TokenGraph class."""

    def test_empty_registry(self) -> None:
        registry = PoolRegistry()
        graph = TokenGraph.from_registry(registry)
        assert graph.token_count == 0

    def test_single_pool(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B))
        graph = TokenGraph.from_registry(registry)

        assert graph.token_count == 2
        assert graph.has_token(TOKEN_A)
        assert graph.has_token(TOKEN_B)
        assert TOKEN_B.lower() in graph.get_neighbors(TOKEN_A)
        assert TOKEN_A.lower() in graph.get_neighbors(TOKEN_B)

    def test_multiple_pools_form_chain(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B, "01"))
        registry.add_pool(make_v2_pool(TOKEN_B, TOKEN_C, "02"))
        graph = TokenGraph.from_registry(registry)

        assert graph.token_count == 3
        # A connects to B
        assert TOKEN_B.lower() in graph.get_neighbors(TOKEN_A)
        # B connects to A and C
        assert TOKEN_A.lower() in graph.get_neighbors(TOKEN_B)
        assert TOKEN_C.lower() in graph.get_neighbors(TOKEN_B)
        # C connects to B
        assert TOKEN_B.lower() in graph.get_neighbors(TOKEN_C)

    def test_case_insensitive_tokens(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B))
        graph = TokenGraph.from_registry(registry)

        # Should work with different cases
        assert graph.has_token(TOKEN_A.upper())
        assert graph.has_token(TOKEN_A.lower())


class TestPathFinder:
    """Tests for PathFinder class."""

    def test_direct_path(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B))
        finder = PathFinder(registry)

        paths = finder.find_all_paths(TOKEN_A, TOKEN_B)
        assert len(paths) == 1
        assert paths[0] == [TOKEN_A.lower(), TOKEN_B.lower()]

    def test_two_hop_path(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B, "01"))
        registry.add_pool(make_v2_pool(TOKEN_B, TOKEN_C, "02"))
        finder = PathFinder(registry)

        paths = finder.find_all_paths(TOKEN_A, TOKEN_C)
        assert len(paths) == 1
        assert paths[0] == [TOKEN_A.lower(), TOKEN_B.lower(), TOKEN_C.lower()]

    def test_multiple_paths(self) -> None:
        registry = PoolRegistry()
        # Direct path A -> C
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_C, "01"))
        # Two-hop path A -> B -> C
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B, "02"))
        registry.add_pool(make_v2_pool(TOKEN_B, TOKEN_C, "03"))
        finder = PathFinder(registry)

        paths = finder.find_all_paths(TOKEN_A, TOKEN_C)
        assert len(paths) == 2
        # Should include both direct and two-hop
        path_lengths = sorted(len(p) for p in paths)
        assert path_lengths == [2, 3]  # Direct (2 nodes) and two-hop (3 nodes)

    def test_no_path(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B))
        finder = PathFinder(registry)

        paths = finder.find_all_paths(TOKEN_A, TOKEN_C)
        assert paths == []

    def test_same_token(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B))
        finder = PathFinder(registry)

        paths = finder.find_all_paths(TOKEN_A, TOKEN_A)
        assert paths == []

    def test_max_hops_limit(self) -> None:
        registry = PoolRegistry()
        # Chain: A -> B -> C -> D
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B, "01"))
        registry.add_pool(make_v2_pool(TOKEN_B, TOKEN_C, "02"))
        registry.add_pool(make_v2_pool(TOKEN_C, TOKEN_D, "03"))
        finder = PathFinder(registry)

        # With default max_hops=3, should find A -> B -> C -> D
        paths = finder.find_all_paths(TOKEN_A, TOKEN_D)
        assert len(paths) == 1  # 3 hops path found

        # With max_hops=2, should NOT find it (3 hops needed)
        paths = finder.find_all_paths(TOKEN_A, TOKEN_D, max_hops=2)
        assert paths == []

        # With explicit max_hops=3, should find it
        paths = finder.find_all_paths(TOKEN_A, TOKEN_D, max_hops=3)
        assert len(paths) == 1

    def test_find_shortest_path(self) -> None:
        registry = PoolRegistry()
        # Direct path
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_C, "01"))
        # Longer path A -> B -> C
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B, "02"))
        registry.add_pool(make_v2_pool(TOKEN_B, TOKEN_C, "03"))
        finder = PathFinder(registry)

        path = finder.find_shortest_path(TOKEN_A, TOKEN_C)
        assert path is not None
        # Should be the direct path
        assert len(path) == 2
        assert path == [TOKEN_A.lower(), TOKEN_C.lower()]

    def test_find_shortest_path_no_route(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B))
        finder = PathFinder(registry)

        path = finder.find_shortest_path(TOKEN_A, TOKEN_C)
        assert path is None

    def test_find_shortest_path_same_token(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B))
        finder = PathFinder(registry)

        path = finder.find_shortest_path(TOKEN_A, TOKEN_A)
        assert path == [TOKEN_A.lower()]


class TestPathFinderCaching:
    """Tests for PathFinder cache invalidation."""

    def test_cache_invalidation_on_add_pool(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B, "01"))
        # Use registry.pathfinder so invalidation callback is registered
        finder = registry.pathfinder

        # Build graph
        paths = finder.find_all_paths(TOKEN_A, TOKEN_C)
        assert paths == []  # No path yet

        # Add new pool - this triggers _invalidate_pathfinder()
        registry.add_pool(make_v2_pool(TOKEN_B, TOKEN_C, "02"))

        # Graph should be invalidated, new paths should be found
        paths = finder.find_all_paths(TOKEN_A, TOKEN_C)
        assert len(paths) == 1

    def test_explicit_invalidation(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B))
        finder = PathFinder(registry)

        # Build graph
        _ = finder.graph

        # Explicit invalidation
        finder.invalidate()

        # Should rebuild on next access
        assert finder._graph is None
        _ = finder.graph
        assert finder._graph is not None


class TestPoolRegistryPathfindingDelegation:
    """Tests that PoolRegistry correctly delegates to PathFinder."""

    def test_get_all_candidate_paths_delegation(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B, "01"))
        registry.add_pool(make_v2_pool(TOKEN_B, TOKEN_C, "02"))

        # Use registry's method (delegates to PathFinder)
        paths = registry.get_all_candidate_paths(TOKEN_A, TOKEN_C)
        assert len(paths) == 1
        assert paths[0] == [TOKEN_A.lower(), TOKEN_B.lower(), TOKEN_C.lower()]

    def test_find_path_delegation(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B, "01"))
        registry.add_pool(make_v2_pool(TOKEN_B, TOKEN_C, "02"))

        # Use registry's method (delegates to PathFinder)
        path = registry.find_path(TOKEN_A, TOKEN_C)
        assert path is not None
        assert path == [TOKEN_A.lower(), TOKEN_B.lower(), TOKEN_C.lower()]

    def test_pathfinder_property_lazy_init(self) -> None:
        registry = PoolRegistry()
        registry.add_pool(make_v2_pool(TOKEN_A, TOKEN_B))

        # PathFinder should be lazy-initialized
        assert registry._pathfinder is None

        # Access pathfinder property
        finder = registry.pathfinder
        assert finder is not None
        assert registry._pathfinder is finder

        # Should return same instance
        assert registry.pathfinder is finder
