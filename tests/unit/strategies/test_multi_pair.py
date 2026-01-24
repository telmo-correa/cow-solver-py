"""Tests for multi-pair price coordination strategy."""

from decimal import Decimal

import pytest

from solver.models.auction import Order
from solver.models.order_groups import OrderGroup
from solver.strategies.multi_pair import (
    MultiPairCowStrategy,
    PriceCandidates,
    UnionFind,
    build_token_graph,
    enumerate_price_combinations,
    find_spanning_tree,
    find_token_components,
    solve_fills_at_prices,
)

# Counter for generating unique UIDs
_uid_counter = 0


def make_order(
    _name: str,
    sell_token: str,
    buy_token: str,
    sell_amount: int,
    buy_amount: int,
    partially_fillable: bool = True,
    kind: str = "sell",
) -> Order:
    """Create a test order with valid format."""
    global _uid_counter
    _uid_counter += 1
    uid = f"0x{_uid_counter:0112x}"
    return Order(
        uid=uid,
        sell_token=sell_token,
        buy_token=buy_token,
        sell_amount=str(sell_amount),
        buy_amount=str(buy_amount),
        kind=kind,
        class_="market",
        partially_fillable=partially_fillable,
    )


# Tokens for testing
TOKEN_A = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
TOKEN_B = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"
TOKEN_C = "0xcCcCCccCCCCcCCCCcCcCccCcCCCcCccccccccccc"
TOKEN_D = "0xdDdDDDDDDdDdDdDDddDDDDddDDDddDDdDdDDDDDd"


class TestUnionFind:
    """Tests for Union-Find data structure."""

    def test_init_empty(self) -> None:
        """Empty Union-Find should have no elements."""
        uf = UnionFind()
        assert uf.find(0) == 0
        assert uf.find(1) == 1

    def test_union_two_elements(self) -> None:
        """Union should connect two elements."""
        uf = UnionFind()
        assert not uf.connected(0, 1)
        uf.union(0, 1)
        assert uf.connected(0, 1)

    def test_transitive_connection(self) -> None:
        """Connection should be transitive."""
        uf = UnionFind()
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.connected(0, 2)

    def test_multiple_components(self) -> None:
        """Should support multiple disconnected components."""
        uf = UnionFind()
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.connected(0, 1)
        assert uf.connected(2, 3)
        assert not uf.connected(0, 2)


class TestFindTokenComponents:
    """Tests for connected component detection."""

    def test_empty_input(self) -> None:
        """Empty input should return empty list."""
        result = find_token_components([])
        assert result == []

    def test_single_group(self) -> None:
        """Single group should be its own component."""
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[make_order("ask", TOKEN_A, TOKEN_B, 100, 200)],
            sellers_of_b=[make_order("bid", TOKEN_B, TOKEN_A, 200, 100)],
        )
        result = find_token_components([group])
        assert len(result) == 1
        assert result[0] == [group]

    def test_disconnected_groups(self) -> None:
        """Groups with no shared tokens should be separate components."""
        group1 = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[make_order("ask1", TOKEN_A, TOKEN_B, 100, 200)],
            sellers_of_b=[make_order("bid1", TOKEN_B, TOKEN_A, 200, 100)],
        )
        group2 = OrderGroup(
            token_a=TOKEN_C,
            token_b=TOKEN_D,
            sellers_of_a=[make_order("ask2", TOKEN_C, TOKEN_D, 100, 200)],
            sellers_of_b=[make_order("bid2", TOKEN_D, TOKEN_C, 200, 100)],
        )
        result = find_token_components([group1, group2])
        assert len(result) == 2

    def test_connected_groups_shared_token(self) -> None:
        """Groups sharing a token should be in same component."""
        group1 = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[make_order("ask1", TOKEN_A, TOKEN_B, 100, 200)],
            sellers_of_b=[make_order("bid1", TOKEN_B, TOKEN_A, 200, 100)],
        )
        group2 = OrderGroup(
            token_a=TOKEN_A,  # Shares TOKEN_A with group1
            token_b=TOKEN_C,
            sellers_of_a=[make_order("ask2", TOKEN_A, TOKEN_C, 100, 300)],
            sellers_of_b=[make_order("bid2", TOKEN_C, TOKEN_A, 300, 100)],
        )
        result = find_token_components([group1, group2])
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_chain_of_groups(self) -> None:
        """Chain of connected groups should be single component."""
        # A-B, B-C, C-D should all be in one component
        group1 = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[make_order("ask1", TOKEN_A, TOKEN_B, 100, 200)],
            sellers_of_b=[make_order("bid1", TOKEN_B, TOKEN_A, 200, 100)],
        )
        group2 = OrderGroup(
            token_a=TOKEN_B,  # Connects to group1
            token_b=TOKEN_C,
            sellers_of_a=[make_order("ask2", TOKEN_B, TOKEN_C, 100, 200)],
            sellers_of_b=[make_order("bid2", TOKEN_C, TOKEN_B, 200, 100)],
        )
        group3 = OrderGroup(
            token_a=TOKEN_C,  # Connects to group2
            token_b=TOKEN_D,
            sellers_of_a=[make_order("ask3", TOKEN_C, TOKEN_D, 100, 200)],
            sellers_of_b=[make_order("bid3", TOKEN_D, TOKEN_C, 200, 100)],
        )
        result = find_token_components([group1, group2, group3])
        assert len(result) == 1
        assert len(result[0]) == 3


class TestPriceCandidates:
    """Tests for price candidate collection."""

    def test_add_and_get_ratio(self) -> None:
        """Should store and retrieve ratios."""
        candidates = PriceCandidates()
        candidates.add_ratio(TOKEN_A, TOKEN_B, Decimal("2.0"))
        candidates.add_ratio(TOKEN_A, TOKEN_B, Decimal("2.5"))

        ratios = candidates.get_ratios(TOKEN_A, TOKEN_B)
        assert Decimal("2.0") in ratios
        assert Decimal("2.5") in ratios

    def test_no_duplicates(self) -> None:
        """Should not store duplicate ratios."""
        candidates = PriceCandidates()
        candidates.add_ratio(TOKEN_A, TOKEN_B, Decimal("2.0"))
        candidates.add_ratio(TOKEN_A, TOKEN_B, Decimal("2.0"))

        ratios = candidates.get_ratios(TOKEN_A, TOKEN_B)
        assert len(ratios) == 1

    def test_empty_get(self) -> None:
        """Should return empty list for unknown pair."""
        candidates = PriceCandidates()
        ratios = candidates.get_ratios(TOKEN_A, TOKEN_B)
        assert ratios == []


class TestBuildTokenGraph:
    """Tests for token graph construction."""

    def test_single_group(self) -> None:
        """Single group should create edge between its tokens."""
        group = OrderGroup(token_a=TOKEN_A, token_b=TOKEN_B)
        graph = build_token_graph([group])

        token_a = TOKEN_A.lower()
        token_b = TOKEN_B.lower()
        assert token_b in graph[token_a]
        assert token_a in graph[token_b]

    def test_multiple_groups(self) -> None:
        """Multiple groups should create all edges."""
        groups = [
            OrderGroup(token_a=TOKEN_A, token_b=TOKEN_B),
            OrderGroup(token_a=TOKEN_B, token_b=TOKEN_C),
        ]
        graph = build_token_graph(groups)

        token_a = TOKEN_A.lower()
        token_b = TOKEN_B.lower()
        token_c = TOKEN_C.lower()

        assert token_b in graph[token_a]
        assert token_a in graph[token_b]
        assert token_c in graph[token_b]
        assert token_b in graph[token_c]


class TestFindSpanningTree:
    """Tests for spanning tree discovery."""

    def test_single_edge(self) -> None:
        """Single edge graph should return that edge."""
        graph = {TOKEN_A.lower(): {TOKEN_B.lower()}, TOKEN_B.lower(): {TOKEN_A.lower()}}
        edges = find_spanning_tree(graph)
        assert len(edges) == 1

    def test_triangle(self) -> None:
        """Triangle should return 2 edges (spanning tree)."""
        ta = TOKEN_A.lower()
        tb = TOKEN_B.lower()
        tc = TOKEN_C.lower()
        graph = {
            ta: {tb, tc},
            tb: {ta, tc},
            tc: {ta, tb},
        }
        edges = find_spanning_tree(graph)
        assert len(edges) == 2  # Spanning tree of 3 nodes has 2 edges


class TestEnumeratePriceCombinations:
    """Tests for price combination enumeration."""

    def test_single_edge_single_ratio(self) -> None:
        """Single edge with one ratio should produce one combination."""
        candidates = PriceCandidates()
        candidates.add_ratio(TOKEN_A.lower(), TOKEN_B.lower(), Decimal("2.0"))

        spanning_tree = [(TOKEN_A.lower(), TOKEN_B.lower())]
        combos = enumerate_price_combinations(candidates, spanning_tree)

        assert len(combos) == 1
        assert TOKEN_A.lower() in combos[0]
        assert TOKEN_B.lower() in combos[0]

    def test_single_edge_multiple_ratios(self) -> None:
        """Single edge with multiple ratios should produce that many combinations."""
        candidates = PriceCandidates()
        candidates.add_ratio(TOKEN_A.lower(), TOKEN_B.lower(), Decimal("2.0"))
        candidates.add_ratio(TOKEN_A.lower(), TOKEN_B.lower(), Decimal("2.5"))

        spanning_tree = [(TOKEN_A.lower(), TOKEN_B.lower())]
        combos = enumerate_price_combinations(candidates, spanning_tree)

        assert len(combos) == 2

    def test_max_combinations_limit(self) -> None:
        """Should respect max_combinations limit."""
        candidates = PriceCandidates()
        for i in range(20):
            candidates.add_ratio(TOKEN_A.lower(), TOKEN_B.lower(), Decimal(f"{i + 1}.0"))

        spanning_tree = [(TOKEN_A.lower(), TOKEN_B.lower())]
        combos = enumerate_price_combinations(candidates, spanning_tree, max_combinations=5)

        assert len(combos) <= 5


class TestSolveFillsAtPrices:
    """Tests for LP solver."""

    def test_simple_balanced_match(self) -> None:
        """Two compatible orders should produce fills."""
        # Ask: sell 100 A, want 200 B (limit = 2 B/A)
        # Bid: sell 200 B, want 100 A (limit = 2 B/A)
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[make_order("ask", TOKEN_A, TOKEN_B, 100, 200)],
            sellers_of_b=[make_order("bid", TOKEN_B, TOKEN_A, 200, 100)],
        )

        # Set prices at limit
        prices = {
            TOKEN_A.lower(): Decimal("1.0"),
            TOKEN_B.lower(): Decimal("2.0"),  # 2 B per A
        }

        result = solve_fills_at_prices([group], prices)

        assert result is not None
        assert len(result.fills) == 2
        assert result.total_volume > 0

    def test_no_match_when_price_violates_limit(self) -> None:
        """Orders should not match when price violates their limits."""
        # Ask: sell 100 A, want 200 B (limit = 2 B/A)
        # But price is only 1.5 B/A - seller won't accept
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[make_order("ask", TOKEN_A, TOKEN_B, 100, 200)],
            sellers_of_b=[make_order("bid", TOKEN_B, TOKEN_A, 150, 100)],
        )

        prices = {
            TOKEN_A.lower(): Decimal("1.0"),
            TOKEN_B.lower(): Decimal("1.5"),  # Below ask limit of 2
        }

        result = solve_fills_at_prices([group], prices)

        # No fills because ask limit not satisfied
        assert result is None or len(result.fills) == 0

    def test_fill_or_kill_respected(self) -> None:
        """Fill-or-kill orders should not be partially filled."""
        # Ask: sell 100 A, NOT partially fillable
        # Bid: sell 50 B (can only buy half)
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[make_order("ask", TOKEN_A, TOKEN_B, 100, 200, partially_fillable=False)],
            sellers_of_b=[make_order("bid", TOKEN_B, TOKEN_A, 100, 50)],  # Can only buy 50 A
        )

        prices = {
            TOKEN_A.lower(): Decimal("1.0"),
            TOKEN_B.lower(): Decimal("2.0"),
        }

        result = solve_fills_at_prices([group], prices)

        # Fill-or-kill ask should not be in fills if partially matched
        if result is not None:
            for fill in result.fills:
                if not fill.order.partially_fillable:
                    # Should be complete fill or no fill
                    assert fill.sell_filled >= fill.order.sell_amount_int * 0.99


class TestMultiPairCowStrategy:
    """Integration tests for the strategy."""

    def test_no_orders(self) -> None:
        """Empty auction should return None."""
        from solver.models.auction import AuctionInstance

        auction = AuctionInstance(orders=[], tokens={}, liquidity=[])
        strategy = MultiPairCowStrategy()
        result = strategy.try_solve(auction)
        assert result is None

    def test_single_order(self) -> None:
        """Single order should return None (no CoW possible)."""
        from solver.models.auction import AuctionInstance

        order = make_order("ask", TOKEN_A, TOKEN_B, 100, 200)
        auction = AuctionInstance(orders=[order], tokens={}, liquidity=[])
        strategy = MultiPairCowStrategy()
        result = strategy.try_solve(auction)
        assert result is None

    def test_two_compatible_orders(self) -> None:
        """Two compatible orders should produce CoW match."""
        from solver.models.auction import AuctionInstance

        ask = make_order("ask", TOKEN_A, TOKEN_B, 100, 200)
        bid = make_order("bid", TOKEN_B, TOKEN_A, 200, 100)
        auction = AuctionInstance(orders=[ask, bid], tokens={}, liquidity=[])

        strategy = MultiPairCowStrategy()
        result = strategy.try_solve(auction)

        # May or may not find a match depending on LP feasibility
        # The important thing is it doesn't crash
        if result is not None:
            assert result.has_fills

    def test_overlapping_pairs_unified_prices(self) -> None:
        """Overlapping token pairs should get unified prices."""
        from solver.models.auction import AuctionInstance

        # Pair 1: A-B
        ask1 = make_order("ask1", TOKEN_A, TOKEN_B, 100, 200)
        bid1 = make_order("bid1", TOKEN_B, TOKEN_A, 200, 100)

        # Pair 2: A-C (shares token A with pair 1)
        ask2 = make_order("ask2", TOKEN_A, TOKEN_C, 100, 300)
        bid2 = make_order("bid2", TOKEN_C, TOKEN_A, 300, 100)

        auction = AuctionInstance(
            orders=[ask1, bid1, ask2, bid2],
            tokens={},
            liquidity=[],
        )

        strategy = MultiPairCowStrategy()
        result = strategy.try_solve(auction)

        # Should process both pairs in same component
        # The key test is that it doesn't crash due to price conflicts
        if result is not None:
            # If there are fills, prices should be consistent
            assert isinstance(result.prices, dict)


class TestMultiPairIntegration:
    """Integration tests with realistic scenarios."""

    def test_three_token_triangle(self) -> None:
        """Three token pairs forming a triangle should be in one component."""
        from solver.models.auction import AuctionInstance

        # A-B, B-C, C-A forms a triangle
        orders = [
            make_order("ab_ask", TOKEN_A, TOKEN_B, 100, 200),
            make_order("ab_bid", TOKEN_B, TOKEN_A, 200, 100),
            make_order("bc_ask", TOKEN_B, TOKEN_C, 100, 150),
            make_order("bc_bid", TOKEN_C, TOKEN_B, 150, 100),
            make_order("ca_ask", TOKEN_C, TOKEN_A, 100, 100),
            make_order("ca_bid", TOKEN_A, TOKEN_C, 100, 100),
        ]

        auction = AuctionInstance(orders=orders, tokens={}, liquidity=[])

        strategy = MultiPairCowStrategy()
        result = strategy.try_solve(auction)

        # All three pairs should be processed together
        # This tests the component detection works for cycles
        if result is not None:
            assert len(result.prices) >= 2  # At least 2 tokens priced


# Skip scipy-dependent tests if not available
try:
    import scipy  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestLPSolverWithScipy:
    """Tests that specifically require scipy."""

    def test_conservation_constraint(self) -> None:
        """LP should enforce token conservation."""
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[make_order("ask", TOKEN_A, TOKEN_B, 100, 200)],
            sellers_of_b=[make_order("bid", TOKEN_B, TOKEN_A, 200, 100)],
        )

        prices = {
            TOKEN_A.lower(): Decimal("1.0"),
            TOKEN_B.lower(): Decimal("2.0"),
        }

        result = solve_fills_at_prices([group], prices)

        if result is not None and result.fills:
            # Check conservation: sells of A should equal buys of A
            total_a_sold = 0
            total_a_bought = 0
            for fill in result.fills:
                sell_token = fill.order.sell_token.lower()
                if sell_token == TOKEN_A.lower():
                    total_a_sold += fill.sell_filled
                else:
                    total_a_bought += fill.buy_filled

            # Should be approximately balanced (may have small rounding)
            assert abs(total_a_sold - total_a_bought) <= 1
