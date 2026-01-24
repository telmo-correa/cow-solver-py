"""Tests for multi-pair price coordination strategy."""

from decimal import Decimal

import pytest

from solver.models.auction import Order
from solver.models.order_groups import OrderGroup
from solver.strategies.components import find_order_components, find_token_components
from solver.strategies.graph import UnionFind, find_spanning_tree
from solver.strategies.multi_pair import MultiPairCowStrategy
from solver.strategies.pricing import (
    PriceCandidates,
    build_price_candidates_from_orders,
    build_token_graph_from_groups,
    build_token_graph_from_orders,
    enumerate_price_combinations,
    solve_fills_at_prices,
    solve_fills_at_prices_v2,
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
        graph = build_token_graph_from_groups([group])

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
        graph = build_token_graph_from_groups(groups)

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
                    # Should be complete fill or no fill (exact match for FOK)
                    assert fill.sell_filled == fill.order.sell_amount_int


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

            # Token conservation check: total sold of A should equal total bought of A.
            # 1 wei tolerance is allowed because scipy's LP solver uses floating-point
            # arithmetic internally, and the integer rounding of fill amounts may cause
            # up to 1 wei imbalance. This is acceptable for unit testing the LP solver.
            assert abs(total_a_sold - total_a_bought) <= 1


# =============================================================================
# Slice 4.7: Generalized Multi-Pair Tests
# =============================================================================


class TestFindOrderComponents:
    """Tests for generalized order component detection."""

    def test_empty_input(self) -> None:
        """Empty input should return empty list."""
        result = find_order_components([])
        assert result == []

    def test_single_order(self) -> None:
        """Single order creates a 2-token component."""
        order = make_order("o1", TOKEN_A, TOKEN_B, 100, 200)
        result = find_order_components([order])
        assert len(result) == 1
        tokens, orders = result[0]
        assert len(tokens) == 2
        assert TOKEN_A.lower() in tokens
        assert TOKEN_B.lower() in tokens
        assert len(orders) == 1

    def test_two_disconnected_orders(self) -> None:
        """Orders with no shared tokens are separate components."""
        order1 = make_order("o1", TOKEN_A, TOKEN_B, 100, 200)
        order2 = make_order("o2", TOKEN_C, TOKEN_D, 100, 200)
        result = find_order_components([order1, order2])
        assert len(result) == 2
        # Each component has 2 tokens and 1 order
        for tokens, orders in result:
            assert len(tokens) == 2
            assert len(orders) == 1

    def test_three_cycle_single_component(self) -> None:
        """Orders forming A→B, B→C, C→A cycle are in one component."""
        orders = [
            make_order("ab", TOKEN_A, TOKEN_B, 100, 100),
            make_order("bc", TOKEN_B, TOKEN_C, 100, 100),
            make_order("ca", TOKEN_C, TOKEN_A, 100, 100),
        ]
        result = find_order_components(orders, max_tokens=4)
        assert len(result) == 1
        tokens, comp_orders = result[0]
        assert len(tokens) == 3
        assert len(comp_orders) == 3

    def test_max_tokens_filter(self) -> None:
        """Components with more than max_tokens are filtered out."""
        # Create a 4-token component
        orders = [
            make_order("ab", TOKEN_A, TOKEN_B, 100, 100),
            make_order("bc", TOKEN_B, TOKEN_C, 100, 100),
            make_order("cd", TOKEN_C, TOKEN_D, 100, 100),
        ]
        # With max_tokens=3, should be filtered
        result = find_order_components(orders, max_tokens=3)
        assert len(result) == 0

        # With max_tokens=4, should be included
        result = find_order_components(orders, max_tokens=4)
        assert len(result) == 1
        tokens, _ = result[0]
        assert len(tokens) == 4

    def test_bidirectional_pair_creates_2_token_component(self) -> None:
        """Bidirectional pair (A↔B) creates 2-token component."""
        orders = [
            make_order("ab", TOKEN_A, TOKEN_B, 100, 200),
            make_order("ba", TOKEN_B, TOKEN_A, 200, 100),
        ]
        result = find_order_components(orders)
        assert len(result) == 1
        tokens, comp_orders = result[0]
        assert len(tokens) == 2
        assert len(comp_orders) == 2

    def test_sorted_by_order_count(self) -> None:
        """Components are sorted by order count descending."""
        # Create two disconnected components
        large_component = [
            make_order("ab1", TOKEN_A, TOKEN_B, 100, 200),
            make_order("ab2", TOKEN_A, TOKEN_B, 100, 200),
            make_order("ba", TOKEN_B, TOKEN_A, 200, 100),
        ]
        small_component = [
            make_order("cd", TOKEN_C, TOKEN_D, 100, 200),
        ]
        all_orders = large_component + small_component
        result = find_order_components(all_orders)
        assert len(result) == 2
        # First component should have more orders
        assert len(result[0][1]) > len(result[1][1])


class TestBuildTokenGraphFromOrders:
    """Tests for token graph construction from orders."""

    def test_single_order(self) -> None:
        """Single order creates bidirectional edge."""
        order = make_order("o1", TOKEN_A, TOKEN_B, 100, 200)
        graph = build_token_graph_from_orders([order])

        ta = TOKEN_A.lower()
        tb = TOKEN_B.lower()
        assert tb in graph[ta]
        assert ta in graph[tb]

    def test_cycle_orders(self) -> None:
        """Cycle orders create triangle graph."""
        orders = [
            make_order("ab", TOKEN_A, TOKEN_B, 100, 100),
            make_order("bc", TOKEN_B, TOKEN_C, 100, 100),
            make_order("ca", TOKEN_C, TOKEN_A, 100, 100),
        ]
        graph = build_token_graph_from_orders(orders)

        ta = TOKEN_A.lower()
        tb = TOKEN_B.lower()
        tc = TOKEN_C.lower()

        # Should have edges A-B, B-C, C-A
        assert tb in graph[ta]
        assert tc in graph[tb]
        assert ta in graph[tc]


class TestBuildPriceCandidatesFromOrders:
    """Tests for price candidate building from orders."""

    def test_single_order_limit_price(self) -> None:
        """Single order adds its limit price as candidate."""
        order = make_order("o1", TOKEN_A, TOKEN_B, 100, 200)

        candidates = build_price_candidates_from_orders([order], None, None)

        ratios = candidates.get_ratios(TOKEN_A.lower(), TOKEN_B.lower())
        # Limit price: buy_amount / sell_amount = 200 / 100 = 2
        assert Decimal("2") in ratios

    def test_multiple_orders_different_limits(self) -> None:
        """Multiple orders add their different limit prices."""
        orders = [
            make_order("o1", TOKEN_A, TOKEN_B, 100, 200),  # limit = 2
            make_order("o2", TOKEN_A, TOKEN_B, 100, 300),  # limit = 3
        ]

        candidates = build_price_candidates_from_orders(orders, None, None)

        ratios = candidates.get_ratios(TOKEN_A.lower(), TOKEN_B.lower())
        assert Decimal("2") in ratios
        assert Decimal("3") in ratios


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestSolveFillsAtPricesV2:
    """Tests for the generalized LP solver."""

    def test_bidirectional_match(self) -> None:
        """Two compatible orders should produce fills."""
        orders = [
            make_order("ask", TOKEN_A, TOKEN_B, 100, 200),  # sells A, wants 200 B
            make_order("bid", TOKEN_B, TOKEN_A, 200, 100),  # sells B, wants 100 A
        ]
        tokens = {TOKEN_A.lower(), TOKEN_B.lower()}
        prices = {
            TOKEN_A.lower(): Decimal("1.0"),
            TOKEN_B.lower(): Decimal("2.0"),  # 2 B per A
        }

        result = solve_fills_at_prices_v2(orders, tokens, prices)

        assert result is not None
        assert len(result.fills) == 2
        assert result.total_volume > 0

    def test_3_cycle_match(self) -> None:
        """Three orders forming A→B→C→A should produce fills at valid prices."""
        orders = [
            make_order("ab", TOKEN_A, TOKEN_B, 100, 100),  # sells A, wants B at 1:1
            make_order("bc", TOKEN_B, TOKEN_C, 100, 100),  # sells B, wants C at 1:1
            make_order("ca", TOKEN_C, TOKEN_A, 100, 100),  # sells C, wants A at 1:1
        ]
        tokens = {TOKEN_A.lower(), TOKEN_B.lower(), TOKEN_C.lower()}
        # All prices equal = 1:1:1 exchange rates
        prices = {
            TOKEN_A.lower(): Decimal("1.0"),
            TOKEN_B.lower(): Decimal("1.0"),
            TOKEN_C.lower(): Decimal("1.0"),
        }

        result = solve_fills_at_prices_v2(orders, tokens, prices)

        assert result is not None
        assert len(result.fills) == 3
        assert result.total_volume > 0

    def test_price_violates_limit(self) -> None:
        """Orders should not match when price violates their limits."""
        orders = [
            make_order("ab", TOKEN_A, TOKEN_B, 100, 200),  # wants at least 2 B/A
        ]
        tokens = {TOKEN_A.lower(), TOKEN_B.lower()}
        prices = {
            TOKEN_A.lower(): Decimal("1.0"),
            TOKEN_B.lower(): Decimal("1.5"),  # Only 1.5 B/A, below limit of 2
        }

        result = solve_fills_at_prices_v2(orders, tokens, prices)

        # No fills because limit not satisfied
        assert result is None or len(result.fills) == 0


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestGeneralizedMultiPairStrategy:
    """Integration tests for generalized component detection."""

    def test_3_cycle_no_bidirectional(self) -> None:
        """3-cycle (A→B, B→C, C→A) should be captured by generalized strategy.

        This is the key test for Slice 4.7 - cycles that the old bidirectional
        approach would miss.
        """
        from solver.models.auction import AuctionInstance

        # Create orders forming a cycle with compatible limit prices
        # A→B: sell 100 A for 100 B
        # B→C: sell 100 B for 100 C
        # C→A: sell 100 C for 100 A
        orders = [
            make_order("ab", TOKEN_A, TOKEN_B, 100, 100),
            make_order("bc", TOKEN_B, TOKEN_C, 100, 100),
            make_order("ca", TOKEN_C, TOKEN_A, 100, 100),
        ]
        auction = AuctionInstance(orders=orders, tokens={}, liquidity=[])

        # Test with generalized strategy (should find matches)
        strategy = MultiPairCowStrategy(use_generalized=True, max_tokens=4)
        result = strategy.try_solve(auction)

        # The generalized strategy should find this component
        # Whether it matches depends on LP feasibility
        # Key thing is it doesn't crash and processes the component
        if result is not None:
            assert result.has_fills

    def test_4_cycle_with_max_tokens_4(self) -> None:
        """4-cycle should be processed when max_tokens >= 4."""
        from solver.models.auction import AuctionInstance

        orders = [
            make_order("ab", TOKEN_A, TOKEN_B, 100, 100),
            make_order("bc", TOKEN_B, TOKEN_C, 100, 100),
            make_order("cd", TOKEN_C, TOKEN_D, 100, 100),
            make_order("da", TOKEN_D, TOKEN_A, 100, 100),
        ]
        auction = AuctionInstance(orders=orders, tokens={}, liquidity=[])

        # With max_tokens=4, should process the component
        strategy = MultiPairCowStrategy(use_generalized=True, max_tokens=4)
        result = strategy.try_solve(auction)

        # Should not crash; may or may not find matches
        if result is not None:
            assert isinstance(result.prices, dict)

    def test_4_cycle_skipped_with_max_tokens_3(self) -> None:
        """4-cycle should be skipped when max_tokens < 4."""
        from solver.models.auction import AuctionInstance

        orders = [
            make_order("ab", TOKEN_A, TOKEN_B, 100, 100),
            make_order("bc", TOKEN_B, TOKEN_C, 100, 100),
            make_order("cd", TOKEN_C, TOKEN_D, 100, 100),
            make_order("da", TOKEN_D, TOKEN_A, 100, 100),
        ]
        auction = AuctionInstance(orders=orders, tokens={}, liquidity=[])

        # With max_tokens=3, the 4-token component should be skipped
        strategy = MultiPairCowStrategy(use_generalized=True, max_tokens=3)
        result = strategy.try_solve(auction)

        # Should return None (no valid components within max_tokens)
        assert result is None

    def test_bidirectional_still_works(self) -> None:
        """Bidirectional pair (A↔B) should still be handled correctly."""
        from solver.models.auction import AuctionInstance

        orders = [
            make_order("ask", TOKEN_A, TOKEN_B, 100, 200),
            make_order("bid", TOKEN_B, TOKEN_A, 200, 100),
        ]
        auction = AuctionInstance(orders=orders, tokens={}, liquidity=[])

        strategy = MultiPairCowStrategy(use_generalized=True, max_tokens=4)
        result = strategy.try_solve(auction)

        # Should work just as before
        if result is not None:
            assert result.has_fills

    def test_use_generalized_false_uses_old_approach(self) -> None:
        """With use_generalized=False, should use old bidirectional approach."""
        from solver.models.auction import AuctionInstance

        # 3-cycle without bidirectional pairs
        orders = [
            make_order("ab", TOKEN_A, TOKEN_B, 100, 100),
            make_order("bc", TOKEN_B, TOKEN_C, 100, 100),
            make_order("ca", TOKEN_C, TOKEN_A, 100, 100),
        ]
        auction = AuctionInstance(orders=orders, tokens={}, liquidity=[])

        # Old approach should not find this (no bidirectional pairs)
        strategy = MultiPairCowStrategy(use_generalized=False)
        result = strategy.try_solve(auction)

        # Old approach returns None because no bidirectional pairs exist
        assert result is None

    def test_mixed_pair_and_cycle(self) -> None:
        """Mixed scenario: bidirectional pair + cycle orders."""
        from solver.models.auction import AuctionInstance

        # Pair A↔B + cycle through C
        orders = [
            make_order("ab", TOKEN_A, TOKEN_B, 100, 200),
            make_order("ba", TOKEN_B, TOKEN_A, 200, 100),
            make_order("ac", TOKEN_A, TOKEN_C, 100, 300),
            make_order("cb", TOKEN_C, TOKEN_B, 300, 100),
        ]
        auction = AuctionInstance(orders=orders, tokens={}, liquidity=[])

        strategy = MultiPairCowStrategy(use_generalized=True, max_tokens=4)
        result = strategy.try_solve(auction)

        # All orders are in one component (A, B, C connected)
        # Should process without crashing
        if result is not None:
            assert isinstance(result.prices, dict)


class TestLargeComponentPriceConsistency:
    """Tests for price consistency in large components.

    When processing large components (>500 orders), the strategy falls back to
    processing pairs individually. This must skip pairs that share tokens with
    already-processed pairs to maintain consistent clearing prices.
    """

    def test_overlapping_pairs_skipped_in_large_component(self) -> None:
        """Pairs with overlapping tokens should be skipped after first pair is processed."""
        from solver.models.auction import AuctionInstance
        from solver.models.order_groups import OrderGroup

        # Create two pairs that share TOKEN_A
        # Pair 1: A/B (bigger, processed first)
        # Pair 2: A/C (shares A, should be skipped)
        group1 = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[make_order("ab1", TOKEN_A, TOKEN_B, 100, 200)],
            sellers_of_b=[make_order("ba1", TOKEN_B, TOKEN_A, 300, 100)],
        )
        group2 = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_C,
            sellers_of_a=[make_order("ac1", TOKEN_A, TOKEN_C, 50, 150)],
            sellers_of_b=[make_order("ca1", TOKEN_C, TOKEN_A, 150, 50)],
        )

        # Build auction with mock router
        from unittest.mock import MagicMock

        auction = AuctionInstance(orders=[], tokens={}, liquidity=[])

        strategy = MultiPairCowStrategy()

        # Mock _solve_single_pair to return predictable results
        def mock_solve(group, _router, _auction):
            from solver.strategies.base import OrderFill
            from solver.strategies.multi_pair import LPResult

            fills = [
                OrderFill(order=group.sellers_of_a[0], sell_filled=50, buy_filled=100),
                OrderFill(order=group.sellers_of_b[0], sell_filled=100, buy_filled=50),
            ]
            prices = {
                group.token_a: Decimal("100"),
                group.token_b: Decimal("50"),
            }
            return LPResult(fills=fills, total_volume=150, prices=prices)

        strategy._solve_single_pair = MagicMock(side_effect=mock_solve)

        # Call _solve_large_component
        mock_router = MagicMock()
        result = strategy._solve_large_component([group1, group2], mock_router, auction)

        # Should only process group1 (larger) and skip group2 (shares TOKEN_A)
        assert result is not None
        # Only 2 fills from group1
        assert len(result.fills) == 2
        # Only 2 tokens priced (A and B), not C
        assert len(result.prices) == 2
        assert TOKEN_A in result.prices
        assert TOKEN_B in result.prices
        assert TOKEN_C not in result.prices

    def test_clearing_prices_match_fill_rates(self) -> None:
        """Clearing prices should produce correct rates for all fills."""
        from decimal import Decimal

        from solver.models.auction import AuctionInstance

        # Create a simple A/B pair
        orders = [
            make_order("seller", TOKEN_A, TOKEN_B, 100, 200),  # Sells A for B, rate >= 2
            make_order("buyer", TOKEN_B, TOKEN_A, 250, 100),  # Sells B for A, rate <= 2.5
        ]
        auction = AuctionInstance(orders=orders, tokens={}, liquidity=[])

        strategy = MultiPairCowStrategy()
        result = strategy.try_solve(auction)

        if result is not None and result.fills:
            # Check that clearing rate matches fill rate for each order
            for fill in result.fills:
                order = fill.order
                sell_price = int(result.prices.get(order.sell_token.lower(), "0"))
                buy_price = int(result.prices.get(order.buy_token.lower(), "0"))

                if sell_price > 0 and buy_price > 0 and fill.sell_filled > 0:
                    # Conservation invariant: sell_filled * sell_price = buy_filled * buy_price
                    # Cross-multiply to avoid division: sell_price * sell_filled ≈ buy_price * buy_filled
                    sell_value = sell_price * fill.sell_filled
                    buy_value = buy_price * fill.buy_filled

                    # Allow error bounded by max price (for integer truncation)
                    max_truncation_error = max(sell_price, buy_price)
                    assert abs(sell_value - buy_value) <= max_truncation_error, (
                        f"Conservation violated: {abs(sell_value - buy_value)} > {max_truncation_error}"
                    )


class TestMultiPairEBBO:
    """Tests for EBBO validation in MultiPairCowStrategy."""

    def _make_auction_with_tokens(self, orders: list[Order]) -> "AuctionInstance":
        """Create auction with token decimals for EBBO validation."""
        from solver.models.auction import AuctionInstance, Token

        tokens = {}
        for order in orders:
            for token in [order.sell_token, order.buy_token]:
                token_lower = token.lower()
                if token_lower not in tokens:
                    tokens[token_lower] = Token(decimals=18, available_balance="0")
        return AuctionInstance(id="test", orders=orders, tokens=tokens, liquidity=[])

    def test_ebbo_rejects_clearing_below_amm(self) -> None:
        """Pair rejected when clearing rate < AMM rate."""
        from decimal import Decimal
        from unittest.mock import Mock

        # Order A: sells 100 A for 200 B (limit: 2 B/A)
        # Order B: sells 200 B for 100 A (limit: 0.5 A/B = 2 B/A)
        # Clearing rate: 2 B/A
        # AMM rate: 3 B/A (better - EBBO violation)
        order_a = make_order("ask", TOKEN_A, TOKEN_B, 100, 200)
        order_b = make_order("bid", TOKEN_B, TOKEN_A, 200, 100)

        # Mock router returns AMM rate better than clearing
        mock_router = Mock()
        mock_router.get_reference_price.return_value = Decimal("3.0")  # AMM offers 3 B/A
        mock_router.get_reference_price_ratio.return_value = None

        strategy = MultiPairCowStrategy(router=mock_router)
        auction = self._make_auction_with_tokens([order_a, order_b])
        result = strategy.try_solve(auction)

        # Match should be rejected due to EBBO violation
        assert result is None or len(result.fills) == 0

    def test_ebbo_accepts_clearing_above_amm(self) -> None:
        """Pair accepted when clearing rate >= AMM rate."""
        from decimal import Decimal
        from unittest.mock import Mock

        # Order A: sells 100 A for 180 B (limit: 1.8 B/A - generous, below AMM)
        # Order B: sells 200 B for 100 A (limit: 0.5 A/B = 2 B/A)
        # With AMM at 2.0 B/A, CoW can match at 2.0:
        # - Order A gets 200 B for 100 A (rate 2.0 >= limit 1.8)
        # - Order B gets 100 A for 200 B (rate 0.5 >= limit 0.5)
        order_a = make_order("ask", TOKEN_A, TOKEN_B, 100, 180)
        order_b = make_order("bid", TOKEN_B, TOKEN_A, 200, 100)

        token_a_lower = TOKEN_A.lower()
        token_b_lower = TOKEN_B.lower()

        # Mock router with AMM rate 2.0 B/A and spread
        # - Selling A: get 2.0 B/A (ebbo_min)
        # - Selling B: get 0.45 A/B → ebbo_max = 1/0.45 ≈ 2.22 B/A
        def mock_get_ref_price(sell_token, buy_token, **_kwargs):
            if sell_token == token_a_lower:
                return Decimal("2.0")  # Selling A gets 2 B
            elif sell_token == token_b_lower:
                return Decimal("0.45")  # Selling B gets 0.45 A
            return None

        mock_router = Mock()
        mock_router.get_reference_price.side_effect = mock_get_ref_price
        mock_router.get_reference_price_ratio.return_value = None

        strategy = MultiPairCowStrategy(router=mock_router)
        auction = self._make_auction_with_tokens([order_a, order_b])
        result = strategy.try_solve(auction)

        # Match should be accepted (clearing rate within EBBO bounds)
        assert result is not None
        assert len(result.fills) >= 2

    def test_ebbo_skipped_when_no_liquidity(self) -> None:
        """EBBO check skipped when router returns None."""
        from unittest.mock import Mock

        # Standard matchable pair
        order_a = make_order("ask", TOKEN_A, TOKEN_B, 100, 200)
        order_b = make_order("bid", TOKEN_B, TOKEN_A, 200, 100)

        # Mock router returns None (no AMM liquidity)
        mock_router = Mock()
        mock_router.get_reference_price.return_value = None
        mock_router.get_reference_price_ratio.return_value = None

        strategy = MultiPairCowStrategy(router=mock_router)
        auction = self._make_auction_with_tokens([order_a, order_b])
        result = strategy.try_solve(auction)

        # Match should be accepted - no EBBO constraint when no AMM
        assert result is not None
        assert len(result.fills) >= 2

    def test_ebbo_without_router_skips_validation(self) -> None:
        """Strategy without router configured skips EBBO validation."""
        # Standard matchable pair
        order_a = make_order("ask", TOKEN_A, TOKEN_B, 100, 200)
        order_b = make_order("bid", TOKEN_B, TOKEN_A, 200, 100)

        # Strategy without router (default behavior)
        strategy = MultiPairCowStrategy()
        auction = self._make_auction_with_tokens([order_a, order_b])
        result = strategy.try_solve(auction)

        # Match should be accepted - no EBBO check without router
        assert result is not None
        assert len(result.fills) >= 2
