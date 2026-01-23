"""Tests for order grouping utilities."""

from solver.models.auction import Order
from solver.models.order_groups import find_cow_opportunities, group_orders_by_pair


def make_order(
    sell_token: str,
    buy_token: str,
    sell_amount: str = "1000000000000000000",
    buy_amount: str = "1000000000000000000",
    uid_suffix: str = "01",
) -> Order:
    """Create a test order with minimal required fields."""
    return Order(
        uid="0x" + uid_suffix * 56,
        sellToken=sell_token,
        buyToken=buy_token,
        sellAmount=sell_amount,
        buyAmount=buy_amount,
        kind="sell",
        **{"class": "market"},  # Use dict expansion for reserved keyword
        partiallyFillable=False,
        validTo=9999999999,
        appData="0x" + "00" * 32,
        receiver="0x" + "00" * 20,
    )


# Token addresses for testing
TOKEN_A = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
TOKEN_B = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
TOKEN_C = "0xcccccccccccccccccccccccccccccccccccccccc"


class TestGroupOrdersByPair:
    """Tests for group_orders_by_pair function."""

    def test_empty_orders(self) -> None:
        groups = group_orders_by_pair([])
        assert len(groups) == 0

    def test_single_order(self) -> None:
        order = make_order(TOKEN_A, TOKEN_B)
        groups = group_orders_by_pair([order])

        assert len(groups) == 1
        # Canonical ordering: TOKEN_A < TOKEN_B
        pair = (TOKEN_A.lower(), TOKEN_B.lower())
        assert pair in groups

        group = groups[pair]
        assert len(group.sellers_of_a) == 1
        assert len(group.sellers_of_b) == 0
        assert group.sellers_of_a[0] == order

    def test_two_orders_same_direction(self) -> None:
        order1 = make_order(TOKEN_A, TOKEN_B, uid_suffix="01")
        order2 = make_order(TOKEN_A, TOKEN_B, uid_suffix="02")
        groups = group_orders_by_pair([order1, order2])

        assert len(groups) == 1
        pair = (TOKEN_A.lower(), TOKEN_B.lower())
        group = groups[pair]

        assert len(group.sellers_of_a) == 2
        assert len(group.sellers_of_b) == 0
        assert order1 in group.sellers_of_a
        assert order2 in group.sellers_of_a

    def test_two_orders_opposite_directions(self) -> None:
        order_a_to_b = make_order(TOKEN_A, TOKEN_B, uid_suffix="01")
        order_b_to_a = make_order(TOKEN_B, TOKEN_A, uid_suffix="02")
        groups = group_orders_by_pair([order_a_to_b, order_b_to_a])

        assert len(groups) == 1
        pair = (TOKEN_A.lower(), TOKEN_B.lower())
        group = groups[pair]

        assert len(group.sellers_of_a) == 1
        assert len(group.sellers_of_b) == 1
        assert group.sellers_of_a[0] == order_a_to_b
        assert group.sellers_of_b[0] == order_b_to_a

    def test_multiple_pairs(self) -> None:
        order_ab = make_order(TOKEN_A, TOKEN_B, uid_suffix="01")
        order_ac = make_order(TOKEN_A, TOKEN_C, uid_suffix="02")
        order_bc = make_order(TOKEN_B, TOKEN_C, uid_suffix="03")
        groups = group_orders_by_pair([order_ab, order_ac, order_bc])

        assert len(groups) == 3

    def test_canonical_ordering_consistency(self) -> None:
        """Orders with reversed token pairs should be in the same group."""
        # TOKEN_B > TOKEN_A lexicographically
        order_b_to_a = make_order(TOKEN_B, TOKEN_A, uid_suffix="01")
        order_a_to_b = make_order(TOKEN_A, TOKEN_B, uid_suffix="02")
        groups = group_orders_by_pair([order_b_to_a, order_a_to_b])

        # Should be single group with canonical ordering
        assert len(groups) == 1
        pair = (TOKEN_A.lower(), TOKEN_B.lower())  # Canonical: smaller first
        assert pair in groups


class TestOrderGroup:
    """Tests for OrderGroup dataclass."""

    def test_has_cow_potential_true(self) -> None:
        order_a = make_order(TOKEN_A, TOKEN_B)
        order_b = make_order(TOKEN_B, TOKEN_A)
        groups = group_orders_by_pair([order_a, order_b])

        pair = (TOKEN_A.lower(), TOKEN_B.lower())
        assert groups[pair].has_cow_potential is True

    def test_has_cow_potential_false_single_direction(self) -> None:
        order = make_order(TOKEN_A, TOKEN_B)
        groups = group_orders_by_pair([order])

        pair = (TOKEN_A.lower(), TOKEN_B.lower())
        assert groups[pair].has_cow_potential is False

    def test_total_sell_a(self) -> None:
        order1 = make_order(TOKEN_A, TOKEN_B, sell_amount="100", uid_suffix="01")
        order2 = make_order(TOKEN_A, TOKEN_B, sell_amount="200", uid_suffix="02")
        groups = group_orders_by_pair([order1, order2])

        pair = (TOKEN_A.lower(), TOKEN_B.lower())
        assert groups[pair].total_sell_a == 300

    def test_total_sell_b(self) -> None:
        order1 = make_order(TOKEN_B, TOKEN_A, sell_amount="150", uid_suffix="01")
        order2 = make_order(TOKEN_B, TOKEN_A, sell_amount="250", uid_suffix="02")
        groups = group_orders_by_pair([order1, order2])

        pair = (TOKEN_A.lower(), TOKEN_B.lower())
        assert groups[pair].total_sell_b == 400

    def test_total_buy_amounts(self) -> None:
        order_a_to_b = make_order(TOKEN_A, TOKEN_B, buy_amount="500", uid_suffix="01")
        order_b_to_a = make_order(TOKEN_B, TOKEN_A, buy_amount="600", uid_suffix="02")
        groups = group_orders_by_pair([order_a_to_b, order_b_to_a])

        pair = (TOKEN_A.lower(), TOKEN_B.lower())
        group = groups[pair]

        # Sellers of A buy B
        assert group.total_buy_b == 500
        # Sellers of B buy A
        assert group.total_buy_a == 600

    def test_order_count(self) -> None:
        orders = [
            make_order(TOKEN_A, TOKEN_B, uid_suffix="01"),
            make_order(TOKEN_A, TOKEN_B, uid_suffix="02"),
            make_order(TOKEN_B, TOKEN_A, uid_suffix="03"),
        ]
        groups = group_orders_by_pair(orders)

        pair = (TOKEN_A.lower(), TOKEN_B.lower())
        assert groups[pair].order_count == 3

    def test_all_orders(self) -> None:
        order1 = make_order(TOKEN_A, TOKEN_B, uid_suffix="01")
        order2 = make_order(TOKEN_B, TOKEN_A, uid_suffix="02")
        groups = group_orders_by_pair([order1, order2])

        pair = (TOKEN_A.lower(), TOKEN_B.lower())
        all_orders = groups[pair].all_orders

        assert len(all_orders) == 2
        assert order1 in all_orders
        assert order2 in all_orders


class TestFindCowOpportunities:
    """Tests for find_cow_opportunities function."""

    def test_empty_orders(self) -> None:
        opportunities = find_cow_opportunities([])
        assert len(opportunities) == 0

    def test_no_cow_potential(self) -> None:
        orders = [
            make_order(TOKEN_A, TOKEN_B, uid_suffix="01"),
            make_order(TOKEN_A, TOKEN_C, uid_suffix="02"),
        ]
        opportunities = find_cow_opportunities(orders)
        assert len(opportunities) == 0

    def test_single_cow_opportunity(self) -> None:
        orders = [
            make_order(TOKEN_A, TOKEN_B, uid_suffix="01"),
            make_order(TOKEN_B, TOKEN_A, uid_suffix="02"),
        ]
        opportunities = find_cow_opportunities(orders)

        assert len(opportunities) == 1
        assert opportunities[0].has_cow_potential is True

    def test_multiple_cow_opportunities_sorted_by_count(self) -> None:
        orders = [
            # A/B pair: 3 orders
            make_order(TOKEN_A, TOKEN_B, uid_suffix="01"),
            make_order(TOKEN_A, TOKEN_B, uid_suffix="02"),
            make_order(TOKEN_B, TOKEN_A, uid_suffix="03"),
            # A/C pair: 2 orders
            make_order(TOKEN_A, TOKEN_C, uid_suffix="04"),
            make_order(TOKEN_C, TOKEN_A, uid_suffix="05"),
        ]
        opportunities = find_cow_opportunities(orders)

        assert len(opportunities) == 2
        # Sorted by order count descending
        assert opportunities[0].order_count == 3
        assert opportunities[1].order_count == 2

    def test_filters_out_non_cow_groups(self) -> None:
        orders = [
            # A/B: Has CoW potential
            make_order(TOKEN_A, TOKEN_B, uid_suffix="01"),
            make_order(TOKEN_B, TOKEN_A, uid_suffix="02"),
            # A/C: No CoW potential (single direction)
            make_order(TOKEN_A, TOKEN_C, uid_suffix="03"),
        ]
        opportunities = find_cow_opportunities(orders)

        assert len(opportunities) == 1
        assert opportunities[0].token_a.lower() == TOKEN_A.lower()
        assert opportunities[0].token_b.lower() == TOKEN_B.lower()


class TestPropertyInvariants:
    """Test invariants that should always hold."""

    def test_all_orders_appear_exactly_once(self) -> None:
        """Every order should appear in exactly one group."""
        orders = [
            make_order(TOKEN_A, TOKEN_B, uid_suffix="01"),
            make_order(TOKEN_B, TOKEN_A, uid_suffix="02"),
            make_order(TOKEN_A, TOKEN_C, uid_suffix="03"),
            make_order(TOKEN_B, TOKEN_C, uid_suffix="04"),
        ]
        groups = group_orders_by_pair(orders)

        # Collect all orders from groups
        all_grouped_orders = []
        for group in groups.values():
            all_grouped_orders.extend(group.all_orders)

        # Each order should appear exactly once
        assert len(all_grouped_orders) == len(orders)
        for order in orders:
            assert order in all_grouped_orders

    def test_canonical_ordering_is_consistent(self) -> None:
        """Token pairs should always have token_a < token_b."""
        orders = [
            make_order(TOKEN_B, TOKEN_A),  # Reversed
            make_order(TOKEN_C, TOKEN_A),  # Reversed
            make_order(TOKEN_A, TOKEN_B),  # Normal
        ]
        groups = group_orders_by_pair(orders)

        for (token_a, token_b), _group in groups.items():
            assert token_a < token_b, f"Canonical ordering violated: {token_a} >= {token_b}"
