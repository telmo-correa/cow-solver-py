"""Tests for hybrid CoW+AMM double auction."""

from decimal import Decimal

from solver.models.auction import Order
from solver.models.order_groups import OrderGroup
from solver.strategies.double_auction import (
    run_hybrid_auction,
)

# Counter for generating unique UIDs
_uid_counter = 0


def make_order(
    _name: str,  # Used for documentation in tests, not in the order itself
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
    # Generate a valid UID (0x + 112 hex chars)
    uid = f"0x{_uid_counter:0112x}"
    return Order(
        uid=uid,
        sell_token=sell_token,
        buy_token=buy_token,
        sell_amount=str(sell_amount),
        buy_amount=str(buy_amount),
        kind=kind,
        class_="market",  # Market orders don't need fee calculation
        partially_fillable=partially_fillable,
    )


# Tokens for testing (valid 40-char hex addresses)
TOKEN_A = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
TOKEN_B = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"


class TestHybridAuctionBasic:
    """Basic tests for hybrid CoW+AMM auction."""

    def test_no_amm_price_falls_back_to_pure_auction(self) -> None:
        """Without AMM price, hybrid auction behaves like pure auction."""
        # Ask: sell 100 A, want 200 B (price = 2 B/A)
        # Bid: sell 300 B, want 100 A (price = 3 B/A)
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 300, 100),
            ],
        )

        # No AMM price - should work like pure auction
        result = run_hybrid_auction(group, amm_price=None)

        # Should match at midpoint (2.5)
        # 100 A matched, 100 * 2.5 = 250 B matched
        # Bid has 300 - 250 = 50 B remaining -> routes to AMM
        assert len(result.cow_matches) == 1
        assert result.clearing_price == Decimal("2.5")
        assert result.total_cow_a == 100
        assert result.total_cow_b == 250
        # Bid remainder (50 B) routes to AMM
        assert len(result.amm_routes) == 1
        assert result.amm_routes[0].amount == 50
        assert result.amm_routes[0].is_selling_a is False

    def test_amm_price_unlocks_crossing_orders(self) -> None:
        """AMM price allows matching orders with crossed limit prices."""
        # Ask: sell 100 A, want 300 B (price = 3 B/A) - too expensive
        # Bid: sell 200 B, want 100 A (price = 2 B/A) - willing to pay less
        # These orders DON'T match directly (ask=3 > bid=2)
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 300),  # Wants 3 B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 200, 100),  # Offers 2 B/A
            ],
        )

        # Pure auction: no match (prices crossed)
        pure_result = run_hybrid_auction(group, amm_price=None)
        assert len(pure_result.cow_matches) == 0

        # Hybrid with AMM at 2.5 B/A:
        # - Ask wants 3, AMM offers 2.5 -> ask routes to AMM (worse than limit, will fail)
        # - Bid offers 2, AMM wants 2.5 -> bid routes to AMM
        # Actually: AMM price determines which orders CAN be satisfied
        # If AMM price is between the crossed prices, orders route to AMM
        hybrid_result = run_hybrid_auction(group, amm_price=Decimal("2.5"))

        # Both orders should route to AMM since they can't match each other
        assert len(hybrid_result.cow_matches) == 0
        assert len(hybrid_result.amm_routes) == 2

    def test_amm_price_between_ask_and_bid(self) -> None:
        """When AMM price is between ask and bid, CoW matching happens at AMM price."""
        # Ask: sell 100 A, want 200 B (price = 2 B/A)
        # Bid: sell 400 B, want 100 A (price = 4 B/A)
        # AMM price: 3 B/A (between 2 and 4)
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),  # Min 2 B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 400, 100),  # Max 4 B/A
            ],
        )

        result = run_hybrid_auction(group, amm_price=Decimal("3"))

        # Should match via CoW at AMM price (3 B/A), not midpoint
        assert len(result.cow_matches) == 1
        assert result.clearing_price == Decimal("3")
        assert result.cow_matches[0].amount_a == 100
        assert result.cow_matches[0].amount_b == 300  # 100 * 3

    def test_partial_fill_routes_remainder_to_amm(self) -> None:
        """Unmatched portion of orders should route to AMM."""
        # Ask: sell 100 A, want 200 B (price = 2 B/A)
        # Bid: sell 150 B, want 50 A (limit price = 150/50 = 3 B/A)
        # AMM price: 2.5 B/A
        #
        # At AMM price 2.5, bid can buy: 150 / 2.5 = 60 A
        # (The buy_amount is the MINIMUM the bid wants, but at a better price
        # they can buy more with their sell_amount)
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 150, 50),
            ],
        )

        result = run_hybrid_auction(group, amm_price=Decimal("2.5"))

        # At AMM price 2.5, bid can buy 150/2.5 = 60 A
        # So 60 A should match via CoW
        assert len(result.cow_matches) >= 1
        total_cow_a = sum(m.amount_a for m in result.cow_matches)
        assert total_cow_a == 60

        # Remaining 40 A (100 - 60) should route to AMM
        assert len(result.amm_routes) >= 1
        amm_a = sum(r.amount for r in result.amm_routes if r.is_selling_a)
        assert amm_a == 40


class TestHybridAuctionEdgeCases:
    """Edge cases for hybrid auction."""

    def test_empty_group(self) -> None:
        """Empty group returns empty result."""
        group = OrderGroup(token_a=TOKEN_A, token_b=TOKEN_B)
        result = run_hybrid_auction(group, amm_price=Decimal("2"))

        assert len(result.cow_matches) == 0
        assert len(result.amm_routes) == 0

    def test_single_direction_routes_to_amm(self) -> None:
        """Orders in only one direction all route to AMM."""
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),
                make_order("ask2", TOKEN_A, TOKEN_B, 50, 100),
            ],
            sellers_of_b=[],  # No bids
        )

        result = run_hybrid_auction(group, amm_price=Decimal("2.5"))

        # No CoW possible, all route to AMM
        assert len(result.cow_matches) == 0
        assert len(result.amm_routes) == 2


class TestHybridAuctionMultipleOrders:
    """Tests with multiple orders on each side."""

    def test_multiple_orders_sorted_by_price(self) -> None:
        """Multiple orders should be matched in price priority order."""
        # Asks (cheapest first):
        # ask1: 50 A @ 1.5 B/A
        # ask2: 50 A @ 2.0 B/A
        # ask3: 50 A @ 3.0 B/A (too expensive for AMM at 2.5)
        #
        # Bids (highest first):
        # bid1: 200 B @ 4.0 B/A (willing to pay up to 4)
        #
        # AMM price: 2.5 B/A
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 50, 75),  # 1.5 B/A
                make_order("ask2", TOKEN_A, TOKEN_B, 50, 100),  # 2.0 B/A
                make_order("ask3", TOKEN_A, TOKEN_B, 50, 150),  # 3.0 B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 200, 50),  # 4.0 B/A max
            ],
        )

        result = run_hybrid_auction(group, amm_price=Decimal("2.5"))

        # ask1 (1.5) and ask2 (2.0) should match via CoW at AMM price 2.5
        # ask3 (3.0) wants more than AMM offers, should route to AMM
        # (but AMM only gives 2.5, so ask3 might not fill)
        assert len(result.cow_matches) >= 1

        # Total CoW volume should be limited by bid capacity (200 B / 2.5 = 80 A max)
        total_cow_a = sum(m.amount_a for m in result.cow_matches)
        assert total_cow_a <= 100  # ask1 + ask2 capacity


class TestHybridAuctionValidation:
    """Tests for input validation and edge cases."""

    def test_zero_amm_price_falls_back_to_pure_auction(self) -> None:
        """Zero AMM price should fall back to pure auction."""
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),  # 2 B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 300, 100),  # 3 B/A
            ],
        )

        # Zero price should fall back to pure auction (midpoint matching)
        result = run_hybrid_auction(group, amm_price=Decimal("0"))

        # Should match like pure auction at midpoint (2.5)
        assert len(result.cow_matches) == 1
        assert result.clearing_price == Decimal("2.5")

    def test_negative_amm_price_falls_back_to_pure_auction(self) -> None:
        """Negative AMM price should fall back to pure auction."""
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 300, 100),
            ],
        )

        result = run_hybrid_auction(group, amm_price=Decimal("-1"))

        # Should fall back to pure auction
        assert len(result.cow_matches) == 1

    def test_limit_price_respected_after_truncation(self) -> None:
        """Integer truncation should not violate limit prices."""
        # Create a scenario where truncation could violate limits
        # Ask: sell 3 A, want 7 B (price = 7/3 = 2.333... B/A)
        # Bid: sell 10 B, want 4 A (price = 10/4 = 2.5 B/A)
        # AMM price: 2.34 B/A (barely above ask's limit)
        #
        # At AMM price 2.34:
        # match_a could be 3, match_b = int(3 * 2.34) = int(7.02) = 7
        # actual_price = 7/3 = 2.333... which is < 2.34 but >= ask_limit (2.333...)
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 3, 7),  # 2.333... B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 10, 4),  # 2.5 B/A max
            ],
        )

        result = run_hybrid_auction(group, amm_price=Decimal("2.34"))

        # If there's a match, verify limits are satisfied
        for match in result.cow_matches:
            actual_price = Decimal(match.amount_b) / Decimal(match.amount_a)
            # Ask wanted at least 7/3 = 2.333... B/A
            assert actual_price >= Decimal("7") / Decimal("3")
            # Bid offered at most 10/4 = 2.5 B/A
            assert actual_price <= Decimal("10") / Decimal("4")


class TestHybridAuctionEBBOBounds:
    """Tests for EBBO bounds (ebbo_min, ebbo_max) with hybrid auction."""

    def test_ebbo_min_rejects_amm_price_below_floor(self) -> None:
        """AMM price below ebbo_min should result in no CoW match.

        When the AMM price is below the EBBO minimum (what sellers must get),
        no EBBO-compliant CoW match is possible.
        """
        # Ask: sell 100 A, want 200 B (limit = 2 B/A)
        # Bid: sell 400 B, want 100 A (limit = 4 B/A)
        # AMM price: 2.5 B/A (matches can happen)
        # But ebbo_min = 3.0 means sellers must get at least 3 B/A
        # Since AMM price (2.5) < ebbo_min (3.0), no match possible
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 400, 100),
            ],
        )

        # Without EBBO bounds, should match at AMM price
        result_no_ebbo = run_hybrid_auction(group, amm_price=Decimal("2.5"))
        assert len(result_no_ebbo.cow_matches) == 1

        # With ebbo_min = 3.0 > AMM price (2.5), no CoW match
        result_with_ebbo = run_hybrid_auction(
            group, amm_price=Decimal("2.5"), ebbo_min=Decimal("3.0")
        )
        assert len(result_with_ebbo.cow_matches) == 0
        # All orders should route to AMM
        assert len(result_with_ebbo.amm_routes) == 2

    def test_ebbo_max_rejects_amm_price_above_ceiling(self) -> None:
        """AMM price above ebbo_max should result in no CoW match.

        When the AMM price is above the EBBO maximum (what buyers can pay),
        no EBBO-compliant CoW match is possible.
        """
        # Ask: sell 100 A, want 200 B (limit = 2 B/A)
        # Bid: sell 400 B, want 100 A (limit = 4 B/A)
        # AMM price: 3.5 B/A (matches can happen)
        # But ebbo_max = 3.0 means buyers must pay at most 3 B/A
        # Since AMM price (3.5) > ebbo_max (3.0), no match possible
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 400, 100),
            ],
        )

        # Without EBBO bounds, should match at AMM price
        result_no_ebbo = run_hybrid_auction(group, amm_price=Decimal("3.5"))
        assert len(result_no_ebbo.cow_matches) == 1

        # With ebbo_max = 3.0 < AMM price (3.5), no CoW match
        result_with_ebbo = run_hybrid_auction(
            group, amm_price=Decimal("3.5"), ebbo_max=Decimal("3.0")
        )
        assert len(result_with_ebbo.cow_matches) == 0
        assert len(result_with_ebbo.amm_routes) == 2

    def test_ebbo_bounds_with_no_amm_price(self) -> None:
        """EBBO bounds should work with pure double auction (no AMM price).

        When AMM price is None, hybrid auction falls back to pure double auction,
        which should still respect EBBO bounds.
        """
        # Ask: sell 100 A, want 150 B (limit = 1.5 B/A)
        # Bid: sell 300 B, want 100 A (limit = 3 B/A)
        # Pure auction range: [1.5, 3.0], midpoint = 2.25
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 150),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 300, 100),
            ],
        )

        # Without EBBO bounds, matches at midpoint
        result_no_ebbo = run_hybrid_auction(group, amm_price=None)
        assert result_no_ebbo.clearing_price == Decimal("2.25")

        # With ebbo_min = 2.5, range becomes [2.5, 3.0]
        result_with_ebbo = run_hybrid_auction(group, amm_price=None, ebbo_min=Decimal("2.5"))
        assert len(result_with_ebbo.cow_matches) == 1
        # Midpoint of [2.5, 3.0] = 2.75
        assert result_with_ebbo.clearing_price == Decimal("2.75")

    def test_ebbo_both_bounds_within_valid_range(self) -> None:
        """Both EBBO bounds should constrain the AMM price validation."""
        # Ask: sell 100 A, want 200 B (limit = 2 B/A)
        # Bid: sell 400 B, want 100 A (limit = 4 B/A)
        # AMM price: 3.0 B/A
        # ebbo_min = 2.5, ebbo_max = 3.5 - AMM price (3.0) is within bounds
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 400, 100),
            ],
        )

        result = run_hybrid_auction(
            group,
            amm_price=Decimal("3.0"),
            ebbo_min=Decimal("2.5"),
            ebbo_max=Decimal("3.5"),
        )

        # AMM price is within EBBO bounds, should match
        assert len(result.cow_matches) == 1
        assert result.clearing_price == Decimal("3.0")

    def test_ebbo_bounds_crossed_no_valid_range(self) -> None:
        """When ebbo_min > ebbo_max, no match is possible."""
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 400, 100),
            ],
        )

        # ebbo_min (3.5) > ebbo_max (2.5) - no valid range
        result = run_hybrid_auction(
            group,
            amm_price=Decimal("3.0"),
            ebbo_min=Decimal("3.5"),
            ebbo_max=Decimal("2.5"),
        )

        # No match possible when bounds are crossed
        assert len(result.cow_matches) == 0

    def test_multiple_orders_with_ebbo_filtering(self) -> None:
        """EBBO bounds should filter appropriately with multiple orders."""
        # Multiple asks and bids
        # AMM price needs to satisfy ebbo_min for ALL sellers
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 50, 100),  # 2 B/A
                make_order("ask2", TOKEN_A, TOKEN_B, 50, 125),  # 2.5 B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 300, 75),  # 4 B/A
            ],
        )

        # AMM price = 3.0 is fine for both asks
        result_no_ebbo = run_hybrid_auction(group, amm_price=Decimal("3.0"))
        assert len(result_no_ebbo.cow_matches) >= 1

        # ebbo_min = 3.5 - now AMM price (3.0) < ebbo_min, no CoW match
        result_with_ebbo = run_hybrid_auction(
            group, amm_price=Decimal("3.0"), ebbo_min=Decimal("3.5")
        )
        assert len(result_with_ebbo.cow_matches) == 0
