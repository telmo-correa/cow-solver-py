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
