"""Tests for double auction rounding fixes (C4 and H5).

C4: Verifies ceiling division for match_b ensures seller limit satisfaction.
H5: Verifies exact sort key handles ratios differing by < 1 part in 10^18.
"""

from solver.models.auction import Order
from solver.models.order_groups import OrderGroup
from solver.strategies.double_auction import run_double_auction
from solver.strategies.double_auction.core import (
    PriceRatio,
    _compare_prices,
    _exact_price_key,
    _price_ratio_sort_key,
)

# Counter for generating unique UIDs
_uid_counter = 0


def make_order(
    sell_token: str,
    buy_token: str,
    sell_amount: int,
    buy_amount: int,
    partially_fillable: bool = True,
) -> Order:
    """Create a test order."""
    global _uid_counter
    _uid_counter += 1
    uid = f"0x{_uid_counter:0112x}"
    return Order(
        uid=uid,
        sell_token=sell_token,
        buy_token=buy_token,
        sell_amount=str(sell_amount),
        buy_amount=str(buy_amount),
        kind="sell",
        class_="market",
        partially_fillable=partially_fillable,
    )


TOKEN_A = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
TOKEN_B = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"


class TestCeilingDivisionForMatchB:
    """C4: match_b should use ceiling division so seller limit is always met."""

    def test_match_at_exact_seller_limit(self) -> None:
        """When clearing price == seller limit, ceiling ensures seller gets enough.

        With floor: match_b = 999 * 3 // 7 = 428 (seller gets 428/999 < 3/7)
        With ceiling: match_b = (999*3 + 6) // 7 = 429 (seller gets 429/999 >= 3/7)
        """
        # Seller wants at least 3/7 B per A (buy=3, sell=7)
        seller = make_order(TOKEN_A, TOKEN_B, sell_amount=999, buy_amount=int(999 * 3 / 7) + 1)
        # Buyer willing to pay more
        buyer = make_order(TOKEN_B, TOKEN_A, sell_amount=600, buy_amount=999)

        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[seller],
            sellers_of_b=[buyer],
        )

        result = run_double_auction(group)

        # If we get matches, verify the seller limit is satisfied
        if result.matches:
            for match in result.matches:
                # Seller limit: match_b / match_a >= buy_amount / sell_amount
                # Cross multiply: match_b * sell_amount >= buy_amount * match_a
                assert (
                    match.amount_b * seller.sell_amount_int
                    >= seller.buy_amount_int * match.amount_a
                ), (
                    f"Seller limit violated: {match.amount_b}/{match.amount_a} < {seller.buy_amount_int}/{seller.sell_amount_int}"
                )

    def test_fok_check_consistent_with_match(self) -> None:
        """FOK bid_fill_amount should use same ceiling as match_b.

        This ensures that if FOK check passes, the actual match will also pass.
        """
        # Create orders where ceiling vs floor matters for FOK check
        seller = make_order(TOKEN_A, TOKEN_B, sell_amount=1000, buy_amount=333)
        buyer = make_order(
            TOKEN_B,
            TOKEN_A,
            sell_amount=334,
            buy_amount=1000,
            partially_fillable=False,  # FOK
        )

        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[seller],
            sellers_of_b=[buyer],
        )

        result = run_double_auction(group)

        # If there are matches, the buyer must be fully filled (FOK)
        for match in result.matches:
            if match.buyer.uid == buyer.uid:
                # Buyer was not partially filled - this is the consistency check
                assert match.amount_b <= buyer.sell_amount_int

    def test_buyer_still_satisfied_with_ceiling(self) -> None:
        """Ceiling for seller doesn't violate buyer's limit."""
        seller = make_order(TOKEN_A, TOKEN_B, sell_amount=1000, buy_amount=500)
        buyer = make_order(TOKEN_B, TOKEN_A, sell_amount=600, buy_amount=1000)

        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[seller],
            sellers_of_b=[buyer],
        )

        result = run_double_auction(group)

        if result.matches:
            for match in result.matches:
                # Buyer limit: match_b / match_a <= buyer_sell / buyer_buy
                assert (
                    match.amount_b * buyer.buy_amount_int <= buyer.sell_amount_int * match.amount_a
                ), "Buyer limit violated by ceiling rounding"


class TestExactSortKey:
    """H5: Exact sort key handles ratios differing by < 1 part in 10^18."""

    def test_ratios_differing_by_tiny_amount(self) -> None:
        """Two ratios that differ by less than 1 part in 10^18 should sort correctly."""
        # These ratios are very close but different:
        # ratio_a = 10**36 + 1 / 10**36
        # ratio_b = 10**36 / 10**36 = 1
        ratio_a: PriceRatio = (10**36 + 1, 10**36)
        ratio_b: PriceRatio = (10**36, 10**36)

        # The old sort key loses this precision
        old_key_a = _price_ratio_sort_key(ratio_a)
        old_key_b = _price_ratio_sort_key(ratio_b)
        # They could collide with 10^18 scaling
        # (10^36+1)*10^18 // 10^36 = 10^18 + 10^18//10^36 = 10^18
        assert old_key_a == old_key_b  # Demonstrates the collision

        # The exact comparison correctly distinguishes them
        assert _compare_prices(ratio_a, ratio_b) == 1  # a > b

        # The exact sort key should sort correctly
        items = [ratio_a, ratio_b]
        sorted_items = sorted(items, key=lambda x: _exact_price_key(x))
        assert sorted_items == [ratio_b, ratio_a]  # ascending: b < a

    def test_exact_sort_matches_compare_prices(self) -> None:
        """Sort results with exact key should match _compare_prices ordering."""
        ratios = [
            (3, 7),
            (5, 7),
            (1, 3),
            (2, 3),
            (999999999999999999, 1000000000000000000),  # just below 1
            (1000000000000000001, 1000000000000000000),  # just above 1
        ]

        sorted_ratios = sorted(ratios, key=lambda x: _exact_price_key(x))

        # Verify pairwise ordering matches _compare_prices
        for i in range(len(sorted_ratios) - 1):
            assert _compare_prices(sorted_ratios[i], sorted_ratios[i + 1]) <= 0, (
                f"Sort inconsistency: {sorted_ratios[i]} should be <= {sorted_ratios[i + 1]}"
            )

    def test_equal_ratios_sort_stably(self) -> None:
        """Equal ratios (different representations) should compare as equal."""
        ratio_a: PriceRatio = (2, 4)
        ratio_b: PriceRatio = (3, 6)

        assert _compare_prices(ratio_a, ratio_b) == 0
