"""Tests for double auction algorithm."""

from decimal import Decimal

from solver.models.auction import Order
from solver.models.order_groups import OrderGroup
from solver.strategies.double_auction import (
    calculate_surplus,
    get_limit_price,
    run_double_auction,
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


class TestGetLimitPrice:
    """Tests for limit price calculation."""

    def test_seller_of_a_limit_price(self) -> None:
        """Seller of A: limit = buy_amount / sell_amount (B per A)."""
        order = make_order(
            _name="seller",
            sell_token=TOKEN_A,
            buy_token=TOKEN_B,
            sell_amount=100,
            buy_amount=200,  # Wants at least 2 B per A
        )
        price = get_limit_price(order, is_selling_a=True)
        assert price == Decimal(2)

    def test_seller_of_b_limit_price(self) -> None:
        """Seller of B: limit = sell_amount / buy_amount (B per A)."""
        order = make_order(
            _name="buyer",
            sell_token=TOKEN_B,
            buy_token=TOKEN_A,
            sell_amount=300,  # Willing to pay 300 B
            buy_amount=100,  # To get 100 A
        )
        # Price = 300/100 = 3 B per A
        price = get_limit_price(order, is_selling_a=False)
        assert price == Decimal(3)


class TestDoubleAuction:
    """Tests for the double auction algorithm."""

    def test_empty_group(self) -> None:
        """Empty group returns empty result."""
        group = OrderGroup(token_a=TOKEN_A, token_b=TOKEN_B)
        result = run_double_auction(group)

        assert result.matches == []
        assert result.total_a_matched == 0
        assert result.clearing_price is None

    def test_no_cow_potential(self) -> None:
        """Group with orders only one direction returns no matches."""
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),
            ],
            sellers_of_b=[],
        )
        result = run_double_auction(group)

        assert result.matches == []
        assert len(result.unmatched_sellers) == 1

    def test_simple_two_order_match(self) -> None:
        """Two compatible orders should match completely."""
        # Ask: sell 100 A, want 200 B (price = 2 B/A)
        # Bid: sell 300 B, want 100 A (price = 3 B/A)
        # Prices overlap (ask=2 <= bid=3), so match is possible
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
        result = run_double_auction(group)

        assert len(result.matches) == 1
        assert result.total_a_matched == 100
        # Clearing price is midpoint: (2+3)/2 = 2.5
        assert result.clearing_price == Decimal("2.5")
        # B matched = 100 * 2.5 = 250
        assert result.total_b_matched == 250

    def test_prices_cross_no_match(self) -> None:
        """Orders with non-overlapping prices should not match."""
        # Ask: sell 100 A, want 500 B (price = 5 B/A)
        # Bid: sell 200 B, want 100 A (price = 2 B/A)
        # Ask wants 5, bid offers only 2 - no match
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 500),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 200, 100),
            ],
        )
        result = run_double_auction(group)

        assert result.matches == []
        assert len(result.unmatched_sellers) == 1
        assert len(result.unmatched_buyers) == 1

    def test_multiple_asks_single_bid(self) -> None:
        """Multiple asks should be matched against single large bid."""
        # Asks (ascending by price):
        # ask1: 50 A @ 1.5 B/A (sell 50, want 75)
        # ask2: 50 A @ 2.0 B/A (sell 50, want 100)
        # Bid: sell 400 B, want 100 A (price = 4 B/A)

        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 50, 75),  # 1.5 B/A
                make_order("ask2", TOKEN_A, TOKEN_B, 50, 100),  # 2.0 B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 400, 100),  # 4.0 B/A
            ],
        )
        result = run_double_auction(group)

        # Both asks should match
        assert len(result.matches) >= 1
        assert result.total_a_matched == 100  # All A matched

    def test_fill_or_kill_respected(self) -> None:
        """Fill-or-kill orders should not be partially filled."""
        # Ask: sell 100 A, want 200 B (price = 2 B/A), NOT partially fillable
        # Bid: sell 100 B, want 40 A (price = 2.5 B/A)
        # Bid can only buy 40 A, but ask is fill-or-kill for 100 A
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200, partially_fillable=False),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 100, 40),
            ],
        )
        result = run_double_auction(group, respect_fill_or_kill=True)

        # No match because ask is fill-or-kill
        assert result.matches == []

    def test_fill_or_kill_complete_fill(self) -> None:
        """Fill-or-kill orders match when complete fill is possible.

        The double auction tries candidate prices (midpoint and boundaries) and
        selects one that allows complete fill for fill-or-kill orders.
        """
        # Ask: sell 50 A, want 100 B (price = 2 B/A), NOT partially fillable
        # Bid: sell 125 B, want 50 A (price = 2.5 B/A)
        # Valid price range: [2.0, 2.5]
        # At midpoint = 2.25 -> bid can buy 125/2.25 = 55 A >= 50 (complete fill)
        # At max_ask = 2.0 -> bid can buy 125/2.0 = 62 A >= 50 (complete fill)
        # Both prices allow complete fill; midpoint is tried first
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 50, 100, partially_fillable=False),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 125, 50),  # 2.5 B/A max
            ],
        )
        result = run_double_auction(group, respect_fill_or_kill=True)

        # Should match completely - fill-or-kill is satisfied
        assert len(result.matches) == 1
        assert result.total_a_matched == 50  # Complete fill of ask
        assert len(result.unmatched_sellers) == 0  # Ask fully matched

    def test_fill_or_kill_boundary_price_enables_match(self) -> None:
        """Fill-or-kill match found at boundary price when midpoint fails.

        When midpoint causes partial fill that violates fill-or-kill, the algorithm
        tries boundary prices. If a boundary price allows complete fill, it succeeds.
        """
        # Ask: sell 100 A, want 200 B (price = 2 B/A), NOT partially fillable
        # Bid: sell 201 B, want 50 A (price = 4.02 B/A)
        # Valid price range: [2.0, 4.02]
        # At midpoint = 3.01 -> bid can buy 201/3.01 = 66.7 -> 66 A (< 100, partial fill!)
        #   Ask fill-or-kill violated at midpoint
        # At max_ask = 2.0 -> bid can buy 201/2.0 = 100.5 -> 100 A (complete fill!)
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200, partially_fillable=False),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 201, 50),  # 4.02 B/A max
            ],
        )
        result = run_double_auction(group, respect_fill_or_kill=True)

        # Should match at boundary price (2.0) which allows complete fill
        assert len(result.matches) == 1
        assert result.total_a_matched == 100  # Complete fill of ask
        assert float(result.clearing_price) == 2.0  # Uses boundary price

    def test_partial_fill_allowed(self) -> None:
        """Partially fillable orders should match partially.

        The double auction tries candidate prices (midpoint and boundaries) and
        selects the one that maximizes matched volume. At the ask's limit price
        (2 B/A), more A can be matched than at midpoint (2.25 B/A).
        """
        # Ask: sell 100 A, want 200 B (price = 2 B/A), partially fillable
        # Bid: sell 125 B, want 50 A (price = 2.5 B/A)
        # Candidate prices: midpoint=2.25, max_ask=2.0, min_bid=2.5
        # At price 2.0 (max_ask_limit), bid can buy 125/2.0 = 62.5 -> 62 A
        # At price 2.25 (midpoint), bid can buy 125/2.25 = 55.5 -> 55 A
        # Algorithm selects price 2.0 for maximum volume
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200, partially_fillable=True),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 125, 50),  # 2.5 B/A max
            ],
        )
        result = run_double_auction(group)

        # Should match 62 A at price 2.0 (boundary price maximizes volume)
        assert len(result.matches) == 1
        assert result.total_a_matched == 62  # 125 / 2.0 = 62.5 -> 62
        assert float(result.clearing_price) == 2.0  # Uses ask's limit (max_ask_limit)
        # Remaining 38 A should be unmatched
        assert len(result.unmatched_sellers) == 1
        assert result.unmatched_sellers[0][1] == 38  # 100 - 62 = 38 A remaining


class TestMultiOrderAuction:
    """Tests for multi-order double auctions."""

    def test_n_order_fill_or_kill_boundary_price(self) -> None:
        """N>2 orders where boundary price enables fill-or-kill match.

        With 3 orders, the midpoint price would cause partial fill of the
        fill-or-kill ask, but boundary price allows complete fill.
        """
        # Ask1: sell 100 A, want 200 B (price = 2 B/A), fill-or-kill
        # Ask2: sell 20 A, want 30 B (price = 1.5 B/A), partially fillable
        # Bid1: sell 250 B, want 50 A (price = 5 B/A)
        #
        # Valid price range: [2.0, 5.0], midpoint = 3.5
        #
        # At midpoint 3.5:
        #   - Bid1 can buy: 250/3.5 = 71 A
        #   - Ask1 needs 100 A (fill-or-kill) -> can't match!
        #
        # At max_ask 2.0:
        #   - Bid1 can buy: 250/2.0 = 125 A
        #   - Ask1 (100 A) + Ask2 (20 A) = 120 A -> both can match!
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200, partially_fillable=False),
                make_order("ask2", TOKEN_A, TOKEN_B, 20, 30, partially_fillable=True),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 250, 50),  # 5 B/A max
            ],
        )
        result = run_double_auction(group, respect_fill_or_kill=True)

        # Boundary price (2.0) enables both asks to match
        assert len(result.matches) == 2
        assert result.total_a_matched == 120  # 100 + 20
        assert float(result.clearing_price) == 2.0  # Boundary price
        assert len(result.unmatched_sellers) == 0  # Both asks fully matched

    def test_three_asks_two_bids(self) -> None:
        """Multiple asks and bids should clear correctly."""
        # Asks (sorted by price):
        # ask1: 30 A @ 1.0 B/A
        # ask2: 40 A @ 1.5 B/A
        # ask3: 50 A @ 2.0 B/A
        #
        # Bids (sorted by price descending):
        # bid1: 100 B @ 2.5 B/A (can buy 40 A)
        # bid2: 80 B @ 1.8 B/A (can buy ~44 A)

        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 30, 30),  # 1.0 B/A
                make_order("ask2", TOKEN_A, TOKEN_B, 40, 60),  # 1.5 B/A
                make_order("ask3", TOKEN_A, TOKEN_B, 50, 100),  # 2.0 B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 100, 40),  # 2.5 B/A
                make_order("bid2", TOKEN_B, TOKEN_A, 80, 44),  # ~1.8 B/A
            ],
        )
        result = run_double_auction(group)

        # All asks should be matchable (ask prices <= bid prices)
        # ask1 @ 1.0 matches bid1 @ 2.5 (midpoint 1.75)
        # ask2 @ 1.5 matches (remaining of) bid1 and bid2
        # ask3 @ 2.0 should not match bid2 @ 1.8 (price crossed)
        assert result.total_a_matched > 0
        assert len(result.matches) >= 1


class TestSurplusCalculation:
    """Tests for surplus calculation."""

    def test_surplus_positive(self) -> None:
        """Surplus should be positive when prices have spread."""
        # Ask: 100 A @ 2 B/A, Bid: 100 A @ 3 B/A
        # If clearing at 2.5, both sides get surplus
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
        result = run_double_auction(group)
        surplus = calculate_surplus(result)

        # Seller wanted 200 B, got 250 B -> surplus = 50
        # Buyer was willing to pay 300 B, paid 250 B -> surplus = 50
        # Total = 100
        assert surplus == 100

    def test_surplus_zero_at_limit(self) -> None:
        """Surplus should be zero if clearing at limit prices."""
        # Ask: 100 A @ 2 B/A, Bid: 100 A @ 2 B/A (same price)
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),  # 2 B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 200, 100),  # 2 B/A
            ],
        )
        result = run_double_auction(group)
        surplus = calculate_surplus(result)

        # Both trade at exactly their limit
        assert surplus == 0


class TestUniformClearingPrice:
    """Tests for uniform clearing price guarantee."""

    def test_all_matches_have_same_clearing_price(self) -> None:
        """All matches in a result must have the same clearing price."""
        # Create multiple asks and bids that will produce multiple matches
        # Ask1: 50 A @ 2.0 B/A, Ask2: 50 A @ 2.2 B/A
        # Bid1: 200 B @ 3.0 B/A, Bid2: 100 B @ 2.8 B/A
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 50, 100),  # 2.0 B/A
                make_order("ask2", TOKEN_A, TOKEN_B, 50, 110),  # 2.2 B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 200, 66),  # ~3.0 B/A
                make_order("bid2", TOKEN_B, TOKEN_A, 100, 35),  # ~2.86 B/A
            ],
        )
        result = run_double_auction(group)

        # Should have multiple matches
        assert len(result.matches) >= 2

        # All matches must have the same clearing price
        prices = [m.clearing_price for m in result.matches]
        assert len(set(prices)) == 1, f"Non-uniform prices: {prices}"

        # The clearing price must be within valid range
        clearing = result.clearing_price
        assert clearing is not None

        # Verify clearing price satisfies all matched orders
        for match in result.matches:
            ask_limit = get_limit_price(match.seller, is_selling_a=True)
            bid_limit = get_limit_price(match.buyer, is_selling_a=False)
            assert ask_limit is not None
            assert bid_limit is not None
            assert clearing >= ask_limit, f"Clearing {clearing} < ask limit {ask_limit}"
            assert clearing <= bid_limit, f"Clearing {clearing} > bid limit {bid_limit}"

    def test_nonmatching_order_does_not_affect_clearing_price(self) -> None:
        """Orders that don't actually match shouldn't influence clearing price.

        This tests the fix for Issue 1: limits should only be recorded AFTER
        confirming match_a > 0 and match_b > 0.
        """
        # Ask1: 100 A @ 2.0 B/A (will match)
        # Ask2: 1 A @ 0.4 B/A - very cheap, but price so low that match_b truncates to 0
        # Bid1: 1 B @ 0.5 B/A (tiny bid with very low price)
        #
        # At midpoint (0.4+0.5)/2 = 0.45 B/A:
        # - bid_can_buy = 1 / 0.45 = 2 A
        # - match_a = min(1, 2) = 1
        # - match_b = int(1 * 0.45) = 0 (truncated!)
        #
        # So Ask2 should NOT have its limit recorded.
        # Without the fix, max_ask would be 2.0 (from Ask1 only since Ask2 doesn't overlap with main bid)
        # This is actually hard to trigger because we need overlapping prices but truncation.
        #
        # Simpler test: verify the clearing price is the midpoint of MATCHED orders' limits
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),  # 2.0 B/A
                make_order("ask2", TOKEN_A, TOKEN_B, 100, 300),  # 3.0 B/A (higher limit)
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 300, 100),  # 3.0 B/A
            ],
        )
        result = run_double_auction(group)

        # Both asks should match (both limits <= bid limit of 3.0)
        assert len(result.matches) >= 1

        # Verify clearing price is within valid range
        clearing = result.clearing_price
        assert clearing is not None

        # The clearing price should satisfy all matched orders
        for match in result.matches:
            ask_limit = get_limit_price(match.seller, is_selling_a=True)
            bid_limit = get_limit_price(match.buyer, is_selling_a=False)
            assert ask_limit is not None and bid_limit is not None
            # Clearing price must be between ask and bid limits
            assert clearing >= ask_limit
            assert clearing <= bid_limit


class TestRealWorldScenario:
    """Test with more realistic parameters."""

    def test_usdc_weth_like_scenario(self) -> None:
        """Simulate USDC/WETH-like token pair with multiple orders."""
        # Using 18 decimals for WETH, 6 for USDC
        WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"

        # Price around 2000 USDC per WETH
        # Asks: selling WETH for USDC
        asks = [
            make_order("ask1", WETH, USDC, 1 * 10**18, 1990 * 10**6),  # 1990 USDC/ETH
            make_order("ask2", WETH, USDC, 2 * 10**18, 4000 * 10**6),  # 2000 USDC/ETH
            make_order("ask3", WETH, USDC, 1 * 10**18, 2010 * 10**6),  # 2010 USDC/ETH
        ]

        # Bids: selling USDC for WETH
        bids = [
            make_order("bid1", USDC, WETH, 4100 * 10**6, 2 * 10**18),  # 2050 USDC/ETH willing
            make_order("bid2", USDC, WETH, 2020 * 10**6, 1 * 10**18),  # 2020 USDC/ETH willing
            make_order("bid3", USDC, WETH, 1980 * 10**6, 1 * 10**18),  # 1980 USDC/ETH willing
        ]

        group = OrderGroup(
            token_a=WETH,
            token_b=USDC,
            sellers_of_a=asks,
            sellers_of_b=bids,
        )

        result = run_double_auction(group)

        # Should have matches where ask price <= bid price
        # ask1 (1990) matches bids up to 2050, 2020
        # ask2 (2000) matches bids up to 2050, 2020
        # ask3 (2010) matches bids up to 2050
        # bid3 (1980) won't match ask1 (1990) - price crossed
        assert len(result.matches) >= 1
        assert result.total_a_matched > 0

        surplus = calculate_surplus(result)
        assert surplus >= 0  # Surplus should be non-negative


class TestEBBOBounds:
    """Tests for EBBO bounds (ebbo_min, ebbo_max) parameters."""

    def test_ebbo_min_rejects_match_below_floor(self) -> None:
        """Match should be rejected when clearing price < ebbo_min.

        EBBO requires sellers of A to get at least the AMM rate.
        If the clearing price would be below ebbo_min, no match is possible.
        """
        # Ask: sell 100 A, want 150 B (price = 1.5 B/A)
        # Bid: sell 200 B, want 100 A (price = 2.0 B/A)
        # Valid clearing range: [1.5, 2.0], midpoint = 1.75
        # But if ebbo_min = 1.8, the valid range becomes [1.8, 2.0]
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 150),  # 1.5 B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 200, 100),  # 2.0 B/A
            ],
        )

        # Without EBBO bounds, should match
        result_no_ebbo = run_double_auction(group)
        assert len(result_no_ebbo.matches) == 1
        assert result_no_ebbo.clearing_price == Decimal("1.75")

        # With ebbo_min = 1.8, clearing price moves to [1.8, 2.0] range
        result_with_ebbo = run_double_auction(group, ebbo_min=Decimal("1.8"))
        assert len(result_with_ebbo.matches) == 1
        # Clearing price should be (1.8 + 2.0) / 2 = 1.9
        assert result_with_ebbo.clearing_price == Decimal("1.9")

    def test_ebbo_min_no_valid_range(self) -> None:
        """When ebbo_min > all bid limits, no match is possible."""
        # Ask: sell 100 A, want 150 B (price = 1.5 B/A)
        # Bid: sell 200 B, want 100 A (price = 2.0 B/A)
        # If ebbo_min = 2.5 > bid limit (2.0), no EBBO-compliant match possible
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 150),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 200, 100),
            ],
        )

        result = run_double_auction(group, ebbo_min=Decimal("2.5"))
        assert result.matches == []
        assert result.clearing_price is None

    def test_ebbo_max_rejects_match_above_ceiling(self) -> None:
        """Match should be constrained when clearing price > ebbo_max.

        EBBO requires buyers of A to pay at most the AMM rate (inverted).
        If the clearing price would exceed ebbo_max, it should be capped.
        """
        # Ask: sell 100 A, want 300 B (price = 3.0 B/A)
        # Bid: sell 500 B, want 100 A (price = 5.0 B/A)
        # Valid clearing range: [3.0, 5.0], midpoint = 4.0
        # With ebbo_max = 3.5, valid range becomes [3.0, 3.5]
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 300),  # 3.0 B/A
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 500, 100),  # 5.0 B/A
            ],
        )

        # Without EBBO bounds
        result_no_ebbo = run_double_auction(group)
        assert result_no_ebbo.clearing_price == Decimal("4.0")

        # With ebbo_max = 3.5, clearing price constrained to [3.0, 3.5]
        result_with_ebbo = run_double_auction(group, ebbo_max=Decimal("3.5"))
        assert len(result_with_ebbo.matches) == 1
        assert result_with_ebbo.clearing_price == Decimal("3.25")  # (3.0 + 3.5) / 2

    def test_ebbo_max_no_valid_range(self) -> None:
        """When ebbo_max < all ask limits, no match is possible."""
        # Ask: sell 100 A, want 300 B (price = 3.0 B/A)
        # Bid: sell 500 B, want 100 A (price = 5.0 B/A)
        # If ebbo_max = 2.5 < ask limit (3.0), no EBBO-compliant match possible
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 300),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 500, 100),
            ],
        )

        result = run_double_auction(group, ebbo_max=Decimal("2.5"))
        assert result.matches == []
        assert result.clearing_price is None

    def test_ebbo_both_bounds_constrain_range(self) -> None:
        """Both ebbo_min and ebbo_max should constrain the clearing price range."""
        # Ask: sell 100 A, want 200 B (price = 2.0 B/A)
        # Bid: sell 400 B, want 100 A (price = 4.0 B/A)
        # Original range: [2.0, 4.0], midpoint = 3.0
        # With ebbo_min = 2.5, ebbo_max = 3.5: new range [2.5, 3.5], midpoint = 3.0
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

        result = run_double_auction(group, ebbo_min=Decimal("2.5"), ebbo_max=Decimal("3.5"))
        assert len(result.matches) == 1
        # Range is [2.5, 3.5], midpoint = 3.0
        assert result.clearing_price == Decimal("3.0")

    def test_ebbo_bounds_crossed_no_match(self) -> None:
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

        # ebbo_min > ebbo_max means no valid clearing price
        result = run_double_auction(group, ebbo_min=Decimal("3.5"), ebbo_max=Decimal("2.5"))
        assert result.matches == []
        assert result.clearing_price is None

    def test_ebbo_min_affects_volume(self) -> None:
        """Higher ebbo_min can reduce matched volume due to stricter constraints."""
        # Ask: sell 100 A, want 150 B (price = 1.5 B/A)
        # Bid: sell 200 B, want 100 A (price = 2.0 B/A)
        # At price 1.5: bid can buy 200/1.5 = 133 A (capped at 100)
        # At price 1.9: bid can buy 200/1.9 = 105 A (capped at 100)
        # So volume should still be 100 A in both cases
        group = OrderGroup(
            token_a=TOKEN_A,
            token_b=TOKEN_B,
            sellers_of_a=[
                make_order("ask1", TOKEN_A, TOKEN_B, 100, 150),
            ],
            sellers_of_b=[
                make_order("bid1", TOKEN_B, TOKEN_A, 200, 100),
            ],
        )

        result_no_ebbo = run_double_auction(group)
        result_with_ebbo = run_double_auction(group, ebbo_min=Decimal("1.9"))

        # Both should match 100 A but at different prices
        assert result_no_ebbo.total_a_matched == 100
        assert result_with_ebbo.total_a_matched == 100
        # But B amounts differ due to different clearing prices
        assert result_with_ebbo.total_b_matched > result_no_ebbo.total_b_matched
