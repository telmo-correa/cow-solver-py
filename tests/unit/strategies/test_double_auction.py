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
            name="seller",
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
            name="buyer",
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

    def test_partial_fill_allowed(self) -> None:
        """Partially fillable orders should match partially."""
        # Ask: sell 100 A, want 200 B (price = 2 B/A), partially fillable
        # Bid: sell 125 B, want 50 A (price = 2.5 B/A)
        # Midpoint price = 2.25 B/A
        # At price 2.25, bid can buy 125/2.25 = 55.5 -> 55 A
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

        # Should match ~55 A (limited by bid's B at clearing price)
        assert len(result.matches) == 1
        assert result.total_a_matched == 55  # 125 / 2.25 = 55.5 -> 55
        # Remaining 45 A should be unmatched
        assert len(result.unmatched_sellers) == 1
        assert result.unmatched_sellers[0][1] == 45  # 100 - 55 = 45 A remaining


class TestMultiOrderAuction:
    """Tests for multi-order double auctions."""

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
