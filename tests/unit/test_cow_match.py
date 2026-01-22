"""Tests for the CowMatchStrategy."""

from solver.models.auction import AuctionInstance, Order
from solver.strategies.cow_match import CowMatchStrategy


def make_order(
    uid: str,
    sell_token: str,
    buy_token: str,
    sell_amount: str,
    buy_amount: str,
    kind: str = "sell",
    partially_fillable: bool = False,
    order_class: str = "market",  # Market orders don't require fee calculation
) -> Order:
    """Create a minimal Order for testing.

    Args:
        partially_fillable: If True, order can be partially filled.
            If False (default), order is fill-or-kill.
        order_class: Order class ("market" or "limit"). Defaults to "market"
            to avoid fee calculation requirements in CoW matching tests.
    """
    return Order(
        uid=uid,
        sell_token=sell_token,
        buy_token=buy_token,
        sell_amount=sell_amount,
        buy_amount=buy_amount,
        kind=kind,
        class_=order_class,
        partially_fillable=partially_fillable,
    )


def make_auction(orders: list[Order]) -> AuctionInstance:
    """Create a minimal AuctionInstance for testing."""
    return AuctionInstance(id="test", orders=orders)


# Token addresses for tests
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
DAI = "0x6B175474E89094C44Da98b954EedeaCB5f6f8fa0"


class TestCowMatchDetection:
    """Tests for detecting CoW matches."""

    def test_perfect_match_found(self):
        """Two opposite orders with matching amounts are detected."""
        # Order A: sells 1 WETH, wants 3000 USDC
        # Order B: sells 3000 USDC, wants 1 WETH
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="3000000000",  # 3000 USDC
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2
        assert len(result.interactions) == 0  # No AMM needed
        # Perfect match means no remainders
        assert len(result.remainder_orders) == 0

    def test_surplus_for_both_parties(self):
        """Match where both parties get better than their limit price."""
        # Order A: sells 1 WETH, wants min 2000 USDC (limit: 2000 USDC/WETH)
        # Order B: sells 3000 USDC, wants 1 WETH (limit: 3000 USDC/WETH)
        # Result: A gets 3000 USDC (surplus!), B gets 1 WETH
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="2000000000",  # Wants only 2000 USDC
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # Offering 3000 USDC
            buy_amount="1000000000000000000",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        # Check fills have correct executed amounts
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)

        # A (sell): executed_amount = sell amount
        assert fill_a.executed_amount == 1000000000000000000
        # B (sell): executed_amount = sell amount
        assert fill_b.executed_amount == 3000000000

    def test_no_match_wrong_token_direction(self):
        """Orders with same direction don't match."""
        # Both want to sell WETH for USDC
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="3000000000",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",
            buy_amount="6000000000",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is None

    def test_no_match_different_tokens(self):
        """Orders with different token pairs don't match."""
        # Order A: WETH -> USDC
        # Order B: DAI -> WETH (A wants USDC but B offers WETH, not USDC)
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="3000000000",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=DAI,
            buy_token=WETH,
            sell_amount="3000000000000000000",
            buy_amount="1000000000000000000",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is None

    def test_no_match_limit_price_not_met_for_a(self):
        """No match when order A's limit price isn't satisfied."""
        # Order A: sells 1 WETH, wants 3000 USDC
        # Order B: sells only 2000 USDC, wants 1 WETH
        # A would only get 2000 USDC < 3000 limit
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="3000000000",  # Wants 3000 USDC
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2000000000",  # Only offering 2000 USDC
            buy_amount="1000000000000000000",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is None

    def test_partial_match_b_wants_more_than_a_offers(self):
        """Partial match when B wants more than A offers.

        With partial matching, A fills completely and B gets a partial fill
        with a remainder order for the unfilled portion.
        B must have partially_fillable=True since B will be partially filled.
        """
        # Order A: sells 0.5 WETH, wants 1500 USDC
        # Order B: sells 3000 USDC, wants 1 WETH (partially_fillable)
        # Partial CoW: A sells 0.5 WETH, B sells 1500 USDC
        # B remainder: 1500 USDC wanting 0.5 WETH
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="500000000000000000",  # 0.5 WETH
            buy_amount="1500000000",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",
            buy_amount="1000000000000000000",  # Wants 1 WETH
            partially_fillable=True,  # B will be partially filled
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Partial match found
        assert result is not None
        assert len(result.fills) == 2
        assert len(result.remainder_orders) == 1  # B has remainder

        # A is completely filled
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        assert fill_a.sell_filled == 500000000000000000  # 0.5 WETH
        assert fill_a.buy_filled == 1500000000  # 1500 USDC
        assert fill_a.is_complete

        # B is partially filled
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert fill_b.sell_filled == 1500000000  # 1500 USDC
        assert fill_b.buy_filled == 500000000000000000  # 0.5 WETH
        assert not fill_b.is_complete

        # B's remainder wants 0.5 WETH for 1500 USDC
        # Remainder has a NEW derived UID, not the original
        remainder = result.remainder_orders[0]
        assert remainder.uid != order_b.uid
        assert remainder.uid.startswith("0x")
        assert int(remainder.sell_amount) == 1500000000  # 1500 USDC
        assert int(remainder.buy_amount) == 500000000000000000  # 0.5 WETH

    def test_single_order_returns_none(self):
        """Single-order auctions don't produce CoW matches."""
        order = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="3000000000",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order])
        result = strategy.try_solve(auction)

        assert result is None

    def test_three_orders_returns_none(self):
        """Three-order auctions not yet supported."""
        orders = [
            make_order(f"0x{str(i).zfill(2) * 56}", WETH, USDC, "1000", "3000") for i in range(3)
        ]

        strategy = CowMatchStrategy()
        auction = make_auction(orders)
        result = strategy.try_solve(auction)

        assert result is None

    def test_sell_buy_match(self):
        """Sell order matched with buy order (sell-buy)."""
        # Order A (sell): sells 1 WETH, wants min 2500 USDC
        # Order B (buy): wants 1 WETH, willing to pay up to 3000 USDC
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # sells 1 WETH
            buy_amount="2500000000",  # wants min 2500 USDC
            kind="sell",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # max to pay: 3000 USDC
            buy_amount="1000000000000000000",  # wants exactly 1 WETH
            kind="buy",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2
        assert len(result.interactions) == 0

        # Check fills have correct executed amounts
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)

        # A (sell): executed_amount = sell amount
        assert fill_a.executed_amount == 1000000000000000000
        # B (buy): executed_amount = buy amount
        assert fill_b.executed_amount == 1000000000000000000

    def test_buy_sell_match(self):
        """Buy order matched with sell order (buy-sell)."""
        # Order A (buy): wants 3000 USDC, willing to pay up to 1 WETH
        # Order B (sell): sells 3000 USDC, wants min 1 WETH
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # max to pay: 1 WETH
            buy_amount="3000000000",  # wants exactly 3000 USDC
            kind="buy",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # sells 3000 USDC
            buy_amount="1000000000000000000",  # wants min 1 WETH
            kind="sell",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2
        assert len(result.interactions) == 0

        # Check fills have correct executed amounts
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)

        # A (buy): executed_amount = buy amount
        assert fill_a.executed_amount == 3000000000
        # B (sell): executed_amount = sell amount
        assert fill_b.executed_amount == 3000000000

    def test_buy_buy_match(self):
        """Two buy orders matched (buy-buy)."""
        # Order A (buy): wants 1 WETH, willing to pay up to 3000 USDC
        # Order B (buy): wants 2500 USDC, willing to pay up to 1 WETH
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # max to pay: 3000 USDC
            buy_amount="1000000000000000000",  # wants 1 WETH
            kind="buy",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # max to pay: 1 WETH
            buy_amount="2500000000",  # wants 2500 USDC
            kind="buy",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2
        assert len(result.interactions) == 0

        # Check fills have correct executed amounts
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)

        # A (buy): executed_amount = buy amount (1 WETH)
        assert fill_a.executed_amount == 1000000000000000000
        # B (buy): executed_amount = buy amount (2500 USDC)
        assert fill_b.executed_amount == 2500000000

    def test_sell_buy_partial_match_when_amounts_differ(self):
        """Sell-buy produces partial match when sell amount != buy amount."""
        # Order A (sell): sells 2 WETH, wants min 5000 USDC (partially_fillable)
        # Order B (buy): wants only 1 WETH, pays up to 3000 USDC
        # Partial match: A sells 1 WETH, B buys 1 WETH
        # A is partially filled, so A needs partially_fillable=True
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="5000000000",  # min 5000 USDC
            kind="sell",
            partially_fillable=True,  # A will be partially filled
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # max 3000 USDC
            buy_amount="1000000000000000000",  # wants exactly 1 WETH
            kind="buy",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Now produces partial match (A partially fills, B completely fills)
        assert result is not None
        assert len(result.fills) == 2
        assert len(result.remainder_orders) == 1  # A has remainder

    def test_sell_buy_no_match_limit_not_met(self):
        """Sell-buy doesn't match when sell order's limit not satisfied."""
        # Order A (sell): sells 1 WETH, wants min 3500 USDC
        # Order B (buy): wants 1 WETH, max payment 3000 USDC
        # B can't pay enough for A's limit
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="3500000000",  # wants 3500 USDC
            kind="sell",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # can only pay 3000 USDC
            buy_amount="1000000000000000000",
            kind="buy",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is None

    def test_buy_buy_partial_match_a_cant_fully_afford(self):
        """Buy-buy partial match when A can't afford all B wants.

        A gets complete fill (1 WETH), B gets partial fill (2000 of 2500 USDC).
        B is partially filled, so B needs partially_fillable=True.
        """
        # Order A (buy): wants 1 WETH, max payment 2000 USDC
        # Order B (buy): wants 2500 USDC, max payment 1 WETH (partially_fillable)
        # B can satisfy A (1 WETH), A can't fully satisfy B (2000 < 2500)
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2000000000",  # max payment: 2000 USDC
            buy_amount="1000000000000000000",  # wants 1 WETH
            kind="buy",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # max payment: 1 WETH
            buy_amount="2500000000",  # wants 2500 USDC
            kind="buy",
            partially_fillable=True,  # B will be partially filled
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Partial match: A complete (got 1 WETH), B partial (got 2000 USDC)
        assert result is not None
        assert len(result.fills) == 2

        # A is complete (buy order got exact amount)
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        assert fill_a.is_complete

        # B is not complete (wanted 2500, got 2000)
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert not fill_b.is_complete

    def test_buy_buy_partial_match_b_cant_fully_afford(self):
        """Buy-buy partial match when B can't afford all A wants.

        B gets complete fill (2500 USDC), A gets partial fill (0.5 of 1 WETH).
        A is partially filled, so A needs partially_fillable=True.
        """
        # Order A (buy): wants 1 WETH, max payment 3000 USDC (partially_fillable)
        # Order B (buy): wants 2500 USDC, max payment 0.5 WETH
        # A can satisfy B (3000 >= 2500), B can't satisfy A (0.5 < 1)
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # max payment: 3000 USDC
            buy_amount="1000000000000000000",  # wants 1 WETH
            kind="buy",
            partially_fillable=True,  # A will be partially filled
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="500000000000000000",  # max payment: 0.5 WETH
            buy_amount="2500000000",  # wants 2500 USDC
            kind="buy",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Partial match: B complete (got 2500 USDC), A partial (got 0.5 WETH)
        assert result is not None
        assert len(result.fills) == 2

        # A is not complete (wanted 1 WETH, got 0.5)
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        assert not fill_a.is_complete

        # B is complete (got 2500 USDC)
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert fill_b.is_complete

    def test_partial_match_a_offers_more_than_b_wants(self):
        """Partial match when A offers more than B wants.

        B fills completely, A has remainder for AMM routing.
        A is partially filled, so A needs partially_fillable=True.
        """
        # Order A: sells 2 WETH, wants 5000 USDC (partially_fillable)
        # Order B: sells 3000 USDC, wants 1 WETH (rate: 3000 USDC/WETH)
        # Partial CoW: B fills completely (1 WETH for 3000 USDC)
        # A remainder: 1 WETH wanting 2000 USDC (5000-3000)
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="5000000000",  # 5000 USDC
            partially_fillable=True,  # A will be partially filled
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Partial match found
        assert result is not None
        assert len(result.fills) == 2
        assert len(result.remainder_orders) == 1  # A has remainder

        # B is completely filled
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert fill_b.sell_filled == 3000000000  # 3000 USDC
        assert fill_b.buy_filled == 1000000000000000000  # 1 WETH
        assert fill_b.is_complete

        # A is partially filled
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        assert fill_a.sell_filled == 1000000000000000000  # 1 WETH
        assert fill_a.buy_filled == 3000000000  # 3000 USDC
        assert not fill_a.is_complete

        # A's remainder: unfilled portion (5000-3000=2000 USDC for 1 WETH)
        # Remainder has a NEW derived UID, not the original
        remainder = result.remainder_orders[0]
        assert remainder.uid != order_a.uid
        assert remainder.uid.startswith("0x")
        assert int(remainder.sell_amount) == 1000000000000000000  # 1 WETH
        assert int(remainder.buy_amount) == 2000000000  # 2000 USDC (5000-3000)

    def test_partial_match_incompatible_limits(self):
        """No partial match when limit prices are incompatible."""
        # Order A: sells 1 WETH, wants 4000 USDC (rate: 4000 USDC/WETH)
        # Order B: sells 3000 USDC, wants 1 WETH (rate: 3000 USDC/WETH)
        # Limits incompatible: A wants more than B is willing to give
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="4000000000",  # 4000 USDC (too high)
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # No match - limits incompatible
        assert result is None


class TestCowMatchSolution:
    """Tests for the solution built from a CoW match."""

    def test_clearing_prices_are_set(self):
        """Result has clearing prices for both tokens."""
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="3000000000",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",
            buy_amount="1000000000000000000",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        # Prices should be set for both tokens (normalized to lowercase)
        weth_lower = WETH.lower()
        usdc_lower = USDC.lower()
        assert weth_lower in result.prices
        assert usdc_lower in result.prices

    def test_clearing_prices_satisfy_limit_constraints(self):
        """Clearing prices satisfy both orders' limit constraints."""
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2500000000",  # wants min 2500 USDC
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # wants 1 WETH
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None

        weth_lower = WETH.lower()
        usdc_lower = USDC.lower()
        price_weth = int(result.prices[weth_lower])
        price_usdc = int(result.prices[usdc_lower])

        # Check A's constraint: sell_a * price_weth >= limit_a * price_usdc
        sell_a = 1000000000000000000
        limit_a = 2500000000  # A's minimum buy amount (from order)
        assert sell_a * price_weth >= limit_a * price_usdc

        # Check B's constraint: sell_b * price_usdc >= limit_b * price_weth
        sell_b = 3000000000
        limit_b = 1000000000000000000  # B's minimum buy amount (from order)
        assert sell_b * price_usdc >= limit_b * price_weth

    def test_trades_have_correct_executed_amounts(self):
        """Trades built from fills have correct executed amounts."""
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="3000000000",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",
            buy_amount="1000000000000000000",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None

        # Build solution and check trades
        solution = result.build_solution()
        trade_a = next(t for t in solution.trades if t.order == order_a.uid)
        trade_b = next(t for t in solution.trades if t.order == order_b.uid)

        # For sell orders, executed_amount is the sell amount
        assert trade_a.executed_amount == "1000000000000000000"
        assert trade_b.executed_amount == "3000000000"

    def test_no_interactions_in_cow_solution(self):
        """CoW results have no AMM interactions."""
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="3000000000",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",
            buy_amount="1000000000000000000",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert result.interactions == []
        assert result.gas == 0  # No AMM = no gas for swaps


class TestCowMatchAddressNormalization:
    """Tests for address handling in CoW matching."""

    def test_matches_with_different_case_addresses(self):
        """Addresses are normalized for comparison."""
        # Same addresses but different case
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH.lower(),  # lowercase
            buy_token=USDC.upper().replace("X", "x"),  # mixed case
            sell_amount="1000000000000000000",
            buy_amount="3000000000",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,  # original case
            buy_token=WETH,  # original case
            sell_amount="3000000000",
            buy_amount="1000000000000000000",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Should still match because addresses are normalized
        assert result is not None


class TestPartialMatchAllOrderTypes:
    """Tests for partial matching across all order type combinations.

    Partial matching fills one order completely while partially filling the other,
    creating a remainder order for subsequent strategies (e.g., AMM routing).
    """

    def test_partial_match_sell_buy(self):
        """Partial matching for sell-buy orders.

        A (sell) offers more X than B (buy) wants, so B fills completely
        and A has a remainder. A needs partially_fillable=True.
        """
        # Order A (sell): sells 2 WETH, wants min 5000 USDC (partially_fillable)
        # Order B (buy): wants exactly 1 WETH, willing to pay up to 3000 USDC
        # Partial: A sells 1 WETH to B, receives 3000 USDC
        # A remainder: 1 WETH wanting 2000 USDC (5000-3000)
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="5000000000",  # min 5000 USDC
            kind="sell",
            partially_fillable=True,  # A will be partially filled
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # max 3000 USDC
            buy_amount="1000000000000000000",  # wants exactly 1 WETH
            kind="buy",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2
        assert len(result.remainder_orders) == 1

        # A is partially filled
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        assert fill_a.sell_filled == 1000000000000000000  # 1 WETH
        assert fill_a.buy_filled == 3000000000  # 3000 USDC
        assert not fill_a.is_complete

        # B is completely filled (buy order got exact amount)
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert fill_b.sell_filled == 3000000000  # paid 3000 USDC
        assert fill_b.buy_filled == 1000000000000000000  # got 1 WETH
        assert fill_b.is_complete

        # A's remainder (with NEW derived UID)
        remainder = result.remainder_orders[0]
        assert remainder.uid != order_a.uid
        assert remainder.uid.startswith("0x")
        assert int(remainder.sell_amount) == 1000000000000000000  # 1 WETH
        assert int(remainder.buy_amount) == 2000000000  # 2000 USDC

    def test_partial_match_buy_sell(self):
        """Partial matching for buy-sell orders.

        B (sell) offers more Y than A (buy) wants, so A fills completely
        and B has a remainder. B needs partially_fillable=True.

        Note: B only has a remainder if B's requirement is NOT fully met.
        If A's payment >= B's min requirement, B has no remainder.
        """
        # Order A (buy): wants exactly 1000 USDC, willing to pay up to 0.5 WETH
        # Order B (sell): sells 3000 USDC, wants min 1 WETH (partially_fillable)
        # Partial: B sells 1000 USDC to A, receives 0.5 WETH
        # B remainder: 2000 USDC wanting 0.5 WETH (1 WETH - 0.5 already received)
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="500000000000000000",  # max 0.5 WETH
            buy_amount="1000000000",  # wants exactly 1000 USDC
            kind="buy",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # sells 3000 USDC
            buy_amount="1000000000000000000",  # wants min 1 WETH
            kind="sell",
            partially_fillable=True,  # B will be partially filled
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2
        assert len(result.remainder_orders) == 1

        # A is completely filled (buy order got exact amount)
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        assert fill_a.sell_filled == 500000000000000000  # paid 0.5 WETH
        assert fill_a.buy_filled == 1000000000  # got 1000 USDC
        assert fill_a.is_complete

        # B is partially filled
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert fill_b.sell_filled == 1000000000  # sold 1000 USDC
        assert fill_b.buy_filled == 500000000000000000  # got 0.5 WETH
        assert not fill_b.is_complete

        # B's remainder: 2000 USDC wanting 0.5 WETH (with NEW derived UID)
        remainder = result.remainder_orders[0]
        assert remainder.uid != order_b.uid
        assert remainder.uid.startswith("0x")
        assert int(remainder.sell_amount) == 2000000000  # 2000 USDC
        assert int(remainder.buy_amount) == 500000000000000000  # 0.5 WETH

    def test_partial_match_buy_sell_no_remainder_when_b_satisfied(self):
        """Buy-sell partial match where B has no remainder because B's requirement is met.

        When A's payment >= B's min requirement, B has no unfulfilled need.
        B still needs partially_fillable=True since B doesn't sell full amount.
        """
        # Order A (buy): wants exactly 2000 USDC, willing to pay up to 1 WETH
        # Order B (sell): sells 3000 USDC, wants min 1 WETH (partially_fillable)
        # Partial: B sells 2000 USDC, receives 1 WETH
        # B got 1 WETH (>= min 1 WETH), so B has no remainder even though
        # B has 1000 USDC left unsold
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # max 1 WETH
            buy_amount="2000000000",  # wants exactly 2000 USDC
            kind="buy",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # sells 3000 USDC
            buy_amount="1000000000000000000",  # wants min 1 WETH
            kind="sell",
            partially_fillable=True,  # B will be partially filled (sells only 2000/3000)
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2

        # A is complete
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        assert fill_a.is_complete

        # B got 1 WETH which meets their min requirement
        # Even though B only sold 2000/3000 USDC, B has no remainder
        # because B's "need" (1 WETH) is fully satisfied
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert fill_b.buy_filled == 1000000000000000000  # got 1 WETH
        assert fill_b.sell_filled == 2000000000  # only sold 2000 USDC

        # No remainder because B's requirement is met
        assert len(result.remainder_orders) == 0

    def test_partial_match_buy_buy_b_complete(self):
        """Partial matching for buy-buy orders where B fills completely.

        A can pay what B wants, but B can't pay all A wants.
        B gets complete fill, A gets partial fill.
        A needs partially_fillable=True.
        """
        # Order A (buy): wants 1 WETH, willing to pay up to 3000 USDC (partially_fillable)
        # Order B (buy): wants 2000 USDC, willing to pay up to 0.5 WETH
        # A can satisfy B (3000 >= 2000), but B can't satisfy A (0.5 < 1)
        # B gets complete fill, A gets partial (0.5 WETH instead of 1)
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # max 3000 USDC
            buy_amount="1000000000000000000",  # wants 1 WETH
            kind="buy",
            partially_fillable=True,  # A will be partially filled
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="500000000000000000",  # max 0.5 WETH
            buy_amount="2000000000",  # wants 2000 USDC
            kind="buy",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2
        assert len(result.remainder_orders) == 1

        # A is partially filled
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        assert fill_a.sell_filled == 2000000000  # paid 2000 USDC
        assert fill_a.buy_filled == 500000000000000000  # got 0.5 WETH
        assert not fill_a.is_complete  # wanted 1 WETH, got 0.5

        # B is completely filled
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert fill_b.sell_filled == 500000000000000000  # paid 0.5 WETH
        assert fill_b.buy_filled == 2000000000  # got 2000 USDC
        assert fill_b.is_complete

        # A's remainder (with NEW derived UID)
        remainder = result.remainder_orders[0]
        assert remainder.uid != order_a.uid
        assert remainder.uid.startswith("0x")

    def test_partial_match_buy_buy_a_complete(self):
        """Partial matching for buy-buy orders where A fills completely.

        B can pay what A wants, but A can't pay all B wants.
        A gets complete fill, B gets partial fill.
        B needs partially_fillable=True.
        """
        # Order A (buy): wants 0.5 WETH, willing to pay up to 2000 USDC
        # Order B (buy): wants 3000 USDC, willing to pay up to 1 WETH (partially_fillable)
        # B can satisfy A (1 >= 0.5), but A can't satisfy B (2000 < 3000)
        # A gets complete fill, B gets partial (2000 USDC instead of 3000)
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2000000000",  # max 2000 USDC
            buy_amount="500000000000000000",  # wants 0.5 WETH
            kind="buy",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # max 1 WETH
            buy_amount="3000000000",  # wants 3000 USDC
            kind="buy",
            partially_fillable=True,  # B will be partially filled
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2
        assert len(result.remainder_orders) == 1

        # A is completely filled
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        assert fill_a.sell_filled == 2000000000  # paid 2000 USDC
        assert fill_a.buy_filled == 500000000000000000  # got 0.5 WETH
        assert fill_a.is_complete

        # B is partially filled
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert fill_b.sell_filled == 500000000000000000  # paid 0.5 WETH
        assert fill_b.buy_filled == 2000000000  # got 2000 USDC
        assert not fill_b.is_complete  # wanted 3000 USDC, got 2000

        # B's remainder (with NEW derived UID)
        remainder = result.remainder_orders[0]
        assert remainder.uid != order_b.uid
        assert remainder.uid.startswith("0x")

    def test_partial_match_sell_buy_limit_not_satisfied(self):
        """No partial sell-buy match when A's limit can't be satisfied."""
        # Order A (sell): sells 2 WETH, wants min 6000 USDC (rate: 3000 USDC/WETH)
        # Order B (buy): wants 1 WETH, willing to pay up to 2000 USDC (rate: 2000 USDC/WETH)
        # A's limit (3000/WETH) > B's rate (2000/WETH), so no match
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="6000000000",  # min 6000 USDC
            kind="sell",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2000000000",  # max 2000 USDC
            buy_amount="1000000000000000000",  # wants 1 WETH
            kind="buy",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is None


class TestRemainderOrderBehavior:
    """Tests for remainder order generation and properties.

    These tests verify that remainder orders:
    1. Preserve the original order's UID (for fill merging)
    2. Have correctly reduced amounts
    3. Maintain the same limit price ratio

    See Issue #11 in code review.
    """

    def test_remainder_order_has_new_uid_but_tracks_original(self):
        """Remainder order has a NEW UID but tracks original_uid for merging.

        Remainder orders get new UIDs for uniqueness, but preserve the
        original_uid attribute so fill merging works across strategies.
        """
        original_uid = "0x" + "01" * 56
        order_a = make_order(
            uid=original_uid,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="5000000000",  # 5000 USDC
            partially_fillable=True,  # A will be partially filled
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.remainder_orders) == 1

        remainder = result.remainder_orders[0]
        # Remainder has a NEW UID (different from original)
        assert remainder.uid != original_uid
        assert remainder.uid.startswith("0x")
        # But tracks the original_uid for fill merging
        assert hasattr(remainder, "original_uid")
        assert remainder.original_uid == original_uid

    def test_remainder_order_has_reduced_amounts(self):
        """Remainder order has correctly reduced sell and buy amounts."""
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="5000000000",  # 5000 USDC (rate: 2500 USDC/WETH)
            partially_fillable=True,  # A will be partially filled
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.remainder_orders) == 1

        remainder = result.remainder_orders[0]
        # A sold 1 WETH in CoW match, remainder should be 1 WETH
        assert int(remainder.sell_amount) == 1000000000000000000  # 1 WETH
        # A received 3000 USDC in CoW match, remainder should want 2000 USDC
        assert int(remainder.buy_amount) == 2000000000  # 2000 USDC

    def test_remainder_order_ensures_total_requirement_met(self):
        """Remainder order is calculated to meet the original total requirement.

        This is the correct behavior: if A wants 5000 USDC total for 2 WETH,
        and CoW gives 3000 USDC for 1 WETH, the remainder needs only 2000 USDC
        for 1 WETH (not 2500 USDC to preserve the rate).

        This can result in a LOWER limit price for the remainder, which is
        actually favorable - A already got a better rate from CoW.
        """
        original_sell = 2000000000000000000  # 2 WETH
        original_buy = 5000000000  # 5000 USDC total
        # Original rate: 5000/2 = 2500 USDC/WETH

        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount=str(original_sell),
            buy_amount=str(original_buy),
            partially_fillable=True,  # A will be partially filled
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.remainder_orders) == 1

        remainder = result.remainder_orders[0]
        remainder_sell = int(remainder.sell_amount)
        remainder_buy = int(remainder.buy_amount)

        # Verify remainder ensures total requirement is met
        # A got 3000 USDC from CoW for 1 WETH
        # Remainder should need 2000 USDC (5000 - 3000) for 1 WETH
        cow_received = 3000000000  # From CoW match
        assert remainder_buy == original_buy - cow_received

        # The remainder's limit price (2000/1 = 2000 USDC/WETH) is LOWER
        # than original (2500 USDC/WETH), which is correct and favorable
        original_rate = original_buy / original_sell
        remainder_rate = remainder_buy / remainder_sell
        assert remainder_rate < original_rate, (
            "Remainder rate should be lower (easier to fill) since CoW gave a good rate"
        )

    def test_remainder_order_full_amounts_updated(self):
        """Remainder order has full_sell_amount and full_buy_amount updated.

        These fields should reflect the remainder amounts, not the original.
        """
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",
            buy_amount="5000000000",
            partially_fillable=True,  # A will be partially filled
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",
            buy_amount="1000000000000000000",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        remainder = result.remainder_orders[0]

        # full_sell_amount should match sell_amount
        assert remainder.full_sell_amount == remainder.sell_amount
        # full_buy_amount should match buy_amount
        assert remainder.full_buy_amount == remainder.buy_amount

    def test_no_remainder_when_order_fully_filled(self):
        """No remainder order generated when an order is completely filled."""
        # Order B will be fully filled (sells all 3000 USDC)
        # Order A will be partially filled, so A needs partially_fillable=True
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="5000000000",  # 5000 USDC
            partially_fillable=True,  # A will be partially filled
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None

        # Only A should have a remainder (B is fully filled)
        # Remainder has a NEW UID (derived from original), not the original UID
        assert len(result.remainder_orders) == 1
        assert result.remainder_orders[0].uid != order_a.uid
        assert result.remainder_orders[0].uid.startswith("0x")

        # Verify B's fill is complete
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert fill_b.is_complete


class TestFillOrKillBehavior:
    """Tests for fill-or-kill (partiallyFillable=false) order behavior.

    Per CoW Protocol spec:
    - partiallyFillable=false (default): Fill-or-kill, must fill completely or not at all
    - partiallyFillable=true: Can be partially filled

    Key scenarios:
    1. Fill-or-kill orders can participate in PERFECT matches (both fully filled)
    2. Fill-or-kill orders CANNOT be partially filled
    3. Fill-or-kill order CAN be fully filled while the OTHER order is partial
    """

    def test_fill_or_kill_perfect_match_works(self):
        """Fill-or-kill orders can be perfectly matched.

        Both orders are fill-or-kill (default), and both are fully filled.
        This should work because neither is partially filled.
        """
        # Order A: sells 1 WETH, wants 3000 USDC (fill-or-kill)
        # Order B: sells 3000 USDC, wants 1 WETH (fill-or-kill)
        # Perfect match: both fully filled
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="3000000000",  # 3000 USDC
            # partiallyFillable=False (default)
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
            # partiallyFillable=False (default)
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Perfect match should work
        assert result is not None
        assert len(result.fills) == 2
        assert len(result.remainder_orders) == 0  # No remainders

        # Both fills are complete
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert fill_a.is_complete
        assert fill_b.is_complete

    def test_fill_or_kill_cannot_be_partially_filled(self):
        """Fill-or-kill order cannot be partially filled.

        If order A is fill-or-kill and would need to be partially filled,
        no match should occur.
        """
        # Order A: sells 2 WETH, wants 5000 USDC (fill-or-kill)
        # Order B: sells 3000 USDC, wants 1 WETH (fill-or-kill)
        # Would be partial match: A partially fills (1 WETH), B fully fills
        # BUT A is fill-or-kill, so NO match
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="5000000000",  # 5000 USDC
            # partiallyFillable=False (default) - A is fill-or-kill
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
            # partiallyFillable=False (default)
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # No match because A would need to be partially filled
        assert result is None

    def test_fill_or_kill_fully_filled_while_other_partial(self):
        """Fill-or-kill order can be fully filled while the other is partial.

        Order A is fill-or-kill but gets FULLY filled.
        Order B is partiallyFillable and gets partially filled.
        This should work!
        """
        # Order A: sells 1 WETH, wants 2500 USDC (fill-or-kill)
        # Order B: sells 5000 USDC, wants 2 WETH (partiallyFillable)
        # Partial match: A fully fills (1 WETH), B partially fills (2500 USDC)
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2500000000",  # 2500 USDC
            # partiallyFillable=False (default) - fill-or-kill
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="5000000000",  # 5000 USDC
            buy_amount="2000000000000000000",  # 2 WETH
            partially_fillable=True,  # B can be partially filled
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Match should work: A is fully filled, B is partially filled
        assert result is not None
        assert len(result.fills) == 2
        assert len(result.remainder_orders) == 1  # B has remainder

        # A is completely filled
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        assert fill_a.is_complete

        # B is partially filled
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert not fill_b.is_complete

        # Remainder is B's (with new derived UID)
        assert len(result.remainder_orders) == 1
        assert result.remainder_orders[0].uid != order_b.uid
        assert result.remainder_orders[0].uid.startswith("0x")

    def test_both_fill_or_kill_no_partial_match(self):
        """When both orders are fill-or-kill, no partial match is possible.

        Even if limits are compatible, if one would need partial fill, no match.
        """
        # Order A: sells 2 WETH, wants 4000 USDC (fill-or-kill)
        # Order B: sells 3000 USDC, wants 1 WETH (fill-or-kill)
        # Limits compatible, but would require partial fill of A
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="4000000000",  # 4000 USDC (rate: 2000 USDC/WETH)
            # partiallyFillable=False (default)
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH (rate: 3000 USDC/WETH)
            # partiallyFillable=False (default)
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # No match: A would need to be partially filled
        assert result is None

    def test_fill_or_kill_b_cannot_be_partially_filled(self):
        """Fill-or-kill order B cannot be partially filled.

        Reverse scenario: B is fill-or-kill and would be partially filled.
        """
        # Order A: sells 0.5 WETH, wants 1500 USDC (partiallyFillable)
        # Order B: sells 3000 USDC, wants 1 WETH (fill-or-kill)
        # Would be partial: A fully fills, B partially fills
        # BUT B is fill-or-kill, so NO match
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="500000000000000000",  # 0.5 WETH
            buy_amount="1500000000",  # 1500 USDC
            partially_fillable=True,  # A can be partial (but won't need to be)
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
            # partiallyFillable=False (default) - B is fill-or-kill
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # No match: B would need to be partially filled but is fill-or-kill
        assert result is None

    def test_mixed_partial_fillable_a_partial_b_fok(self):
        """A is partiallyFillable, B is fill-or-kill, B gets fully filled.

        A offers more than B wants. B gets fully filled, A is partially filled.
        Since A allows partial fills and B is fully filled, this should work.
        """
        # Order A: sells 2 WETH, wants 5000 USDC (partiallyFillable)
        # Order B: sells 3000 USDC, wants 1 WETH (fill-or-kill)
        # B wants 1 WETH, A has 2 WETH -> B fully fills, A partial fills
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="5000000000",  # 5000 USDC
            partially_fillable=True,  # A can be partially filled
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # 3000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
            # partiallyFillable=False (default) - B is fill-or-kill
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Match works: B fully fills, A partially fills
        assert result is not None
        assert len(result.fills) == 2

        # A is partially filled (sold 1 WETH of 2)
        fill_a = next(f for f in result.fills if f.order.uid == order_a.uid)
        assert not fill_a.is_complete
        assert fill_a.sell_filled == 1000000000000000000  # 1 WETH

        # B is fully filled
        fill_b = next(f for f in result.fills if f.order.uid == order_b.uid)
        assert fill_b.is_complete

        # A has remainder with a new derived UID
        assert len(result.remainder_orders) == 1
        # Remainder has a NEW UID (derived from original), not the original UID
        assert result.remainder_orders[0].uid != order_a.uid
        assert result.remainder_orders[0].uid.startswith("0x")


class TestStrategyResultCombine:
    """Tests for StrategyResult.combine() behavior."""

    def test_combine_merges_fills_by_original_uid(self):
        """Fills with different UIDs but same original_uid are merged."""
        from solver.strategies.base import OrderFill, StrategyResult

        original_uid = "0x" + "01" * 56
        remainder_uid = "0x" + "FF" * 56

        # Create original order
        order = make_order(
            uid=original_uid,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="4000000000",  # 4000 USDC
            partially_fillable=True,
        )

        # Create remainder order (different UID, tracks original)
        remainder = order.model_copy(
            update={
                "uid": remainder_uid,
                "sell_amount": "1000000000000000000",  # 1 WETH
                "buy_amount": "2000000000",  # 2000 USDC
                "original_uid": original_uid,
            }
        )

        # First result: partial fill of original
        result1 = StrategyResult(
            fills=[OrderFill(order=order, sell_filled=1000000000000000000, buy_filled=3000000000)],
            prices={WETH.lower(): "3000000000", USDC.lower(): "1000000000000000000"},
        )

        # Second result: fill of remainder (different UID, same original_uid)
        result2 = StrategyResult(
            fills=[
                OrderFill(order=remainder, sell_filled=1000000000000000000, buy_filled=2500000000)
            ],
            prices={WETH.lower(): "2500000000", USDC.lower(): "1000000000000000000"},
        )

        combined = StrategyResult.combine([result1, result2])

        # Should have ONE merged fill, not two separate fills
        assert len(combined.fills) == 1
        fill = combined.fills[0]

        # Merged amounts: 2 WETH sold, 5500 USDC received
        assert fill.sell_filled == 2000000000000000000
        assert fill.buy_filled == 5500000000

        # Should use original order (not remainder)
        assert fill.order.uid == original_uid

    def test_combine_validates_fills_satisfy_limits(self):
        """combine() raises PriceWorsened if fill violates order limit."""
        import pytest

        from solver.strategies.base import OrderFill, PriceWorsened, StrategyResult

        # Order wants 3000 USDC for 1 WETH (rate: 3000)
        order = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="3000000000",  # 3000 USDC
        )

        # Fill only gives 2000 USDC - VIOLATES limit!
        bad_fill = OrderFill(
            order=order,
            sell_filled=1000000000000000000,
            buy_filled=2000000000,  # Only 2000 USDC, below 3000 limit
        )

        result = StrategyResult(
            fills=[bad_fill],
            prices={WETH.lower(): "2000000000", USDC.lower(): "1000000000000000000"},
        )

        with pytest.raises(PriceWorsened) as exc_info:
            StrategyResult.combine([result])

        assert "limit violated" in str(exc_info.value)

    def test_combine_accepts_fills_at_or_above_limit(self):
        """combine() accepts fills that meet or exceed limit."""
        from solver.strategies.base import OrderFill, StrategyResult

        # Order wants 3000 USDC for 1 WETH
        order = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="3000000000",  # 3000 USDC
        )

        # Fill gives exactly 3000 USDC - meets limit
        good_fill = OrderFill(
            order=order,
            sell_filled=1000000000000000000,
            buy_filled=3000000000,  # Exactly 3000 USDC
        )

        result = StrategyResult(
            fills=[good_fill],
            prices={WETH.lower(): "3000000000", USDC.lower(): "1000000000000000000"},
        )

        # Should not raise
        combined = StrategyResult.combine([result])
        assert len(combined.fills) == 1

    def test_combine_accepts_surplus_fills(self):
        """combine() accepts fills that give user better than limit."""
        from solver.strategies.base import OrderFill, StrategyResult

        # Order wants 3000 USDC for 1 WETH
        order = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="3000000000",  # 3000 USDC minimum
        )

        # Fill gives 4000 USDC - exceeds limit (surplus!)
        surplus_fill = OrderFill(
            order=order,
            sell_filled=1000000000000000000,
            buy_filled=4000000000,  # 4000 USDC, above 3000 limit
        )

        result = StrategyResult(
            fills=[surplus_fill],
            prices={WETH.lower(): "4000000000", USDC.lower(): "1000000000000000000"},
        )

        # Should not raise
        combined = StrategyResult.combine([result])
        assert len(combined.fills) == 1


class TestRemainderOrderUidGeneration:
    """Tests for remainder order UID generation."""

    def test_remainder_uid_is_deterministic(self):
        """Same original UID always produces same remainder UID."""
        from solver.strategies.base import OrderFill

        original_uid = "0x" + "AB" * 56

        order = make_order(
            uid=original_uid,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",
            buy_amount="4000000000",
            partially_fillable=True,
        )

        fill1 = OrderFill(order=order, sell_filled=1000000000000000000, buy_filled=2000000000)
        fill2 = OrderFill(order=order, sell_filled=1000000000000000000, buy_filled=2000000000)

        remainder1 = fill1.get_remainder_order()
        remainder2 = fill2.get_remainder_order()

        # Same input produces same remainder UID
        assert remainder1 is not None
        assert remainder2 is not None
        assert remainder1.uid == remainder2.uid

    def test_remainder_uid_differs_from_original(self):
        """Remainder UID is different from original UID."""
        from solver.strategies.base import OrderFill

        original_uid = "0x" + "CD" * 56

        order = make_order(
            uid=original_uid,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",
            buy_amount="4000000000",
            partially_fillable=True,
        )

        fill = OrderFill(order=order, sell_filled=1000000000000000000, buy_filled=2000000000)
        remainder = fill.get_remainder_order()

        assert remainder is not None
        assert remainder.uid != original_uid
        assert len(remainder.uid) == len(original_uid)  # Same format

    def test_remainder_tracks_original_uid(self):
        """Remainder order has original_uid field set."""
        from solver.strategies.base import OrderFill

        original_uid = "0x" + "EF" * 56

        order = make_order(
            uid=original_uid,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",
            buy_amount="4000000000",
            partially_fillable=True,
        )

        fill = OrderFill(order=order, sell_filled=1000000000000000000, buy_filled=2000000000)
        remainder = fill.get_remainder_order()

        assert remainder is not None
        assert remainder.original_uid == original_uid

    def test_chained_remainders_preserve_root_original_uid(self):
        """Multiple remainder generations preserve the root original UID."""
        from solver.strategies.base import OrderFill

        root_uid = "0x" + "11" * 56

        # Original order
        order = make_order(
            uid=root_uid,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="3000000000000000000",  # 3 WETH
            buy_amount="6000000000",  # 6000 USDC
            partially_fillable=True,
        )

        # First partial fill -> first remainder
        fill1 = OrderFill(order=order, sell_filled=1000000000000000000, buy_filled=2000000000)
        remainder1 = fill1.get_remainder_order()
        assert remainder1 is not None
        assert remainder1.original_uid == root_uid

        # Second partial fill of remainder -> second remainder
        fill2 = OrderFill(order=remainder1, sell_filled=1000000000000000000, buy_filled=2000000000)
        remainder2 = fill2.get_remainder_order()
        assert remainder2 is not None

        # Second remainder should still track the ROOT original UID
        assert remainder2.original_uid == root_uid
        # But have a different UID from both original and first remainder
        assert remainder2.uid != root_uid
        assert remainder2.uid != remainder1.uid
