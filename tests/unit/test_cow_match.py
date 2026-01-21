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
) -> Order:
    """Create a minimal Order for testing."""
    return Order(
        uid=uid,
        sell_token=sell_token,
        buy_token=buy_token,
        sell_amount=sell_amount,
        buy_amount=buy_amount,
        kind=kind,
        class_="limit",
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
        solution = strategy.try_solve(auction)

        assert solution is not None
        assert len(solution.trades) == 2
        assert len(solution.interactions) == 0  # No AMM needed

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
        solution = strategy.try_solve(auction)

        assert solution is not None
        # A's trade: executed 1 WETH
        assert solution.trades[0].executed_amount == "1000000000000000000"
        # B's trade: executed 3000 USDC
        assert solution.trades[1].executed_amount == "3000000000"

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
        solution = strategy.try_solve(auction)

        assert solution is None

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
        solution = strategy.try_solve(auction)

        assert solution is None

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
        solution = strategy.try_solve(auction)

        assert solution is None

    def test_no_match_limit_price_not_met_for_b(self):
        """No match when order B's limit price isn't satisfied."""
        # Order A: sells 0.5 WETH, wants 1500 USDC
        # Order B: sells 3000 USDC, wants 1 WETH
        # B would only get 0.5 WETH < 1 WETH limit
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
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        solution = strategy.try_solve(auction)

        assert solution is None

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
        solution = strategy.try_solve(auction)

        assert solution is None

    def test_three_orders_returns_none(self):
        """Three-order auctions not yet supported."""
        orders = [
            make_order(f"0x{str(i).zfill(2) * 56}", WETH, USDC, "1000", "3000") for i in range(3)
        ]

        strategy = CowMatchStrategy()
        auction = make_auction(orders)
        solution = strategy.try_solve(auction)

        assert solution is None

    def test_no_match_buy_orders(self):
        """Buy orders are not yet supported for CoW matching."""
        # Order A: buy order wanting 1 WETH, willing to pay up to 3000 USDC
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="3000000000",  # max to sell
            buy_amount="1000000000000000000",  # wants 1 WETH
            kind="buy",
        )
        # Order B: sell order selling 1 WETH for 3000 USDC
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="3000000000",
            kind="sell",
        )

        strategy = CowMatchStrategy()
        auction = make_auction([order_a, order_b])
        solution = strategy.try_solve(auction)

        # Buy orders should not match (not yet supported)
        assert solution is None


class TestCowMatchSolution:
    """Tests for the solution built from a CoW match."""

    def test_clearing_prices_are_set(self):
        """Solution has clearing prices for both tokens."""
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
        solution = strategy.try_solve(auction)

        assert solution is not None
        # Prices should be set for both tokens (normalized to lowercase)
        weth_lower = WETH.lower()
        usdc_lower = USDC.lower()
        assert weth_lower in solution.prices
        assert usdc_lower in solution.prices

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
        solution = strategy.try_solve(auction)

        assert solution is not None

        weth_lower = WETH.lower()
        usdc_lower = USDC.lower()
        price_weth = int(solution.prices[weth_lower])
        price_usdc = int(solution.prices[usdc_lower])

        # Check A's constraint: sell_a * price_weth >= limit_a * price_usdc
        sell_a = 1000000000000000000
        limit_a = 2500000000  # A's minimum buy amount (from order)
        assert sell_a * price_weth >= limit_a * price_usdc

        # Check B's constraint: sell_b * price_usdc >= limit_b * price_weth
        sell_b = 3000000000
        limit_b = 1000000000000000000  # B's minimum buy amount (from order)
        assert sell_b * price_usdc >= limit_b * price_weth

    def test_trades_have_correct_executed_amounts(self):
        """Trades have correct executed amounts for sell orders."""
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
        solution = strategy.try_solve(auction)

        assert solution is not None

        # Find trades by order UID
        trade_a = next(t for t in solution.trades if t.order == order_a.uid)
        trade_b = next(t for t in solution.trades if t.order == order_b.uid)

        # For sell orders, executed_amount is the sell amount
        assert trade_a.executed_amount == "1000000000000000000"
        assert trade_b.executed_amount == "3000000000"

    def test_no_interactions_in_cow_solution(self):
        """CoW solutions have no AMM interactions."""
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
        solution = strategy.try_solve(auction)

        assert solution is not None
        assert solution.interactions == []
        assert solution.gas == 0  # No AMM = no gas for swaps


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
        solution = strategy.try_solve(auction)

        # Should still match because addresses are normalized
        assert solution is not None
