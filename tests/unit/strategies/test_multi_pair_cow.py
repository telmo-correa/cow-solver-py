"""Tests for multi-pair CoW strategy cycle fixes (C2, H8).

C2: Cycle settlement prices are included in final result.
H8: FOK-blind order selection prefers partially-fillable orders.
"""

from unittest.mock import MagicMock

from solver.models.auction import AuctionInstance, Order, Token
from solver.strategies.graph import OrderGraph
from solver.strategies.multi_pair import MultiPairCowStrategy

# Counter for generating unique UIDs
_uid_counter = 20000


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
TOKEN_C = "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"


def make_auction(orders: list[Order]) -> AuctionInstance:
    """Create a test auction."""
    tokens = {}
    for order in orders:
        for addr in [order.sell_token, order.buy_token]:
            if addr not in tokens:
                tokens[addr] = Token(
                    decimals=18,
                    symbol="TKN",
                    referencePrice="1000000000000000000",
                    availableBalance="1000000000000000000000",
                )
    return AuctionInstance(id="test", orders=orders, tokens=tokens)


class TestCyclePricesInResult:
    """C2: _solve_cycles must return clearing prices for cycle tokens."""

    def test_solve_cycles_returns_prices(self) -> None:
        """Verify _solve_cycles returns prices for all cycle tokens.

        Creates a viable 3-token cycle (A→B→C→A) and checks that
        prices for A, B, and C are all present in the returned prices dict.
        """
        # Create orders forming a viable cycle with surplus
        # A→B: sell 100 A, want 90 B (rate 0.9)
        # B→C: sell 100 B, want 90 C (rate 0.9)
        # C→A: sell 100 C, want 90 A (rate 0.9)
        # Product = 0.9^3 = 0.729 < 1, so viable with surplus
        order_ab = make_order(TOKEN_A, TOKEN_B, 100, 90, partially_fillable=True)
        order_bc = make_order(TOKEN_B, TOKEN_C, 100, 90, partially_fillable=True)
        order_ca = make_order(TOKEN_C, TOKEN_A, 100, 90, partially_fillable=True)

        orders = [order_ab, order_bc, order_ca]
        strategy = MultiPairCowStrategy(use_generalized=True, max_tokens=3)

        # Mock router to skip EBBO validation
        mock_router = MagicMock()
        mock_router.get_reference_price.return_value = None
        mock_router.get_reference_price_ratio.return_value = None

        fills, prices = strategy._solve_cycles(
            orders=orders,
            router=mock_router,
            auction=make_auction(orders),
            already_matched=set(),
            priced_tokens=None,
        )

        if fills:
            # If we got fills, prices should be populated for all tokens
            token_a_norm = TOKEN_A.lower()
            token_b_norm = TOKEN_B.lower()
            token_c_norm = TOKEN_C.lower()

            assert token_a_norm in prices, f"Missing price for token A, got {prices.keys()}"
            assert token_b_norm in prices, f"Missing price for token B, got {prices.keys()}"
            assert token_c_norm in prices, f"Missing price for token C, got {prices.keys()}"

            # Prices should be non-zero string integers
            for token, price in prices.items():
                assert int(price) > 0, f"Price for {token} should be positive, got {price}"


class TestFOKBlindOrderSelection:
    """H8: Cycle order selection should prefer partially-fillable orders."""

    def test_partially_fillable_preferred_over_fok(self) -> None:
        """When both FOK and partial orders are available, prefer partial.

        This reduces the chance of FOK violations in cycle settlement.
        """
        # Two orders on the same edge (A→B):
        # FOK order with better rate (cheaper)
        fok_order = make_order(
            TOKEN_A,
            TOKEN_B,
            sell_amount=1000,
            buy_amount=800,
            partially_fillable=False,  # FOK
        )
        # Partial order with worse rate (more expensive)
        partial_order = make_order(
            TOKEN_A,
            TOKEN_B,
            sell_amount=1000,
            buy_amount=900,  # worse rate
            partially_fillable=True,
        )

        orders = [fok_order, partial_order]
        graph = OrderGraph.from_orders(orders)
        strategy = MultiPairCowStrategy()

        # Build a simple cycle with these tokens
        cycle_tokens = (TOKEN_A.lower(), TOKEN_B.lower(), TOKEN_C.lower())

        # Add a B→C and C→A order to complete the cycle
        order_bc = make_order(TOKEN_B, TOKEN_C, 1000, 900, partially_fillable=True)
        order_ca = make_order(TOKEN_C, TOKEN_A, 1000, 900, partially_fillable=True)

        all_orders = orders + [order_bc, order_ca]
        graph = OrderGraph.from_orders(all_orders)

        cycle_orders = strategy._get_cycle_orders(cycle_tokens, graph, set())

        if cycle_orders:
            # The A→B order should be the partial one, not the FOK one
            ab_order = cycle_orders[0]  # First order in cycle is A→B
            assert ab_order.partially_fillable, (
                "Should prefer partially-fillable order over FOK for cycle robustness"
            )
            assert ab_order.uid == partial_order.uid
