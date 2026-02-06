"""Tests for cycle settlement (C3: FOK bypass fix).

Validates fill-or-kill enforcement in cycle settlement calculations.
"""

from solver.models.auction import Order
from solver.strategies.settlement import (
    CycleViability,
    calculate_cycle_settlement,
    solve_cycle,
)

# Counter for generating unique UIDs
_uid_counter = 10000


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


class TestCycleFillOrKill:
    """C3: FOK orders must not be partially filled in cycles."""

    def test_fok_order_with_partial_fill_returns_none(self) -> None:
        """A FOK order that would be partially filled should reject the cycle.

        Cycle: A→B→C→A
        Order 0 (A→B): large FOK order (bottleneck is elsewhere)
        Order 1 (B→C): small order (bottleneck)
        Order 2 (C→A): normal order

        The small order 1 is the bottleneck, so order 0 gets partially filled.
        Since order 0 is FOK, the cycle should be rejected.
        """
        # Order 0 sells 1000 A for 800 B - FOK, will be partially filled
        order_a_to_b = make_order(
            TOKEN_A,
            TOKEN_B,
            sell_amount=1000,
            buy_amount=800,
            partially_fillable=False,  # FOK
        )
        # Order 1 sells 100 B for 90 C - bottleneck (small)
        order_b_to_c = make_order(
            TOKEN_B,
            TOKEN_C,
            sell_amount=100,
            buy_amount=90,
            partially_fillable=True,
        )
        # Order 2 sells 90 C for 80 A
        order_c_to_a = make_order(
            TOKEN_C,
            TOKEN_A,
            sell_amount=90,
            buy_amount=80,
            partially_fillable=True,
        )

        result = solve_cycle([order_a_to_b, order_b_to_c, order_c_to_a])

        # Should return None because order_a_to_b is FOK but would be partially filled
        assert result is None

    def test_all_partial_orders_succeed(self) -> None:
        """When all orders are partially fillable, cycle should settle normally."""
        order_a_to_b = make_order(
            TOKEN_A,
            TOKEN_B,
            sell_amount=1000,
            buy_amount=800,
            partially_fillable=True,
        )
        order_b_to_c = make_order(
            TOKEN_B,
            TOKEN_C,
            sell_amount=100,
            buy_amount=90,
            partially_fillable=True,
        )
        order_c_to_a = make_order(
            TOKEN_C,
            TOKEN_A,
            sell_amount=90,
            buy_amount=80,
            partially_fillable=True,
        )

        result = solve_cycle([order_a_to_b, order_b_to_c, order_c_to_a])

        # Should succeed since all orders allow partial fills
        assert result is not None
        assert len(result.fills) == 3

    def test_fok_order_at_bottleneck_succeeds(self) -> None:
        """A FOK order that IS the bottleneck gets fully filled (success).

        When the FOK order is the smallest, it determines the bottleneck.
        Its sell_filled == sell_amount, so FOK is satisfied.
        """
        # Order 0 sells 100 A for 80 B - FOK, IS the bottleneck
        order_a_to_b = make_order(
            TOKEN_A,
            TOKEN_B,
            sell_amount=100,
            buy_amount=80,
            partially_fillable=False,  # FOK
        )
        # Order 1 sells 1000 B for 900 C - large, partially fillable
        order_b_to_c = make_order(
            TOKEN_B,
            TOKEN_C,
            sell_amount=1000,
            buy_amount=900,
            partially_fillable=True,
        )
        # Order 2 sells 1000 C for 900 A - large, partially fillable
        order_c_to_a = make_order(
            TOKEN_C,
            TOKEN_A,
            sell_amount=1000,
            buy_amount=900,
            partially_fillable=True,
        )

        result = solve_cycle([order_a_to_b, order_b_to_c, order_c_to_a])

        # Should succeed - FOK order is the bottleneck so it's fully filled
        assert result is not None
        assert len(result.fills) == 3

        # Verify the FOK order is indeed fully filled
        fok_fill = next(f for f in result.fills if f.order.uid == order_a_to_b.uid)
        assert fok_fill.sell_filled == order_a_to_b.sell_amount_int


class TestCalculateCycleSettlementFOK:
    """C3: FOK enforcement in calculate_cycle_settlement (via CycleViability)."""

    def test_fok_order_partial_returns_none(self) -> None:
        """calculate_cycle_settlement rejects FOK partial fills."""
        # FOK order that won't be the bottleneck
        order_a = make_order(
            TOKEN_A, TOKEN_B, sell_amount=1000, buy_amount=500, partially_fillable=False
        )
        order_b = make_order(
            TOKEN_B, TOKEN_C, sell_amount=100, buy_amount=50, partially_fillable=True
        )
        order_c = make_order(
            TOKEN_C, TOKEN_A, sell_amount=50, buy_amount=25, partially_fillable=True
        )

        viability = CycleViability(
            viable=True,
            surplus_ratio=0.5,
            product=0.5,
            orders=[order_a, order_b, order_c],
            product_num=1,
            product_denom=2,
        )

        result = calculate_cycle_settlement(viability)

        # FOK order_a cannot be partially filled
        assert result is None
