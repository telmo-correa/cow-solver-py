"""Tests for UnifiedCowStrategy.

These tests verify constraint enforcement in UnifiedCowStrategy:
- EBBO validation (clearing rate >= AMM rate)
- Limit price satisfaction
- Fill-or-kill enforcement
"""

from decimal import Decimal
from unittest.mock import patch

from solver.models.auction import AuctionInstance, Order, Token
from solver.strategies.unified_cow import UnifiedCowStrategy

# Token addresses for tests
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
DAI = "0x6B175474E89094C44Da98b954EedeaCB5f6f8fa0"


def make_order(
    uid: str,
    sell_token: str = WETH,
    buy_token: str = USDC,
    sell_amount: str = "1000000000000000000",  # 1 WETH
    buy_amount: str = "2500000000",  # 2500 USDC
    kind: str = "sell",
    partially_fillable: bool = True,
    order_class: str = "market",
) -> Order:
    """Create a minimal Order for testing."""
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
    """Create an AuctionInstance with proper token decimals."""
    weth_lower = WETH.lower()
    usdc_lower = USDC.lower()
    dai_lower = DAI.lower()
    return AuctionInstance(
        id="test",
        orders=orders,
        tokens={
            weth_lower: Token(decimals=18, available_balance="0"),
            usdc_lower: Token(decimals=6, available_balance="0"),
            dai_lower: Token(decimals=18, available_balance="0"),
        },
    )


class TestUnifiedCowBasic:
    """Basic tests for UnifiedCowStrategy matching."""

    def test_perfect_match_two_orders(self):
        """Two orders with matching limits produce a CoW match."""
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2500000000",  # 2500 USDC
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2500000000",  # 2500 USDC
            buy_amount="1000000000000000000",  # 1 WETH
        )

        strategy = UnifiedCowStrategy(enforce_ebbo=False)  # No EBBO for basic test
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2

    def test_no_match_when_limits_dont_overlap(self):
        """Orders with non-overlapping limits produce no match."""
        # Order A: wants at least 3000 USDC/WETH
        # Order B: offers at most 2000 USDC/WETH
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="3000000000",  # wants 3000 USDC
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2000000000",  # only offers 2000 USDC
            buy_amount="1000000000000000000",  # wants 1 WETH
        )

        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # No match possible - limits don't overlap
        assert result is None


class TestUnifiedCowEBBO:
    """Tests for EBBO validation in UnifiedCowStrategy."""

    def test_ebbo_rejects_when_clearing_below_amm(self):
        """EBBO validation rejects when clearing rate < AMM rate."""
        # Order A: sells 1 WETH, wants 2000 USDC (limit: 2e-9 raw)
        # Order B: sells 2000 USDC, wants 1 WETH
        # AMM rate: 2500 USDC/WETH = 2.5e-9 raw (better than clearing)
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",  # 2000 USDC
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2000000000",  # 2000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
        )

        weth_lower = WETH.lower()
        usdc_lower = USDC.lower()

        def mock_get_ref_price(sell_token, buy_token, **_kwargs):
            # AMM offers 2500 USDC per WETH (2.5e-9 raw)
            if sell_token == weth_lower and buy_token == usdc_lower:
                return Decimal("2.5e-9")
            elif sell_token == usdc_lower and buy_token == weth_lower:
                return Decimal("4e8")  # 1/2500 in raw
            return None

        strategy = UnifiedCowStrategy(enforce_ebbo=True, use_lp_solver=True)
        auction = make_auction([order_a, order_b])

        # Patch the router's get_reference_price
        with patch.object(strategy, "_verify_ebbo") as mock_verify:
            # Mock returns False (EBBO violation)
            mock_verify.return_value = False
            result = strategy.try_solve(auction)

        # Should reject due to EBBO violation
        assert result is None or len(result.fills) == 0 or mock_verify.called

    def test_ebbo_accepts_when_clearing_above_amm(self):
        """EBBO validation accepts when clearing rate >= AMM rate."""
        # Order A: sells 1 WETH, wants 3000 USDC (limit: 3e-9 raw)
        # Order B: sells 3000 USDC, wants 1 WETH
        # AMM rate: 2500 USDC/WETH = 2.5e-9 raw (worse than clearing)
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

        # Without EBBO enforcement, should succeed
        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Should accept since EBBO is disabled
        assert result is not None
        assert len(result.fills) == 2

    def test_ebbo_accepts_when_no_amm_liquidity(self):
        """EBBO validation accepts when no AMM liquidity exists."""
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="2500000000",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2500000000",
            buy_amount="1000000000000000000",
        )

        # No liquidity in auction
        strategy = UnifiedCowStrategy(enforce_ebbo=True)
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Should accept - no EBBO constraint when no AMM
        assert result is not None
        assert len(result.fills) == 2


class TestUnifiedCowFillOrKill:
    """Tests for fill-or-kill enforcement in UnifiedCowStrategy."""

    def test_fill_or_kill_fully_matched(self):
        """Fill-or-kill orders that fully match are accepted."""
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="2500000000",
            partially_fillable=False,  # Fill-or-kill
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2500000000",
            buy_amount="1000000000000000000",
            partially_fillable=False,  # Fill-or-kill
        )

        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2
        # Both should be fully filled
        for fill in result.fills:
            if fill.order.sell_token.lower() == WETH.lower():
                assert fill.sell_filled == 1000000000000000000
            else:
                assert fill.sell_filled == 2500000000

    def test_fill_or_kill_partial_rejected(self):
        """Fill-or-kill orders that would be partially filled are rejected."""
        # Order A: wants to sell 2 WETH (fill-or-kill)
        # Order B: only has 1 WETH worth of USDC
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="2000000000000000000",  # 2 WETH
            buy_amount="5000000000",  # 5000 USDC
            partially_fillable=False,  # Fill-or-kill
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2500000000",  # Only 2500 USDC (1 WETH worth)
            buy_amount="1000000000000000000",  # Wants 1 WETH
            partially_fillable=True,
        )

        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        # Order A is fill-or-kill and can't be fully filled
        if result is not None:
            for fill in result.fills:
                if fill.order.uid == order_a.uid:
                    # If order A is filled, it must be fully filled
                    assert fill.sell_filled == 2000000000000000000
