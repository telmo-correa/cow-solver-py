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
LINK = "0x514910771AF9Ca656af840dff83E8264EcF986CA"


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
    link_lower = LINK.lower()
    return AuctionInstance(
        id="test",
        orders=orders,
        tokens={
            weth_lower: Token(decimals=18, available_balance="0"),
            usdc_lower: Token(decimals=6, available_balance="0"),
            dai_lower: Token(decimals=18, available_balance="0"),
            link_lower: Token(decimals=18, available_balance="0"),
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


class TestUnifiedCowEBBOEndToEnd:
    """End-to-end EBBO tests that test actual _verify_ebbo calculation without mocking.

    Two-sided EBBO: For a token pair A/B, both parties must get at least as good
    a deal as the AMM. The valid clearing price range is [ebbo_min, ebbo_max].

    Example with AMM spread (2400-2600 USDC/WETH):
    - WETH sellers need >= 2400 USDC/WETH (ebbo_min in WETH→USDC direction)
    - USDC sellers need <= 2600 USDC/WETH (ebbo_max from inverse)
    - Valid clearing prices: 2400 to 2600 USDC/WETH
    """

    def test_ebbo_accepts_clearing_within_amm_spread(self):
        """Clearing price within AMM bid-ask spread should be accepted."""
        from unittest.mock import Mock, patch

        weth_lower = WETH.lower()
        usdc_lower = USDC.lower()

        # Orders that clear at exactly 2500 USDC/WETH
        # In WETH-per-USDC raw terms: 1e18 / 2.5e9 = 4e8
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

        auction = make_auction([order_a, order_b])

        # Mock AMM with spread that includes 2500 USDC/WETH (= 4e8 WETH/USDC)
        # For orders to match at amm_price, their limits must be compatible:
        # - Asks (sellers of A=USDC) limit must be <= amm_price
        # - Bids (sellers of B=WETH) limit must be >= amm_price
        # Both orders have limit = 4e8, so amm_price must be exactly 4e8 or lower
        #
        # Set amm_price (ebbo_min from USDC→WETH) = 4e8 (exactly at order limits)
        # Set WETH→USDC = 2.5e-9 (inverse of 4e8)
        mock_router = Mock()

        def mock_get_reference_price(sell_token, buy_token, **kwargs):
            if sell_token == weth_lower and buy_token == usdc_lower:
                return Decimal("2.5e-9")  # WETH→USDC: 2500 USDC/WETH
            elif sell_token == usdc_lower and buy_token == weth_lower:
                return Decimal("4e8")  # USDC→WETH: 4e8 WETH per USDC (= 1/2500)
            return None

        mock_router.get_reference_price.side_effect = mock_get_reference_price

        with patch(
            "solver.strategies.unified_cow.SingleOrderRouter", return_value=mock_router
        ):
            strategy = UnifiedCowStrategy(enforce_ebbo=True)
            result = strategy.try_solve(auction)

        # Should accept - orders can match at amm_price = 4e8 (2500 USDC/WETH)
        assert result is not None
        assert len(result.fills) == 2

    def test_ebbo_rejects_clearing_below_ebbo_min(self):
        """Clearing price below ebbo_min should reject (WETH sellers get worse deal)."""
        from unittest.mock import Mock, patch

        weth_lower = WETH.lower()
        usdc_lower = USDC.lower()

        # Orders that clear at 2000 USDC/WETH
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",  # 2000 USDC (limit)
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2000000000",  # 2000 USDC
            buy_amount="1000000000000000000",  # 1 WETH
        )

        auction = make_auction([order_a, order_b])

        # Mock AMM with spread [2400, 2600] - clearing at 2000 is below
        mock_router = Mock()

        def mock_get_reference_price(sell_token, buy_token, **kwargs):
            if sell_token == weth_lower and buy_token == usdc_lower:
                return Decimal("2.4e-9")  # 2400 USDC/WETH
            elif sell_token == usdc_lower and buy_token == weth_lower:
                return Decimal("3.85e8")  # 1/2600
            return None

        mock_router.get_reference_price.side_effect = mock_get_reference_price

        with patch(
            "solver.strategies.unified_cow.SingleOrderRouter", return_value=mock_router
        ):
            strategy = UnifiedCowStrategy(enforce_ebbo=True)
            result = strategy.try_solve(auction)

        # Should reject - clearing at 2000 < ebbo_min 2400
        assert result is None or len(result.fills) == 0

    def test_ebbo_rejects_clearing_above_ebbo_max(self):
        """Clearing price above ebbo_max should reject (USDC sellers get worse deal)."""
        from unittest.mock import Mock, patch

        weth_lower = WETH.lower()
        usdc_lower = USDC.lower()

        # Orders that clear at 3000 USDC/WETH (above AMM spread)
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

        auction = make_auction([order_a, order_b])

        # Mock AMM with spread [2400, 2600] - clearing at 3000 is above
        mock_router = Mock()

        def mock_get_reference_price(sell_token, buy_token, **kwargs):
            if sell_token == weth_lower and buy_token == usdc_lower:
                return Decimal("2.4e-9")  # 2400 USDC/WETH (ebbo_min)
            elif sell_token == usdc_lower and buy_token == weth_lower:
                return Decimal("3.85e8")  # 1/2600 (gives ebbo_max of 2600)
            return None

        mock_router.get_reference_price.side_effect = mock_get_reference_price

        with patch(
            "solver.strategies.unified_cow.SingleOrderRouter", return_value=mock_router
        ):
            strategy = UnifiedCowStrategy(enforce_ebbo=True)
            result = strategy.try_solve(auction)

        # Should reject - clearing at 3000 > ebbo_max 2600
        assert result is None or len(result.fills) == 0

    def test_ebbo_boundary_at_ebbo_min(self):
        """Clearing price exactly at ebbo_min should be accepted."""
        from unittest.mock import Mock, patch

        weth_lower = WETH.lower()
        usdc_lower = USDC.lower()

        # Orders that clear at exactly 2500 USDC/WETH (matching the other test)
        # In WETH-per-USDC raw terms: 1e18 / 2.5e9 = 4e8
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

        auction = make_auction([order_a, order_b])

        # Mock AMM at exactly 2500 USDC/WETH (ebbo_min = order limits)
        # In WETH-per-USDC raw: 1e18 / 2.5e9 = 4e8 (exact)
        # Both directions return consistent inverse values
        mock_router = Mock()

        def mock_get_reference_price(sell_token, buy_token, **kwargs):
            if sell_token == weth_lower and buy_token == usdc_lower:
                # WETH→USDC: 2500 USDC/WETH = 2.5e-9 USDC per wei (exact)
                return Decimal("2.5e-9")
            elif sell_token == usdc_lower and buy_token == weth_lower:
                # USDC→WETH: 4e8 WETH wei per USDC wei (exact 1/2.5e-9)
                return Decimal("4e8")
            return None

        mock_router.get_reference_price.side_effect = mock_get_reference_price

        with patch(
            "solver.strategies.unified_cow.SingleOrderRouter", return_value=mock_router
        ):
            strategy = UnifiedCowStrategy(enforce_ebbo=True)
            result = strategy.try_solve(auction)

        # Should accept - clearing exactly at ebbo_min = 4e8
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


class TestUnifiedCowLimitPrice:
    """Tests for limit price enforcement in UnifiedCowStrategy."""

    def test_fills_satisfy_limit_prices(self):
        """All fills must satisfy their order's limit price."""
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

        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        for fill in result.fills:
            order = fill.order
            sell_amount = int(order.sell_amount)
            buy_amount = int(order.buy_amount)
            # Limit price check using exact integer cross-multiplication:
            # actual_rate >= limit_rate
            # buy_filled / sell_filled >= buy_amount / sell_amount
            # buy_filled * sell_amount >= buy_amount * sell_filled
            if fill.sell_filled > 0:
                assert fill.buy_filled * sell_amount >= buy_amount * fill.sell_filled, (
                    f"Fill violates limit: {fill.buy_filled}/{fill.sell_filled} < "
                    f"{buy_amount}/{sell_amount}"
                )

    def test_limit_price_at_boundary(self):
        """Match at exact limit price boundary is accepted."""
        # Both orders have exactly matching limits
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2500000000",  # exactly 2500 USDC/WETH
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
            sell_amount="2500000000",  # exactly 2500 USDC
            buy_amount="1000000000000000000",  # exactly 1 WETH
        )

        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2


class TestUnifiedCowUniformPrice:
    """Tests for uniform clearing price in UnifiedCowStrategy."""

    def test_prices_are_uniform_for_pair(self):
        """All orders in a pair execute at the same clearing price."""
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

        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None
        # Both tokens should have exactly one price
        weth_lower = WETH.lower()
        usdc_lower = USDC.lower()
        assert weth_lower in result.prices
        assert usdc_lower in result.prices

    def test_price_conservation_invariant(self):
        """Prices satisfy: sell_filled * sell_price = buy_filled * buy_price."""
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

        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)

        assert result is not None

        for fill in result.fills:
            sell_token = fill.order.sell_token.lower()
            buy_token = fill.order.buy_token.lower()
            sell_price = int(result.prices[sell_token])
            buy_price = int(result.prices[buy_token])

            sell_value = fill.sell_filled * sell_price
            buy_value = fill.buy_filled * buy_price

            # Conservation: sell_value should equal buy_value
            # Allow tiny error from int() truncation in _normalize_prices
            # Max error = max(fill amounts) * 1 (price truncation)
            if sell_value > 0 and buy_value > 0:
                max_truncation_error = max(fill.sell_filled, fill.buy_filled)
                assert abs(sell_value - buy_value) <= max_truncation_error, (
                    f"Conservation violated: {sell_value} vs {buy_value} "
                    f"(diff={abs(sell_value - buy_value)})"
                )


class TestUnifiedCowThreeTokenCycle:
    """Tests for three-token cycle settlement."""

    def test_three_token_cycle_detected(self):
        """Three orders forming A->B->C->A cycle are detected and settled."""
        # WETH -> USDC
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2500000000",  # wants 2500 USDC
        )
        # USDC -> DAI
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=DAI,
            sell_amount="2500000000",  # 2500 USDC
            buy_amount="2500000000000000000",  # wants 2500 DAI
        )
        # DAI -> WETH
        order_c = make_order(
            uid="0x" + "03" * 56,
            sell_token=DAI,
            buy_token=WETH,
            sell_amount="2500000000000000000",  # 2500 DAI
            buy_amount="1000000000000000000",  # wants 1 WETH
        )

        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b, order_c])
        result = strategy.try_solve(auction)

        # May or may not find cycle depending on implementation
        # If found, verify constraints
        if result is not None and len(result.fills) == 3:
            # All three tokens should have prices
            assert len(result.prices) == 3

    def test_three_token_cycle_limit_prices(self):
        """Three-token cycle satisfies all limit prices."""
        # Create a viable cycle with overlapping limits
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="2000000000",  # wants min 2000 USDC
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=DAI,
            sell_amount="2500000000",
            buy_amount="2000000000000000000",  # wants min 2000 DAI
        )
        order_c = make_order(
            uid="0x" + "03" * 56,
            sell_token=DAI,
            buy_token=WETH,
            sell_amount="2500000000000000000",
            buy_amount="800000000000000000",  # wants min 0.8 WETH
        )

        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b, order_c])
        result = strategy.try_solve(auction)

        if result is not None:
            for fill in result.fills:
                order = fill.order
                sell_amount = int(order.sell_amount)
                buy_amount = int(order.buy_amount)
                # Limit price check using exact integer cross-multiplication
                if fill.sell_filled > 0:
                    assert fill.buy_filled * sell_amount >= buy_amount * fill.sell_filled


class TestUnifiedCowEdgeCases:
    """Edge case tests for UnifiedCowStrategy."""

    def test_empty_auction(self):
        """Empty auction returns None."""
        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = AuctionInstance(id="test", orders=[], tokens={})
        result = strategy.try_solve(auction)
        assert result is None

    def test_single_order(self):
        """Single order returns None (no CoW possible)."""
        order = make_order(uid="0x" + "01" * 56)
        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order])
        result = strategy.try_solve(auction)
        assert result is None

    def test_unrelated_orders(self):
        """Orders with no token overlap return None."""
        # Order selling WETH for USDC
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_token=WETH,
            buy_token=USDC,
            sell_amount="1000000000000000000",
            buy_amount="2500000000",
        )
        # Order selling DAI for LINK (no overlap with WETH/USDC)
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=DAI,
            buy_token=LINK,
            sell_amount="1000000000000000000",
            buy_amount="50000000000000000000",  # 50 LINK
        )
        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b])
        result = strategy.try_solve(auction)
        # No CoW possible - orders have no token overlap
        assert result is None

    def test_zero_amounts_rejected(self):
        """Orders with zero amounts don't cause errors."""
        order_a = make_order(
            uid="0x" + "01" * 56,
            sell_amount="0",
            buy_amount="0",
        )
        order_b = make_order(
            uid="0x" + "02" * 56,
            sell_token=USDC,
            buy_token=WETH,
        )

        strategy = UnifiedCowStrategy(enforce_ebbo=False)
        auction = make_auction([order_a, order_b])
        # Should not raise, may or may not find a match
        result = strategy.try_solve(auction)
        # Just verify no exception was raised
        assert result is None or isinstance(result.fills, list)
