"""Unit tests for ring trade detection and execution."""

from __future__ import annotations

from solver.models.auction import AuctionInstance, Order, Token
from solver.strategies.graph import OrderGraph
from solver.strategies.research.ring_trade import RingTrade, RingTradeStrategy
from solver.strategies.settlement import (
    CycleViability,
    check_cycle_viability,
    find_viable_cycle_direction,
)

# Near viable threshold from CycleViability dataclass
NEAR_VIABLE_THRESHOLD = CycleViability.NEAR_VIABLE_THRESHOLD

# --- Fixtures ---


def make_uid(short_id: str) -> str:
    """Create a valid 112-character hex UID from a short identifier."""
    # UID format: 0x + 112 hex chars
    # We use the short_id encoded as hex and pad to 112 chars
    short_hex = short_id.encode().hex()
    return "0x" + short_hex.ljust(112, "0")


def make_order(
    uid: str,
    sell_token: str,
    buy_token: str,
    sell_amount: str,
    buy_amount: str,
) -> Order:
    """Create a test order."""
    # Convert short UID to valid format
    valid_uid = make_uid(uid) if not uid.startswith("0x") or len(uid) != 114 else uid
    return Order(
        uid=valid_uid,
        sell_token=sell_token,
        buy_token=buy_token,
        sell_amount=sell_amount,
        buy_amount=buy_amount,
        kind="sell",
        partially_fillable=False,
        fee_amount="0",
        **{
            "class": "limit",
            "valid_to": 0,
            "app_data": "0x0",
            "sell_token_balance": "erc20",
            "buy_token_balance": "erc20",
        },
    )


def make_auction(orders: list[Order]) -> AuctionInstance:
    """Create a test auction with given orders."""
    tokens = {}
    for order in orders:
        for token in [order.sell_token, order.buy_token]:
            if token not in tokens:
                tokens[token] = Token(
                    address=token,
                    decimals=18,
                    symbol=token[:6],
                    availableBalance="1000000000000000000000000",  # 1M tokens
                )
    return AuctionInstance(
        id="test",
        orders=orders,
        tokens=tokens,
        liquidity=[],
        effective_gas_price="1000000000",
        deadline="2099-01-01T00:00:00Z",
    )


# Token addresses
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
DAI = "0x6b175474e89094c44da98b954eedeac495271d0f"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"


# --- OrderGraph Tests ---


class TestOrderGraph:
    """Tests for OrderGraph class."""

    def test_from_orders_empty(self):
        """Empty orders create empty graph."""
        graph = OrderGraph.from_orders([])
        assert graph.tokens == set()
        assert graph.edge_count == 0

    def test_from_orders_single(self):
        """Single order creates one edge."""
        order = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000")
        graph = OrderGraph.from_orders([order])

        assert WETH.lower() in graph.tokens
        assert USDC.lower() in graph.tokens
        assert graph.edge_count == 1
        assert graph.has_edge(WETH.lower(), USDC.lower())
        assert not graph.has_edge(USDC.lower(), WETH.lower())

    def test_from_orders_multiple_same_pair(self):
        """Multiple orders on same pair go to same edge."""
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000")
        o2 = make_order("o2", WETH, USDC, "2000000000000000000", "3900000000")
        graph = OrderGraph.from_orders([o1, o2])

        assert graph.edge_count == 1
        orders = graph.get_orders(WETH.lower(), USDC.lower())
        assert len(orders) == 2

    def test_from_orders_bidirectional(self):
        """Orders in both directions create two edges."""
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000")
        o2 = make_order("o2", USDC, WETH, "2000000000", "1000000000000000000")
        graph = OrderGraph.from_orders([o1, o2])

        assert graph.edge_count == 2
        assert graph.has_edge(WETH.lower(), USDC.lower())
        assert graph.has_edge(USDC.lower(), WETH.lower())

    def test_get_orders_empty(self):
        """get_orders returns empty list for missing edge."""
        graph = OrderGraph.from_orders([])
        assert graph.get_orders(WETH, USDC) == []

    def test_tokens_property(self):
        """tokens property returns all unique tokens."""
        o1 = make_order("o1", WETH, USDC, "1000", "2000")
        o2 = make_order("o2", DAI, WBTC, "3000", "1")
        graph = OrderGraph.from_orders([o1, o2])

        tokens = graph.tokens
        assert len(tokens) == 4
        assert WETH.lower() in tokens
        assert USDC.lower() in tokens
        assert DAI.lower() in tokens
        assert WBTC.lower() in tokens


class TestOrderGraphCycleDetection:
    """Tests for cycle detection in OrderGraph."""

    def test_find_3_cycles_none(self):
        """No 3-cycles when graph is acyclic."""
        o1 = make_order("o1", WETH, USDC, "1000", "2000")
        o2 = make_order("o2", USDC, DAI, "2000", "2000")
        graph = OrderGraph.from_orders([o1, o2])

        cycles = graph.find_3_cycles()
        assert cycles == []

    def test_find_3_cycles_one(self):
        """Find a single 3-cycle."""
        o1 = make_order("o1", WETH, USDC, "1000", "2000")
        o2 = make_order("o2", USDC, DAI, "2000", "2000")
        o3 = make_order("o3", DAI, WETH, "2000", "1000")
        graph = OrderGraph.from_orders([o1, o2, o3])

        cycles = graph.find_3_cycles()
        assert len(cycles) == 1
        # Cycles are sorted tuples
        cycle = cycles[0]
        assert set(cycle) == {WETH.lower(), USDC.lower(), DAI.lower()}

    def test_find_3_cycles_deduplication(self):
        """Cycles are deduplicated (same tokens in different order)."""
        o1 = make_order("o1", WETH, USDC, "1000", "2000")
        o2 = make_order("o2", USDC, DAI, "2000", "2000")
        o3 = make_order("o3", DAI, WETH, "2000", "1000")
        # Add reverse direction orders too
        o4 = make_order("o4", USDC, WETH, "2000", "1000")
        o5 = make_order("o5", DAI, USDC, "2000", "2000")
        o6 = make_order("o6", WETH, DAI, "1000", "2000")
        graph = OrderGraph.from_orders([o1, o2, o3, o4, o5, o6])

        cycles = graph.find_3_cycles()
        # Should only have one unique cycle (deduplicated)
        assert len(cycles) == 1

    def test_find_4_cycles_none(self):
        """No 4-cycles in simple graph."""
        o1 = make_order("o1", WETH, USDC, "1000", "2000")
        o2 = make_order("o2", USDC, DAI, "2000", "2000")
        o3 = make_order("o3", DAI, WETH, "2000", "1000")
        graph = OrderGraph.from_orders([o1, o2, o3])

        cycles = graph.find_4_cycles()
        assert cycles == []

    def test_find_4_cycles_one(self):
        """Find a single 4-cycle."""
        o1 = make_order("o1", WETH, USDC, "1000", "2000")
        o2 = make_order("o2", USDC, DAI, "2000", "2000")
        o3 = make_order("o3", DAI, WBTC, "2000", "1")
        o4 = make_order("o4", WBTC, WETH, "1", "1000")
        graph = OrderGraph.from_orders([o1, o2, o3, o4])

        cycles = graph.find_4_cycles()
        assert len(cycles) == 1
        cycle = cycles[0]
        assert set(cycle) == {WETH.lower(), USDC.lower(), DAI.lower(), WBTC.lower()}

    def test_find_4_cycles_limit(self):
        """Respects the limit parameter."""
        # Create multiple 4-cycles would require a larger graph
        # For now just test that limit=0 returns empty
        o1 = make_order("o1", WETH, USDC, "1000", "2000")
        graph = OrderGraph.from_orders([o1])

        cycles = graph.find_4_cycles(limit=0)
        assert cycles == []


# --- CycleViability Tests ---


class TestCycleViability:
    """Tests for CycleViability dataclass."""

    def test_near_viable_true(self):
        """near_viable is True when product is slightly > 1 (1.0 < product <= 1/THRESHOLD)."""
        # product = 1.02 means traders demand 2% more than available
        # This is close enough that an AMM could fill the gap
        viability = CycleViability(
            viable=False,
            surplus_ratio=-0.02,  # Negative means deficit
            product=1.02,
            orders=[],
        )
        assert viability.near_viable is True

    def test_near_viable_false_when_viable(self):
        """near_viable is False when already viable."""
        viability = CycleViability(
            viable=True,
            surplus_ratio=0.0,  # No surplus at exact match
            product=1.0,
            orders=[],
        )
        assert viability.near_viable is False

    def test_near_viable_false_when_too_high(self):
        """near_viable is False when product too high (gap too large)."""
        # product = 1.5 means traders demand 50% more than available
        # Too large a gap for AMM to fill profitably
        viability = CycleViability(
            viable=False,
            surplus_ratio=-0.5,  # 50% deficit
            product=1.5,
            orders=[],
        )
        assert viability.near_viable is False


# --- check_cycle_viability Tests ---


class TestCheckCycleViability:
    """Tests for check_cycle_viability function."""

    def test_viable_cycle(self):
        """Viable when product <= 1 (traders are collectively generous)."""
        # WETH->USDC: sell 1 WETH, want 2000 USDC (rate = 2000/1 = 2000)
        # USDC->DAI: sell 2000 USDC, want 2000 DAI (rate = 2000/2000 = 1.0)
        # DAI->WETH: sell 2000 DAI, want 1 WETH (rate = 1/2000 = 0.0005)
        # Product = 2000 * 1.0 * 0.0005 = 1.0 <= 1 (exactly viable!)
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000000")
        o2 = make_order("o2", USDC, DAI, "2000000000000", "2000000000000000000000")
        o3 = make_order("o3", DAI, WETH, "2000000000000000000000", "1000000000000000000")
        graph = OrderGraph.from_orders([o1, o2, o3])

        cycle = (WETH.lower(), USDC.lower(), DAI.lower())
        result = check_cycle_viability(cycle, graph.get_orders)

        assert result.viable is True
        assert result.product <= 1.0
        assert result.surplus_ratio >= 0.0  # surplus_ratio = 1 - product >= 0
        assert len(result.orders) == 3

    def test_non_viable_cycle(self):
        """Non-viable when product > 1/THRESHOLD (traders demand too much)."""
        # Create a cycle where traders demand significantly more than available
        # WETH->USDC: sell 1 WETH, want 3000 USDC (rate = 3000)
        # USDC->DAI: sell 1000 USDC, want 1500 DAI (rate = 1.5)
        # DAI->WETH: sell 1000 DAI, want 1 WETH (rate = 0.001)
        # Product = 3000 * 1.5 * 0.001 = 4.5 >> 1 (not viable)
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "3000000000000")
        o2 = make_order("o2", USDC, DAI, "1000000000000", "1500000000000000000000")
        o3 = make_order("o3", DAI, WETH, "1000000000000000000000", "1000000000000000000")
        graph = OrderGraph.from_orders([o1, o2, o3])

        cycle = (WETH.lower(), USDC.lower(), DAI.lower())
        result = check_cycle_viability(cycle, graph.get_orders)

        assert result.viable is False
        assert result.near_viable is False
        assert result.product > 1.0 / NEAR_VIABLE_THRESHOLD  # > ~1.053

    def test_near_viable_cycle(self):
        """Near-viable when 1.0 < product <= 1/THRESHOLD (~1.053)."""
        # Create a cycle where product is just slightly > 1
        # WETH->USDC: sell 1 WETH, want 2000 USDC (rate = 2000)
        # USDC->DAI: sell 2000 USDC, want 2000 DAI (rate = 1.0)
        # DAI->WETH: sell 2000 DAI, want 1.02 WETH (rate = 0.00051)
        # Product = 2000 * 1.0 * 0.00051 = 1.02 (near-viable!)
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000000")
        o2 = make_order("o2", USDC, DAI, "2000000000000", "2000000000000000000000")
        o3 = make_order("o3", DAI, WETH, "2000000000000000000000", "1020000000000000000")
        graph = OrderGraph.from_orders([o1, o2, o3])

        cycle = (WETH.lower(), USDC.lower(), DAI.lower())
        result = check_cycle_viability(cycle, graph.get_orders)

        assert result.viable is False
        assert result.near_viable is True
        assert result.product > 1.0
        assert result.product <= 1.0 / NEAR_VIABLE_THRESHOLD

    def test_missing_edge(self):
        """Non-viable when missing order on edge."""
        o1 = make_order("o1", WETH, USDC, "1000", "2000")
        o2 = make_order("o2", USDC, DAI, "2000", "2000")
        # Missing DAI->WETH
        graph = OrderGraph.from_orders([o1, o2])

        cycle = (WETH.lower(), USDC.lower(), DAI.lower())
        result = check_cycle_viability(cycle, graph.get_orders)

        assert result.viable is False
        assert result.product == float("inf")  # Indicates missing edge
        assert result.orders == []

    def test_zero_sell_amount(self):
        """Non-viable when order has zero sell amount."""
        o1 = make_order("o1", WETH, USDC, "0", "2000")  # Zero sell
        o2 = make_order("o2", USDC, DAI, "2000", "2000")
        o3 = make_order("o3", DAI, WETH, "2000", "1000")
        graph = OrderGraph.from_orders([o1, o2, o3])

        cycle = (WETH.lower(), USDC.lower(), DAI.lower())
        result = check_cycle_viability(cycle, graph.get_orders)

        assert result.viable is False

    def test_best_order_selected(self):
        """Selects order with LOWEST rate (most generous) on each edge."""
        # Two orders WETH->USDC: one has lower rate (more generous)
        # Lower rate = willing to accept less = more generous
        o1_greedy = make_order("o1", WETH, USDC, "1000", "2000")  # rate = 2.0 (greedy)
        o1_generous = make_order("o1b", WETH, USDC, "1000", "1500")  # rate = 1.5 (generous)
        o2 = make_order("o2", USDC, DAI, "2000", "2000")
        o3 = make_order("o3", DAI, WETH, "2000", "1000")
        graph = OrderGraph.from_orders([o1_greedy, o1_generous, o2, o3])

        cycle = (WETH.lower(), USDC.lower(), DAI.lower())
        result = check_cycle_viability(cycle, graph.get_orders)

        # Should use the more generous (lower rate) order
        assert result.orders[0].uid == make_uid("o1b")


# --- find_viable_cycle_direction Tests ---


class TestFindViableCycleDirection:
    """Tests for find_viable_cycle_direction function."""

    def test_finds_viable_direction(self):
        """Finds viable direction from sorted tokens."""
        # Create a cycle that's viable (product <= 1)
        # WETH->USDC->DAI->WETH with product = 1.0
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000000")
        o2 = make_order("o2", USDC, DAI, "2000000000000", "2000000000000000000000")
        o3 = make_order("o3", DAI, WETH, "2000000000000000000000", "1000000000000000000")
        graph = OrderGraph.from_orders([o1, o2, o3])

        # Sorted tuple may not match the viable direction
        sorted_tokens = tuple(sorted([WETH.lower(), USDC.lower(), DAI.lower()]))
        result = find_viable_cycle_direction(sorted_tokens, graph.get_orders)

        assert result is not None
        assert result.viable is True

    def test_returns_none_when_no_viable_direction(self):
        """Returns None when product >> 1 in all directions (not even near-viable)."""
        # Create orders where traders demand way too much
        # product >> 1/THRESHOLD in all directions
        o1 = make_order("o1", WETH, USDC, "1000", "5000")  # rate = 5.0
        o2 = make_order("o2", USDC, DAI, "1000", "5000")  # rate = 5.0
        o3 = make_order("o3", DAI, WETH, "1000", "5000")  # rate = 5.0
        # Product = 5.0 * 5.0 * 5.0 = 125 >> 1.053 (not viable, not near-viable)
        graph = OrderGraph.from_orders([o1, o2, o3])

        sorted_tokens = tuple(sorted([WETH.lower(), USDC.lower(), DAI.lower()]))
        result = find_viable_cycle_direction(sorted_tokens, graph.get_orders)

        # No viable direction, and product too high for near-viable
        assert result is None

    def test_returns_best_near_viable(self):
        """Returns best near-viable when no viable direction (1 < product <= 1/THRESHOLD)."""
        # Rates that give product ~1.02 (near-viable, not viable)
        # WETH->USDC: rate = 2000
        # USDC->DAI: rate = 1.0
        # DAI->WETH: rate = 0.00051 (wants 1.02 WETH per 2000 DAI)
        # Product = 2000 * 1.0 * 0.00051 = 1.02 (near-viable)
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000000")
        o2 = make_order("o2", USDC, DAI, "2000000000000", "2000000000000000000000")
        o3 = make_order("o3", DAI, WETH, "2000000000000000000000", "1020000000000000000")
        graph = OrderGraph.from_orders([o1, o2, o3])

        sorted_tokens = tuple(sorted([WETH.lower(), USDC.lower(), DAI.lower()]))
        result = find_viable_cycle_direction(sorted_tokens, graph.get_orders)

        assert result is not None
        assert result.viable is False
        assert result.near_viable is True


# --- RingTrade Tests ---


class TestRingTrade:
    """Tests for RingTrade dataclass."""

    def test_order_uids(self):
        """order_uids returns set of order UIDs."""
        o1 = make_order("uid1", WETH, USDC, "1000", "2000")
        o2 = make_order("uid2", USDC, DAI, "2000", "2100")
        o3 = make_order("uid3", DAI, WETH, "2100", "1050")

        ring = RingTrade(
            cycle=(WETH.lower(), USDC.lower(), DAI.lower()),
            orders=[o1, o2, o3],
            fills=[(1000, 2000), (2000, 2100), (2100, 1050)],
            clearing_prices={
                WETH.lower(): 10**18,
                USDC.lower(): 5 * 10**14,
                DAI.lower(): 5 * 10**14,
            },
            surplus=1000,
        )

        assert ring.order_uids == {make_uid("uid1"), make_uid("uid2"), make_uid("uid3")}

    def test_to_strategy_result(self):
        """to_strategy_result creates valid StrategyResult."""
        o1 = make_order("uid1", WETH, USDC, "1000", "2000")
        o2 = make_order("uid2", USDC, DAI, "2000", "2100")
        o3 = make_order("uid3", DAI, WETH, "2100", "1050")

        ring = RingTrade(
            cycle=(WETH.lower(), USDC.lower(), DAI.lower()),
            orders=[o1, o2, o3],
            fills=[(1000, 2000), (2000, 2100), (2100, 1050)],
            clearing_prices={
                WETH.lower(): 10**18,
                USDC.lower(): 5 * 10**14,
                DAI.lower(): 5 * 10**14,
            },
            surplus=1000,
        )

        result = ring.to_strategy_result()

        assert len(result.fills) == 3
        assert len(result.interactions) == 0  # No AMM calls
        assert result.gas == 0  # Peer-to-peer
        assert len(result.prices) == 3

        # Check fills
        assert result.fills[0].sell_filled == 1000
        assert result.fills[0].buy_filled == 2000


# --- RingTradeStrategy Tests ---


class TestRingTradeStrategy:
    """Tests for RingTradeStrategy class."""

    def test_try_solve_no_orders(self):
        """Returns None for empty auction."""
        strategy = RingTradeStrategy()
        auction = make_auction([])

        result = strategy.try_solve(auction)

        assert result is None

    def test_try_solve_no_cycles(self):
        """Returns None when no cycles exist."""
        strategy = RingTradeStrategy()
        o1 = make_order("o1", WETH, USDC, "1000", "2000")
        auction = make_auction([o1])

        result = strategy.try_solve(auction)

        assert result is None

    def test_try_solve_viable_ring(self):
        """Returns result with viable ring trade."""
        strategy = RingTradeStrategy()

        # Create viable 3-cycle (product <= 1)
        # WETH->USDC->DAI->WETH with product = 1.0
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000000")
        o2 = make_order("o2", USDC, DAI, "2000000000000", "2000000000000000000000")
        o3 = make_order("o3", DAI, WETH, "2000000000000000000000", "1000000000000000000")
        auction = make_auction([o1, o2, o3])

        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 3
        assert len(result.interactions) == 0
        assert result.gas == 0

    def test_try_solve_non_viable_ring(self):
        """Returns None when ring not viable (product >> 1)."""
        strategy = RingTradeStrategy()

        # Create non-viable 3-cycle (product >> 1, not even near-viable)
        o1 = make_order("o1", WETH, USDC, "1000", "5000")  # rate = 5.0
        o2 = make_order("o2", USDC, DAI, "1000", "5000")  # rate = 5.0
        o3 = make_order("o3", DAI, WETH, "1000", "5000")  # rate = 5.0
        # Product = 125 >> 1.053
        auction = make_auction([o1, o2, o3])

        result = strategy.try_solve(auction)

        # No viable rings
        assert result is None

    def test_select_rings_non_overlapping(self):
        """Selects non-overlapping rings by surplus."""
        strategy = RingTradeStrategy()

        # Create two overlapping rings (share an order)
        o1 = make_order("shared", WETH, USDC, "1000", "2000")
        o2 = make_order("r1_o2", USDC, DAI, "2000", "2100")
        o3 = make_order("r1_o3", DAI, WETH, "2100", "1050")
        # Second ring using same o1
        o4 = make_order("r2_o2", USDC, WBTC, "2000", "1")
        o5 = make_order("r2_o3", WBTC, WETH, "1", "1050")

        rings = [
            RingTrade(
                cycle=(WETH.lower(), USDC.lower(), DAI.lower()),
                orders=[o1, o2, o3],
                fills=[(1000, 2000), (2000, 2100), (2100, 1050)],
                clearing_prices={},
                surplus=1000,  # Lower surplus
            ),
            RingTrade(
                cycle=(WETH.lower(), USDC.lower(), WBTC.lower()),
                orders=[o1, o4, o5],  # Shares o1!
                fills=[(1000, 2000), (2000, 1), (1, 1050)],
                clearing_prices={},
                surplus=2000,  # Higher surplus - should win
            ),
        ]

        selected = strategy._select_rings(rings)

        # Only one ring selected (higher surplus wins)
        assert len(selected) == 1
        assert selected[0].surplus == 2000

    def test_tracks_near_viable(self):
        """Tracks near-viable cycles for metrics."""
        strategy = RingTradeStrategy()

        # Create near-viable cycle (1.0 < product <= 1.053)
        # Product ~1.02 (near-viable, not viable)
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000000")
        o2 = make_order("o2", USDC, DAI, "2000000000000", "2000000000000000000000")
        o3 = make_order("o3", DAI, WETH, "2000000000000000000000", "1020000000000000000")
        auction = make_auction([o1, o2, o3])

        result = strategy.try_solve(auction)

        # No viable rings, but near_viable tracked
        assert result is None
        assert strategy._near_viable_count >= 1  # Should have found at least one near-viable


class TestRingTradeSettlement:
    """Tests for ring trade settlement calculation."""

    def test_fills_respect_limit_prices(self):
        """Fill amounts respect order limit prices."""
        strategy = RingTradeStrategy()

        # Viable cycle with product = 1.0
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000000")
        o2 = make_order("o2", USDC, DAI, "2000000000000", "2000000000000000000000")
        o3 = make_order("o3", DAI, WETH, "2000000000000000000000", "1000000000000000000")
        auction = make_auction([o1, o2, o3])

        result = strategy.try_solve(auction)

        assert result is not None
        for fill in result.fills:
            order = fill.order
            sell_amt = int(order.sell_amount)
            buy_amt = int(order.buy_amount)

            # Fill should not exceed order amounts
            assert fill.sell_filled <= sell_amt
            # Fill rate should be at least as good as limit price
            if fill.sell_filled > 0:
                limit_rate = buy_amt / sell_amt
                actual_rate = fill.buy_filled / fill.sell_filled
                assert actual_rate >= limit_rate * 0.99  # Allow 1% rounding

    def test_token_conservation(self):
        """Total tokens in = total tokens out for each token."""
        strategy = RingTradeStrategy()

        # Create viable ring (product = 1.0)
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000000")
        o2 = make_order("o2", USDC, DAI, "2000000000000", "2000000000000000000000")
        o3 = make_order("o3", DAI, WETH, "2000000000000000000000", "1000000000000000000")
        auction = make_auction([o1, o2, o3])

        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 3

        # In a ring: what's sold by one = what's bought by previous
        # o1 sells WETH, o3 buys WETH
        # o2 sells USDC, o1 buys USDC
        # o3 sells DAI, o2 buys DAI

        weth_sold = result.fills[0].sell_filled
        weth_bought = result.fills[2].buy_filled

        usdc_sold = result.fills[1].sell_filled
        usdc_bought = result.fills[0].buy_filled

        dai_sold = result.fills[2].sell_filled
        dai_bought = result.fills[1].buy_filled

        # Conservation should be exact (by construction)
        assert weth_sold == weth_bought
        assert usdc_sold == usdc_bought
        assert dai_sold == dai_bought
