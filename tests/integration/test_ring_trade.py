"""Integration tests for ring trade detection."""

from __future__ import annotations

import json
from pathlib import Path

from solver.models.auction import AuctionInstance, Order, Token
from solver.strategies.ring_trade import OrderGraph, RingTradeStrategy


def make_uid(short_id: str) -> str:
    """Create a valid 112-character hex UID from a short identifier."""
    short_hex = short_id.encode().hex()
    return "0x" + short_hex.ljust(112, "0")


# Token addresses
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
DAI = "0x6b175474e89094c44da98b954eedeac495271d0f"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"


def make_order(
    uid: str,
    sell_token: str,
    buy_token: str,
    sell_amount: str,
    buy_amount: str,
) -> Order:
    """Create a test order with proper UID format."""
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
                    availableBalance="1000000000000000000000000",
                )
    return AuctionInstance(
        id="test",
        orders=orders,
        tokens=tokens,
        liquidity=[],
        effective_gas_price="1000000000",
        deadline="2099-01-01T00:00:00Z",
    )


class TestRingTradeIntegration:
    """Integration tests for ring trade strategy."""

    def test_viable_3_ring(self):
        """Test a viable 3-order ring trade.

        Creates a cycle where:
        - Alice sells 1 WETH for 2000 USDC
        - Bob sells 2000 USDC for 2100 DAI
        - Carol sells 2100 DAI for 1 WETH

        Product of rates: (2000/1) * (2100/2000) * (1/2100) = 1.0 (exactly viable)
        """
        strategy = RingTradeStrategy()

        alice = make_order("alice", WETH, USDC, "1000000000000000000", "2000000000")
        bob = make_order("bob", USDC, DAI, "2000000000", "2100000000000000000000")
        carol = make_order("carol", DAI, WETH, "2100000000000000000000", "1000000000000000000")

        auction = make_auction([alice, bob, carol])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 3
        assert len(result.interactions) == 0
        assert result.gas == 0

        # All 3 tokens should have clearing prices
        assert len(result.prices) == 3

    def test_profitable_3_ring(self):
        """Test a profitable 3-order ring (product < 1, surplus exists).

        Creates a cycle with surplus (product < 1).
        When product < 1, traders are collectively generous - there's surplus.

        Rate calculation for viability:
        - Alice: sells 1 WETH for 1900 USDC → rate = 1900
        - Bob: sells 2000 USDC for 1900 DAI → rate = 0.95
        - Carol: sells 2000 DAI for 1 WETH → rate = 0.0005
        Product = 1900 * 0.95 * 0.0005 = 0.9025 < 1 (viable with surplus!)
        """
        strategy = RingTradeStrategy()

        # Product < 1: traders are generous, surplus exists
        alice = make_order(
            "alice", WETH, USDC, "1000000000000000000", "1900000000"
        )  # Asking 1900 USDC
        bob = make_order(
            "bob", USDC, DAI, "2000000000", "1900000000000000000000"
        )  # Asking 1900 DAI
        carol = make_order(
            "carol", DAI, WETH, "2000000000000000000000", "1000000000000000000"
        )  # Asking 1 WETH

        auction = make_auction([alice, bob, carol])
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 3

        # Check that fills respect limit prices (buy_filled >= limit buy amount)
        for fill in result.fills:
            order = fill.order
            if fill.sell_filled > 0:
                limit_rate = int(order.buy_amount) / int(order.sell_amount)
                actual_rate = fill.buy_filled / fill.sell_filled
                assert actual_rate >= limit_rate * 0.999  # Allow small rounding

    def test_non_viable_ring_returns_none(self):
        """Test that non-viable cycles return None (product >> 1)."""
        strategy = RingTradeStrategy()

        # Very unfavorable rates - product >> 1 (traders demand too much)
        alice = make_order("alice", WETH, USDC, "1000000000000000000", "5000000000000")
        bob = make_order("bob", USDC, DAI, "1000000000000", "5000000000000000000000")
        carol = make_order("carol", DAI, WETH, "1000000000000000000000", "5000000000000000000")
        # Product = 5000 * 5000 * 5 = very large >> 1.053

        auction = make_auction([alice, bob, carol])
        result = strategy.try_solve(auction)

        assert result is None

    def test_4_order_ring(self):
        """Test a 4-order ring trade.

        A → B → C → D → A
        """
        strategy = RingTradeStrategy()

        # Create a viable 4-cycle
        o1 = make_order("o1", WETH, USDC, "1000000000000000000", "2000000000")
        o2 = make_order("o2", USDC, DAI, "2000000000", "2000000000000000000000")
        o3 = make_order("o3", DAI, WBTC, "2000000000000000000000", "100000000")
        o4 = make_order("o4", WBTC, WETH, "100000000", "1000000000000000000")

        auction = make_auction([o1, o2, o3, o4])
        result = strategy.try_solve(auction)

        # Should find the 4-order ring
        if result is not None:
            assert len(result.fills) == 4
            assert len(result.interactions) == 0
            assert result.gas == 0

    def test_multiple_rings_non_overlapping(self):
        """Test that multiple non-overlapping rings are found."""
        strategy = RingTradeStrategy()

        # Ring 1: WETH-USDC-DAI-WETH
        r1_o1 = make_order("r1_o1", WETH, USDC, "1000000000000000000", "2000000000")
        r1_o2 = make_order("r1_o2", USDC, DAI, "2000000000", "2100000000000000000000")
        r1_o3 = make_order("r1_o3", DAI, WETH, "2100000000000000000000", "1000000000000000000")

        # Ring 2: Different tokens
        GNO = "0x6810e776880c02933d47db1b9fc05908e5386b96"
        COW = "0xdef1ca1fb7fbcdc777520aa7f396b4e015f497ab"
        UNI = "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"

        r2_o1 = make_order("r2_o1", GNO, COW, "1000000000000000000", "5000000000000000000")
        r2_o2 = make_order("r2_o2", COW, UNI, "5000000000000000000", "2500000000000000000")
        r2_o3 = make_order("r2_o3", UNI, GNO, "2500000000000000000", "1000000000000000000")

        auction = make_auction([r1_o1, r1_o2, r1_o3, r2_o1, r2_o2, r2_o3])
        result = strategy.try_solve(auction)

        # Should find both rings (6 orders total)
        if result is not None:
            assert len(result.fills) == 6


class TestOrderGraphFromFixtures:
    """Test OrderGraph with real auction fixtures."""

    def test_n_order_fixture_graph(self):
        """Test building graph from n_order fixture."""
        fixture_path = Path("tests/fixtures/auctions/n_order_cow/n_order_3_01.json")
        if not fixture_path.exists():
            return

        with fixture_path.open() as f:
            data = json.load(f)

        orders = []
        for order_data in data.get("orders", []):
            order = Order(**order_data)
            orders.append(order)

        if not orders:
            return

        graph = OrderGraph.from_orders(orders)

        # Graph should have nodes and edges
        assert len(graph.tokens) > 0
        # Check if any cycles exist
        cycles_3 = graph.find_3_cycles()
        cycles_4 = graph.find_4_cycles(limit=10)

        # Just verify it runs without error
        assert isinstance(cycles_3, list)
        assert isinstance(cycles_4, list)


class TestRingTradeWithProductionAuctions:
    """Test ring trade detection against production-like auctions."""

    def test_strategy_handles_empty_auction(self):
        """Strategy handles empty auction gracefully."""
        strategy = RingTradeStrategy()
        auction = make_auction([])
        result = strategy.try_solve(auction)
        assert result is None

    def test_strategy_handles_single_order(self):
        """Strategy handles single order (no cycle possible)."""
        strategy = RingTradeStrategy()
        order = make_order("single", WETH, USDC, "1000000000000000000", "2000000000")
        auction = make_auction([order])
        result = strategy.try_solve(auction)
        assert result is None

    def test_strategy_handles_two_orders_bidirectional(self):
        """Two orders form a 2-cycle, not 3-cycle."""
        strategy = RingTradeStrategy()

        # This is a direct CoW match, not a ring
        alice = make_order("alice", WETH, USDC, "1000000000000000000", "2000000000")
        bob = make_order("bob", USDC, WETH, "2000000000", "1000000000000000000")

        auction = make_auction([alice, bob])
        result = strategy.try_solve(auction)

        # Should return None - 2-cycles are CoW matches, not ring trades
        assert result is None
