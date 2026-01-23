"""Tests for HybridCowStrategy."""

import pytest

from solver.amm.uniswap_v2 import UniswapV2, UniswapV2Pool
from solver.models.auction import AuctionInstance, Order, Token
from solver.pools import PoolRegistry
from solver.routing.router import SingleOrderRouter
from solver.strategies.hybrid_cow import HybridCowStrategy

# Counter for generating unique UIDs
_uid_counter = 0


def make_order(
    _name: str,
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


def make_auction(orders: list[Order]) -> AuctionInstance:
    """Create a test auction."""
    # Collect unique tokens
    tokens_dict = {}
    for order in orders:
        if order.sell_token not in tokens_dict:
            tokens_dict[order.sell_token] = Token(
                address=order.sell_token,
                decimals=18,
                symbol="TKN",
                reference_price="1000000000000000000",
                available_balance="1000000000000000000000",
            )
        if order.buy_token not in tokens_dict:
            tokens_dict[order.buy_token] = Token(
                address=order.buy_token,
                decimals=18,
                symbol="TKN",
                reference_price="1000000000000000000",
                available_balance="1000000000000000000000",
            )

    return AuctionInstance(
        id="test_auction",
        orders=orders,
        tokens=tokens_dict,
        liquidity=[],  # We'll use the router's registry instead
    )


# Test token addresses
TOKEN_A = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
TOKEN_B = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"
TOKEN_C = "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"


class TestHybridCowStrategyBasic:
    """Basic tests for HybridCowStrategy."""

    @pytest.fixture(autouse=True)
    def reset_uid_counter(self) -> None:
        """Reset UID counter before each test."""
        global _uid_counter
        _uid_counter = 0

    def test_two_order_cow_match(self) -> None:
        """Two compatible orders should match via CoW."""
        # Ask: sell 100 A, want 200 B (limit = 2 B/A)
        # Bid: sell 300 B, want 100 A (limit = 3 B/A)
        # AMM price: 2.5 B/A (between limits)
        orders = [
            make_order("ask", TOKEN_A, TOKEN_B, 100, 200),
            make_order("bid", TOKEN_B, TOKEN_A, 300, 100),
        ]
        auction = make_auction(orders)

        # Create pool with price ~2.5 B/A
        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=1000 * 10**18,
            reserve1=2500 * 10**18,  # ~2.5 B/A
        )
        registry = PoolRegistry()
        registry.add_pool(pool)
        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)

        strategy = HybridCowStrategy(router=router)
        result = strategy.try_solve(auction)

        assert result is not None
        assert len(result.fills) == 2
        # CoW match has no AMM interactions
        assert len(result.interactions) == 0
        assert result.gas == 0

    def test_no_cow_potential_returns_none(self) -> None:
        """Orders in same direction should return None (no CoW potential)."""
        orders = [
            make_order("ask1", TOKEN_A, TOKEN_B, 100, 200),
            make_order("ask2", TOKEN_A, TOKEN_B, 50, 100),
        ]
        auction = make_auction(orders)

        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=1000 * 10**18,
            reserve1=2500 * 10**18,
        )
        registry = PoolRegistry()
        registry.add_pool(pool)
        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)

        strategy = HybridCowStrategy(router=router)
        result = strategy.try_solve(auction)

        # No CoW potential - strategy returns None for AMM routing to handle
        assert result is None

    def test_partial_match_creates_remainders(self) -> None:
        """Partial CoW match should create remainder orders."""
        # Ask: sell 100 A, want 200 B (limit price = 2 B/A)
        # Bid: sell 150 B, want 50 A (limit price = 3 B/A max)
        # AMM price: ~2.4925 B/A (slightly less than 2.5 due to fee)
        #
        # At AMM price ~2.4925 B/A:
        # - Bid sells ~149 B → gets 60 A
        # - Ask sells 60 A to match → gets ~149 B
        # - Ask remainder: 40 A to sell
        orders = [
            make_order("ask", TOKEN_A, TOKEN_B, 100, 200),
            make_order("bid", TOKEN_B, TOKEN_A, 150, 50),
        ]
        auction = make_auction(orders)

        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=1000 * 10**18,
            reserve1=2500 * 10**18,
        )
        registry = PoolRegistry()
        registry.add_pool(pool)
        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)

        strategy = HybridCowStrategy(router=router)
        result = strategy.try_solve(auction)

        assert result is not None
        # Should have 2 fills (ask partially filled, bid partially filled)
        assert len(result.fills) == 2

        # Check that the ask order has a partial fill
        ask_fills = [f for f in result.fills if f.order.sell_token == TOKEN_A]
        assert len(ask_fills) == 1
        ask_fill = ask_fills[0]
        # Ask sold 60 A at AMM price ~2.4925
        assert ask_fill.sell_filled == 60
        assert ask_fill.buy_filled == 149  # 60 * 2.4925 ≈ 149

        # Check that bid has a matching fill
        bid_fills = [f for f in result.fills if f.order.sell_token == TOKEN_B]
        assert len(bid_fills) == 1
        bid_fill = bid_fills[0]
        assert bid_fill.sell_filled == 149  # Matches ask's buy_filled
        assert bid_fill.buy_filled == 60  # Matches ask's sell_filled

        # Should have remainder orders for AMM routing
        assert len(result.remainder_orders) > 0
        # The ask order should have exactly one remainder of 40 A (100 - 60)
        # No duplication - only the computed remainder, not the original order
        ask_remainders = [r for r in result.remainder_orders if r.sell_token == TOKEN_A]
        assert len(ask_remainders) == 1  # Only one remainder, not two
        assert int(ask_remainders[0].sell_amount) == 40
        assert ask_remainders[0].original_uid is not None  # Computed remainder

    def test_empty_auction_returns_none(self) -> None:
        """Empty auction should return None."""
        auction = make_auction([])
        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=PoolRegistry())

        strategy = HybridCowStrategy(router=router)
        result = strategy.try_solve(auction)

        assert result is None

    def test_single_order_returns_none(self) -> None:
        """Single order auction should return None (no CoW possible)."""
        orders = [make_order("ask", TOKEN_A, TOKEN_B, 100, 200)]
        auction = make_auction(orders)
        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=PoolRegistry())

        strategy = HybridCowStrategy(router=router)
        result = strategy.try_solve(auction)

        assert result is None


class TestHybridCowStrategyMultiOrder:
    """Tests with multiple orders."""

    @pytest.fixture(autouse=True)
    def reset_uid_counter(self) -> None:
        """Reset UID counter before each test."""
        global _uid_counter
        _uid_counter = 0

    def test_multiple_orders_same_pair(self) -> None:
        """Multiple orders on same pair should be processed together."""
        # 3 asks, 2 bids on same pair
        orders = [
            make_order("ask1", TOKEN_A, TOKEN_B, 50, 100),  # 2 B/A
            make_order("ask2", TOKEN_A, TOKEN_B, 50, 100),  # 2 B/A
            make_order("ask3", TOKEN_A, TOKEN_B, 50, 150),  # 3 B/A (above AMM)
            make_order("bid1", TOKEN_B, TOKEN_A, 200, 50),  # 4 B/A max
            make_order("bid2", TOKEN_B, TOKEN_A, 100, 30),  # 3.33 B/A max
        ]
        auction = make_auction(orders)

        pool = UniswapV2Pool(
            address="0x1111111111111111111111111111111111111111",
            token0=TOKEN_A,
            token1=TOKEN_B,
            reserve0=1000 * 10**18,
            reserve1=2500 * 10**18,  # ~2.5 B/A
        )
        registry = PoolRegistry()
        registry.add_pool(pool)
        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=registry)

        strategy = HybridCowStrategy(router=router)
        result = strategy.try_solve(auction)

        assert result is not None
        # Should have some fills
        assert len(result.fills) > 0
        # No AMM interactions for CoW matches
        assert len(result.interactions) == 0


class TestHybridCowStrategyNoLiquidity:
    """Tests when no AMM liquidity exists."""

    @pytest.fixture(autouse=True)
    def reset_uid_counter(self) -> None:
        """Reset UID counter before each test."""
        global _uid_counter
        _uid_counter = 0

    def test_no_amm_liquidity_falls_back_to_pure_auction(self) -> None:
        """Without AMM liquidity, should use pure double auction."""
        # Compatible orders that can match directly
        orders = [
            make_order("ask", TOKEN_A, TOKEN_B, 100, 200),  # 2 B/A
            make_order("bid", TOKEN_B, TOKEN_A, 300, 100),  # 3 B/A
        ]
        auction = make_auction(orders)

        # No pools registered
        router = SingleOrderRouter(amm=UniswapV2(), pool_registry=PoolRegistry())

        strategy = HybridCowStrategy(router=router)
        result = strategy.try_solve(auction)

        # Should still match using pure double auction (midpoint price)
        assert result is not None
        assert len(result.fills) == 2
