"""Tests for strategy base module - OrderFill, StrategyResult, and fee calculation."""

import pytest

from solver.fees.config import DEFAULT_FEE_CONFIG
from solver.models.auction import (
    AuctionInstance,
    Order,
    OrderClass,
    OrderKind,
    Token,
)
from solver.strategies.base import OrderFill, StrategyResult

# --- Test Fixtures ---


def make_order(
    *,
    sell_token: str = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    buy_token: str = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    sell_amount: str = "1000000000",  # 1000 USDC
    buy_amount: str = "400000000000000000",  # 0.4 WETH
    kind: OrderKind = OrderKind.SELL,
    order_class: OrderClass = OrderClass.MARKET,
    partially_fillable: bool = False,
    uid: str = "0xaa01020304050607080910111213141516171819202122232425262728293031333435363738394041424344454647484950515253545556",
) -> Order:
    """Create a test order with sensible defaults."""
    return Order(
        uid=uid,
        sellToken=sell_token,
        buyToken=buy_token,
        sellAmount=sell_amount,
        buyAmount=buy_amount,
        fullSellAmount=sell_amount,
        fullBuyAmount=buy_amount,
        kind=kind,
        **{"class": order_class},
        partiallyFillable=partially_fillable,
    )


def make_auction(
    orders: list[Order],
    gas_price: str = "15000000000",  # 15 gwei
    # Reference prices: value in wei to buy 1e18 of the token
    # For fees to be reasonable, these need to be high enough
    # that gas_cost * 1e18 / ref_price gives a small fee
    usdc_ref_price: str = "450000000000000000000000000",  # ~4.5e26 (gives ~5 USDC fee for 150k gas at 15 gwei)
    weth_ref_price: str = "1000000000000000000",  # 1e18 (1:1 with ETH)
) -> AuctionInstance:
    """Create a test auction with token reference prices."""
    return AuctionInstance(
        id="test-auction",
        tokens={
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": Token(
                decimals=6,
                symbol="USDC",
                referencePrice=usdc_ref_price,
                availableBalance="10000000000",
                trusted=True,
            ),
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": Token(
                decimals=18,
                symbol="WETH",
                referencePrice=weth_ref_price,
                availableBalance="10000000000000000000",
                trusted=True,
            ),
        },
        orders=orders,
        liquidity=[],
        effectiveGasPrice=gas_price,
    )


# --- Fee Calculation Tests ---


class TestFeeCalculation:
    """Tests for limit order fee calculation."""

    def test_market_order_no_fee(self):
        """Market orders should NOT have a fee in the trade response."""
        order = make_order(order_class=OrderClass.MARKET)
        fill = OrderFill(order=order, sell_filled=1000000000, buy_filled=400000000000000000)

        result = StrategyResult(
            fills=[fill],
            gas=150000,
            prices={
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "400000000000000000",
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "1000000000",
            },
        )

        auction = make_auction([order])
        solution = result.build_solution(auction=auction)

        assert len(solution.trades) == 1
        trade = solution.trades[0]
        assert trade.fee is None, "Market orders should not have a fee"
        assert trade.executed_amount == "1000000000", "Full amount should be executed"

    def test_limit_order_has_fee(self):
        """Limit orders MUST have a fee in the trade response."""
        order = make_order(order_class=OrderClass.LIMIT)
        fill = OrderFill(order=order, sell_filled=1000000000, buy_filled=400000000000000000)

        result = StrategyResult(
            fills=[fill],
            gas=150000,
            prices={
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "400000000000000000",
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "1000000000",
            },
        )

        auction = make_auction([order])
        solution = result.build_solution(auction=auction)

        assert len(solution.trades) == 1
        trade = solution.trades[0]
        assert trade.fee is not None, "Limit orders MUST have a fee"
        assert int(trade.fee) > 0, "Fee should be positive"

    def test_fee_formula_matches_rust(self):
        """Fee calculation should match Rust: fee = gas_cost_wei * 1e18 / reference_price."""
        order = make_order(order_class=OrderClass.LIMIT)
        sell_amount = 1000000000  # 1e9
        fill = OrderFill(order=order, sell_filled=sell_amount, buy_filled=400000000000000000)

        gas = 150000
        gas_price = 15000000000  # 15 gwei
        # Reference price must be large enough that fee < sell_amount
        # fee = gas * gas_price * 1e18 / ref_price
        # For fee = 5e6 (5 USDC): ref_price = 150000 * 15e9 * 1e18 / 5e6 = 4.5e26
        usdc_ref_price = 450000000000000000000000000  # 4.5e26

        result = StrategyResult(
            fills=[fill],
            gas=gas,
            prices={
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "400000000000000000",
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "1000000000",
            },
        )

        auction = make_auction(
            [order], gas_price=str(gas_price), usdc_ref_price=str(usdc_ref_price)
        )
        solution = result.build_solution(auction=auction)

        # Expected fee: gas_cost * 1e18 / reference_price
        gas_cost_wei = gas * gas_price
        expected_fee = (gas_cost_wei * DEFAULT_FEE_CONFIG.fee_base) // usdc_ref_price

        assert len(solution.trades) == 1, "Should have one trade when fee < order amount"
        trade = solution.trades[0]
        actual_fee = int(trade.fee)
        assert actual_fee == expected_fee, f"Expected fee {expected_fee}, got {actual_fee}"
        assert actual_fee < sell_amount, (
            f"Fee ({actual_fee}) should be less than order ({sell_amount})"
        )

    def test_sell_order_executed_amount_reduced_by_fee(self):
        """For sell orders, executed amount should be reduced by fee."""
        order = make_order(order_class=OrderClass.LIMIT, kind=OrderKind.SELL)
        sell_amount = 1000000000
        fill = OrderFill(order=order, sell_filled=sell_amount, buy_filled=400000000000000000)

        result = StrategyResult(
            fills=[fill],
            gas=150000,
            prices={
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "400000000000000000",
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "1000000000",
            },
        )

        auction = make_auction([order])
        solution = result.build_solution(auction=auction)

        trade = solution.trades[0]
        fee = int(trade.fee)
        executed = int(trade.executed_amount)

        # For sell orders: executed + fee should equal the original sell_filled
        assert executed + fee == sell_amount, (
            f"executed ({executed}) + fee ({fee}) should equal sell_amount ({sell_amount})"
        )

    def test_limit_order_without_auction_rejects_trade(self):
        """Limit orders without auction data should be rejected (can't calculate fee)."""
        order = make_order(order_class=OrderClass.LIMIT)
        fill = OrderFill(order=order, sell_filled=1000000000, buy_filled=400000000000000000)

        result = StrategyResult(fills=[fill], gas=150000, prices={})

        # Build without auction
        solution = result.build_solution(auction=None)

        # Without auction data, we can't calculate fee - trade is rejected
        assert len(solution.trades) == 0, "Trade should be rejected without auction data"

    def test_zero_gas_means_zero_fee(self):
        """Zero gas should result in zero fee for limit orders."""
        order = make_order(order_class=OrderClass.LIMIT)
        fill = OrderFill(order=order, sell_filled=1000000000, buy_filled=400000000000000000)

        result = StrategyResult(
            fills=[fill],
            gas=0,  # Zero gas
            prices={
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "400000000000000000",
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "1000000000",
            },
        )

        auction = make_auction([order])
        solution = result.build_solution(auction=auction)

        trade = solution.trades[0]
        # Zero gas means zero fee, but still a limit order
        assert trade.fee == "0" or trade.fee is None

    def test_missing_reference_price_rejects_trade(self):
        """Missing reference price should reject trade (can't calculate fee)."""
        order = make_order(order_class=OrderClass.LIMIT)
        fill = OrderFill(order=order, sell_filled=1000000000, buy_filled=400000000000000000)

        result = StrategyResult(fills=[fill], gas=150000, prices={})

        # Auction without reference price for sell token
        auction = AuctionInstance(
            id="test",
            tokens={
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": Token(
                    decimals=6,
                    symbol="USDC",
                    referencePrice=None,  # No reference price
                    availableBalance="10000000000",
                ),
            },
            orders=[order],
            liquidity=[],
            effectiveGasPrice="15000000000",
        )

        solution = result.build_solution(auction=auction)

        # Trade should be rejected when fee can't be calculated
        assert len(solution.trades) == 0, "Trade should be rejected without reference price"

    def test_fee_exceeds_order_rejects_trade(self):
        """When fee > order amount, trade should be rejected (matching Rust checked_sub behavior)."""
        order = make_order(order_class=OrderClass.LIMIT)
        fill = OrderFill(order=order, sell_filled=1000000000, buy_filled=400000000000000000)

        result = StrategyResult(
            fills=[fill],
            gas=150000,
            prices={
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "400000000000000000",
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "1000000000",
            },
        )

        # Use a very small reference price to make fee astronomically large
        # This simulates the case where fee calculation overflows
        auction = make_auction(
            [order],
            usdc_ref_price="1000000000",  # Very small ref price = huge fee
        )
        solution = result.build_solution(auction=auction)

        # Trade should be rejected - no trades in solution
        # This matches Rust's checked_sub returning None
        assert len(solution.trades) == 0, "Trade should be rejected when fee exceeds order amount"

    def test_fee_overflow_produces_no_trades(self):
        """Extreme fee overflow should produce empty solution, not crash."""
        order = make_order(order_class=OrderClass.LIMIT)
        fill = OrderFill(order=order, sell_filled=1000000000, buy_filled=400000000000000000)

        result = StrategyResult(
            fills=[fill],
            gas=999999999,  # Very high gas
            prices={},
        )

        # Tiny reference price causes fee = gas * gas_price * 1e18 / ref_price to be huge
        auction = make_auction(
            [order],
            gas_price="100000000000",  # 100 gwei
            usdc_ref_price="1",  # Minimal ref price
        )
        solution = result.build_solution(auction=auction)

        # Should not crash, should just produce no trades
        assert len(solution.trades) == 0


class TestOrderClassBehavior:
    """Tests verifying order class affects fee behavior."""

    @pytest.mark.parametrize("order_class", [OrderClass.MARKET, OrderClass.LIQUIDITY])
    def test_non_limit_orders_no_fee(self, order_class: OrderClass):
        """Non-limit orders should not have solver-determined fee."""
        order = make_order(order_class=order_class)
        fill = OrderFill(order=order, sell_filled=1000000000, buy_filled=400000000000000000)

        result = StrategyResult(fills=[fill], gas=150000, prices={})
        auction = make_auction([order])
        solution = result.build_solution(auction=auction)

        trade = solution.trades[0]
        assert trade.fee is None, f"Order class {order_class} should not have fee"

    def test_limit_order_requires_fee(self):
        """Limit orders must have fee for driver validation."""
        order = make_order(order_class=OrderClass.LIMIT)
        fill = OrderFill(order=order, sell_filled=1000000000, buy_filled=400000000000000000)

        result = StrategyResult(fills=[fill], gas=150000, prices={})
        auction = make_auction([order])
        solution = result.build_solution(auction=auction)

        trade = solution.trades[0]
        # With valid auction data, limit order MUST have fee
        assert trade.fee is not None, "Limit order must have fee"
