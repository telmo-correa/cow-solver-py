"""Coincidence of Wants (CoW) matching strategy.

Detects when orders can be settled directly against each other without
needing AMM liquidity, saving users from swap fees.

This module uses a data-driven approach where matching rules are defined
in matching_rules.py as data structures, making the logic:
- Easy to audit and verify
- Self-documenting
- Testable in isolation
"""

from dataclasses import dataclass

import structlog

from solver.models.auction import AuctionInstance, Order
from solver.models.types import normalize_address
from solver.strategies.base import OrderFill, StrategyResult
from solver.strategies.matching_rules import (
    ExecutionAmounts,
    OrderAmounts,
    evaluate_partial_match,
    evaluate_perfect_match,
)

logger = structlog.get_logger()


@dataclass
class CowMatch:
    """A matched pair of orders that can settle directly.

    Attributes:
        order_a: First order (sells token X, buys token Y)
        order_b: Second order (sells token Y, buys token X)
        exec_sell_a: Amount of X that A sells
        exec_buy_a: Amount of Y that A receives
        exec_sell_b: Amount of Y that B sells
        exec_buy_b: Amount of X that B receives
    """

    order_a: Order
    order_b: Order
    exec_sell_a: int
    exec_buy_a: int
    exec_sell_b: int
    exec_buy_b: int

    @classmethod
    def from_execution(cls, order_a: Order, order_b: Order, exec: ExecutionAmounts) -> "CowMatch":
        """Create a CowMatch from execution amounts."""
        return cls(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=exec.exec_sell_a,
            exec_buy_a=exec.exec_buy_a,
            exec_sell_b=exec.exec_sell_b,
            exec_buy_b=exec.exec_buy_b,
        )


class CowMatchStrategy:
    """Strategy that matches orders directly (Coincidence of Wants).

    This strategy detects when two orders can be settled against each other:
    - Order A sells token X, wants token Y
    - Order B sells token Y, wants token X
    - Both orders' limit prices are satisfied

    CoW matching is preferred over AMM routing because:
    - No swap fees for users
    - Better prices (users trade at their limit or better)
    - No slippage from AMM price impact

    The matching logic is defined in matching_rules.py as data structures,
    making it easy to audit and extend.
    """

    def try_solve(self, auction: AuctionInstance) -> StrategyResult | None:
        """Try to find a CoW match in the auction.

        Handles both perfect and partial matches between 2 orders.
        For partial matches, returns remainder orders for subsequent strategies.

        Fill semantics (per CoW Protocol spec):
        - partiallyFillable=true: Order can be partially filled
        - partiallyFillable=false (default): Fill-or-kill, must fill completely

        Args:
            auction: The auction to solve

        Returns:
            A StrategyResult if a CoW match is found, None otherwise
        """
        # Currently only handle 2-order auctions
        if auction.order_count != 2:
            return None

        order_a, order_b = auction.orders[0], auction.orders[1]

        # Validate token pair compatibility
        if not self._validate_cow_pair(order_a, order_b):
            return None

        # Get parsed amounts
        amounts = self._get_order_amounts(order_a, order_b)

        # Try perfect match first
        match = self._find_perfect_match(order_a, order_b, amounts)
        if match is not None:
            logger.info(
                "cow_match_found",
                match_type="perfect",
                order_a=match.order_a.uid[:18] + "...",
                order_b=match.order_b.uid[:18] + "...",
                exec_sell_a=match.exec_sell_a,
                exec_buy_a=match.exec_buy_a,
            )
            return self._build_result(match)

        # Try partial match
        match = self._find_partial_match(order_a, order_b, amounts)
        if match is not None:
            logger.info(
                "cow_match_found",
                match_type="partial",
                order_a=match.order_a.uid[:18] + "...",
                order_b=match.order_b.uid[:18] + "...",
                exec_sell_a=match.exec_sell_a,
                exec_buy_a=match.exec_buy_a,
            )
            return self._build_result(match)

        return None

    def _validate_cow_pair(self, order_a: Order, order_b: Order) -> bool:
        """Validate that two orders have opposite token directions.

        For a CoW match, A must sell what B buys and B must sell what A buys.
        """
        a_sell = normalize_address(order_a.sell_token)
        a_buy = normalize_address(order_a.buy_token)
        b_sell = normalize_address(order_b.sell_token)
        b_buy = normalize_address(order_b.buy_token)

        if not (a_sell == b_buy and a_buy == b_sell):
            logger.debug(
                "cow_no_match_tokens",
                reason="Token pairs don't match",
                a_sells=a_sell[-8:],
                a_buys=a_buy[-8:],
                b_sells=b_sell[-8:],
                b_buys=b_buy[-8:],
            )
            return False

        return True

    def _get_order_amounts(self, order_a: Order, order_b: Order) -> OrderAmounts:
        """Extract order amounts and types for matching calculations."""
        return OrderAmounts(
            sell_a=order_a.sell_amount_int,
            buy_a=order_a.buy_amount_int,
            sell_b=order_b.sell_amount_int,
            buy_b=order_b.buy_amount_int,
            a_is_sell=order_a.is_sell_order,
            b_is_sell=order_b.is_sell_order,
        )

    def _find_perfect_match(
        self, order_a: Order, order_b: Order, amounts: OrderAmounts
    ) -> CowMatch | None:
        """Check if two orders form a perfect CoW match.

        Uses the rule-based evaluation from matching_rules.py.
        """
        exec_amounts = evaluate_perfect_match(amounts)
        if exec_amounts is None:
            return None

        return CowMatch.from_execution(order_a, order_b, exec_amounts)

    def _find_partial_match(
        self, order_a: Order, order_b: Order, amounts: OrderAmounts
    ) -> CowMatch | None:
        """Find a partial CoW match between two orders.

        Uses the rule-based evaluation from matching_rules.py.
        Checks fill-or-kill constraints for partially filled orders.
        """
        result = evaluate_partial_match(amounts)
        if result is None:
            return None

        exec_amounts, a_is_partial, b_is_partial = result

        # Check fill-or-kill constraints
        if a_is_partial and not order_a.partially_fillable:
            logger.debug(
                "cow_partial_rejected_fill_or_kill",
                reason="Order A would be partially filled but is fill-or-kill",
                order_uid=order_a.uid[:18] + "...",
            )
            return None

        if b_is_partial and not order_b.partially_fillable:
            logger.debug(
                "cow_partial_rejected_fill_or_kill",
                reason="Order B would be partially filled but is fill-or-kill",
                order_uid=order_b.uid[:18] + "...",
            )
            return None

        logger.debug(
            "cow_partial_match_computed",
            exec_sell_a=exec_amounts.exec_sell_a,
            exec_buy_a=exec_amounts.exec_buy_a,
            a_is_partial=a_is_partial,
            b_is_partial=b_is_partial,
        )

        return CowMatch.from_execution(order_a, order_b, exec_amounts)

    def _build_result(self, match: CowMatch) -> StrategyResult:
        """Build a StrategyResult from a CoW match."""
        # Normalize token addresses
        token_x = normalize_address(match.order_a.sell_token)  # A sells, B buys
        token_y = normalize_address(match.order_a.buy_token)  # A buys, B sells

        # Clearing prices: set so that the exchange rate equals the actual trade
        prices = {
            token_x: str(match.exec_sell_b),
            token_y: str(match.exec_sell_a),
        }

        # Create fills for both orders
        fill_a = OrderFill(
            order=match.order_a,
            sell_filled=match.exec_sell_a,
            buy_filled=match.exec_buy_a,
        )
        fill_b = OrderFill(
            order=match.order_b,
            sell_filled=match.exec_sell_b,
            buy_filled=match.exec_buy_b,
        )

        # Compute remainder orders for any unfilled portions
        remainder_orders = []
        remainder_a = fill_a.get_remainder_order()
        remainder_b = fill_b.get_remainder_order()
        if remainder_a:
            remainder_orders.append(remainder_a)
        if remainder_b:
            remainder_orders.append(remainder_b)

        # No AMM interactions needed - pure peer-to-peer settlement
        return StrategyResult(
            fills=[fill_a, fill_b],
            interactions=[],
            prices=prices,
            gas=0,  # No on-chain swaps, just transfers
            remainder_orders=remainder_orders,
        )
