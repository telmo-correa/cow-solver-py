"""Coincidence of Wants (CoW) matching strategy.

Detects when orders can be settled directly against each other without
needing AMM liquidity, saving users from swap fees.
"""

from dataclasses import dataclass
from typing import NamedTuple

import structlog

from solver.models.auction import AuctionInstance, Order
from solver.models.types import normalize_address
from solver.strategies.base import OrderFill, StrategyResult

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


class OrderAmounts(NamedTuple):
    """Parsed order amounts for CoW matching calculations."""

    sell_a: int
    buy_a: int
    sell_b: int
    buy_b: int
    a_is_sell: bool
    b_is_sell: bool


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
    """

    def try_solve(self, auction: AuctionInstance) -> StrategyResult | None:
        """Try to find a CoW match in the auction.

        Handles both perfect and partial matches between 2 orders.
        For partial matches, returns remainder orders for subsequent strategies.

        Fill semantics (per CoW Protocol spec):
        - partiallyFillable=true: Order can be partially filled
        - partiallyFillable=false (default): Fill-or-kill, must fill completely or not at all

        Perfect matches work for any order type since both orders are fully filled.
        Partial matches require that the order being partially filled has
        partiallyFillable=true. A fill-or-kill order CAN participate in a partial
        match if it gets completely filled (the other order has the remainder).

        Args:
            auction: The auction to solve

        Returns:
            A StrategyResult if a CoW match is found (full or partial), None otherwise
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

        # Try partial match (supports all order type combinations)
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

        Args:
            order_a: First order
            order_b: Second order

        Returns:
            True if tokens are compatible for CoW matching, False otherwise
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
        """Extract order amounts and types for matching calculations.

        Args:
            order_a: First order
            order_b: Second order

        Returns:
            OrderAmounts tuple with all relevant values
        """
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

        A perfect match requires both limit prices satisfied when fully executed.

        Args:
            order_a: First order
            order_b: Second order
            amounts: Pre-parsed order amounts

        Returns:
            CowMatch if orders match, None otherwise
        """
        if amounts.a_is_sell and amounts.b_is_sell:
            return self._match_sell_sell(order_a, order_b, amounts)
        elif amounts.a_is_sell and not amounts.b_is_sell:
            return self._match_sell_buy(order_a, order_b, amounts)
        elif not amounts.a_is_sell and amounts.b_is_sell:
            return self._match_buy_sell(order_a, order_b, amounts)
        else:
            return self._match_buy_buy(order_a, order_b, amounts)

    def _find_partial_match(
        self, order_a: Order, order_b: Order, amounts: OrderAmounts
    ) -> CowMatch | None:
        """Find a partial CoW match between two orders.

        A partial match occurs when:
        1. Both limit prices can be satisfied
        2. One order can be completely filled while the other is partially filled
        3. The partially-filled order must have partiallyFillable=true

        Args:
            order_a: First order
            order_b: Second order
            amounts: Pre-parsed order amounts

        Returns:
            CowMatch for partial execution, None if no partial match possible
        """
        if amounts.a_is_sell and amounts.b_is_sell:
            return self._partial_match_sell_sell(order_a, order_b, amounts)
        elif amounts.a_is_sell and not amounts.b_is_sell:
            return self._partial_match_sell_buy(order_a, order_b, amounts)
        elif not amounts.a_is_sell and amounts.b_is_sell:
            return self._partial_match_buy_sell(order_a, order_b, amounts)
        else:
            return self._partial_match_buy_buy(order_a, order_b, amounts)

    # =========================================================================
    # Perfect Match Methods
    # =========================================================================

    def _match_sell_sell(
        self, order_a: Order, order_b: Order, amounts: OrderAmounts
    ) -> CowMatch | None:
        """Match two sell orders.

        Both orders fully sell their amounts:
        - A sells sell_a of X, receives sell_b of Y
        - B sells sell_b of Y, receives sell_a of X

        Constraints:
        - A receives >= buy_a: sell_b >= buy_a
        - B receives >= buy_b: sell_a >= buy_b
        """
        if amounts.sell_b < amounts.buy_a:
            logger.debug(
                "cow_no_match_limit_a",
                reason="Order A's limit price not satisfied",
                a_wants=amounts.buy_a,
                b_offers=amounts.sell_b,
            )
            return None

        if amounts.sell_a < amounts.buy_b:
            logger.debug(
                "cow_no_match_limit_b",
                reason="Order B's limit price not satisfied",
                b_wants=amounts.buy_b,
                a_offers=amounts.sell_a,
            )
            return None

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=amounts.sell_a,
            exec_buy_a=amounts.sell_b,
            exec_sell_b=amounts.sell_b,
            exec_buy_b=amounts.sell_a,
        )

    def _match_sell_buy(
        self, order_a: Order, order_b: Order, amounts: OrderAmounts
    ) -> CowMatch | None:
        """Match a sell order (A) with a buy order (B).

        For full execution of both:
        - A sells sell_a of X, B buys buy_b of X (must equal sell_a)
        - A receives sell_b of Y (B's max payment)

        Constraints:
        - Amounts must match: sell_a == buy_b
        - A receives >= buy_a: sell_b >= buy_a
        """
        if amounts.sell_a != amounts.buy_b:
            logger.debug(
                "cow_no_match_amounts",
                reason="Sell amount doesn't match buy amount for sell-buy match",
                a_sells=amounts.sell_a,
                b_wants=amounts.buy_b,
            )
            return None

        if amounts.sell_b < amounts.buy_a:
            logger.debug(
                "cow_no_match_limit_a",
                reason="Order A's limit price not satisfied",
                a_wants=amounts.buy_a,
                b_max_payment=amounts.sell_b,
            )
            return None

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=amounts.sell_a,
            exec_buy_a=amounts.sell_b,
            exec_sell_b=amounts.sell_b,
            exec_buy_b=amounts.buy_b,
        )

    def _match_buy_sell(
        self, order_a: Order, order_b: Order, amounts: OrderAmounts
    ) -> CowMatch | None:
        """Match a buy order (A) with a sell order (B).

        For full execution of both:
        - A buys buy_a of Y (must equal sell_b), B sells sell_b of Y
        - A pays sell_a of X (A's max payment), B receives sell_a of X

        Constraints:
        - Amounts must match: buy_a == sell_b
        - B receives >= buy_b: sell_a >= buy_b
        """
        if amounts.buy_a != amounts.sell_b:
            logger.debug(
                "cow_no_match_amounts",
                reason="Buy amount doesn't match sell amount for buy-sell match",
                a_wants=amounts.buy_a,
                b_sells=amounts.sell_b,
            )
            return None

        if amounts.sell_a < amounts.buy_b:
            logger.debug(
                "cow_no_match_limit_b",
                reason="Order B's limit price not satisfied",
                b_wants=amounts.buy_b,
                a_max_payment=amounts.sell_a,
            )
            return None

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=amounts.sell_a,
            exec_buy_a=amounts.buy_a,
            exec_sell_b=amounts.sell_b,
            exec_buy_b=amounts.sell_a,
        )

    def _match_buy_buy(
        self, order_a: Order, order_b: Order, amounts: OrderAmounts
    ) -> CowMatch | None:
        """Match two buy orders.

        Both orders get exactly what they want:
        - A buys buy_a of Y, pays buy_b of X (what B wants)
        - B buys buy_b of X, pays buy_a of Y (what A wants)

        Constraints:
        - A can afford B's want: buy_b <= sell_a
        - B can afford A's want: buy_a <= sell_b
        """
        if amounts.buy_b > amounts.sell_a:
            logger.debug(
                "cow_no_match_limit_a",
                reason="Order A can't afford what B wants",
                b_wants=amounts.buy_b,
                a_max_payment=amounts.sell_a,
            )
            return None

        if amounts.buy_a > amounts.sell_b:
            logger.debug(
                "cow_no_match_limit_b",
                reason="Order B can't afford what A wants",
                a_wants=amounts.buy_a,
                b_max_payment=amounts.sell_b,
            )
            return None

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=amounts.buy_b,
            exec_buy_a=amounts.buy_a,
            exec_sell_b=amounts.buy_a,
            exec_buy_b=amounts.buy_b,
        )

    # =========================================================================
    # Partial Match Methods
    # =========================================================================

    def _check_fill_or_kill(self, order: Order, is_partial: bool) -> bool:
        """Check if a fill-or-kill order would be partially filled.

        Args:
            order: The order to check
            is_partial: True if this order would be partially filled

        Returns:
            True if the fill is allowed, False if rejected
        """
        if is_partial and not order.partially_fillable:
            logger.debug(
                "cow_partial_rejected_fill_or_kill",
                reason="Order would be partially filled but is fill-or-kill",
                order_uid=order.uid[:18] + "...",
            )
            return False
        return True

    def _partial_match_sell_sell(
        self, order_a: Order, order_b: Order, amounts: OrderAmounts
    ) -> CowMatch | None:
        """Find partial CoW match for two sell orders.

        For sell-sell partial matching:
        - A sells sell_a of X, wants at least buy_a of Y
        - B sells sell_b of Y, wants at least buy_b of X

        Partial match when limits are compatible but amounts don't fully match.
        """
        # Check if limits are compatible
        # A's limit: Y/X >= buy_a / sell_a
        # B's limit: Y/X <= sell_b / buy_b
        # Compatible if: buy_a * buy_b <= sell_a * sell_b
        if amounts.buy_a * amounts.buy_b > amounts.sell_a * amounts.sell_b:
            logger.debug(
                "cow_partial_no_match_limits",
                reason="Limit prices not compatible",
            )
            return None

        # CoW amount of X = min(sell_a, buy_b)
        cow_x = min(amounts.sell_a, amounts.buy_b)

        # If cow_x equals both, it's a perfect match (handled elsewhere)
        if cow_x == amounts.sell_a and cow_x == amounts.buy_b:
            return None

        # Determine which order would be partially filled
        a_is_partial = cow_x == amounts.buy_b  # B is fully filled, A is partial
        b_is_partial = cow_x == amounts.sell_a  # A is fully filled, B is partial

        if not self._check_fill_or_kill(order_a, a_is_partial):
            return None
        if not self._check_fill_or_kill(order_b, b_is_partial):
            return None

        # Calculate proportional Y for the CoW match
        if cow_x == amounts.buy_b:
            # B is fully satisfied, B gives all their Y
            cow_y = amounts.sell_b
        else:
            # A is fully satisfied, receives proportional Y using B's rate
            cow_y = (cow_x * amounts.sell_b) // amounts.buy_b

        # Verify A's limit is satisfied (ceiling division for min requirements)
        a_min_receive = (cow_x * amounts.buy_a + amounts.sell_a - 1) // amounts.sell_a
        if cow_y < a_min_receive:
            logger.debug(
                "cow_partial_no_match_a_limit",
                reason="Partial fill doesn't satisfy A's limit",
                cow_y=cow_y,
                a_min_receive=a_min_receive,
            )
            return None

        # Verify B's limit is satisfied
        b_min_receive = (cow_y * amounts.buy_b + amounts.sell_b - 1) // amounts.sell_b
        if cow_x < b_min_receive:
            logger.debug(
                "cow_partial_no_match_b_limit",
                reason="Partial fill doesn't satisfy B's limit",
                cow_x=cow_x,
                b_min_receive=b_min_receive,
            )
            return None

        logger.debug(
            "cow_partial_match_found",
            cow_x=cow_x,
            cow_y=cow_y,
            a_remainder=amounts.sell_a - cow_x,
            b_remainder=amounts.sell_b - cow_y,
        )

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=cow_x,
            exec_buy_a=cow_y,
            exec_sell_b=cow_y,
            exec_buy_b=cow_x,
        )

    def _partial_match_sell_buy(
        self, order_a: Order, order_b: Order, amounts: OrderAmounts
    ) -> CowMatch | None:
        """Find partial CoW match for sell order A and buy order B.

        A (sell): sells sell_a of X, wants min buy_a of Y
        B (buy): wants exactly buy_b of X, pays up to sell_b of Y

        Partial match when A offers more X than B wants.
        """
        # Only partial if A offers MORE than B wants
        if amounts.sell_a <= amounts.buy_b:
            return None

        # A would be partially filled
        if not self._check_fill_or_kill(order_a, is_partial=True):
            return None

        # Check if A's limit can be satisfied by B's payment
        # sell_b * sell_a >= buy_b * buy_a
        if amounts.sell_b * amounts.sell_a < amounts.buy_b * amounts.buy_a:
            logger.debug(
                "cow_partial_sell_buy_limit_not_met",
                reason="A's limit not satisfied by B's max payment",
            )
            return None

        cow_x = amounts.buy_b  # X transferred (A→B)
        cow_y = amounts.sell_b  # Y transferred (B→A)

        logger.debug(
            "cow_partial_match_found",
            match_type="sell_buy",
            cow_x=cow_x,
            cow_y=cow_y,
            a_remainder=amounts.sell_a - cow_x,
        )

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=cow_x,
            exec_buy_a=cow_y,
            exec_sell_b=cow_y,
            exec_buy_b=cow_x,
        )

    def _partial_match_buy_sell(
        self, order_a: Order, order_b: Order, amounts: OrderAmounts
    ) -> CowMatch | None:
        """Find partial CoW match for buy order A and sell order B.

        A (buy): wants exactly buy_a of Y, pays up to sell_a of X
        B (sell): sells sell_b of Y, wants min buy_b of X

        Partial match when B offers more Y than A wants.
        """
        # Only partial if B offers MORE than A wants
        if amounts.sell_b <= amounts.buy_a:
            return None

        # B would be partially filled
        if not self._check_fill_or_kill(order_b, is_partial=True):
            return None

        # Check if B's limit can be satisfied by A's payment
        # sell_a * sell_b >= buy_a * buy_b
        if amounts.sell_a * amounts.sell_b < amounts.buy_a * amounts.buy_b:
            logger.debug(
                "cow_partial_buy_sell_limit_not_met",
                reason="B's limit not satisfied by A's max payment",
            )
            return None

        cow_x = amounts.sell_a  # X transferred (A→B)
        cow_y = amounts.buy_a  # Y transferred (B→A)

        logger.debug(
            "cow_partial_match_found",
            match_type="buy_sell",
            cow_x=cow_x,
            cow_y=cow_y,
            b_remainder=amounts.sell_b - cow_y,
        )

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=cow_x,
            exec_buy_a=cow_y,
            exec_sell_b=cow_y,
            exec_buy_b=cow_x,
        )

    def _partial_match_buy_buy(
        self, order_a: Order, order_b: Order, amounts: OrderAmounts
    ) -> CowMatch | None:
        """Find partial CoW match for two buy orders.

        A (buy): wants exactly buy_a of Y, pays up to sell_a of X
        B (buy): wants exactly buy_b of X, pays up to sell_b of Y

        Partial match when one can fully satisfy the other but not vice versa.
        """
        a_can_satisfy_b = amounts.sell_a >= amounts.buy_b
        b_can_satisfy_a = amounts.sell_b >= amounts.buy_a

        # If both or neither can satisfy, no partial match
        if a_can_satisfy_b == b_can_satisfy_a:
            return None

        if a_can_satisfy_b:
            # Case 1: B gets complete fill, A gets partial fill
            if not self._check_fill_or_kill(order_a, is_partial=True):
                return None

            cow_x = amounts.buy_b  # X transferred (A→B)
            cow_y = amounts.sell_b  # Y transferred (B→A)

            logger.debug(
                "cow_partial_match_found",
                match_type="buy_buy_b_complete",
                cow_x=cow_x,
                cow_y=cow_y,
                a_receives=cow_y,
                a_wanted=amounts.buy_a,
            )
        else:
            # Case 2: A gets complete fill, B gets partial fill
            if not self._check_fill_or_kill(order_b, is_partial=True):
                return None

            cow_x = amounts.sell_a  # X transferred (A→B)
            cow_y = amounts.buy_a  # Y transferred (B→A)

            logger.debug(
                "cow_partial_match_found",
                match_type="buy_buy_a_complete",
                cow_x=cow_x,
                cow_y=cow_y,
                b_receives=cow_x,
                b_wanted=amounts.buy_b,
            )

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=cow_x,
            exec_buy_a=cow_y,
            exec_sell_b=cow_y,
            exec_buy_b=cow_x,
        )

    # =========================================================================
    # Result Building
    # =========================================================================

    def _build_result(self, match: CowMatch) -> StrategyResult:
        """Build a StrategyResult from a CoW match.

        Args:
            match: The matched order pair

        Returns:
            A StrategyResult with fills, clearing prices, and no interactions
        """
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
