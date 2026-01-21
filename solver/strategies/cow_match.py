"""Coincidence of Wants (CoW) matching strategy.

Detects when orders can be settled directly against each other without
needing AMM liquidity, saving users from swap fees.
"""

from dataclasses import dataclass

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

        # Try perfect match first
        match = self._find_perfect_match(order_a, order_b)
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
        match = self._find_partial_match(order_a, order_b)
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

    def _find_perfect_match(self, order_a: Order, order_b: Order) -> CowMatch | None:
        """Check if two orders form a perfect CoW match.

        A perfect match requires:
        1. Opposite token directions (A sells what B buys, B sells what A buys)
        2. Both limit prices satisfied when fully executed

        Supports all order type combinations:
        - sell-sell: Both orders fully sell their amounts
        - sell-buy: Sell amount must equal buy amount for full execution
        - buy-sell: Buy amount must equal sell amount for full execution
        - buy-buy: Both orders get exactly what they want

        Args:
            order_a: First order
            order_b: Second order

        Returns:
            CowMatch if orders match, None otherwise
        """
        # Normalize addresses for comparison
        a_sell = normalize_address(order_a.sell_token)
        a_buy = normalize_address(order_a.buy_token)
        b_sell = normalize_address(order_b.sell_token)
        b_buy = normalize_address(order_b.buy_token)

        # Check opposite directions: A sells X/buys Y, B sells Y/buys X
        if not (a_sell == b_buy and a_buy == b_sell):
            logger.debug(
                "cow_no_match_tokens",
                reason="Token pairs don't match",
                a_sells=a_sell[-8:],
                a_buys=a_buy[-8:],
                b_sells=b_sell[-8:],
                b_buys=b_buy[-8:],
            )
            return None

        # Parse amounts
        try:
            sell_amount_a = int(order_a.sell_amount)
            buy_amount_a = int(order_a.buy_amount)
            sell_amount_b = int(order_b.sell_amount)
            buy_amount_b = int(order_b.buy_amount)
        except (ValueError, TypeError):
            return None

        a_is_sell = order_a.is_sell_order
        b_is_sell = order_b.is_sell_order

        if a_is_sell and b_is_sell:
            return self._match_sell_sell(
                order_a,
                order_b,
                sell_amount_a,
                buy_amount_a,
                sell_amount_b,
                buy_amount_b,
            )
        elif a_is_sell and not b_is_sell:
            return self._match_sell_buy(
                order_a,
                order_b,
                sell_amount_a,
                buy_amount_a,
                sell_amount_b,
                buy_amount_b,
            )
        elif not a_is_sell and b_is_sell:
            return self._match_buy_sell(
                order_a,
                order_b,
                sell_amount_a,
                buy_amount_a,
                sell_amount_b,
                buy_amount_b,
            )
        else:
            return self._match_buy_buy(
                order_a,
                order_b,
                sell_amount_a,
                buy_amount_a,
                sell_amount_b,
                buy_amount_b,
            )

    def _match_sell_sell(
        self,
        order_a: Order,
        order_b: Order,
        sell_amount_a: int,
        buy_amount_a: int,
        sell_amount_b: int,
        buy_amount_b: int,
    ) -> CowMatch | None:
        """Match two sell orders.

        Both orders fully sell their amounts:
        - A sells sell_amount_a of X, receives sell_amount_b of Y
        - B sells sell_amount_b of Y, receives sell_amount_a of X

        Constraints:
        - A receives >= buy_amount_a: sell_amount_b >= buy_amount_a
        - B receives >= buy_amount_b: sell_amount_a >= buy_amount_b
        """
        if sell_amount_b < buy_amount_a:
            logger.debug(
                "cow_no_match_limit_a",
                reason="Order A's limit price not satisfied",
                a_wants=buy_amount_a,
                b_offers=sell_amount_b,
            )
            return None

        if sell_amount_a < buy_amount_b:
            logger.debug(
                "cow_no_match_limit_b",
                reason="Order B's limit price not satisfied",
                b_wants=buy_amount_b,
                a_offers=sell_amount_a,
            )
            return None

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=sell_amount_a,
            exec_buy_a=sell_amount_b,
            exec_sell_b=sell_amount_b,
            exec_buy_b=sell_amount_a,
        )

    def _match_sell_buy(
        self,
        order_a: Order,
        order_b: Order,
        sell_amount_a: int,
        buy_amount_a: int,
        sell_amount_b: int,
        buy_amount_b: int,
    ) -> CowMatch | None:
        """Match a sell order (A) with a buy order (B).

        For full execution of both:
        - A sells sell_amount_a of X
        - B buys buy_amount_b of X (must equal sell_amount_a)
        - A receives sell_amount_b of Y (B's max payment)
        - B pays sell_amount_b of Y

        Constraints:
        - Amounts must match: sell_amount_a == buy_amount_b
        - A receives >= buy_amount_a: sell_amount_b >= buy_amount_a
        """
        if sell_amount_a != buy_amount_b:
            logger.debug(
                "cow_no_match_amounts",
                reason="Sell amount doesn't match buy amount for sell-buy match",
                a_sells=sell_amount_a,
                b_wants=buy_amount_b,
            )
            return None

        if sell_amount_b < buy_amount_a:
            logger.debug(
                "cow_no_match_limit_a",
                reason="Order A's limit price not satisfied",
                a_wants=buy_amount_a,
                b_max_payment=sell_amount_b,
            )
            return None

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=sell_amount_a,
            exec_buy_a=sell_amount_b,
            exec_sell_b=sell_amount_b,
            exec_buy_b=buy_amount_b,
        )

    def _match_buy_sell(
        self,
        order_a: Order,
        order_b: Order,
        sell_amount_a: int,
        buy_amount_a: int,
        sell_amount_b: int,
        buy_amount_b: int,
    ) -> CowMatch | None:
        """Match a buy order (A) with a sell order (B).

        For full execution of both:
        - A buys buy_amount_a of Y (must equal sell_amount_b)
        - B sells sell_amount_b of Y
        - A pays sell_amount_a of X (A's max payment)
        - B receives sell_amount_a of X

        Constraints:
        - Amounts must match: buy_amount_a == sell_amount_b
        - B receives >= buy_amount_b: sell_amount_a >= buy_amount_b
        """
        if buy_amount_a != sell_amount_b:
            logger.debug(
                "cow_no_match_amounts",
                reason="Buy amount doesn't match sell amount for buy-sell match",
                a_wants=buy_amount_a,
                b_sells=sell_amount_b,
            )
            return None

        if sell_amount_a < buy_amount_b:
            logger.debug(
                "cow_no_match_limit_b",
                reason="Order B's limit price not satisfied",
                b_wants=buy_amount_b,
                a_max_payment=sell_amount_a,
            )
            return None

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=sell_amount_a,
            exec_buy_a=buy_amount_a,
            exec_sell_b=sell_amount_b,
            exec_buy_b=sell_amount_a,
        )

    def _match_buy_buy(
        self,
        order_a: Order,
        order_b: Order,
        sell_amount_a: int,
        buy_amount_a: int,
        sell_amount_b: int,
        buy_amount_b: int,
    ) -> CowMatch | None:
        """Match two buy orders.

        Both orders get exactly what they want:
        - A buys buy_amount_a of Y, pays buy_amount_b of X (what B wants)
        - B buys buy_amount_b of X, pays buy_amount_a of Y (what A wants)

        Constraints:
        - A can afford B's want: buy_amount_b <= sell_amount_a
        - B can afford A's want: buy_amount_a <= sell_amount_b
        """
        if buy_amount_b > sell_amount_a:
            logger.debug(
                "cow_no_match_limit_a",
                reason="Order A can't afford what B wants",
                b_wants=buy_amount_b,
                a_max_payment=sell_amount_a,
            )
            return None

        if buy_amount_a > sell_amount_b:
            logger.debug(
                "cow_no_match_limit_b",
                reason="Order B can't afford what A wants",
                a_wants=buy_amount_a,
                b_max_payment=sell_amount_b,
            )
            return None

        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=buy_amount_b,
            exec_buy_a=buy_amount_a,
            exec_sell_b=buy_amount_a,
            exec_buy_b=buy_amount_b,
        )

    def _find_partial_match(self, order_a: Order, order_b: Order) -> CowMatch | None:
        """Find a partial CoW match between two orders.

        A partial match occurs when:
        1. Orders have opposite token directions
        2. Both limit prices can be satisfied
        3. One order can be completely filled while the other is partially filled
        4. The partially-filled order must have partiallyFillable=true

        Per CoW Protocol spec: orders with partiallyFillable=false are "fill-or-kill"
        and must be completely filled or not at all. However, a fill-or-kill order
        CAN participate in a partial match if IT gets completely filled (while the
        other order is partially filled).

        Supports all order type combinations:
        - sell-sell: Both orders specify exact sell amounts
        - sell-buy: A sells exact amount, B wants exact amount
        - buy-sell: A wants exact amount, B sells exact amount
        - buy-buy: Both orders want exact amounts

        Args:
            order_a: First order
            order_b: Second order

        Returns:
            CowMatch for partial execution, None if no partial match possible
        """
        # Normalize addresses for comparison
        a_sell = normalize_address(order_a.sell_token)
        a_buy = normalize_address(order_a.buy_token)
        b_sell = normalize_address(order_b.sell_token)
        b_buy = normalize_address(order_b.buy_token)

        # Check opposite directions
        if not (a_sell == b_buy and a_buy == b_sell):
            return None

        # Parse amounts
        try:
            sell_amount_a = int(order_a.sell_amount)
            buy_amount_a = int(order_a.buy_amount)
            sell_amount_b = int(order_b.sell_amount)
            buy_amount_b = int(order_b.buy_amount)
        except (ValueError, TypeError):
            return None

        a_is_sell = order_a.is_sell_order
        b_is_sell = order_b.is_sell_order

        if a_is_sell and b_is_sell:
            return self._partial_match_sell_sell(
                order_a,
                order_b,
                sell_amount_a,
                buy_amount_a,
                sell_amount_b,
                buy_amount_b,
            )
        elif a_is_sell and not b_is_sell:
            return self._partial_match_sell_buy(
                order_a,
                order_b,
                sell_amount_a,
                buy_amount_a,
                sell_amount_b,
                buy_amount_b,
            )
        elif not a_is_sell and b_is_sell:
            return self._partial_match_buy_sell(
                order_a,
                order_b,
                sell_amount_a,
                buy_amount_a,
                sell_amount_b,
                buy_amount_b,
            )
        else:
            return self._partial_match_buy_buy(
                order_a,
                order_b,
                sell_amount_a,
                buy_amount_a,
                sell_amount_b,
                buy_amount_b,
            )

    def _partial_match_sell_sell(
        self,
        order_a: Order,
        order_b: Order,
        sell_amount_a: int,
        buy_amount_a: int,
        sell_amount_b: int,
        buy_amount_b: int,
    ) -> CowMatch | None:
        """Find partial CoW match for two sell orders.

        For sell-sell partial matching:
        - A sells sell_amount_a of X, wants at least buy_amount_a of Y
        - B sells sell_amount_b of Y, wants at least buy_amount_b of X

        Partial match when limits are compatible but amounts don't fully match:
        - B wants buy_amount_b of X, A has sell_amount_a of X
        - If buy_amount_b < sell_amount_a: fill B completely, A has remainder

        The exchange rate must satisfy both limits:
        - A's limit: receives >= buy_amount_a for sell_amount_a (rate >= buy_amount_a/sell_amount_a)
        - B's limit: receives >= buy_amount_b for sell_amount_b (rate <= sell_amount_b/buy_amount_b)

        For partial match at the favorable rate for both:
        - Match buy_amount_b of X from A to B
        - B gives proportional Y: cow_y = sell_amount_b (B fully depletes Y)
        - Check if this satisfies A's limit proportionally

        Rounding behavior:
        - When calculating proportional Y (cow_y), we use floor division
        - This is conservative: A receives slightly less in edge cases
        - Verification uses ceiling division for min requirements
        - This ensures we NEVER give users less than their limit price
        - We might reject marginal matches, but never harm users
        """
        # Check if limits are compatible (can we find a valid exchange rate?)
        # A's limit: Y/X >= buy_amount_a / sell_amount_a
        # B's limit: Y/X <= sell_amount_b / buy_amount_b
        # Compatible if: buy_amount_a / sell_amount_a <= sell_amount_b / buy_amount_b
        # i.e.: buy_amount_a * buy_amount_b <= sell_amount_a * sell_amount_b
        if buy_amount_a * buy_amount_b > sell_amount_a * sell_amount_b:
            logger.debug(
                "cow_partial_no_match_limits",
                reason="Limit prices not compatible",
            )
            return None

        # Determine which order constrains the CoW volume
        # A offers sell_amount_a of X
        # B wants buy_amount_b of X
        # CoW amount of X = min(sell_amount_a, buy_amount_b)
        cow_x = min(sell_amount_a, buy_amount_b)

        # If cow_x equals both, it's a perfect match (handled elsewhere)
        if cow_x == sell_amount_a and cow_x == buy_amount_b:
            # This should have been caught by _find_perfect_match
            return None

        # Determine which order would be partially filled and check if allowed
        # If cow_x == buy_amount_b: B is fully filled, A is partial
        # If cow_x == sell_amount_a: A is fully filled, B is partial
        if cow_x == buy_amount_b and not order_a.partially_fillable:
            logger.debug(
                "cow_partial_rejected_fill_or_kill",
                reason="Order A would be partially filled but is fill-or-kill",
                order_uid=order_a.uid[:18] + "...",
            )
            return None
        if cow_x == sell_amount_a and not order_b.partially_fillable:
            logger.debug(
                "cow_partial_rejected_fill_or_kill",
                reason="Order B would be partially filled but is fill-or-kill",
                order_uid=order_b.uid[:18] + "...",
            )
            return None

        # Calculate proportional Y for the CoW match
        # Use the exchange rate that fully satisfies the constraining order
        # (noqa: SIM108 - ternary would lose explanatory comments below)
        if cow_x == buy_amount_b:  # noqa: SIM108
            # B is fully satisfied, B gives all their Y (exact, no rounding)
            cow_y = sell_amount_b
        else:
            # A is fully satisfied (cow_x == sell_amount_a < buy_amount_b)
            # A gives all their X, receives proportional Y
            # Use B's rate: cow_y = cow_x * (sell_amount_b / buy_amount_b)
            # Floor division: rounds DOWN, conservative for A (receives slightly less)
            # This is safe because we verify A's limit with ceiling division below
            cow_y = (cow_x * sell_amount_b) // buy_amount_b

        # Verify A's limit is satisfied for the partial fill
        # A sells cow_x, must receive at least cow_x * (buy_amount_a / sell_amount_a)
        a_min_receive = (cow_x * buy_amount_a + sell_amount_a - 1) // sell_amount_a  # Round up
        if cow_y < a_min_receive:
            logger.debug(
                "cow_partial_no_match_a_limit",
                reason="Partial fill doesn't satisfy A's limit",
                cow_y=cow_y,
                a_min_receive=a_min_receive,
            )
            return None

        # Verify B's limit is satisfied
        # B sells cow_y, must receive at least cow_y * (buy_amount_b / sell_amount_b)
        b_min_receive = (cow_y * buy_amount_b + sell_amount_b - 1) // sell_amount_b  # Round up
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
            a_remainder=sell_amount_a - cow_x,
            b_remainder=sell_amount_b - cow_y,
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
        self,
        order_a: Order,
        order_b: Order,
        sell_amount_a: int,
        buy_amount_a: int,
        sell_amount_b: int,
        buy_amount_b: int,
    ) -> CowMatch | None:
        """Find partial CoW match for sell order A and buy order B.

        A (sell): sells sell_amount_a of X, wants min buy_amount_a of Y
        B (buy): wants exactly buy_amount_b of X, pays up to sell_amount_b of Y

        Partial match when A offers more X than B wants:
        - B gets exactly buy_amount_b of X (complete)
        - B pays sell_amount_b of Y (max payment)
        - A sells buy_amount_b of X, receives sell_amount_b of Y (partial)
        - A has remainder: (sell_amount_a - buy_amount_b) of X

        We don't partial-fill B since buy orders want exact amounts.
        """
        # Only partial if A offers MORE than B wants
        # (if A offers less, we can't partially fill B's exact want)
        if sell_amount_a <= buy_amount_b:
            return None

        # A would be partially filled - check if allowed
        if not order_a.partially_fillable:
            logger.debug(
                "cow_partial_rejected_fill_or_kill",
                reason="Order A would be partially filled but is fill-or-kill",
                order_uid=order_a.uid[:18] + "...",
            )
            return None

        # Check if A's limit can be satisfied by B's payment
        # A wants at least buy_amount_a for sell_amount_a
        # For partial fill of buy_amount_b, A needs proportionally:
        # min_receive = buy_amount_b * (buy_amount_a / sell_amount_a)
        # B pays sell_amount_b, so check: sell_amount_b >= min_receive
        # i.e., sell_amount_b * sell_amount_a >= buy_amount_b * buy_amount_a
        if sell_amount_b * sell_amount_a < buy_amount_b * buy_amount_a:
            logger.debug(
                "cow_partial_sell_buy_limit_not_met",
                reason="A's limit not satisfied by B's max payment",
            )
            return None

        # Execute: A sells buy_amount_b of X, receives sell_amount_b of Y
        # B gets exactly buy_amount_b of X, pays sell_amount_b of Y
        cow_x = buy_amount_b  # X transferred (A→B)
        cow_y = sell_amount_b  # Y transferred (B→A)

        logger.debug(
            "cow_partial_match_found",
            match_type="sell_buy",
            cow_x=cow_x,
            cow_y=cow_y,
            a_remainder=sell_amount_a - cow_x,
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
        self,
        order_a: Order,
        order_b: Order,
        sell_amount_a: int,
        buy_amount_a: int,
        sell_amount_b: int,
        buy_amount_b: int,
    ) -> CowMatch | None:
        """Find partial CoW match for buy order A and sell order B.

        A (buy): wants exactly buy_amount_a of Y, pays up to sell_amount_a of X
        B (sell): sells sell_amount_b of Y, wants min buy_amount_b of X

        Partial match when B offers more Y than A wants:
        - A gets exactly buy_amount_a of Y (complete)
        - A pays sell_amount_a of X (max payment)
        - B sells buy_amount_a of Y, receives sell_amount_a of X (partial)
        - B has remainder: (sell_amount_b - buy_amount_a) of Y

        We don't partial-fill A since buy orders want exact amounts.
        """
        # Only partial if B offers MORE than A wants
        if sell_amount_b <= buy_amount_a:
            return None

        # B would be partially filled - check if allowed
        if not order_b.partially_fillable:
            logger.debug(
                "cow_partial_rejected_fill_or_kill",
                reason="Order B would be partially filled but is fill-or-kill",
                order_uid=order_b.uid[:18] + "...",
            )
            return None

        # Check if B's limit can be satisfied by A's payment
        # B wants at least buy_amount_b for sell_amount_b
        # For partial fill of buy_amount_a, B needs proportionally:
        # min_receive = buy_amount_a * (buy_amount_b / sell_amount_b)
        # A pays sell_amount_a, so check: sell_amount_a >= min_receive
        # i.e., sell_amount_a * sell_amount_b >= buy_amount_a * buy_amount_b
        if sell_amount_a * sell_amount_b < buy_amount_a * buy_amount_b:
            logger.debug(
                "cow_partial_buy_sell_limit_not_met",
                reason="B's limit not satisfied by A's max payment",
            )
            return None

        # Execute: B sells buy_amount_a of Y, receives sell_amount_a of X
        # A gets exactly buy_amount_a of Y, pays sell_amount_a of X
        cow_x = sell_amount_a  # X transferred (A→B)
        cow_y = buy_amount_a  # Y transferred (B→A)

        logger.debug(
            "cow_partial_match_found",
            match_type="buy_sell",
            cow_x=cow_x,
            cow_y=cow_y,
            b_remainder=sell_amount_b - cow_y,
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
        self,
        order_a: Order,
        order_b: Order,
        sell_amount_a: int,
        buy_amount_a: int,
        sell_amount_b: int,
        buy_amount_b: int,
    ) -> CowMatch | None:
        """Find partial CoW match for two buy orders.

        A (buy): wants exactly buy_amount_a of Y, pays up to sell_amount_a of X
        B (buy): wants exactly buy_amount_b of X, pays up to sell_amount_b of Y

        Partial match possibilities:
        1. A can fully satisfy B, but B can only partially satisfy A:
           - sell_amount_a >= buy_amount_b (A can pay what B wants)
           - sell_amount_b < buy_amount_a (B can't pay all A wants)
           - B gets complete fill, A gets partial fill

        2. B can fully satisfy A, but A can only partially satisfy B:
           - sell_amount_b >= buy_amount_a (B can pay what A wants)
           - sell_amount_a < buy_amount_b (A can't pay all B wants)
           - A gets complete fill, B gets partial fill
        """
        a_can_satisfy_b = sell_amount_a >= buy_amount_b
        b_can_satisfy_a = sell_amount_b >= buy_amount_a

        # If both can satisfy each other, it's a perfect match (handled elsewhere)
        if a_can_satisfy_b and b_can_satisfy_a:
            return None

        # If neither can satisfy the other, no match possible
        if not a_can_satisfy_b and not b_can_satisfy_a:
            return None

        if a_can_satisfy_b and not b_can_satisfy_a:
            # Case 1: B gets complete fill, A gets partial fill
            # A would be partially filled - check if allowed
            if not order_a.partially_fillable:
                logger.debug(
                    "cow_partial_rejected_fill_or_kill",
                    reason="Order A would be partially filled but is fill-or-kill",
                    order_uid=order_a.uid[:18] + "...",
                )
                return None

            # B gets exactly buy_amount_b of X, pays sell_amount_b of Y
            # A pays buy_amount_b of X, receives sell_amount_b of Y (partial)
            cow_x = buy_amount_b  # X transferred (A→B)
            cow_y = sell_amount_b  # Y transferred (B→A), less than A wants

            logger.debug(
                "cow_partial_match_found",
                match_type="buy_buy_b_complete",
                cow_x=cow_x,
                cow_y=cow_y,
                a_receives=cow_y,
                a_wanted=buy_amount_a,
            )

            return CowMatch(
                order_a=order_a,
                order_b=order_b,
                exec_sell_a=cow_x,
                exec_buy_a=cow_y,
                exec_sell_b=cow_y,
                exec_buy_b=cow_x,
            )

        else:
            # Case 2: A gets complete fill, B gets partial fill
            # B would be partially filled - check if allowed
            if not order_b.partially_fillable:
                logger.debug(
                    "cow_partial_rejected_fill_or_kill",
                    reason="Order B would be partially filled but is fill-or-kill",
                    order_uid=order_b.uid[:18] + "...",
                )
                return None

            # A gets exactly buy_amount_a of Y, pays sell_amount_a of X
            # B pays buy_amount_a of Y, receives sell_amount_a of X (partial)
            cow_x = sell_amount_a  # X transferred (A→B), less than B wants
            cow_y = buy_amount_a  # Y transferred (B→A)

            logger.debug(
                "cow_partial_match_found",
                match_type="buy_buy_a_complete",
                cow_x=cow_x,
                cow_y=cow_y,
                b_receives=cow_x,
                b_wanted=buy_amount_b,
            )

            return CowMatch(
                order_a=order_a,
                order_b=order_b,
                exec_sell_a=cow_x,
                exec_buy_a=cow_y,
                exec_sell_b=cow_y,
                exec_buy_b=cow_x,
            )

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
        # price[X] / price[Y] = exec_buy_a / exec_sell_a = exec_sell_b / exec_sell_a
        #
        # We use:
        #   price[X] = exec_sell_b (amount of Y per unit of X, scaled)
        #   price[Y] = exec_sell_a (amount of X per unit of Y, scaled)
        #
        # This ensures both orders' constraints are satisfied with equality
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
