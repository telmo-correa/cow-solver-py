"""Coincidence of Wants (CoW) matching strategy.

Detects when orders can be settled directly against each other without
needing AMM liquidity, saving users from swap fees.
"""

from dataclasses import dataclass

import structlog

from solver.models.auction import AuctionInstance, Order
from solver.models.solution import Solution, Trade, TradeKind
from solver.models.types import normalize_address

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

    def try_solve(self, auction: AuctionInstance) -> Solution | None:
        """Try to find a CoW match in the auction.

        Currently only handles perfect matches between exactly 2 orders.

        Args:
            auction: The auction to solve

        Returns:
            A Solution if a CoW match is found, None otherwise
        """
        # Currently only handle 2-order auctions
        if auction.order_count != 2:
            return None

        match = self._find_perfect_match(auction.orders[0], auction.orders[1])
        if match is None:
            return None

        logger.info(
            "cow_match_found",
            order_a=match.order_a.uid[:18] + "...",
            order_b=match.order_b.uid[:18] + "...",
            exec_sell_a=match.exec_sell_a,
            exec_buy_a=match.exec_buy_a,
        )

        return self._build_solution(match)

    def _find_perfect_match(self, order_a: Order, order_b: Order) -> CowMatch | None:
        """Check if two orders form a perfect CoW match.

        A perfect match requires:
        1. Opposite token directions (A sells what B buys, B sells what A buys)
        2. Both limit prices satisfied when fully executed

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

        # For now, only handle sell orders
        # TODO: Extend to buy orders in future slices
        if not order_a.is_sell_order or not order_b.is_sell_order:
            logger.debug(
                "cow_no_match_order_type",
                reason="Buy orders not yet supported for CoW matching",
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

        # For a perfect match with full execution:
        # - A sells sell_amount_a of X, receives sell_amount_b of Y
        # - B sells sell_amount_b of Y, receives sell_amount_a of X
        #
        # Limit price constraints:
        # - A's limit: receives >= buy_amount_a → sell_amount_b >= buy_amount_a
        # - B's limit: receives >= buy_amount_b → sell_amount_a >= buy_amount_b

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

        # Perfect match found!
        return CowMatch(
            order_a=order_a,
            order_b=order_b,
            exec_sell_a=sell_amount_a,
            exec_buy_a=sell_amount_b,  # A receives what B sells
            exec_sell_b=sell_amount_b,
            exec_buy_b=sell_amount_a,  # B receives what A sells
        )

    def _build_solution(self, match: CowMatch) -> Solution:
        """Build a solution from a CoW match.

        Args:
            match: The matched order pair

        Returns:
            A Solution with trades and clearing prices (no AMM interactions)
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

        # Create trades for both orders
        # For sell orders, executed_amount is the sell amount
        trades = [
            Trade(
                kind=TradeKind.FULFILLMENT,
                order=match.order_a.uid,
                executedAmount=str(match.exec_sell_a),
            ),
            Trade(
                kind=TradeKind.FULFILLMENT,
                order=match.order_b.uid,
                executedAmount=str(match.exec_sell_b),
            ),
        ]

        # No AMM interactions needed - pure peer-to-peer settlement
        return Solution(
            id=0,
            prices=prices,
            trades=trades,
            interactions=[],
            gas=0,  # No on-chain swaps, just transfers
        )
