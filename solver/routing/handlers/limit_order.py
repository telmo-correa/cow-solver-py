"""0x limit order routing handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from solver.models.auction import Order
from solver.routing.handlers.base import BaseHandler
from solver.routing.types import RoutingResult

if TYPE_CHECKING:
    from solver.amm.limit_order import LimitOrderAMM
    from solver.pools.limit_order import LimitOrderPool


class LimitOrderHandler(BaseHandler):
    """Handler for 0x limit order routing.

    Limit orders have fixed exchange rates (no slippage curve).
    They are unidirectional: taker_token -> maker_token only.
    """

    def __init__(self, amm: LimitOrderAMM | None) -> None:
        """Initialize the limit order handler.

        Args:
            amm: LimitOrderAMM instance for swap simulation.
                 If None, routing through limit orders will fail.
        """
        self.amm = amm

    def route(
        self,
        order: Order,
        pool: LimitOrderPool,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route an order through a limit order.

        Args:
            order: The order to route
            pool: The limit order to use
            sell_amount: Order's sell amount
            buy_amount: Order's buy amount (minimum for sell, exact for buy)

        Returns:
            RoutingResult with the routing outcome
        """
        if self.amm is None:
            return self._error_result(order, "Limit order: AMM not configured")

        if order.is_sell_order:
            return self._route_sell_order(order, pool, sell_amount, buy_amount)
        return self._route_buy_order(order, pool, sell_amount, buy_amount)

    def _route_sell_order(
        self,
        order: Order,
        pool: LimitOrderPool,
        sell_amount: int,
        min_buy_amount: int,
    ) -> RoutingResult:
        """Route a sell order through a limit order."""
        assert self.amm is not None  # Checked by caller

        result = self.amm.simulate_swap(pool, order.sell_token, order.buy_token, sell_amount)
        if result is None:
            return self._error_result(
                order,
                f"Limit order: invalid direction (expected {pool.taker_token[:10]}... -> {pool.maker_token[:10]}...)",
            )

        # For partial fills (when actual_amount_in < sell_amount), check proportional limit
        # Proportional minimum = (actual_amount_in * min_buy_amount) / sell_amount
        actual_amount_in = result.amount_in
        if actual_amount_in < sell_amount:
            # Calculate proportional minimum (round up to protect user)
            proportional_min = (actual_amount_in * min_buy_amount + sell_amount - 1) // sell_amount
        else:
            proportional_min = min_buy_amount

        # Check if output meets proportional minimum
        if result.amount_out < proportional_min:
            return RoutingResult(
                order=order,
                amount_in=actual_amount_in,
                amount_out=result.amount_out,
                pool=pool,
                success=False,
                error=f"Output {result.amount_out} below minimum {proportional_min}",
            )

        return self._build_success_result(
            order, pool, actual_amount_in, result.amount_out, pool.gas_estimate
        )

    def _route_buy_order(
        self,
        order: Order,
        pool: LimitOrderPool,
        max_sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route a buy order through a limit order."""
        assert self.amm is not None  # Checked by caller

        result = self.amm.simulate_swap_exact_output(
            pool, order.sell_token, order.buy_token, buy_amount
        )
        if result is None:
            # For exact output, None means either wrong direction or amount exceeds capacity
            if not pool.supports_pair(order.sell_token, order.buy_token):
                return self._error_result(
                    order,
                    f"Limit order: invalid direction (expected {pool.taker_token[:10]}... -> {pool.maker_token[:10]}...)",
                )
            return self._error_result(
                order,
                f"Limit order: requested output {buy_amount} exceeds available {pool.maker_amount}",
            )

        # Check if required input exceeds maximum
        if result.amount_in > max_sell_amount:
            return RoutingResult(
                order=order,
                amount_in=result.amount_in,
                amount_out=buy_amount,
                pool=pool,
                success=False,
                error=f"Required input {result.amount_in} exceeds maximum {max_sell_amount}",
            )

        return self._build_success_result(
            order, pool, result.amount_in, buy_amount, pool.gas_estimate
        )


__all__ = ["LimitOrderHandler"]
