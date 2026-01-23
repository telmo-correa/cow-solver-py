"""UniswapV3 pool routing handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from solver.models.auction import Order
from solver.routing.handlers.base import BaseHandler
from solver.routing.types import RoutingResult

if TYPE_CHECKING:
    from solver.amm.uniswap_v3 import UniswapV3AMM, UniswapV3Pool


class UniswapV3Handler(BaseHandler):
    """Handler for UniswapV3 pool routing.

    Uses the V3 quoter to simulate swaps. V3 pools require on-chain quotes
    due to concentrated liquidity (no local reserve state).
    """

    def __init__(self, amm: UniswapV3AMM | None) -> None:
        """Initialize the V3 handler.

        Args:
            amm: UniswapV3 AMM instance for swap simulation.
                 If None, routing through V3 pools will fail.
        """
        self.amm = amm

    def route(
        self,
        order: Order,
        pool: UniswapV3Pool,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route an order through a V3 pool.

        Args:
            order: The order to route
            pool: The V3 pool to use
            sell_amount: Order's sell amount
            buy_amount: Order's buy amount (minimum for sell, exact for buy)

        Returns:
            RoutingResult with the routing outcome
        """
        if self.amm is None:
            return self._error_result(order, "V3: AMM not configured")

        if order.is_sell_order:
            return self._route_sell_order(order, pool, sell_amount, buy_amount)
        return self._route_buy_order(order, pool, sell_amount, buy_amount)

    def _route_sell_order(
        self,
        order: Order,
        pool: UniswapV3Pool,
        sell_amount: int,
        min_buy_amount: int,
    ) -> RoutingResult:
        """Route a sell order (exact input) through V3 pool.

        Args:
            order: The order being routed
            pool: V3 pool to use for the swap
            sell_amount: Exact amount to sell
            min_buy_amount: Minimum acceptable output

        Returns:
            RoutingResult with success=True if output >= min_buy_amount
        """
        assert self.amm is not None  # Checked by caller

        result = self.amm.simulate_swap(pool, order.sell_token, sell_amount)
        if result is None:
            return self._error_result(order, "V3: quote failed")

        # Check if output meets minimum
        if result.amount_out < min_buy_amount:
            return RoutingResult(
                order=order,
                amount_in=sell_amount,
                amount_out=result.amount_out,
                pool=pool,
                success=False,
                error=f"Output {result.amount_out} below minimum {min_buy_amount}",
            )

        return self._build_success_result(
            order, pool, sell_amount, result.amount_out, pool.gas_estimate
        )

    def _route_buy_order(
        self,
        order: Order,
        pool: UniswapV3Pool,
        max_sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route a buy order (exact output) through V3 pool.

        Args:
            order: The order being routed
            pool: V3 pool to use for the swap
            max_sell_amount: Maximum willing to pay
            buy_amount: Exact amount to receive

        Returns:
            RoutingResult with success=True if required input <= max_sell_amount
        """
        assert self.amm is not None  # Checked by caller

        result = self.amm.simulate_swap_exact_output(pool, order.sell_token, buy_amount)
        if result is None:
            return self._error_result(order, "V3: quote failed")

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

        # For buy orders: use requested buy_amount for trade/prices,
        # actual forward-simulated output for interaction
        return self._build_success_result(
            order,
            pool,
            result.amount_in,
            buy_amount,
            pool.gas_estimate,
            actual_amount_out=result.amount_out,
        )


__all__ = ["UniswapV3Handler"]
