"""UniswapV3 pool routing handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from solver.models.auction import Order
from solver.models.types import normalize_address
from solver.routing.types import HopResult, RoutingResult

if TYPE_CHECKING:
    from solver.amm.uniswap_v3 import UniswapV3AMM, UniswapV3Pool


class UniswapV3Handler:
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

    def _error_result(self, order: Order, error: str) -> RoutingResult:
        """Create a failed routing result."""
        return RoutingResult(
            order=order,
            amount_in=0,
            amount_out=0,
            pool=None,
            success=False,
            error=error,
        )

    def _route_sell_order(
        self,
        order: Order,
        pool: UniswapV3Pool,
        sell_amount: int,
        min_buy_amount: int,
    ) -> RoutingResult:
        """Route a sell order through V3 pool."""
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

        hop = HopResult(
            pool=pool,
            input_token=normalize_address(order.sell_token),
            output_token=normalize_address(order.buy_token),
            amount_in=sell_amount,
            amount_out=result.amount_out,
        )

        return RoutingResult(
            order=order,
            amount_in=sell_amount,
            amount_out=result.amount_out,
            pool=pool,
            pools=[pool],
            hops=[hop],
            success=True,
            gas_estimate=pool.gas_estimate,
        )

    def _route_buy_order(
        self,
        order: Order,
        pool: UniswapV3Pool,
        max_sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route a buy order through V3 pool."""
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

        hop = HopResult(
            pool=pool,
            input_token=normalize_address(order.sell_token),
            output_token=normalize_address(order.buy_token),
            amount_in=result.amount_in,
            amount_out=buy_amount,
        )

        return RoutingResult(
            order=order,
            amount_in=result.amount_in,
            amount_out=buy_amount,
            pool=pool,
            pools=[pool],
            hops=[hop],
            success=True,
            gas_estimate=pool.gas_estimate,
        )


__all__ = ["UniswapV3Handler"]
