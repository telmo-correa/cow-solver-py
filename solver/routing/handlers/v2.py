"""UniswapV2 pool routing handler."""

from __future__ import annotations

import structlog

from solver.amm.uniswap_v2 import UniswapV2, UniswapV2Pool
from solver.constants import POOL_SWAP_GAS_COST
from solver.models.auction import Order
from solver.routing.handlers.base import BaseHandler
from solver.routing.types import RoutingResult

logger = structlog.get_logger()


class UniswapV2Handler(BaseHandler):
    """Handler for UniswapV2 pool routing.

    Handles both sell orders (exact input) and buy orders (exact output),
    including partial fills for partially fillable orders.
    """

    def __init__(self, amm: UniswapV2) -> None:
        """Initialize the V2 handler.

        Args:
            amm: UniswapV2 AMM instance for swap simulation
        """
        self.amm = amm

    def route(
        self,
        order: Order,
        pool: UniswapV2Pool,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route an order through a V2 pool.

        Args:
            order: The order to route
            pool: The V2 pool to use
            sell_amount: Order's sell amount
            buy_amount: Order's buy amount

        Returns:
            RoutingResult with the routing outcome
        """
        if order.is_sell_order:
            return self._route_sell_order(order, pool, sell_amount, buy_amount)
        return self._route_buy_order(order, pool, sell_amount, buy_amount)

    def _route_sell_order(
        self,
        order: Order,
        pool: UniswapV2Pool,
        sell_amount: int,
        min_buy_amount: int,
    ) -> RoutingResult:
        """Route a sell order (exact input, minimum output).

        For sell orders:
        - sell_amount is the exact amount to sell
        - buy_amount is the minimum acceptable output

        If the order is partially fillable and full fill fails, attempts
        to find the maximum partial fill that satisfies the limit price.
        """
        swap_result = self.amm.simulate_swap(
            pool=pool,
            token_in=order.sell_token,
            amount_in=sell_amount,
        )

        # Check if output meets minimum
        if swap_result.amount_out < min_buy_amount:
            # Full fill fails - try partial fill if allowed
            if order.partially_fillable:
                return self._try_partial_sell_order(order, pool, sell_amount, min_buy_amount)

            return RoutingResult(
                order=order,
                amount_in=sell_amount,
                amount_out=swap_result.amount_out,
                pool=pool,
                success=False,
                error=f"Output {swap_result.amount_out} below minimum {min_buy_amount}",
            )

        return self._build_success_result(
            order, pool, sell_amount, swap_result.amount_out, POOL_SWAP_GAS_COST
        )

    def _route_buy_order(
        self,
        order: Order,
        pool: UniswapV2Pool,
        max_sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route a buy order (exact output, maximum input).

        For buy orders:
        - buy_amount is the exact amount to receive
        - sell_amount is the maximum willing to pay

        If the order is partially fillable and full fill fails, attempts
        to find the maximum partial fill that satisfies the limit price.
        """
        swap_result = self.amm.simulate_swap_exact_output(
            pool=pool,
            token_in=order.sell_token,
            amount_out=buy_amount,
        )

        # Check if required input exceeds maximum
        if swap_result.amount_in > max_sell_amount:
            # Full fill fails - try partial fill if allowed
            if order.partially_fillable:
                return self._try_partial_buy_order(order, pool, max_sell_amount, buy_amount)

            return RoutingResult(
                order=order,
                amount_in=swap_result.amount_in,
                amount_out=buy_amount,
                pool=pool,
                success=False,
                error=f"Required input {swap_result.amount_in} exceeds maximum {max_sell_amount}",
            )

        return self._build_success_result(
            order, pool, swap_result.amount_in, buy_amount, POOL_SWAP_GAS_COST
        )

    def _try_partial_sell_order(
        self,
        order: Order,
        pool: UniswapV2Pool,
        sell_amount: int,
        min_buy_amount: int,
    ) -> RoutingResult:
        """Try to find maximum partial fill for a sell order.

        Calculates the maximum input amount that satisfies the order's
        limit price and simulates the swap at that amount.
        """
        reserve_in, reserve_out = pool.get_reserves(order.sell_token)

        # Calculate maximum partial fill that satisfies limit price
        max_input = self.amm.max_fill_sell_order(
            reserve_in=reserve_in,
            reserve_out=reserve_out,
            sell_amount=sell_amount,
            buy_amount=min_buy_amount,
            fee_multiplier=pool.fee_multiplier,
        )

        if max_input <= 0:
            logger.debug(
                "partial_sell_order_no_valid_fill",
                order_uid=order.uid[:18] + "...",
                reason="pool_rate_worse_than_limit",
                sell_amount=sell_amount,
                min_buy_amount=min_buy_amount,
            )
            return self._partial_fill_result(
                order,
                pool,
                0,
                0,
                success=False,
                error="Pool rate worse than limit price, no partial fill possible",
            )

        # Simulate swap at partial amount
        swap_result = self.amm.simulate_swap(
            pool=pool,
            token_in=order.sell_token,
            amount_in=max_input,
        )

        # Verify the partial fill satisfies the limit (defensive check)
        # Limit: output/input >= min_buy_amount/sell_amount
        if swap_result.amount_out * sell_amount < min_buy_amount * max_input:
            logger.warning(
                "partial_fill_limit_check_failed",
                order_uid=order.uid[:18] + "...",
                max_input=max_input,
                amount_out=swap_result.amount_out,
                expected_min=min_buy_amount * max_input // sell_amount,
            )
            return self._partial_fill_result(
                order,
                pool,
                max_input,
                swap_result.amount_out,
                success=False,
                error="Partial fill calculation error",
            )

        logger.info(
            "partial_fill_sell_order",
            order_uid=order.uid[:18] + "...",
            original_sell=sell_amount,
            partial_sell=max_input,
            fill_ratio=f"{max_input * 100 // sell_amount}%",
            amount_out=swap_result.amount_out,
        )

        return self._partial_fill_result(
            order,
            pool,
            max_input,
            swap_result.amount_out,
            success=True,
        )

    def _try_partial_buy_order(
        self,
        order: Order,
        pool: UniswapV2Pool,
        max_sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Try to find maximum partial fill for a buy order.

        Calculates the maximum output amount that satisfies the order's
        limit price and simulates the swap for that amount.
        """
        reserve_in, reserve_out = pool.get_reserves(order.sell_token)

        # Calculate maximum partial fill that satisfies limit price
        max_output = self.amm.max_fill_buy_order(
            reserve_in=reserve_in,
            reserve_out=reserve_out,
            sell_amount=max_sell_amount,
            buy_amount=buy_amount,
            fee_multiplier=pool.fee_multiplier,
        )

        if max_output <= 0:
            logger.debug(
                "partial_buy_order_no_valid_fill",
                order_uid=order.uid[:18] + "...",
                reason="pool_rate_worse_than_limit",
                max_sell_amount=max_sell_amount,
                buy_amount=buy_amount,
            )
            return self._partial_fill_result(
                order,
                pool,
                0,
                0,
                success=False,
                error="Pool rate worse than limit price, no partial fill possible",
            )

        # Simulate swap for partial output
        swap_result = self.amm.simulate_swap_exact_output(
            pool=pool,
            token_in=order.sell_token,
            amount_out=max_output,
        )

        # Verify the partial fill satisfies the limit (defensive check)
        # Limit: input/output <= max_sell_amount/buy_amount
        if swap_result.amount_in * buy_amount > max_sell_amount * max_output:
            logger.warning(
                "partial_fill_limit_check_failed",
                order_uid=order.uid[:18] + "...",
                max_output=max_output,
                amount_in=swap_result.amount_in,
                expected_max=max_sell_amount * max_output // buy_amount,
            )
            return self._partial_fill_result(
                order,
                pool,
                swap_result.amount_in,
                max_output,
                success=False,
                error="Partial fill calculation error",
            )

        logger.info(
            "partial_fill_buy_order",
            order_uid=order.uid[:18] + "...",
            original_buy=buy_amount,
            partial_buy=max_output,
            fill_ratio=f"{max_output * 100 // buy_amount}%",
            amount_in=swap_result.amount_in,
        )

        return self._partial_fill_result(
            order,
            pool,
            swap_result.amount_in,
            max_output,
            success=True,
        )

    def _partial_fill_result(
        self,
        order: Order,
        pool: UniswapV2Pool,
        amount_in: int,
        amount_out: int,
        success: bool,
        error: str | None = None,
    ) -> RoutingResult:
        """Create a routing result for partial fill attempts.

        Used by both _try_partial_sell_order and _try_partial_buy_order
        to reduce duplication in result construction.
        """
        if not success:
            return RoutingResult(
                order=order,
                amount_in=amount_in,
                amount_out=amount_out,
                pool=pool,
                success=False,
                error=error,
            )

        return self._build_success_result(order, pool, amount_in, amount_out, POOL_SWAP_GAS_COST)


__all__ = ["UniswapV2Handler"]
