"""Balancer pool routing handler with singledispatch for type-safe dispatch."""

from __future__ import annotations

from collections.abc import Callable
from functools import singledispatchmethod

import structlog

from solver.amm.balancer import (
    BalancerStableAMM,
    BalancerStablePool,
    BalancerWeightedAMM,
    BalancerWeightedPool,
)
from solver.models.auction import Order
from solver.routing.handlers.base import BaseHandler
from solver.routing.types import RoutingResult

logger = structlog.get_logger()

# Type alias for AMM types
BalancerAMM = BalancerWeightedAMM | BalancerStableAMM
BalancerPool = BalancerWeightedPool | BalancerStablePool


class BalancerHandler(BaseHandler):
    """Handler for Balancer pool routing (weighted and stable).

    Uses singledispatch to route to the correct AMM based on pool type,
    with fully type-safe implementations for each pool type.

    Supports both weighted product pools and stable (StableSwap) pools,
    including partial fills for partially fillable orders.
    """

    def __init__(
        self,
        weighted_amm: BalancerWeightedAMM | None,
        stable_amm: BalancerStableAMM | None,
    ) -> None:
        """Initialize the Balancer handler.

        Args:
            weighted_amm: AMM for weighted pools. If None, weighted pools fail.
            stable_amm: AMM for stable pools. If None, stable pools fail.
        """
        self.weighted_amm = weighted_amm
        self.stable_amm = stable_amm

    def route(
        self,
        order: Order,
        pool: BalancerWeightedPool | BalancerStablePool,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route an order through a Balancer pool.

        Uses singledispatch to delegate to the appropriate method based on
        the pool's concrete type.

        Args:
            order: The order to route
            pool: The Balancer pool (weighted or stable)
            sell_amount: Order's sell amount
            buy_amount: Order's buy amount (minimum for sell, exact for buy)

        Returns:
            RoutingResult with the routing outcome
        """
        return self._route_pool(pool, order, sell_amount, buy_amount)

    @singledispatchmethod
    def _route_pool(
        self,
        pool: BalancerWeightedPool | BalancerStablePool,
        order: Order,
        _sell_amount: int,
        _buy_amount: int,
    ) -> RoutingResult:
        """Base dispatch method - raises for unknown pool types."""
        return self._error_result(order, f"Unknown Balancer pool type: {type(pool)}")

    @_route_pool.register(BalancerWeightedPool)
    def _route_weighted(
        self,
        pool: BalancerWeightedPool,
        order: Order,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route through a Balancer weighted pool."""
        if self.weighted_amm is None:
            return self._error_result(order, "Balancer weighted: AMM not configured")
        return self._route_balancer_pool(
            pool, self.weighted_amm, "weighted", order, sell_amount, buy_amount
        )

    @_route_pool.register(BalancerStablePool)
    def _route_stable(
        self,
        pool: BalancerStablePool,
        order: Order,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route through a Balancer stable pool."""
        if self.stable_amm is None:
            return self._error_result(order, "Balancer stable: AMM not configured")
        return self._route_balancer_pool(
            pool, self.stable_amm, "stable", order, sell_amount, buy_amount
        )

    def _route_balancer_pool(
        self,
        pool: BalancerPool,
        amm: BalancerAMM,
        pool_type: str,
        order: Order,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Common routing logic for all Balancer pool types.

        Args:
            pool: The Balancer pool (weighted or stable)
            amm: The AMM implementation (must match pool type)
            pool_type: String identifier for logging ("weighted" or "stable")
            order: The order to route
            sell_amount: Order's sell amount
            buy_amount: Order's buy amount

        Returns:
            RoutingResult with the routing outcome

        Note: Type unions are safe here because callers ensure pool/amm types match.
        """

        # Define partial fill callback for this pool/amm combination
        def try_partial() -> RoutingResult:
            return self._try_partial_fill_impl(order, pool, amm, pool_type, sell_amount, buy_amount)

        if order.is_sell_order:
            return self._route_sell_order(
                pool, amm, pool_type, order, sell_amount, buy_amount, try_partial
            )
        else:
            return self._route_buy_order(
                pool, amm, pool_type, order, sell_amount, buy_amount, try_partial
            )

    def _route_sell_order(
        self,
        pool: BalancerPool,
        amm: BalancerAMM,
        pool_type: str,
        order: Order,
        sell_amount: int,
        buy_amount: int,
        try_partial: Callable[[], RoutingResult],
    ) -> RoutingResult:
        """Route a sell order through a Balancer pool."""
        # Pool/AMM type match guaranteed by _route_weighted/_route_stable callers
        result = amm.simulate_swap(pool, order.sell_token, order.buy_token, sell_amount)  # type: ignore[arg-type]
        if result is None:
            if order.partially_fillable:
                return try_partial()
            return self._error_result(order, f"Balancer {pool_type}: quote failed")

        if result.amount_out < buy_amount:
            if order.partially_fillable:
                return try_partial()
            return RoutingResult(
                order=order,
                amount_in=sell_amount,
                amount_out=result.amount_out,
                pool=pool,
                success=False,
                error=f"Output {result.amount_out} below minimum {buy_amount}",
            )

        return self._build_success_result(
            order, pool, sell_amount, result.amount_out, pool.gas_estimate
        )

    def _route_buy_order(
        self,
        pool: BalancerPool,
        amm: BalancerAMM,
        pool_type: str,
        order: Order,
        sell_amount: int,
        buy_amount: int,
        try_partial: Callable[[], RoutingResult],
    ) -> RoutingResult:
        """Route a buy order through a Balancer pool."""
        # Pool/AMM type match guaranteed by _route_weighted/_route_stable callers
        result = amm.simulate_swap_exact_output(
            pool,  # type: ignore[arg-type]
            order.sell_token,
            order.buy_token,
            buy_amount,
        )
        if result is None:
            if order.partially_fillable:
                return try_partial()
            return self._error_result(order, f"Balancer {pool_type}: quote failed")

        if result.amount_in > sell_amount:
            if order.partially_fillable:
                return try_partial()
            return RoutingResult(
                order=order,
                amount_in=result.amount_in,
                amount_out=buy_amount,
                pool=pool,
                success=False,
                error=f"Required input {result.amount_in} exceeds maximum {sell_amount}",
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

    def _try_partial_fill_impl(
        self,
        order: Order,
        pool: BalancerPool,
        amm: BalancerAMM,
        pool_type: str,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Implementation for partial fill calculation.

        Note: This method accepts union types but is only called from contexts
        where pool and amm types are guaranteed to match.
        """
        is_sell = order.is_sell_order
        order_uid_short = order.uid[:18] + "..."

        # Calculate maximum fill amount
        # Pool/AMM type match guaranteed by _route_balancer_pool caller
        if is_sell:
            max_fill = amm.max_fill_sell_order(
                pool=pool,  # type: ignore[arg-type]
                token_in=order.sell_token,
                token_out=order.buy_token,
                sell_amount=sell_amount,
                buy_amount=buy_amount,
            )
        else:
            max_fill = amm.max_fill_buy_order(
                pool=pool,  # type: ignore[arg-type]
                token_in=order.sell_token,
                token_out=order.buy_token,
                sell_amount=sell_amount,
                buy_amount=buy_amount,
            )

        if max_fill <= 0:
            logger.debug(
                f"partial_{pool_type}_{'sell' if is_sell else 'buy'}_order_no_valid_fill",
                order_uid=order_uid_short,
                reason="pool_rate_worse_than_limit",
            )
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=pool,
                success=False,
                error="Pool rate worse than limit price, no partial fill possible",
            )

        # Simulate the swap at the max fill amount
        if is_sell:
            result = amm.simulate_swap(
                pool=pool,  # type: ignore[arg-type]
                token_in=order.sell_token,
                token_out=order.buy_token,
                amount_in=max_fill,
            )
            fail_amount_in = max_fill
            fail_amount_out = 0
        else:
            result = amm.simulate_swap_exact_output(
                pool=pool,  # type: ignore[arg-type]
                token_in=order.sell_token,
                token_out=order.buy_token,
                amount_out=max_fill,
            )
            fail_amount_in = 0
            fail_amount_out = max_fill

        if result is None:
            return RoutingResult(
                order=order,
                amount_in=fail_amount_in,
                amount_out=fail_amount_out,
                pool=pool,
                success=False,
                error=f"Balancer {pool_type}: swap simulation failed for partial fill",
            )

        # Verify the limit price constraint
        actual_out: int | None = None  # For buy orders, tracks actual forward-simulated output
        if is_sell:
            # Sell: output/input >= buy_amount/sell_amount
            limit_satisfied = result.amount_out * sell_amount >= buy_amount * max_fill
            final_in, final_out = max_fill, result.amount_out
            log_key = "partial_sell"
            fill_ratio = f"{max_fill * 100 // sell_amount}%"
        else:
            # Buy: input/output <= sell_amount/buy_amount
            # Use result.amount_out for limit check (actual forward-simulated output)
            limit_satisfied = result.amount_in * buy_amount <= sell_amount * result.amount_out
            # For trade/prices: use requested max_fill
            # For interaction: use actual result.amount_out
            final_in, final_out = result.amount_in, max_fill
            actual_out = result.amount_out
            log_key = "partial_buy"
            fill_ratio = f"{max_fill * 100 // buy_amount}%"

        if not limit_satisfied:
            logger.warning(
                "partial_fill_limit_check_failed",
                order_uid=order_uid_short,
                max_fill=max_fill,
                amount_in=final_in,
                amount_out=final_out,
            )
            return RoutingResult(
                order=order,
                amount_in=final_in,
                amount_out=final_out,
                pool=pool,
                success=False,
                error="Partial fill calculation error",
            )

        logger.info(
            f"partial_fill_{pool_type}_{'sell' if is_sell else 'buy'}_order",
            order_uid=order_uid_short,
            **{log_key: max_fill},
            fill_ratio=fill_ratio,
        )

        return self._build_success_result(
            order,
            pool,
            final_in,
            final_out,
            pool.gas_estimate,
            actual_amount_out=actual_out,
        )


__all__ = ["BalancerHandler"]
