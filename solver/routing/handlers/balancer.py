"""Balancer pool routing handler with singledispatch for type-safe dispatch."""

from __future__ import annotations

from functools import singledispatchmethod

import structlog

from solver.amm.balancer import (
    BalancerStableAMM,
    BalancerStablePool,
    BalancerWeightedAMM,
    BalancerWeightedPool,
)
from solver.constants import (
    BALANCER_STABLE_SWAP_GAS_COST,
    BALANCER_WEIGHTED_SWAP_GAS_COST,
)
from solver.models.auction import Order
from solver.routing.handlers.base import BaseHandler
from solver.routing.types import RoutingResult

logger = structlog.get_logger()

# TODO: Revert to using pool.gas_estimate instead of hardcoded constants.
# The Rust baseline solver ignores the gasEstimate from the auction JSON and uses
# hardcoded constants (WEIGHTED=100000, STABLE=183520). This is likely a bug since
# the driver provides per-pool gas estimates for a reason. For now we match Rust's
# behavior for parity testing. Once parity is achieved, revert to pool.gas_estimate
# and update test fixtures to expect the correct (non-hardcoded) gas values.
# See: cow-services/crates/shared/src/sources/balancer_v2/swap/mod.rs:26-28


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

        amm = self.weighted_amm
        pool_type = "weighted"

        if order.is_sell_order:
            result = amm.simulate_swap(pool, order.sell_token, order.buy_token, sell_amount)
            if result is None:
                if order.partially_fillable:
                    return self._try_partial_weighted(order, pool, amm, sell_amount, buy_amount)
                return self._error_result(order, f"Balancer {pool_type}: quote failed")

            if result.amount_out < buy_amount:
                if order.partially_fillable:
                    return self._try_partial_weighted(order, pool, amm, sell_amount, buy_amount)
                return RoutingResult(
                    order=order,
                    amount_in=sell_amount,
                    amount_out=result.amount_out,
                    pool=pool,
                    success=False,
                    error=f"Output {result.amount_out} below minimum {buy_amount}",
                )

            return self._build_success_result(
                order, pool, sell_amount, result.amount_out, BALANCER_WEIGHTED_SWAP_GAS_COST
            )
        else:
            result = amm.simulate_swap_exact_output(
                pool, order.sell_token, order.buy_token, buy_amount
            )
            if result is None:
                if order.partially_fillable:
                    return self._try_partial_weighted(order, pool, amm, sell_amount, buy_amount)
                return self._error_result(order, f"Balancer {pool_type}: quote failed")

            if result.amount_in > sell_amount:
                if order.partially_fillable:
                    return self._try_partial_weighted(order, pool, amm, sell_amount, buy_amount)
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
                BALANCER_WEIGHTED_SWAP_GAS_COST,
                actual_amount_out=result.amount_out,
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

        amm = self.stable_amm
        pool_type = "stable"

        if order.is_sell_order:
            result = amm.simulate_swap(pool, order.sell_token, order.buy_token, sell_amount)
            if result is None:
                if order.partially_fillable:
                    return self._try_partial_stable(order, pool, amm, sell_amount, buy_amount)
                return self._error_result(order, f"Balancer {pool_type}: quote failed")

            if result.amount_out < buy_amount:
                if order.partially_fillable:
                    return self._try_partial_stable(order, pool, amm, sell_amount, buy_amount)
                return RoutingResult(
                    order=order,
                    amount_in=sell_amount,
                    amount_out=result.amount_out,
                    pool=pool,
                    success=False,
                    error=f"Output {result.amount_out} below minimum {buy_amount}",
                )

            return self._build_success_result(
                order, pool, sell_amount, result.amount_out, BALANCER_STABLE_SWAP_GAS_COST
            )
        else:
            result = amm.simulate_swap_exact_output(
                pool, order.sell_token, order.buy_token, buy_amount
            )
            if result is None:
                if order.partially_fillable:
                    return self._try_partial_stable(order, pool, amm, sell_amount, buy_amount)
                return self._error_result(order, f"Balancer {pool_type}: quote failed")

            if result.amount_in > sell_amount:
                if order.partially_fillable:
                    return self._try_partial_stable(order, pool, amm, sell_amount, buy_amount)
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
                BALANCER_STABLE_SWAP_GAS_COST,
                actual_amount_out=result.amount_out,
            )

    def _try_partial_weighted(
        self,
        order: Order,
        pool: BalancerWeightedPool,
        amm: BalancerWeightedAMM,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Try partial fill for a weighted pool order."""
        return self._try_partial_fill_impl(order, pool, amm, "weighted", sell_amount, buy_amount)

    def _try_partial_stable(
        self,
        order: Order,
        pool: BalancerStablePool,
        amm: BalancerStableAMM,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Try partial fill for a stable pool order."""
        return self._try_partial_fill_impl(order, pool, amm, "stable", sell_amount, buy_amount)

    def _try_partial_fill_impl(
        self,
        order: Order,
        pool: BalancerWeightedPool | BalancerStablePool,
        amm: BalancerWeightedAMM | BalancerStableAMM,
        pool_type: str,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Implementation for partial fill calculation.

        Note: This method accepts union types but is only called from type-safe
        wrapper methods that ensure pool and amm types match.
        """
        is_sell = order.is_sell_order
        order_uid_short = order.uid[:18] + "..."

        # Calculate maximum fill amount
        # The caller ensures pool/amm types match via type-safe wrapper methods
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

        gas_cost = (
            BALANCER_WEIGHTED_SWAP_GAS_COST
            if pool_type == "weighted"
            else BALANCER_STABLE_SWAP_GAS_COST
        )
        return self._build_success_result(
            order,
            pool,
            final_in,
            final_out,
            gas_cost,
            actual_amount_out=actual_out,
        )


__all__ = ["BalancerHandler"]
