"""AMM routing strategy - routes orders through liquidity pools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from solver.amm.uniswap_v2 import (
    UniswapV2,
    UniswapV2Pool,
    uniswap_v2,
)
from solver.models.auction import AuctionInstance, Order
from solver.models.solution import Interaction, Solution
from solver.pools import PoolRegistry, build_registry_from_liquidity
from solver.routing.router import RoutingResult, SingleOrderRouter
from solver.strategies.base import OrderFill, StrategyResult

if TYPE_CHECKING:
    from solver.amm.balancer import BalancerStableAMM, BalancerWeightedAMM
    from solver.amm.uniswap_v3 import UniswapV3AMM

logger = structlog.get_logger()


@dataclass
class _OrderRoutingResult:
    """Internal result from routing a single order."""

    fill: OrderFill
    solution: Solution
    routing_result: RoutingResult


class AmmRoutingStrategy:
    """Strategy that routes orders through AMM liquidity pools.

    This strategy handles:
    - Single-order auctions (direct and multi-hop routing)
    - Multi-order auctions (each order routed independently)
    - Both sell orders (exact input) and buy orders (exact output)

    For multi-order auctions (e.g., remainder orders from partial CoW):
    - Each order is routed independently through the pools
    - Pool reserves are updated after each swap for accurate subsequent routing
    - Orders that can't be routed are skipped (partial success is OK)
    - Results are combined into a single StrategyResult

    Note: This is not optimal for multi-order batches (no cross-order
    optimization). Full batch optimization is future work (Slice 2.3+).

    It builds a PoolRegistry from the auction's liquidity data and uses
    the SingleOrderRouter to find routes through available pools.
    """

    def __init__(
        self,
        amm: UniswapV2 | None = None,
        router: SingleOrderRouter | None = None,
        v3_amm: UniswapV3AMM | None = None,
        weighted_amm: BalancerWeightedAMM | None = None,
        stable_amm: BalancerStableAMM | None = None,
    ) -> None:
        """Initialize the AMM routing strategy.

        Args:
            amm: AMM implementation for swap math. Defaults to UniswapV2.
            router: Injected router for testing. If provided, used directly
                    instead of creating one from auction liquidity. Note that
                    injected routers are stateless - reserve updates between
                    orders won't affect the injected router's pool finder.
                    This is acceptable for unit tests with mocked behavior.
            v3_amm: UniswapV3 AMM for V3 pool routing. If None, V3 pools are skipped.
            weighted_amm: Balancer weighted AMM. If None, weighted pools are skipped.
            stable_amm: Balancer stable AMM. If None, stable pools are skipped.
        """
        self.amm = amm if amm is not None else uniswap_v2
        self._injected_router = router
        self.v3_amm = v3_amm
        self.weighted_amm = weighted_amm
        self.stable_amm = stable_amm

    def _get_router(self, pool_registry: PoolRegistry) -> SingleOrderRouter:
        """Get the router to use for routing orders.

        Returns the injected router if available, otherwise creates a new one.

        Args:
            pool_registry: Registry of available liquidity pools

        Returns:
            Router instance to use for routing
        """
        if self._injected_router is not None:
            return self._injected_router
        return SingleOrderRouter(
            amm=self.amm,
            pool_registry=pool_registry,
            v3_amm=self.v3_amm,
            weighted_amm=self.weighted_amm,
            stable_amm=self.stable_amm,
        )

    def _route_and_build(
        self, order: Order, pool_registry: PoolRegistry
    ) -> _OrderRoutingResult | None:
        """Route an order and build the fill and solution.

        This is the core method that handles routing a single order:
        1. Gets the appropriate router
        2. Routes the order through pools
        3. Builds the solution
        4. Creates the fill record

        Args:
            order: The order to route
            pool_registry: Registry of available liquidity pools

        Returns:
            _OrderRoutingResult with fill, solution, and routing details,
            or None if routing fails
        """
        router = self._get_router(pool_registry)

        # Route the order
        routing_result = router.route_order(order)
        if not routing_result.success:
            logger.debug(
                "amm_routing_failed",
                order_uid=order.uid,
                error=routing_result.error,
            )
            return None

        # Build solution from routing result
        solution = router.build_solution(routing_result, solution_id=0)
        if solution is None:
            return None

        # Create fill record
        fill = OrderFill(
            order=order,
            sell_filled=routing_result.amount_in,
            buy_filled=routing_result.amount_out,
        )

        return _OrderRoutingResult(
            fill=fill,
            solution=solution,
            routing_result=routing_result,
        )

    def try_solve(self, auction: AuctionInstance) -> StrategyResult | None:
        """Try to route orders through AMM pools.

        Handles both single-order and multi-order auctions. For multi-order
        auctions, each order is routed independently.

        Args:
            auction: The auction to solve

        Returns:
            A StrategyResult if at least one order is routed, None otherwise
        """
        if auction.order_count == 0:
            return None

        # Build pool registry from auction liquidity
        pool_registry = build_registry_from_liquidity(auction.liquidity)

        logger.debug(
            "amm_strategy_pool_registry",
            pool_count=pool_registry.pool_count,
            liquidity_count=len(auction.liquidity),
        )

        if auction.order_count == 1:
            return self._solve_single_order(auction, pool_registry)

        # Multi-order: route each order independently
        return self._solve_multiple_orders(auction, pool_registry)

    def _solve_single_order(
        self, auction: AuctionInstance, pool_registry: PoolRegistry
    ) -> StrategyResult | None:
        """Route a single order through AMM pools.

        For partially fillable orders, may return a partial fill with
        a remainder order for the unfilled portion.
        """
        order = auction.orders[0]

        result = self._route_and_build(order, pool_registry)
        if result is None:
            return None

        # Check if this is a partial fill
        remainder_orders = []
        if not result.fill.is_complete and order.partially_fillable:
            remainder = result.fill.get_remainder_order()
            if remainder is not None:
                remainder_orders.append(remainder)
                logger.info(
                    "amm_partial_fill_remainder",
                    order_uid=order.uid[:18] + "...",
                    fill_ratio=f"{result.fill.fill_ratio:.1%}",
                    remainder_sell=remainder.sell_amount,
                    remainder_buy=remainder.buy_amount,
                )

        return StrategyResult(
            fills=[result.fill],
            interactions=list(result.solution.interactions),
            prices=result.solution.prices,
            gas=result.solution.gas or 0,
            remainder_orders=remainder_orders,
        )

    def _update_reserves_after_swap(
        self, pool_registry: PoolRegistry, routing_result: RoutingResult
    ) -> None:
        """Update pool reserves in the registry after a successful swap.

        For each hop in the route, calculates new reserves based on the
        swap amounts and updates the pool in the registry.

        Args:
            pool_registry: Registry to update (mutated in place)
            routing_result: The routing result containing hop details
        """
        if routing_result.hops is None:
            # Single-hop route - use pool and amounts from result
            if routing_result.pool is None:
                return

            pool = routing_result.pool
            token_in = routing_result.order.sell_token
            amount_in = routing_result.amount_in
            amount_out = routing_result.amount_out

            # Only update V2 pools (V3 pools use quoter, no local reserve tracking)
            if isinstance(pool, UniswapV2Pool):
                updated_pool = self._create_updated_pool(pool, token_in, amount_in, amount_out)
                pool_registry.add_pool(updated_pool)
            return

        # Multi-hop route - update each pool (multi-hop is V2-only)
        for hop in routing_result.hops:
            if isinstance(hop.pool, UniswapV2Pool):
                updated_pool = self._create_updated_pool(
                    hop.pool, hop.input_token, hop.amount_in, hop.amount_out
                )
                pool_registry.add_pool(updated_pool)

    def _create_updated_pool(
        self,
        pool: UniswapV2Pool,
        token_in: str,
        amount_in: int,
        amount_out: int,
    ) -> UniswapV2Pool:
        """Create a new pool with updated reserves after a swap.

        After a swap:
        - reserve_in increases by amount_in
        - reserve_out decreases by amount_out

        Args:
            pool: Original pool
            token_in: Token being swapped in
            amount_in: Amount of token_in added to pool
            amount_out: Amount of token_out removed from pool

        Returns:
            New UniswapV2Pool with updated reserves
        """
        token_in_norm = token_in.lower()
        is_token0 = token_in_norm == pool.token0.lower()

        if is_token0:
            new_reserve0 = pool.reserve0 + amount_in
            new_reserve1 = pool.reserve1 - amount_out
        else:
            new_reserve0 = pool.reserve0 - amount_out
            new_reserve1 = pool.reserve1 + amount_in

        logger.debug(
            "pool_reserves_updated",
            pool=pool.address[-8:],
            old_reserves=(pool.reserve0, pool.reserve1),
            new_reserves=(new_reserve0, new_reserve1),
        )

        return UniswapV2Pool(
            address=pool.address,
            token0=pool.token0,
            token1=pool.token1,
            reserve0=new_reserve0,
            reserve1=new_reserve1,
            fee_bps=pool.fee_bps,
            liquidity_id=pool.liquidity_id,
        )

    def _solve_multiple_orders(
        self, auction: AuctionInstance, pool_registry: PoolRegistry
    ) -> StrategyResult | None:
        """Route multiple orders independently through AMM pools.

        Each order is routed separately. After each successful route, pool
        reserves are updated so subsequent orders use accurate reserve data.
        Orders that fail routing are skipped (partial success is OK).

        Args:
            auction: The auction containing multiple orders
            pool_registry: Registry of available liquidity pools

        Returns:
            A StrategyResult if at least one order is routed, None otherwise
        """
        all_fills: list[OrderFill] = []
        all_interactions: list[Interaction] = []
        all_prices: dict[str, str] = {}
        all_remainders: list[Order] = []
        total_gas = 0
        failed_order_uids: list[str] = []

        for order in auction.orders:
            result = self._route_and_build(order, pool_registry)

            if result is not None:
                all_fills.append(result.fill)
                all_interactions.extend(result.solution.interactions)
                all_prices.update(result.solution.prices)
                total_gas += result.solution.gas or 0

                # Update pool reserves for next order
                self._update_reserves_after_swap(pool_registry, result.routing_result)

                # Check for partial fill remainder
                if not result.fill.is_complete and order.partially_fillable:
                    remainder = result.fill.get_remainder_order()
                    if remainder is not None:
                        all_remainders.append(remainder)
            else:
                failed_order_uids.append(order.uid[:18] + "...")

        if not all_fills:
            logger.debug(
                "amm_routing_all_orders_failed",
                order_count=auction.order_count,
                failed_order_uids=failed_order_uids,
            )
            return None

        logger.info(
            "amm_routing_multiple_orders",
            total_orders=auction.order_count,
            routed_orders=len(all_fills),
            partial_fills=len(all_remainders),
            failed_count=len(failed_order_uids),
            failed_order_uids=failed_order_uids if failed_order_uids else None,
        )

        return StrategyResult(
            fills=all_fills,
            interactions=all_interactions,
            prices=all_prices,
            gas=total_gas,
            remainder_orders=all_remainders,
        )
