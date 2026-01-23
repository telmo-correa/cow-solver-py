"""Order routing and solution building.

This module handles routing orders through AMM pools and building solutions
for the CoW Protocol settlement.

Supports:
- Single-order auctions and multi-order auctions
- Direct and multi-hop routing through UniswapV2 and UniswapV3 pools
- Both sell orders (exact input) and buy orders (exact output)
- Partial fills for partially fillable orders (single-hop only)
- Best-quote selection across V2 and V3 pools

For partially fillable orders, when full fill isn't possible due to
insufficient liquidity, the router calculates the maximum partial fill
that still satisfies the order's limit price.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog

from solver.amm.balancer import (
    BalancerStableAMM,
    BalancerStablePool,
    BalancerWeightedAMM,
    BalancerWeightedPool,
)
from solver.amm.uniswap_v2 import (
    UniswapV2,
    UniswapV2Pool,
    uniswap_v2,
)
from solver.models.auction import Order
from solver.models.solution import Solution
from solver.pools import AnyPool, PoolRegistry
from solver.routing.handlers import BalancerHandler, UniswapV2Handler, UniswapV3Handler
from solver.routing.multihop import MultihopRouter
from solver.routing.solution import build_solution
from solver.routing.types import HopResult, RoutingResult

if TYPE_CHECKING:
    from solver.amm.uniswap_v3 import UniswapV3AMM

logger = structlog.get_logger()


class SingleOrderRouter:
    """Routes single orders through AMM pools.

    This router handles one order at a time, finding the best pool
    for the token pair across V2, V3, and Balancer liquidity.

    Args:
        amm: AMM implementation for V2 swap simulation and encoding.
             Defaults to the UniswapV2 singleton.
        v3_amm: AMM implementation for V3 swap simulation and encoding.
                If None, V3 pools are not used for routing.
        weighted_amm: AMM implementation for Balancer weighted pools.
                      If None, weighted pools are not used for routing.
        stable_amm: AMM implementation for Balancer stable pools.
                    If None, stable pools are not used for routing.
        pool_registry: Registry of available pools for routing.
                       If None, an empty registry is used.
        pool_finder: DEPRECATED - Use pool_registry instead.
                     Callable to find a pool for a token pair.
    """

    def __init__(
        self,
        amm: UniswapV2 | None = None,
        v3_amm: UniswapV3AMM | None = None,
        weighted_amm: BalancerWeightedAMM | None = None,
        stable_amm: BalancerStableAMM | None = None,
        pool_registry: PoolRegistry | None = None,
        pool_finder: Callable[[str, str], UniswapV2Pool | None] | None = None,
    ) -> None:
        """Initialize the router with optional dependencies.

        Args:
            amm: AMM implementation for V2. Defaults to uniswap_v2 singleton.
            v3_amm: AMM implementation for V3. If None, V3 is disabled.
            weighted_amm: AMM implementation for Balancer weighted pools. If None, disabled.
            stable_amm: AMM implementation for Balancer stable pools. If None, disabled.
            pool_registry: Pool registry for lookups. If None, uses empty registry.
            pool_finder: DEPRECATED - Legacy pool lookup function. If provided
                         without pool_registry, wraps it in a simple registry.
        """
        self.amm = amm if amm is not None else uniswap_v2
        self.v3_amm = v3_amm
        self.weighted_amm = weighted_amm
        self.stable_amm = stable_amm
        self._pool_finder: Callable[[str, str], UniswapV2Pool | None]

        # Handle pool_registry vs legacy pool_finder
        if pool_registry is not None:
            self._registry = pool_registry
            self._pool_finder = pool_registry.get_pool
        elif pool_finder is not None:
            # Legacy mode: use provided pool_finder directly
            self._registry = PoolRegistry()
            self._pool_finder = pool_finder
        else:
            # Default: empty registry
            self._registry = PoolRegistry()
            self._pool_finder = self._registry.get_pool

        # Initialize handlers
        self._v2_handler = UniswapV2Handler(self.amm)
        self._v3_handler = UniswapV3Handler(v3_amm)
        self._balancer_handler = BalancerHandler(weighted_amm, stable_amm)

        # Initialize multi-hop router
        self._multihop = MultihopRouter(
            v2_amm=self.amm,
            v3_amm=v3_amm,
            weighted_amm=weighted_amm,
            stable_amm=stable_amm,
            registry=self._registry,
        )

    # Backward compatibility property
    @property
    def pool_finder(self) -> Callable[[str, str], UniswapV2Pool | None]:
        """Return the pool finder function (for backward compatibility)."""
        return self._pool_finder

    def _error_result(self, order: Order, error: str) -> RoutingResult:
        """Create a failed routing result with the given error message."""
        return RoutingResult(
            order=order,
            amount_in=0,
            amount_out=0,
            pool=None,
            success=False,
            error=error,
        )

    def route_order(self, order: Order) -> RoutingResult:
        """Find a route for a single order.

        Handles both sell orders (exact input) and buy orders (exact output).
        Gets quotes from all available pools (V2 + V3) and selects the best.
        Falls back to multi-hop routing if no direct pool exists.

        Args:
            order: The order to route

        Returns:
            RoutingResult with success=True if route found, False otherwise
        """
        # Validate input amounts
        try:
            sell_amount = order.sell_amount_int
            buy_amount = order.buy_amount_int
        except (ValueError, TypeError) as e:
            return self._error_result(order, f"Invalid amount format: {e}")

        if sell_amount <= 0:
            return self._error_result(order, "Sell amount must be positive")

        if buy_amount <= 0:
            return self._error_result(order, "Buy amount must be positive")

        # Get all pools (V2 + V3) for this token pair from registry
        all_pools = self._registry.get_pools_for_pair(order.sell_token, order.buy_token)

        # Also check legacy pool_finder for backward compatibility
        # This handles cases where pool_finder returns a pool not in the registry
        if not all_pools:
            legacy_pool = self._pool_finder(order.sell_token, order.buy_token)
            if legacy_pool is not None:
                all_pools = [legacy_pool]

        if all_pools:
            # Find the best pool based on quote (V2, V3, and Balancer pools)
            best_result = self._find_best_direct_route(order, all_pools, sell_amount, buy_amount)
            if best_result is not None:
                return best_result

        # No direct pool or all failed - try multi-hop routing using registry
        path = self._registry.find_path(order.sell_token, order.buy_token)
        if path is None or len(path) < 2:
            return self._error_result(
                order, f"No route found for {order.sell_token}/{order.buy_token}"
            )

        # Select best pools for each hop based on quotes
        selection_result = self._multihop.select_best_pools_for_path(
            path, sell_amount, is_sell=order.is_sell_order
        )
        if selection_result is None:
            return self._error_result(order, f"No valid pools found for multi-hop path {path}")
        pools, _ = selection_result

        logger.info(
            "using_multihop_route",
            order_uid=order.uid,
            path=[p[-8:] for p in path],  # Log last 8 chars of addresses
            hops=len(path) - 1,
            pool_types=[self._get_pool_type(p) for p in pools],
        )

        if order.is_sell_order:
            return self._multihop.route_sell_order(order, pools, path, sell_amount, buy_amount)
        else:
            return self._multihop.route_buy_order(order, pools, path, sell_amount, buy_amount)

    def _find_best_direct_route(
        self,
        order: Order,
        pools: list[AnyPool],
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult | None:
        """Find the best route among candidate pools.

        Gets quotes from all pools and selects the best based on:
        - For sell orders: highest output amount
        - For buy orders: lowest input amount

        Args:
            order: The order to route
            pools: List of candidate pools (V2, V3, Balancer weighted, Balancer stable)
            sell_amount: Order's sell amount
            buy_amount: Order's buy amount

        Returns:
            Best RoutingResult, or None if all pools failed
        """
        from solver.amm.uniswap_v3 import UniswapV3Pool

        candidates: list[tuple[AnyPool, int, int]] = []

        for pool in pools:
            if isinstance(pool, UniswapV3Pool):
                # V3 pool - use V3 AMM
                if self.v3_amm is None:
                    continue

                if order.is_sell_order:
                    result = self.v3_amm.simulate_swap(pool, order.sell_token, sell_amount)
                    if result is not None:
                        candidates.append((pool, result.amount_in, result.amount_out))
                else:
                    result = self.v3_amm.simulate_swap_exact_output(
                        pool, order.sell_token, buy_amount
                    )
                    if result is not None:
                        candidates.append((pool, result.amount_in, result.amount_out))

            elif isinstance(pool, BalancerWeightedPool):
                # Balancer weighted pool - use weighted AMM
                if self.weighted_amm is None:
                    continue

                if order.is_sell_order:
                    result = self.weighted_amm.simulate_swap(
                        pool, order.sell_token, order.buy_token, sell_amount
                    )
                    if result is not None:
                        candidates.append((pool, result.amount_in, result.amount_out))
                else:
                    result = self.weighted_amm.simulate_swap_exact_output(
                        pool, order.sell_token, order.buy_token, buy_amount
                    )
                    if result is not None:
                        candidates.append((pool, result.amount_in, result.amount_out))

            elif isinstance(pool, BalancerStablePool):
                # Balancer stable pool - use stable AMM
                if self.stable_amm is None:
                    continue

                if order.is_sell_order:
                    result = self.stable_amm.simulate_swap(
                        pool, order.sell_token, order.buy_token, sell_amount
                    )
                    if result is not None:
                        candidates.append((pool, result.amount_in, result.amount_out))
                else:
                    result = self.stable_amm.simulate_swap_exact_output(
                        pool, order.sell_token, order.buy_token, buy_amount
                    )
                    if result is not None:
                        candidates.append((pool, result.amount_in, result.amount_out))

            else:
                # V2 pool - use V2 AMM
                v2_pool = pool  # type is narrowed by isinstance checks above
                if order.is_sell_order:
                    result = self.amm.simulate_swap(v2_pool, order.sell_token, sell_amount)
                    candidates.append((pool, result.amount_in, result.amount_out))
                else:
                    result = self.amm.simulate_swap_exact_output(
                        v2_pool, order.sell_token, buy_amount
                    )
                    candidates.append((pool, result.amount_in, result.amount_out))

        if not candidates:
            return None

        # Select best candidate
        if order.is_sell_order:
            # For sell orders, maximize output
            best_pool, best_in, best_out = max(candidates, key=lambda x: x[2])
        else:
            # For buy orders, minimize input
            best_pool, best_in, best_out = min(candidates, key=lambda x: x[1])

        # Log pool selection if multiple candidates
        if len(candidates) > 1:
            logger.info(
                "best_pool_selected",
                order_uid=order.uid[:18] + "...",
                pool_type=self._get_pool_type(best_pool),
                pool_address=best_pool.address[:10] + "...",
                candidates=len(candidates),
                amount_in=best_in,
                amount_out=best_out,
            )

        # Route through the best pool using the appropriate handler
        return self._route_through_pool(order, best_pool, sell_amount, buy_amount)

    def _route_through_pool(
        self,
        order: Order,
        pool: AnyPool,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route an order through a specific pool using the appropriate handler.

        Args:
            order: The order to route
            pool: The pool to route through
            sell_amount: Order's sell amount
            buy_amount: Order's buy amount

        Returns:
            RoutingResult from the appropriate handler
        """
        from solver.amm.uniswap_v3 import UniswapV3Pool

        if isinstance(pool, UniswapV3Pool):
            return self._v3_handler.route(order, pool, sell_amount, buy_amount)
        elif isinstance(pool, (BalancerWeightedPool, BalancerStablePool)):
            return self._balancer_handler.route(order, pool, sell_amount, buy_amount)
        else:
            # V2 pool
            v2_pool = pool  # type is narrowed by isinstance checks above
            return self._v2_handler.route(order, v2_pool, sell_amount, buy_amount)

    def _get_pool_type(self, pool: AnyPool) -> str:
        """Get the type string for a pool."""
        from solver.amm.uniswap_v3 import UniswapV3Pool

        if isinstance(pool, UniswapV3Pool):
            return "v3"
        elif isinstance(pool, BalancerWeightedPool):
            return "balancer_weighted"
        elif isinstance(pool, BalancerStablePool):
            return "balancer_stable"
        return "v2"

    def build_solution(
        self,
        routing_result: RoutingResult,
        solution_id: int = 0,
    ) -> Solution | None:
        """Build a complete solution from a routing result.

        Delegates to the solution module's build_solution function.

        Args:
            routing_result: The result of routing an order
            solution_id: ID for this solution

        Returns:
            Solution if successful, None otherwise
        """
        return build_solution(routing_result, solution_id)


__all__ = ["SingleOrderRouter", "RoutingResult", "HopResult"]
