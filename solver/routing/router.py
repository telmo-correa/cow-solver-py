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
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from solver.amm.balancer import (
    BalancerStableAMM,
    BalancerStablePool,
    BalancerWeightedAMM,
    BalancerWeightedPool,
)
from solver.amm.limit_order import LimitOrderAMM
from solver.amm.uniswap_v2 import (
    UniswapV2,
    UniswapV2Pool,
    uniswap_v2,
)
from solver.models.auction import Order
from solver.models.solution import Solution
from solver.pools import AnyPool, LimitOrderPool, PoolRegistry
from solver.routing.handlers import (
    BalancerHandler,
    LimitOrderHandler,
    UniswapV2Handler,
    UniswapV3Handler,
)
from solver.routing.multihop import MultihopRouter
from solver.routing.registry import HandlerRegistry
from solver.routing.solution import build_solution
from solver.routing.types import HopResult, RoutingResult

if TYPE_CHECKING:
    from solver.amm.uniswap_v3 import UniswapV3AMM

logger = structlog.get_logger()


class SingleOrderRouter:
    """Routes single orders through AMM pools and limit orders.

    This router handles one order at a time, finding the best pool
    for the token pair across V2, V3, Balancer, and 0x limit order liquidity.

    Args:
        amm: AMM implementation for V2 swap simulation and encoding.
             Defaults to the UniswapV2 singleton.
        v3_amm: AMM implementation for V3 swap simulation and encoding.
                If None, V3 pools are not used for routing.
        weighted_amm: AMM implementation for Balancer weighted pools.
                      If None, weighted pools are not used for routing.
        stable_amm: AMM implementation for Balancer stable pools.
                    If None, stable pools are not used for routing.
        limit_order_amm: AMM implementation for 0x limit orders.
                         If None, limit orders are not used for routing.
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
        limit_order_amm: LimitOrderAMM | None = None,
        pool_registry: PoolRegistry | None = None,
        pool_finder: Callable[[str, str], UniswapV2Pool | None] | None = None,
    ) -> None:
        """Initialize the router with optional dependencies.

        Args:
            amm: AMM implementation for V2. Defaults to uniswap_v2 singleton.
            v3_amm: AMM implementation for V3. If None, V3 is disabled.
            weighted_amm: AMM implementation for Balancer weighted pools. If None, disabled.
            stable_amm: AMM implementation for Balancer stable pools. If None, disabled.
            limit_order_amm: AMM implementation for 0x limit orders. If None, disabled.
            pool_registry: Pool registry for lookups. If None, uses empty registry.
            pool_finder: DEPRECATED - Legacy pool lookup function. If provided
                         without pool_registry, wraps it in a simple registry.
        """
        self.amm = amm if amm is not None else uniswap_v2
        self.v3_amm = v3_amm
        self.weighted_amm = weighted_amm
        self.stable_amm = stable_amm
        self.limit_order_amm = limit_order_amm
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
        self._limit_order_handler = LimitOrderHandler(limit_order_amm)

        # Initialize handler registry for centralized dispatch
        self._handler_registry = self._build_handler_registry()

        # Initialize multi-hop router with the handler registry
        self._multihop = MultihopRouter(
            v2_amm=self.amm,
            v3_amm=v3_amm,
            weighted_amm=weighted_amm,
            stable_amm=stable_amm,
            registry=self._registry,
            handler_registry=self._handler_registry,
        )

    def _build_handler_registry(self) -> HandlerRegistry:
        """Build the handler registry with all configured pool types.

        Returns:
            HandlerRegistry with handlers for all available pool types

        Note: The registry maps specific pool types to handlers. The Protocol-based
        simulator types accept AnyPool, and the registry ensures type-correct
        dispatch at runtime based on pool type.
        """
        from solver.amm.uniswap_v3 import UniswapV3Pool

        registry = HandlerRegistry()

        # Register V2 (always available)
        registry.register(
            UniswapV2Pool,
            handler=self._v2_handler,  # type: ignore[arg-type]
            simulator=lambda p, ti, _to, ai: self.amm.simulate_swap(p, ti, ai),
            exact_output_simulator=lambda p, ti, _to, ao: self.amm.simulate_swap_exact_output(
                p,
                ti,
                ao,
            ),
            type_name="v2",
            gas_estimate=lambda p: p.gas_estimate,
        )

        # Register V3 (if available)
        if self.v3_amm is not None:
            v3_amm = self.v3_amm  # Capture for lambda
            registry.register(
                UniswapV3Pool,
                handler=self._v3_handler,  # type: ignore[arg-type]
                simulator=lambda p, ti, _to, ai: v3_amm.simulate_swap(p, ti, ai),
                exact_output_simulator=lambda p, ti, _to, ao: v3_amm.simulate_swap_exact_output(
                    p,
                    ti,
                    ao,
                ),
                type_name="v3",
                gas_estimate=lambda p: p.gas_estimate,
            )

        # Register Balancer weighted (if available)
        if self.weighted_amm is not None:
            weighted_amm = self.weighted_amm  # Capture for lambda
            registry.register(
                BalancerWeightedPool,
                handler=self._balancer_handler,  # type: ignore[arg-type]
                simulator=lambda p, ti, to, ai: weighted_amm.simulate_swap(p, ti, to, ai),
                exact_output_simulator=lambda p,
                ti,
                to,
                ao: weighted_amm.simulate_swap_exact_output(
                    p,
                    ti,
                    to,
                    ao,
                ),
                type_name="balancer_weighted",
                gas_estimate=lambda p: p.gas_estimate,
            )

        # Register Balancer stable (if available)
        if self.stable_amm is not None:
            stable_amm = self.stable_amm  # Capture for lambda
            registry.register(
                BalancerStablePool,
                handler=self._balancer_handler,  # type: ignore[arg-type]
                simulator=lambda p, ti, to, ai: stable_amm.simulate_swap(p, ti, to, ai),
                exact_output_simulator=lambda p, ti, to, ao: stable_amm.simulate_swap_exact_output(
                    p,
                    ti,
                    to,
                    ao,
                ),
                type_name="balancer_stable",
                gas_estimate=lambda p: p.gas_estimate,
            )

        # Register 0x limit orders (if available)
        if self.limit_order_amm is not None:
            lo_amm = self.limit_order_amm  # Capture for lambda
            registry.register(
                LimitOrderPool,
                handler=self._limit_order_handler,  # type: ignore[arg-type]
                simulator=lambda p, ti, to, ai: lo_amm.simulate_swap(p, ti, to, ai),
                exact_output_simulator=lambda p, ti, to, ao: lo_amm.simulate_swap_exact_output(
                    p,
                    ti,
                    to,
                    ao,
                ),
                type_name="limit_order",
                gas_estimate=lambda p: p.gas_estimate,
            )

        return registry

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
        Generates all candidate paths (direct + multi-hop), estimates each,
        and selects the best based on output (sell) or input (buy).

        This matches Rust's baseline solver behavior which considers all paths
        and picks the optimal one rather than preferring direct routes.

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

        # Get all candidate paths (direct + multi-hop)
        candidate_paths = self._registry.get_all_candidate_paths(order.sell_token, order.buy_token)

        if not candidate_paths:
            # Try legacy pool_finder for backward compatibility
            legacy_pool = self._pool_finder(order.sell_token, order.buy_token)
            if legacy_pool is not None:
                # Create a direct path for the legacy pool
                candidate_paths = [[order.sell_token, order.buy_token]]

        if not candidate_paths:
            return self._error_result(
                order, f"No route found for {order.sell_token}/{order.buy_token}"
            )

        # Estimate each path and collect successful results
        path_results: list[tuple[list[str], RoutingResult]] = []

        for path in candidate_paths:
            result = self._estimate_path(order, path, sell_amount, buy_amount)
            if result is not None and result.success:
                path_results.append((path, result))

        if not path_results:
            return self._error_result(
                order, f"No route found for {order.sell_token}/{order.buy_token}"
            )

        # Select the best path based on order type
        if order.is_sell_order:
            # For sell orders, maximize output
            best_path, best_result = max(path_results, key=lambda x: x[1].amount_out)
        else:
            # For buy orders, minimize input
            best_path, best_result = min(path_results, key=lambda x: x[1].amount_in)

        # Log if we chose a multi-hop route
        if len(best_path) > 2:
            logger.info(
                "using_multihop_route",
                order_uid=order.uid,
                path=[p[-8:] for p in best_path],
                hops=len(best_path) - 1,
            )

        return best_result

    def get_reference_price(
        self,
        token_in: str,
        token_out: str,
        probe_amount: int | None = None,
    ) -> Decimal | None:
        """Get the reference market price for a token pair.

        Queries all available pools for the pair and returns the best price
        (highest output per input). This price can be used as a reference
        for CoW matching - orders that cross this price can potentially
        match directly instead of routing through AMMs.

        The price is calculated by simulating a small swap to get the
        marginal exchange rate at current liquidity levels.

        Args:
            token_in: Token being sold (numerator token)
            token_out: Token being bought (denominator token)
            probe_amount: Amount to use for price discovery. If None,
                         uses 1e15 (0.001 with 18 decimals) as a small
                         probe that minimizes price impact.

        Returns:
            Price as Decimal (token_out per token_in), or None if no
            liquidity exists for the pair.

        Example:
            >>> price = router.get_reference_price(WETH, USDC)
            >>> # price = Decimal("2500.00") means 1 WETH = 2500 USDC
        """
        # Default probe amount: 0.001 tokens (with 18 decimals)
        # Small enough to minimize price impact, large enough to avoid dust issues
        if probe_amount is None:
            probe_amount = 10**15

        # Get all pools for this pair
        pools = self._registry.get_pools_for_pair(token_in, token_out)

        if not pools:
            return None

        best_price: Decimal | None = None

        for pool in pools:
            # Skip pools without registered handlers
            if not self._handler_registry.is_registered(pool):
                continue

            # Simulate a small swap to get the exchange rate
            result = self._handler_registry.simulate_swap(pool, token_in, token_out, probe_amount)

            if result is None or result.amount_out <= 0:
                continue

            # Calculate price: output / input
            price = Decimal(result.amount_out) / Decimal(result.amount_in)

            # Keep the best price (highest output per input)
            if best_price is None or price > best_price:
                best_price = price

        return best_price

    def _estimate_path(
        self,
        order: Order,
        path: list[str],
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult | None:
        """Estimate a single path and return a routing result.

        For direct paths (2 tokens), uses the best direct pool.
        For multi-hop paths (3+ tokens), uses multi-hop routing.

        Args:
            order: The order to route
            path: Token path to estimate
            sell_amount: Order's sell amount
            buy_amount: Order's buy amount

        Returns:
            RoutingResult if path is valid, None otherwise
        """
        if len(path) < 2:
            return None

        if len(path) == 2:
            # Direct path - use best pool for this pair
            pools = self._registry.get_pools_for_pair(path[0], path[1])

            # Also check legacy pool_finder
            if not pools:
                legacy_pool = self._pool_finder(path[0], path[1])
                if legacy_pool is not None:
                    pools = [legacy_pool]

            if not pools:
                return None

            return self._find_best_direct_route(order, pools, sell_amount, buy_amount)
        else:
            # Multi-hop path
            selection_result = self._multihop.select_best_pools_for_path(
                path, sell_amount, is_sell=order.is_sell_order
            )
            if selection_result is None:
                return None

            pools, _ = selection_result

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
        candidates: list[tuple[AnyPool, int, int]] = []

        for pool in pools:
            # Skip pools without registered handlers
            if not self._handler_registry.is_registered(pool):
                continue

            if order.is_sell_order:
                result = self._handler_registry.simulate_swap(
                    pool, order.sell_token, order.buy_token, sell_amount
                )
                if result is not None:
                    candidates.append((pool, result.amount_in, result.amount_out))
            else:
                result = self._handler_registry.simulate_swap_exact_output(
                    pool, order.sell_token, order.buy_token, buy_amount
                )
                if result is not None:
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
                pool_type=self._handler_registry.get_type_name(best_pool),
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
        handler = self._handler_registry.get_handler(pool)
        if handler is None:
            return self._error_result(
                order, f"No handler registered for pool type: {type(pool).__name__}"
            )
        return handler.route(order, pool, sell_amount, buy_amount)

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
