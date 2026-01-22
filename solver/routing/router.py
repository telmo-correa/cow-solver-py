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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from solver.amm.uniswap_v2 import (
    PoolRegistry,
    UniswapV2,
    UniswapV2Pool,
    uniswap_v2,
)
from solver.constants import POOL_SWAP_GAS_COST, SETTLEMENT_OVERHEAD
from solver.models.auction import Order
from solver.models.solution import (
    Interaction,
    LiquidityInteraction,
    Solution,
    Trade,
    TradeKind,
)
from solver.models.types import normalize_address

if TYPE_CHECKING:
    from solver.amm.balancer import (
        BalancerStableAMM,
        BalancerStablePool,
        BalancerWeightedAMM,
        BalancerWeightedPool,
    )
    from solver.amm.uniswap_v3 import UniswapV3AMM, UniswapV3Pool

# Type alias for any pool type
AnyPool = "UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool"

logger = structlog.get_logger()


@dataclass
class HopResult:
    """Result of a single hop in a multi-hop route."""

    pool: UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool
    input_token: str
    output_token: str
    amount_in: int
    amount_out: int


@dataclass
class RoutingResult:
    """Result of routing an order."""

    order: Order
    amount_in: int
    amount_out: int
    pool: (
        UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool | None
    )  # None when no pool found
    success: bool
    error: str | None = None
    # Multi-hop routing fields
    path: list[str] | None = None  # Token path for multi-hop swaps
    pools: (
        list[UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool] | None
    ) = None  # Pools along the path
    hops: list[HopResult] | None = None  # Detailed results for each hop
    gas_estimate: int = POOL_SWAP_GAS_COST  # Default single-hop gas (60k per swap)

    @property
    def is_multihop(self) -> bool:
        """Check if this is a multi-hop route."""
        return self.path is not None and len(self.path) > 2


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

        hop = HopResult(
            pool=pool,
            input_token=normalize_address(order.sell_token),
            output_token=normalize_address(order.buy_token),
            amount_in=amount_in,
            amount_out=amount_out,
        )

        return RoutingResult(
            order=order,
            amount_in=amount_in,
            amount_out=amount_out,
            pool=pool,
            pools=[pool],
            hops=[hop],
            success=True,
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
        selection_result = self._select_best_pools_for_path(
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
            return self._route_sell_order_multihop(order, pools, path, sell_amount, buy_amount)
        else:
            return self._route_buy_order_multihop(order, pools, path, sell_amount, buy_amount)

    def _find_best_direct_route(
        self,
        order: Order,
        pools: list[UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool],
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
        from solver.amm.balancer import BalancerStablePool, BalancerWeightedPool
        from solver.amm.uniswap_v3 import UniswapV3Pool

        candidates: list[
            tuple[
                UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool,
                int,
                int,
            ]
        ] = []

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
                if order.is_sell_order:
                    result = self.amm.simulate_swap(pool, order.sell_token, sell_amount)
                    candidates.append((pool, result.amount_in, result.amount_out))
                else:
                    result = self.amm.simulate_swap_exact_output(pool, order.sell_token, buy_amount)
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

        # Route through the best pool
        if isinstance(best_pool, UniswapV3Pool):
            return self._route_through_v3_pool(order, best_pool, sell_amount, buy_amount)
        elif isinstance(best_pool, (BalancerWeightedPool, BalancerStablePool)):
            return self._route_through_balancer_pool(order, best_pool, sell_amount, buy_amount)
        else:
            if order.is_sell_order:
                return self._route_sell_order(order, best_pool, sell_amount, buy_amount)
            else:
                return self._route_buy_order(order, best_pool, sell_amount, buy_amount)

    def _get_pool_type(
        self,
        pool: (UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool),
    ) -> str:
        """Get the type string for a pool."""
        from solver.amm.balancer import BalancerStablePool, BalancerWeightedPool
        from solver.amm.uniswap_v3 import UniswapV3Pool

        if isinstance(pool, UniswapV3Pool):
            return "v3"
        elif isinstance(pool, BalancerWeightedPool):
            return "balancer_weighted"
        elif isinstance(pool, BalancerStablePool):
            return "balancer_stable"
        return "v2"

    def _select_best_pools_for_path(
        self,
        path: list[str],
        amount_in: int,
        is_sell: bool,
    ) -> (
        tuple[
            list[UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool],
            int,
        ]
        | None
    ):
        """Select the best pool for each hop in a multi-hop path based on quotes.

        Uses a greedy approach: for each hop, selects the pool that gives the
        best output given the current input amount. This is O(n*m) where n is
        the number of hops and m is the average number of pools per hop.

        Args:
            path: List of token addresses forming the swap path
            amount_in: Initial input amount for sell orders
            is_sell: True for sell orders (forward simulation),
                     False for buy orders (backward simulation)

        Returns:
            Tuple of (selected_pools, final_amount) if successful, None if any hop fails.
            For sell orders, final_amount is the output. For buy orders, it's the input.
        """
        if not is_sell:
            # For buy orders, we'd need to simulate backward which is more complex.
            # Fall back to the registry's default selection for now.
            try:
                pools = self._registry.get_all_pools_on_path(path)
                return pools, amount_in
            except ValueError:
                return None

        selected_pools: list[
            UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool
        ] = []
        current_amount = amount_in

        for i in range(len(path) - 1):
            token_in = path[i]
            token_out = path[i + 1]
            candidate_pools = self._registry.get_pools_for_pair(token_in, token_out)

            if not candidate_pools:
                return None

            best_pool = None
            best_output = 0

            for pool in candidate_pools:
                output = self._simulate_hop_output(pool, token_in, token_out, current_amount)
                if output is not None and output > best_output:
                    best_output = output
                    best_pool = pool

            if best_pool is None or best_output == 0:
                return None

            selected_pools.append(best_pool)
            current_amount = best_output

        return selected_pools, current_amount

    def _simulate_hop_output(
        self,
        pool: UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> int | None:
        """Simulate a single hop and return the output amount.

        Args:
            pool: The pool to simulate through
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount

        Returns:
            Output amount, or None if simulation fails
        """
        from solver.amm.balancer import BalancerStablePool, BalancerWeightedPool
        from solver.amm.uniswap_v3 import UniswapV3Pool

        if isinstance(pool, UniswapV3Pool):
            if self.v3_amm is None:
                return None
            result = self.v3_amm.simulate_swap(pool, token_in, amount_in)
            return result.amount_out if result else None

        elif isinstance(pool, BalancerWeightedPool):
            if self.weighted_amm is None:
                return None
            result = self.weighted_amm.simulate_swap(pool, token_in, token_out, amount_in)
            return result.amount_out if result else None

        elif isinstance(pool, BalancerStablePool):
            if self.stable_amm is None:
                return None
            result = self.stable_amm.simulate_swap(pool, token_in, token_out, amount_in)
            return result.amount_out if result else None

        else:
            # V2 pool
            try:
                reserve_in, reserve_out = pool.get_reserves(token_in)
                return self.amm.get_amount_out(
                    amount_in, reserve_in, reserve_out, pool.fee_multiplier
                )
            except (ValueError, ZeroDivisionError):
                return None

    def _route_through_v3_pool(
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
            buy_amount: Order's buy amount (minimum for sell orders, exact for buy)

        Returns:
            RoutingResult with the routing outcome
        """
        if self.v3_amm is None:
            return self._error_result(order, "V3: AMM not configured")

        if order.is_sell_order:
            result = self.v3_amm.simulate_swap(pool, order.sell_token, sell_amount)
            if result is None:
                return self._error_result(order, "V3: quote failed")

            # Check if output meets minimum
            if result.amount_out < buy_amount:
                return RoutingResult(
                    order=order,
                    amount_in=sell_amount,
                    amount_out=result.amount_out,
                    pool=pool,
                    success=False,
                    error=f"Output {result.amount_out} below minimum {buy_amount}",
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
        else:
            # Buy order - exact output
            result = self.v3_amm.simulate_swap_exact_output(pool, order.sell_token, buy_amount)
            if result is None:
                return self._error_result(order, "V3: quote failed")

            # Check if required input exceeds maximum
            if result.amount_in > sell_amount:
                return RoutingResult(
                    order=order,
                    amount_in=result.amount_in,
                    amount_out=buy_amount,
                    pool=pool,
                    success=False,
                    error=f"Required input {result.amount_in} exceeds maximum {sell_amount}",
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

    def _route_through_balancer_pool(
        self,
        order: Order,
        pool: BalancerWeightedPool | BalancerStablePool,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route an order through a Balancer pool (weighted or stable).

        Args:
            order: The order to route
            pool: The Balancer pool to use (weighted or stable)
            sell_amount: Order's sell amount
            buy_amount: Order's buy amount (minimum for sell orders, exact for buy)

        Returns:
            RoutingResult with the routing outcome
        """
        from solver.amm.balancer import BalancerWeightedPool

        # Dispatch to type-specific handler
        if isinstance(pool, BalancerWeightedPool):
            return self._route_through_weighted_pool(order, pool, sell_amount, buy_amount)
        else:
            return self._route_through_stable_pool(order, pool, sell_amount, buy_amount)

    def _route_through_weighted_pool(
        self,
        order: Order,
        pool: BalancerWeightedPool,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route an order through a Balancer weighted pool."""
        if self.weighted_amm is None:
            return self._error_result(order, "Balancer weighted: AMM not configured")

        if order.is_sell_order:
            result = self.weighted_amm.simulate_swap(
                pool, order.sell_token, order.buy_token, sell_amount
            )
            if result is None:
                if order.partially_fillable:
                    return self._try_partial_balancer_fill(
                        order, pool, self.weighted_amm, "weighted", sell_amount, buy_amount
                    )
                return self._error_result(order, "Balancer weighted: quote failed")

            if result.amount_out < buy_amount:
                if order.partially_fillable:
                    return self._try_partial_balancer_fill(
                        order, pool, self.weighted_amm, "weighted", sell_amount, buy_amount
                    )
                return RoutingResult(
                    order=order,
                    amount_in=sell_amount,
                    amount_out=result.amount_out,
                    pool=pool,
                    success=False,
                    error=f"Output {result.amount_out} below minimum {buy_amount}",
                )

            return self._build_balancer_result(order, pool, sell_amount, result.amount_out)
        else:
            result = self.weighted_amm.simulate_swap_exact_output(
                pool, order.sell_token, order.buy_token, buy_amount
            )
            if result is None:
                if order.partially_fillable:
                    return self._try_partial_balancer_fill(
                        order, pool, self.weighted_amm, "weighted", sell_amount, buy_amount
                    )
                return self._error_result(order, "Balancer weighted: quote failed")

            if result.amount_in > sell_amount:
                if order.partially_fillable:
                    return self._try_partial_balancer_fill(
                        order, pool, self.weighted_amm, "weighted", sell_amount, buy_amount
                    )
                return RoutingResult(
                    order=order,
                    amount_in=result.amount_in,
                    amount_out=buy_amount,
                    pool=pool,
                    success=False,
                    error=f"Required input {result.amount_in} exceeds maximum {sell_amount}",
                )

            return self._build_balancer_result(order, pool, result.amount_in, buy_amount)

    def _route_through_stable_pool(
        self,
        order: Order,
        pool: BalancerStablePool,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route an order through a Balancer stable pool."""
        if self.stable_amm is None:
            return self._error_result(order, "Balancer stable: AMM not configured")

        if order.is_sell_order:
            result = self.stable_amm.simulate_swap(
                pool, order.sell_token, order.buy_token, sell_amount
            )
            if result is None:
                if order.partially_fillable:
                    return self._try_partial_balancer_fill(
                        order, pool, self.stable_amm, "stable", sell_amount, buy_amount
                    )
                return self._error_result(order, "Balancer stable: quote failed")

            if result.amount_out < buy_amount:
                if order.partially_fillable:
                    return self._try_partial_balancer_fill(
                        order, pool, self.stable_amm, "stable", sell_amount, buy_amount
                    )
                return RoutingResult(
                    order=order,
                    amount_in=sell_amount,
                    amount_out=result.amount_out,
                    pool=pool,
                    success=False,
                    error=f"Output {result.amount_out} below minimum {buy_amount}",
                )

            return self._build_balancer_result(order, pool, sell_amount, result.amount_out)
        else:
            result = self.stable_amm.simulate_swap_exact_output(
                pool, order.sell_token, order.buy_token, buy_amount
            )
            if result is None:
                if order.partially_fillable:
                    return self._try_partial_balancer_fill(
                        order, pool, self.stable_amm, "stable", sell_amount, buy_amount
                    )
                return self._error_result(order, "Balancer stable: quote failed")

            if result.amount_in > sell_amount:
                if order.partially_fillable:
                    return self._try_partial_balancer_fill(
                        order, pool, self.stable_amm, "stable", sell_amount, buy_amount
                    )
                return RoutingResult(
                    order=order,
                    amount_in=result.amount_in,
                    amount_out=buy_amount,
                    pool=pool,
                    success=False,
                    error=f"Required input {result.amount_in} exceeds maximum {sell_amount}",
                )

            return self._build_balancer_result(order, pool, result.amount_in, buy_amount)

    def _build_balancer_result(
        self,
        order: Order,
        pool: BalancerWeightedPool | BalancerStablePool,
        amount_in: int,
        amount_out: int,
    ) -> RoutingResult:
        """Build a successful routing result for a Balancer pool."""
        hop = HopResult(
            pool=pool,
            input_token=normalize_address(order.sell_token),
            output_token=normalize_address(order.buy_token),
            amount_in=amount_in,
            amount_out=amount_out,
        )

        return RoutingResult(
            order=order,
            amount_in=amount_in,
            amount_out=amount_out,
            pool=pool,
            pools=[pool],
            hops=[hop],
            success=True,
            gas_estimate=pool.gas_estimate,
        )

    def _try_partial_balancer_fill(
        self,
        order: Order,
        pool: BalancerWeightedPool | BalancerStablePool,
        amm: Any,  # BalancerWeightedAMM or BalancerStableAMM - caller ensures match
        pool_type: str,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Try to find maximum partial fill for a Balancer pool order.

        This is a unified method for both weighted and stable pools,
        handling both sell and buy orders.

        Args:
            order: The order to partially fill
            pool: The Balancer pool (weighted or stable)
            amm: The AMM instance (BalancerWeightedAMM or BalancerStableAMM).
                 Caller must ensure pool type matches AMM type.
            pool_type: "weighted" or "stable" (for logging)
            sell_amount: Order's sell amount
            buy_amount: Order's buy amount

        Returns:
            RoutingResult with the partial fill outcome
        """
        is_sell = order.is_sell_order
        order_uid_short = order.uid[:18] + "..."

        # Calculate maximum fill amount
        if is_sell:
            max_fill = amm.max_fill_sell_order(
                pool=pool,
                token_in=order.sell_token,
                token_out=order.buy_token,
                sell_amount=sell_amount,
                buy_amount=buy_amount,
            )
        else:
            max_fill = amm.max_fill_buy_order(
                pool=pool,
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
                pool=pool,
                token_in=order.sell_token,
                token_out=order.buy_token,
                amount_in=max_fill,
            )
            fail_amount_in = max_fill
            fail_amount_out = 0
        else:
            result = amm.simulate_swap_exact_output(
                pool=pool,
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
        if is_sell:
            # Sell: output/input >= buy_amount/sell_amount
            limit_satisfied = result.amount_out * sell_amount >= buy_amount * max_fill
            final_in, final_out = max_fill, result.amount_out
            log_key = "partial_sell"
            fill_ratio = f"{max_fill * 100 // sell_amount}%"
        else:
            # Buy: input/output <= sell_amount/buy_amount
            limit_satisfied = result.amount_in * buy_amount <= sell_amount * max_fill
            final_in, final_out = result.amount_in, max_fill
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

        return self._build_balancer_result(order, pool, final_in, final_out)

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

        # Create hop result for single-hop route
        hop = HopResult(
            pool=pool,
            input_token=normalize_address(order.sell_token),
            output_token=normalize_address(order.buy_token),
            amount_in=sell_amount,
            amount_out=swap_result.amount_out,
        )

        return RoutingResult(
            order=order,
            amount_in=sell_amount,
            amount_out=swap_result.amount_out,
            pool=pool,
            pools=[pool],
            hops=[hop],
            success=True,
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

        # Create hop result for single-hop route
        hop = HopResult(
            pool=pool,
            input_token=normalize_address(order.sell_token),
            output_token=normalize_address(order.buy_token),
            amount_in=swap_result.amount_in,
            amount_out=buy_amount,
        )

        return RoutingResult(
            order=order,
            amount_in=swap_result.amount_in,
            amount_out=buy_amount,
            pool=pool,
            pools=[pool],
            hops=[hop],
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

    def _route_sell_order_multihop(
        self,
        order: Order,
        pools: list[UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool],
        path: list[str],
        sell_amount: int,
        min_buy_amount: int,
    ) -> RoutingResult:
        """Route a sell order through multiple hops.

        For sell orders:
        - sell_amount is the exact amount to sell
        - buy_amount is the minimum acceptable output

        Supports V2, V3, and Balancer pools in the multi-hop path.
        """
        from solver.amm.balancer import BalancerStablePool, BalancerWeightedPool
        from solver.amm.uniswap_v3 import UniswapV3Pool

        # Compute intermediate amounts for each hop
        hops: list[HopResult] = []
        current_amount = sell_amount
        total_gas = 0

        for i, pool in enumerate(pools):
            token_in = normalize_address(path[i])
            token_out = normalize_address(path[i + 1])

            # Dispatch to correct AMM based on pool type
            if isinstance(pool, UniswapV3Pool):
                if self.v3_amm is None:
                    return self._error_result(order, "V3: AMM not configured for multi-hop")
                result = self.v3_amm.simulate_swap(pool, path[i], current_amount)
                if result is None:
                    return self._error_result(order, f"V3: swap failed at hop {i}")
                amount_out = result.amount_out
                total_gas += pool.gas_estimate
            elif isinstance(pool, BalancerWeightedPool):
                if self.weighted_amm is None:
                    return self._error_result(
                        order, "Balancer weighted: AMM not configured for multi-hop"
                    )
                result = self.weighted_amm.simulate_swap(pool, path[i], path[i + 1], current_amount)
                if result is None:
                    return self._error_result(order, f"Balancer weighted: swap failed at hop {i}")
                amount_out = result.amount_out
                total_gas += pool.gas_estimate
            elif isinstance(pool, BalancerStablePool):
                if self.stable_amm is None:
                    return self._error_result(
                        order, "Balancer stable: AMM not configured for multi-hop"
                    )
                result = self.stable_amm.simulate_swap(pool, path[i], path[i + 1], current_amount)
                if result is None:
                    return self._error_result(order, f"Balancer stable: swap failed at hop {i}")
                amount_out = result.amount_out
                total_gas += pool.gas_estimate
            else:
                # V2 pool
                reserve_in, reserve_out = pool.get_reserves(path[i])
                amount_out = self.amm.get_amount_out(
                    current_amount, reserve_in, reserve_out, pool.fee_multiplier
                )
                total_gas += POOL_SWAP_GAS_COST

            hops.append(
                HopResult(
                    pool=pool,
                    input_token=token_in,
                    output_token=token_out,
                    amount_in=current_amount,
                    amount_out=amount_out,
                )
            )
            current_amount = amount_out

        final_amount_out = current_amount

        # Check if output meets minimum
        if final_amount_out < min_buy_amount:
            return RoutingResult(
                order=order,
                amount_in=sell_amount,
                amount_out=final_amount_out,
                pool=pools[0],  # First pool for compatibility
                pools=pools,
                path=path,
                hops=hops,
                success=False,
                error=f"Output {final_amount_out} below minimum {min_buy_amount}",
                gas_estimate=total_gas,
            )

        return RoutingResult(
            order=order,
            amount_in=sell_amount,
            amount_out=final_amount_out,
            pool=pools[0],
            pools=pools,
            path=path,
            hops=hops,
            success=True,
            gas_estimate=total_gas,
        )

    def _route_buy_order_multihop(
        self,
        order: Order,
        pools: list[UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool],
        path: list[str],
        max_sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route a buy order through multiple hops.

        For buy orders:
        - buy_amount is the exact amount to receive
        - sell_amount is the maximum willing to pay

        Supports V2, V3, and Balancer pools in the multi-hop path.
        """
        from solver.amm.balancer import BalancerStablePool, BalancerWeightedPool
        from solver.amm.uniswap_v3 import UniswapV3Pool

        # Work backwards to compute required inputs for each hop
        amounts: list[int] = [0] * (len(pools) + 1)
        amounts[-1] = buy_amount  # Final output is the desired buy amount
        total_gas = 0

        for i in range(len(pools) - 1, -1, -1):
            pool = pools[i]
            token_in = path[i]
            token_out = path[i + 1]

            # Dispatch to correct AMM based on pool type
            if isinstance(pool, UniswapV3Pool):
                if self.v3_amm is None:
                    return self._error_result(order, "V3: AMM not configured for multi-hop")
                result = self.v3_amm.simulate_swap_exact_output(pool, token_in, amounts[i + 1])
                if result is None:
                    return self._error_result(order, f"V3: exact output failed at hop {i}")
                amounts[i] = result.amount_in
                total_gas += pool.gas_estimate
            elif isinstance(pool, BalancerWeightedPool):
                if self.weighted_amm is None:
                    return self._error_result(
                        order, "Balancer weighted: AMM not configured for multi-hop"
                    )
                result = self.weighted_amm.simulate_swap_exact_output(
                    pool, token_in, token_out, amounts[i + 1]
                )
                if result is None:
                    return self._error_result(
                        order, f"Balancer weighted: exact output failed at hop {i}"
                    )
                amounts[i] = result.amount_in
                total_gas += pool.gas_estimate
            elif isinstance(pool, BalancerStablePool):
                if self.stable_amm is None:
                    return self._error_result(
                        order, "Balancer stable: AMM not configured for multi-hop"
                    )
                result = self.stable_amm.simulate_swap_exact_output(
                    pool, token_in, token_out, amounts[i + 1]
                )
                if result is None:
                    return self._error_result(
                        order, f"Balancer stable: exact output failed at hop {i}"
                    )
                amounts[i] = result.amount_in
                total_gas += pool.gas_estimate
            else:
                # V2 pool
                reserve_in, reserve_out = pool.get_reserves(token_in)
                amounts[i] = self.amm.get_amount_in(
                    amounts[i + 1], reserve_in, reserve_out, pool.fee_multiplier
                )
                total_gas += POOL_SWAP_GAS_COST

        required_input = amounts[0]

        # Check if required input exceeds maximum
        if required_input > max_sell_amount:
            return RoutingResult(
                order=order,
                amount_in=required_input,
                amount_out=buy_amount,
                pool=pools[0],
                pools=pools,
                path=path,
                success=False,
                error=f"Required input {required_input} exceeds maximum {max_sell_amount}",
                gas_estimate=total_gas,
            )

        # Now build hop results with actual amounts
        hops: list[HopResult] = []
        for i, pool in enumerate(pools):
            token_in = normalize_address(path[i])
            token_out = normalize_address(path[i + 1])
            hops.append(
                HopResult(
                    pool=pool,
                    input_token=token_in,
                    output_token=token_out,
                    amount_in=amounts[i],
                    amount_out=amounts[i + 1],
                )
            )

        return RoutingResult(
            order=order,
            amount_in=required_input,
            amount_out=buy_amount,
            pool=pools[0],
            pools=pools,
            path=path,
            hops=hops,
            success=True,
            gas_estimate=total_gas,
        )

    def build_solution(
        self,
        routing_result: RoutingResult,
        solution_id: int = 0,
    ) -> Solution | None:
        """Build a complete solution from a routing result.

        Creates LiquidityInteraction objects for each hop, referencing the
        liquidity pools by their auction ID. This matches the Rust solver's
        output format and enables CIP-2 internalization.

        Args:
            routing_result: The result of routing an order
            solution_id: ID for this solution

        Returns:
            Solution if successful, None otherwise
        """
        if not routing_result.success:
            logger.debug(
                "build_solution_skipped_failed_routing",
                order_uid=routing_result.order.uid[:18] + "...",
                error=routing_result.error,
            )
            return None

        if not routing_result.hops:
            logger.error("routing_result_missing_hops", order=routing_result.order.uid)
            return None

        order = routing_result.order

        # Build the trade (order execution)
        # executedAmount semantics:
        # - For sell orders: the amount sold (amount_in)
        # - For buy orders: the amount bought (amount_out)
        if order.is_sell_order:
            executed_amount = routing_result.amount_in
        else:
            executed_amount = routing_result.amount_out

        trade = Trade(
            kind=TradeKind.FULFILLMENT,
            order=order.uid,
            executedAmount=str(executed_amount),
        )

        # Build LiquidityInteraction for each hop
        # This references the auction's liquidity by ID, allowing the driver
        # to use CIP-2 internalization when possible.
        interactions: list[Interaction] = []
        for hop in routing_result.hops:
            if hop.pool.liquidity_id is None:
                logger.error(
                    "pool_missing_liquidity_id",
                    pool_address=hop.pool.address,
                    order=order.uid,
                )
                return None

            interaction = LiquidityInteraction(
                internalize=True,  # Enable CIP-2 internalization
                id=hop.pool.liquidity_id,
                inputToken=hop.input_token,
                outputToken=hop.output_token,
                inputAmount=str(hop.amount_in),
                outputAmount=str(hop.amount_out),
            )
            interactions.append(interaction)

        # Calculate clearing prices
        # CoW Protocol uses uniform clearing prices where:
        #   executed_sell * price[sell_token] >= executed_buy * price[buy_token]
        #
        # The pricing works the same for both sell and buy orders:
        # We set prices such that the exchange rate is amount_out / amount_in
        #
        # Using buy_token as the reference (price = amount_in) to match Rust:
        #   price[sell_token] = amount_out
        #   price[buy_token] = amount_in
        #
        # This maintains the ratio: amount_in / amount_out = price[buy] / price[sell]
        #
        # Note: Addresses are normalized to lowercase for consistent lookup
        sell_token_normalized = normalize_address(order.sell_token)
        buy_token_normalized = normalize_address(order.buy_token)
        prices = {
            buy_token_normalized: str(routing_result.amount_in),
            sell_token_normalized: str(routing_result.amount_out),
        }

        # Calculate total gas: 60k per swap + 106k settlement overhead
        # Formula: sum(POOL_SWAP_GAS_COST per hop) + SETTLEMENT_OVERHEAD
        # Source: cow-services/crates/shared/src/price_estimation/gas.rs
        gas_estimate = routing_result.gas_estimate + SETTLEMENT_OVERHEAD

        return Solution(
            id=solution_id,
            prices=prices,
            trades=[trade],
            interactions=interactions,
            gas=gas_estimate,
        )


__all__ = ["SingleOrderRouter", "RoutingResult", "HopResult"]
