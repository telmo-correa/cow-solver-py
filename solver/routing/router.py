"""Order routing and solution building.

This module handles routing orders through AMM pools and building solutions
for the CoW Protocol settlement.

Supports:
- Single-order auctions and multi-order auctions
- Direct and multi-hop routing through UniswapV2-style pools
- Both sell orders (exact input) and buy orders (exact output)
- Partial fills for partially fillable orders (single-hop only)

For partially fillable orders, when full fill isn't possible due to
insufficient liquidity, the router calculates the maximum partial fill
that still satisfies the order's limit price.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from solver.strategies.base import SolutionStrategy

from solver.amm.uniswap_v2 import (
    PoolRegistry,
    UniswapV2,
    UniswapV2Pool,
    uniswap_v2,
)
from solver.constants import POOL_SWAP_GAS_COST, SETTLEMENT_OVERHEAD
from solver.models.auction import AuctionInstance, Order
from solver.models.solution import (
    Interaction,
    LiquidityInteraction,
    Solution,
    SolverResponse,
    Trade,
    TradeKind,
)
from solver.models.types import normalize_address

logger = structlog.get_logger()


@dataclass
class HopResult:
    """Result of a single hop in a multi-hop route."""

    pool: UniswapV2Pool
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
    pool: UniswapV2Pool | None  # None when no pool found (kept for backward compatibility)
    success: bool
    error: str | None = None
    # Multi-hop routing fields
    path: list[str] | None = None  # Token path for multi-hop swaps
    pools: list[UniswapV2Pool] | None = None  # Pools along the path
    hops: list[HopResult] | None = None  # Detailed results for each hop
    gas_estimate: int = POOL_SWAP_GAS_COST  # Default single-hop gas (60k per swap)

    @property
    def is_multihop(self) -> bool:
        """Check if this is a multi-hop route."""
        return self.path is not None and len(self.path) > 2


class SingleOrderRouter:
    """Routes single orders through AMM pools.

    This router handles one order at a time, finding the best direct pool
    for the token pair. It uses a PoolRegistry to look up available liquidity.

    Args:
        amm: AMM implementation for swap simulation and encoding.
             Defaults to the UniswapV2 singleton.
        pool_registry: Registry of available pools for routing.
                       If None, an empty registry is used.
        pool_finder: DEPRECATED - Use pool_registry instead.
                     Callable to find a pool for a token pair.
    """

    def __init__(
        self,
        amm: UniswapV2 | None = None,
        pool_registry: PoolRegistry | None = None,
        pool_finder: Callable[[str, str], UniswapV2Pool | None] | None = None,
    ) -> None:
        """Initialize the router with optional dependencies.

        Args:
            amm: AMM implementation. Defaults to uniswap_v2 singleton.
            pool_registry: Pool registry for lookups. If None, uses empty registry.
            pool_finder: DEPRECATED - Legacy pool lookup function. If provided
                         without pool_registry, wraps it in a simple registry.
        """
        self.amm = amm if amm is not None else uniswap_v2
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

    def route_order(self, order: Order) -> RoutingResult:
        """Find a route for a single order.

        Handles both sell orders (exact input) and buy orders (exact output).
        Tries direct routing first, then multi-hop if no direct pool exists.

        Args:
            order: The order to route

        Returns:
            RoutingResult with success=True if route found, False otherwise
        """
        # Validate input amounts
        try:
            sell_amount = int(order.sell_amount)
            buy_amount = int(order.buy_amount)
        except (ValueError, TypeError) as e:
            return self._error_result(order, f"Invalid amount format: {e}")

        if sell_amount <= 0:
            return self._error_result(order, "Sell amount must be positive")

        if buy_amount <= 0:
            return self._error_result(order, "Buy amount must be positive")

        # Try direct pool first
        pool = self._pool_finder(order.sell_token, order.buy_token)
        if pool is not None:
            if order.is_sell_order:
                return self._route_sell_order(order, pool, sell_amount, buy_amount)
            else:
                return self._route_buy_order(order, pool, sell_amount, buy_amount)

        # No direct pool - try multi-hop routing using registry
        path = self._registry.find_path(order.sell_token, order.buy_token)
        if path is None or len(path) < 2:
            return self._error_result(
                order, f"No route found for {order.sell_token}/{order.buy_token}"
            )

        # Get pools for the path
        try:
            pools = self._registry.get_all_pools_on_path(path)
        except ValueError as e:
            return self._error_result(order, str(e))

        logger.info(
            "using_multihop_route",
            order_uid=order.uid,
            path=[p[-8:] for p in path],  # Log last 8 chars of addresses
            hops=len(path) - 1,
        )

        if order.is_sell_order:
            return self._route_sell_order_multihop(order, pools, path, sell_amount, buy_amount)
        else:
            return self._route_buy_order_multihop(order, pools, path, sell_amount, buy_amount)

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

        Args:
            order: The order to route
            pool: The pool to swap through
            sell_amount: Original sell amount
            min_buy_amount: Minimum buy amount for the full order

        Returns:
            RoutingResult with the partial fill, or failure if no partial possible
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
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=pool,
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
            return RoutingResult(
                order=order,
                amount_in=max_input,
                amount_out=swap_result.amount_out,
                pool=pool,
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

        hop = HopResult(
            pool=pool,
            input_token=normalize_address(order.sell_token),
            output_token=normalize_address(order.buy_token),
            amount_in=max_input,
            amount_out=swap_result.amount_out,
        )

        return RoutingResult(
            order=order,
            amount_in=max_input,
            amount_out=swap_result.amount_out,
            pool=pool,
            pools=[pool],
            hops=[hop],
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

        Args:
            order: The order to route
            pool: The pool to swap through
            max_sell_amount: Maximum input the user is willing to pay
            buy_amount: Desired output amount for the full order

        Returns:
            RoutingResult with the partial fill, or failure if no partial possible
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
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=pool,
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
            return RoutingResult(
                order=order,
                amount_in=swap_result.amount_in,
                amount_out=max_output,
                pool=pool,
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

        hop = HopResult(
            pool=pool,
            input_token=normalize_address(order.sell_token),
            output_token=normalize_address(order.buy_token),
            amount_in=swap_result.amount_in,
            amount_out=max_output,
        )

        return RoutingResult(
            order=order,
            amount_in=swap_result.amount_in,
            amount_out=max_output,
            pool=pool,
            pools=[pool],
            hops=[hop],
            success=True,
        )

    def _route_sell_order_multihop(
        self,
        order: Order,
        pools: list[UniswapV2Pool],
        path: list[str],
        sell_amount: int,
        min_buy_amount: int,
    ) -> RoutingResult:
        """Route a sell order through multiple hops.

        For sell orders:
        - sell_amount is the exact amount to sell
        - buy_amount is the minimum acceptable output
        """
        # Compute intermediate amounts for each hop
        hops: list[HopResult] = []
        current_amount = sell_amount

        for i, pool in enumerate(pools):
            token_in = normalize_address(path[i])
            token_out = normalize_address(path[i + 1])
            reserve_in, reserve_out = pool.get_reserves(path[i])
            amount_out = self.amm.get_amount_out(
                current_amount, reserve_in, reserve_out, pool.fee_multiplier
            )

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
        gas_estimate = POOL_SWAP_GAS_COST * len(pools)

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
                gas_estimate=gas_estimate,
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
            gas_estimate=gas_estimate,
        )

    def _route_buy_order_multihop(
        self,
        order: Order,
        pools: list[UniswapV2Pool],
        path: list[str],
        max_sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route a buy order through multiple hops.

        For buy orders:
        - buy_amount is the exact amount to receive
        - sell_amount is the maximum willing to pay
        """
        # Work backwards to compute required inputs for each hop
        # First, calculate amounts working backwards
        amounts: list[int] = [0] * (len(pools) + 1)
        amounts[-1] = buy_amount  # Final output is the desired buy amount

        for i in range(len(pools) - 1, -1, -1):
            pool = pools[i]
            token_in = path[i]
            reserve_in, reserve_out = pool.get_reserves(token_in)
            amounts[i] = self.amm.get_amount_in(
                amounts[i + 1], reserve_in, reserve_out, pool.fee_multiplier
            )

        required_input = amounts[0]
        gas_estimate = POOL_SWAP_GAS_COST * len(pools)

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
                gas_estimate=gas_estimate,
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
            gas_estimate=gas_estimate,
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


class Solver:
    """Main solver that processes auction instances using a strategy pattern.

    The solver tries each strategy in order, composing partial results.
    If a strategy returns a result with unfilled remainders, subsequent
    strategies are tried on the remainder orders.

    Default strategies (in priority order):
    1. CowMatchStrategy - Direct peer-to-peer matching (no AMM fees)
    2. AmmRoutingStrategy - Route through liquidity pools

    Args:
        strategies: List of strategies to try in order. If None, uses defaults.
        router: Deprecated. Use AmmRoutingStrategy with injected router instead.
        amm: Deprecated. Use AmmRoutingStrategy with injected AMM instead.
    """

    def __init__(
        self,
        strategies: "list[SolutionStrategy] | None" = None,
        router: SingleOrderRouter | None = None,
        amm: UniswapV2 | None = None,
    ) -> None:
        """Initialize the solver with strategies.

        Args:
            strategies: List of SolutionStrategy instances to try in order.
                       If None, uses [CowMatchStrategy(), AmmRoutingStrategy()].
            router: Deprecated. For backwards compatibility, if provided,
                    creates AmmRoutingStrategy with this router.
            amm: Deprecated. For backwards compatibility, if provided,
                 creates AmmRoutingStrategy with this AMM.
        """
        if strategies is not None:
            self.strategies = strategies
        elif router is not None or amm is not None:
            # Backwards compatibility: create strategies from legacy params
            from solver.strategies import AmmRoutingStrategy, CowMatchStrategy

            self.strategies = [
                CowMatchStrategy(),
                AmmRoutingStrategy(amm=amm, router=router),
            ]
        else:
            # Default strategies
            from solver.strategies import AmmRoutingStrategy, CowMatchStrategy

            self.strategies = [
                CowMatchStrategy(),
                AmmRoutingStrategy(),
            ]

    def solve(self, auction: AuctionInstance) -> SolverResponse:
        """Solve an auction batch by composing strategy results.

        Strategies are tried in priority order. Each strategy may return
        a partial result with remainder orders. Subsequent strategies
        are tried on the remainders until all orders are filled or no
        strategy can make progress.

        This enables composition: e.g., CoW matching fills part of orders,
        then AMM routing fills the remainder.

        Args:
            auction: The auction to solve

        Returns:
            SolverResponse with proposed solutions
        """
        from solver.strategies.base import StrategyResult

        if auction.order_count == 0:
            return SolverResponse.empty()

        # Collect results from strategies
        results: list[StrategyResult] = []
        current_auction = auction

        for strategy in self.strategies:
            if current_auction.order_count == 0:
                break

            result = strategy.try_solve(current_auction)
            if result is not None and result.has_fills:
                logger.info(
                    "strategy_succeeded",
                    strategy=type(strategy).__name__,
                    order_count=current_auction.order_count,
                    fills=len(result.fills),
                    remainders=len(result.remainder_orders),
                )
                results.append(result)

                # If there are remainders, create a sub-auction for next strategy
                if result.remainder_orders:
                    current_auction = self._create_sub_auction(
                        current_auction, result.remainder_orders
                    )
                else:
                    # All orders filled, we're done
                    break

        if not results:
            logger.debug(
                "no_strategy_found_solution",
                order_count=auction.order_count,
                strategies_tried=[type(s).__name__ for s in self.strategies],
            )
            return SolverResponse.empty()

        # Combine all results into final solution
        combined = StrategyResult.combine(results)
        solution = combined.build_solution(solution_id=0)

        return SolverResponse(solutions=[solution])

    def _create_sub_auction(
        self, original: AuctionInstance, orders: list[Order]
    ) -> AuctionInstance:
        """Create a sub-auction with the given orders.

        Preserves tokens and liquidity from the original auction.

        Args:
            original: The original auction
            orders: Orders to include in the sub-auction

        Returns:
            A new AuctionInstance with the specified orders
        """
        logger.debug(
            "sub_auction_created",
            original_order_count=original.order_count,
            remainder_order_count=len(orders),
            order_uids=[o.uid[:18] + "..." for o in orders],
        )
        return original.model_copy(update={"orders": orders})


# Singleton solver instance
solver = Solver()
