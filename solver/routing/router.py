"""Order routing and solution building.

This module handles routing orders through AMM pools and building solutions
for the CoW Protocol settlement.

Current limitations:
- Only single-order auctions supported
- Partially fillable orders are executed fully or not at all
- Multi-hop routing not yet implemented (Slice 1.4)
"""

from collections.abc import Callable
from dataclasses import dataclass

import structlog

from solver.amm.uniswap_v2 import (
    PoolRegistry,
    UniswapV2,
    UniswapV2Pool,
    build_registry_from_liquidity,
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
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=None,
                success=False,
                error=f"Invalid amount format: {e}",
            )

        if sell_amount <= 0:
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=None,
                success=False,
                error="Sell amount must be positive",
            )

        if buy_amount <= 0:
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=None,
                success=False,
                error="Buy amount must be positive",
            )

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
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=None,
                success=False,
                error=f"No route found for {order.sell_token}/{order.buy_token}",
            )

        # Get pools for the path
        try:
            pools = self._registry.get_all_pools_on_path(path)
        except ValueError as e:
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=None,
                success=False,
                error=str(e),
            )

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
        """
        swap_result = self.amm.simulate_swap(
            pool=pool,
            token_in=order.sell_token,
            amount_in=sell_amount,
        )

        # Check if output meets minimum
        if swap_result.amount_out < min_buy_amount:
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
        """
        swap_result = self.amm.simulate_swap_exact_output(
            pool=pool,
            token_in=order.sell_token,
            amount_out=buy_amount,
        )

        # Check if required input exceeds maximum
        if swap_result.amount_in > max_sell_amount:
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
    """Main solver that processes auction instances.

    The solver builds a PoolRegistry from the auction's liquidity data and
    uses it for routing. This ensures the solver uses the actual available
    liquidity rather than hardcoded pool data.

    Args:
        router: DEPRECATED - Router instance is now created per-auction.
                Only used for testing/mocking purposes.
        amm: AMM implementation for swap math. Defaults to UniswapV2.
    """

    def __init__(
        self,
        router: SingleOrderRouter | None = None,
        amm: UniswapV2 | None = None,
    ) -> None:
        """Initialize the solver with optional dependencies.

        Args:
            router: Legacy router for testing. If provided, used directly.
            amm: AMM implementation. Defaults to uniswap_v2 singleton.
        """
        self._legacy_router = router
        self.amm = amm if amm is not None else uniswap_v2

    def solve(self, auction: AuctionInstance) -> SolverResponse:
        """Solve an auction batch.

        Creates a PoolRegistry from the auction's liquidity and routes
        orders through available pools.

        Currently handles single-order auctions only.

        Args:
            auction: The auction to solve

        Returns:
            SolverResponse with proposed solutions
        """
        if auction.order_count == 0:
            return SolverResponse.empty()

        # Build pool registry from auction liquidity
        pool_registry = build_registry_from_liquidity(auction.liquidity)

        logger.debug(
            "built_pool_registry",
            pool_count=pool_registry.pool_count,
            liquidity_count=len(auction.liquidity),
        )

        # For now, only handle single-order auctions
        if auction.order_count == 1:
            return self._solve_single_order(auction, pool_registry)

        # Multi-order auctions not yet supported
        return SolverResponse.empty()

    def _solve_single_order(
        self, auction: AuctionInstance, pool_registry: PoolRegistry
    ) -> SolverResponse:
        """Solve a single-order auction."""
        order = auction.orders[0]

        # Log if order is partially fillable (we still process it, but execute fully)
        if order.partially_fillable:
            logger.info(
                "partially_fillable_order",
                order_uid=order.uid,
                message="Partially fillable orders are executed fully or not at all",
            )

        # Create router with auction's liquidity
        # Use legacy router if provided (for testing), otherwise create new one
        if self._legacy_router is not None:
            router = self._legacy_router
        else:
            router = SingleOrderRouter(amm=self.amm, pool_registry=pool_registry)

        # Route the order
        routing_result = router.route_order(order)
        if not routing_result.success:
            return SolverResponse.empty()

        # Build solution
        solution = router.build_solution(routing_result, solution_id=0)
        if solution is None:
            return SolverResponse.empty()

        return SolverResponse(solutions=[solution])


# Singleton solver instance
solver = Solver()
