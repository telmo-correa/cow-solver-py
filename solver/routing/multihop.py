"""Multi-hop routing through multiple pools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from solver.amm.balancer import (
    BalancerStableAMM,
    BalancerStablePool,
    BalancerWeightedAMM,
    BalancerWeightedPool,
)
from solver.amm.uniswap_v2 import UniswapV2
from solver.constants import (
    BALANCER_STABLE_SWAP_GAS_COST,
    BALANCER_WEIGHTED_SWAP_GAS_COST,
    POOL_SWAP_GAS_COST,
)
from solver.models.auction import Order
from solver.models.types import normalize_address
from solver.pools import AnyPool, LimitOrderPool, PoolRegistry
from solver.routing.types import HopResult, RoutingResult

if TYPE_CHECKING:
    from solver.amm.uniswap_v3 import UniswapV3AMM
    from solver.routing.registry import HandlerRegistry


class MultihopRouter:
    """Handles multi-hop routing through multiple pools.

    Supports routing through any combination of V2, V3, and Balancer pools
    for both sell orders (exact input) and buy orders (exact output).
    """

    def __init__(
        self,
        v2_amm: UniswapV2,
        v3_amm: UniswapV3AMM | None,
        weighted_amm: BalancerWeightedAMM | None,
        stable_amm: BalancerStableAMM | None,
        registry: PoolRegistry,
        handler_registry: HandlerRegistry | None = None,
    ) -> None:
        """Initialize the multi-hop router.

        Args:
            v2_amm: UniswapV2 AMM for V2 pools
            v3_amm: UniswapV3 AMM for V3 pools (if available)
            weighted_amm: Balancer weighted AMM (if available)
            stable_amm: Balancer stable AMM (if available)
            registry: Pool registry for path finding
            handler_registry: Optional handler registry for centralized dispatch.
                             If provided, uses registry for simulation. Otherwise
                             falls back to legacy isinstance-based dispatch.
        """
        self.v2_amm = v2_amm
        self.v3_amm = v3_amm
        self.weighted_amm = weighted_amm
        self.stable_amm = stable_amm
        self.registry = registry
        self._handler_registry = handler_registry

    def route_sell_order(
        self,
        order: Order,
        pools: list[AnyPool],
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
        # Compute intermediate amounts for each hop
        hops: list[HopResult] = []
        current_amount = sell_amount
        total_gas = 0

        for i, pool in enumerate(pools):
            token_in = normalize_address(path[i])
            token_out = normalize_address(path[i + 1])

            # Use handler registry if available, otherwise fall back to isinstance
            if self._handler_registry is not None and self._handler_registry.is_registered(pool):
                result = self._handler_registry.simulate_swap(
                    pool, path[i], path[i + 1], current_amount
                )
                if result is None:
                    pool_type = self._handler_registry.get_type_name(pool)
                    return self._error_result(order, f"{pool_type}: swap failed at hop {i}")
                amount_out = result.amount_out
                total_gas += self._handler_registry.get_gas_estimate(pool)
            else:
                # Legacy dispatch
                sim_result = self._simulate_hop_legacy(pool, path[i], path[i + 1], current_amount)
                if sim_result is None:
                    return self._error_result(order, f"Swap failed at hop {i}")
                amount_out, gas = sim_result
                total_gas += gas

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

    def route_buy_order(
        self,
        order: Order,
        pools: list[AnyPool],
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
        # Work backwards to compute required inputs for each hop
        amounts: list[int] = [0] * (len(pools) + 1)
        amounts[-1] = buy_amount  # Final output is the desired buy amount
        total_gas = 0

        for i in range(len(pools) - 1, -1, -1):
            pool = pools[i]
            token_in = path[i]
            token_out = path[i + 1]

            # Use handler registry if available, otherwise fall back to isinstance
            if self._handler_registry is not None and self._handler_registry.is_registered(pool):
                result = self._handler_registry.simulate_swap_exact_output(
                    pool, token_in, token_out, amounts[i + 1]
                )
                if result is None:
                    pool_type = self._handler_registry.get_type_name(pool)
                    return self._error_result(order, f"{pool_type}: exact output failed at hop {i}")
                amounts[i] = result.amount_in
                total_gas += self._handler_registry.get_gas_estimate(pool)
            else:
                # Legacy dispatch
                sim_result = self._simulate_hop_exact_output_legacy(
                    pool, token_in, token_out, amounts[i + 1]
                )
                if sim_result is None:
                    return self._error_result(order, f"Exact output failed at hop {i}")
                amount_in, gas = sim_result
                amounts[i] = amount_in
                total_gas += gas

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

        # Forward pass to compute actual outputs at each hop
        # (due to forward verification, actual outputs may be slightly > requested)
        actual_amounts: list[int] = [amounts[0]]  # Start with required input
        for i, pool in enumerate(pools):
            token_in = path[i]
            token_out = path[i + 1]

            # Simulate forward to get actual output
            if self._handler_registry is not None and self._handler_registry.is_registered(pool):
                result = self._handler_registry.simulate_swap(
                    pool, token_in, token_out, actual_amounts[i]
                )
                if result is None:
                    return self._error_result(order, f"Forward verification failed at hop {i}")
                actual_amounts.append(result.amount_out)
            else:
                sim_result = self._simulate_hop_legacy(pool, token_in, token_out, actual_amounts[i])
                if sim_result is None:
                    return self._error_result(order, f"Forward verification failed at hop {i}")
                actual_amounts.append(sim_result[0])

        # Build hop results with actual amounts from forward pass
        hops: list[HopResult] = []
        for i, pool in enumerate(pools):
            token_in = normalize_address(path[i])
            token_out = normalize_address(path[i + 1])
            hops.append(
                HopResult(
                    pool=pool,
                    input_token=token_in,
                    output_token=token_out,
                    amount_in=actual_amounts[i],
                    amount_out=actual_amounts[i + 1],
                )
            )

        # For buy orders: RoutingResult.amount_out should be the requested buy_amount
        # (used for trade executedAmount and clearing prices)
        # The hops already have the actual forward-simulated outputs (used for interactions)
        return RoutingResult(
            order=order,
            amount_in=required_input,
            amount_out=buy_amount,  # Requested amount for trade/prices
            pool=pools[0],
            pools=pools,
            path=path,
            hops=hops,
            success=True,
            gas_estimate=total_gas,
        )

    def select_best_pools_for_path(
        self,
        path: list[str],
        amount_in: int,
        is_sell: bool,
    ) -> tuple[list[AnyPool], int] | None:
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
                pools = self.registry.get_all_pools_on_path(path)
                return pools, amount_in
            except ValueError:
                return None

        selected_pools: list[AnyPool] = []
        current_amount = amount_in

        for i in range(len(path) - 1):
            token_in = path[i]
            token_out = path[i + 1]
            candidate_pools = self.registry.get_pools_for_pair(token_in, token_out)

            if not candidate_pools:
                return None

            best_pool: AnyPool | None = None
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
        pool: AnyPool,
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
        # Use handler registry if available
        if self._handler_registry is not None and self._handler_registry.is_registered(pool):
            result = self._handler_registry.simulate_swap(pool, token_in, token_out, amount_in)
            return result.amount_out if result else None

        # Legacy dispatch
        sim_result = self._simulate_hop_legacy(pool, token_in, token_out, amount_in)
        return sim_result[0] if sim_result else None

    def _simulate_hop_legacy(
        self,
        pool: AnyPool,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> tuple[int, int] | None:
        """Legacy simulation using isinstance dispatch.

        Returns tuple of (amount_out, gas_estimate) or None if simulation fails.
        """
        from solver.amm.uniswap_v3 import UniswapV3Pool

        if isinstance(pool, UniswapV3Pool):
            if self.v3_amm is None:
                return None
            result = self.v3_amm.simulate_swap(pool, token_in, amount_in)
            return (result.amount_out, pool.gas_estimate) if result else None

        elif isinstance(pool, BalancerWeightedPool):
            if self.weighted_amm is None:
                return None
            result = self.weighted_amm.simulate_swap(pool, token_in, token_out, amount_in)
            # TODO: Revert to pool.gas_estimate once Rust is fixed.
            # See comment in solver/routing/handlers/balancer.py for details.
            return (result.amount_out, BALANCER_WEIGHTED_SWAP_GAS_COST) if result else None

        elif isinstance(pool, BalancerStablePool):
            if self.stable_amm is None:
                return None
            result = self.stable_amm.simulate_swap(pool, token_in, token_out, amount_in)
            # TODO: Revert to pool.gas_estimate once Rust is fixed.
            return (result.amount_out, BALANCER_STABLE_SWAP_GAS_COST) if result else None

        elif isinstance(pool, LimitOrderPool):
            # Limit orders should be handled by handler registry, not legacy path
            # If we get here, limit order AMM is not configured
            return None

        else:
            # V2 pool - type is narrowed by isinstance checks above
            try:
                reserve_in, reserve_out = pool.get_reserves(token_in)
                amount_out = self.v2_amm.get_amount_out(
                    amount_in, reserve_in, reserve_out, pool.fee_multiplier
                )
                return (amount_out, POOL_SWAP_GAS_COST)
            except (ValueError, ZeroDivisionError):
                return None

    def _simulate_hop_exact_output_legacy(
        self,
        pool: AnyPool,
        token_in: str,
        token_out: str,
        amount_out: int,
    ) -> tuple[int, int] | None:
        """Legacy exact output simulation using isinstance dispatch.

        Returns tuple of (amount_in, gas_estimate) or None if simulation fails.
        """
        from solver.amm.uniswap_v3 import UniswapV3Pool

        if isinstance(pool, UniswapV3Pool):
            if self.v3_amm is None:
                return None
            result = self.v3_amm.simulate_swap_exact_output(pool, token_in, amount_out)
            return (result.amount_in, pool.gas_estimate) if result else None

        elif isinstance(pool, BalancerWeightedPool):
            if self.weighted_amm is None:
                return None
            result = self.weighted_amm.simulate_swap_exact_output(
                pool, token_in, token_out, amount_out
            )
            # TODO: Revert to pool.gas_estimate once Rust is fixed.
            return (result.amount_in, BALANCER_WEIGHTED_SWAP_GAS_COST) if result else None

        elif isinstance(pool, BalancerStablePool):
            if self.stable_amm is None:
                return None
            result = self.stable_amm.simulate_swap_exact_output(
                pool, token_in, token_out, amount_out
            )
            # TODO: Revert to pool.gas_estimate once Rust is fixed.
            return (result.amount_in, BALANCER_STABLE_SWAP_GAS_COST) if result else None

        elif isinstance(pool, LimitOrderPool):
            # Limit orders should be handled by handler registry, not legacy path
            # If we get here, limit order AMM is not configured
            return None

        else:
            # V2 pool - type is narrowed by isinstance checks above
            try:
                reserve_in, reserve_out = pool.get_reserves(token_in)
                amount_in = self.v2_amm.get_amount_in(
                    amount_out, reserve_in, reserve_out, pool.fee_multiplier
                )
                return (amount_in, POOL_SWAP_GAS_COST)
            except (ValueError, ZeroDivisionError):
                return None

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


__all__ = ["MultihopRouter"]
