"""Centralized registry for pool handlers and simulators.

This module provides a HandlerRegistry class that eliminates isinstance chains
by mapping pool types to their handlers and simulators. This makes it easy to
add new pool types without modifying routing code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from solver.pools import AnyPool
from solver.routing.handlers.base import PoolHandler

if TYPE_CHECKING:
    from solver.amm.base import SwapResult


# Type alias for simulator functions
SwapSimulator = Callable[[AnyPool, str, str, int], "SwapResult | None"]
ExactOutputSimulator = Callable[[AnyPool, str, str, int], "SwapResult | None"]


class HandlerRegistry:
    """Registry for pool-specific routing handlers and simulators.

    This registry centralizes the mapping from pool types to their handlers,
    eliminating isinstance chains throughout the routing code.

    Usage:
        registry = HandlerRegistry()
        registry.register(
            UniswapV2Pool,
            handler=v2_handler,
            simulator=lambda p, ti, to, ai: amm.simulate_swap(p, ti, ai),
            exact_output_simulator=lambda p, ti, to, ao: amm.simulate_swap_exact_output(p, ti, ao),
            type_name="v2",
        )

        # Later in routing code:
        handler = registry.get_handler(pool)
        result = registry.simulate_swap(pool, token_in, token_out, amount_in)
    """

    def __init__(self) -> None:
        """Initialize an empty handler registry."""
        self._handlers: dict[type, PoolHandler] = {}
        self._simulators: dict[type, SwapSimulator] = {}
        self._exact_output_simulators: dict[type, ExactOutputSimulator] = {}
        self._type_names: dict[type, str] = {}
        self._gas_estimates: dict[type, Callable[[AnyPool], int]] = {}

    def register(
        self,
        pool_type: type,
        handler: PoolHandler,
        simulator: SwapSimulator,
        exact_output_simulator: ExactOutputSimulator | None = None,
        type_name: str = "unknown",
        gas_estimate: Callable[[AnyPool], int] | None = None,
    ) -> None:
        """Register a handler for a pool type.

        Args:
            pool_type: The pool class to register (e.g., UniswapV2Pool)
            handler: The PoolHandler implementation for this pool type
            simulator: Function to simulate a swap (exact input)
            exact_output_simulator: Function to simulate a swap (exact output)
            type_name: Human-readable name for logging (e.g., "v2", "v3")
            gas_estimate: Function to get gas estimate for a pool
        """
        self._handlers[pool_type] = handler
        self._simulators[pool_type] = simulator
        if exact_output_simulator is not None:
            self._exact_output_simulators[pool_type] = exact_output_simulator
        self._type_names[pool_type] = type_name
        if gas_estimate is not None:
            self._gas_estimates[pool_type] = gas_estimate

    def get_handler(self, pool: AnyPool) -> PoolHandler | None:
        """Get handler for a pool by its type.

        Args:
            pool: The pool to get a handler for

        Returns:
            PoolHandler if registered, None otherwise
        """
        return self._handlers.get(type(pool))

    def simulate_swap(
        self,
        pool: AnyPool,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> SwapResult | None:
        """Simulate a swap using the registered simulator.

        Args:
            pool: The pool to simulate through
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount

        Returns:
            SwapResult if successful, None if no simulator registered or simulation fails
        """
        simulator = self._simulators.get(type(pool))
        if simulator is None:
            return None
        return simulator(pool, token_in, token_out, amount_in)

    def simulate_swap_exact_output(
        self,
        pool: AnyPool,
        token_in: str,
        token_out: str,
        amount_out: int,
    ) -> SwapResult | None:
        """Simulate a swap for exact output using the registered simulator.

        Args:
            pool: The pool to simulate through
            token_in: Input token address
            token_out: Output token address
            amount_out: Desired output amount

        Returns:
            SwapResult if successful, None if no simulator registered or simulation fails
        """
        simulator = self._exact_output_simulators.get(type(pool))
        if simulator is None:
            return None
        return simulator(pool, token_in, token_out, amount_out)

    def get_type_name(self, pool: AnyPool) -> str:
        """Get human-readable pool type name.

        Args:
            pool: The pool to get the type name for

        Returns:
            Type name if registered, "unknown" otherwise
        """
        return self._type_names.get(type(pool), "unknown")

    def get_gas_estimate(self, pool: AnyPool) -> int:
        """Get gas estimate for a pool.

        Args:
            pool: The pool to get gas estimate for

        Returns:
            Gas estimate, or 0 if no estimate function registered
        """
        estimate_fn = self._gas_estimates.get(type(pool))
        if estimate_fn is None:
            return 0
        return estimate_fn(pool)

    def is_registered(self, pool: AnyPool) -> bool:
        """Check if a handler is registered for this pool type.

        Args:
            pool: The pool to check

        Returns:
            True if a handler is registered, False otherwise
        """
        return type(pool) in self._handlers


__all__ = ["HandlerRegistry", "SwapSimulator", "ExactOutputSimulator"]
