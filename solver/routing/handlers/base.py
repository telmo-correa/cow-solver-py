"""Base class and protocol for pool routing handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from solver.models.types import normalize_address
from solver.routing.types import HopResult, RoutingResult

if TYPE_CHECKING:
    from solver.models.auction import Order
    from solver.pools import AnyPool


class PoolHandler(Protocol):
    """Protocol for pool-specific routing handlers.

    Each handler implementation is responsible for routing orders through
    a specific pool type (V2, V3, Balancer weighted, Balancer stable).

    The handler pattern centralizes pool-specific logic:
    - Swap simulation
    - Limit price checking
    - Partial fill calculation
    - Result building
    """

    def route(
        self,
        order: Order,
        pool: AnyPool,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route an order through this pool type.

        Args:
            order: The order to route
            pool: The pool to route through (type must match handler)
            sell_amount: Order's sell amount
            buy_amount: Order's buy amount (minimum for sell, exact for buy)

        Returns:
            RoutingResult with the routing outcome
        """
        ...


class BaseHandler:
    """Base class with shared handler utilities.

    Provides common methods for building routing results that are used
    across all handler implementations (V2, V3, Balancer).
    """

    def _error_result(self, order: Order, error: str) -> RoutingResult:
        """Create a failed routing result.

        Args:
            order: The order that failed to route
            error: Human-readable error message

        Returns:
            RoutingResult with success=False
        """
        return RoutingResult(
            order=order,
            amount_in=0,
            amount_out=0,
            pool=None,
            success=False,
            error=error,
        )

    def _build_hop(
        self,
        pool: AnyPool,
        order: Order,
        amount_in: int,
        amount_out: int,
    ) -> HopResult:
        """Build a HopResult for a single-hop route.

        Args:
            pool: The pool used for the swap
            order: The order being routed
            amount_in: Input amount for the swap
            amount_out: Output amount from the swap

        Returns:
            HopResult with normalized token addresses
        """
        return HopResult(
            pool=pool,
            input_token=normalize_address(order.sell_token),
            output_token=normalize_address(order.buy_token),
            amount_in=amount_in,
            amount_out=amount_out,
        )

    def _build_success_result(
        self,
        order: Order,
        pool: AnyPool,
        amount_in: int,
        amount_out: int,
        gas_estimate: int,
        *,
        actual_amount_out: int | None = None,
    ) -> RoutingResult:
        """Build a successful routing result.

        Args:
            order: The routed order
            pool: The pool used for the swap
            amount_in: Input amount
            amount_out: Output amount (used for trade executedAmount and clearing prices)
            gas_estimate: Estimated gas cost for the swap
            actual_amount_out: Actual forward-simulated output for interactions.
                              If provided, used for HopResult.amount_out (interaction outputAmount).
                              If None, amount_out is used for both.
                              For buy orders, this allows using requested amount for trade/prices
                              while using actual amount for interaction output.

        Returns:
            RoutingResult with success=True and hop details
        """
        # For interactions, use actual forward-simulated output if provided
        hop_amount_out = actual_amount_out if actual_amount_out is not None else amount_out
        hop = self._build_hop(pool, order, amount_in, hop_amount_out)
        return RoutingResult(
            order=order,
            amount_in=amount_in,
            amount_out=amount_out,  # Used for trade and prices
            pool=pool,
            pools=[pool],
            hops=[hop],
            success=True,
            gas_estimate=gas_estimate,
        )


__all__ = ["PoolHandler", "BaseHandler"]
