"""Base protocol for pool routing handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from solver.models.auction import Order
    from solver.pools import AnyPool
    from solver.routing.types import RoutingResult


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


__all__ = ["PoolHandler"]
