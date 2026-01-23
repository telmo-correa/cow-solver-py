"""Type definitions for routing module."""

from __future__ import annotations

from dataclasses import dataclass

from solver.constants import POOL_SWAP_GAS_COST
from solver.models.auction import Order
from solver.pools import AnyPool


@dataclass
class HopResult:
    """Result of a single hop in a multi-hop route."""

    pool: AnyPool
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
    pool: AnyPool | None  # None when no pool found
    success: bool
    error: str | None = None
    # Multi-hop routing fields
    path: list[str] | None = None  # Token path for multi-hop swaps
    pools: list[AnyPool] | None = None  # Pools along the path
    hops: list[HopResult] | None = None  # Detailed results for each hop
    gas_estimate: int = POOL_SWAP_GAS_COST  # Default single-hop gas (60k per swap)

    @property
    def is_multihop(self) -> bool:
        """Check if this is a multi-hop route."""
        return self.path is not None and len(self.path) > 2


__all__ = ["HopResult", "RoutingResult"]
