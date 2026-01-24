"""Order grouping utilities for batch optimization.

This module provides utilities for grouping orders by token pair, which is
essential for Phase 4's unified optimization:
- N-order CoW detection
- Batch optimization per token pair
- Aggregate demand/supply analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from solver.models.auction import Order
from solver.models.types import normalize_address


@dataclass
class OrderGroup:
    """Orders sharing the same token pair (in either direction).

    For tokens A and B:
    - sellers_of_a: Orders selling A to get B
    - sellers_of_b: Orders selling B to get A

    This represents all orders that could potentially CoW match
    or share liquidity pools.

    Attributes:
        token_a: First token (canonical ordering: token_a < token_b)
        token_b: Second token
        sellers_of_a: Orders selling token_a to buy token_b
        sellers_of_b: Orders selling token_b to buy token_a

    Note:
        Aggregate properties (total_*, all_orders) are cached for performance.
        Do not mutate sellers_of_a/sellers_of_b after accessing cached properties.
    """

    token_a: str  # Canonical ordering: token_a < token_b
    token_b: str
    sellers_of_a: list[Order] = field(default_factory=list)  # A -> B
    sellers_of_b: list[Order] = field(default_factory=list)  # B -> A

    @property
    def has_cow_potential(self) -> bool:
        """True if orders exist in both directions (CoW matching possible)."""
        return bool(self.sellers_of_a) and bool(self.sellers_of_b)

    @cached_property
    def total_sell_a(self) -> int:
        """Total amount of token_a being sold across all orders."""
        return sum(o.sell_amount_int for o in self.sellers_of_a)

    @cached_property
    def total_buy_a(self) -> int:
        """Total amount of token_a being bought (by sellers of B)."""
        return sum(o.buy_amount_int for o in self.sellers_of_b)

    @cached_property
    def total_sell_b(self) -> int:
        """Total amount of token_b being sold across all orders."""
        return sum(o.sell_amount_int for o in self.sellers_of_b)

    @cached_property
    def total_buy_b(self) -> int:
        """Total amount of token_b being bought (by sellers of A)."""
        return sum(o.buy_amount_int for o in self.sellers_of_a)

    @property
    def order_count(self) -> int:
        """Total number of orders in this group."""
        return len(self.sellers_of_a) + len(self.sellers_of_b)

    @cached_property
    def all_orders(self) -> tuple[Order, ...]:
        """All orders in this group (immutable view)."""
        return tuple(self.sellers_of_a) + tuple(self.sellers_of_b)


def group_orders_by_pair(orders: list[Order]) -> dict[tuple[str, str], OrderGroup]:
    """Group orders by canonical token pair.

    Returns dict mapping (token_a, token_b) to OrderGroup.
    Token pair is canonical: token_a < token_b lexicographically.

    Args:
        orders: List of orders to group

    Returns:
        Dict of (token_a, token_b) -> OrderGroup

    Example:
        >>> groups = group_orders_by_pair(auction.orders)
        >>> for pair, group in groups.items():
        ...     if group.has_cow_potential:
        ...         print(f"{pair}: {group.order_count} orders with CoW potential")
    """
    # Build intermediate lists first to avoid mutating OrderGroup after construction
    sellers_a: dict[tuple[str, str], list[Order]] = {}
    sellers_b: dict[tuple[str, str], list[Order]] = {}

    for order in orders:
        sell = normalize_address(order.sell_token)
        buy = normalize_address(order.buy_token)

        # Canonical pair ordering (ensures consistent grouping)
        if sell < buy:
            pair = (sell, buy)
            sellers_a.setdefault(pair, []).append(order)
        else:
            pair = (buy, sell)
            sellers_b.setdefault(pair, []).append(order)

    # Construct OrderGroups with complete lists (safe for cached_property)
    all_pairs = set(sellers_a.keys()) | set(sellers_b.keys())
    return {
        pair: OrderGroup(
            token_a=pair[0],
            token_b=pair[1],
            sellers_of_a=sellers_a.get(pair, []),
            sellers_of_b=sellers_b.get(pair, []),
        )
        for pair in all_pairs
    }


def find_cow_opportunities(orders: list[Order]) -> list[OrderGroup]:
    """Find all order groups with CoW matching potential.

    Returns groups where orders exist in both directions,
    sorted by total order count (largest groups first).

    Args:
        orders: List of orders to analyze

    Returns:
        List of OrderGroups with CoW potential, sorted by order count descending

    Example:
        >>> opportunities = find_cow_opportunities(auction.orders)
        >>> for group in opportunities:
        ...     print(f"{group.token_a[-6:]}↔{group.token_b[-6:]}: "
        ...           f"{len(group.sellers_of_a)} vs {len(group.sellers_of_b)}")
    """
    groups = group_orders_by_pair(orders)
    cow_groups = [g for g in groups.values() if g.has_cow_potential]
    return sorted(cow_groups, key=lambda g: g.order_count, reverse=True)


__all__ = ["OrderGroup", "group_orders_by_pair", "find_cow_opportunities"]
