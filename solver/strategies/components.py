"""Component detection algorithms for order grouping.

This module provides algorithms for finding connected components
of orders and token pairs for batch optimization.
"""

from __future__ import annotations

from collections import defaultdict

from solver.models.auction import Order
from solver.models.order_groups import OrderGroup
from solver.models.types import normalize_address
from solver.strategies.graph import UnionFind


def find_token_components(cow_groups: list[OrderGroup]) -> list[list[OrderGroup]]:
    """Find connected components of overlapping token pairs using Union-Find.

    Two pairs are connected if they share a token. For example:
    - (WETH, USDC) and (WETH, DAI) are connected via WETH
    - (USDC, DAI) would also connect to the above component

    Args:
        cow_groups: List of OrderGroups with CoW potential

    Returns:
        List of components, each component is a list of OrderGroups
    """
    if not cow_groups:
        return []

    # Map tokens to group indices
    token_to_groups: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(cow_groups):
        token_a = normalize_address(group.token_a)
        token_b = normalize_address(group.token_b)
        token_to_groups[token_a].append(i)
        token_to_groups[token_b].append(i)

    # Union groups that share tokens
    uf = UnionFind()
    for indices in token_to_groups.values():
        if len(indices) > 1:
            # All groups sharing this token should be in same component
            first = indices[0]
            for other in indices[1:]:
                uf.union(first, other)

    # Group by component root
    components: dict[int, list[OrderGroup]] = defaultdict(list)
    for i, group in enumerate(cow_groups):
        root = uf.find(i)
        components[root].append(group)

    # Sort components by total order count (largest first)
    result = list(components.values())
    result.sort(key=lambda c: sum(g.order_count for g in c), reverse=True)

    return result


def find_order_components(
    orders: list[Order],
    max_tokens: int = 4,
) -> list[tuple[set[str], list[Order]]]:
    """Find connected components of tokens from ALL orders.

    This generalizes beyond bidirectional pairs to capture cycles like A→B→C→A
    where no single pair has orders in both directions.

    Args:
        orders: All orders in the auction
        max_tokens: Maximum tokens per component (default 4). Components with
                    more tokens are skipped to bound complexity.

    Returns:
        List of (token_set, orders) for components with ≤ max_tokens tokens,
        sorted by order count descending.
    """
    if not orders:
        return []

    # Map each token to an index for Union-Find
    all_tokens: set[str] = set()
    for order in orders:
        all_tokens.add(normalize_address(order.sell_token))
        all_tokens.add(normalize_address(order.buy_token))

    token_to_idx = {token: i for i, token in enumerate(sorted(all_tokens))}

    # Build Union-Find on tokens (union sell_token and buy_token for each order)
    uf = UnionFind()
    for order in orders:
        sell_idx = token_to_idx[normalize_address(order.sell_token)]
        buy_idx = token_to_idx[normalize_address(order.buy_token)]
        uf.union(sell_idx, buy_idx)

    # Group tokens by component root
    component_tokens: dict[int, set[str]] = defaultdict(set)
    for token, idx in token_to_idx.items():
        root = uf.find(idx)
        component_tokens[root].add(token)

    # Group orders by component root (via their sell_token)
    component_orders: dict[int, list[Order]] = defaultdict(list)
    for order in orders:
        sell_idx = token_to_idx[normalize_address(order.sell_token)]
        root = uf.find(sell_idx)
        component_orders[root].append(order)

    # Build result list, filtering by max_tokens
    result: list[tuple[set[str], list[Order]]] = []
    for root in component_tokens:
        tokens = component_tokens[root]
        if len(tokens) <= max_tokens:
            result.append((tokens, component_orders[root]))

    # Sort by order count descending
    result.sort(key=lambda x: len(x[1]), reverse=True)

    return result


__all__ = [
    "find_token_components",
    "find_order_components",
]
