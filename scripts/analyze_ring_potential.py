#!/usr/bin/env python3
"""Analyze ring trade potential in historical auctions.

This script determines if ring trades would increase CoW match rate
by finding cycles in the order graph and checking economic viability.

A ring trade is viable if the product of limit prices around the cycle >= 1:
  (sell_A/buy_A) * (sell_B/buy_B) * (sell_C/buy_C) >= 1

Usage:
    python scripts/analyze_ring_potential.py [--limit N]
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.models.auction import AuctionInstance


@dataclass
class RingAnalysis:
    """Analysis results for ring trade potential."""

    auction_id: str
    order_count: int
    unique_tokens: int

    # Graph structure
    edges: int  # Directed edges (token pairs with orders)

    # Cycle detection
    cycles_3: int  # 3-node cycles found
    cycles_4: int  # 4-node cycles found

    # Economic viability
    viable_cycles_3: int  # Cycles where limit prices align
    viable_cycles_4: int

    # Orders in viable cycles (UNIQUE, not overcounted)
    unique_orders_in_viable_cycles: int
    unique_order_uids: set[str] = field(default_factory=set)


@dataclass
class Summary:
    """Summary across all auctions."""

    total_auctions: int = 0
    total_orders: int = 0
    auctions_with_cycles: int = 0
    auctions_with_viable_cycles: int = 0
    total_cycles_3: int = 0
    total_cycles_4: int = 0
    total_viable_3: int = 0
    total_viable_4: int = 0
    total_unique_orders_in_viable: int = 0  # Unique orders across all auctions
    results: list[RingAnalysis] = field(default_factory=list)


def build_order_graph(auction: AuctionInstance) -> dict[str, dict[str, list]]:
    """Build directed graph: token -> token -> list of orders.

    Returns adjacency list where graph[A][B] = list of orders selling A for B.
    """
    graph: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for order in auction.orders:
        sell_token = order.sell_token.lower()
        buy_token = order.buy_token.lower()
        graph[sell_token][buy_token].append(order)

    return graph


def find_cycles_3(graph: dict[str, dict[str, list]]) -> list[tuple[str, str, str]]:
    """Find all 3-node cycles (A→B→C→A)."""
    cycles = []
    tokens = list(graph.keys())

    for a in tokens:
        for b in graph[a]:
            if b == a:
                continue
            for c in graph[b]:
                if c in (a, b):
                    continue
                # Check if C→A exists
                if a in graph[c]:
                    # Normalize cycle to avoid duplicates (smallest token first)
                    cycle = tuple(sorted([a, b, c]))
                    if cycle not in cycles:
                        cycles.append(cycle)

    return cycles


def find_cycles_4(
    graph: dict[str, dict[str, list]], limit: int = 100
) -> list[tuple[str, str, str, str]]:
    """Find 4-node cycles (A→B→C→D→A). Limited to avoid explosion."""
    cycles = []
    tokens = list(graph.keys())

    for a in tokens:
        if len(cycles) >= limit:
            break
        for b in graph[a]:
            if b == a:
                continue
            for c in graph[b]:
                if c in (a, b):
                    continue
                for d in graph[c]:
                    if d in (a, b, c):
                        continue
                    if a in graph[d]:
                        cycle = tuple(sorted([a, b, c, d]))
                        if cycle not in cycles:
                            cycles.append(cycle)
                            if len(cycles) >= limit:
                                break

    return cycles


def is_cycle_viable(
    cycle: tuple[str, ...], graph: dict[str, dict[str, list]]
) -> tuple[bool, set[str]]:
    """Check if a cycle is economically viable.

    A cycle is viable if we can find orders around it where the product
    of exchange rates >= 1.

    Returns (is_viable, set of order UIDs involved).
    """
    n = len(cycle)

    # For each edge in cycle, get best exchange rate
    best_rates = []
    orders_used = []

    for i in range(n):
        from_token = cycle[i]
        to_token = cycle[(i + 1) % n]

        orders = graph[from_token].get(to_token, [])
        if not orders:
            return False, set()

        # Best rate = max(buy_amount / sell_amount) for orders on this edge
        # This is the best price someone is willing to pay
        best_rate = 0
        best_order = None
        for order in orders:
            sell_amt = int(order.sell_amount) if order.sell_amount else 0
            buy_amt = int(order.buy_amount) if order.buy_amount else 0
            rate = buy_amt / sell_amt if sell_amt > 0 else 0
            if rate > best_rate:
                best_rate = rate
                best_order = order

        if best_rate == 0:
            return False, set()

        best_rates.append(best_rate)
        orders_used.append(best_order)

    # Product of rates around cycle
    product = 1.0
    for rate in best_rates:
        product *= rate

    # Viable if product >= 1 (no loss around the cycle)
    # In practice, need product > 1 to cover gas, but >= 1 shows potential
    is_viable = product >= 1.0

    if is_viable:
        return True, {order.uid for order in orders_used if order}
    return False, set()


def analyze_auction(auction: AuctionInstance) -> RingAnalysis:
    """Analyze ring trade potential for a single auction."""
    graph = build_order_graph(auction)

    # Count unique tokens and edges
    unique_tokens = set()
    edge_count = 0
    for from_token, destinations in graph.items():
        unique_tokens.add(from_token)
        for to_token in destinations:
            unique_tokens.add(to_token)
            edge_count += 1

    # Find cycles
    cycles_3 = find_cycles_3(graph)
    cycles_4 = find_cycles_4(graph, limit=50)  # Limit to avoid explosion

    # Check viability - collect UNIQUE order UIDs across all cycles
    viable_3 = 0
    viable_4 = 0
    all_viable_order_uids: set[str] = set()

    for cycle in cycles_3:
        # Need to check the actual directed cycle, not just the sorted tuple
        # Try all rotations to find one that works
        for i in range(3):
            rotated = (cycle[i], cycle[(i + 1) % 3], cycle[(i + 2) % 3])
            viable, order_uids = is_cycle_viable(rotated, graph)
            if viable:
                viable_3 += 1
                all_viable_order_uids.update(order_uids)
                break

    for cycle in cycles_4:
        for i in range(4):
            rotated = (cycle[i], cycle[(i + 1) % 4], cycle[(i + 2) % 4], cycle[(i + 3) % 4])
            viable, order_uids = is_cycle_viable(rotated, graph)
            if viable:
                viable_4 += 1
                all_viable_order_uids.update(order_uids)
                break

    return RingAnalysis(
        auction_id=auction.id,
        order_count=len(list(auction.orders)),
        unique_tokens=len(unique_tokens),
        edges=edge_count,
        cycles_3=len(cycles_3),
        cycles_4=len(cycles_4),
        viable_cycles_3=viable_3,
        viable_cycles_4=viable_4,
        unique_orders_in_viable_cycles=len(all_viable_order_uids),
        unique_order_uids=all_viable_order_uids,
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze ring trade potential")
    parser.add_argument("--limit", type=int, default=10, help="Max auctions to analyze")
    parser.add_argument("--data-dir", type=str, default="data/historical_auctions")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    auction_files = sorted(data_dir.glob("*.json"))[: args.limit]
    if not auction_files:
        print(f"Error: No auction files found in {data_dir}")
        sys.exit(1)

    print(f"Analyzing {len(auction_files)} auctions for ring trade potential...\n")

    summary = Summary()

    for i, auction_file in enumerate(auction_files, 1):
        print(f"[{i}/{len(auction_files)}] {auction_file.name}...", end=" ")

        try:
            with open(auction_file) as f:
                auction_data = json.load(f)
            auction = AuctionInstance.model_validate(auction_data)
        except Exception as e:
            print(f"Error: {e}")
            continue

        result = analyze_auction(auction)
        summary.results.append(result)
        summary.total_auctions += 1
        summary.total_orders += result.order_count
        summary.total_cycles_3 += result.cycles_3
        summary.total_cycles_4 += result.cycles_4
        summary.total_viable_3 += result.viable_cycles_3
        summary.total_viable_4 += result.viable_cycles_4
        summary.total_unique_orders_in_viable += result.unique_orders_in_viable_cycles

        if result.cycles_3 > 0 or result.cycles_4 > 0:
            summary.auctions_with_cycles += 1
        if result.viable_cycles_3 > 0 or result.viable_cycles_4 > 0:
            summary.auctions_with_viable_cycles += 1

        print(
            f"tokens={result.unique_tokens}, 3-cycles={result.cycles_3} (viable={result.viable_cycles_3}), "
            f"4-cycles={result.cycles_4} (viable={result.viable_cycles_4}), "
            f"unique_orders={result.unique_orders_in_viable_cycles}"
        )

    # Print summary
    print("\n" + "=" * 70)
    print("RING TRADE POTENTIAL ANALYSIS")
    print("=" * 70)

    print(f"\nAuctions analyzed: {summary.total_auctions}")
    print(f"Total orders: {summary.total_orders}")

    print("\n--- Cycle Detection ---")
    print(
        f"Auctions with any cycles: {summary.auctions_with_cycles} ({summary.auctions_with_cycles / summary.total_auctions * 100:.1f}%)"
    )
    print(f"Total 3-node cycles: {summary.total_cycles_3}")
    print(f"Total 4-node cycles: {summary.total_cycles_4} (capped at 50/auction)")

    print("\n--- Economic Viability ---")
    print(
        f"Auctions with viable cycles: {summary.auctions_with_viable_cycles} ({summary.auctions_with_viable_cycles / summary.total_auctions * 100:.1f}%)"
    )
    print(
        f"Viable 3-node cycles: {summary.total_viable_3} ({summary.total_viable_3 / max(1, summary.total_cycles_3) * 100:.1f}% of detected)"
    )
    print(
        f"Viable 4-node cycles: {summary.total_viable_4} ({summary.total_viable_4 / max(1, summary.total_cycles_4) * 100:.1f}% of detected)"
    )

    print("\n--- Potential Impact (UNIQUE ORDERS) ---")
    print(f"Unique orders in viable cycles: {summary.total_unique_orders_in_viable}")
    ring_match_rate = (
        summary.total_unique_orders_in_viable / summary.total_orders * 100
        if summary.total_orders > 0
        else 0
    )
    print(f"Ring match rate: {ring_match_rate:.2f}% of orders")

    # Compare to CoW match rate from Slice 4.3
    cow_match_rate = 1.4  # From previous evaluation
    print("\nComparison to CoW matching:")
    print(f"  CoW match rate (Slice 4.3): {cow_match_rate:.2f}%")
    print(f"  Ring match rate (potential): {ring_match_rate:.2f}%")

    if ring_match_rate > cow_match_rate * 2:
        print(
            f"\n✅ Ring trades could SIGNIFICANTLY increase match rate ({ring_match_rate / cow_match_rate:.1f}x)"
        )
        print("   Recommendation: Consider implementing ring trades")
    elif ring_match_rate > cow_match_rate:
        print(
            f"\n⚠️  Ring trades offer MODEST improvement ({ring_match_rate / cow_match_rate:.1f}x)"
        )
        print("   Recommendation: Weigh implementation cost vs benefit")
    else:
        print("\n❌ Ring trades offer NO significant improvement")
        print("   Recommendation: Skip ring trades, focus on split routing")


if __name__ == "__main__":
    main()
