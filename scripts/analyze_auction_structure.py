#!/usr/bin/env python3
"""Analyze auction structure for Phase 4 optimization planning.

This script analyzes historical auctions to understand:
1. Order count distribution
2. Token pair patterns
3. CoW matching potential (single pair and multi-pair)
4. Ring trade opportunities
5. Order size vs liquidity ratios

Run with: python scripts/analyze_auction_structure.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from solver.models.auction import AuctionInstance
from solver.models.order_groups import OrderGroup, find_cow_opportunities, group_orders_by_pair
from solver.models.types import normalize_address


@dataclass
class AuctionAnalysis:
    """Analysis results for a single auction."""

    auction_id: str
    order_count: int
    unique_tokens: int
    token_pairs: int
    cow_pairs: int  # Pairs with orders in both directions
    total_cow_orders: int  # Orders that could participate in CoW
    has_ring_potential: bool  # A→B, B→C, C→A all exist
    ring_tokens: list[str] = field(default_factory=list)
    liquidity_types: set[str] = field(default_factory=set)
    max_order_to_liquidity_ratio: float = 0.0


@dataclass
class AggregateStats:
    """Aggregate statistics across all auctions."""

    total_auctions: int = 0
    total_orders: int = 0

    # Order count distribution
    order_count_distribution: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # CoW potential
    auctions_with_cow_potential: int = 0
    auctions_with_multi_pair_cow: int = 0
    total_cow_pairs: int = 0
    total_cow_orders: int = 0  # Total orders that could participate in CoW

    # CoW pair size distribution (how many orders per CoW pair)
    cow_pair_size_2: int = 0  # Pairs with exactly 2 orders
    cow_pair_size_3_9: int = 0  # Pairs with 3-9 orders
    cow_pair_size_10_plus: int = 0  # Pairs with 10+ orders
    orders_in_large_pairs: int = 0  # Orders in 10+ order pairs

    # Ring trades
    auctions_with_ring_potential: int = 0

    # Liquidity
    liquidity_type_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Size ratios
    high_impact_orders: int = 0  # Orders > 10% of pool liquidity


def load_auction(path: Path) -> AuctionInstance | None:
    """Load an auction from JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        # Skip expected output files
        if "expected" in path.name:
            return None
        return AuctionInstance.model_validate(data)
    except Exception as e:
        print(f"  Warning: Could not load {path.name}: {e}")
        return None


def find_ring_potential(groups: dict[tuple[str, str], OrderGroup]) -> tuple[bool, list[str]]:
    """Check if there's ring trade potential (A→B→C→A cycle).

    For a ring trade, we need:
    - Orders selling A for B (or B for A)
    - Orders selling B for C (or C for B)
    - Orders selling C for A (or A for C)

    This forms a cycle that could be exploited.
    """
    # Build adjacency from order groups (not just CoW pairs)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for (token_a, token_b), group in groups.items():
        if group.sellers_of_a:  # Orders selling A for B
            adjacency[token_a].add(token_b)
        if group.sellers_of_b:  # Orders selling B for A
            adjacency[token_b].add(token_a)

    # Look for 3-cycles using DFS
    tokens = list(adjacency.keys())
    for start in tokens:
        # Try to find path start → X → Y → start
        for mid1 in adjacency.get(start, set()):
            if mid1 == start:
                continue
            for mid2 in adjacency.get(mid1, set()):
                if mid2 in (start, mid1):
                    continue
                if start in adjacency.get(mid2, set()):
                    # Found cycle: start → mid1 → mid2 → start
                    return True, [start, mid1, mid2]

    return False, []


def calculate_order_liquidity_ratio(
    auction: AuctionInstance,
) -> float:
    """Calculate max ratio of order size to available liquidity."""
    max_ratio = 0.0

    # Build liquidity map: (token_in, token_out) -> total_liquidity
    liquidity_map: dict[tuple[str, str], int] = defaultdict(int)

    for liq in auction.liquidity:
        if liq.kind == "constantProduct" and liq.tokens:
            tokens = list(liq.tokens.keys())
            if len(tokens) >= 2:
                t0, t1 = normalize_address(tokens[0]), normalize_address(tokens[1])
                bal0 = int(liq.tokens[tokens[0]].get("balance", 0))
                bal1 = int(liq.tokens[tokens[1]].get("balance", 0))
                liquidity_map[(t0, t1)] += bal0
                liquidity_map[(t1, t0)] += bal1

    # Check each order against available liquidity
    for order in auction.orders:
        sell_token = normalize_address(order.sell_token)
        buy_token = normalize_address(order.buy_token)
        sell_amount = order.sell_amount_int

        available = liquidity_map.get((sell_token, buy_token), 0)
        if available > 0:
            ratio = sell_amount / available
            max_ratio = max(max_ratio, ratio)

    return max_ratio


def analyze_auction(auction: AuctionInstance) -> AuctionAnalysis:
    """Analyze a single auction."""
    # Group orders by pair
    groups = group_orders_by_pair(auction.orders)

    # Find CoW opportunities
    cow_opportunities = find_cow_opportunities(auction.orders)

    # Count orders in CoW pairs
    total_cow_orders = sum(g.order_count for g in cow_opportunities)

    # Check for ring potential
    has_ring, ring_tokens = find_ring_potential(groups)

    # Collect unique tokens
    tokens: set[str] = set()
    for order in auction.orders:
        tokens.add(normalize_address(order.sell_token))
        tokens.add(normalize_address(order.buy_token))

    # Collect liquidity types
    liq_types: set[str] = set()
    for liq in auction.liquidity:
        liq_types.add(liq.kind)

    # Calculate order/liquidity ratio
    max_ratio = calculate_order_liquidity_ratio(auction)

    return AuctionAnalysis(
        auction_id=auction.id or "unknown",
        order_count=len(auction.orders),
        unique_tokens=len(tokens),
        token_pairs=len(groups),
        cow_pairs=len(cow_opportunities),
        total_cow_orders=total_cow_orders,
        has_ring_potential=has_ring,
        ring_tokens=ring_tokens,
        liquidity_types=liq_types,
        max_order_to_liquidity_ratio=max_ratio,
    )


def aggregate_stats(
    analyses: list[AuctionAnalysis],
    auctions: list[AuctionInstance] | None = None,
) -> AggregateStats:
    """Compute aggregate statistics.

    Args:
        analyses: List of analysis results
        auctions: Optional list of original auctions for detailed CoW analysis
    """
    stats = AggregateStats()

    for i, a in enumerate(analyses):
        stats.total_auctions += 1
        stats.total_orders += a.order_count
        stats.order_count_distribution[a.order_count] += 1

        if a.cow_pairs > 0:
            stats.auctions_with_cow_potential += 1
            stats.total_cow_pairs += a.cow_pairs
            stats.total_cow_orders += a.total_cow_orders

        if a.cow_pairs > 1:
            stats.auctions_with_multi_pair_cow += 1

        if a.has_ring_potential:
            stats.auctions_with_ring_potential += 1

        for liq_type in a.liquidity_types:
            stats.liquidity_type_counts[liq_type] += 1

        if a.max_order_to_liquidity_ratio > 0.1:
            stats.high_impact_orders += 1

        # Detailed CoW pair analysis if we have the auction
        if auctions and i < len(auctions):
            cow_groups = find_cow_opportunities(auctions[i].orders)
            for g in cow_groups:
                if g.order_count == 2:
                    stats.cow_pair_size_2 += 1
                elif g.order_count <= 9:
                    stats.cow_pair_size_3_9 += 1
                else:
                    stats.cow_pair_size_10_plus += 1
                    stats.orders_in_large_pairs += g.order_count

    return stats


def print_analysis(analyses: list[AuctionAnalysis], stats: AggregateStats) -> None:
    """Print analysis results."""
    print("\n" + "=" * 70)
    print("AUCTION STRUCTURE ANALYSIS - Phase 4 Planning")
    print("=" * 70)

    print("\n## Summary")
    print(f"Total auctions analyzed: {stats.total_auctions}")
    print(f"Total orders: {stats.total_orders}")
    print(f"Average orders per auction: {stats.total_orders / max(1, stats.total_auctions):.1f}")

    print("\n## Order Count Distribution")
    for count in sorted(stats.order_count_distribution.keys()):
        freq = stats.order_count_distribution[count]
        pct = 100 * freq / max(1, stats.total_auctions)
        bar = "#" * int(pct / 2)
        print(f"  {count:3d} orders: {freq:3d} auctions ({pct:5.1f}%) {bar}")

    print("\n## CoW Matching Potential")
    cow_pct = 100 * stats.auctions_with_cow_potential / max(1, stats.total_auctions)
    multi_pct = 100 * stats.auctions_with_multi_pair_cow / max(1, stats.total_auctions)
    cow_order_pct = 100 * stats.total_cow_orders / max(1, stats.total_orders)
    print(f"  Auctions with CoW potential: {stats.auctions_with_cow_potential} ({cow_pct:.1f}%)")
    print(
        f"  Auctions with multi-pair CoW: {stats.auctions_with_multi_pair_cow} ({multi_pct:.1f}%)"
    )
    print(f"  Total CoW pairs found: {stats.total_cow_pairs}")
    print(f"  Total CoW orders: {stats.total_cow_orders} ({cow_order_pct:.1f}% of all orders)")

    # CoW pair size distribution
    if stats.cow_pair_size_2 > 0 or stats.cow_pair_size_3_9 > 0:
        print("\n## CoW Pair Size Distribution")
        total_pairs = stats.cow_pair_size_2 + stats.cow_pair_size_3_9 + stats.cow_pair_size_10_plus
        if total_pairs > 0:
            print(
                f"  Pairs with 2 orders (simple): {stats.cow_pair_size_2} ({100 * stats.cow_pair_size_2 / total_pairs:.1f}%)"
            )
            print(
                f"  Pairs with 3-9 orders: {stats.cow_pair_size_3_9} ({100 * stats.cow_pair_size_3_9 / total_pairs:.1f}%)"
            )
            print(
                f"  Pairs with 10+ orders (double auction): {stats.cow_pair_size_10_plus} ({100 * stats.cow_pair_size_10_plus / total_pairs:.1f}%)"
            )
            print(f"  Orders in 10+ pairs: {stats.orders_in_large_pairs}")

    print("\n## Ring Trade Potential")
    ring_pct = 100 * stats.auctions_with_ring_potential / max(1, stats.total_auctions)
    print(f"  Auctions with ring potential: {stats.auctions_with_ring_potential} ({ring_pct:.1f}%)")

    # Show which auctions have ring potential
    ring_auctions = [a for a in analyses if a.has_ring_potential]
    if ring_auctions:
        print("  Ring trade auctions:")
        for a in ring_auctions[:5]:  # Show first 5
            print(f"    - {a.auction_id}: {' → '.join(a.ring_tokens[:3])} → ...")

    print("\n## Liquidity Types")
    for liq_type, count in sorted(stats.liquidity_type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(1, stats.total_auctions)
        print(f"  {liq_type}: {count} auctions ({pct:.1f}%)")

    print("\n## Order Size Impact")
    impact_pct = 100 * stats.high_impact_orders / max(1, stats.total_auctions)
    print(
        f"  Auctions with high-impact orders (>10% of liquidity): {stats.high_impact_orders} ({impact_pct:.1f}%)"
    )

    # Detailed per-auction analysis for CoW auctions
    cow_auctions = [a for a in analyses if a.cow_pairs > 0]
    if cow_auctions:
        print("\n## Detailed CoW Auction Analysis")
        print(f"{'Auction':<40} {'Orders':>7} {'Pairs':>6} {'CoW':>5} {'CoW Orders':>10}")
        print("-" * 70)
        for a in sorted(cow_auctions, key=lambda x: -x.cow_pairs)[:15]:
            print(
                f"{a.auction_id:<40} {a.order_count:>7} {a.token_pairs:>6} {a.cow_pairs:>5} {a.total_cow_orders:>10}"
            )

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR PHASE 4")
    print("=" * 70)

    print("\n## Priority Assessment")

    # Multi-order CoW
    if cow_pct > 20:
        print(f"  [HIGH] Multi-order CoW matching: {cow_pct:.0f}% of auctions have CoW potential")
    else:
        print(
            f"  [LOW] Multi-order CoW matching: Only {cow_pct:.0f}% of auctions have CoW potential"
        )

    # Ring trades
    if ring_pct > 5:
        print(f"  [MEDIUM] Ring trades: {ring_pct:.0f}% of auctions have ring potential")
    else:
        print(f"  [LOW] Ring trades: Only {ring_pct:.0f}% of auctions have ring potential")

    # Split routing
    if impact_pct > 10:
        print(f"  [MEDIUM] Split routing: {impact_pct:.0f}% of auctions have high-impact orders")
    else:
        print(f"  [LOW] Split routing: Only {impact_pct:.0f}% have high-impact orders")

    print("\n")


def main() -> None:
    """Run the analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze auction structure")
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Analyze historical auctions from data/historical_auctions/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of auctions to analyze",
    )
    args = parser.parse_args()

    # Find auction files
    if args.historical:
        # Real historical auctions
        auctions_dir = Path("data/historical_auctions")
        if not auctions_dir.exists():
            print("No historical auctions found. Run download script first.")
            return
        auction_files = list(auctions_dir.glob("*.json"))[: args.limit]
    else:
        # Test fixtures
        fixtures_dir = Path("tests/fixtures/auctions")
        auction_files = []
        for subdir in fixtures_dir.iterdir():
            if subdir.is_dir():
                auction_files.extend(subdir.glob("*.json"))

    print(f"Found {len(auction_files)} auction files")

    # Load and analyze each auction
    analyses: list[AuctionAnalysis] = []
    auctions: list[AuctionInstance] = []
    for path in sorted(auction_files):
        auction = load_auction(path)
        if auction is None:
            continue
        if not auction.orders:
            continue

        analysis = analyze_auction(auction)
        analyses.append(analysis)
        auctions.append(auction)
        print(
            f"  Analyzed: {analysis.auction_id} ({analysis.order_count} orders, {analysis.cow_pairs} CoW pairs)"
        )

    if not analyses:
        print("No auctions to analyze!")
        return

    # Compute aggregate stats (pass auctions for detailed CoW analysis)
    stats = aggregate_stats(analyses, auctions)

    # Print results
    print_analysis(analyses, stats)


if __name__ == "__main__":
    main()
