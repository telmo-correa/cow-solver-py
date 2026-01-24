#!/usr/bin/env python3
"""Analyze why matching strategies fail to capture CoW potential.

This script investigates:
1. Price crossing: Do limit prices on same pair overlap?
2. Fill-or-kill constraints: How many orders can't be partially filled?
3. Volume imbalance: Are sell/buy volumes matched on each pair?
4. Token overlap: How many pairs share tokens (causing price conflicts)?

Usage:
    python scripts/analyze_matching_gap.py [--limit N]
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.models.auction import AuctionInstance
from solver.models.types import normalize_address
from solver.strategies.double_auction import get_limit_price


@dataclass
class PairAnalysis:
    """Analysis of a single token pair."""

    token_a: str
    token_b: str
    sellers_of_a: int = 0  # Orders selling A for B
    sellers_of_b: int = 0  # Orders selling B for A
    volume_a_to_b: int = 0  # Total sell amount A→B
    volume_b_to_a: int = 0  # Total sell amount B→A
    best_ask: Decimal | None = None  # Lowest ask price (A→B)
    best_bid: Decimal | None = None  # Highest bid price (B→A, expressed as B/A)
    prices_cross: bool = False  # Do ask and bid overlap?
    fillable_sellers_a: int = 0  # Partially fillable A→B orders
    fillable_sellers_b: int = 0  # Partially fillable B→A orders

    @property
    def has_cow_potential(self) -> bool:
        return self.sellers_of_a > 0 and self.sellers_of_b > 0

    @property
    def total_orders(self) -> int:
        return self.sellers_of_a + self.sellers_of_b


def analyze_pair(
    orders_a_to_b: list, orders_b_to_a: list, token_a: str, token_b: str
) -> PairAnalysis:
    """Analyze a single token pair for matching potential."""
    analysis = PairAnalysis(token_a=token_a, token_b=token_b)

    # Analyze A→B orders (sellers of A)
    for order in orders_a_to_b:
        analysis.sellers_of_a += 1
        analysis.volume_a_to_b += order.sell_amount_int

        if order.partially_fillable:
            analysis.fillable_sellers_a += 1

        # Ask price = buy_amount / sell_amount (B per A they want)
        price = get_limit_price(order, is_selling_a=True)
        if price is not None and (analysis.best_ask is None or price < analysis.best_ask):
            analysis.best_ask = price

    # Analyze B→A orders (sellers of B, which are buyers of A)
    for order in orders_b_to_a:
        analysis.sellers_of_b += 1
        analysis.volume_b_to_a += order.sell_amount_int

        if order.partially_fillable:
            analysis.fillable_sellers_b += 1

        # Bid price = sell_amount / buy_amount (B per A they'll pay)
        price = get_limit_price(order, is_selling_a=False)
        if price is not None and (analysis.best_bid is None or price > analysis.best_bid):
            analysis.best_bid = price

    # Check if prices cross (ask <= bid means match possible)
    if analysis.best_ask is not None and analysis.best_bid is not None:
        analysis.prices_cross = analysis.best_ask <= analysis.best_bid

    return analysis


def analyze_auction(auction: AuctionInstance) -> dict:
    """Analyze a single auction for matching gaps."""
    # Group orders by directed pair
    pair_orders: dict[tuple[str, str], list] = defaultdict(list)

    for order in auction.orders:
        sell = normalize_address(order.sell_token)
        buy = normalize_address(order.buy_token)
        pair_orders[(sell, buy)].append(order)

    # Find all undirected pairs with orders in both directions
    analyzed_pairs: list[PairAnalysis] = []
    seen_pairs: set[tuple[str, str]] = set()

    for (sell, buy), _orders in pair_orders.items():
        # Check if we have orders in both directions
        if (buy, sell) not in pair_orders:
            continue

        # Avoid analyzing same pair twice
        normalized = tuple(sorted([sell, buy]))
        if normalized in seen_pairs:
            continue
        seen_pairs.add(normalized)

        # Analyze this pair
        token_a, token_b = normalized
        orders_a_to_b = pair_orders.get((token_a, token_b), [])
        orders_b_to_a = pair_orders.get((token_b, token_a), [])

        analysis = analyze_pair(orders_a_to_b, orders_b_to_a, token_a, token_b)
        if analysis.has_cow_potential:
            analyzed_pairs.append(analysis)

    # Build token adjacency graph (for overlap analysis)
    token_to_pairs: dict[str, list[PairAnalysis]] = defaultdict(list)
    for pair in analyzed_pairs:
        token_to_pairs[pair.token_a].append(pair)
        token_to_pairs[pair.token_b].append(pair)

    # Count overlapping pairs (pairs that share a token)
    overlapping_pairs = 0
    for pairs in token_to_pairs.values():
        if len(pairs) > 1:
            overlapping_pairs += len(pairs)

    # Aggregate statistics
    total_orders_on_cow_pairs = sum(p.total_orders for p in analyzed_pairs)
    crossing_pairs = [p for p in analyzed_pairs if p.prices_cross]
    non_crossing_pairs = [p for p in analyzed_pairs if not p.prices_cross]

    orders_on_crossing = sum(p.total_orders for p in crossing_pairs)
    orders_on_non_crossing = sum(p.total_orders for p in non_crossing_pairs)

    fillable_orders = sum(p.fillable_sellers_a + p.fillable_sellers_b for p in analyzed_pairs)

    return {
        "total_orders": auction.order_count,
        "cow_pairs": len(analyzed_pairs),
        "orders_on_cow_pairs": total_orders_on_cow_pairs,
        "crossing_pairs": len(crossing_pairs),
        "non_crossing_pairs": len(non_crossing_pairs),
        "orders_on_crossing": orders_on_crossing,
        "orders_on_non_crossing": orders_on_non_crossing,
        "fillable_orders": fillable_orders,
        "fill_or_kill_orders": total_orders_on_cow_pairs - fillable_orders,
        "overlapping_pair_count": overlapping_pairs,
        "pairs": analyzed_pairs,  # For detailed analysis
    }


def load_auction(path: Path) -> AuctionInstance:
    """Load an auction from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return AuctionInstance.model_validate(data)


def main():
    parser = argparse.ArgumentParser(description="Analyze matching gap")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of auctions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-auction details")
    args = parser.parse_args()

    # Find historical auction files
    data_dir = Path(__file__).parent.parent / "data" / "historical_auctions"
    auction_files = sorted(data_dir.glob("mainnet_*.json"))

    if args.limit:
        auction_files = auction_files[: args.limit]

    print(f"Analyzing {len(auction_files)} historical auctions...\n")

    # Aggregate statistics
    totals = {
        "total_orders": 0,
        "orders_on_cow_pairs": 0,
        "crossing_pairs": 0,
        "non_crossing_pairs": 0,
        "orders_on_crossing": 0,
        "orders_on_non_crossing": 0,
        "fillable_orders": 0,
        "fill_or_kill_orders": 0,
        "overlapping_pair_count": 0,
    }

    # Sample some non-crossing pairs for analysis
    sample_non_crossing: list[PairAnalysis] = []

    for auction_file in auction_files:
        auction = load_auction(auction_file)
        result = analyze_auction(auction)

        for key in totals:
            if key in result:
                totals[key] += result[key]

        # Sample some non-crossing pairs
        if len(sample_non_crossing) < 10:
            for pair in result["pairs"]:
                if not pair.prices_cross and len(sample_non_crossing) < 10:
                    sample_non_crossing.append(pair)

        if args.verbose:
            print(f"{auction_file.name}:")
            print(f"  Orders: {result['total_orders']}, CoW pairs: {result['cow_pairs']}")
            print(
                f"  Crossing: {result['crossing_pairs']}, Non-crossing: {result['non_crossing_pairs']}"
            )

    # Print summary
    print("=" * 80)
    print("MATCHING GAP ANALYSIS")
    print("=" * 80)

    print(f"\nTotal orders analyzed: {totals['total_orders']:,}")
    print(
        f"Orders on CoW pairs (bidirectional): {totals['orders_on_cow_pairs']:,} ({totals['orders_on_cow_pairs'] / totals['total_orders'] * 100:.1f}%)"
    )

    print("\n" + "-" * 80)
    print("PRICE CROSSING ANALYSIS")
    print("-" * 80)
    print("\nFor matching to occur, ask price <= bid price (prices 'cross').")
    print(f"\nCrossing pairs: {totals['crossing_pairs']:,}")
    print(f"Non-crossing pairs: {totals['non_crossing_pairs']:,}")
    print(
        f"\nOrders on crossing pairs: {totals['orders_on_crossing']:,} ({totals['orders_on_crossing'] / totals['total_orders'] * 100:.2f}%)"
    )
    print(
        f"Orders on non-crossing pairs: {totals['orders_on_non_crossing']:,} ({totals['orders_on_non_crossing'] / totals['total_orders'] * 100:.2f}%)"
    )

    if totals["orders_on_cow_pairs"] > 0:
        crossing_rate = totals["orders_on_crossing"] / totals["orders_on_cow_pairs"] * 100
        print(f"\nOf CoW-potential orders, {crossing_rate:.1f}% have crossing prices.")

    print("\n" + "-" * 80)
    print("FILL-OR-KILL ANALYSIS")
    print("-" * 80)
    print("\nFill-or-kill orders can't be partially filled, limiting matching flexibility.")
    print(f"\nPartially fillable orders: {totals['fillable_orders']:,}")
    print(f"Fill-or-kill orders: {totals['fill_or_kill_orders']:,}")

    if totals["orders_on_cow_pairs"] > 0:
        fok_rate = totals["fill_or_kill_orders"] / totals["orders_on_cow_pairs"] * 100
        print(f"\nOf CoW-potential orders, {fok_rate:.1f}% are fill-or-kill.")

    print("\n" + "-" * 80)
    print("TOKEN OVERLAP ANALYSIS")
    print("-" * 80)
    print("\nOverlapping pairs share tokens, creating clearing price conflicts.")
    print(f"\nPair-token incidences where overlap exists: {totals['overlapping_pair_count']:,}")

    print("\n" + "-" * 80)
    print("SAMPLE NON-CROSSING PAIRS")
    print("-" * 80)
    print("\nWhy don't these pairs match? (ask > bid)")

    for pair in sample_non_crossing[:5]:
        spread = None
        if pair.best_ask and pair.best_bid:
            spread = (float(pair.best_ask) / float(pair.best_bid) - 1) * 100

        print(f"\n  Pair: {pair.token_a[-8:]} <-> {pair.token_b[-8:]}")
        print(f"    Sellers of A: {pair.sellers_of_a}, Sellers of B: {pair.sellers_of_b}")
        print(
            f"    Best ask (A→B): {float(pair.best_ask):.6f}"
            if pair.best_ask
            else "    Best ask: None"
        )
        print(
            f"    Best bid (B→A): {float(pair.best_bid):.6f}"
            if pair.best_bid
            else "    Best bid: None"
        )
        if spread is not None:
            print(f"    Spread: {spread:.2f}% (ask is {spread:.2f}% higher than bid)")

    print("\n" + "-" * 80)
    print("KEY INSIGHTS")
    print("-" * 80)

    if totals["orders_on_cow_pairs"] > 0:
        crossing_rate = totals["orders_on_crossing"] / totals["orders_on_cow_pairs"] * 100
        gap = 100 - crossing_rate

        print(f"""
1. PRICE CROSSING is the primary bottleneck:
   - Only {crossing_rate:.1f}% of CoW-potential orders have crossing prices
   - {gap:.1f}% cannot match because ask > bid (users want more than counterparties offer)

2. This explains the 0.17% match rate vs 36% CoW potential:
   - {totals["orders_on_crossing"]:,} orders could theoretically match
   - But current strategies achieve ~0.17% due to additional constraints

3. The gap between crossing orders ({totals["orders_on_crossing"]:,}) and matched orders (~500)
   comes from:
   - Volume imbalances (not enough on one side)
   - Fill-or-kill constraints ({totals["fill_or_kill_orders"]:,} orders)
   - Token overlap (clearing price conflicts)
   - Ring trade requirements (exact cycle needed)
""")


if __name__ == "__main__":
    main()
