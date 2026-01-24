#!/usr/bin/env python3
"""Benchmark existing strategies on historical auction data.

This script establishes a baseline for strategy performance:
1. CowMatchStrategy (2-order direct matching)
2. HybridCowStrategy (N-order with AMM price reference)
3. RingTradeStrategy (N-token cycles)

Metrics collected:
- Orders matched (count and percentage)
- Volume matched (in sell token amounts)
- Surplus generated (improvement over limit prices)
- Execution time

Usage:
    python scripts/benchmark_strategies.py [--limit N]
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.models.auction import AuctionInstance
from solver.strategies.cow_match import CowMatchStrategy
from solver.strategies.hybrid_cow import HybridCowStrategy
from solver.strategies.ring_trade import RingTradeStrategy


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy run."""

    name: str
    orders_matched: int = 0
    orders_total: int = 0
    volume_matched: int = 0  # Sum of sell amounts filled
    surplus: int = 0  # Improvement over limit prices
    time_ms: float = 0.0
    auctions_with_matches: int = 0
    auctions_total: int = 0


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    strategy_name: str
    metrics: list[StrategyMetrics] = field(default_factory=list)

    @property
    def total_orders_matched(self) -> int:
        return sum(m.orders_matched for m in self.metrics)

    @property
    def total_orders(self) -> int:
        return sum(m.orders_total for m in self.metrics)

    @property
    def match_rate(self) -> float:
        if self.total_orders == 0:
            return 0.0
        return self.total_orders_matched / self.total_orders * 100

    @property
    def total_volume(self) -> int:
        return sum(m.volume_matched for m in self.metrics)

    @property
    def total_surplus(self) -> int:
        return sum(m.surplus for m in self.metrics)

    @property
    def auctions_with_matches(self) -> int:
        return sum(1 for m in self.metrics if m.orders_matched > 0)

    @property
    def total_time_ms(self) -> float:
        return sum(m.time_ms for m in self.metrics)

    @property
    def avg_time_ms(self) -> float:
        if not self.metrics:
            return 0.0
        return self.total_time_ms / len(self.metrics)


def calculate_surplus(result, auction: AuctionInstance) -> int:  # noqa: ARG001
    """Calculate surplus from a strategy result.

    Surplus = sum of (what user got - minimum they required)
    """
    if result is None:
        return 0

    total_surplus = 0
    for fill in result.fills:
        order = fill.order
        # Limit price = buy_amount / sell_amount (minimum rate)
        limit_rate = Decimal(order.buy_amount_int) / Decimal(order.sell_amount_int)
        # Actual rate = buy_filled / sell_filled
        if fill.sell_filled > 0:
            actual_rate = Decimal(fill.buy_filled) / Decimal(fill.sell_filled)
            # Surplus = (actual - limit) * sell_filled (in buy token units)
            surplus = int((actual_rate - limit_rate) * fill.sell_filled)
            total_surplus += max(0, surplus)

    return total_surplus


def run_strategy(strategy, auction: AuctionInstance, strategy_name: str) -> StrategyMetrics:
    """Run a strategy and collect metrics."""
    start_time = time.perf_counter()
    result = strategy.try_solve(auction)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    metrics = StrategyMetrics(
        name=strategy_name,
        orders_total=auction.order_count,
        time_ms=elapsed_ms,
        auctions_total=1,
    )

    if result is not None and result.fills:
        metrics.orders_matched = len(result.fills)
        metrics.volume_matched = sum(f.sell_filled for f in result.fills)
        metrics.surplus = calculate_surplus(result, auction)
        metrics.auctions_with_matches = 1

    return metrics


def analyze_matching_potential(auction: AuctionInstance) -> dict:
    """Analyze the theoretical matching potential of an auction.

    Returns stats about:
    - Token pairs with orders in both directions (CoW potential)
    - Cycles in the order graph (ring trade potential)
    """
    from collections import defaultdict

    from solver.models.types import normalize_address
    from solver.strategies.ring_trade import OrderGraph

    # Build pair statistics
    pair_orders: dict[tuple[str, str], list] = defaultdict(list)
    for order in auction.orders:
        sell = normalize_address(order.sell_token)
        buy = normalize_address(order.buy_token)
        pair_orders[(sell, buy)].append(order)

    # Find pairs with potential CoW (orders in both directions)
    cow_pairs = 0
    cow_potential_orders = 0
    for (sell, buy), orders in pair_orders.items():
        if (buy, sell) in pair_orders:
            cow_pairs += 1
            cow_potential_orders += len(orders)

    # Build order graph and find cycles
    graph = OrderGraph.from_orders(auction.orders)
    cycles_3 = graph.find_3_cycles()
    cycles_4 = graph.find_4_cycles(limit=100)

    return {
        "total_orders": auction.order_count,
        "unique_tokens": len(graph.tokens),
        "directed_pairs": graph.edge_count,
        "cow_pairs": cow_pairs // 2,  # Each pair counted twice
        "cow_potential_orders": cow_potential_orders,
        "cycles_3": len(cycles_3),
        "cycles_4": len(cycles_4),
    }


def load_auction(path: Path) -> AuctionInstance:
    """Load an auction from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return AuctionInstance.model_validate(data)


def main():
    parser = argparse.ArgumentParser(description="Benchmark strategies on historical data")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of auctions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Find historical auction files
    data_dir = Path(__file__).parent.parent / "data" / "historical_auctions"
    auction_files = sorted(data_dir.glob("mainnet_*.json"))

    if args.limit:
        auction_files = auction_files[: args.limit]

    print(f"Benchmarking {len(auction_files)} historical auctions...\n")

    # Initialize strategies
    strategies = [
        ("CowMatch", CowMatchStrategy()),
        ("HybridCow", HybridCowStrategy()),
        ("RingTrade", RingTradeStrategy(max_4_cycles=100)),
    ]

    results = {name: BenchmarkResult(strategy_name=name) for name, _ in strategies}

    # Aggregate potential analysis
    total_potential = {
        "total_orders": 0,
        "cow_potential_orders": 0,
        "cow_pairs": 0,
        "cycles_3": 0,
        "cycles_4": 0,
    }

    for i, auction_file in enumerate(auction_files):
        auction = load_auction(auction_file)

        if args.verbose:
            print(
                f"[{i + 1}/{len(auction_files)}] {auction_file.name}: {auction.order_count} orders"
            )

        # Analyze potential
        potential = analyze_matching_potential(auction)
        for key in total_potential:
            if key in potential:
                total_potential[key] += potential[key]

        # Run each strategy
        for name, strategy in strategies:
            metrics = run_strategy(strategy, auction, name)
            results[name].metrics.append(metrics)

            if args.verbose and metrics.orders_matched > 0:
                print(f"  {name}: {metrics.orders_matched} orders matched")

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\nAuctions analyzed: {len(auction_files)}")
    print(f"Total orders: {total_potential['total_orders']:,}")
    print(
        f"Orders with CoW potential (on pairs with both directions): {total_potential['cow_potential_orders']:,}"
    )
    print(f"Token pairs with CoW potential: {total_potential['cow_pairs']:,}")
    print(f"3-cycles found: {total_potential['cycles_3']:,}")
    print(f"4-cycles found: {total_potential['cycles_4']:,}")

    print("\n" + "-" * 80)
    print("STRATEGY PERFORMANCE")
    print("-" * 80)

    print(
        f"\n{'Strategy':<15} {'Matched':<12} {'Rate':<10} {'Auctions':<12} {'Avg Time':<12} {'Volume':<20}"
    )
    print("-" * 80)

    for name, result in results.items():
        print(
            f"{name:<15} "
            f"{result.total_orders_matched:<12,} "
            f"{result.match_rate:<10.2f}% "
            f"{result.auctions_with_matches}/{len(result.metrics):<8} "
            f"{result.avg_time_ms:<12.2f}ms "
            f"{result.total_volume:<20,}"
        )

    # Calculate theoretical gap
    print("\n" + "-" * 80)
    print("GAP ANALYSIS")
    print("-" * 80)

    best_matched = max(r.total_orders_matched for r in results.values())
    cow_potential = total_potential["cow_potential_orders"]

    print(
        f"\nBest strategy matched: {best_matched:,} orders ({best_matched / total_potential['total_orders'] * 100:.2f}%)"
    )
    print(
        f"CoW potential (orders on bidirectional pairs): {cow_potential:,} ({cow_potential / total_potential['total_orders'] * 100:.2f}%)"
    )
    print(f"Potential improvement: {cow_potential - best_matched:,} orders")

    # Detailed per-strategy analysis
    print("\n" + "-" * 80)
    print("DETAILED ANALYSIS")
    print("-" * 80)

    for name, result in results.items():
        print(f"\n{name}:")
        print(
            f"  - Matched {result.total_orders_matched:,} of {result.total_orders:,} orders ({result.match_rate:.2f}%)"
        )
        print(
            f"  - Found matches in {result.auctions_with_matches} of {len(result.metrics)} auctions"
        )
        print(f"  - Total volume: {result.total_volume:,}")
        print(f"  - Total surplus: {result.total_surplus:,}")
        print(
            f"  - Total time: {result.total_time_ms:.1f}ms (avg {result.avg_time_ms:.2f}ms per auction)"
        )


if __name__ == "__main__":
    main()
