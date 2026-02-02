#!/usr/bin/env python3
"""Profile the solver on historical auctions to identify performance bottlenecks."""

import argparse
import cProfile
import json
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.models.auction import AuctionInstance
from solver.solver import Solver


def load_auction(path: Path) -> AuctionInstance:
    """Load auction from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return AuctionInstance.model_validate(data)


def profile_solve(auction: AuctionInstance, solver: Solver) -> tuple[float, pstats.Stats]:
    """Profile a single solve and return timing + stats."""
    profiler = cProfile.Profile()

    start = time.perf_counter()
    profiler.enable()
    result = solver.solve(auction)
    profiler.disable()
    elapsed = time.perf_counter() - start

    # Create stats object (will print to stdout by default)
    stats = pstats.Stats(profiler)

    return elapsed, stats, result


def main():
    parser = argparse.ArgumentParser(description="Profile solver on historical auctions")
    parser.add_argument(
        "--auction",
        type=Path,
        help="Specific auction file to profile",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of auctions to profile (default: 1)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top functions to show (default: 30)",
    )
    parser.add_argument(
        "--sort",
        choices=["cumtime", "tottime", "calls", "ncalls"],
        default="cumtime",
        help="Sort key for profile output (default: cumtime)",
    )
    parser.add_argument(
        "--callers",
        action="store_true",
        help="Show callers for top functions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save profile stats to file (for later analysis)",
    )
    args = parser.parse_args()

    # Find auction files
    data_dir = Path(__file__).parent.parent / "data" / "historical_auctions"
    if args.auction:
        auction_files = [args.auction]
    elif data_dir.exists():
        auction_files = sorted(data_dir.glob("*.json"))[: args.limit]
    else:
        print(f"Error: No auction files found in {data_dir}")
        sys.exit(1)

    if not auction_files:
        print("No auction files to profile")
        sys.exit(1)

    # Create solver (no V3 RPC to keep it simple)
    solver = Solver()

    # Aggregate stats across all auctions
    all_stats = None
    total_time = 0.0
    total_orders = 0
    total_solutions = 0

    for auction_file in auction_files:
        print(f"\n{'='*60}")
        print(f"Profiling: {auction_file.name}")
        print(f"{'='*60}")

        auction = load_auction(auction_file)
        print(f"Orders: {len(auction.orders)}, Tokens: {len(auction.tokens)}, Liquidity: {len(auction.liquidity)}")

        elapsed, stats, result = profile_solve(auction, solver)
        total_time += elapsed
        total_orders += len(auction.orders)
        total_solutions += len(result.solutions) if result.solutions else 0

        print(f"Solve time: {elapsed:.3f}s")
        print(f"Solutions: {len(result.solutions) if result.solutions else 0}")

        if all_stats is None:
            all_stats = stats
        else:
            all_stats.add(stats)

    # Print summary
    print(f"\n{'='*60}")
    print("AGGREGATE PROFILE RESULTS")
    print(f"{'='*60}")
    print(f"Auctions profiled: {len(auction_files)}")
    print(f"Total orders: {total_orders}")
    print(f"Total solutions: {total_solutions}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Avg time per auction: {total_time / len(auction_files):.3f}s")
    print(f"Avg time per order: {total_time / total_orders * 1000:.3f}ms")

    # Print top functions
    print(f"\n{'='*60}")
    print(f"TOP {args.top} FUNCTIONS BY {args.sort.upper()}")
    print(f"{'='*60}")

    all_stats.sort_stats(args.sort)
    all_stats.print_stats(args.top)

    if args.callers:
        print(f"\n{'='*60}")
        print("CALLERS FOR TOP FUNCTIONS")
        print(f"{'='*60}")
        all_stats.print_callers(args.top // 2)

    # Save stats if requested
    if args.output:
        all_stats.dump_stats(str(args.output))
        print(f"\nProfile stats saved to: {args.output}")


if __name__ == "__main__":
    main()
