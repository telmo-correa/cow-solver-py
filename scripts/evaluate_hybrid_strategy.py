#!/usr/bin/env python3
"""Slice 4.3: Evaluate Hybrid CoW+AMM Strategy vs Pure-AMM Baseline.

This script runs both strategies on historical auctions and measures:
1. Win rate (which strategy produces better solutions)
2. Surplus improvement (how much better is hybrid when it wins)
3. Gas savings from CoW matches
4. Failure cases (where hybrid loses)

Usage:
    python scripts/evaluate_hybrid_strategy.py [--limit N] [--verbose]
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.models.auction import AuctionInstance
from solver.models.solution import SolverResponse
from solver.solver import Solver
from solver.strategies import AmmRoutingStrategy, CowMatchStrategy, HybridCowStrategy


@dataclass
class AuctionResult:
    """Results for a single auction."""

    auction_id: str
    order_count: int

    # Pure AMM results
    amm_solution_count: int = 0
    amm_trade_count: int = 0
    amm_gas: int = 0
    amm_has_solution: bool = False

    # Hybrid results
    hybrid_solution_count: int = 0
    hybrid_trade_count: int = 0
    hybrid_gas: int = 0
    hybrid_has_solution: bool = False
    hybrid_cow_matches: int = 0  # Trades with gas=0

    # Comparison
    winner: str = "tie"  # "hybrid", "amm", "tie", "neither"
    gas_savings: int = 0  # hybrid_gas - amm_gas (negative = savings)


@dataclass
class EvaluationSummary:
    """Summary of all auction evaluations."""

    total_auctions: int = 0
    hybrid_wins: int = 0
    amm_wins: int = 0
    ties: int = 0
    neither: int = 0  # Both failed

    total_orders: int = 0
    total_cow_matches: int = 0
    total_gas_savings: int = 0

    results: list[AuctionResult] = field(default_factory=list)

    @property
    def hybrid_win_rate(self) -> float:
        contested = self.hybrid_wins + self.amm_wins
        if contested == 0:
            return 0.0
        return self.hybrid_wins / contested * 100

    @property
    def cow_match_rate(self) -> float:
        if self.total_orders == 0:
            return 0.0
        return self.total_cow_matches / self.total_orders * 100


def create_pure_amm_solver() -> Solver:
    """Create a solver with only AMM routing (no CoW matching)."""
    return Solver(strategies=[AmmRoutingStrategy()])


def create_hybrid_solver() -> Solver:
    """Create a solver with CoW + Hybrid + AMM strategies."""
    return Solver(
        strategies=[
            CowMatchStrategy(),
            HybridCowStrategy(),
            AmmRoutingStrategy(),
        ]
    )


def count_cow_matches(response: SolverResponse) -> int:
    """Count trades that were CoW-matched (gas=0)."""
    count = 0
    for solution in response.solutions:
        # CoW solutions have no interactions
        if len(solution.interactions) == 0:
            count += len(solution.trades)
    return count


def total_gas(response: SolverResponse) -> int:
    """Calculate total gas across all solutions."""
    return sum(s.gas or 0 for s in response.solutions)


def evaluate_auction(
    auction: AuctionInstance,
    amm_solver: Solver,
    hybrid_solver: Solver,
) -> AuctionResult:
    """Run both solvers on an auction and compare results."""
    result = AuctionResult(
        auction_id=auction.id,
        order_count=auction.order_count,
    )

    # Run pure AMM solver
    try:
        amm_response = amm_solver.solve(auction)
        result.amm_solution_count = len(amm_response.solutions)
        result.amm_trade_count = sum(len(s.trades) for s in amm_response.solutions)
        result.amm_gas = total_gas(amm_response)
        result.amm_has_solution = result.amm_trade_count > 0
    except Exception as e:
        print(f"  AMM solver error: {e}")
        result.amm_has_solution = False

    # Run hybrid solver
    try:
        hybrid_response = hybrid_solver.solve(auction)
        result.hybrid_solution_count = len(hybrid_response.solutions)
        result.hybrid_trade_count = sum(len(s.trades) for s in hybrid_response.solutions)
        result.hybrid_gas = total_gas(hybrid_response)
        result.hybrid_has_solution = result.hybrid_trade_count > 0
        result.hybrid_cow_matches = count_cow_matches(hybrid_response)
    except Exception as e:
        print(f"  Hybrid solver error: {e}")
        result.hybrid_has_solution = False

    # Determine winner
    # Metric: more trades filled is better, tie-break by gas
    if not result.amm_has_solution and not result.hybrid_has_solution:
        result.winner = "neither"
    elif not result.amm_has_solution:
        result.winner = "hybrid"
    elif not result.hybrid_has_solution:
        result.winner = "amm"
    elif result.hybrid_trade_count > result.amm_trade_count:
        result.winner = "hybrid"
    elif result.amm_trade_count > result.hybrid_trade_count:
        result.winner = "amm"
    elif result.hybrid_gas < result.amm_gas:
        result.winner = "hybrid"  # Same trades, less gas
    elif result.amm_gas < result.hybrid_gas:
        result.winner = "amm"
    else:
        result.winner = "tie"

    result.gas_savings = result.amm_gas - result.hybrid_gas

    return result


def print_summary(summary: EvaluationSummary, verbose: bool = False) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("SLICE 4.3: HYBRID COW+AMM STRATEGY EVALUATION")
    print("=" * 60)

    print(f"\nTotal auctions evaluated: {summary.total_auctions}")
    print(f"Total orders processed: {summary.total_orders}")

    print("\n--- Win Rate ---")
    print(
        f"Hybrid wins: {summary.hybrid_wins} ({summary.hybrid_wins / summary.total_auctions * 100:.1f}%)"
    )
    print(
        f"AMM wins:    {summary.amm_wins} ({summary.amm_wins / summary.total_auctions * 100:.1f}%)"
    )
    print(f"Ties:        {summary.ties} ({summary.ties / summary.total_auctions * 100:.1f}%)")
    print(f"Neither:     {summary.neither} ({summary.neither / summary.total_auctions * 100:.1f}%)")

    contested = summary.hybrid_wins + summary.amm_wins
    if contested > 0:
        print(f"\nContested win rate (Hybrid): {summary.hybrid_win_rate:.1f}%")

    print("\n--- CoW Matching ---")
    print(f"Total CoW matches: {summary.total_cow_matches}")
    print(f"CoW match rate: {summary.cow_match_rate:.2f}% of orders")

    print("\n--- Gas Analysis ---")
    print(f"Total gas savings: {summary.total_gas_savings:,}")
    avg_savings = (
        summary.total_gas_savings / summary.total_auctions if summary.total_auctions > 0 else 0
    )
    print(f"Average gas savings per auction: {avg_savings:,.0f}")

    # Decision criteria
    print("\n" + "=" * 60)
    print("DECISION CRITERIA")
    print("=" * 60)

    surplus_improvement = summary.hybrid_win_rate

    if surplus_improvement >= 5:
        print(f"\n✅ Hybrid CoW adds SIGNIFICANT value ({surplus_improvement:.1f}% win rate)")
        print("   Recommendation: Proceed to RING TRADES (Slice 4.4)")
        print("   Ring trades extend CoW concept across token cycles")
    else:
        print(f"\n⚠️  Hybrid CoW adds MARGINAL value ({surplus_improvement:.1f}% win rate)")
        print("   Recommendation: Consider SPLIT ROUTING or PERFORMANCE OPTIMIZATION")

    if verbose:
        print("\n--- Detailed Results ---")
        for r in summary.results:
            winner_mark = "🏆" if r.winner == "hybrid" else ("❌" if r.winner == "amm" else "➖")
            print(
                f"{winner_mark} {r.auction_id}: orders={r.order_count}, "
                f"hybrid_trades={r.hybrid_trade_count}, amm_trades={r.amm_trade_count}, "
                f"cow_matches={r.hybrid_cow_matches}"
            )


def sample_orders_randomly(
    auction: AuctionInstance, max_orders: int = 50, seed: int = 42
) -> AuctionInstance:
    """Randomly sample orders from auction for unbiased evaluation.

    Uses deterministic seed for reproducibility.
    """
    import random

    rng = random.Random(seed)

    orders = list(auction.orders)
    if len(orders) <= max_orders:
        return auction

    sampled = rng.sample(orders, max_orders)
    return auction.model_copy(update={"orders": sampled})


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid CoW+AMM Strategy")
    parser.add_argument("--limit", type=int, default=50, help="Max auctions to process")
    parser.add_argument(
        "--max-orders",
        type=int,
        default=100,
        help="Max orders per auction (samples CoW-eligible first)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/historical_auctions",
        help="Directory containing historical auction JSON files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please ensure historical auctions are available.")
        sys.exit(1)

    auction_files = sorted(data_dir.glob("*.json"))[: args.limit]

    if not auction_files:
        print(f"Error: No auction files found in {data_dir}")
        sys.exit(1)

    print(f"Loading {len(auction_files)} historical auctions...")

    # Create solvers
    amm_solver = create_pure_amm_solver()
    hybrid_solver = create_hybrid_solver()

    summary = EvaluationSummary()

    for i, auction_file in enumerate(auction_files, 1):
        print(f"\n[{i}/{len(auction_files)}] Processing {auction_file.name}...")

        try:
            with open(auction_file) as f:
                auction_data = json.load(f)
            auction = AuctionInstance.model_validate(auction_data)
            original_count = auction.order_count
            # Random sample for unbiased evaluation
            auction = sample_orders_randomly(auction, max_orders=args.max_orders, seed=i)
            print(f"  Sampled {auction.order_count} of {original_count} orders (random)")
        except Exception as e:
            print(f"  Error loading auction: {e}")
            continue

        result = evaluate_auction(auction, amm_solver, hybrid_solver)

        # Update summary
        summary.total_auctions += 1
        summary.total_orders += result.order_count
        summary.total_cow_matches += result.hybrid_cow_matches
        summary.total_gas_savings += result.gas_savings
        summary.results.append(result)

        if result.winner == "hybrid":
            summary.hybrid_wins += 1
        elif result.winner == "amm":
            summary.amm_wins += 1
        elif result.winner == "tie":
            summary.ties += 1
        else:
            summary.neither += 1

        # Progress indicator
        print(
            f"  Orders: {result.order_count}, Winner: {result.winner}, "
            f"CoW matches: {result.hybrid_cow_matches}"
        )

    print_summary(summary, args.verbose)

    # Save results to file
    output_file = Path("docs/slice-4.3-evaluation-results.md")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        f.write("# Slice 4.3: Hybrid Strategy Evaluation Results\n\n")
        f.write("**Date:** 2026-01-23\n")
        f.write(f"**Auctions evaluated:** {summary.total_auctions}\n")
        f.write(f"**Total orders:** {summary.total_orders}\n\n")

        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(
            f"| Hybrid wins | {summary.hybrid_wins} ({summary.hybrid_wins / summary.total_auctions * 100:.1f}%) |\n"
        )
        f.write(
            f"| AMM wins | {summary.amm_wins} ({summary.amm_wins / summary.total_auctions * 100:.1f}%) |\n"
        )
        f.write(f"| Ties | {summary.ties} ({summary.ties / summary.total_auctions * 100:.1f}%) |\n")
        f.write(
            f"| Neither | {summary.neither} ({summary.neither / summary.total_auctions * 100:.1f}%) |\n"
        )
        f.write(f"| **Contested win rate** | **{summary.hybrid_win_rate:.1f}%** |\n")
        f.write(f"| CoW matches | {summary.total_cow_matches} |\n")
        f.write(f"| CoW match rate | {summary.cow_match_rate:.2f}% |\n")
        f.write(f"| Total gas savings | {summary.total_gas_savings:,} |\n\n")

        f.write("## Decision\n\n")
        if summary.hybrid_win_rate >= 5:
            f.write("**Recommendation:** Proceed to Ring Trades (Slice 4.4)\n\n")
            f.write(
                "Hybrid CoW adds significant value. Ring trades extend the CoW concept across token cycles.\n"
            )
        else:
            f.write("**Recommendation:** Consider Split Routing or Performance Optimization\n\n")
            f.write(
                "Hybrid CoW adds marginal value. Focus on split routing for large orders or performance.\n"
            )

    print(f"\n📄 Results saved to {output_file}")


if __name__ == "__main__":
    main()
