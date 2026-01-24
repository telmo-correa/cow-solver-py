#!/usr/bin/env python3
"""Compare HybridCowStrategy vs AmmRoutingStrategy on CoW-eligible auctions.

This script validates the Slice 4.2 exit criteria:
"Hybrid strategy outperforms pure-AMM routing on at least 20% of CoW-eligible auctions."

Usage:
    python scripts/compare_strategies.py \
        --auctions tests/fixtures/auctions/benchmark_python_only

    python scripts/compare_strategies.py \
        --auctions tests/fixtures/auctions/n_order_cow \
        --verbose
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import structlog

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from solver.models.auction import AuctionInstance  # noqa: E402
from solver.models.order_groups import find_cow_opportunities  # noqa: E402
from solver.strategies.amm_routing import AmmRoutingStrategy  # noqa: E402
from solver.strategies.base import StrategyResult  # noqa: E402
from solver.strategies.research.hybrid_cow import HybridCowStrategy  # noqa: E402

logger = structlog.get_logger()


@dataclass
class ComparisonResult:
    """Result of comparing strategies on a single auction."""

    auction_name: str
    auction_id: str
    order_count: int
    is_cow_eligible: bool

    # Strategy results
    hybrid_result: StrategyResult | None = None
    amm_result: StrategyResult | None = None

    # Surplus (computed from fills)
    hybrid_surplus: int = 0
    amm_surplus: int = 0

    # Winner
    winner: str = "tie"  # "hybrid", "amm", "tie"

    error: str | None = None


@dataclass
class ComparisonSummary:
    """Summary statistics from comparing strategies."""

    total_auctions: int = 0
    cow_eligible_auctions: int = 0
    hybrid_wins: int = 0
    amm_wins: int = 0
    ties: int = 0

    results: list[ComparisonResult] = field(default_factory=list)

    @property
    def hybrid_win_rate(self) -> float:
        """Percentage of CoW-eligible auctions where hybrid wins."""
        if self.cow_eligible_auctions == 0:
            return 0.0
        return self.hybrid_wins / self.cow_eligible_auctions

    @property
    def passes_exit_criteria(self) -> bool:
        """Check if exit criteria is met (20% improvement rate)."""
        return self.hybrid_win_rate >= 0.20


def compute_surplus(result: StrategyResult | None, auction: AuctionInstance) -> int:
    """Compute total user surplus from a strategy result.

    Surplus is the total value delivered to users, combining:
    1. Price surplus: extra tokens received beyond minimum required
    2. Gas savings: value of gas saved from 0-gas CoW matches

    For sell orders: surplus = buy_filled - buy_amount (at execution price)
    For buy orders: surplus = sell_amount - sell_filled (at execution price)

    Gas savings are computed as:
        gas_cost_wei = gas * effective_gas_price
        Gas is converted to a comparable unit using token reference prices.

    Returns 0 if result is None or has no fills.
    """
    if result is None or not result.fills:
        return 0

    # Price surplus
    total_price_surplus = 0
    for fill in result.fills:
        order = fill.order
        if order.kind == "sell":
            # Sell order: surplus is extra tokens received beyond minimum
            minimum_buy = int(int(order.buy_amount) * fill.sell_filled // int(order.sell_amount))
            surplus = fill.buy_filled - minimum_buy
        else:
            # Buy order: surplus is tokens saved from maximum allowed sell
            maximum_sell = int(int(order.sell_amount) * fill.buy_filled // int(order.buy_amount))
            surplus = maximum_sell - fill.sell_filled

        total_price_surplus += max(0, surplus)

    # Gas savings (value of avoided gas costs)
    # Use effective gas price from auction to convert gas to value
    # For CoW matches (gas=0), this represents savings vs AMM routing
    effective_gas_price = int(auction.effective_gas_price or "30000000000")  # Default 30 gwei

    # Estimate gas cost of AMM routing (110,000 gas per order)
    num_fills = len(result.fills)
    amm_gas_estimate = 110_000 * num_fills  # Gas if routed through AMM

    # Actual gas used by this solution
    actual_gas = result.gas or 0

    # Gas savings in wei
    gas_savings_wei = (amm_gas_estimate - actual_gas) * effective_gas_price

    # Convert gas savings to a normalized unit (use 1e18 as base)
    # This puts gas savings on a scale comparable to token amounts
    # For a typical order of 1 ETH (~2500 USDC), gas of 110k at 30 gwei = 0.0033 ETH
    # That's about 0.33% of the trade value, so gas_savings_wei is a reasonable metric

    return total_price_surplus + max(0, gas_savings_wei)


def is_cow_eligible(auction: AuctionInstance) -> bool:
    """Check if an auction has CoW potential.

    An auction is CoW-eligible if it has 2+ orders on the same token pair
    with orders in both directions.
    """
    if auction.order_count < 2:
        return False

    cow_groups = find_cow_opportunities(auction.orders)
    return len(cow_groups) > 0


def compare_auction(
    auction: AuctionInstance,
    auction_name: str,
    hybrid_strategy: HybridCowStrategy,
    amm_strategy: AmmRoutingStrategy,
) -> ComparisonResult:
    """Compare strategies on a single auction."""
    result = ComparisonResult(
        auction_name=auction_name,
        auction_id=auction.id or auction_name,
        order_count=auction.order_count,
        is_cow_eligible=is_cow_eligible(auction),
    )

    try:
        # Run hybrid strategy
        result.hybrid_result = hybrid_strategy.try_solve(auction)

        # Run AMM-only strategy
        result.amm_result = amm_strategy.try_solve(auction)

        # Compute surplus
        result.hybrid_surplus = compute_surplus(result.hybrid_result, auction)
        result.amm_surplus = compute_surplus(result.amm_result, auction)

        # Determine winner
        if result.hybrid_surplus > result.amm_surplus:
            result.winner = "hybrid"
        elif result.amm_surplus > result.hybrid_surplus:
            result.winner = "amm"
        else:
            result.winner = "tie"

    except Exception as e:
        result.error = str(e)
        logger.error("comparison_error", auction=auction_name, error=str(e))

    return result


def run_comparison(auctions_dirs: list[Path], verbose: bool = False) -> ComparisonSummary:
    """Run strategy comparison on all auctions in the given directories."""
    summary = ComparisonSummary()

    # Initialize strategies
    hybrid_strategy = HybridCowStrategy()
    amm_strategy = AmmRoutingStrategy()

    # Find all auction files
    auction_files: list[Path] = []
    for auctions_dir in auctions_dirs:
        if not auctions_dir.exists():
            logger.warning("auctions_dir_not_found", path=str(auctions_dir))
            continue
        auction_files.extend(auctions_dir.rglob("*.json"))

    logger.info("found_auctions", count=len(auction_files))

    for path in sorted(auction_files):
        try:
            with open(path) as f:
                auction_json = json.load(f)

            # Skip expected output files
            if path.name.endswith("_expected.json"):
                continue

            auction = AuctionInstance.model_validate(auction_json)
            name = path.stem

            result = compare_auction(auction, name, hybrid_strategy, amm_strategy)
            summary.results.append(result)
            summary.total_auctions += 1

            if result.is_cow_eligible:
                summary.cow_eligible_auctions += 1
                if result.winner == "hybrid":
                    summary.hybrid_wins += 1
                elif result.winner == "amm":
                    summary.amm_wins += 1
                else:
                    summary.ties += 1

            if verbose:
                cow_status = "CoW" if result.is_cow_eligible else "---"
                winner_emoji = (
                    "H" if result.winner == "hybrid" else ("A" if result.winner == "amm" else "=")
                )
                print(
                    f"  [{cow_status}] {name}: {winner_emoji} "
                    f"(hybrid={result.hybrid_surplus}, amm={result.amm_surplus})"
                )

        except Exception as e:
            logger.error("auction_load_error", path=str(path), error=str(e))

    return summary


def print_summary(summary: ComparisonSummary) -> None:
    """Print comparison summary."""
    print()
    print("=" * 60)
    print("Strategy Comparison Results")
    print("=" * 60)
    print()
    print(f"Total auctions:      {summary.total_auctions}")
    print(f"CoW-eligible:        {summary.cow_eligible_auctions}")
    print()

    if summary.cow_eligible_auctions > 0:
        print("On CoW-eligible auctions:")
        print(
            f"  HybridCow wins:    {summary.hybrid_wins} "
            f"({summary.hybrid_wins / summary.cow_eligible_auctions:.1%})"
        )
        print(
            f"  AmmRouting wins:   {summary.amm_wins} "
            f"({summary.amm_wins / summary.cow_eligible_auctions:.1%})"
        )
        print(
            f"  Ties:              {summary.ties} "
            f"({summary.ties / summary.cow_eligible_auctions:.1%})"
        )
        print()

        # Exit criteria check
        print("-" * 60)
        threshold = 0.20
        status = "PASS" if summary.passes_exit_criteria else "FAIL"
        print(f"Exit Criteria: {summary.hybrid_win_rate:.1%} >= {threshold:.0%} -> {status}")
        print("-" * 60)

        if not summary.passes_exit_criteria:
            print()
            print("NOTE: Exit criteria not met. This may indicate:")
            print("  - Need more CoW-eligible fixtures with N>2 orders")
            print("  - Fixtures need overlapping limit prices")
            print("  - Current fixtures are already well-handled by CowMatchStrategy")
    else:
        print("No CoW-eligible auctions found.")
        print("Add fixtures to tests/fixtures/auctions/n_order_cow/")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare HybridCowStrategy vs AmmRoutingStrategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--auctions",
        type=Path,
        action="append",
        default=[],
        help="Directory containing auction JSON fixtures (can be specified multiple times)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show results for each auction",
    )

    args = parser.parse_args()

    # Default to benchmark_python_only if no directories specified
    if not args.auctions:
        args.auctions = [
            project_root / "tests" / "fixtures" / "auctions" / "benchmark_python_only",
        ]

    # Configure logging
    import logging

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

    print("CoW Protocol Strategy Comparison")
    print("=" * 60)
    print(f"Auction directories: {[str(p) for p in args.auctions]}")
    print()

    summary = run_comparison(args.auctions, verbose=args.verbose)
    print_summary(summary)

    # Return exit code based on criteria
    return 0 if summary.passes_exit_criteria else 1


if __name__ == "__main__":
    sys.exit(main())
