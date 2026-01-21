#!/usr/bin/env python3
"""CLI script for running benchmark comparisons between Python and Rust solvers.

Usage:
    # Compare both solvers via HTTP (recommended for fair comparison)
    python scripts/run_benchmarks.py \\
        --python-url http://localhost:8000 \\
        --rust-url http://localhost:8080

    # Run with just one solver for sanity checking
    python scripts/run_benchmarks.py --rust-url http://localhost:8080
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import structlog

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.harness import BenchmarkResult, BenchmarkSummary, SolverResult  # noqa: E402
from benchmarks.metrics import compute_metrics  # noqa: E402
from benchmarks.report import save_report  # noqa: E402
from solver.models.solution import (  # noqa: E402
    CustomInteraction,
    LiquidityInteraction,
    Solution,
    SolverResponse,
)

logger = structlog.get_logger()


def extract_output_amounts(solution: Solution) -> dict[str, int]:
    """Extract output amounts from a solution's interactions by token.

    Returns a dict mapping token address (lowercased) to total output amount.
    Works with both CustomInteraction (Python) and LiquidityInteraction (Rust).
    """
    outputs: dict[str, int] = {}

    for interaction in solution.interactions:
        if isinstance(interaction, LiquidityInteraction):
            # Rust solver uses LiquidityInteraction with output_amount
            token = interaction.output_token.lower()
            amount = int(interaction.output_amount)
            outputs[token] = outputs.get(token, 0) + amount
        elif isinstance(interaction, CustomInteraction):
            # Python solver uses CustomInteraction with outputs list
            for output in interaction.outputs:
                token = output.token.lower()
                amount = int(output.amount)
                outputs[token] = outputs.get(token, 0) + amount

    return outputs


def compare_solutions(
    python_response: SolverResponse | None,
    rust_response: SolverResponse | None,
) -> dict[str, str | bool | dict]:
    """Compare solutions from Python and Rust solvers.

    Returns a dict with comparison results:
    - match: bool indicating if solutions are equivalent (common outputs match)
    - python_outputs: dict of token -> amount for Python solution
    - rust_outputs: dict of token -> amount for Rust solution
    - diff: description of any differences

    Note: For multi-hop routes, Rust reports intermediate token amounts while Python
    uses path encoding that only reports final outputs. We compare only the tokens
    that both solvers report (the final output tokens).
    """
    result: dict[str, str | bool | dict] = {
        "match": False,
        "python_outputs": {},
        "rust_outputs": {},
        "diff": "",
    }

    # Handle missing responses
    if not python_response or not python_response.solutions:
        if not rust_response or not rust_response.solutions:
            result["match"] = True  # Both empty
            result["diff"] = "Both solvers returned no solutions"
            return result
        result["diff"] = "Python returned no solutions, Rust did"
        rust_outputs = extract_output_amounts(rust_response.solutions[0])
        result["rust_outputs"] = rust_outputs
        return result

    if not rust_response or not rust_response.solutions:
        result["diff"] = "Rust returned no solutions, Python did"
        python_outputs = extract_output_amounts(python_response.solutions[0])
        result["python_outputs"] = python_outputs
        return result

    # Extract outputs from first solution of each
    python_outputs = extract_output_amounts(python_response.solutions[0])
    rust_outputs = extract_output_amounts(rust_response.solutions[0])

    result["python_outputs"] = python_outputs
    result["rust_outputs"] = rust_outputs

    # Compare outputs
    if not python_outputs and not rust_outputs:
        result["match"] = True
        result["diff"] = "Both have no interaction outputs"
        return result

    # Find common tokens (present in both outputs)
    common_tokens = set(python_outputs.keys()) & set(rust_outputs.keys())
    python_only_tokens = set(python_outputs.keys()) - set(rust_outputs.keys())
    rust_only_tokens = set(rust_outputs.keys()) - set(python_outputs.keys())

    differences = []
    notes = []

    # Compare common tokens (must match exactly)
    for token in common_tokens:
        py_amt = python_outputs[token]
        rs_amt = rust_outputs[token]
        if py_amt != rs_amt:
            # Calculate relative difference
            if rs_amt > 0:
                rel_diff = abs(py_amt - rs_amt) / rs_amt * 100
                differences.append(
                    f"{token[:10]}...: Python={py_amt}, Rust={rs_amt} ({rel_diff:.2f}% diff)"
                )
            else:
                differences.append(f"{token[:10]}...: Python={py_amt}, Rust={rs_amt}")

    # Note tokens only in one solver (likely intermediate tokens in multi-hop)
    if rust_only_tokens:
        # Rust often reports intermediate tokens in multi-hop that Python doesn't
        intermediate_note = f"Rust has {len(rust_only_tokens)} extra token(s) (likely intermediate)"
        notes.append(intermediate_note)

    if python_only_tokens:
        # Less common, but note if Python has tokens Rust doesn't
        extra_note = f"Python has {len(python_only_tokens)} extra token(s)"
        notes.append(extra_note)

    # Build result
    if differences:
        result["diff"] = "; ".join(differences)
        if notes:
            result["diff"] += f" [{', '.join(notes)}]"
    else:
        result["match"] = True
        if notes:
            result["diff"] = f"Common outputs match [{', '.join(notes)}]"
        else:
            result["diff"] = "Solutions match"

    return result


async def run_http_solver(
    client: httpx.AsyncClient,
    solver_name: str,
    base_url: str,
    auction_json: dict,
    auction_id: str,
    timeout: float = 30.0,
) -> SolverResult:
    """Run a solver via HTTP and return the result."""
    # Python solver uses /{env}/{network} endpoint, Rust uses /solve
    url = f"{base_url}/benchmark/mainnet" if solver_name == "python" else f"{base_url}/solve"

    start = time.perf_counter()
    try:
        response = await client.post(url, json=auction_json, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException:
        return SolverResult(
            solver_name=solver_name,
            auction_id=auction_id,
            solution_count=0,
            best_score=None,
            elapsed_ms=timeout * 1000,
            error=f"Timeout after {timeout}s",
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SolverResult(
            solver_name=solver_name,
            auction_id=auction_id,
            solution_count=0,
            best_score=None,
            elapsed_ms=elapsed_ms,
            error=str(e),
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    try:
        solver_response = SolverResponse.model_validate(data)
    except Exception as e:
        return SolverResult(
            solver_name=solver_name,
            auction_id=auction_id,
            solution_count=0,
            best_score=None,
            elapsed_ms=elapsed_ms,
            error=f"Failed to parse response: {e}",
        )

    # Extract best score
    best_score = None
    for sol in solver_response.solutions:
        if sol.score is not None:
            score = int(sol.score)
            if best_score is None or score > best_score:
                best_score = score

    return SolverResult(
        solver_name=solver_name,
        auction_id=auction_id,
        solution_count=len(solver_response.solutions),
        best_score=best_score,
        elapsed_ms=elapsed_ms,
        response=solver_response,
    )


async def run_benchmarks_http(
    auctions_dir: Path,
    python_url: str | None = None,
    rust_url: str | None = None,
) -> BenchmarkSummary:
    """Run benchmarks comparing both solvers via HTTP.

    This is the recommended approach for fair comparison since both
    solvers are tested through their HTTP interfaces.
    """
    if not auctions_dir.exists():
        logger.warning("auctions_dir_not_found", path=str(auctions_dir))
        return BenchmarkSummary(total_auctions=0, successful_auctions=0)

    auction_files = list(auctions_dir.rglob("*.json"))
    logger.info("found_auctions", count=len(auction_files), dir=str(auctions_dir))

    results: list[BenchmarkResult] = []

    async with httpx.AsyncClient() as client:
        for path in auction_files:
            try:
                with open(path) as f:
                    auction_json = json.load(f)

                name = path.stem
                auction_id = auction_json.get("id", name)
                order_count = len(auction_json.get("orders", []))

                logger.info("benchmarking_auction", name=name, order_count=order_count)

                # Run Python solver if URL provided
                if python_url:
                    python_result = await run_http_solver(
                        client, "python", python_url, auction_json, auction_id
                    )
                else:
                    python_result = SolverResult(
                        solver_name="python",
                        auction_id=auction_id,
                        solution_count=0,
                        best_score=None,
                        elapsed_ms=0,
                        error="Python solver not configured",
                    )

                # Run Rust solver if URL provided
                if rust_url:
                    rust_result = await run_http_solver(
                        client, "rust", rust_url, auction_json, auction_id
                    )
                else:
                    rust_result = SolverResult(
                        solver_name="rust",
                        auction_id=auction_id,
                        solution_count=0,
                        best_score=None,
                        elapsed_ms=0,
                        error="Rust solver not configured",
                    )

                results.append(
                    BenchmarkResult(
                        auction_name=name,
                        auction_id=auction_id,
                        order_count=order_count,
                        python_result=python_result,
                        rust_result=rust_result,
                    )
                )
            except Exception as e:
                logger.error("auction_error", path=str(path), error=str(e))

    return BenchmarkSummary(
        total_auctions=len(auction_files),
        successful_auctions=len(results),
        results=results,
    )


async def main() -> int:
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run benchmark comparisons between Python and Rust CoW solvers via HTTP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmarks.py \\
      --python-url http://localhost:8000 --rust-url http://localhost:8080

Before running:
  1. Start Python solver: python -m solver.api.main (port 8000)
  2. Start Rust solver: ./target/release/solvers --addr 127.0.0.1:8080 ...
        """,
    )

    parser.add_argument(
        "--auctions",
        type=Path,
        default=project_root / "tests" / "fixtures" / "auctions" / "benchmark",
        help="Directory containing auction JSON fixtures",
    )
    parser.add_argument(
        "--python-url",
        type=str,
        default=None,
        help="URL of Python solver HTTP server (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--rust-url",
        type=str,
        default=None,
        help="URL of Rust solver HTTP server (e.g., http://localhost:8080)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "benchmarks" / "results",
        help="Directory for output reports (default: benchmarks/results)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    import logging

    log_level = logging.DEBUG if args.verbose else logging.INFO
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

    # Validate inputs
    if not args.auctions.exists():
        logger.error("auctions_dir_not_found", path=str(args.auctions))
        print(f"Error: Auctions directory not found: {args.auctions}")
        return 1

    if not args.python_url and not args.rust_url:
        print("Error: At least one of --python-url or --rust-url must be provided")
        return 1

    # Run benchmarks
    print("=" * 60)
    print("CoW Protocol Solver Benchmark (HTTP)")
    print("=" * 60)
    print(f"Auctions directory: {args.auctions}")
    print(f"Python solver: {args.python_url or 'Not configured'}")
    print(f"Rust solver:   {args.rust_url or 'Not configured'}")
    print()

    summary = await run_benchmarks_http(
        auctions_dir=args.auctions,
        python_url=args.python_url,
        rust_url=args.rust_url,
    )

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)

    # Compute and display metrics
    metrics = compute_metrics(summary.results)

    print(f"Total auctions: {summary.total_auctions}")
    print(f"Successful: {summary.successful_auctions}")
    print()

    # Count solutions
    python_solutions = sum(1 for r in summary.results if r.python_result.solution_count > 0)
    rust_solutions = sum(1 for r in summary.results if r.rust_result.solution_count > 0)
    print(f"Python found solutions: {python_solutions}/{len(summary.results)}")
    print(f"Rust found solutions:   {rust_solutions}/{len(summary.results)}")
    print()

    if metrics.time_ratio_stats:
        ts = metrics.time_ratio_stats
        print("Time Comparison (Python / Rust):")
        print(f"  Mean:   {ts.mean:.2f}x")
        print(f"  Median: {ts.median:.2f}x")
        print(f"  Range:  {ts.min:.2f}x - {ts.max:.2f}x")
        faster = "Python" if ts.mean < 1 else "Rust"
        print(f"  Faster: {faster}")
        print()

    if metrics.score_ratio_stats:
        ss = metrics.score_ratio_stats
        print("Score Comparison (Python / Rust):")
        print(f"  Mean:   {ss.mean:.2%}")
        print(f"  Median: {ss.median:.2%}")
        print(f"  Range:  {ss.min:.2%} - {ss.max:.2%}")
        print()

    # Show individual results with solution comparison
    if summary.results:
        print("Individual Results:")
        print("-" * 60)

        matches = 0
        mismatches = 0

        for r in summary.results:
            py_ok = r.python_result.error is None and r.python_result.solution_count > 0
            rs_ok = r.rust_result.error is None and r.rust_result.solution_count > 0
            py_status = "✓" if py_ok else ("○" if r.python_result.error is None else "✗")
            rs_status = "✓" if rs_ok else ("○" if r.rust_result.error is None else "✗")

            # Compare solutions
            comparison = compare_solutions(
                r.python_result.response,
                r.rust_result.response,
            )
            match_status = "✓" if comparison["match"] else "✗"
            if comparison["match"]:
                matches += 1
            else:
                mismatches += 1

            print(f"  {r.auction_name}:")
            print(
                f"    Python [{py_status}]: {r.python_result.elapsed_ms:.1f}ms, "
                f"solutions={r.python_result.solution_count}"
            )
            print(
                f"    Rust   [{rs_status}]: {r.rust_result.elapsed_ms:.1f}ms, "
                f"solutions={r.rust_result.solution_count}"
            )
            print(f"    Match  [{match_status}]: {comparison['diff']}")

            # Show output amounts if available and different
            if not comparison["match"]:
                if comparison["python_outputs"]:
                    for token, amt in comparison["python_outputs"].items():
                        print(f"      Python output: {token[:16]}... = {amt}")
                if comparison["rust_outputs"]:
                    for token, amt in comparison["rust_outputs"].items():
                        print(f"      Rust output:   {token[:16]}... = {amt}")

            if r.python_result.error:
                print(f"      Python error: {r.python_result.error[:80]}")
            if r.rust_result.error:
                print(f"      Rust error: {r.rust_result.error[:80]}")
        print()

        # Summary of solution comparisons
        print("Solution Comparison Summary:")
        print(f"  Matching: {matches}/{len(summary.results)}")
        print(f"  Mismatching: {mismatches}/{len(summary.results)}")
        if mismatches > 0:
            print("  WARNING: Some solutions differ between Python and Rust solvers!")
        print()

    # Save reports
    args.output.mkdir(parents=True, exist_ok=True)
    formats = ["markdown", "json"] if args.format == "both" else [args.format]
    paths = save_report(summary, args.output, formats)

    print("Reports saved to:")
    for p in paths:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
