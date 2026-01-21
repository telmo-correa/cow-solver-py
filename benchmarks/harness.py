"""Main benchmark harness for comparing Python and Rust solvers."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import structlog

from benchmarks.rust_runner import RustSolverRunner
from solver.models.auction import AuctionInstance
from solver.models.solution import SolverResponse

logger = structlog.get_logger()


@dataclass
class SolverResult:
    """Result from running a solver on an auction."""

    solver_name: str
    auction_id: str | None
    solution_count: int
    best_score: int | None  # Parsed from solution, may be None if not computed
    elapsed_ms: float
    error: str | None = None
    response: SolverResponse | None = None


@dataclass
class BenchmarkResult:
    """Comparison result for a single auction."""

    auction_name: str
    auction_id: str | None
    order_count: int
    python_result: SolverResult
    rust_result: SolverResult

    @property
    def time_ratio(self) -> float | None:
        """Python time / Rust time. >1 means Python is slower."""
        if self.rust_result.elapsed_ms == 0:
            return None
        return self.python_result.elapsed_ms / self.rust_result.elapsed_ms

    @property
    def score_ratio(self) -> float | None:
        """Python score / Rust score. <1 means Python found worse solution."""
        if self.rust_result.best_score is None or self.rust_result.best_score == 0:
            return None
        if self.python_result.best_score is None:
            return 0.0
        return self.python_result.best_score / self.rust_result.best_score


@dataclass
class BenchmarkSummary:
    """Summary statistics across all benchmarked auctions."""

    total_auctions: int
    successful_auctions: int
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def avg_time_ratio(self) -> float | None:
        """Average time ratio across successful auctions."""
        ratios = [r.time_ratio for r in self.results if r.time_ratio is not None]
        if not ratios:
            return None
        return sum(ratios) / len(ratios)

    @property
    def avg_score_ratio(self) -> float | None:
        """Average score ratio across successful auctions."""
        ratios = [r.score_ratio for r in self.results if r.score_ratio is not None]
        if not ratios:
            return None
        return sum(ratios) / len(ratios)


class SolverProtocol(Protocol):
    """Protocol for solver implementations."""

    async def solve(self, auction: AuctionInstance) -> SolverResponse:
        """Solve an auction and return proposed solutions."""
        ...


class PythonSolverRunner:
    """Runner for the Python solver."""

    def __init__(self, solver: SolverProtocol):
        self.solver = solver

    async def run(self, auction: AuctionInstance) -> SolverResult:
        """Run the Python solver on an auction."""
        start = time.perf_counter()
        error = None
        response = None

        try:
            response = await self.solver.solve(auction)
        except Exception as e:
            error = str(e)
            logger.exception("python_solver_error", error=error)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract best score from solutions
        best_score = None
        solution_count = 0
        if response:
            solution_count = len(response.solutions)
            for sol in response.solutions:
                if sol.score is not None:
                    score = int(sol.score)
                    if best_score is None or score > best_score:
                        best_score = score

        return SolverResult(
            solver_name="python",
            auction_id=auction.id,
            solution_count=solution_count,
            best_score=best_score,
            elapsed_ms=elapsed_ms,
            error=error,
            response=response,
        )


class BenchmarkHarness:
    """Harness for benchmarking solvers against historical auctions."""

    def __init__(
        self,
        python_solver: SolverProtocol,
        rust_solver_path: str | Path | None = None,
    ):
        self.python_runner = PythonSolverRunner(python_solver)
        self.rust_runner = RustSolverRunner(rust_solver_path) if rust_solver_path else None

    async def benchmark_auction(
        self,
        auction: AuctionInstance,
        auction_name: str = "unknown",
    ) -> BenchmarkResult:
        """Run both solvers on an auction and compare results."""
        logger.info("benchmarking_auction", name=auction_name, order_count=auction.order_count)

        # Run Python solver
        python_result = await self.python_runner.run(auction)

        # Run Rust solver if available
        if self.rust_runner:
            rust_result = await self.rust_runner.run(auction)
        else:
            rust_result = SolverResult(
                solver_name="rust",
                auction_id=auction.id,
                solution_count=0,
                best_score=None,
                elapsed_ms=0,
                error="Rust solver not configured",
            )

        return BenchmarkResult(
            auction_name=auction_name,
            auction_id=auction.id,
            order_count=auction.order_count,
            python_result=python_result,
            rust_result=rust_result,
        )

    async def benchmark_directory(
        self,
        auctions_dir: Path,
        category: str | None = None,
    ) -> BenchmarkSummary:
        """Benchmark all auctions in a directory."""
        search_dir = auctions_dir / category if category else auctions_dir
        if not search_dir.exists():
            logger.warning("auctions_dir_not_found", path=str(search_dir))
            return BenchmarkSummary(total_auctions=0, successful_auctions=0)

        auction_files = list(search_dir.rglob("*.json"))
        logger.info("found_auctions", count=len(auction_files), dir=str(search_dir))

        results: list[BenchmarkResult] = []
        errors = 0

        for path in auction_files:
            try:
                with open(path) as f:
                    data = json.load(f)
                auction = AuctionInstance.model_validate(data)
                name = path.stem
                result = await self.benchmark_auction(auction, name)
                results.append(result)
            except Exception as e:
                logger.error("auction_load_error", path=str(path), error=str(e))
                errors += 1

        return BenchmarkSummary(
            total_auctions=len(auction_files),
            successful_auctions=len(results),
            results=results,
        )


async def run_benchmarks(
    auctions_dir: Path,
    rust_solver_path: Path | None = None,
    category: str | None = None,
) -> BenchmarkSummary:
    """Run benchmarks on historical auctions.

    Args:
        auctions_dir: Directory containing auction JSON files
        rust_solver_path: Path to Rust solver binary (optional)
        category: Subdirectory to filter (e.g., "single_order")

    Returns:
        BenchmarkSummary with results
    """
    # Import here to avoid circular imports
    from solver.api.endpoints import get_solver, solve

    # Create a simple adapter for the solve endpoint
    class SolverAdapter:
        async def solve(self, auction: AuctionInstance) -> SolverResponse:
            # Manually resolve the dependency since we're calling outside FastAPI
            solver_instance = get_solver()
            return await solve("benchmark", "mainnet", auction, solver_instance)

    harness = BenchmarkHarness(
        python_solver=SolverAdapter(),
        rust_solver_path=rust_solver_path,
    )

    return await harness.benchmark_directory(auctions_dir, category)


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    auctions_path = Path(__file__).parent.parent / "tests" / "fixtures" / "auctions"
    summary = asyncio.run(run_benchmarks(auctions_path))
    print(f"Benchmarked {summary.successful_auctions}/{summary.total_auctions} auctions")
    if summary.avg_time_ratio:
        print(f"Average time ratio (Python/Rust): {summary.avg_time_ratio:.2f}x")
    if summary.avg_score_ratio:
        print(f"Average score ratio (Python/Rust): {summary.avg_score_ratio:.2%}")
