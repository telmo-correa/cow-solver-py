"""Metrics calculation for benchmark comparisons."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, median, stdev

from benchmarks.harness import BenchmarkResult


@dataclass
class DistributionStats:
    """Statistical summary of a distribution."""

    count: int
    mean: float
    median: float
    min: float
    max: float
    stdev: float | None  # None if count < 2

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "DistributionStats | None":
        """Compute statistics from a sequence of values."""
        if not values:
            return None

        values = list(values)
        return cls(
            count=len(values),
            mean=mean(values),
            median=median(values),
            min=min(values),
            max=max(values),
            stdev=stdev(values) if len(values) >= 2 else None,
        )


@dataclass
class BenchmarkMetrics:
    """Computed metrics for a benchmark run."""

    total_auctions: int
    successful_both: int  # Both solvers succeeded
    python_only_success: int
    rust_only_success: int
    both_failed: int

    time_ratio_stats: DistributionStats | None
    score_ratio_stats: DistributionStats | None

    python_time_stats: DistributionStats | None
    rust_time_stats: DistributionStats | None

    # Categorized results
    python_faster_count: int
    python_better_score_count: int


def compute_metrics(results: list[BenchmarkResult]) -> BenchmarkMetrics:
    """Compute detailed metrics from benchmark results."""
    if not results:
        return BenchmarkMetrics(
            total_auctions=0,
            successful_both=0,
            python_only_success=0,
            rust_only_success=0,
            both_failed=0,
            time_ratio_stats=None,
            score_ratio_stats=None,
            python_time_stats=None,
            rust_time_stats=None,
            python_faster_count=0,
            python_better_score_count=0,
        )

    successful_both = 0
    python_only = 0
    rust_only = 0
    both_failed = 0

    time_ratios: list[float] = []
    score_ratios: list[float] = []
    python_times: list[float] = []
    rust_times: list[float] = []

    python_faster = 0
    python_better_score = 0

    for r in results:
        py_ok = r.python_result.error is None
        rust_ok = r.rust_result.error is None

        if py_ok and rust_ok:
            successful_both += 1
            python_times.append(r.python_result.elapsed_ms)
            rust_times.append(r.rust_result.elapsed_ms)

            if r.time_ratio is not None:
                time_ratios.append(r.time_ratio)
                if r.time_ratio < 1.0:
                    python_faster += 1

            if r.score_ratio is not None:
                score_ratios.append(r.score_ratio)
                if r.score_ratio > 1.0:
                    python_better_score += 1

        elif py_ok and not rust_ok:
            python_only += 1
            python_times.append(r.python_result.elapsed_ms)
        elif not py_ok and rust_ok:
            rust_only += 1
            rust_times.append(r.rust_result.elapsed_ms)
        else:
            both_failed += 1

    return BenchmarkMetrics(
        total_auctions=len(results),
        successful_both=successful_both,
        python_only_success=python_only,
        rust_only_success=rust_only,
        both_failed=both_failed,
        time_ratio_stats=DistributionStats.from_values(time_ratios),
        score_ratio_stats=DistributionStats.from_values(score_ratios),
        python_time_stats=DistributionStats.from_values(python_times),
        rust_time_stats=DistributionStats.from_values(rust_times),
        python_faster_count=python_faster,
        python_better_score_count=python_better_score,
    )
