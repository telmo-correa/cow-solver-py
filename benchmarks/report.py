"""Report generation for benchmark results."""

import json
from datetime import datetime
from pathlib import Path

from benchmarks.harness import BenchmarkResult, BenchmarkSummary
from benchmarks.metrics import BenchmarkMetrics, compute_metrics


def format_markdown_report(summary: BenchmarkSummary, metrics: BenchmarkMetrics | None = None) -> str:
    """Generate a markdown report from benchmark results."""
    if metrics is None:
        metrics = compute_metrics(summary.results)

    lines = [
        "# Benchmark Report",
        "",
        f"**Date:** {datetime.now().isoformat()}",
        f"**Total Auctions:** {summary.total_auctions}",
        f"**Successful:** {summary.successful_auctions}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Both solvers succeeded | {metrics.successful_both} |",
        f"| Python only succeeded | {metrics.python_only_success} |",
        f"| Rust only succeeded | {metrics.rust_only_success} |",
        f"| Both failed | {metrics.both_failed} |",
        "",
    ]

    # Time comparison
    if metrics.time_ratio_stats:
        ts = metrics.time_ratio_stats
        lines.extend([
            "## Time Comparison (Python / Rust)",
            "",
            "| Statistic | Value |",
            "|-----------|-------|",
            f"| Mean | {ts.mean:.2f}x |",
            f"| Median | {ts.median:.2f}x |",
            f"| Min | {ts.min:.2f}x |",
            f"| Max | {ts.max:.2f}x |",
            f"| Std Dev | {ts.stdev:.2f}x |" if ts.stdev else "",
            "",
            f"Python was faster in {metrics.python_faster_count}/{ts.count} auctions "
            f"({100*metrics.python_faster_count/ts.count:.1f}%)",
            "",
        ])

    # Score comparison
    if metrics.score_ratio_stats:
        ss = metrics.score_ratio_stats
        lines.extend([
            "## Score Comparison (Python / Rust)",
            "",
            "| Statistic | Value |",
            "|-----------|-------|",
            f"| Mean | {ss.mean:.2%} |",
            f"| Median | {ss.median:.2%} |",
            f"| Min | {ss.min:.2%} |",
            f"| Max | {ss.max:.2%} |",
            f"| Std Dev | {ss.stdev:.2%} |" if ss.stdev else "",
            "",
            f"Python found better score in {metrics.python_better_score_count}/{ss.count} auctions "
            f"({100*metrics.python_better_score_count/ss.count:.1f}%)",
            "",
        ])

    # Individual results table
    if summary.results:
        lines.extend([
            "## Individual Results",
            "",
            "| Auction | Orders | Python (ms) | Rust (ms) | Time Ratio | Score Ratio |",
            "|---------|--------|-------------|-----------|------------|-------------|",
        ])

        for r in summary.results:
            time_ratio = f"{r.time_ratio:.2f}x" if r.time_ratio else "N/A"
            score_ratio = f"{r.score_ratio:.2%}" if r.score_ratio else "N/A"
            lines.append(
                f"| {r.auction_name} | {r.order_count} | "
                f"{r.python_result.elapsed_ms:.1f} | {r.rust_result.elapsed_ms:.1f} | "
                f"{time_ratio} | {score_ratio} |"
            )

    return "\n".join(lines)


def format_json_report(summary: BenchmarkSummary, metrics: BenchmarkMetrics | None = None) -> str:
    """Generate a JSON report from benchmark results."""
    if metrics is None:
        metrics = compute_metrics(summary.results)

    def result_to_dict(r: BenchmarkResult) -> dict:
        return {
            "auction_name": r.auction_name,
            "auction_id": r.auction_id,
            "order_count": r.order_count,
            "python": {
                "elapsed_ms": r.python_result.elapsed_ms,
                "solution_count": r.python_result.solution_count,
                "best_score": r.python_result.best_score,
                "error": r.python_result.error,
            },
            "rust": {
                "elapsed_ms": r.rust_result.elapsed_ms,
                "solution_count": r.rust_result.solution_count,
                "best_score": r.rust_result.best_score,
                "error": r.rust_result.error,
            },
            "time_ratio": r.time_ratio,
            "score_ratio": r.score_ratio,
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_auctions": summary.total_auctions,
            "successful_auctions": summary.successful_auctions,
            "avg_time_ratio": summary.avg_time_ratio,
            "avg_score_ratio": summary.avg_score_ratio,
        },
        "metrics": {
            "successful_both": metrics.successful_both,
            "python_only_success": metrics.python_only_success,
            "rust_only_success": metrics.rust_only_success,
            "both_failed": metrics.both_failed,
            "python_faster_count": metrics.python_faster_count,
            "python_better_score_count": metrics.python_better_score_count,
        },
        "results": [result_to_dict(r) for r in summary.results],
    }

    return json.dumps(report, indent=2)


def save_report(
    summary: BenchmarkSummary,
    output_dir: Path,
    formats: list[str] | None = None,
) -> list[Path]:
    """Save benchmark report to files.

    Args:
        summary: Benchmark summary to report
        output_dir: Directory to save reports
        formats: List of formats ("markdown", "json"). Defaults to both.

    Returns:
        List of paths to saved report files
    """
    if formats is None:
        formats = ["markdown", "json"]

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(summary.results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: list[Path] = []

    if "markdown" in formats:
        md_path = output_dir / f"benchmark_{timestamp}.md"
        md_path.write_text(format_markdown_report(summary, metrics))
        paths.append(md_path)

    if "json" in formats:
        json_path = output_dir / f"benchmark_{timestamp}.json"
        json_path.write_text(format_json_report(summary, metrics))
        paths.append(json_path)

    return paths
