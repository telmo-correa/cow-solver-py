"""Subprocess runner for the reference Rust solver."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from solver.models.auction import AuctionInstance
from solver.models.solution import SolverResponse

if TYPE_CHECKING:
    from benchmarks.harness import SolverResult

logger = structlog.get_logger()


class RustSolverRunner:
    """Runner for the reference Rust solver via subprocess.

    The Rust solver can be run in two modes:
    1. As a standalone HTTP server (like the Python solver)
    2. As a CLI tool that reads JSON from stdin

    This runner supports the CLI mode for benchmarking.
    """

    def __init__(
        self,
        solver_path: str | Path | None = None,
        solver_type: str = "baseline",
        timeout_seconds: float = 30.0,
    ):
        """Initialize the Rust solver runner.

        Args:
            solver_path: Path to the solver binary or cargo workspace
            solver_type: Which solver to use ("baseline" or "naive")
            timeout_seconds: Maximum time to wait for solver response
        """
        self.solver_path = Path(solver_path) if solver_path else None
        self.solver_type = solver_type
        self.timeout_seconds = timeout_seconds

    def _build_command(self, auction_file: Path) -> list[str]:
        """Build the command to run the Rust solver.

        The exact command depends on how the solver is built/run.
        This may need adjustment based on the actual Rust solver interface.
        """
        if self.solver_path is None:
            raise ValueError("Rust solver path not configured")

        # If pointing to a cargo workspace, use cargo run
        if (self.solver_path / "Cargo.toml").exists():
            return [
                "cargo",
                "run",
                "--manifest-path",
                str(self.solver_path / "Cargo.toml"),
                "--release",
                "-p",
                "solvers",
                "--",
                "--solver-type",
                self.solver_type,
                "--auction",
                str(auction_file),
            ]

        # If pointing to a binary
        return [
            str(self.solver_path),
            "--solver-type",
            self.solver_type,
            "--auction",
            str(auction_file),
        ]

    async def run(self, auction: AuctionInstance) -> SolverResult:
        """Run the Rust solver on an auction.

        Args:
            auction: The auction to solve

        Returns:
            SolverResult with timing and solution info
        """
        from benchmarks.harness import SolverResult

        if self.solver_path is None:
            return SolverResult(
                solver_name="rust",
                auction_id=auction.id,
                solution_count=0,
                best_score=None,
                elapsed_ms=0,
                error="Rust solver path not configured",
            )

        # Write auction to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Use model_dump with by_alias for proper JSON serialization
            auction_data = auction.model_dump(by_alias=True, mode="json")
            json.dump(auction_data, f)
            auction_file = Path(f.name)

        try:
            cmd = self._build_command(auction_file)
            logger.debug("running_rust_solver", command=" ".join(cmd))

            start = time.perf_counter()

            # Run the solver subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError:
                process.kill()
                return SolverResult(
                    solver_name="rust",
                    auction_id=auction.id,
                    solution_count=0,
                    best_score=None,
                    elapsed_ms=self.timeout_seconds * 1000,
                    error=f"Timeout after {self.timeout_seconds}s",
                )

            elapsed_ms = (time.perf_counter() - start) * 1000

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error("rust_solver_error", error=error_msg)
                return SolverResult(
                    solver_name="rust",
                    auction_id=auction.id,
                    solution_count=0,
                    best_score=None,
                    elapsed_ms=elapsed_ms,
                    error=error_msg,
                )

            # Parse response
            try:
                response_data = json.loads(stdout.decode())
                response = SolverResponse.model_validate(response_data)
            except (json.JSONDecodeError, Exception) as e:
                logger.error("rust_response_parse_error", error=str(e))
                return SolverResult(
                    solver_name="rust",
                    auction_id=auction.id,
                    solution_count=0,
                    best_score=None,
                    elapsed_ms=elapsed_ms,
                    error=f"Failed to parse response: {e}",
                )

            # Extract best score
            best_score = None
            for sol in response.solutions:
                if sol.score is not None:
                    score = int(sol.score)
                    if best_score is None or score > best_score:
                        best_score = score

            return SolverResult(
                solver_name="rust",
                auction_id=auction.id,
                solution_count=len(response.solutions),
                best_score=best_score,
                elapsed_ms=elapsed_ms,
                response=response,
            )

        finally:
            # Clean up temp file
            auction_file.unlink(missing_ok=True)


class RustSolverHTTPRunner:
    """Runner for the Rust solver via HTTP (when running as a server).

    Use this when the Rust solver is running as an HTTP server.

    Note: The auction must already be in the Rust solver's expected format,
    including fields like fullSellAmount, fullBuyAmount, validTo, owner, etc.
    Use fixtures from tests/fixtures/auctions/benchmark/ as examples.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout_seconds: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def run(self, auction: AuctionInstance) -> SolverResult:
        """Run the Rust solver via HTTP request."""
        import httpx

        from benchmarks.harness import SolverResult

        url = f"{self.base_url}/solve"

        start = time.perf_counter()

        # Serialize auction to JSON (must already be in Rust-compatible format)
        auction_json = auction.model_dump(by_alias=True, exclude_none=True, mode="json")

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    url,
                    json=auction_json,
                )
                response.raise_for_status()
                data = response.json()

        except httpx.TimeoutException:
            return SolverResult(
                solver_name="rust",
                auction_id=auction.id,
                solution_count=0,
                best_score=None,
                elapsed_ms=self.timeout_seconds * 1000,
                error=f"Timeout after {self.timeout_seconds}s",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return SolverResult(
                solver_name="rust",
                auction_id=auction.id,
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
                solver_name="rust",
                auction_id=auction.id,
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
            solver_name="rust",
            auction_id=auction.id,
            solution_count=len(solver_response.solutions),
            best_score=best_score,
            elapsed_ms=elapsed_ms,
            response=solver_response,
        )
