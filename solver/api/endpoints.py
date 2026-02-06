"""API endpoints for the CoW solver."""

import asyncio
import os
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends

from solver.models.auction import AuctionInstance
from solver.models.solution import SolverResponse
from solver.solver import Solver, get_default_solver

logger = structlog.get_logger()

router = APIRouter()

# Networks that this solver supports (has liquidity data for)
# Configurable via environment variable COW_SUPPORTED_NETWORKS (comma-separated)
SUPPORTED_NETWORKS = set(os.environ.get("COW_SUPPORTED_NETWORKS", "mainnet").split(","))

# Valid environment values
VALID_ENVIRONMENTS = {"production", "staging", "shadow", "local"}


def get_solver() -> Solver:
    """Dependency provider for the solver instance.

    Override this in tests to inject a mock solver:
        app.dependency_overrides[get_solver] = lambda: mock_solver

    Returns:
        The solver instance to use for solving auctions.
    """
    return get_default_solver()


@router.post("/{environment}/{network}", response_model_exclude_none=True)
async def solve(
    environment: str,
    network: str,
    auction: AuctionInstance,
    solver_instance: Solver = Depends(get_solver),
) -> SolverResponse:
    """Solve an auction batch.

    This is the main entry point called by the CoW driver.

    Args:
        environment: Environment name (e.g., "production", "staging", "shadow")
        network: Network name (e.g., "mainnet", "arbitrum-one", "xdai")
        auction: The auction instance to solve
        solver_instance: Injected solver (via FastAPI Depends)

    Returns:
        SolverResponse containing proposed solutions.
        Uses `response_model_exclude_none=True` to omit None fields from JSON,
        reducing payload size and matching CoW Protocol API expectations.

    Error Handling:
        - Invalid request schema: Returns 422 Validation Error (Pydantic)
        - Unsupported network: Returns empty solutions (graceful degradation)
        - Solver exception: Logs error, returns empty solutions
    """
    logger.info(
        "received_auction",
        environment=environment,
        network=network,
        auction_id=auction.id,
        order_count=auction.order_count,
        token_pairs=len(auction.token_pairs),
    )

    # Validate environment (log warning but don't fail)
    if environment not in VALID_ENVIRONMENTS:
        logger.warning(
            "unknown_environment",
            environment=environment,
            valid_environments=list(VALID_ENVIRONMENTS),
        )

    # Check if network is supported
    if network not in SUPPORTED_NETWORKS:
        logger.warning(
            "unsupported_network",
            network=network,
            supported_networks=list(SUPPORTED_NETWORKS),
            message="Returning empty solution - no liquidity data for this network",
        )
        return SolverResponse.empty()

    # Calculate timeout from auction deadline
    timeout_seconds: float | None = None
    if auction.deadline is not None:
        now = datetime.now(UTC)
        remaining = (auction.deadline - now).total_seconds()
        if remaining <= 0:
            logger.warning(
                "auction_deadline_passed",
                auction_id=auction.id,
                deadline=str(auction.deadline),
            )
            return SolverResponse.empty()
        # Leave 0.5s buffer for response serialization
        timeout_seconds = max(remaining - 0.5, 0.1)

    # Use the solver to find solutions
    # Wrap in try-except to prevent uncaught exceptions from crashing the server
    try:
        loop = asyncio.get_event_loop()
        if timeout_seconds is not None:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, solver_instance.solve, auction),
                timeout=timeout_seconds,
            )
        else:
            response = await loop.run_in_executor(None, solver_instance.solve, auction)
    except TimeoutError:
        logger.warning(
            "solver_timeout",
            auction_id=auction.id,
            timeout_seconds=timeout_seconds,
            message="Solver exceeded deadline, returning empty solution",
        )
        return SolverResponse.empty()
    except Exception:
        # Log error with full traceback for debugging
        logger.exception(
            "solver_error",
            auction_id=auction.id,
            order_count=auction.order_count,
            message="Solver raised an exception, returning empty solution",
        )
        # Return empty response rather than 500 error
        # This allows the driver to continue and try other solvers
        return SolverResponse.empty()

    logger.info(
        "returning_solutions",
        auction_id=auction.id,
        solution_count=len(response.solutions),
        has_trades=any(len(s.trades) > 0 for s in response.solutions),
    )

    return response
