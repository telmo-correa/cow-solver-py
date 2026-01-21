"""API endpoints for the CoW solver."""

import structlog
from fastapi import APIRouter, Depends

from solver.models.auction import AuctionInstance
from solver.models.solution import SolverResponse
from solver.routing.router import Solver, solver

logger = structlog.get_logger()

router = APIRouter()

# Networks that this solver supports (has liquidity data for)
SUPPORTED_NETWORKS = {"mainnet"}

# Valid environment values
VALID_ENVIRONMENTS = {"production", "staging", "shadow", "local"}


def get_solver() -> Solver:
    """Dependency provider for the solver instance.

    Override this in tests to inject a mock solver:
        app.dependency_overrides[get_solver] = lambda: mock_solver

    Returns:
        The solver instance to use for solving auctions.
    """
    return solver


@router.post("/{environment}/{network}")
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
        SolverResponse containing proposed solutions
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

    # Use the solver to find solutions
    response = solver_instance.solve(auction)

    logger.info(
        "returning_solutions",
        auction_id=auction.id,
        solution_count=len(response.solutions),
        has_trades=any(len(s.trades) > 0 for s in response.solutions),
    )

    return response
