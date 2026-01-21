"""Base protocol for solution strategies."""

from typing import Protocol

from solver.models.auction import AuctionInstance
from solver.models.solution import Solution


class SolutionStrategy(Protocol):
    """Protocol for solution strategies.

    Each strategy attempts to solve an auction. If the strategy can produce
    a valid solution, it returns it. Otherwise, it returns None to indicate
    the next strategy should be tried.

    Strategies are tried in priority order (e.g., CoW matching before AMM routing)
    since CoW matches are better for users (no AMM fees).
    """

    def try_solve(self, auction: AuctionInstance) -> Solution | None:
        """Attempt to solve the auction.

        Args:
            auction: The auction to solve

        Returns:
            A Solution if this strategy can handle the auction, None otherwise
        """
        ...
