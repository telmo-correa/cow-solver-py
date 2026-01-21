"""Main solver that orchestrates solution strategies.

The Solver class is the entry point for solving CoW Protocol auctions.
It composes multiple strategies (CoW matching, AMM routing) to find
optimal solutions.
"""

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from solver.strategies.base import SolutionStrategy

from solver.amm.uniswap_v2 import UniswapV2
from solver.models.auction import AuctionInstance, Order
from solver.models.solution import SolverResponse
from solver.routing.router import SingleOrderRouter

logger = structlog.get_logger()


class Solver:
    """Main solver that processes auction instances using a strategy pattern.

    The solver tries each strategy in order, composing partial results.
    If a strategy returns a result with unfilled remainders, subsequent
    strategies are tried on the remainder orders.

    Default strategies (in priority order):
    1. CowMatchStrategy - Direct peer-to-peer matching (no AMM fees)
    2. AmmRoutingStrategy - Route through liquidity pools

    Args:
        strategies: List of strategies to try in order. If None, uses defaults.
        router: Deprecated. Use AmmRoutingStrategy with injected router instead.
        amm: Deprecated. Use AmmRoutingStrategy with injected AMM instead.
    """

    def __init__(
        self,
        strategies: "list[SolutionStrategy] | None" = None,
        router: SingleOrderRouter | None = None,
        amm: UniswapV2 | None = None,
    ) -> None:
        """Initialize the solver with strategies.

        Args:
            strategies: List of SolutionStrategy instances to try in order.
                       If None, uses [CowMatchStrategy(), AmmRoutingStrategy()].
            router: Deprecated. For backwards compatibility, if provided,
                    creates AmmRoutingStrategy with this router.
            amm: Deprecated. For backwards compatibility, if provided,
                 creates AmmRoutingStrategy with this AMM.
        """
        if strategies is not None:
            self.strategies = strategies
        elif router is not None or amm is not None:
            # Backwards compatibility: create strategies from legacy params
            from solver.strategies import AmmRoutingStrategy, CowMatchStrategy

            self.strategies = [
                CowMatchStrategy(),
                AmmRoutingStrategy(amm=amm, router=router),
            ]
        else:
            # Default strategies
            from solver.strategies import AmmRoutingStrategy, CowMatchStrategy

            self.strategies = [
                CowMatchStrategy(),
                AmmRoutingStrategy(),
            ]

    def solve(self, auction: AuctionInstance) -> SolverResponse:
        """Solve an auction batch by composing strategy results.

        Strategies are tried in priority order. Each strategy may return
        a partial result with remainder orders. Subsequent strategies
        are tried on the remainders until all orders are filled or no
        strategy can make progress.

        This enables composition: e.g., CoW matching fills part of orders,
        then AMM routing fills the remainder.

        Args:
            auction: The auction to solve

        Returns:
            SolverResponse with proposed solutions
        """
        from solver.strategies.base import StrategyResult

        if auction.order_count == 0:
            return SolverResponse.empty()

        # Collect results from strategies
        results: list[StrategyResult] = []
        current_auction = auction

        for strategy in self.strategies:
            if current_auction.order_count == 0:
                break

            result = strategy.try_solve(current_auction)
            if result is not None and result.has_fills:
                logger.info(
                    "strategy_succeeded",
                    strategy=type(strategy).__name__,
                    order_count=current_auction.order_count,
                    fills=len(result.fills),
                    remainders=len(result.remainder_orders),
                )
                results.append(result)

                # If there are remainders, create a sub-auction for next strategy
                if result.remainder_orders:
                    current_auction = self._create_sub_auction(
                        current_auction, result.remainder_orders
                    )
                else:
                    # All orders filled, we're done
                    break

        if not results:
            logger.debug(
                "no_strategy_found_solution",
                order_count=auction.order_count,
                strategies_tried=[type(s).__name__ for s in self.strategies],
            )
            return SolverResponse.empty()

        # Combine all results into final solution
        combined = StrategyResult.combine(results)
        solution = combined.build_solution(solution_id=0)

        return SolverResponse(solutions=[solution])

    def _create_sub_auction(
        self, original: AuctionInstance, orders: list[Order]
    ) -> AuctionInstance:
        """Create a sub-auction with the given orders.

        Preserves tokens and liquidity from the original auction.

        Args:
            original: The original auction
            orders: Orders to include in the sub-auction

        Returns:
            A new AuctionInstance with the specified orders
        """
        logger.debug(
            "sub_auction_created",
            original_order_count=original.order_count,
            remainder_order_count=len(orders),
            order_uids=[o.uid[:18] + "..." for o in orders],
        )
        return original.model_copy(update={"orders": orders})


# Singleton solver instance
solver = Solver()
