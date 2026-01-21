"""AMM routing strategy - routes orders through liquidity pools."""

import structlog

from solver.amm.uniswap_v2 import (
    PoolRegistry,
    UniswapV2,
    build_registry_from_liquidity,
    uniswap_v2,
)
from solver.models.auction import AuctionInstance
from solver.models.solution import Solution
from solver.routing.router import SingleOrderRouter

logger = structlog.get_logger()


class AmmRoutingStrategy:
    """Strategy that routes orders through AMM liquidity pools.

    This strategy handles:
    - Single-order auctions (direct and multi-hop routing)
    - Both sell orders (exact input) and buy orders (exact output)

    It builds a PoolRegistry from the auction's liquidity data and uses
    the SingleOrderRouter to find routes through available pools.
    """

    def __init__(
        self,
        amm: UniswapV2 | None = None,
        router: SingleOrderRouter | None = None,
    ) -> None:
        """Initialize the AMM routing strategy.

        Args:
            amm: AMM implementation for swap math. Defaults to UniswapV2.
            router: Injected router for testing. If provided, used directly
                    instead of creating one from auction liquidity.
        """
        self.amm = amm if amm is not None else uniswap_v2
        self._injected_router = router

    def try_solve(self, auction: AuctionInstance) -> Solution | None:
        """Try to route orders through AMM pools.

        Currently only handles single-order auctions.

        Args:
            auction: The auction to solve

        Returns:
            A Solution if routing succeeds, None otherwise
        """
        if auction.order_count == 0:
            return None

        # Build pool registry from auction liquidity
        pool_registry = build_registry_from_liquidity(auction.liquidity)

        logger.debug(
            "amm_strategy_pool_registry",
            pool_count=pool_registry.pool_count,
            liquidity_count=len(auction.liquidity),
        )

        # Currently only handle single-order auctions
        if auction.order_count == 1:
            return self._solve_single_order(auction, pool_registry)

        # Multi-order AMM routing not yet supported
        return None

    def _solve_single_order(
        self, auction: AuctionInstance, pool_registry: PoolRegistry
    ) -> Solution | None:
        """Route a single order through AMM pools."""
        order = auction.orders[0]

        # Log if order is partially fillable
        if order.partially_fillable:
            logger.info(
                "partially_fillable_order",
                order_uid=order.uid,
                message="Partially fillable orders are executed fully or not at all",
            )

        # Create router with auction's liquidity
        # Use injected router if provided (for testing)
        if self._injected_router is not None:
            router = self._injected_router
        else:
            router = SingleOrderRouter(amm=self.amm, pool_registry=pool_registry)

        # Route the order
        routing_result = router.route_order(order)
        if not routing_result.success:
            logger.debug(
                "amm_routing_failed",
                order_uid=order.uid,
                error=routing_result.error,
            )
            return None

        # Build solution
        solution = router.build_solution(routing_result, solution_id=0)
        return solution
