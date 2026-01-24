"""Base class for AMM-backed strategies.

This module provides a common base class for strategies that need router
access for AMM price queries and routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from solver.amm.uniswap_v2 import UniswapV2, uniswap_v2
from solver.fees import DefaultFeeCalculator, PoolBasedPriceEstimator
from solver.pools import PoolRegistry
from solver.routing.router import SingleOrderRouter

if TYPE_CHECKING:
    from solver.amm.balancer import BalancerStableAMM, BalancerWeightedAMM
    from solver.amm.limit_order import LimitOrderAMM
    from solver.amm.uniswap_v3 import UniswapV3AMM


class AMMBackedStrategy:
    """Base class for strategies that need AMM router access.

    Provides common router initialization and management shared by:
    - AmmRoutingStrategy
    - MultiPairCowStrategy
    - HybridCowStrategy

    Subclasses get:
    - Consistent AMM component injection for testing
    - Shared _get_router() method
    - Fee calculator creation with pool-based price estimation

    Args:
        amm: AMM implementation for swap math. Defaults to UniswapV2.
        router: Injected router for testing. If provided, used directly
                instead of creating one from auction liquidity.
        v3_amm: UniswapV3 AMM for V3 pool routing. If None, V3 pools are skipped.
        weighted_amm: Balancer weighted AMM. If None, weighted pools are skipped.
        stable_amm: Balancer stable AMM. If None, stable pools are skipped.
        limit_order_amm: 0x limit order AMM. If None, limit orders are skipped.
    """

    def __init__(
        self,
        amm: UniswapV2 | None = None,
        router: SingleOrderRouter | None = None,
        v3_amm: UniswapV3AMM | None = None,
        weighted_amm: BalancerWeightedAMM | None = None,
        stable_amm: BalancerStableAMM | None = None,
        limit_order_amm: LimitOrderAMM | None = None,
    ) -> None:
        """Initialize with optional AMM components."""
        self.amm = amm if amm is not None else uniswap_v2
        self._injected_router = router
        self.v3_amm = v3_amm
        self.weighted_amm = weighted_amm
        self.stable_amm = stable_amm
        self.limit_order_amm = limit_order_amm

    def _get_router(self, pool_registry: PoolRegistry) -> SingleOrderRouter:
        """Get the router to use for AMM operations.

        Returns the injected router if available, otherwise creates a new one
        from the pool registry.

        Args:
            pool_registry: Registry of available liquidity pools

        Returns:
            Router instance configured with available AMM components
        """
        if self._injected_router is not None:
            return self._injected_router
        return SingleOrderRouter(
            amm=self.amm,
            pool_registry=pool_registry,
            v3_amm=self.v3_amm,
            weighted_amm=self.weighted_amm,
            stable_amm=self.stable_amm,
            limit_order_amm=self.limit_order_amm,
        )

    def _create_fee_calculator(self, router: SingleOrderRouter) -> DefaultFeeCalculator:
        """Create a fee calculator with pool-based price estimation.

        This enables fee calculation for limit orders when reference prices
        are missing, by estimating prices through pool routing to native token.

        Args:
            router: Router for simulating price estimation swaps

        Returns:
            Fee calculator with price estimator configured
        """
        price_estimator = PoolBasedPriceEstimator(router=router)
        return DefaultFeeCalculator(price_estimator=price_estimator)


__all__ = ["AMMBackedStrategy"]
