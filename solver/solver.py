"""Main solver that orchestrates solution strategies.

The Solver class is the entry point for solving CoW Protocol auctions.
It composes multiple strategies (CoW matching, AMM routing) to find
optimal solutions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from solver.amm.balancer import BalancerStableAMM, BalancerWeightedAMM
    from solver.amm.limit_order import LimitOrderAMM
    from solver.amm.uniswap_v3 import UniswapV3AMM
    from solver.strategies.base import SolutionStrategy, StrategyResult

from solver.amm.uniswap_v2 import UniswapV2
from solver.ebbo import EBBO_TOLERANCE, EBBOPrices, EBBOValidator
from solver.models.auction import AuctionInstance, Order
from solver.models.solution import Interaction, Solution, SolverResponse
from solver.models.types import normalize_address
from solver.routing.router import SingleOrderRouter

logger = structlog.get_logger()


class Solver:
    """Main solver that processes auction instances using a strategy pattern.

    The solver tries each strategy in order, composing partial results.
    If a strategy returns a result with unfilled remainders, subsequent
    strategies are tried on the remainder orders.

    Default strategies (in priority order):
    1. MultiPairCowStrategy - N-order CoW matching with joint price optimization
    2. AmmRoutingStrategy - Route through liquidity pools

    Args:
        strategies: List of strategies to try in order. If None, uses defaults.
        router: Deprecated. Use AmmRoutingStrategy with injected router instead.
        amm: Deprecated. Use AmmRoutingStrategy with injected AMM instead.
    """

    def __init__(
        self,
        strategies: list[SolutionStrategy] | None = None,
        router: SingleOrderRouter | None = None,
        amm: UniswapV2 | None = None,
        v3_amm: UniswapV3AMM | None = None,
        weighted_amm: BalancerWeightedAMM | None = None,
        stable_amm: BalancerStableAMM | None = None,
        limit_order_amm: LimitOrderAMM | None = None,
        ebbo_prices: EBBOPrices | None = None,
    ) -> None:
        """Initialize the solver with strategies.

        Args:
            strategies: List of SolutionStrategy instances to try in order.
                       If None, uses [CowMatchStrategy(), HybridCowStrategy(),
                       AmmRoutingStrategy()].
            router: Deprecated. For backwards compatibility, if provided,
                    creates AmmRoutingStrategy with this router.
            amm: Deprecated. For backwards compatibility, if provided,
                 creates AmmRoutingStrategy with this AMM.
            v3_amm: UniswapV3 AMM for V3 pool routing. If None, V3 pools are skipped.
            weighted_amm: Balancer weighted AMM. If None, weighted pools are skipped.
            stable_amm: Balancer stable AMM. If None, stable pools are skipped.
            limit_order_amm: 0x limit order AMM. If None, limit orders are skipped.
            ebbo_prices: Precomputed EBBO prices for validation. If provided,
                        strategy results are filtered to remove EBBO violations.
        """
        self.ebbo_validator = (
            EBBOValidator(ebbo_prices=ebbo_prices, tolerance=EBBO_TOLERANCE)
            if ebbo_prices is not None
            else None
        )

        if strategies is not None:
            self.strategies = strategies
        elif (
            router is not None
            or amm is not None
            or v3_amm is not None
            or weighted_amm is not None
            or stable_amm is not None
            or limit_order_amm is not None
        ):
            # Backwards compatibility: create strategies from legacy params
            from solver.strategies import (
                AmmRoutingStrategy,
                MultiPairCowStrategy,
            )

            self.strategies = [
                # MultiPairCowStrategy handles N-order CoW matching with multi-pair
                # price coordination using LP optimization. It finds connected
                # components of overlapping token pairs and optimizes prices jointly.
                MultiPairCowStrategy(
                    amm=amm,
                    router=router,
                    v3_amm=v3_amm,
                    weighted_amm=weighted_amm,
                    stable_amm=stable_amm,
                    limit_order_amm=limit_order_amm,
                ),
                AmmRoutingStrategy(
                    amm=amm,
                    router=router,
                    v3_amm=v3_amm,
                    weighted_amm=weighted_amm,
                    stable_amm=stable_amm,
                    limit_order_amm=limit_order_amm,
                ),
            ]
        else:
            # Default strategies
            from solver.strategies import (
                AmmRoutingStrategy,
                MultiPairCowStrategy,
            )

            self.strategies = [
                # MultiPairCowStrategy handles N-order CoW matching with multi-pair
                # price coordination using LP optimization. It finds connected
                # components of overlapping token pairs and optimizes prices jointly.
                MultiPairCowStrategy(),
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

        Solution structure (for Rust parity):
        - CoW matches: Combined into one solution (orders trade with each other)
        - AMM routes: Separate solutions (one per order, like Rust baseline)

        See CLAUDE.md section "Solution Structure: One Solution Per Order vs Combined
        Solutions" for detailed documentation on this architectural decision.

        Args:
            auction: The auction to solve

        Returns:
            SolverResponse with proposed solutions
        """
        if auction.order_count == 0:
            return SolverResponse.empty()

        # Collect results from strategies
        results: list[StrategyResult] = []
        current_auction = auction

        for strategy in self.strategies:
            if current_auction.order_count == 0:
                break

            result = strategy.try_solve(current_auction)

            # Apply EBBO filtering if validator is configured.
            # This is a mandatory safety net when enabled - violations are logged as
            # errors since they indicate bugs in strategy-level EBBO validation.
            if result is not None and result.has_fills and self.ebbo_validator is not None:
                result = self._filter_ebbo_violations(
                    result, current_auction, self.ebbo_validator, type(strategy).__name__
                )

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

        # Build solutions from results
        # For Rust parity: AMM-routed orders with MULTIPLE fills get separate solutions,
        # single-order results and CoW matches are combined
        solutions: list[Solution] = []

        for result in results:
            if self._should_combine_fills(result):
                # Single order or CoW match: all fills go in one solution
                solution = result.build_solution(solution_id=len(solutions), auction=auction)
                solutions.append(solution)
            else:
                # Multiple AMM-routed orders: each fill gets its own solution (Rust parity)
                solutions.extend(self._build_separate_solutions(result, len(solutions), auction))

        return SolverResponse(solutions=solutions)

    def _should_combine_fills(self, result: StrategyResult) -> bool:
        """Check if fills should be combined into one solution.

        Combine fills when:
        - Single fill (single order, no splitting needed)
        - CoW match (no AMM interactions, trades between users)

        Separate fills when:
        - Multiple fills AND each has AMM interactions (independent AMM routes)

        This matches Rust baseline behavior: one solution per order for AMM routing,
        combined solutions for CoW matches.

        Args:
            result: The strategy result to check

        Returns:
            True if fills should be combined, False if they should be separate
        """
        # Single fill: nothing to separate
        if len(result.fills) <= 1:
            return True

        # CoW matches have no interactions (direct peer-to-peer trading)
        # Multiple fills with interactions should be separated (Rust parity)
        return len(result.interactions) == 0

    def _build_separate_solutions(
        self,
        result: StrategyResult,
        start_id: int,
        auction: AuctionInstance,
    ) -> list[Solution]:
        """Build separate solutions for each fill in an AMM routing result.

        For Rust parity, each order routed through AMMs gets its own solution
        with its own clearing prices and interactions.

        Args:
            result: The strategy result with multiple fills
            start_id: Starting solution ID
            auction: The auction instance (for fee calculation)

        Returns:
            List of solutions, one per fill
        """
        from solver.models.types import normalize_address
        from solver.strategies.base import StrategyResult

        solutions: list[Solution] = []

        # Pre-index interactions by token pair for O(1) lookup instead of O(n²)
        # Key: (input_token, output_token) -> list of (index, interaction)
        interaction_index: dict[tuple[str, str], list[tuple[int, Interaction]]] = {}
        for j, interaction in enumerate(result.interactions):
            if hasattr(interaction, "input_token") and hasattr(interaction, "output_token"):
                key = (
                    normalize_address(interaction.input_token),
                    normalize_address(interaction.output_token),
                )
                if key not in interaction_index:
                    interaction_index[key] = []
                interaction_index[key].append((j, interaction))

        # Track which interactions have been assigned to avoid duplicates
        assigned_interactions: set[int] = set()

        # Group interactions by order using the index
        for i, fill in enumerate(result.fills):
            order = fill.order

            # Find interactions for this order using pre-built index
            order_interactions = []
            key = (normalize_address(order.sell_token), normalize_address(order.buy_token))
            for j, interaction in interaction_index.get(key, []):
                if j not in assigned_interactions:
                    order_interactions.append(interaction)
                    assigned_interactions.add(j)
                    break  # One interaction per order for single-hop

            # Calculate gas for this order's interactions
            order_gas = 0
            if order_interactions:
                # Use proportional gas from the result
                order_gas = result.gas // len(result.fills) if result.fills else 0

            # Calculate prices from this fill's amounts
            # CoW Protocol pricing: price[buy_token] = amount_in, price[sell_token] = amount_out
            # This matches the Rust solver's per-order price calculation
            sell_token_normalized = normalize_address(order.sell_token)
            buy_token_normalized = normalize_address(order.buy_token)
            order_prices = {
                buy_token_normalized: str(fill.sell_filled),  # price[buy] = input amount
                sell_token_normalized: str(fill.buy_filled),  # price[sell] = output amount
            }

            # Build a single-fill result for this order
            single_result = StrategyResult(
                fills=[fill],
                interactions=order_interactions,
                prices=order_prices,  # Use per-order prices
                gas=order_gas,
                remainder_orders=[],
                fee_calculator=result.fee_calculator,
            )

            solution = single_result.build_solution(
                solution_id=start_id + i,
                auction=auction,
            )
            solutions.append(solution)

        return solutions

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

    def _filter_ebbo_violations(
        self,
        result: StrategyResult,
        auction: AuctionInstance,
        ebbo_validator: EBBOValidator | None = None,
        strategy_name: str = "unknown",
    ) -> StrategyResult | None:
        """Filter out fills that violate EBBO constraints.

        EBBO (Ethereum Best Bid/Offer) requires that clearing prices are at
        least as good as what users could get from base protocols.

        For CoW matches (no interactions), we check that each filled order's
        clearing rate is at least as good as the EBBO rate for that pair.

        This is a mandatory safety net. If violations are caught here, it
        indicates a bug in the strategy's EBBO checking logic and an error
        is logged.

        Args:
            result: Strategy result to filter
            auction: Auction context for token info
            ebbo_validator: Validator to use (if None, uses self.ebbo_validator)
            strategy_name: Name of the strategy that produced this result

        Returns:
            Filtered result with EBBO-compliant fills only, or None if all violate
        """
        from solver.strategies.base import StrategyResult

        # Use provided validator or fall back to instance validator
        validator = ebbo_validator if ebbo_validator is not None else self.ebbo_validator
        if validator is None:
            return result

        # Build clearing prices dict
        clearing_prices = {normalize_address(k): int(v) for k, v in result.prices.items()}

        # Check each fill for EBBO compliance
        valid_fills = []
        violation_count = 0

        for fill in result.fills:
            order = fill.order
            violations = validator.check_clearing_prices(
                clearing_prices,
                [order],
                auction,
            )

            if violations:
                # This fill violates EBBO - reject it
                # Log as ERROR because this indicates a bug in the strategy
                violation_count += 1
                for v in violations:
                    logger.error(
                        "ebbo_violation_caught_by_solver",
                        strategy=strategy_name,
                        order_uid=order.uid[:18] + "...",
                        sell_token=order.sell_token[-8:],
                        buy_token=order.buy_token[-8:],
                        clearing_rate=str(v.clearing_rate),
                        ebbo_rate=str(v.ebbo_rate),
                        deficit_pct=f"{v.deficit_pct:.2f}%",
                        message="Strategy produced EBBO violation - this indicates a bug in strategy EBBO validation",
                    )
            else:
                valid_fills.append(fill)

        if not valid_fills:
            logger.warning(
                "ebbo_all_fills_rejected_by_solver",
                strategy=strategy_name,
                total_fills=len(result.fills),
                violations=violation_count,
                message="All fills from strategy violated EBBO - strategy EBBO validation is broken",
            )
            return None

        if violation_count > 0:
            logger.warning(
                "ebbo_partial_rejection_by_solver",
                strategy=strategy_name,
                original=len(result.fills),
                valid=len(valid_fills),
                rejected=violation_count,
                message="Some fills violated EBBO - strategy EBBO validation is incomplete",
            )

        # Rebuild the result with valid fills only
        # Keep only interactions that correspond to valid fills
        valid_orders = {f.order.uid for f in valid_fills}
        valid_interactions = []
        for interaction in result.interactions:
            # For AMM interactions, we keep all since they may be shared
            # For CoW matches, there are no interactions
            valid_interactions.append(interaction)

        # Recalculate remainder orders - add rejected orders back
        rejected_orders = [f.order for f in result.fills if f.order.uid not in valid_orders]
        new_remainders = list(result.remainder_orders) + rejected_orders

        return StrategyResult(
            fills=valid_fills,
            interactions=valid_interactions,
            prices=result.prices,
            gas=result.gas,
            remainder_orders=new_remainders,
            fee_calculator=result.fee_calculator,
        )


# Singleton solver instance with optional V3 support via environment
def _create_default_solver() -> Solver:
    """Create the default solver with Balancer, limit orders, and optional V3.

    V3 support is enabled if RPC_URL environment variable is set.
    This allows the solver to get quotes from UniswapV3 QuoterV2 contract.

    Balancer support (weighted and stable pools) and 0x limit order support
    are always enabled since the math is computed locally without RPC calls.

    Returns:
        Configured Solver instance
    """
    import os

    from solver.amm.balancer import BalancerStableAMM, BalancerWeightedAMM
    from solver.amm.limit_order import LimitOrderAMM

    # Local AMMs are always enabled (no RPC required)
    weighted_amm = BalancerWeightedAMM()
    stable_amm = BalancerStableAMM()
    limit_order_amm = LimitOrderAMM()

    rpc_url = os.environ.get("RPC_URL")
    if rpc_url:
        # V3 support enabled
        from solver.amm.uniswap_v3 import UniswapV3AMM, Web3UniswapV3Quoter

        logger.info("v3_support_enabled", rpc_url=rpc_url[:50] + "...")
        quoter = Web3UniswapV3Quoter(rpc_url)
        v3_amm = UniswapV3AMM(quoter=quoter)
        return Solver(
            v3_amm=v3_amm,
            weighted_amm=weighted_amm,
            stable_amm=stable_amm,
            limit_order_amm=limit_order_amm,
        )
    else:
        logger.info("v3_support_disabled", reason="RPC_URL not set")
        return Solver(
            weighted_amm=weighted_amm,
            stable_amm=stable_amm,
            limit_order_amm=limit_order_amm,
        )


solver = _create_default_solver()
