"""Order routing and solution building.

This module handles routing orders through AMM pools and building solutions
for the CoW Protocol settlement.

Current limitations:
- Only single-order auctions supported
- Only sell orders supported (buy orders in Slice 1.3)
- Partially fillable orders are executed fully or not at all
- Multi-hop routing not yet implemented (Slice 1.4)
"""

from dataclasses import dataclass

import structlog

from solver.amm.uniswap_v2 import UniswapV2Pool, get_pool, uniswap_v2
from solver.models.auction import AuctionInstance, Order
from solver.models.solution import (
    CustomInteraction,
    Solution,
    SolverResponse,
    TokenAmount,
    Trade,
    TradeKind,
)
from solver.models.types import normalize_address

logger = structlog.get_logger()

# CoW Protocol Settlement contract address (lowercase for consistency)
COW_SETTLEMENT = "0x9008d19f58aabd9ed0d60971565aa8510560ab41"

# Reference price scaling factor (1e18 for precision)
PRICE_SCALE = 10**18


@dataclass
class RoutingResult:
    """Result of routing an order."""

    order: Order
    amount_in: int
    amount_out: int
    pool: UniswapV2Pool | None  # None when no pool found
    success: bool
    error: str | None = None


class SingleOrderRouter:
    """Routes single orders through UniswapV2 pools.

    This is a simple router that handles one order at a time,
    finding the best direct pool for the token pair.
    """

    def route_order(self, order: Order) -> RoutingResult:
        """Find a route for a single order.

        Args:
            order: The order to route

        Returns:
            RoutingResult with success=True if route found, False otherwise
        """
        # Validate input amounts
        try:
            sell_amount = int(order.sell_amount)
            min_buy_amount = int(order.buy_amount)
        except (ValueError, TypeError) as e:
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=None,
                success=False,
                error=f"Invalid amount format: {e}",
            )

        if sell_amount <= 0:
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=None,
                success=False,
                error="Sell amount must be positive",
            )

        if min_buy_amount <= 0:
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=None,
                success=False,
                error="Buy amount must be positive",
            )

        # Find a pool for this token pair
        pool = get_pool(order.sell_token, order.buy_token)
        if pool is None:
            return RoutingResult(
                order=order,
                amount_in=0,
                amount_out=0,
                pool=None,
                success=False,
                error=f"No pool found for {order.sell_token}/{order.buy_token}",
            )

        # Simulate the swap
        swap_result = uniswap_v2.simulate_swap(
            pool=pool,
            token_in=order.sell_token,
            amount_in=sell_amount,
        )

        # Check if output meets minimum (buy_amount is the minimum acceptable)
        if swap_result.amount_out < min_buy_amount:
            return RoutingResult(
                order=order,
                amount_in=sell_amount,
                amount_out=swap_result.amount_out,
                pool=pool,
                success=False,
                error=f"Output {swap_result.amount_out} below minimum {min_buy_amount}",
            )

        return RoutingResult(
            order=order,
            amount_in=sell_amount,
            amount_out=swap_result.amount_out,
            pool=pool,
            success=True,
        )

    def build_solution(
        self,
        routing_result: RoutingResult,
        solution_id: int = 0,
    ) -> Solution | None:
        """Build a complete solution from a routing result.

        Args:
            routing_result: The result of routing an order
            solution_id: ID for this solution

        Returns:
            Solution if successful, None otherwise
        """
        if not routing_result.success:
            return None

        order = routing_result.order

        # Build the trade (order execution)
        trade = Trade(
            kind=TradeKind.FULFILLMENT,
            order=order.uid,
            executedAmount=str(routing_result.amount_in),
        )

        # Encode the swap interaction
        # For a sell order, we swap exact input for minimum output
        #
        # NOTE on token transfers in CoW Protocol:
        # The CoW Settlement contract (0x9008D19f58AAbD9eD0D60971565AA8510560ab41)
        # already has custody of user tokens via ERC20 approvals. The settlement
        # contract handles:
        # 1. Transferring sell tokens from user to the AMM (via transferFrom)
        # 2. Receiving buy tokens from the AMM
        # 3. Transferring buy tokens to the user
        #
        # The solver only needs to encode the swap interaction. The driver/settlement
        # contract handles all token movements based on the solution's trades and
        # interaction inputs/outputs.
        target, calldata = uniswap_v2.encode_swap(
            token_in=order.sell_token,
            token_out=order.buy_token,
            amount_in=routing_result.amount_in,
            amount_out_min=int(order.buy_amount),  # Minimum acceptable
            recipient=COW_SETTLEMENT,
        )

        interaction = CustomInteraction(
            target=target,
            value="0",
            callData=calldata,  # Using alias for mypy compatibility
            internalize=False,
            # Inputs: tokens the interaction consumes from the settlement contract
            # Note: Addresses normalized to lowercase for consistency
            inputs=[
                TokenAmount(
                    token=normalize_address(order.sell_token),
                    amount=str(routing_result.amount_in),
                )
            ],
            # Outputs: tokens the interaction produces for the settlement contract
            outputs=[
                TokenAmount(
                    token=normalize_address(order.buy_token),
                    amount=str(routing_result.amount_out),
                )
            ],
        )

        # Calculate clearing prices
        # CoW Protocol uses uniform clearing prices where:
        #   executed_sell * price[sell_token] >= executed_buy * price[buy_token]
        #
        # For a sell order that sells `amount_in` of sell_token for `amount_out` of buy_token:
        # We set prices such that the exchange rate is amount_out / amount_in
        #
        # Using sell_token as the reference (price = PRICE_SCALE = 1e18):
        #   price[buy_token] = (amount_in * PRICE_SCALE) / amount_out
        #
        # This means: 1 unit of buy_token costs (amount_in/amount_out) units of sell_token
        #
        # Note: Addresses are normalized to lowercase for consistent lookup
        sell_token_normalized = normalize_address(order.sell_token)
        buy_token_normalized = normalize_address(order.buy_token)
        prices = {
            sell_token_normalized: str(PRICE_SCALE),
            buy_token_normalized: str(
                (routing_result.amount_in * PRICE_SCALE) // routing_result.amount_out
            ),
        }

        # Estimate gas for the solution
        gas_estimate = uniswap_v2.SWAP_GAS

        return Solution(
            id=solution_id,
            prices=prices,
            trades=[trade],
            interactions=[interaction],
            gas=gas_estimate,
        )


class Solver:
    """Main solver that processes auction instances."""

    def __init__(self) -> None:
        self.router = SingleOrderRouter()

    def solve(self, auction: AuctionInstance) -> SolverResponse:
        """Solve an auction batch.

        Currently handles single-order auctions only.

        Args:
            auction: The auction to solve

        Returns:
            SolverResponse with proposed solutions
        """
        if auction.order_count == 0:
            return SolverResponse.empty()

        # For now, only handle single-order auctions
        if auction.order_count == 1:
            return self._solve_single_order(auction)

        # Multi-order auctions not yet supported
        return SolverResponse.empty()

    def _solve_single_order(self, auction: AuctionInstance) -> SolverResponse:
        """Solve a single-order auction."""
        order = auction.orders[0]

        # Only handle sell orders for now
        if not order.is_sell_order:
            logger.info(
                "skipping_buy_order",
                order_uid=order.uid,
                message="Buy orders not yet supported (Slice 1.3)",
            )
            return SolverResponse.empty()

        # Log if order is partially fillable (we still process it, but execute fully)
        if order.partially_fillable:
            logger.info(
                "partially_fillable_order",
                order_uid=order.uid,
                message="Partially fillable orders are executed fully or not at all",
            )

        # Route the order
        routing_result = self.router.route_order(order)
        if not routing_result.success:
            return SolverResponse.empty()

        # Build solution
        solution = self.router.build_solution(routing_result, solution_id=0)
        if solution is None:
            return SolverResponse.empty()

        return SolverResponse(solutions=[solution])


# Singleton solver instance
solver = Solver()
