"""Solution building from routing results."""

from __future__ import annotations

import structlog

from solver.constants import SETTLEMENT_OVERHEAD
from solver.models.solution import (
    Interaction,
    LiquidityInteraction,
    Solution,
    Trade,
    TradeKind,
)
from solver.models.types import normalize_address
from solver.routing.types import RoutingResult

logger = structlog.get_logger()


def build_solution(
    routing_result: RoutingResult,
    solution_id: int = 0,
) -> Solution | None:
    """Build a complete solution from a routing result.

    Creates LiquidityInteraction objects for each hop, referencing the
    liquidity pools by their auction ID. This matches the Rust solver's
    output format and enables CIP-2 internalization.

    Args:
        routing_result: The result of routing an order
        solution_id: ID for this solution

    Returns:
        Solution if successful, None otherwise
    """
    if not routing_result.success:
        logger.debug(
            "build_solution_skipped_failed_routing",
            order_uid=routing_result.order.uid[:18] + "...",
            error=routing_result.error,
        )
        return None

    if not routing_result.hops:
        logger.error("routing_result_missing_hops", order=routing_result.order.uid)
        return None

    order = routing_result.order

    # Build the trade (order execution)
    # executedAmount semantics:
    # - For sell orders: the amount sold (amount_in)
    # - For buy orders: the amount bought (amount_out)
    if order.is_sell_order:  # noqa: SIM108 - explicit if-else for clarity with comments
        executed_amount = routing_result.amount_in
    else:
        executed_amount = routing_result.amount_out

    trade = Trade(
        kind=TradeKind.FULFILLMENT,
        order=order.uid,
        executedAmount=str(executed_amount),
    )

    # Build LiquidityInteraction for each hop
    # This references the auction's liquidity by ID, allowing the driver
    # to use CIP-2 internalization when possible.
    interactions: list[Interaction] = []
    for hop in routing_result.hops:
        if hop.pool.liquidity_id is None:
            logger.error(
                "pool_missing_liquidity_id",
                pool_address=hop.pool.address,
                order=order.uid,
            )
            return None

        interaction = LiquidityInteraction(
            internalize=True,  # Enable CIP-2 internalization
            id=hop.pool.liquidity_id,
            inputToken=hop.input_token,
            outputToken=hop.output_token,
            inputAmount=str(hop.amount_in),
            outputAmount=str(hop.amount_out),
        )
        interactions.append(interaction)

    # Calculate clearing prices
    # CoW Protocol uses uniform clearing prices where:
    #   executed_sell * price[sell_token] >= executed_buy * price[buy_token]
    #
    # The pricing works the same for both sell and buy orders:
    # We set prices such that the exchange rate is amount_out / amount_in
    #
    # Using buy_token as the reference (price = amount_in) to match Rust:
    #   price[sell_token] = amount_out
    #   price[buy_token] = amount_in
    #
    # This maintains the ratio: amount_in / amount_out = price[buy] / price[sell]
    #
    # Note: Addresses are normalized to lowercase for consistent lookup
    sell_token_normalized = normalize_address(order.sell_token)
    buy_token_normalized = normalize_address(order.buy_token)
    prices = {
        buy_token_normalized: str(routing_result.amount_in),
        sell_token_normalized: str(routing_result.amount_out),
    }

    # Calculate total gas: 60k per swap + 106k settlement overhead
    # Formula: sum(POOL_SWAP_GAS_COST per hop) + SETTLEMENT_OVERHEAD
    # Source: cow-services/crates/shared/src/price_estimation/gas.rs
    gas_estimate = routing_result.gas_estimate + SETTLEMENT_OVERHEAD

    return Solution(
        id=solution_id,
        prices=prices,
        trades=[trade],
        interactions=interactions,
        gas=gas_estimate,
    )


__all__ = ["build_solution"]
