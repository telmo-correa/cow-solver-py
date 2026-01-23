"""Hybrid CoW+AMM matching strategy.

This strategy extends simple 2-order CoW matching to handle N orders
by using AMM reference prices to unlock more matches.

Key insight: Most CoW pairs have "crossing prices" where ask > bid,
meaning pure peer-to-peer matching rarely works. By using the AMM
price as a reference, orders that can both execute against the AMM
can instead match directly via CoW, saving gas fees.

Algorithm:
1. Group orders by token pair
2. For each pair with CoW potential (orders in both directions):
   a. Get AMM reference price
   b. Run hybrid auction at AMM price
   c. Convert matches to fills
3. Return fills and remainders for AMM routing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from solver.models.auction import AuctionInstance
from solver.models.order_groups import find_cow_opportunities
from solver.models.types import normalize_address
from solver.strategies.base import OrderFill, StrategyResult
from solver.strategies.double_auction import run_hybrid_auction

if TYPE_CHECKING:
    from solver.routing.router import SingleOrderRouter

logger = structlog.get_logger()


class HybridCowStrategy:
    """Strategy that matches orders using hybrid CoW+AMM auction.

    For each token pair with orders in both directions:
    1. Query AMM for reference price
    2. Match orders at AMM price via double auction
    3. Route unmatched orders to AMM

    This captures gas savings from CoW matching while using AMM
    prices to ensure fair execution.

    Args:
        router: Router to query AMM reference prices
    """

    def __init__(self, router: SingleOrderRouter) -> None:
        """Initialize with router for AMM price queries."""
        self.router = router

    def try_solve(self, auction: AuctionInstance) -> StrategyResult | None:
        """Try to find CoW matches using hybrid auction.

        Args:
            auction: The auction to solve

        Returns:
            StrategyResult with CoW fills and remainders, or None if no matches
        """
        if auction.order_count < 2:
            return None

        # Find token pairs with CoW potential (orders in both directions)
        cow_groups = find_cow_opportunities(auction.orders)

        if not cow_groups:
            logger.debug(
                "hybrid_cow_no_opportunities",
                order_count=auction.order_count,
            )
            return None

        # Process each CoW group
        all_fills: list[OrderFill] = []
        all_remainders = []
        all_prices: dict[str, str] = {}

        for group in cow_groups:
            # Get AMM reference price for this pair
            amm_price = self.router.get_reference_price(group.token_a, group.token_b)

            logger.debug(
                "hybrid_cow_processing_pair",
                token_a=group.token_a[-8:],
                token_b=group.token_b[-8:],
                sellers_of_a=len(group.sellers_of_a),
                sellers_of_b=len(group.sellers_of_b),
                amm_price=float(amm_price) if amm_price else None,
            )

            # Run hybrid auction (uses AMM price if available, falls back to pure
            # double auction with uniform clearing price if not)
            result = run_hybrid_auction(group, amm_price=amm_price)

            if not result.cow_matches:
                # No matches - add all orders as remainders for AMM routing
                for order in group.sellers_of_a:
                    all_remainders.append(order)
                for order in group.sellers_of_b:
                    all_remainders.append(order)
                continue

            # Convert matches to fills
            for match in result.cow_matches:
                # Seller (sells A, receives B)
                fill_seller = OrderFill(
                    order=match.seller,
                    sell_filled=match.amount_a,
                    buy_filled=match.amount_b,
                )
                all_fills.append(fill_seller)

                # Buyer (sells B, receives A)
                fill_buyer = OrderFill(
                    order=match.buyer,
                    sell_filled=match.amount_b,
                    buy_filled=match.amount_a,
                )
                all_fills.append(fill_buyer)

            # Set clearing prices for this pair
            # Price is set so token conservation holds: sum(sell) = sum(buy)
            # NOTE: For multi-pair auctions where tokens overlap, later pair
            # prices overwrite earlier. This is acceptable for the single-pair
            # optimization focus of this strategy.
            token_a_norm = normalize_address(group.token_a)
            token_b_norm = normalize_address(group.token_b)
            all_prices[token_a_norm] = str(result.total_cow_b)
            all_prices[token_b_norm] = str(result.total_cow_a)

            # Add AMM routes as remainders (only for orders with NO fill)
            # Orders with partial fills will have remainders computed from fills
            matched_uids = {match.seller.uid for match in result.cow_matches}
            matched_uids.update(match.buyer.uid for match in result.cow_matches)
            for route in result.amm_routes:
                if route.order.uid not in matched_uids:
                    all_remainders.append(route.order)

            logger.info(
                "hybrid_cow_pair_matched",
                token_a=group.token_a[-8:],
                token_b=group.token_b[-8:],
                cow_matches=len(result.cow_matches),
                total_cow_a=result.total_cow_a,
                amm_routes=len(result.amm_routes),
            )

        if not all_fills:
            return None

        # Deduplicate fills by order UID (same order might appear in multiple matches)
        fills_by_uid: dict[str, OrderFill] = {}
        for fill in all_fills:
            uid = fill.order.uid
            if uid in fills_by_uid:
                # Merge fills for same order
                existing = fills_by_uid[uid]
                merged_sell = existing.sell_filled + fill.sell_filled
                merged_buy = existing.buy_filled + fill.buy_filled

                # Cap sell_filled at order's maximum sell amount
                # Note: buy_filled has no maximum - users WANT to receive more
                # (buy_amount is the MINIMUM they require, not maximum)
                max_sell = fill.order.sell_amount_int
                if merged_sell > max_sell:
                    logger.warning(
                        "hybrid_cow_fill_overflow_capped",
                        uid=uid[:16] + "...",
                        merged_sell=merged_sell,
                        max_sell=max_sell,
                    )
                    merged_sell = max_sell

                fills_by_uid[uid] = OrderFill(
                    order=fill.order,
                    sell_filled=merged_sell,
                    buy_filled=merged_buy,
                )
            else:
                fills_by_uid[uid] = fill

        # Compute remainder orders from partial fills
        final_remainders = list(all_remainders)
        for fill in fills_by_uid.values():
            remainder = fill.get_remainder_order()
            if remainder:
                final_remainders.append(remainder)

        logger.info(
            "hybrid_cow_complete",
            total_fills=len(fills_by_uid),
            total_remainders=len(final_remainders),
            pairs_processed=len(cow_groups),
        )

        return StrategyResult(
            fills=list(fills_by_uid.values()),
            interactions=[],  # CoW matches have no AMM interactions
            prices=all_prices,
            gas=0,  # No on-chain swaps
            remainder_orders=final_remainders,
        )
