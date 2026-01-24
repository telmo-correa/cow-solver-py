"""Hybrid CoW+AMM matching strategy.

.. deprecated::
    This strategy has been **superseded by MultiPairCowStrategy** (Slice 4.6).
    MultiPairCowStrategy provides joint optimization across overlapping token
    pairs, which HybridCowStrategy cannot do. This module is kept for reference
    and research purposes but is NOT part of the default solver chain.

    Use MultiPairCowStrategy for production.

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
from solver.pools import build_registry_from_liquidity
from solver.strategies.base import OrderFill, StrategyResult
from solver.strategies.base_amm import AMMBackedStrategy
from solver.strategies.double_auction import run_hybrid_auction
from solver.strategies.ebbo_bounds import get_ebbo_bounds

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class HybridCowStrategy(AMMBackedStrategy):
    """Strategy that matches orders using hybrid CoW+AMM auction.

    For each token pair with orders in both directions:
    1. Query AMM for reference price
    2. Match orders at AMM price via double auction
    3. Route unmatched orders to AMM

    This captures gas savings from CoW matching while using AMM
    prices to ensure fair execution.

    Like AmmRoutingStrategy, this builds a router from auction liquidity
    at solve time to get reference prices from available pools.

    Args:
        amm: AMM implementation for swap math. Defaults to UniswapV2.
        router: Injected router for testing. If provided, used directly.
        v3_amm: UniswapV3 AMM for V3 pool routing. If None, V3 pools are skipped.
        weighted_amm: Balancer weighted AMM. If None, weighted pools are skipped.
        stable_amm: Balancer stable AMM. If None, stable pools are skipped.
        limit_order_amm: 0x limit order AMM. If None, limit orders are skipped.
    """

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

        # Build pool registry from auction liquidity for price queries
        pool_registry = build_registry_from_liquidity(auction.liquidity)
        router = self._get_router(pool_registry)

        logger.debug(
            "hybrid_cow_pool_registry",
            pool_count=pool_registry.pool_count,
            liquidity_count=len(auction.liquidity),
        )

        # Check for overlapping tokens across pairs
        # If tokens overlap, we can only safely process one pair (the largest)
        # because clearing prices must be consistent across the entire solution
        if len(cow_groups) > 1:
            all_tokens: set[str] = set()
            has_overlap = False
            for group in cow_groups:
                token_a = normalize_address(group.token_a)
                token_b = normalize_address(group.token_b)
                if token_a in all_tokens or token_b in all_tokens:
                    has_overlap = True
                    break
                all_tokens.add(token_a)
                all_tokens.add(token_b)

            if has_overlap:
                # Only process the pair with the most orders
                original_count = len(cow_groups)
                cow_groups = sorted(
                    cow_groups,
                    key=lambda g: len(g.sellers_of_a) + len(g.sellers_of_b),
                    reverse=True,
                )[:1]
                logger.warning(
                    "hybrid_cow_overlapping_tokens",
                    original_pairs=original_count,
                    message="Tokens overlap across pairs; processing only largest pair",
                )

        # Process each CoW group
        all_fills: list[OrderFill] = []
        all_remainders = []
        all_prices: dict[str, str] = {}

        for group in cow_groups:
            # Get two-sided EBBO bounds
            bounds = get_ebbo_bounds(group.token_a, group.token_b, router, auction)

            logger.debug(
                "hybrid_cow_processing_pair",
                token_a=group.token_a[-8:],
                token_b=group.token_b[-8:],
                sellers_of_a=len(group.sellers_of_a),
                sellers_of_b=len(group.sellers_of_b),
                amm_price=float(bounds.amm_price) if bounds.amm_price else None,
                ebbo_min=float(bounds.ebbo_min) if bounds.ebbo_min else None,
                ebbo_max=float(bounds.ebbo_max) if bounds.ebbo_max else None,
            )

            # Run hybrid auction with both EBBO bounds
            # (EBBO validation is now handled inside run_hybrid_auction)
            result = run_hybrid_auction(
                group,
                amm_price=bounds.amm_price,
                ebbo_min=bounds.ebbo_min,
                ebbo_max=bounds.ebbo_max,
            )

            if not result.cow_matches:
                # No matches - add all orders as remainders for AMM routing
                for order in group.sellers_of_a:
                    all_remainders.append(order)
                for order in group.sellers_of_b:
                    all_remainders.append(order)
                continue

            # EBBO validation is now handled by run_hybrid_auction with both bounds

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
            # Note: Overlapping tokens across pairs are handled earlier by
            # filtering to only the largest pair when overlap is detected.
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
                # When capping, also scale buy_filled proportionally to maintain
                # the clearing price ratio (required for valid settlement)
                max_sell = fill.order.sell_amount_int
                if merged_sell > max_sell:
                    # Scale buy_filled proportionally
                    scale_factor = max_sell / merged_sell
                    original_buy = merged_buy
                    merged_buy = int(merged_buy * scale_factor)
                    logger.warning(
                        "hybrid_cow_fill_overflow_capped",
                        uid=uid[:16] + "...",
                        merged_sell=merged_sell,
                        max_sell=max_sell,
                        original_buy=original_buy,
                        scaled_buy=merged_buy,
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
