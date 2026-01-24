#!/usr/bin/env python3
"""Prototype price enumeration approach for CoW matching.

The key insight from the problem formulation:
- If prices are fixed, fill optimization becomes a LINEAR program
- Limit prices create natural "breakpoints" for price enumeration

Approach:
1. For each crossing pair, enumerate candidate prices from order limits
2. For each candidate price, compute optimal fills (simple greedy)
3. Select price that maximizes matched volume

This is a simplified version that processes pairs independently.
A full solution would handle token overlap and multi-pair optimization.

Usage:
    python scripts/prototype_price_enumeration.py [--limit N]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.models.auction import AuctionInstance, Order
from solver.models.types import normalize_address
from solver.strategies.double_auction import get_limit_price


@dataclass
class MatchResult:
    """Result of matching at a specific price."""

    price: Decimal
    orders_matched: int
    volume_a: int  # Volume of token A exchanged
    volume_b: int  # Volume of token B exchanged
    fills: list[tuple[Order, int, int]]  # (order, sell_filled, buy_filled)


def compute_fills_at_price(
    price: Decimal,
    orders_a_to_b: list[Order],
    orders_b_to_a: list[Order],
) -> MatchResult:
    """Compute optimal fills at a given clearing price.

    At price P (B per A):
    - A-sellers with limit <= P can participate
    - B-sellers with limit >= P can participate

    Returns the maximum matching at this price.
    """
    # Filter to orders that can trade at this price
    eligible_a = []
    for order in orders_a_to_b:
        limit = get_limit_price(order, is_selling_a=True)
        if limit is not None and limit <= price:
            eligible_a.append((order, order.sell_amount_int))

    eligible_b = []
    for order in orders_b_to_a:
        limit = get_limit_price(order, is_selling_a=False)
        if limit is not None and limit >= price:
            eligible_b.append((order, order.sell_amount_int))

    # Total supply of A (from A-sellers)
    total_a_supply = sum(amt for _, amt in eligible_a)

    # Total demand for A (from B-sellers at this price)
    total_a_demand = sum(int(Decimal(amt) / price) for _, amt in eligible_b)

    # Match is limited by the smaller side
    match_a = min(total_a_supply, total_a_demand)
    match_b = int(Decimal(match_a) * price)

    if match_a == 0:
        return MatchResult(
            price=price,
            orders_matched=0,
            volume_a=0,
            volume_b=0,
            fills=[],
        )

    # Allocate fills greedily (best prices first)
    fills = []
    remaining_a = match_a

    # Sort A-sellers by limit price (lowest first = most generous)
    eligible_a.sort(key=lambda x: get_limit_price(x[0], is_selling_a=True) or Decimal("inf"))

    for order, amount in eligible_a:
        if remaining_a <= 0:
            break
        fill_a = min(amount, remaining_a)
        fill_b = int(Decimal(fill_a) * price)
        fills.append((order, fill_a, fill_b))
        remaining_a -= fill_a

    remaining_b = match_b

    # Sort B-sellers by limit price (highest first = most generous)
    eligible_b.sort(
        key=lambda x: get_limit_price(x[0], is_selling_a=False) or Decimal(0),
        reverse=True,
    )

    for order, amount in eligible_b:
        if remaining_b <= 0:
            break
        fill_b = min(amount, remaining_b)
        fill_a = int(Decimal(fill_b) / price)
        fills.append((order, fill_b, fill_a))  # B-seller: sells B, gets A
        remaining_b -= fill_b

    return MatchResult(
        price=price,
        orders_matched=len(fills),
        volume_a=match_a,
        volume_b=match_b,
        fills=fills,
    )


def enumerate_prices(
    orders_a_to_b: list[Order],
    orders_b_to_a: list[Order],
) -> list[Decimal]:
    """Enumerate candidate prices from order limit prices.

    Candidate prices are the breakpoints where order participation changes:
    - Each A-seller's limit is a candidate (orders enter at this price)
    - Each B-seller's limit is a candidate
    - Midpoints between crossing limits
    """
    candidates = set()

    for order in orders_a_to_b:
        limit = get_limit_price(order, is_selling_a=True)
        if limit is not None and limit > 0:
            candidates.add(limit)

    for order in orders_b_to_a:
        limit = get_limit_price(order, is_selling_a=False)
        if limit is not None and limit > 0:
            candidates.add(limit)

    # Sort and return
    return sorted(candidates)


def find_best_price(
    orders_a_to_b: list[Order],
    orders_b_to_a: list[Order],
) -> MatchResult | None:
    """Find the price that maximizes matched volume."""
    if not orders_a_to_b or not orders_b_to_a:
        return None

    candidates = enumerate_prices(orders_a_to_b, orders_b_to_a)

    if not candidates:
        return None

    best_result = None

    for price in candidates:
        result = compute_fills_at_price(price, orders_a_to_b, orders_b_to_a)

        if result.orders_matched > 0 and (
            best_result is None or result.volume_a > best_result.volume_a
        ):
            best_result = result

    return best_result


def analyze_auction_with_enumeration(auction: AuctionInstance) -> dict:
    """Analyze an auction using price enumeration."""
    # Group orders by directed pair
    pair_orders: dict[tuple[str, str], list[Order]] = defaultdict(list)

    for order in auction.orders:
        sell = normalize_address(order.sell_token)
        buy = normalize_address(order.buy_token)
        pair_orders[(sell, buy)].append(order)

    # Find pairs with orders in both directions
    total_matched = 0
    total_volume = 0
    pairs_with_matches = 0
    used_tokens: set[str] = set()  # Track for overlap

    results_by_pair = []

    # Process each undirected pair
    seen: set[tuple[str, str]] = set()

    for sell, buy in pair_orders:
        if (buy, sell) not in pair_orders:
            continue

        normalized = tuple(sorted([sell, buy]))
        if normalized in seen:
            continue
        seen.add(normalized)

        token_a, token_b = normalized
        orders_a_to_b = pair_orders.get((token_a, token_b), [])
        orders_b_to_a = pair_orders.get((token_b, token_a), [])

        result = find_best_price(orders_a_to_b, orders_b_to_a)

        if result and result.orders_matched > 0:
            # Check token overlap
            if token_a in used_tokens or token_b in used_tokens:
                # Skip this pair due to overlap
                continue

            used_tokens.add(token_a)
            used_tokens.add(token_b)

            total_matched += result.orders_matched
            total_volume += result.volume_a
            pairs_with_matches += 1
            results_by_pair.append((token_a, token_b, result))

    return {
        "orders_matched": total_matched,
        "volume_matched": total_volume,
        "pairs_with_matches": pairs_with_matches,
        "details": results_by_pair,
    }


def load_auction(path: Path) -> AuctionInstance:
    """Load an auction from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return AuctionInstance.model_validate(data)


def main():
    parser = argparse.ArgumentParser(description="Prototype price enumeration")
    parser.add_argument("--limit", type=int, default=None, help="Limit auctions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data" / "historical_auctions"
    auction_files = sorted(data_dir.glob("mainnet_*.json"))

    if args.limit:
        auction_files = auction_files[: args.limit]

    print(f"Testing price enumeration on {len(auction_files)} auctions...\n")

    total_orders = 0
    total_matched = 0
    total_pairs = 0
    total_time = 0.0

    for auction_file in auction_files:
        auction = load_auction(auction_file)
        total_orders += auction.order_count

        start = time.perf_counter()
        result = analyze_auction_with_enumeration(auction)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        total_matched += result["orders_matched"]
        total_pairs += result["pairs_with_matches"]

        if args.verbose and result["orders_matched"] > 0:
            print(f"{auction_file.name}: {result['orders_matched']} orders matched")
            for token_a, token_b, match in result["details"]:
                print(
                    f"  {token_a[-8:]} <-> {token_b[-8:]}: {match.orders_matched} orders at price {float(match.price):.6f}"
                )

    print("=" * 80)
    print("PRICE ENUMERATION RESULTS")
    print("=" * 80)

    print(f"\nAuctions: {len(auction_files)}")
    print(f"Total orders: {total_orders:,}")
    print(f"Orders matched: {total_matched:,}")
    print(f"Match rate: {total_matched / total_orders * 100:.2f}%")
    print(f"Pairs with matches: {total_pairs}")
    print(f"Total time: {total_time * 1000:.1f}ms")
    print(f"Avg time per auction: {total_time / len(auction_files) * 1000:.2f}ms")

    # Compare to existing strategies
    print("\n" + "-" * 80)
    print("COMPARISON TO EXISTING STRATEGIES")
    print("-" * 80)
    print(f"""
Strategy          | Orders Matched | Rate
------------------|----------------|------
CowMatch          | 0              | 0.00%
HybridCow         | 192            | 0.07%
RingTrade         | 467            | 0.17%
Price Enumeration | {total_matched:<14,} | {total_matched / total_orders * 100:.2f}%

Improvement over best (RingTrade): {total_matched / 467 if total_matched > 0 else 0:.1f}x
""")


if __name__ == "__main__":
    main()
