#!/usr/bin/env python3
"""Prototype LP-based fill optimization.

Given fixed prices, fill optimization becomes a Linear Program:

Variables:
  x_i = fill amount for order i (in sell token units)

Objective:
  maximize sum(x_i) = total fill volume

Constraints:
  1. Fill bounds: 0 <= x_i <= S_i (max sell amount)
  2. Conservation: sum(sells of t) = sum(buys of t) for each token t
  3. Limit satisfaction: implicitly handled by price filtering

This prototype:
1. Enumerates candidate prices
2. For each price, solves LP for optimal fills
3. Compares to greedy approach

Usage:
    python scripts/prototype_lp_solver.py [--limit N]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.models.auction import AuctionInstance, Order
from solver.models.types import normalize_address
from solver.strategies.double_auction import get_limit_price


@dataclass
class LPResult:
    """Result of LP-based fill optimization."""

    price: Decimal
    orders_matched: int
    total_fill: float
    solve_time_ms: float
    fills: list[tuple[Order, float]]  # (order, fill_amount)


def solve_fills_lp(
    price: Decimal,
    orders_a_to_b: list[Order],
    orders_b_to_a: list[Order],
) -> LPResult | None:
    """Solve LP for optimal fills at given price.

    The LP maximizes total fill volume subject to:
    - Fill bounds (0 <= x_i <= S_i)
    - Conservation (sum of A sold = sum of A bought)
    """
    # Filter to eligible orders at this price
    eligible_a = []
    for order in orders_a_to_b:
        limit = get_limit_price(order, is_selling_a=True)
        if limit is not None and limit <= price:
            eligible_a.append(order)

    eligible_b = []
    for order in orders_b_to_a:
        limit = get_limit_price(order, is_selling_a=False)
        if limit is not None and limit >= price:
            eligible_b.append(order)

    if not eligible_a or not eligible_b:
        return None

    # Number of variables: one per order
    n_a = len(eligible_a)
    n_b = len(eligible_b)
    n_vars = n_a + n_b

    # Objective: maximize sum of fills (negative for minimization)
    # We maximize A-side fills (which determines total match)
    c = np.zeros(n_vars)
    c[:n_a] = -1  # Maximize A-side fills

    # Bounds: 0 <= x_i <= S_i
    bounds = []
    for order in eligible_a:
        bounds.append((0, order.sell_amount_int))
    for order in eligible_b:
        # B sellers: their max sell is in B, but we express fill in "A equivalent"
        max_a_from_b = float(Decimal(order.sell_amount_int) / price)
        bounds.append((0, max_a_from_b))

    # Conservation constraint: sum(A sold) = sum(A bought)
    # A_eq @ x = b_eq
    # sum of x[:n_a] = sum of x[n_a:] (both in A terms)
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_a] = 1  # A sellers contribute A
    A_eq[0, n_a:] = -1  # B sellers contribute A (in A terms)
    b_eq = np.array([0.0])

    start = time.perf_counter()
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    solve_time = (time.perf_counter() - start) * 1000

    if not result.success:
        return None

    # Extract fills
    fills = []
    total_fill = 0.0

    for i, order in enumerate(eligible_a):
        fill = result.x[i]
        if fill > 1e-6:  # Numerical tolerance
            fills.append((order, fill))
            total_fill += fill

    for i, order in enumerate(eligible_b):
        fill = result.x[n_a + i]
        if fill > 1e-6:
            fills.append((order, fill))

    return LPResult(
        price=price,
        orders_matched=len(fills),
        total_fill=total_fill,
        solve_time_ms=solve_time,
        fills=fills,
    )


def enumerate_prices(
    orders_a_to_b: list[Order],
    orders_b_to_a: list[Order],
) -> list[Decimal]:
    """Enumerate candidate prices from order limit prices."""
    candidates = set()

    for order in orders_a_to_b:
        limit = get_limit_price(order, is_selling_a=True)
        if limit is not None and limit > 0:
            candidates.add(limit)

    for order in orders_b_to_a:
        limit = get_limit_price(order, is_selling_a=False)
        if limit is not None and limit > 0:
            candidates.add(limit)

    return sorted(candidates)


def find_best_price_lp(
    orders_a_to_b: list[Order],
    orders_b_to_a: list[Order],
) -> LPResult | None:
    """Find price maximizing fill using LP solver."""
    if not orders_a_to_b or not orders_b_to_a:
        return None

    candidates = enumerate_prices(orders_a_to_b, orders_b_to_a)
    if not candidates:
        return None

    best_result = None
    total_solve_time = 0.0

    for price in candidates:
        result = solve_fills_lp(price, orders_a_to_b, orders_b_to_a)

        if result is not None:
            total_solve_time += result.solve_time_ms

            if best_result is None or result.total_fill > best_result.total_fill:
                best_result = result

    if best_result:
        best_result.solve_time_ms = total_solve_time

    return best_result


def analyze_auction_with_lp(auction: AuctionInstance) -> dict:
    """Analyze auction using LP-based fill optimization."""
    pair_orders: dict[tuple[str, str], list[Order]] = defaultdict(list)

    for order in auction.orders:
        sell = normalize_address(order.sell_token)
        buy = normalize_address(order.buy_token)
        pair_orders[(sell, buy)].append(order)

    total_matched = 0
    total_fill = 0.0
    total_solve_time = 0.0
    pairs_with_matches = 0
    used_tokens: set[str] = set()

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

        result = find_best_price_lp(orders_a_to_b, orders_b_to_a)

        if result and result.orders_matched > 0:
            # Skip if token overlap
            if token_a in used_tokens or token_b in used_tokens:
                continue

            used_tokens.add(token_a)
            used_tokens.add(token_b)

            total_matched += result.orders_matched
            total_fill += result.total_fill
            total_solve_time += result.solve_time_ms
            pairs_with_matches += 1

    return {
        "orders_matched": total_matched,
        "total_fill": total_fill,
        "solve_time_ms": total_solve_time,
        "pairs_with_matches": pairs_with_matches,
    }


def load_auction(path: Path) -> AuctionInstance:
    """Load auction from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return AuctionInstance.model_validate(data)


def main():
    parser = argparse.ArgumentParser(description="Prototype LP-based solver")
    parser.add_argument("--limit", type=int, default=None, help="Limit auctions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data" / "historical_auctions"
    auction_files = sorted(data_dir.glob("mainnet_*.json"))

    if args.limit:
        auction_files = auction_files[: args.limit]

    print(f"Testing LP-based solver on {len(auction_files)} auctions...\n")

    total_orders = 0
    total_matched = 0
    total_fill = 0.0
    total_solve_time = 0.0
    total_pairs = 0
    total_wall_time = 0.0

    for auction_file in auction_files:
        auction = load_auction(auction_file)
        total_orders += auction.order_count

        start = time.perf_counter()
        result = analyze_auction_with_lp(auction)
        wall_time = time.perf_counter() - start
        total_wall_time += wall_time

        total_matched += result["orders_matched"]
        total_fill += result["total_fill"]
        total_solve_time += result["solve_time_ms"]
        total_pairs += result["pairs_with_matches"]

        if args.verbose and result["orders_matched"] > 0:
            print(
                f"{auction_file.name}: {result['orders_matched']} orders, "
                f"fill={result['total_fill']:.0f}, "
                f"LP time={result['solve_time_ms']:.1f}ms"
            )

    print("=" * 80)
    print("LP SOLVER RESULTS")
    print("=" * 80)

    print(f"\nAuctions: {len(auction_files)}")
    print(f"Total orders: {total_orders:,}")
    print(f"Orders matched: {total_matched:,}")
    print(f"Match rate: {total_matched / total_orders * 100:.2f}%")
    print(f"Total fill: {total_fill:,.0f}")
    print(f"Pairs with matches: {total_pairs}")

    print("\n" + "-" * 80)
    print("TIMING")
    print("-" * 80)
    print(f"LP solver time: {total_solve_time:.1f}ms total")
    print(f"Wall clock time: {total_wall_time * 1000:.1f}ms total")
    print(f"Avg per auction: {total_wall_time / len(auction_files) * 1000:.2f}ms")

    print("\n" + "-" * 80)
    print("COMPARISON")
    print("-" * 80)
    print(f"""
Strategy          | Orders Matched | Rate   | Time/auction
------------------|----------------|--------|-------------
RingTrade         | 467            | 0.17%  | 56ms
Price Enum        | 522            | 0.19%  | 149ms
LP Solver         | {total_matched:<14,} | {total_matched / total_orders * 100:.2f}%  | {total_wall_time / len(auction_files) * 1000:.0f}ms
""")


if __name__ == "__main__":
    main()
