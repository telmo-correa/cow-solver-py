#!/usr/bin/env python3
"""Quick timing analysis of solver components."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.models.auction import AuctionInstance
from solver.models.types import normalize_address
from solver.pools import build_registry_from_liquidity


def main():
    # Load auction
    auction_path = (
        Path(__file__).parent.parent / "data" / "historical_auctions" / "mainnet_11985000.json"
    )
    with open(auction_path) as f:
        data = json.load(f)
    auction = AuctionInstance.model_validate(data)

    print(f"Orders: {len(auction.orders)}")
    print(f"Tokens: {len(auction.tokens)}")
    print(f"Liquidity: {len(auction.liquidity)}")

    # Time pool registry build
    start = time.perf_counter()
    registry = build_registry_from_liquidity(auction.liquidity)
    build_time = time.perf_counter() - start
    print(f"\nRegistry build: {build_time * 1000:.1f}ms")
    print(f"  Pools: {registry.pool_count}")

    # Count unique token pairs
    pairs = set()
    for order in auction.orders:
        pair = (normalize_address(order.sell_token), normalize_address(order.buy_token))
        pairs.add(pair)
    print(f"\nUnique token pairs: {len(pairs)}")

    # Time path checking for all unique pairs
    start = time.perf_counter()
    pairs_with_paths = 0
    pairs_without_paths = 0
    for sell_token, buy_token in pairs:
        paths = registry.get_all_candidate_paths(sell_token, buy_token)
        if paths:
            pairs_with_paths += 1
        else:
            pairs_without_paths += 1
    path_check_time = time.perf_counter() - start
    print(f"\nPath checking (all unique pairs): {path_check_time * 1000:.1f}ms")
    print(f"  Pairs with paths: {pairs_with_paths}")
    print(f"  Pairs without paths: {pairs_without_paths}")

    # Count orders by path availability
    orders_with_paths = 0
    orders_without_paths = 0
    for order in auction.orders:
        pair = (normalize_address(order.sell_token), normalize_address(order.buy_token))
        paths = registry.get_all_candidate_paths(pair[0], pair[1])  # Uses cache
        if paths:
            orders_with_paths += 1
        else:
            orders_without_paths += 1
    print(f"\nOrders with routable pairs: {orders_with_paths}")
    print(f"Orders with no route: {orders_without_paths}")
    print(f"  → {orders_without_paths / len(auction.orders) * 100:.1f}% can be pre-filtered out")


if __name__ == "__main__":
    main()
