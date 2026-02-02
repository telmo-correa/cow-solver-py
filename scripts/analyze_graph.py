#!/usr/bin/env python3
"""Analyze token graph structure."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.models.auction import AuctionInstance
from solver.pools import build_registry_from_liquidity
from solver.routing.pathfinding import TokenGraph


def main():
    # Load auction
    auction_path = Path(__file__).parent.parent / "data" / "historical_auctions" / "mainnet_11985000.json"
    with open(auction_path) as f:
        data = json.load(f)
    auction = AuctionInstance.model_validate(data)

    registry = build_registry_from_liquidity(auction.liquidity)
    graph = TokenGraph.from_registry(registry)

    print(f"Graph structure:")
    print(f"  Tokens (nodes): {graph.token_count}")

    # Count edges
    total_edges = 0
    max_degree = 0
    degrees = []
    for token in graph._adjacency:
        degree = len(graph._adjacency[token])
        degrees.append(degree)
        total_edges += degree
        max_degree = max(max_degree, degree)
    total_edges //= 2  # Undirected

    print(f"  Edges: {total_edges}")
    print(f"  Max degree: {max_degree}")
    print(f"  Avg degree: {sum(degrees)/len(degrees):.1f}")

    # Find high-degree nodes (likely base tokens like WETH, USDC)
    print(f"\nTop 10 highest degree tokens:")
    sorted_tokens = sorted(graph._adjacency.items(), key=lambda x: len(x[1]), reverse=True)
    for token, neighbors in sorted_tokens[:10]:
        print(f"  {token[-8:]}: {len(neighbors)} neighbors")

    # Time a few path queries
    test_pairs = [
        ("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"),  # WETH/USDC
        ("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "0x6b175474e89094c44da98b954eedeac495271d0f"),  # WETH/DAI
    ]
    print(f"\nPath query timing:")
    for sell, buy in test_pairs:
        start = time.perf_counter()
        paths = registry.pathfinder.find_all_paths(sell, buy, max_hops=3)
        elapsed = time.perf_counter() - start
        print(f"  {sell[-8:]}/{buy[-8:]}: {len(paths)} paths, {elapsed*1000:.1f}ms")

        # Second call (cached)
        start = time.perf_counter()
        paths = registry.pathfinder.find_all_paths(sell, buy, max_hops=3)
        elapsed = time.perf_counter() - start
        print(f"    (cached): {elapsed*1000:.2f}ms")


if __name__ == "__main__":
    main()
