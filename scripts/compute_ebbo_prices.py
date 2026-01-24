#!/usr/bin/env python3
"""Compute EBBO reference prices for historical auctions.

EBBO (Ethereum Best Bid and Offer) prices are the best AMM spot prices
available for each token pair. This script:

1. Loads historical auction data
2. Builds a router from the auction's liquidity
3. Computes EBBO prices for all order pairs
4. Saves EBBO data to a companion file

Usage:
    python scripts/compute_ebbo_prices.py [--limit N] [--output-dir DIR]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.amm import uniswap_v2
from solver.amm.balancer import BalancerStableAMM, BalancerWeightedAMM
from solver.amm.limit_order import LimitOrderAMM
from solver.models.auction import AuctionInstance
from solver.models.types import normalize_address
from solver.pools import build_registry_from_liquidity
from solver.routing import SingleOrderRouter


def compute_ebbo_for_auction(auction: AuctionInstance) -> dict:
    """Compute EBBO reference prices for all pairs in an auction.

    Returns:
        Dict mapping (sell_token, buy_token) -> Decimal price ratio
    """
    # Build router from auction liquidity
    pool_registry = build_registry_from_liquidity(auction.liquidity)

    router = SingleOrderRouter(
        amm=uniswap_v2,
        pool_registry=pool_registry,
        v3_amm=None,  # Skip V3 (requires RPC)
        weighted_amm=BalancerWeightedAMM(),
        stable_amm=BalancerStableAMM(),
        limit_order_amm=LimitOrderAMM(),
    )

    # Collect unique order pairs
    pairs: set[tuple[str, str]] = set()
    for order in auction.orders:
        sell = normalize_address(order.sell_token)
        buy = normalize_address(order.buy_token)
        pairs.add((sell, buy))

    # Compute EBBO for each pair
    ebbo_prices: dict[str, dict[str, str]] = defaultdict(dict)

    for sell_token, buy_token in pairs:
        # Get token decimals from auction
        token_info = auction.tokens.get(sell_token)
        decimals = token_info.decimals if token_info and token_info.decimals else 18

        # Query AMM spot price
        price = router.get_reference_price(sell_token, buy_token, token_in_decimals=decimals)

        if price is not None:
            ebbo_prices[sell_token][buy_token] = str(price)

    return dict(ebbo_prices)


def load_auction(path: Path) -> AuctionInstance:
    """Load auction from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return AuctionInstance.model_validate(data)


def main():
    parser = argparse.ArgumentParser(description="Compute EBBO prices for historical auctions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of auctions")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Find historical auction files
    data_dir = Path(__file__).parent.parent / "data" / "historical_auctions"
    auction_files = sorted(data_dir.glob("mainnet_*.json"))

    if args.limit:
        auction_files = auction_files[: args.limit]

    output_dir = args.output_dir or data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing EBBO prices for {len(auction_files)} auctions...")

    total_pairs = 0
    total_prices = 0

    for auction_file in auction_files:
        auction = load_auction(auction_file)

        # Compute EBBO prices
        ebbo = compute_ebbo_for_auction(auction)

        # Count pairs and prices
        pair_count = len(
            {
                (normalize_address(o.sell_token), normalize_address(o.buy_token))
                for o in auction.orders
            }
        )
        price_count = sum(len(v) for v in ebbo.values())
        total_pairs += pair_count
        total_prices += price_count

        # Save EBBO data
        ebbo_file = output_dir / f"{auction_file.stem}_ebbo.json"
        with open(ebbo_file, "w") as f:
            json.dump(
                {
                    "auction_id": auction.id,
                    "auction_file": auction_file.name,
                    "pair_count": len(ebbo),
                    "ebbo_prices": ebbo,
                },
                f,
                indent=2,
            )

        if args.verbose:
            print(f"  {auction_file.name}: {price_count} EBBO prices for {len(ebbo)} pairs")

    print(f"\nDone. Computed {total_prices:,} EBBO prices across {len(auction_files)} auctions.")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
