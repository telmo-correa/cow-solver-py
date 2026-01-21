"""Script to collect historical auctions from CoW Protocol API.

This script fetches recent auctions from the CoW Protocol orderbook API
and saves them as JSON fixtures for testing and benchmarking.

Usage:
    python -m scripts.collect_auctions --output tests/fixtures/auctions --count 100
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger()

# CoW Protocol API endpoints
COW_API_URLS = {
    "mainnet": "https://api.cow.fi/mainnet",
    "arbitrum-one": "https://api.cow.fi/arbitrum_one",
    "base": "https://api.cow.fi/base",
    "xdai": "https://api.cow.fi/xdai",
}

# Alternative: solver competition archive (if available)
SOLVER_ARCHIVE_URL = "https://solver-instances.s3.eu-central-1.amazonaws.com"


async def fetch_recent_orders(
    network: str = "mainnet",
    limit: int = 100,
) -> list[dict]:
    """Fetch recent orders from the CoW API.

    Note: The API provides orders, not complete auctions.
    We'll need to synthesize auctions from these orders.
    """
    base_url = COW_API_URLS.get(network, COW_API_URLS["mainnet"])
    url = f"{base_url}/api/v1/orders"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            url,
            params={"limit": limit, "status": "open,fulfilled"},
        )
        response.raise_for_status()
        return response.json()


async def fetch_auction_from_archive(
    auction_id: str,
    network: str = "mainnet",
) -> dict | None:
    """Fetch a specific auction from the solver archive (if available).

    The CoW team may publish historical auctions for solver development.
    """
    # This URL structure is hypothetical - actual archive may differ
    url = f"{SOLVER_ARCHIVE_URL}/{network}/auctions/{auction_id}.json"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError:
            logger.debug("auction_not_found", auction_id=auction_id)
            return None


def synthesize_auction_from_orders(
    orders: list[dict],
    auction_id: str | None = None,
) -> dict:
    """Create a synthetic auction instance from a list of orders.

    This creates a minimal auction structure for testing.
    Real auctions would include liquidity sources, gas prices, etc.
    """
    # Extract unique tokens
    tokens: dict[str, dict] = {}
    for order in orders:
        for token_addr in [order.get("sellToken"), order.get("buyToken")]:
            if token_addr and token_addr not in tokens:
                tokens[token_addr] = {
                    "decimals": 18,  # Default, would need to fetch actual
                    "symbol": None,
                    "referencePrice": None,
                    "availableBalance": "0",
                    "trusted": True,
                }

    # Transform orders to match auction schema
    auction_orders = []
    for order in orders:
        auction_orders.append(
            {
                "uid": order.get("uid", ""),
                "sellToken": order.get("sellToken", ""),
                "buyToken": order.get("buyToken", ""),
                "sellAmount": order.get("sellAmount", "0"),
                "buyAmount": order.get("buyAmount", "0"),
                "feeAmount": order.get("feeAmount", "0"),
                "kind": order.get("kind", "sell"),
                "partiallyFillable": order.get("partiallyFillable", False),
                "class": order.get("class", "limit"),
            }
        )

    return {
        "id": auction_id or f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "tokens": tokens,
        "orders": auction_orders,
        "liquidity": [],  # Would need to fetch from on-chain
        "effectiveGasPrice": "30000000000",  # 30 gwei default
        "deadline": None,
        "surplusCapturingJitOrderOwners": [],
    }


def categorize_auction(auction: dict) -> str:
    """Determine the category for an auction based on its characteristics."""
    orders = auction.get("orders", [])
    order_count = len(orders)

    if order_count == 0:
        return "empty"
    elif order_count == 1:
        return "single_order"

    # Check for CoW potential (opposite direction orders on same pair)
    pairs_with_both_directions: set[tuple[str, str]] = set()
    sell_pairs: set[tuple[str, str]] = set()
    buy_pairs: set[tuple[str, str]] = set()

    for order in orders:
        pair = (order["sellToken"].lower(), order["buyToken"].lower())
        reverse_pair = (pair[1], pair[0])

        if order["kind"] == "sell":
            sell_pairs.add(pair)
            if reverse_pair in sell_pairs or pair in buy_pairs:
                pairs_with_both_directions.add(tuple(sorted([pair[0], pair[1]])))
        else:
            buy_pairs.add(pair)
            if reverse_pair in buy_pairs or pair in sell_pairs:
                pairs_with_both_directions.add(tuple(sorted([pair[0], pair[1]])))

    if pairs_with_both_directions:
        return "cow_pairs"

    # Check if multi-hop routing might be needed
    unique_tokens: set[str] = set()
    for order in orders:
        unique_tokens.add(order["sellToken"].lower())
        unique_tokens.add(order["buyToken"].lower())

    if len(unique_tokens) > 4:
        return "multi_hop"

    return "standard"


async def collect_auctions(
    output_dir: Path,
    network: str = "mainnet",
    count: int = 100,
    _batch_size: int = 10,  # Reserved for future batched fetching
) -> int:
    """Collect historical auctions and save as fixtures.

    Returns the number of auctions collected.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create category subdirectories
    for category in ["single_order", "cow_pairs", "multi_hop", "standard", "empty"]:
        (output_dir / category).mkdir(exist_ok=True)

    logger.info("fetching_orders", network=network, count=count)

    # Fetch orders
    try:
        orders = await fetch_recent_orders(network=network, limit=count * 2)
    except Exception as e:
        logger.error("failed_to_fetch_orders", error=str(e))
        return 0

    logger.info("fetched_orders", count=len(orders))

    # Group orders into synthetic auctions
    # For now, create single-order auctions + some multi-order batches
    auctions_saved = 0

    # Single-order auctions
    for i, order in enumerate(orders[: count // 2]):
        auction = synthesize_auction_from_orders(
            [order],
            auction_id=f"single_{i}",
        )
        category = categorize_auction(auction)
        path = output_dir / category / f"auction_{i:04d}.json"

        with open(path, "w") as f:
            json.dump(auction, f, indent=2)

        auctions_saved += 1

    # Multi-order auctions (batches of 2-5 orders)
    remaining_orders = orders[count // 2 :]
    batch_start = 0

    for batch_idx in range(count // 2):
        if batch_start >= len(remaining_orders):
            break

        batch_end = min(batch_start + (batch_idx % 4) + 2, len(remaining_orders))
        batch_orders = remaining_orders[batch_start:batch_end]

        auction = synthesize_auction_from_orders(
            batch_orders,
            auction_id=f"batch_{batch_idx}",
        )
        category = categorize_auction(auction)
        path = output_dir / category / f"auction_batch_{batch_idx:04d}.json"

        with open(path, "w") as f:
            json.dump(auction, f, indent=2)

        batch_start = batch_end
        auctions_saved += 1

    logger.info("auctions_saved", count=auctions_saved, output_dir=str(output_dir))
    return auctions_saved


def main() -> None:
    """Entry point for the auction collector script."""
    parser = argparse.ArgumentParser(
        description="Collect historical CoW Protocol auctions for testing"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/auctions"),
        help="Output directory for auction fixtures",
    )
    parser.add_argument(
        "--network",
        choices=list(COW_API_URLS.keys()),
        default="mainnet",
        help="Network to fetch auctions from",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of auctions to collect",
    )

    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )

    collected = asyncio.run(
        collect_auctions(
            output_dir=args.output,
            network=args.network,
            count=args.count,
        )
    )

    print(f"\nCollected {collected} auctions to {args.output}")


if __name__ == "__main__":
    main()
