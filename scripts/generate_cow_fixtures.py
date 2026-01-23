#!/usr/bin/env python3
"""Generate test fixtures with N>2 orders for CoW matching benchmarks.

Creates auctions with 3-5 orders on the same token pair where:
- Orders have overlapping limit prices (CoW potential)
- Mix of fill-or-kill and partially fillable orders
- Realistic token pairs (WETH/USDC, WETH/DAI, etc.)

Usage:
    python scripts/generate_cow_fixtures.py \
        --output tests/fixtures/auctions/n_order_cow \
        --count 10
"""

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

# Token addresses (mainnet)
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
DAI = "0x6B175474E89094C44Da98b954EescdeCB5C6F9be"

TOKENS = {
    WETH: {"decimals": 18, "symbol": "WETH", "referencePrice": "1000000000000000000"},
    USDC: {"decimals": 6, "symbol": "USDC", "referencePrice": "400000000000000"},
    DAI: {"decimals": 18, "symbol": "DAI", "referencePrice": "400000000000000000"},
}


def generate_uid() -> str:
    """Generate a unique order UID (must be 112 hex characters)."""
    # UID format: 0x + 112 hex chars = 114 total
    hex_str = uuid.uuid4().hex + uuid.uuid4().hex + uuid.uuid4().hex + uuid.uuid4().hex
    return "0x" + hex_str[:112]


def generate_address() -> str:
    """Generate a random address (must be 40 hex characters)."""
    # Address format: 0x + 40 hex chars = 42 total
    hex_str = uuid.uuid4().hex + uuid.uuid4().hex
    return "0x" + hex_str[:40]


def create_order(
    sell_token: str,
    buy_token: str,
    sell_amount: int,
    buy_amount: int,
    kind: str = "sell",
    partially_fillable: bool = True,
    order_class: str = "limit",
) -> dict:
    """Create an order dictionary."""
    return {
        "uid": generate_uid(),
        "sellToken": sell_token,
        "buyToken": buy_token,
        "sellAmount": str(sell_amount),
        "fullSellAmount": str(sell_amount),
        "buyAmount": str(buy_amount),
        "fullBuyAmount": str(buy_amount),
        "feePolicies": [],
        "validTo": 0,
        "kind": kind,
        "owner": generate_address(),
        "partiallyFillable": partially_fillable,
        "preInteractions": [],
        "postInteractions": [],
        "sellTokenSource": "erc20",
        "buyTokenDestination": "erc20",
        "class": order_class,
        "appData": "0x" + "0" * 64,
        "signingScheme": "presign",
        "signature": "0x",
    }


def create_v2_liquidity(
    token_a: str,
    token_b: str,
    reserve_a: int,
    reserve_b: int,
) -> dict:
    """Create a UniswapV2 liquidity source."""
    return {
        "kind": "constantProduct",
        "id": f"uniswap-v2-{TOKENS[token_a]['symbol'].lower()}-{TOKENS[token_b]['symbol'].lower()}",
        "address": generate_address(),
        "router": "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
        "gasEstimate": "110000",
        "tokens": {
            token_a: {"balance": str(reserve_a)},
            token_b: {"balance": str(reserve_b)},
        },
        "fee": "0.003",
    }


def generate_n_order_auction(
    num_orders: int = 3,
    seed: int | None = None,
    include_fok: bool = True,
) -> dict:
    """Generate an auction with N orders on WETH/USDC pair.

    Creates orders with overlapping limit prices where CoW matching at AMM price
    is beneficial (saves gas while satisfying limit prices).

    Key insight: For HybridCow to win over pure AMM routing, orders must:
    1. Have limit prices that allow execution at AMM reference price
    2. Both sides can match at AMM price, saving gas fees

    Args:
        num_orders: Number of orders (3-5 recommended)
        seed: Random seed for reproducibility
        include_fok: Include fill-or-kill orders in the mix
    """
    if seed is not None:
        random.seed(seed)

    # Use WETH/USDC pair
    token_a = WETH  # 18 decimals
    token_b = USDC  # 6 decimals

    # AMM reference price: 1 WETH = 2500 USDC
    amm_price_a_per_b = 2500  # USDC per WETH

    # Create orders - half sellers of A, half sellers of B
    orders = []
    num_sellers_a = num_orders // 2 + num_orders % 2  # Slightly more sellers of A
    num_sellers_b = num_orders - num_sellers_a

    # Sellers of WETH (A) - want USDC (B)
    # These are "asks" - they want AT LEAST X USDC per WETH
    # For CoW matching at AMM price (2500), they should ask for <= 2500 USDC/WETH
    for i in range(num_sellers_a):
        # Sell 0.5-2 WETH
        sell_amount = random.randint(5, 20) * 10**17  # 0.5 to 2 WETH
        # Ask for slightly LESS than AMM price (willing to accept AMM price)
        # This makes them matchable at AMM price
        price_variation = random.uniform(0.96, 0.99)  # Ask 1-4% below AMM
        buy_amount = int(sell_amount * amm_price_a_per_b * price_variation / 10**12)

        fok = include_fok and i == 0  # First seller is fill-or-kill
        orders.append(
            create_order(
                sell_token=token_a,
                buy_token=token_b,
                sell_amount=sell_amount,
                buy_amount=buy_amount,
                partially_fillable=not fok,
            )
        )

    # Sellers of USDC (B) - want WETH (A)
    # These are "bids" - they're willing to pay UP TO X USDC per WETH
    # For CoW matching at AMM price (2500), they should bid >= 2500 USDC/WETH
    for i in range(num_sellers_b):
        # Sell 1000-5000 USDC
        sell_amount = random.randint(1000, 5000) * 10**6  # 1000 to 5000 USDC
        # Bid slightly MORE than AMM price (willing to pay AMM price)
        # This makes them matchable at AMM price
        price_variation = random.uniform(1.01, 1.04)  # Bid 1-4% above AMM
        buy_amount = int(sell_amount / (amm_price_a_per_b * price_variation) * 10**12)

        fok = include_fok and i == 0  # First seller is fill-or-kill
        orders.append(
            create_order(
                sell_token=token_b,
                buy_token=token_a,
                sell_amount=sell_amount,
                buy_amount=buy_amount,
                partially_fillable=not fok,
            )
        )

    # Create AMM liquidity (50M USDC, 20K WETH)
    liquidity = [
        create_v2_liquidity(
            token_a=token_a,
            token_b=token_b,
            reserve_a=20000 * 10**18,  # 20K WETH
            reserve_b=50_000_000 * 10**6,  # 50M USDC
        )
    ]

    # Build auction
    auction = {
        "id": f"n_order_cow_{num_orders}_{seed or random.randint(0, 9999)}",
        "tokens": {
            token_a: {
                **TOKENS[token_a],
                "availableBalance": str(
                    sum(int(o["sellAmount"]) for o in orders if o["sellToken"] == token_a)
                ),
                "trusted": True,
            },
            token_b: {
                **TOKENS[token_b],
                "availableBalance": str(
                    sum(int(o["sellAmount"]) for o in orders if o["sellToken"] == token_b)
                ),
                "trusted": True,
            },
        },
        "orders": orders,
        "liquidity": liquidity,
        "effectiveGasPrice": "30000000000",
        "deadline": "2106-01-01T00:00:00.000Z",
        "surplusCapturingJitOrderOwners": [],
    }

    return auction


def generate_tight_spread_auction(seed: int | None = None) -> dict:
    """Generate auction with tight bid-ask spread overlapping AMM price.

    Orders have prices that allow matching at AMM price (2500 USDC/WETH):
    - WETH sellers ask for <= 2500 USDC/WETH
    - USDC sellers bid >= 2500 USDC/WETH
    This enables CoW matching at AMM price, saving gas.
    """
    if seed is not None:
        random.seed(seed)

    token_a = WETH
    token_b = USDC

    orders = []

    # Seller of WETH: sell 1 WETH, want 2450 USDC (ask below AMM, can match at 2500)
    orders.append(
        create_order(
            sell_token=token_a,
            buy_token=token_b,
            sell_amount=10**18,
            buy_amount=2450 * 10**6,  # Ask price 2450 USDC/WETH
            partially_fillable=True,
        )
    )

    # Seller of USDC: sell 2550 USDC, want 1 WETH (bid above AMM, can match at 2500)
    # bid price = 2550 USDC/WETH
    orders.append(
        create_order(
            sell_token=token_b,
            buy_token=token_a,
            sell_amount=2550 * 10**6,
            buy_amount=10**18,
            partially_fillable=True,
        )
    )

    # Third order: another WETH seller with tight price
    orders.append(
        create_order(
            sell_token=token_a,
            buy_token=token_b,
            sell_amount=5 * 10**17,  # 0.5 WETH
            buy_amount=1225 * 10**6,  # Ask price 2450 USDC/WETH
            partially_fillable=True,
        )
    )

    liquidity = [create_v2_liquidity(token_a, token_b, 20000 * 10**18, 50_000_000 * 10**6)]

    return {
        "id": f"tight_spread_{seed or random.randint(0, 9999)}",
        "tokens": {
            token_a: {
                **TOKENS[token_a],
                "availableBalance": "2000000000000000000",
                "trusted": True,
            },
            token_b: {**TOKENS[token_b], "availableBalance": "6000000000", "trusted": True},
        },
        "orders": orders,
        "liquidity": liquidity,
        "effectiveGasPrice": "30000000000",
        "deadline": "2106-01-01T00:00:00.000Z",
        "surplusCapturingJitOrderOwners": [],
    }


def generate_mixed_fillability_auction(seed: int | None = None) -> dict:
    """Generate auction with mixed fill-or-kill and partially fillable orders.

    Orders have prices that allow matching at AMM price (2500 USDC/WETH):
    - WETH sellers ask for < 2500 (accept AMM price or better)
    - USDC sellers bid > 2500 (willing to pay AMM price or less)
    """
    if seed is not None:
        random.seed(seed)

    token_a = WETH
    token_b = USDC

    orders = []

    # FOK seller of WETH: sell 1 WETH, want at least 2400 USDC (ask < AMM)
    orders.append(
        create_order(
            sell_token=token_a,
            buy_token=token_b,
            sell_amount=10**18,
            buy_amount=2400 * 10**6,  # Ask price 2400 USDC/WETH
            partially_fillable=False,
        )
    )

    # Partially fillable seller of WETH: sell 2 WETH, want at least 4800 USDC
    orders.append(
        create_order(
            sell_token=token_a,
            buy_token=token_b,
            sell_amount=2 * 10**18,
            buy_amount=4800 * 10**6,  # Ask price 2400 USDC/WETH
            partially_fillable=True,
        )
    )

    # FOK seller of USDC: sell 5200 USDC, want 2 WETH (bid > AMM)
    # bid price = 5200/2 = 2600 USDC/WETH
    orders.append(
        create_order(
            sell_token=token_b,
            buy_token=token_a,
            sell_amount=5200 * 10**6,
            buy_amount=2 * 10**18,
            partially_fillable=False,
        )
    )

    # Partially fillable seller of USDC: sell 2600 USDC, want 1 WETH (bid > AMM)
    # bid price = 2600 USDC/WETH
    orders.append(
        create_order(
            sell_token=token_b,
            buy_token=token_a,
            sell_amount=2600 * 10**6,
            buy_amount=1 * 10**18,
            partially_fillable=True,
        )
    )

    liquidity = [create_v2_liquidity(token_a, token_b, 20000 * 10**18, 50_000_000 * 10**6)]

    return {
        "id": f"mixed_fillability_{seed or random.randint(0, 9999)}",
        "tokens": {
            token_a: {
                **TOKENS[token_a],
                "availableBalance": "5000000000000000000",
                "trusted": True,
            },
            token_b: {**TOKENS[token_b], "availableBalance": "10000000000", "trusted": True},
        },
        "orders": orders,
        "liquidity": liquidity,
        "effectiveGasPrice": "30000000000",
        "deadline": "2106-01-01T00:00:00.000Z",
        "surplusCapturingJitOrderOwners": [],
    }


def main() -> int:
    """Generate fixtures."""
    parser = argparse.ArgumentParser(
        description="Generate N>2 order CoW test fixtures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/auctions/n_order_cow"),
        help="Output directory for fixtures",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of random N-order auctions to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Generating fixtures in {args.output}")

    # Generate fixed scenario fixtures
    fixtures = [
        ("tight_spread_3order", generate_tight_spread_auction(seed=1)),
        ("mixed_fillability_4order", generate_mixed_fillability_auction(seed=2)),
    ]

    # Generate random N-order auctions
    for i in range(args.count):
        num_orders = random.Random(args.seed + i).randint(3, 5)
        auction = generate_n_order_auction(
            num_orders=num_orders,
            seed=args.seed + i + 100,
            include_fok=(i % 2 == 0),  # Alternate with/without FOK
        )
        fixtures.append((f"n_order_{num_orders}_{i:02d}", auction))

    # Write fixtures
    for name, auction in fixtures:
        path = args.output / f"{name}.json"
        with open(path, "w") as f:
            json.dump(auction, f, indent=2)
        print(f"  Created {path.name} ({len(auction['orders'])} orders)")

    print(f"\nGenerated {len(fixtures)} fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
