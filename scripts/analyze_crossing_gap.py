#!/usr/bin/env python3
"""Deep analysis of why crossing pairs don't fully match.

Focus on pairs where ask <= bid (prices cross) but orders aren't matched.

Investigate:
1. Volume imbalance on each side
2. Multiple orders that could be combined
3. AMM integration opportunities

Usage:
    python scripts/analyze_crossing_gap.py [--limit N]
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.models.auction import AuctionInstance, Order
from solver.models.types import normalize_address
from solver.strategies.double_auction import get_limit_price


@dataclass
class CrossingPairAnalysis:
    """Analysis of a pair with crossing prices."""

    token_a: str
    token_b: str
    orders_a_to_b: list[Order] = field(default_factory=list)
    orders_b_to_a: list[Order] = field(default_factory=list)
    best_ask: Decimal | None = None
    best_bid: Decimal | None = None

    @property
    def total_orders(self) -> int:
        return len(self.orders_a_to_b) + len(self.orders_b_to_a)

    @property
    def volume_a_to_b(self) -> int:
        return sum(o.sell_amount_int for o in self.orders_a_to_b)

    @property
    def volume_b_to_a(self) -> int:
        return sum(o.sell_amount_int for o in self.orders_b_to_a)

    def compute_matchable_volume(self) -> dict:
        """Compute how much volume could theoretically match.

        Uses midpoint price and computes overlap.
        """
        if not self.orders_a_to_b or not self.orders_b_to_a:
            return {"matchable_a": 0, "matchable_b": 0, "reason": "one_side_empty"}

        if self.best_ask is None or self.best_bid is None:
            return {"matchable_a": 0, "matchable_b": 0, "reason": "no_prices"}

        if self.best_ask > self.best_bid:
            return {"matchable_a": 0, "matchable_b": 0, "reason": "no_crossing"}

        # Use midpoint price
        midpoint = (self.best_ask + self.best_bid) / 2

        # Volume of A that can be sold at midpoint
        matchable_a = 0
        for order in self.orders_a_to_b:
            limit = get_limit_price(order, is_selling_a=True)
            if limit is not None and limit <= midpoint:
                matchable_a += order.sell_amount_int

        # Volume of B that can be sold at midpoint (buyers of A)
        matchable_b = 0
        for order in self.orders_b_to_a:
            limit = get_limit_price(order, is_selling_a=False)
            if limit is not None and limit >= midpoint:
                matchable_b += order.sell_amount_int

        # At midpoint: A sellers give A, B sellers give B
        # Conservation: A given = A received, B given = B received
        # At price P (B/A): sell_a gives A, receives sell_a * P of B
        # B sellers give B, receive B / P of A
        # Match: min(matchable_a, matchable_b / P) of A can be exchanged

        # Effective match in A terms
        matchable_a_from_b = int(Decimal(matchable_b) / midpoint)
        actual_match_a = min(matchable_a, matchable_a_from_b)

        return {
            "matchable_a": matchable_a,
            "matchable_b": matchable_b,
            "midpoint": float(midpoint),
            "actual_match_a": actual_match_a,
            "limiting_side": "a_sellers" if matchable_a <= matchable_a_from_b else "b_sellers",
        }


def analyze_auction(auction: AuctionInstance) -> list[CrossingPairAnalysis]:
    """Find and analyze crossing pairs in an auction."""
    # Group orders by directed pair
    pair_orders: dict[tuple[str, str], list[Order]] = defaultdict(list)

    for order in auction.orders:
        sell = normalize_address(order.sell_token)
        buy = normalize_address(order.buy_token)
        pair_orders[(sell, buy)].append(order)

    # Find crossing pairs
    crossing_pairs = []
    seen: set[tuple[str, str]] = set()

    for (sell, buy), _orders in pair_orders.items():
        if (buy, sell) not in pair_orders:
            continue

        normalized = tuple(sorted([sell, buy]))
        if normalized in seen:
            continue
        seen.add(normalized)

        token_a, token_b = normalized
        orders_a_to_b = pair_orders.get((token_a, token_b), [])
        orders_b_to_a = pair_orders.get((token_b, token_a), [])

        # Compute best ask and bid
        best_ask = None
        for order in orders_a_to_b:
            price = get_limit_price(order, is_selling_a=True)
            if price is not None and (best_ask is None or price < best_ask):
                best_ask = price

        best_bid = None
        for order in orders_b_to_a:
            price = get_limit_price(order, is_selling_a=False)
            if price is not None and (best_bid is None or price > best_bid):
                best_bid = price

        # Check if prices cross
        if best_ask is not None and best_bid is not None and best_ask <= best_bid:
            crossing_pairs.append(
                CrossingPairAnalysis(
                    token_a=token_a,
                    token_b=token_b,
                    orders_a_to_b=orders_a_to_b,
                    orders_b_to_a=orders_b_to_a,
                    best_ask=best_ask,
                    best_bid=best_bid,
                )
            )

    return crossing_pairs


def load_auction(path: Path) -> AuctionInstance:
    """Load an auction from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return AuctionInstance.model_validate(data)


def main():
    parser = argparse.ArgumentParser(description="Analyze crossing pairs gap")
    parser.add_argument("--limit", type=int, default=None, help="Limit auctions")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data" / "historical_auctions"
    auction_files = sorted(data_dir.glob("mainnet_*.json"))

    if args.limit:
        auction_files = auction_files[: args.limit]

    print(f"Analyzing {len(auction_files)} auctions for crossing pairs...\n")

    # Aggregate
    all_crossing_pairs: list[CrossingPairAnalysis] = []
    total_orders_on_crossing = 0
    total_matchable_a = 0
    total_actual_match_a = 0

    # Track reasons for not matching
    limiting_by_a = 0
    limiting_by_b = 0
    single_order_pairs = 0
    multi_order_pairs = 0

    for auction_file in auction_files:
        auction = load_auction(auction_file)
        crossing = analyze_auction(auction)
        all_crossing_pairs.extend(crossing)

        for pair in crossing:
            total_orders_on_crossing += pair.total_orders

            result = pair.compute_matchable_volume()
            if "actual_match_a" in result:
                total_matchable_a += result.get("matchable_a", 0)
                total_actual_match_a += result.get("actual_match_a", 0)

                if result.get("limiting_side") == "a_sellers":
                    limiting_by_a += 1
                else:
                    limiting_by_b += 1

            if len(pair.orders_a_to_b) == 1 and len(pair.orders_b_to_a) == 1:
                single_order_pairs += 1
            else:
                multi_order_pairs += 1

    print("=" * 80)
    print("CROSSING PAIRS DEEP ANALYSIS")
    print("=" * 80)

    print(f"\nTotal crossing pairs: {len(all_crossing_pairs):,}")
    print(f"Total orders on crossing pairs: {total_orders_on_crossing:,}")

    print("\n" + "-" * 80)
    print("PAIR COMPOSITION")
    print("-" * 80)
    print(f"Single-order pairs (1v1): {single_order_pairs:,}")
    print(f"Multi-order pairs: {multi_order_pairs:,}")

    print("\n" + "-" * 80)
    print("VOLUME BALANCE")
    print("-" * 80)
    print(f"Limited by A-sellers (not enough sell pressure): {limiting_by_a:,}")
    print(f"Limited by B-sellers (not enough buy pressure): {limiting_by_b:,}")

    print("\n" + "-" * 80)
    print("SAMPLE CROSSING PAIRS")
    print("-" * 80)

    # Show some examples
    samples = sorted(all_crossing_pairs, key=lambda p: p.total_orders, reverse=True)[:10]

    for pair in samples:
        result = pair.compute_matchable_volume()
        spread = None
        if pair.best_ask and pair.best_bid:
            spread = (1 - float(pair.best_ask) / float(pair.best_bid)) * 100

        print(f"\nPair: {pair.token_a[-8:]} <-> {pair.token_b[-8:]}")
        print(f"  Orders: {len(pair.orders_a_to_b)} selling A, {len(pair.orders_b_to_a)} selling B")
        print(f"  Volume A→B: {pair.volume_a_to_b:,}")
        print(f"  Volume B→A: {pair.volume_b_to_a:,}")
        print(f"  Ask: {float(pair.best_ask):.8f}, Bid: {float(pair.best_bid):.8f}")
        if spread:
            print(f"  Spread: {spread:.2f}% (profit margin)")
        print(f"  Matchable at midpoint: {result.get('actual_match_a', 0):,} of A")
        print(f"  Limiting factor: {result.get('limiting_side', 'unknown')}")

    print("\n" + "-" * 80)
    print("KEY FINDINGS")
    print("-" * 80)

    print(f"""
CROSSING PAIRS ANALYSIS:
- {len(all_crossing_pairs):,} pairs have crossing prices (matchable)
- {total_orders_on_crossing:,} orders are on these pairs
- {single_order_pairs:,} are 1v1 pairs (simplest case)
- {multi_order_pairs:,} are multi-order (need aggregation)

CURRENT STRATEGY GAPS:
1. CowMatchStrategy only handles 2-order AUCTIONS (not pairs within auctions)
2. HybridCowStrategy processes pairs but:
   - Requires AMM price reference (may not exist)
   - Processes pairs independently (token overlap issues)
3. RingTradeStrategy requires exact cycles (rare)

WHY AREN'T THESE MATCHING?
- Many crossing pairs exist within larger auctions
- Current strategies don't aggregate orders on same pair well
- Need: Process ALL crossing pairs in auction, not just first match
""")


if __name__ == "__main__":
    main()
