# Session 44: Slice 4.2a - AMM Price Integration

**Date:** 2026-01-23
**Phase:** 4 (Unified Optimization)
**Slice:** 4.2a - AMM Price Integration

## Summary

Implemented the first part of the hybrid CoW+AMM strategy: AMM reference price queries and the hybrid auction algorithm that uses AMM prices to unlock more CoW matches.

## What Was Done

### 1. Updated PLAN.md

Revised Phase 4 based on learnings from Slice 4.1:
- Marked Slice 4.1 as complete
- Restructured 4.2 into sub-slices: 4.2a (AMM price), 4.2b (hybrid auction), 4.2c (strategy integration)
- Added key insight: "Most CoW pairs have crossing prices (ask > bid), meaning pure peer-to-peer matching rarely works"
- Made 4.3 a data-driven evaluation checkpoint
- Made ring trades and split routing conditional on 4.3 results

### 2. AMM Reference Price Query

Added `get_reference_price()` method to `SingleOrderRouter`:

```python
def get_reference_price(
    self,
    token_in: str,
    token_out: str,
    probe_amount: int | None = None,
) -> Decimal | None:
    """Get the reference market price for a token pair.

    Queries all available pools and returns the best price
    (highest output per input). Used as reference for CoW matching.
    """
```

**Features:**
- Queries all registered pools (V2, V3, Balancer weighted, stable)
- Returns best price across all venues
- Uses small probe amount (0.001 tokens) to minimize price impact
- Returns `Decimal` for precision
- Returns `None` if no liquidity exists

### 3. Hybrid CoW+AMM Auction

Added `run_hybrid_auction()` function that extends pure double auction:

```python
def run_hybrid_auction(
    group: OrderGroup,
    amm_price: Decimal | None = None,
    respect_fill_or_kill: bool = True,
) -> HybridAuctionResult:
    """Run hybrid CoW+AMM auction on an order group.

    Uses AMM reference price to determine which orders can match
    directly (CoW) vs which should route through AMM.
    """
```

**Algorithm:**
1. If no AMM price, fall back to pure double auction
2. With AMM price:
   - Filter asks where `limit <= AMM price` (willing to sell at or below AMM)
   - Filter bids where `limit >= AMM price` (willing to buy at or above AMM)
   - Match filtered orders at AMM price (fair reference)
   - Route remainders to AMM

**Key insight:** Using AMM as a "virtual participant" at market price allows orders with crossed limits to still benefit from CoW gas savings when they can both execute against the AMM.

### 4. New Data Structures

```python
@dataclass
class AMMRoute:
    """Order that should be routed through AMM."""
    order: Order
    amount: int  # Amount of sell token
    is_selling_a: bool

@dataclass
class HybridAuctionResult:
    """Result combining CoW matches and AMM routes."""
    cow_matches: list[DoubleAuctionMatch]
    amm_routes: list[AMMRoute]
    clearing_price: Decimal | None
    total_cow_a: int
    total_cow_b: int
```

## Files Created/Modified

| File | Change |
|------|--------|
| `PLAN.md` | Updated Phase 4 structure based on 4.1 learnings |
| `solver/routing/router.py` | Added `get_reference_price()` method |
| `solver/strategies/double_auction.py` | Added hybrid auction + data classes |
| `solver/strategies/__init__.py` | Exported new types |
| `tests/unit/routing/test_price_query.py` | Created - 8 tests for price queries |
| `tests/unit/strategies/test_hybrid_auction.py` | Created - 7 tests for hybrid auction |
| `tests/unit/strategies/test_double_auction.py` | Fixed `name=` → `_name=` typo |

## Test Results

```
827 passed, 14 skipped in 0.56s
```

- 8 new price query tests
- 7 new hybrid auction tests
- 2 fixed double auction tests

## Next Steps

**Slice 4.2b: Hybrid Double Auction (continued)**
- Integrate hybrid auction into `CowMatchStrategy`
- Handle multi-pair auctions (process top N pairs by order count)
- Fall back to AMM routing for single-direction pairs

**Slice 4.2c: Strategy Integration & Benchmarking**
- Benchmark hybrid strategy on historical auctions
- Measure surplus improvement vs pure-AMM baseline
- Document results and decide on 4.3 direction
