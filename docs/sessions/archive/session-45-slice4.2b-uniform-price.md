# Session 45: Slice 4.2b - Uniform Clearing Price & HybridCowStrategy

**Date:** 2026-01-23
**Phase:** 4 (Unified Optimization)
**Slice:** 4.2b - Hybrid Strategy Integration

## Summary

Implemented uniform clearing price algorithm in `run_double_auction` to satisfy CoW Protocol settlement constraints, fixed critical bugs discovered during code review, and created the `HybridCowStrategy` class that integrates hybrid CoW+AMM matching into the solver.

## What Was Done

### 1. Uniform Clearing Price Algorithm

CoW Protocol requires all matches to execute at the same clearing price for valid settlement (constraint: `prices[sell_token] * sell = prices[buy_token] * buy`).

Implemented a two-pass algorithm in `run_double_auction`:

**First Pass:** Simulate matching to determine which orders match and their limits
- Walk through sorted asks/bids
- Record limit prices only for orders that actually match
- Compute valid clearing price range: `[max(matched_ask_limits), min(matched_bid_limits)]`
- Select midpoint as uniform clearing price

**Second Pass:** Execute all matches at the computed uniform price
- Reset indices and re-match orders
- All matches use the same `clearing_price` value
- Integer truncation (`match_b = int(match_a * clearing_price)`) may cause limit violations
- Post-truncation validation ensures all executed matches satisfy limits

### 2. Bug Fixes

**Bug 1: Infinite Loop with Small Prices**
- In `test_usdc_weth_like_scenario`, tests hung indefinitely
- Root cause: When `match_b = int(match_a * clearing_price) = 0` due to small prices (WETH/USDC decimal mismatch), the loop never terminated
- Fix: Added check `if match_b <= 0: bid_idx += 1; continue`

**Bug 2: Non-matching Orders Influencing Price**
- Limit prices were recorded BEFORE validating that `match_a > 0` and `match_b > 0`
- This could cause orders that don't actually match to influence the clearing price
- Fix: Moved limit recording to AFTER validating positive match amounts

### 3. HybridCowStrategy Implementation

Created `solver/strategies/hybrid_cow.py` with the `HybridCowStrategy` class:

```python
class HybridCowStrategy:
    """Strategy that matches orders using hybrid CoW+AMM auction.

    For each token pair with orders in both directions:
    1. Query AMM for reference price
    2. Match orders at AMM price via double auction
    3. Route unmatched orders to AMM
    """
```

**Features:**
- Groups orders by token pair using `find_cow_opportunities()`
- Queries AMM reference prices via router
- Runs hybrid auction (uses AMM price if available, falls back to pure double auction)
- Converts matches to `OrderFill` objects
- Computes remainder orders for AMM routing
- Deduplicates fills when same order appears in multiple matches

### 4. Reverted Workaround

Removed the code in `HybridCowStrategy.try_solve()` that skipped pairs without AMM price. With the uniform clearing price implementation, pure double auction now works correctly as a fallback.

## Files Created/Modified

| File | Change |
|------|--------|
| `solver/strategies/double_auction.py` | Two-pass uniform clearing price, bug fixes |
| `solver/strategies/hybrid_cow.py` | Created - HybridCowStrategy class |
| `solver/strategies/__init__.py` | Export HybridCowStrategy |
| `solver/strategies/base.py` | Minor updates |
| `tests/unit/strategies/test_double_auction.py` | Added uniform clearing price tests |
| `tests/unit/strategies/test_hybrid_auction.py` | Additional hybrid auction tests |
| `tests/unit/strategies/test_hybrid_cow_strategy.py` | Created - HybridCowStrategy tests |

## Test Results

```
839 passed, 14 skipped
```

New tests added:
- `TestUniformClearingPrice.test_all_matches_have_same_clearing_price`
- `TestUniformClearingPrice.test_nonmatching_order_does_not_affect_clearing_price`
- Additional hybrid auction edge case tests

## Key Insights

1. **Settlement Constraint:** CoW Protocol's uniform clearing price requirement means all matches in a solution must execute at the same price. This is fundamentally different from traditional order books where each trade can have its own price.

2. **Two-Pass Necessity:** A two-pass algorithm is necessary because we need to know which orders will match before we can compute the uniform clearing price.

3. **Integer Truncation Hazard:** When computing `match_b = int(match_a * price)`, the truncation can cause the actual executed price to differ from the target clearing price, potentially violating limit orders.

## Next Steps

**Slice 4.2c: Strategy Integration & Benchmarking**
- Integrate `HybridCowStrategy` into the main solver
- Benchmark hybrid strategy on historical auctions
- Measure surplus improvement vs pure-AMM baseline
- Document results and decide on Slice 4.3 direction
