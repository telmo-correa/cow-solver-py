# Session 13: AMM Partial Fill Support

**Date:** 2026-01-21
**Focus:** Optimal partial fill calculation for AMM routing

## Summary

Added partial fill support for AMM routing, enabling the Python solver to execute partially fillable orders when full fill is impossible due to slippage. The implementation calculates the **exact mathematical maximum** partial fill that satisfies the order's limit price, outperforming Rust's binary search approach.

## What Was Done

### 1. AMM Partial Fill Calculation (`solver/amm/uniswap_v2.py`)

Added two new methods to UniswapV2:

- **`max_fill_sell_order()`** - Calculates maximum input for a sell order
- **`max_fill_buy_order()`** - Calculates maximum output for a buy order

Formula derivation for sell orders:
```
output(x) = (x * fee * R_out) / (R_in * 1000 + x * fee)
Constraint: output(x) / x >= buy_amount / sell_amount
Solving: x <= (R_out * sell_amount * fee - R_in * 1000 * buy_amount) / (buy_amount * fee)
```

Uses binary search verification to handle integer rounding edge cases efficiently (O(log n)).

### 2. Router Integration (`solver/routing/router.py`)

- **`_try_partial_sell_order()`** - Attempts partial fill when full sell fails
- **`_try_partial_buy_order()`** - Attempts partial fill when full buy fails

Partial fill is only attempted when `order.partially_fillable = true` and the full fill would violate the limit price.

### 3. Strategy Layer (`solver/strategies/amm_routing.py`)

Updated `AmmRoutingStrategy` to:
- Handle partial fills from the router
- Create remainder orders for unfilled portions
- Return remainder orders in StrategyResult for potential composition

### 4. OrderFill Enhancement (`solver/strategies/base.py`)

Added `fill_ratio` property to track what percentage of an order was filled (0.0-1.0).

### 5. Benchmark Script Improvement (`scripts/run_benchmarks.py`)

Enhanced comparison logic to detect **improvements** vs regressions:
- Matches (✓): Solutions identical
- Improvements (▲): Python fills more while respecting limits
- Regressions (▼): Python fills less (none found)

Exit code is now 0 only if no regressions.

### 6. Benchmark Fixtures

Created shared fixtures for Python vs Rust comparison:
- `benchmark/partial_fill_sell.json` - Sell order, ~38.7% fill
- `benchmark/partial_fill_buy.json` - Buy order, ~35.6% fill

## Benchmark Results

| Fixture | Python Fill | Rust Fill | Improvement |
|---------|-------------|-----------|-------------|
| partial_fill_sell | 38.7% | 25.0% | +54.6% |
| partial_fill_buy | 35.6% | 25.0% | +42.3% |

Python fills **~40-55% more** than Rust because:
- **Rust**: Binary search with power-of-2 fractions (1, 1/2, 1/4, 1/8...)
- **Python**: Exact mathematical maximum calculation

Both implementations use identical AMM formulas (verified by code comparison).

## Test Results

```
157 tests passing (was 147)
+10 new tests for partial fill calculation
```

All existing tests continue to pass.

## Files Changed

| File | Changes |
|------|---------|
| `solver/amm/uniswap_v2.py` | +133 lines (partial fill methods) |
| `solver/routing/router.py` | +216 lines (partial fill routing) |
| `solver/strategies/amm_routing.py` | +49/-23 lines (remainder handling) |
| `solver/strategies/base.py` | +14 lines (fill_ratio property) |
| `tests/unit/test_amm.py` | +191 lines (partial fill tests) |
| `scripts/run_benchmarks.py` | +213 lines (improvement detection) |
| `tests/fixtures/auctions/benchmark/partial_fill_sell.json` | New fixture |
| `tests/fixtures/auctions/benchmark/partial_fill_buy.json` | New fixture |

## Algorithm Comparison: Python vs Rust

### Rust Approach (Binary Search)
```rust
let divisor = U256::ONE << i;  // 1, 2, 4, 8, 16
sell.amount / divisor           // Tries 100%, 50%, 25%, 12.5%...
```
Stops at first successful fraction. With `max_partial_attempts=5`, can only achieve: 100%, 50%, 25%, 12.5%, 6.25%.

### Python Approach (Exact Calculation)
```python
max_input = (R_out * sell * fee - R_in * 1000 * buy) / (buy * fee)
# Binary search verification for integer rounding
```
Calculates the exact maximum that satisfies the limit price.

## What's Next

- Phase 2 Slice 2.3: Multi-order CoW detection
- Consider partial CoW + partial AMM composition
