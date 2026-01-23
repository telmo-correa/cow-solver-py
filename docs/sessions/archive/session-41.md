# Session 41 - Solver Fee Feature and Code Review Fixes

**Date:** 2026-01-23

## Completed

- [x] Implemented `solver_fee` parameter for partial fill calculations
- [x] Fixed 4 major issues identified in code review
- [x] All 749 tests passing, all linter checks pass

## Key Changes

### 1. Solver Fee in Partial Fill Calculations

Added `solver_fee` parameter to `max_fill_sell_order()` and `max_fill_buy_order()` in `solver/amm/uniswap_v2.py`.

**Problem:** When calculating partial fills for limit orders, the solver fee must be included in the limit price validation. The naive formula `output / input >= limit_price` doesn't account for the fee deducted from executed amount.

**Solution:** Updated constraint formulas:
- **Sell orders:** `output * sell_amount >= buy_amount * (input + solver_fee)`
- **Buy orders:** `(input + solver_fee) * buy_amount <= sell_amount * output`

Uses binary search when the naive max fill doesn't satisfy the constraint with fee included.

### 2. Code Review Fixes

| Issue | File | Fix |
|-------|------|-----|
| `_p` naming convention | `solver/routing/router.py` | Changed `_p` to `p` with `# noqa: ARG005` comment explaining pool is ignored for Rust parity |
| O(n²) performance | `solver/solver.py` | Pre-index interactions by token pair in dict for O(1) lookup |
| Silent fallback | `solver/fees/price_estimation.py` | Changed `logger.debug` to `logger.warning` for price estimation fallback |
| Type narrowing | `solver/amm/balancer/amm.py` | Renamed closures, used explicit guard clause pattern |

### 3. Additional Cleanup

- Fixed `Callable` import to use `collections.abc` instead of `typing`
- Removed redundant imports and organized module-level imports
- Simplified `_should_combine_fills` return logic per SIM103 linter rule

## Files Modified

```
solver/amm/uniswap_v2.py           # Added solver_fee parameter to max_fill methods
solver/amm/balancer/amm.py         # Improved type narrowing in closures
solver/routing/router.py           # Fixed lambda parameter naming
solver/solver.py                   # Optimized interaction matching, cleaned imports
solver/fees/price_estimation.py    # Changed fallback log level to WARNING
tests/unit/amm/test_partial_fill_with_fee.py  # Enabled tests (removed skip)
```

## Test Results

```
749 passed, 14 skipped (RPC-dependent tests)
All ruff linter checks pass
```

## Technical Details

### Solver Fee Formula

The solver fee represents gas costs that are deducted from the user's executed amount. For limit orders, this affects the effective exchange rate:

```
Without fee: effective_rate = output / input
With fee:    effective_rate = output / (input + fee)
```

The partial fill calculation must find the maximum input where:
```
output / (input + fee) >= limit_price
```

This is implemented using binary search when the initial estimate (ignoring fee) doesn't satisfy the constraint.

### Performance Optimization

The `_build_separate_solutions` method previously used O(n²) nested loops to match interactions to orders. Now uses a pre-built dict index:

```python
# Before: O(n²)
for fill in fills:
    for interaction in interactions:
        if matches(fill, interaction): ...

# After: O(n)
interaction_index = {(token_in, token_out): [...]}
for fill in fills:
    key = (fill.sell_token, fill.buy_token)
    interaction = interaction_index.get(key, [])
```

## What's Next

- Continue Rust parity testing with remaining fixtures
- Multi-order optimization (Phase 4)
