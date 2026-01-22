# Session 33 - Integration Tests for Balancer Pools (Slice 3.2.7)

**Date:** 2026-01-22

## Summary

Completed Slice 3.2.7: Integration tests that verify Python solver output matches Rust baseline exactly for Balancer weighted and stable pools. Fixed critical bugs in fixed-point arithmetic and amplification parameter scaling discovered during test development.

## Completed

### Integration Tests Created
- [x] `test_weighted_pool_solve_gno_to_cow` - V0 weighted pool (GNO→COW)
- [x] `test_weighted_pool_solve_v3plus` - V3Plus weighted pool (xGNO→xCOW)
- [x] `test_stable_pool_sell_order_dai_to_usdc` - 3-token stable pool sell
- [x] `test_stable_pool_buy_order_dai_to_usdc` - 3-token stable pool buy
- [x] `test_composable_stable_pool_with_bpt` - Composable stable (agEUR→EURe)
- [x] `test_solver_selects_weighted_over_v2_when_better` - Best pool selection
- [x] `test_solver_selects_stable_over_v2_when_better` - Best pool selection

### Critical Bug Fixes

#### 1. `pow_raw` Truncation Division
Python's `//` rounds toward -∞ for negative numbers, but Rust/Solidity truncate toward zero.
```python
# Before (wrong for negative numbers)
logx_times_y = (ln_36_x // ONE_18) * y + ((ln_36_x % ONE_18) * y) // ONE_18
logx_times_y //= ONE_18

# After (correct)
div1 = _div_trunc(ln_36_x, ONE_18)
rem1 = ln_36_x - div1 * ONE_18
logx_times_y = div1 * y + _div_trunc(rem1 * y, ONE_18)
logx_times_y = _div_trunc(logx_times_y, ONE_18)
```

#### 2. Stable Pool Amplification Parameter Scaling
JSON provides raw A value (e.g., 5000.0), but stable math expects A × AMP_PRECISION.
```python
# Before (wrong - ~1000x error)
amp=int(pool.amplification_parameter)

# After (correct)
amp=int(pool.amplification_parameter * AMP_PRECISION)
```

#### 3. V3Plus `pow_up_v3` Wiring
Added dispatch to optimized power function for V3Plus pools.
```python
power = base.pow_up_v3(exponent) if _version == "v3Plus" else base.pow_up(exponent)
```

### Code Review Fixes
- [x] Inlined `mul_up` in `pow_down/pow_up` (eliminated 2 temp Bfp objects per call)
- [x] Extracted `_parse_scaling_factor` helper (removed ~20 lines duplication)
- [x] Fixed mypy error in weight parsing (explicit type annotation)
- [x] Added amp scaling documentation to `BalancerStablePool` docstring

## Test Results

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Weighted V0 (GNO→COW) | 1657855325872947866705 | Match | ✅ |
| Weighted V3Plus (xGNO→xCOW) | 1663373703594405548696 | Match | ✅ |
| Stable sell (DAI→USDC) | 9999475 | Match | ✅ |
| Stable buy (DAI→USDC) | 10000524328839166557 | Match | ✅ |
| Composable stable (agEUR→EURe) | 10029862202766050434 | Match | ✅ |

**Total: 647 tests passing, 14 skipped**

## Files Modified

```
solver/math/fixed_point.py         +45   # Truncation fix, pow_up_v3, inlined mul_up
solver/amm/balancer.py             +40   # V3Plus dispatch, amp scaling, helper extraction
solver/solver.py                   +19   # Balancer AMM DI parameters
solver/strategies/amm_routing.py   +14   # Pass Balancer AMMs to router
tests/unit/test_balancer.py        +3    # Updated test for amp scaling
tests/unit/test_fixed_point.py     +6    # Updated mul_up test expectation
tests/integration/test_balancer_integration.py  (new file from previous session)
tests/fixtures/auctions/benchmark/  (fixtures from previous session)
```

## Key Insights

### Division Semantics
- Python `//`: floor division (rounds toward -∞)
- Rust `/` for integers: truncation (rounds toward 0)
- For positive numbers: identical
- For negative numbers: `_div_trunc` required

### Amplification Parameter Format
- JSON: Raw A value (e.g., `"amplificationParameter": "5000.0"`)
- Stable math: A × AMP_PRECISION (e.g., 5,000,000)
- AMM handles the conversion internally

## What's Next

- **Slice 3.2.7 Complete** - All integration tests pass with exact Rust match
- Phase 3 Balancer integration is now complete
- Consider Phase 4: Multi-order CoW detection

## Commits

- feat: add Balancer integration tests with precision fixes (Slice 3.2.7)
