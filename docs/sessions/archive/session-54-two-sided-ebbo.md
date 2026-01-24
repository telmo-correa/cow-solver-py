# Session 54: Two-Sided EBBO Validation

**Date:** 2025-01-24
**Focus:** Implement two-sided EBBO validation and extract helper

## Summary

Extended EBBO validation from one-sided (seller protection only) to two-sided
(protecting both sellers and buyers). Extracted duplicated EBBO bounds
calculation into a reusable helper module.

## Changes

### New Module: `solver/strategies/ebbo_bounds.py`

Created helper module with:
- `EBBOBounds` dataclass - holds `ebbo_min`, `ebbo_max`, and `amm_price`
- `get_ebbo_bounds()` function - computes two-sided EBBO bounds from AMM prices

### Core EBBO Logic: `solver/strategies/double_auction.py`

- Added `ebbo_min` and `ebbo_max` parameters to `run_double_auction()`
- EBBO bounds constrain the clearing price range upfront
- Added `ebbo_min` and `ebbo_max` to `run_hybrid_auction()` with AMM price validation

### Strategy Updates

All strategies now use two-sided EBBO via the helper:

| File | Lines Before | Lines After |
|------|--------------|-------------|
| `multi_pair.py` | 20 lines | 4 lines |
| `unified_cow.py` | 13 lines | 4 lines |
| `hybrid_cow.py` | 31 lines | 12 lines |

### Solver Safety Net: `solver/solver.py`

- Enhanced `_filter_ebbo_violations()` to log errors when violations are caught
- Violations at solver level indicate bugs in strategy-level validation

## EBBO Bounds Explanation

For a token pair A/B where price is expressed as "B per A":
- **ebbo_min**: Floor price for sellers (from A→B AMM quote)
- **ebbo_max**: Ceiling price for buyers (from inverse of B→A AMM quote)

```python
bounds = get_ebbo_bounds(token_a, token_b, router, auction)

result = run_hybrid_auction(
    group,
    amm_price=bounds.amm_price,
    ebbo_min=bounds.ebbo_min,
    ebbo_max=bounds.ebbo_max,
)
```

## Test Results

```
1026 passed, 14 skipped
```

New tests added:
- 10 tests for `ebbo_bounds.py` helper
- 7 tests for `run_double_auction` EBBO bounds
- 6 tests for `run_hybrid_auction` EBBO bounds
- 5 tests for solver-level EBBO filtering

## Files Changed

- `solver/strategies/ebbo_bounds.py` (new)
- `solver/strategies/double_auction.py`
- `solver/strategies/multi_pair.py`
- `solver/strategies/unified_cow.py`
- `solver/strategies/hybrid_cow.py`
- `solver/solver.py`
- `tests/unit/strategies/test_ebbo_bounds.py` (new)
- `tests/unit/strategies/test_double_auction.py`
- `tests/unit/strategies/test_hybrid_auction.py`
- `tests/unit/test_solver.py`
- `CLAUDE.md`
