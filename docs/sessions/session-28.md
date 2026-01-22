# Session 28: Stable Pool Math Bug Fix and Code Review

**Date:** 2026-01-22
**Focus:** Fix critical bug in stable pool invariant calculation and comprehensive code review

## Summary

Fixed a critical bug in `calculate_invariant()` where `A * n^n` was used instead of Balancer's convention of `A * n`. Also performed comprehensive code review and added missing validation and tests.

## Problem

### Bug: Wrong Invariant Formula

The `calculate_invariant()` function was using:
```python
a_times_n_pow_n = amp * n_coins**n_coins  # A * n^n = A * 27 for 3 tokens
```

But Balancer's StableMath.sol uses:
```python
amp_times_n = amp * n_coins  # A * n = A * 3 for 3 tokens
```

This caused the invariant to be ~85 tokens too high for the test case (2,646,242 vs 2,646,157), which propagated through to wrong swap calculations. The Rust test vectors were failing:
- `test_rust_vector_sell_dai_to_usdc`: Expected 9,999,475 USDC, got 0
- `test_rust_vector_buy_usdc_with_dai`: Expected ~10e18 DAI, got ~2646e18 DAI

### Code Review Findings

1. **Missing same-token validation**: `stable_calc_out_given_in` and `stable_calc_in_given_out` didn't check if `token_index_in == token_index_out`
2. **Missing negative index validation**: `get_token_balance_given_invariant_and_all_other_balances` only checked upper bound
3. **Incorrect docstring**: Docstring claimed `A * n^n` formula
4. **Redundant code**: `max(0, amount_out)` was unnecessary after the underflow check

## Solution

### Bug Fix

Changed line 547 from:
```python
n_pow_n = n_coins**n_coins
a_times_n_pow_n = amp * n_pow_n
```

To:
```python
amp_times_n = amp * n_coins
```

### Code Review Fixes

1. Added validation in `stable_calc_out_given_in()`:
   - Check `token_index_in` and `token_index_out` are in valid range
   - Check `token_index_in != token_index_out`

2. Added same validation in `stable_calc_in_given_out()`

3. Added negative index check in `get_token_balance_given_invariant_and_all_other_balances()`

4. Fixed docstring to describe Balancer's actual `A*n` convention

5. Removed redundant `max(0, ...)` call

6. Added 5 new tests for edge cases

## Files Changed

| File | Change |
|------|--------|
| `solver/amm/balancer.py` | Fixed invariant formula, added validations, fixed docstring |
| `tests/unit/test_balancer.py` | Added 5 edge case tests |

## Test Results

- **80 tests** in `test_balancer.py` (up from 75)
- **544 tests** total project-wide
- **14 skipped** (V3 quoter tests requiring RPC)
- **mypy**: Clean
- **ruff**: Clean

### Rust Test Vectors Now Pass

| Test | Input | Expected | Status |
|------|-------|----------|--------|
| Sell DAI→USDC (3-token) | 10 DAI | 9,999,475 USDC | PASS |
| Buy USDC with DAI (3-token) | 10 USDC | 10,000,524e18 DAI | PASS |
| Swap agEUR→EURe (2-token) | 10 agEUR | 10,029,862e18 EURe | PASS |

## Key Learnings

1. **Balancer uses `A*n`, not `A*n^n`**: The `n^n` factor is incorporated through the iterative d_p calculation, not in the amp_times_total variable.

2. **Float vs Integer debugging**: Comparing float approximation results with integer results helped identify where the discrepancy originated.

3. **Validation at boundaries**: Functions that accept indices should validate both positive overflow AND negative values (Python's negative indexing can cause subtle bugs).

## What's Next

- **Slice 3.2.4:** Pool parsing and registry integration
- **Slice 3.2.5:** AMM integration (BalancerWeightedAMM, BalancerStableAMM)

## Commits

- `ce88dc3` - feat: add Balancer stable pool math (Slice 3.2.3)
