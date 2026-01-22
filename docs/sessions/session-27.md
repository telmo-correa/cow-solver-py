# Session 27: Balancer Fixed-Point Math and Weighted Pool Math (Slices 3.2.1-3.2.2)

**Date:** 2026-01-22
**Focus:** Implement Balancer fixed-point math library and weighted pool swap calculations

## Summary

Implemented Slices 3.2.1-3.2.2 for Balancer/Curve integration:
- **Slice 3.2.1:** Fixed-point math library matching Balancer's on-chain LogExpMath.sol
- **Slice 3.2.2:** Weighted pool math with `calc_out_given_in` and `calc_in_given_out` functions

## Problem

### Slice 3.2.1: Fixed-Point Math
Balancer uses 18-decimal fixed-point arithmetic with a Taylor series approximation for power functions. Python's Decimal would produce mathematically accurate results that don't match on-chain behavior. We need to port the exact Rust/Solidity algorithm.

### Slice 3.2.2: Weighted Pool Math
Weighted pool math requires:
- Token scaling (different decimal precision)
- Fee application (subtract before swap / add after swap)
- Core swap formulas matching Rust implementation
- Ratio limits (30% max input/output)

## Solution

### Slice 3.2.1: Created `solver/math/fixed_point.py`

**Core functions matching Balancer's LogExpMath.sol:**
- `_ln()`: Natural logarithm using digit extraction and Taylor series for arctanh
- `_ln_36()`: Higher precision (36-decimal) for values close to 1.0
- `exp()`: Exponential with 12-term Taylor series
- `pow_raw()`: Power function via `exp(y * ln(x))`

**Bfp class (Balancer Fixed Point):**
- `mul_down()`, `mul_up()`: Multiply with rounding control
- `div_down()`, `div_up()`: Divide with rounding control
- `pow_down()`, `pow_up()`: Power with error margin adjustment
- `complement()`: Calculate `1 - x`

**Key implementation detail:**
Used `_div_trunc()` helper for truncation-toward-zero division to match Solidity/Rust behavior (Python's `//` floors toward -inf).

**Tests:** 78 tests with vectors generated from actual Balancer Solidity contract.

### Slice 3.2.2: Created `solver/amm/balancer.py`

**Data classes:**
- `WeightedTokenReserve`: Token reserve with balance, weight, and scaling factor
- `BalancerWeightedPool`: Pool with reserves, fee, version, and gas estimate

**Scaling helpers:**
- `scale_up()`: Convert native decimals to 18-decimal Bfp
- `scale_down_down()`: Convert back, rounding down (for outputs)
- `scale_down_up()`: Convert back, rounding up (for inputs)

**Fee helpers:**
- `subtract_swap_fee_amount()`: Deduct fee from input (sell orders)
- `add_swap_fee_amount()`: Add fee to calculated input (buy orders)

**Core math:**
- `calc_out_given_in()`: Weighted pool sell order formula
- `calc_in_given_out()`: Weighted pool buy order formula

### Key Implementation Details

**Rounding strategy (matching Rust):**
- Sell orders: Round intermediate divisions up, final multiply down → less output
- Buy orders: Round intermediate divisions up, final multiply up → more input required

**Formula for calc_out_given_in:**
```
amount_out = balance_out * (1 - (balance_in / (balance_in + amount_in))^(weight_in / weight_out))
```

**Formula for calc_in_given_out:**
```
amount_in = balance_in * ((balance_out / (balance_out - amount_out))^(weight_out / weight_in) - 1)
```

**Ratio limits:**
- MAX_IN_RATIO = 0.3 (30%) - Cannot swap more than 30% of input reserve
- MAX_OUT_RATIO = 0.3 (30%) - Cannot receive more than 30% of output reserve

### Test Results

24 tests covering:
- Basic Bfp operations (mul_down, mul_up, complement)
- Decimal scaling (up/down, rounding)
- Fee application (subtract/add)
- Core swap math (50/50 pool, Rust test vectors)
- Ratio limit enforcement
- Full integration flow (scale → fee → math → scale back)

**Rust test vectors:**
```
Test 1 (calc_out_given_in):
  balance_in: 100k, balance_out: 10
  weight_in: 0.0003, weight_out: 0.0007
  amount_in: 0.01
  Expected output: 428,571,297,950 wei ✓

Test 2 (calc_in_given_out):
  Same balances/weights
  amount_out: 0.01
  Expected input: 233,722,784,701,541,000,000 wei ✓
```

## Files Changed

| File | Change |
|------|--------|
| `solver/math/__init__.py` | NEW: Package init (empty) |
| `solver/math/fixed_point.py` | NEW: 18-decimal fixed-point math library |
| `tests/unit/test_fixed_point.py` | NEW: 78 tests for fixed-point math |
| `solver/amm/balancer.py` | NEW: Weighted pool dataclasses and math functions |
| `tests/unit/test_balancer.py` | NEW: 24 tests for weighted pool math |

## Key Decisions

1. **Underscore prefix for `_version` parameter**: V3Plus optimization not yet implemented, but parameter kept for API compatibility
2. **Frozen dataclasses**: `WeightedTokenReserve` and `BalancerWeightedPool` are immutable for safety
3. **Case-insensitive token lookup**: `get_reserve()` normalizes addresses to lowercase

## What's Next

- **Slice 3.2.3:** Stable pool math (Newton-Raphson invariant calculation)
- **Slice 3.2.4:** Pool parsing and registry integration
- **Slice 3.2.5:** AMM integration (BalancerWeightedAMM, BalancerStableAMM)

## Test Status

- 488 tests passing (386 existing + 78 fixed-point + 24 balancer)
- 14 skipped
- mypy: clean
- ruff: clean

## Commits

- `18302bd` - feat: add Balancer fixed-point math library (Slice 3.2.1)
- `c5c54a0` - feat: add Balancer weighted pool math (Slice 3.2.2)
