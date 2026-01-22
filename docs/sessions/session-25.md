# Session 25: SafeInt - Safe Arithmetic for Token Calculations

**Date:** 2026-01-22
**Focus:** Implement SafeInt wrapper for underflow/overflow protection

## Summary

Implemented a `SafeInt` wrapper class that provides safe arithmetic operations for token amount calculations. This addresses potential underflow, division-by-zero, and uint256 overflow issues identified in a comprehensive code analysis.

## Problem Analysis

The codebase performs arithmetic on large token amounts (uint256 values) with several risk areas:

1. **Division by zero** - Implicit protection through constraints, not explicit
2. **Subtraction underflow** - Fee subtraction from executed amounts
3. **uint256 overflow** - Values must fit when sent to Ethereum
4. **Implicit safety** - Protections scattered, hard to audit

## Solution: SafeInt Wrapper

Created `solver/safe_int.py` with a lightweight wrapper that makes arithmetic safe by default:

```python
from solver.safe_int import SafeInt, S

# Wrap at function entry
sell_a, buy_a = S(a.sell_a), S(a.buy_a)

# Natural arithmetic - automatically safe
result = (sell_a * buy_a) // other  # Raises DivisionByZero if other == 0
remainder = sell_a - buy_a          # Raises Underflow if buy_a > sell_a

# Unwrap at exit
return result.value
```

### Key Features

- **Safe subtraction**: Raises `Underflow` on negative result
- **Safe division**: Raises `DivisionByZero` on zero divisor
- **True division blocked**: `/` raises `TypeError` directing users to `//` or `ceiling_div()`
- **uint256 validation**: `to_uint256()` validates bounds
- **Natural syntax**: Operator overloading (`+`, `-`, `*`, `//`, `%`)
- **Named operations**: `ceiling_div()`, `min()`, `max()`, `saturating_sub()`
- **Checked variants**: `checked_sub()`, `checked_div()` return `None` instead of raising

### Exception Hierarchy

```
ArithmeticError
└── SafeIntError (base class for all SafeInt errors)
    ├── DivisionByZero
    ├── Underflow
    └── Uint256Overflow
```

## Implementation Details

### Initial Implementation

Created the SafeInt class with:
- Operator overloading for `+`, `-`, `*`, `//`, `%`
- Comparison operators
- Named operations (`ceiling_div`, `min`, `max`, `clamp`, `saturating_sub`)
- Checked variants (`checked_sub`, `checked_div`, `checked_ceiling_div`)
- uint256 validation (`to_uint256`, `is_uint256`)

### Code Review Fixes

After initial implementation, a code review identified several issues that were fixed:

1. **Consistent SafeInt usage**: Applied SafeInt to all partial match functions in `matching_rules.py`:
   - `_partial_sell_sell` (already had it)
   - `_partial_sell_buy` (added)
   - `_partial_buy_sell` (added)
   - `_partial_buy_buy` (added)

2. **Added `__truediv__` and `__rtruediv__`**: Raises `TypeError` with clear message directing users to `//` or `ceiling_div()`, preventing accidental float division

3. **Improved `ceiling_div`**: Added guard for negative divisors that raises `ValueError`, since the formula `(self + other - 1) // other` doesn't work correctly for negative values

4. **Updated `checked_ceiling_div`**: Now returns `None` for non-positive divisors (not just zero)

5. **Changed logger level**: In `base.py`, changed `logger.error` to `logger.critical` for fee subtraction underflow, since this indicates a validation bug

6. **Fixed return type**: Changed `__truediv__` and `__rtruediv__` return type from `SafeInt` to `NoReturn` for semantic accuracy

## Files Changed

### New Files
| File | Lines | Purpose |
|------|-------|---------|
| `solver/safe_int.py` | 380 | SafeInt class with safe arithmetic |
| `tests/unit/test_safe_int.py` | 340 | 74 comprehensive tests |

### Modified Files
| File | Change |
|------|--------|
| `solver/strategies/matching_rules.py` | Use SafeInt in all partial match functions |
| `solver/strategies/base.py` | Use SafeInt for fee subtraction, logger.critical for bugs |
| `solver/fees/calculator.py` | Use SafeInt for fee calculation, validate uint256 |

## Design Decisions

1. **Wrapper approach over domain types**: Minimizes refactoring while providing safety
2. **Exceptions over None**: Matches Python conventions, clearer error messages
3. **Checked variants available**: `checked_sub()`, `checked_div()` for None-based flow
4. **Keep AMM math as-is**: `uniswap_v2.py` already has comprehensive early-return guards
5. **Block true division**: Prevents accidental float results with clear error message
6. **Positive-only ceiling_div**: Formula doesn't work for negative divisors, so reject them

## Why Not Full Domain Types?

Considered wrapping all integers in `PositiveInt`, `NonZeroInt`, etc., but:
- Adds noise to complex AMM formulas
- Existing AMM code has comprehensive early-return guards
- SafeInt provides safety at point of use with less overhead

## Test Results

```
386 passed, 14 skipped
```

- 74 SafeInt tests covering all operations and edge cases
- All existing tests pass unchanged
- ruff and mypy clean

## What's Protected Now

| Operation | Location | Protection |
|-----------|----------|------------|
| Fee - Executed | `base.py:269` | `S(executed) - S(fee)` raises Underflow |
| cow_y = (cow_x * sell_b) // buy_b | `matching_rules.py:283` | Raises DivisionByZero |
| ceiling_div for limits | `matching_rules.py:286,297` | `.ceiling_div()` raises DivisionByZero |
| Fee calculation | `calculator.py:166` | `S(gas) * S(price)` + uint256 validation |
| Accidental float division | All SafeInt uses | `/` raises TypeError |

## What Remains Unchanged

`uniswap_v2.py` was not modified because:
- All divisions are protected by explicit early-return guards
- Adding SafeInt would add overhead without improving safety
- The existing pattern (check preconditions → compute) is clear and auditable

## Usage Guidelines

For new code:
```python
from solver.safe_int import SafeInt, S

def calculate_something(a: int, b: int, c: int) -> int:
    # Wrap inputs
    sa, sb, sc = S(a), S(b), S(c)

    # Safe arithmetic - use // not /
    result = (sa * sb) // sc          # Raises DivisionByZero if sc == 0
    diff = sa - sb                     # Raises Underflow if sb > sa
    rounded_up = sa.ceiling_div(sc)   # Raises if sc <= 0

    # Validate fits in uint256 if needed for chain
    return result.to_uint256()
```

## Commits

1. `5e6a7c8` - feat: add SafeInt wrapper for safe arithmetic on token amounts
2. `f2377bc` - fix: address code review issues for SafeInt implementation
3. (pending) - fix: use NoReturn type for __truediv__ methods

## What's Next

Per `PLAN.md`, options include:
- **Slice 3.2**: Balancer/Curve Integration
- Continue with code quality improvements from the plan file
