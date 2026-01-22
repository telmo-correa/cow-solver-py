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

## Files Changed

### New Files
| File | Lines | Purpose |
|------|-------|---------|
| `solver/safe_int.py` | 345 | SafeInt class with safe arithmetic |
| `tests/unit/test_safe_int.py` | 318 | 70 comprehensive tests |

### Modified Files
| File | Change |
|------|--------|
| `solver/strategies/matching_rules.py` | Use SafeInt in `_partial_sell_sell()` for division |
| `solver/strategies/base.py` | Use SafeInt for fee subtraction |
| `solver/fees/calculator.py` | Use SafeInt for fee calculation, validate uint256 |

## Design Decisions

1. **Wrapper approach over domain types**: Minimizes refactoring while providing safety
2. **Exceptions over None**: Matches Python conventions, clearer error messages
3. **Checked variants available**: `checked_sub()`, `checked_div()` for None-based flow
4. **Keep AMM math as-is**: `uniswap_v2.py` already has comprehensive early-return guards

## Why Not Full Domain Types?

Considered wrapping all integers in `PositiveInt`, `NonZeroInt`, etc., but:
- Adds noise to complex AMM formulas
- Existing AMM code has comprehensive early-return guards
- SafeInt provides safety at point of use with less overhead

## Test Results

```
382 passed, 14 skipped
```

- 70 new SafeInt tests covering all operations and edge cases
- All existing tests pass unchanged
- ruff and mypy clean

## What's Protected Now

| Operation | Location | Protection |
|-----------|----------|------------|
| Fee - Executed | `base.py:265` | `S(executed) - S(fee)` raises Underflow |
| cow_y = (cow_x * sell_b) // buy_b | `matching_rules.py:283` | Raises DivisionByZero |
| ceiling_div for limits | `matching_rules.py:286,291` | `.ceiling_div()` raises DivisionByZero |
| Fee calculation | `calculator.py:166` | `S(gas) * S(price)` + uint256 validation |

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

    # Safe arithmetic
    result = (sa * sb) // sc

    # Validate fits in uint256 if needed for chain
    return result.to_uint256()
```

## Files Changed

```
solver/safe_int.py               (new)
solver/strategies/matching_rules.py (modified)
solver/strategies/base.py        (modified)
solver/fees/calculator.py        (modified)
tests/unit/test_safe_int.py      (new)
```
