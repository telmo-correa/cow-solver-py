# Session 24: Fee Calculator Service

**Date:** 2026-01-22
**Focus:** Centralize fee handling with a dedicated service

## Summary

Implemented a dedicated fee calculator service to centralize all solver fee logic. This improves testability, maintainability, and provides explicit error handling for fee calculation failures.

## Problem Analysis

Fee handling was scattered across multiple files with several issues:
- Fee calculation logic in `StrategyResult._calculate_fee_for_order()`
- Silent failures (returning `None`) when auction data or reference prices were missing
- Inconsistent gas estimate defaults (`SwapResult` used 100k, constant was 60k)
- No explicit error types for debugging
- Case-sensitive token lookups causing failures

## Solution: Fee Calculator Service

Created a new `solver/fees/` module with:

### FeeConfig (`config.py`)
Centralized configuration for fee behavior:
```python
@dataclass(frozen=True)
class FeeConfig:
    swap_gas_cost: int = 60_000
    settlement_overhead: int = 106_391
    fee_base: int = 10**18
    reject_on_missing_reference_price: bool = True
    reject_on_fee_overflow: bool = True
```

### FeeResult (`result.py`)
Explicit success/error handling:
```python
@dataclass(frozen=True)
class FeeResult:
    fee: int | None
    error: FeeError | None = None
    error_detail: str | None = None

    @property
    def is_valid(self) -> bool
    @property
    def requires_fee(self) -> bool

    @classmethod
    def no_fee(cls) -> FeeResult
    @classmethod
    def with_fee(cls, amount: int) -> FeeResult
    @classmethod
    def with_error(cls, error: FeeError, detail: str) -> FeeResult
```

### FeeCalculator (`calculator.py`)
Protocol + default implementation:
```python
class FeeCalculator(Protocol):
    def calculate_solver_fee(
        self, order: Order, gas_estimate: int, auction: AuctionInstance | None
    ) -> FeeResult: ...

    def validate_fee_against_amount(
        self, fee: int, executed_amount: int, is_sell_order: bool
    ) -> FeeResult: ...

class DefaultFeeCalculator:
    # Implementation with case-insensitive token lookup
    # Configurable behavior via FeeConfig
```

## Changes Made

### New Files
| File | Lines | Purpose |
|------|-------|---------|
| `solver/fees/__init__.py` | 32 | Module exports |
| `solver/fees/config.py` | 38 | FeeConfig dataclass |
| `solver/fees/result.py` | 76 | FeeResult + FeeError |
| `solver/fees/calculator.py` | 258 | FeeCalculator service |
| `tests/unit/test_fee_calculator.py` | 215 | 24 unit tests |

### Modified Files
| File | Change |
|------|--------|
| `solver/strategies/base.py` | Refactored `build_solution()` to use FeeCalculator, removed `_calculate_fee_for_order()` |
| `solver/amm/base.py` | Fixed `SwapResult.gas_estimate` default to use `POOL_SWAP_GAS_COST` |
| `tests/unit/test_strategy_base.py` | Updated tests for stricter error handling |
| `tests/unit/test_cow_match.py` | Changed default order class to `market` |
| `tests/integration/test_uniswap_v3.py` | Updated to check `executed + fee = sell_amount` |

## Behavior Changes

| Scenario | Before | After |
|----------|--------|-------|
| Missing auction data | Trade created with no fee | Trade rejected with `MISSING_AUCTION` error |
| Missing reference price | Trade created with no fee | Trade rejected with `MISSING_REFERENCE_PRICE` error |
| Fee > executed amount | Trade skipped silently | Trade rejected with `FEE_EXCEEDS_AMOUNT` error |
| Token lookup | Case-sensitive | Case-insensitive |

The stricter behavior can be disabled via `FeeConfig` if needed.

## Test Results

```
312 passed, 14 skipped
```

- 24 new fee calculator tests
- All existing tests updated and passing
- All benchmark fixtures produce valid solutions

## Architecture

```
StrategyResult.build_solution()
    │
    ├─► FeeCalculator.calculate_solver_fee()
    │       ├─► Check order class (LIMIT vs MARKET)
    │       ├─► Get gas price from auction
    │       ├─► Get reference price (case-insensitive lookup)
    │       └─► Return FeeResult
    │
    └─► FeeCalculator.validate_fee_against_amount()
            ├─► Check fee <= executed (for sell orders)
            └─► Return FeeResult or error
```

## Key Design Decisions

1. **Protocol-based**: `FeeCalculator` is a Protocol for easy mocking in tests
2. **Frozen dataclasses**: `FeeConfig` and `FeeResult` are immutable
3. **Explicit errors**: `FeeError` enum with specific error types
4. **Case-insensitive**: Token lookup handles mixed-case addresses

## What's Next

Per `PLAN.md`, the next focus is:
- **Slice 3.2**: Balancer/Curve Integration
- Or continue with code quality improvements from the plan file

## Files Changed

```
solver/fees/__init__.py          (new)
solver/fees/config.py            (new)
solver/fees/result.py            (new)
solver/fees/calculator.py        (new)
solver/strategies/base.py        (modified)
solver/amm/base.py               (modified)
tests/unit/test_fee_calculator.py (new)
tests/unit/test_strategy_base.py  (modified)
tests/unit/test_cow_match.py      (modified)
tests/integration/test_uniswap_v3.py (modified)
```
