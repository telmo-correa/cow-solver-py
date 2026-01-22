# Session 15: Data-Driven Matching Rules

**Date:** 2026-01-21
**Focus:** Refactor CoW matching to use data-driven rule tables

## Summary

Implemented Approach 2 (Constraint Tables) from the matching refactoring analysis. This transforms the CoW matching logic from procedural code into declarative data structures, making it:
- Easier to audit and verify
- Self-documenting with constraint descriptions
- Testable in isolation
- Extensible for new order types

## Changes Made

### New File: `solver/strategies/matching_rules.py` (483 lines)

Created a comprehensive data-driven matching system:

**Data Structures:**
- `OrderAmounts` - NamedTuple for parsed order values
- `MatchType` - Enum for 4 order type combinations (SELL_SELL, SELL_BUY, BUY_SELL, BUY_BUY)
- `Constraint` - Frozen dataclass with description and check function
- `ExecutionAmounts` - Frozen dataclass for calculated fills
- `PerfectMatchRule` - Rule definition with constraints and execution formula
- `PartialMatchRule` - Rule definition with partial match logic

**Rule Tables:**
- `PERFECT_MATCH_RULES` - Dict mapping MatchType to PerfectMatchRule
- `PARTIAL_MATCH_RULES` - Dict mapping MatchType to PartialMatchRule

**Functions:**
- `evaluate_perfect_match()` - Check all constraints and compute execution
- `evaluate_partial_match()` - Compute partial fill with limit verification

### Simplified: `solver/strategies/cow_match.py` (703 → 267 lines)

**62% reduction in code size.** The strategy now:
1. Validates token pair compatibility
2. Delegates to `evaluate_perfect_match()` / `evaluate_partial_match()`
3. Checks fill-or-kill constraints
4. Builds the result

### New Tests: `tests/unit/test_matching_rules.py` (455 lines, 35 tests)

Comprehensive test coverage for:
- `TestMatchType` - Enum creation and rule coverage
- `TestConstraint` - Constraint satisfaction
- `TestPerfectMatchRules` - All 4 match types with limit checks
- `TestPartialMatchRules` - All 4 match types with remainder scenarios
- `TestExecutionAmounts` - Immutability and equality
- `TestRuleCompleteness` - Verify all rules have required components
- `TestEdgeCases` - Zero amounts, equal amounts, large amounts

## Test Results

```
202 tests passed in 0.24s
```

- Original tests: 167
- New matching_rules tests: 35

## Benchmark Results

**AMM Routing (9 fixtures):** All pass
- Average response time: ~1.2ms
- Multi-hop routing works correctly

**CoW Matching (6 fixtures):** All pass
- Perfect match scenarios
- Partial CoW + AMM remainder
- Fill-or-kill combinations

## Code Quality

```
ruff check . → All checks passed!
mypy solver/ → Success: no issues found in 20 source files
```

## Architecture Notes

The refactoring separates concerns cleanly:

```
matching_rules.py (Data Layer)
├── OrderAmounts      # Input data
├── MatchType         # Classification
├── Constraint        # Validation rules (data)
├── ExecutionAmounts  # Output data
├── PERFECT_MATCH_RULES  # Rule definitions
└── PARTIAL_MATCH_RULES  # Rule definitions

cow_match.py (Strategy Layer)
├── CowMatchStrategy  # Orchestration
├── validate_cow_pair # Token pair check
├── _get_order_amounts # Data extraction
├── _find_perfect_match # Delegates to rules
├── _find_partial_match # Delegates to rules
└── _build_result     # Solution construction
```

## What's Next

Per `PLAN.md`, the next slice is:

**Slice 2.3: Multi-Order CoW Detection**
- Build order flow graph (net demand per token pair)
- Identify netting opportunities
- Greedy matching algorithm

## Files Changed

| File | Lines | Change |
|------|-------|--------|
| `solver/strategies/matching_rules.py` | 483 | New |
| `solver/strategies/cow_match.py` | 267 | Simplified (was 703) |
| `tests/unit/test_matching_rules.py` | 455 | New |
