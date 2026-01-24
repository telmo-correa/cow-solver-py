# Session 49: Slice 4.4 - Ring Trade Detection

**Date:** 2026-01-23

## Goals
- Implement ring trade detection algorithm
- Write comprehensive unit tests
- Write integration tests
- Code review and fixes
- Historical auction tests
- Verify all existing tests still pass

## Completed

### 1. Ring Trade Implementation (`solver/strategies/ring_trade.py`)

Created a complete ring trade detection and execution module with:

**OrderGraph Class**
- Builds directed graph from orders indexed by token pair
- `from_orders()` classmethod to construct from auction
- `find_3_cycles()` - detects 3-node cycles (A→B→C→A)
- `find_4_cycles()` - detects 4-node cycles with configurable limit
- Cached `tokens` property for efficiency

**CycleViability Dataclass**
- Tracks whether a cycle is viable (product <= 1)
- Tracks surplus ratio for ring profitability
- `near_viable` property for cycles with product in (1.0, 1.053]

**Viability Functions**
- `check_cycle_viability()` - calculates product of exchange rates
- `find_viable_cycle_direction()` - tries all rotations/directions, selects best

**RingTrade Dataclass**
- Stores cycle tokens, orders, fills, clearing prices, surplus
- `token_set` property for overlap detection
- `to_strategy_result()` method for solver integration

**RingTradeStrategy Class**
- Implements `try_solve()` for the SolutionStrategy interface
- Finds 3-node and 4-node cycles
- Selects non-overlapping rings by order AND token (prevents clearing price conflicts)
- Returns combined StrategyResult with zero interactions

### 2. Code Review Fixes (2 rounds)

**Round 1 - Critical Fixes:**
- Fixed viability condition: product <= 1 is viable (not >= 1)
- Fixed near-viable comparison: select lower product (closer to viability)
- Added token overlap prevention in `_select_rings()` (prevents clearing price conflicts)
- Added `SETTLEMENT_TOLERANCE = 0.001` constant for consistent tolerances
- Added token caching in `OrderGraph.tokens` property
- Fixed order counting efficiency (use `order_uids` set instead of iterating)

**Round 2 - Cleanup:**
- Removed leftover refactoring comment
- Simplified cache assignment (removed unnecessary `object.__setattr__`)

### 3. Unit Tests (`tests/unit/strategies/test_ring_trade.py`)

27 comprehensive unit tests covering:
- OrderGraph construction and queries
- Cycle detection (3-node and 4-node)
- CycleViability properties
- Viability checking with various rate scenarios
- Direction finding for sorted tuples
- RingTrade settlement calculation
- RingTradeStrategy try_solve behavior
- Non-overlapping ring selection
- Token conservation in settlements

### 4. Integration Tests (`tests/integration/test_ring_trade.py`)

9 integration tests covering:
- Viable 3-order ring trades
- Profitable rings (product < 1, surplus exists)
- Non-viable rings returning None
- 4-order ring trades
- Multiple non-overlapping rings
- Loading from fixture files
- Edge cases (empty, single order, 2-order CoW)

### 5. Historical Auction Tests (`tests/integration/test_ring_trade_historical.py`)

29 tests on real auction data covering:
- **TestRingTradeHistorical** (14 tests)
  - Parametrized tests on 12 n_order_cow fixtures
  - Settlement validity verification (limit prices respected)
  - No duplicate order fills
- **TestNearViableTracking** (2 tests)
  - Near-viable cycle detection
  - Near-viable product range validation (1.0 to 1.053)
- **TestCycleDetectionOnRealData** (2 tests)
  - Cycle detection finds cycles in multi-order auctions
  - Viability check returns valid product values
- **TestRingTradeMatchRate** (1 test)
  - Reports match statistics
- **TestBenchmarkComparison** (3 tests)
  - Ring trades have zero gas
  - Ring trades have no AMM interactions
  - Clearing prices are consistent

### 6. Exports Updated

Updated `solver/strategies/__init__.py` to export:
- `OrderGraph`
- `CycleViability`
- `RingTrade`
- `RingTradeStrategy`
- `find_viable_cycle_direction`

## Test Results

```
942 passed, 14 skipped in 0.69s
```

Ring trade specific tests: 65 (27 unit + 9 integration + 29 historical)

## Code Quality

- All ruff checks pass
- All mypy type checks pass
- Fixed linting issues:
  - Unused loop variables
  - Nested if statements → combined with `and`
  - `zip()` with `strict=True`
  - Proper tuple type hints for cycle return types
  - Float type annotation for price variable
  - Import sorting in test files

## Key Technical Details

| Concept | Implementation |
|---------|---------------|
| Ring viability | Product of rates <= 1 (surplus when < 1) |
| Near-viable | Product in (1.0, 1.053] for future AMM assistance |
| Token conservation | `buy_filled[i] = sell_filled[(i+1) % n]` |
| Order selection | Lowest rate (most generous) per edge |
| Ring selection | Greedy by surplus, no order OR token overlap |
| Settlement tolerance | 0.1% (`SETTLEMENT_TOLERANCE = 0.001`) |

## What's Next

1. **Integration with Solver** - Add RingTradeStrategy to the solver's strategy pipeline (Slice 4.5)
2. **AMM-Assisted Rings** - Future work to close near-viable cycles with AMM fills

## Files Changed

| File | Action |
|------|--------|
| `solver/strategies/ring_trade.py` | Created (~650 LOC) |
| `solver/strategies/__init__.py` | Updated exports |
| `tests/unit/strategies/test_ring_trade.py` | Created (27 tests) |
| `tests/integration/test_ring_trade.py` | Created (9 tests) |
| `tests/integration/test_ring_trade_historical.py` | Created (29 tests) |
