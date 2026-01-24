# Constraint Enforcement Gaps Fix Plan

**Created:** 2026-01-24
**Status:** Pending

This plan addresses gaps identified in the constraint enforcement investigation across all strategies.

---

## Executive Summary

Investigation revealed gaps in how the four constraints (FOK, Limit Price, EBBO, Uniform Price) are enforced and tested across strategies. Key issues:

1. **EBBO testing gaps** - Multiple strategies have zero EBBO tests
2. **FOK not explicitly validated** - Strategies rely on external helpers without internal checks
3. **Test tolerance violations** - Some tests use float/tolerance instead of exact integer math
4. **AmmRouting uniform price broken** - Multi-order routing produces conflicting prices

---

## Priority 1: EBBO Testing Gaps (Critical)

### Issue 1.1: MultiPairCowStrategy Has No EBBO Tests

**File:** `tests/unit/strategies/test_multi_pair.py`
**Problem:** All tests use `liquidity=[]`, so EBBO validation never runs
**Risk:** EBBO violations could slip through undetected

**Fix:** Add dedicated EBBO test class with mock router returning reference prices

```python
class TestMultiPairEBBO:
    def test_ebbo_rejects_clearing_below_amm(self):
        """Pair rejected when clearing rate < AMM rate."""
        # Setup: AMM rate 2.5, orders would clear at 2.0
        # Expected: No fills (EBBO violated)

    def test_ebbo_accepts_clearing_above_amm(self):
        """Pair accepted when clearing rate >= AMM rate."""
        # Setup: AMM rate 2.0, orders clear at 2.5
        # Expected: Fills present

    def test_ebbo_two_sided_bounds(self):
        """Both ebbo_min and ebbo_max are enforced."""
        # Setup: Test seller protection (ebbo_min) and buyer protection (ebbo_max)

    def test_ebbo_skipped_when_no_liquidity(self):
        """EBBO check skipped when router returns None."""
        # Setup: Empty liquidity, router returns None
        # Expected: Match proceeds without EBBO constraint
```

**Acceptance Criteria:**
- [ ] 4+ EBBO tests added to test_multi_pair.py
- [ ] Tests use mock router with configurable reference prices
- [ ] Zero tolerance verified (exact boundary test)

---

### Issue 1.2: RingTradeStrategy Has No EBBO Tests

**File:** `tests/unit/strategies/test_ring_trade.py`
**Problem:** No tests verify EBBO rejection/acceptance for ring trades
**Risk:** Ring trades could execute below AMM rates

**Fix:** Add EBBO test class

```python
class TestRingTradeEBBO:
    def test_ring_rejected_when_clearing_below_amm(self):
        """Ring rejected when any leg violates EBBO."""
        # Setup: 3-token ring with one leg below AMM rate
        # Expected: Ring rejected, returns None

    def test_ring_accepted_when_clearing_above_amm(self):
        """Ring accepted when all legs satisfy EBBO."""
        # Setup: All legs at or above AMM rates
        # Expected: Ring accepted with fills

    def test_ebbo_checked_per_leg(self):
        """Each leg of ring is validated against its AMM rate."""
        # Setup: Different AMM rates for each pair
        # Expected: All legs validated independently

    def test_enforce_ebbo_flag_respected(self):
        """enforce_ebbo=False skips validation."""
        # Setup: Violation exists but enforce_ebbo=False
        # Expected: Ring proceeds (logged but not rejected)
```

**Acceptance Criteria:**
- [ ] 4+ EBBO tests added to test_ring_trade.py
- [ ] Tests cover multi-leg validation
- [ ] Tests verify enforce_ebbo flag behavior

---

### Issue 1.3: UnifiedCowStrategy EBBO Tests Are Mocked

**File:** `tests/unit/strategies/test_unified_cow.py`
**Problem:** Lines 120-190 mock `_verify_ebbo()` instead of testing actual calculation
**Risk:** EBBO formula correctness not verified by tests

**Fix:** Add end-to-end EBBO tests without mocking

```python
class TestUnifiedCowEBBOEndToEnd:
    def test_ebbo_formula_correctness(self):
        """Verify clearing_rate = price_sell / price_buy formula."""
        # Setup: Known prices and AMM rate
        # Calculate expected clearing rate manually
        # Verify strategy produces same result

    def test_ebbo_high_precision_boundary(self):
        """EBBO boundary at high precision (18 decimals)."""
        # Setup: Clearing rate 2.000000000000000001, AMM rate 2.0
        # Expected: Accepted (above AMM)

    def test_ebbo_rejects_at_boundary(self):
        """EBBO rejects when clearing rate exactly below AMM."""
        # Setup: Clearing rate 1.999999999999999999, AMM rate 2.0
        # Expected: Rejected (below AMM)

    def test_cycle_ebbo_validation(self):
        """EBBO validated for cycle settlements."""
        # Setup: 3-token cycle with EBBO bounds
        # Expected: _verify_cycle_ebbo() called and enforced
```

**Acceptance Criteria:**
- [ ] 4+ end-to-end EBBO tests (no mocking of _verify_ebbo)
- [ ] Tests verify actual formula calculation
- [ ] High-precision boundary tests included

---

## Priority 1: Test Tolerance Violations (Critical)

### Issue 1.4: RingTradeStrategy Uses Float Tolerance

**File:** `tests/unit/strategies/test_ring_trade.py`
**Lines:** 140-142, 605-629
**Problem:** Tests use `limit_rate * 0.999` (float with tolerance)

**Current (wrong):**
```python
limit_rate = int(order.buy_amount) / int(order.sell_amount)  # Float!
actual_rate = fill.buy_filled / fill.sell_filled              # Float!
assert actual_rate >= limit_rate * 0.999                      # Tolerance!
```

**Fix (correct):**
```python
# Exact integer cross-multiplication
assert fill.buy_filled * int(order.sell_amount) >= int(order.buy_amount) * fill.sell_filled
```

**Files to Fix:**
- `test_ring_trade.py:140-142` - Integration test limit check
- `test_ring_trade.py:605-629` - `test_fills_respect_limit_prices`

**Acceptance Criteria:**
- [ ] All limit price checks use integer cross-multiplication
- [ ] Zero float division in constraint validation
- [ ] Tests still pass with exact checks

---

### Issue 1.5: MultiPairCowStrategy Uses 1% Tolerance

**File:** `tests/unit/strategies/test_multi_pair.py`
**Line:** 950
**Problem:** `test_clearing_prices_match_fill_rates` uses 1% tolerance

**Current (wrong):**
```python
assert 0.99 <= ratio <= 1.01  # 1% tolerance
```

**Fix:** Use exact conservation check with integer math

```python
# Conservation: sell_filled * sell_price = buy_filled * buy_price
# Allow only integer truncation error (at most 1 unit)
sell_value = fill.sell_filled * int(prices[sell_token])
buy_value = fill.buy_filled * int(prices[buy_token])
assert abs(sell_value - buy_value) <= max(int(prices[sell_token]), int(prices[buy_token]))
```

**Acceptance Criteria:**
- [ ] No percentage tolerance in clearing price tests
- [ ] Truncation error bounded by price magnitude (not percentage)

---

## Priority 2: FOK Not Explicitly Validated

### Issue 2.1: UnifiedCowStrategy Missing FOK Validation

**File:** `solver/strategies/unified_cow.py`
**Problem:** No explicit FOK validation for pair matches or cycle settlements
**Risk:** Could partially fill FOK orders

**Fix:** Add FOK validation after settlement calculation

```python
def _validate_fok_fills(self, fills: list[OrderFill]) -> list[OrderFill]:
    """Filter fills that violate FOK constraint.

    FOK orders must be fully filled or not included at all.
    """
    valid_fills = []
    for fill in fills:
        order = fill.order
        if not order.partially_fillable:
            # FOK order - must be fully filled
            if order.is_sell_order:
                expected = int(order.sell_amount)
                actual = fill.sell_filled
            else:
                expected = int(order.buy_amount)
                actual = fill.buy_filled

            if actual < expected:
                logger.debug(
                    "unified_cow_fok_rejected",
                    order_uid=order.uid[:18],
                    expected=expected,
                    actual=actual,
                )
                continue  # Skip this fill

        valid_fills.append(fill)

    return valid_fills
```

**Call sites:**
- After `_solve_pair_lp()` returns fills
- After `calculate_cycle_settlement()` returns fills

**Acceptance Criteria:**
- [ ] `_validate_fok_fills()` method added
- [ ] Called in both pair and cycle paths
- [ ] Tests verify FOK rejection

---

### Issue 2.2: RingTradeStrategy Missing FOK Validation

**File:** `solver/strategies/ring_trade.py`
**Problem:** Assumes all orders are FOK but never validates fills

**Fix:** Add FOK validation in `_calculate_settlement()`

```python
def _calculate_settlement(self, result: CycleResult, graph: OrderGraph) -> RingTrade | None:
    settlement = calculate_cycle_settlement(result, graph)
    if settlement is None:
        return None

    # Validate FOK constraint
    for fill in settlement.fills:
        order = fill.order
        if not order.partially_fillable:
            if fill.sell_filled != int(order.sell_amount):
                logger.debug(
                    "ring_trade_fok_violated",
                    order_uid=order.uid[:18],
                    expected=int(order.sell_amount),
                    actual=fill.sell_filled,
                )
                return None

    return RingTrade(...)
```

**Acceptance Criteria:**
- [ ] FOK validation added to `_calculate_settlement()`
- [ ] Returns None if any FOK order not fully filled
- [ ] Tests verify FOK enforcement

---

## Priority 2: AmmRoutingStrategy Uniform Price

### Issue 2.3: Document AmmRouting Uniform Price Limitation

**File:** `solver/strategies/amm_routing.py`
**Problem:** Multi-order routing produces non-uniform prices by design
**Current behavior:** Last order's price overwrites previous (line 314)

**Fix Option A:** Document as known limitation (recommended)

Add to class docstring:
```python
"""AMM routing strategy for single-order optimization.

Note: Multi-order routing produces independent prices per order due to
AMM price impact. This means tokens may have different clearing prices
across orders, which technically violates the uniform price constraint.
For batches requiring uniform pricing, use CowMatchStrategy or
MultiPairCowStrategy instead.

This strategy is best used as a fallback for orders that can't be
matched peer-to-peer.
"""
```

**Fix Option B:** Reject conflicting prices (stricter)

```python
# In _solve_multiple_orders(), line 304-314:
for token, price in result.solution.prices.items():
    if token in all_prices and all_prices[token] != price:
        logger.warning("amm_routing_price_conflict", ...)
        continue  # Skip this order entirely instead of overwriting
```

**Recommendation:** Option A (document) - AmmRouting is intentionally for independent single-order routing.

**Acceptance Criteria:**
- [ ] Class docstring updated with uniform price note
- [ ] Behavior documented in strategies/README.md

---

## Priority 3: Additional Improvements

### Issue 3.1: RingTrade Price Conflict Detection

**File:** `solver/strategies/ring_trade.py`
**Problem:** `_combine_rings()` doesn't detect conflicting prices

**Fix:** Add conflict detection before merging

```python
def _combine_rings(self, rings: list[RingTrade]) -> StrategyResult:
    all_fills: list[OrderFill] = []
    all_prices: dict[str, str] = {}

    for ring in rings:
        result = ring.to_strategy_result()

        # Check for price conflicts
        conflict = False
        for token, price in result.prices.items():
            if token in all_prices and all_prices[token] != price:
                logger.debug(
                    "ring_trade_price_conflict",
                    token=token,
                    existing=all_prices[token],
                    new=price,
                )
                conflict = True
                break

        if conflict:
            continue  # Skip this ring

        all_fills.extend(result.fills)
        all_prices.update(result.prices)

    return StrategyResult(fills=all_fills, prices=all_prices, ...)
```

**Acceptance Criteria:**
- [ ] Price conflict detection added
- [ ] Conflicting rings skipped with logging
- [ ] Test verifies conflict handling

---

### Issue 3.2: Add Constraint Documentation to Strategy Docstrings

**Files:** All strategy files
**Problem:** Not all strategies document which constraints they enforce

**Fix:** Add constraint section to each strategy's class docstring

```python
"""[Strategy name and description]

Constraints enforced:
1. Fill-or-Kill: [how it's enforced]
2. Limit Price: [how it's enforced]
3. EBBO: [how it's enforced]
4. Uniform Price: [how it's enforced]
"""
```

**Strategies to update:**
- [ ] `unified_cow.py`
- [ ] `ring_trade.py`
- [ ] `multi_pair.py` (enhance existing)
- [ ] `amm_routing.py` (note limitation)

---

## Implementation Order

### Phase 1: Critical Test Fixes (P1)
| Task | File | Tests |
|------|------|-------|
| 1.4 Fix RingTrade float tolerance | test_ring_trade.py | Existing tests |
| 1.5 Fix MultiPair 1% tolerance | test_multi_pair.py | Existing tests |

### Phase 2: EBBO Test Coverage (P1)
| Task | File | Tests |
|------|------|-------|
| 1.1 MultiPair EBBO tests | test_multi_pair.py | 4 new |
| 1.2 RingTrade EBBO tests | test_ring_trade.py | 4 new |
| 1.3 UnifiedCow EBBO e2e tests | test_unified_cow.py | 4 new |

### Phase 3: FOK Validation (P2)
| Task | File | Tests |
|------|------|-------|
| 2.1 UnifiedCow FOK validation | unified_cow.py | 2 new |
| 2.2 RingTrade FOK validation | ring_trade.py | 2 new |

### Phase 4: Documentation & Cleanup (P2-P3)
| Task | File | Tests |
|------|------|-------|
| 2.3 AmmRouting documentation | amm_routing.py | - |
| 3.1 RingTrade price conflict | ring_trade.py | 1 new |
| 3.2 Strategy docstrings | All strategies | - |

---

## Verification

### After Each Phase

```bash
# Run affected tests
pytest tests/unit/strategies/test_multi_pair.py -v
pytest tests/unit/strategies/test_ring_trade.py -v
pytest tests/unit/strategies/test_unified_cow.py -v

# Run integration tests
pytest tests/integration/test_constraint_enforcement.py -v

# Full test suite
pytest tests/ --tb=short
```

### Final Verification

```bash
# All tests pass
pytest tests/ -q

# Type check
mypy solver/

# Lint
ruff check .
```

---

## Success Criteria

1. **Zero float tolerance** in constraint validation tests
2. **EBBO tests exist** for all strategies that implement EBBO
3. **FOK explicitly validated** in UnifiedCow and RingTrade
4. **All strategies document** their constraint enforcement
5. **All 1070+ tests pass**
6. **No new tolerance/epsilon** in financial comparisons

---

## Files to Modify

| File | Priority | Changes |
|------|----------|---------|
| `tests/unit/strategies/test_ring_trade.py` | P1 | Fix tolerances, add EBBO tests |
| `tests/unit/strategies/test_multi_pair.py` | P1 | Fix tolerance, add EBBO tests |
| `tests/unit/strategies/test_unified_cow.py` | P1 | Add EBBO e2e tests |
| `solver/strategies/unified_cow.py` | P2 | Add FOK validation |
| `solver/strategies/ring_trade.py` | P2 | Add FOK validation, price conflict |
| `solver/strategies/amm_routing.py` | P2 | Document limitation |
| `solver/strategies/multi_pair.py` | P3 | Enhance docstring |
| `solver/strategies/README.md` | P3 | Update constraint notes |
