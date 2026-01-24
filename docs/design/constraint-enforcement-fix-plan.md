# Constraint Enforcement Fix Plan

**Created:** 2026-01-24
**Status:** Phase 2 Complete

This plan addresses all gaps identified in the constraint enforcement analysis across solver strategies.

## Progress

| Phase | Status | Commits |
|-------|--------|---------|
| 1.1 CowMatchStrategy EBBO | ✅ Complete | 685f290 |
| 1.2 AmmRoutingStrategy Limit Tests | ✅ Complete | e195c24 |
| 1.3 AmmRoutingStrategy Price Conflict | ✅ Complete | 987b6c0 |
| 2.1 UnifiedCowStrategy EBBO Fix | ✅ Complete | (this session) |
| 2.2 UnifiedCowStrategy Price Fix | ✅ Complete | (this session) |
| 2.3 UnifiedCowStrategy Tests | ✅ Complete | 17 tests |

---

## Overview

### The Four Constraints

| Constraint | Definition | Validation |
|------------|------------|------------|
| **Fill-or-Kill (FOK)** | `partially_fillable=false` orders must be fully filled or not at all | Check fill ratio = 1.0 |
| **Limit Price** | Actual rate ≥ limit rate for sell orders | `buy_filled/sell_filled >= buy_amount/sell_amount` |
| **EBBO** | Clearing rate ≥ AMM rate (zero tolerance) | Integer comparison, no tolerance |
| **Uniform Price** | All orders in a pair execute at same price | Single clearing price per pair |

### Priority Matrix

| Gap | Severity | Strategy | Risk |
|-----|----------|----------|------|
| CowMatchStrategy: No EBBO | HIGH | Production | EBBO violations in 2-order matches |
| AmmRoutingStrategy: No limit check | HIGH | Production | Limit violations in AMM routes |
| AmmRoutingStrategy: No price conflict | HIGH | Production | Non-uniform prices |
| UnifiedCowStrategy: EBBO formula bug | HIGH | Research | Wrong EBBO validation |
| UnifiedCowStrategy: Price normalization | MEDIUM | Research | Conservation violation |
| UnifiedCowStrategy: Zero tests | HIGH | Research | No verification |

---

## Phase 1: Production Strategy Fixes

### 1.1 CowMatchStrategy EBBO Validation

**File:** `solver/strategies/cow_match.py`

**Problem:** No EBBO check - only validates limit prices, not AMM rates.

**Fix:**

```python
# In CowMatchStrategy class, add router parameter
def __init__(self, router: SingleOrderRouter | None = None):
    self.router = router

# In _build_result(), before returning:
def _build_result(self, match: MatchResult, auction: AuctionInstance) -> StrategyResult:
    # ... existing code to build fills and prices ...

    # NEW: EBBO validation
    if self.router is not None:
        for fill in fills:
            order = fill.order
            sell_token = normalize_address(order.sell_token)
            buy_token = normalize_address(order.buy_token)

            # Get AMM rate
            amm_rate = self.router.get_reference_price(
                sell_token, buy_token, order.sell_amount_int, auction
            )

            if amm_rate is not None:
                # Clearing rate from prices
                sell_price = int(prices[sell_token])
                buy_price = int(prices[buy_token])

                # Use integer comparison (zero tolerance)
                # clearing_rate = sell_price / buy_price
                # EBBO: clearing_rate >= amm_rate
                # Integer: sell_price * amm_rate.denominator >= buy_price * amm_rate.numerator
                clearing_rate = Decimal(sell_price) / Decimal(buy_price)
                if clearing_rate < amm_rate:
                    logger.debug(
                        "cow_match_ebbo_violation",
                        clearing_rate=float(clearing_rate),
                        amm_rate=float(amm_rate),
                    )
                    return None  # Reject match

    return StrategyResult(fills=fills, prices=prices, ...)
```

**Tests to Add:** `tests/unit/strategies/test_cow_match.py`

```python
class TestCowMatchEBBO:
    def test_cow_match_rejects_when_clearing_below_amm(self):
        """CoW match rejected when clearing rate < AMM rate."""
        # Setup: two orders with clearing rate 2.0
        # AMM rate 2.1 (better than clearing)
        # Expected: match rejected

    def test_cow_match_accepts_when_clearing_above_amm(self):
        """CoW match accepted when clearing rate >= AMM rate."""
        # Setup: clearing rate 2.5, AMM rate 2.0
        # Expected: match accepted

    def test_cow_match_accepts_when_no_amm_liquidity(self):
        """CoW match accepted when no AMM price available."""
        # Setup: router returns None for price
        # Expected: match accepted (no EBBO constraint)

    def test_cow_match_ebbo_zero_tolerance(self):
        """EBBO uses zero tolerance (clearing must be >= AMM exactly)."""
        # Setup: clearing rate 1.999999, AMM rate 2.0
        # Expected: match rejected (even tiny deficit fails)
```

**Acceptance Criteria:**
- [ ] CowMatchStrategy accepts optional `router` parameter
- [ ] EBBO validation runs when router provided
- [ ] Uses integer/Decimal comparison (zero tolerance)
- [ ] Returns None if EBBO violated
- [ ] Logs rejection reason
- [ ] 4+ new tests pass

---

### 1.2 AmmRoutingStrategy Limit Price Validation

**File:** `solver/strategies/amm_routing.py`

**Problem:** Routes orders through AMM but doesn't verify result meets limit price.

**Fix:**

```python
# In _route_and_build(), after getting routing result:
def _route_and_build(self, order: Order, ...) -> RoutingBuildResult | None:
    result = self.router.route_order(order, pool_registry, auction)

    if result is None or not result.success:
        return None

    # NEW: Validate limit price
    fill = result.fill
    if not self._validate_limit_price(order, fill):
        logger.debug(
            "amm_routing_limit_violation",
            order_uid=order.uid[:16],
            sell_filled=fill.sell_filled,
            buy_filled=fill.buy_filled,
        )
        return None

    # ... continue with solution building ...

def _validate_limit_price(self, order: Order, fill: OrderFill) -> bool:
    """Validate fill satisfies order's limit price."""
    if fill.sell_filled == 0:
        return True  # No fill, no violation

    # Limit rate = buy_amount / sell_amount (what user wants minimum)
    # Actual rate = buy_filled / sell_filled (what user gets)
    # For sell order: actual >= limit
    # Use integer math to avoid rounding issues

    sell_amount = int(order.sell_amount)
    buy_amount = int(order.buy_amount)

    # actual_rate >= limit_rate
    # buy_filled / sell_filled >= buy_amount / sell_amount
    # buy_filled * sell_amount >= buy_amount * sell_filled
    return fill.buy_filled * sell_amount >= buy_amount * fill.sell_filled
```

**Tests to Add:** `tests/unit/strategies/test_amm_routing.py`

```python
class TestAmmRoutingLimitPrice:
    def test_amm_routing_rejects_below_limit(self):
        """AMM route rejected when fill rate < limit rate."""
        # Setup: order wants 2 B per A, AMM returns 1.9 B per A
        # Expected: route rejected

    def test_amm_routing_accepts_at_limit(self):
        """AMM route accepted when fill rate = limit rate."""
        # Setup: order wants 2 B per A, AMM returns exactly 2 B per A
        # Expected: route accepted

    def test_amm_routing_accepts_above_limit(self):
        """AMM route accepted when fill rate > limit rate."""
        # Setup: order wants 2 B per A, AMM returns 2.5 B per A
        # Expected: route accepted

    def test_amm_routing_partial_fill_respects_limit(self):
        """Partial AMM fill still respects limit price."""
        # Setup: partial fill with proportional limit
        # Expected: limit validated correctly
```

**Acceptance Criteria:**
- [ ] `_validate_limit_price()` method added
- [ ] Called after every successful route
- [ ] Uses integer comparison (no rounding errors)
- [ ] Returns None if limit violated
- [ ] 4+ new tests pass

---

### 1.3 AmmRoutingStrategy Price Conflict Detection

**File:** `solver/strategies/amm_routing.py`

**Problem:** Multiple orders routed independently may get different prices for same token.

**Fix:**

```python
# In try_solve(), track established prices:
def try_solve(self, auction: AuctionInstance) -> StrategyResult | None:
    all_fills = []
    all_prices: dict[str, str] = {}
    all_interactions = []

    for order in auction.orders:
        result = self._route_and_build(order, pool_registry, auction)

        if result is None:
            continue

        # NEW: Check for price conflicts
        conflict = self._check_price_conflict(result.solution.prices, all_prices)
        if conflict:
            logger.debug(
                "amm_routing_price_conflict",
                order_uid=order.uid[:16],
                token=conflict[0],
                existing_price=conflict[1],
                new_price=conflict[2],
            )
            continue  # Skip this order to maintain price consistency

        # Merge results
        all_fills.extend(result.fills)
        all_prices.update(result.solution.prices)
        all_interactions.extend(result.solution.interactions)

    # ... build final result ...

def _check_price_conflict(
    self,
    new_prices: dict[str, str],
    existing_prices: dict[str, str]
) -> tuple[str, str, str] | None:
    """Check if new prices conflict with existing prices.

    Returns (token, existing_price, new_price) if conflict, None otherwise.
    """
    for token, new_price in new_prices.items():
        token_norm = normalize_address(token)
        if token_norm in existing_prices:
            existing = existing_prices[token_norm]
            if existing != new_price:
                return (token_norm, existing, new_price)
    return None
```

**Tests to Add:** `tests/unit/strategies/test_amm_routing.py`

```python
class TestAmmRoutingPriceConflict:
    def test_amm_routing_detects_price_conflict(self):
        """Second order with different price for same token is skipped."""
        # Setup: order 1 gets WETH at price X, order 2 gets WETH at price Y
        # Expected: order 2 skipped, only order 1 in result

    def test_amm_routing_allows_consistent_prices(self):
        """Orders with same price for token are both included."""
        # Setup: order 1 and 2 both get WETH at same price
        # Expected: both orders in result

    def test_amm_routing_independent_tokens_no_conflict(self):
        """Orders with different tokens have no conflict."""
        # Setup: order 1 trades A/B, order 2 trades C/D
        # Expected: both orders in result
```

**Acceptance Criteria:**
- [ ] `_check_price_conflict()` method added
- [ ] Called before merging each routing result
- [ ] Skips orders that would create price conflict
- [ ] Logs conflict details
- [ ] 3+ new tests pass

---

## Phase 2: UnifiedCowStrategy Fixes

### 2.1 Fix EBBO Formula Bug

**File:** `solver/strategies/unified_cow.py`

**Problem:** Lines 720-755 use `price_buy / price_sell` instead of `sell_price / buy_price`.

**Fix:**

```python
# In _verify_ebbo() method (around line 720):
def _verify_ebbo(
    self,
    fills: list[OrderFill],
    prices: dict[str, int],
    router: SingleOrderRouter,
    auction: AuctionInstance,
) -> bool:
    """Verify all fills satisfy EBBO constraints."""
    for fill in fills:
        order = fill.order
        sell_token = normalize_address(order.sell_token)
        buy_token = normalize_address(order.buy_token)

        # Get prices
        sell_price = prices.get(sell_token)
        buy_price = prices.get(buy_token)

        if sell_price is None or buy_price is None:
            continue

        # FIXED: Clearing rate formula
        # User sells sell_token, gets buy_token
        # Rate user gets = buy_filled / sell_filled = sell_price / buy_price
        # (because: sell_filled * sell_price = buy_filled * buy_price)
        clearing_rate = Decimal(sell_price) / Decimal(buy_price)

        # Get AMM rate
        amm_rate = router.get_reference_price(
            sell_token, buy_token, fill.sell_filled, auction
        )

        if amm_rate is not None and clearing_rate < amm_rate:
            logger.debug(
                "unified_cow_ebbo_violation",
                clearing_rate=float(clearing_rate),
                amm_rate=float(amm_rate),
            )
            return False

    return True
```

**Tests to Add:** `tests/unit/strategies/test_unified_cow.py` (new file)

```python
class TestUnifiedCowEBBO:
    def test_ebbo_formula_correct(self):
        """EBBO uses sell_price/buy_price not buy_price/sell_price."""
        # Setup: prices = {A: 100, B: 200}
        # Order sells A for B
        # clearing_rate = 100/200 = 0.5 (user gets 0.5 B per A)
        # If AMM rate is 0.4, should PASS (0.5 >= 0.4)
        # If AMM rate is 0.6, should FAIL (0.5 < 0.6)

    def test_ebbo_rejects_below_amm(self):
        """EBBO validation rejects when clearing < AMM."""

    def test_ebbo_accepts_above_amm(self):
        """EBBO validation accepts when clearing >= AMM."""

    def test_ebbo_skips_when_no_amm(self):
        """EBBO validation skips when no AMM price available."""
```

---

### 2.2 Fix Price Normalization

**File:** `solver/strategies/unified_cow.py`

**Problem:** `_normalize_prices()` uses `max(sells, buys)` which doesn't ensure conservation.

**Fix:**

```python
def _normalize_prices(self, fills: list[OrderFill]) -> dict[str, str]:
    """Compute clearing prices from fills using conservation invariant.

    For each fill, the value exchanged must be equal:
        sell_filled * sell_price = buy_filled * buy_price

    We set a reference price and propagate through the fill graph.
    """
    if not fills:
        return {}

    # Build adjacency: token -> [(other_token, rate)]
    # where rate = sell_filled / buy_filled for order selling token
    token_rates: dict[str, list[tuple[str, Decimal]]] = defaultdict(list)

    for fill in fills:
        sell_token = normalize_address(fill.order.sell_token)
        buy_token = normalize_address(fill.order.buy_token)

        if fill.sell_filled > 0 and fill.buy_filled > 0:
            # Rate from sell_token's perspective
            rate = Decimal(fill.buy_filled) / Decimal(fill.sell_filled)
            token_rates[sell_token].append((buy_token, rate))
            # Inverse rate from buy_token's perspective
            token_rates[buy_token].append((sell_token, 1 / rate))

    # BFS from first token to set consistent prices
    all_tokens = set(token_rates.keys())
    if not all_tokens:
        return {}

    start_token = next(iter(all_tokens))
    prices: dict[str, Decimal] = {start_token: Decimal(10**18)}  # Reference
    visited = {start_token}
    queue = [start_token]

    while queue:
        token = queue.pop(0)
        current_price = prices[token]

        for other_token, rate in token_rates[token]:
            if other_token not in visited:
                # price[other] = price[token] / rate
                prices[other_token] = current_price / rate
                visited.add(other_token)
                queue.append(other_token)

    # Convert to strings
    return {token: str(int(price)) for token, price in prices.items()}
```

**Tests to Add:**

```python
class TestUnifiedCowPriceNormalization:
    def test_price_normalization_two_tokens(self):
        """Two-token fills produce consistent prices."""
        # Fill: sell 100 A, get 200 B
        # Expected: price[A] / price[B] = 200/100 = 2

    def test_price_normalization_three_tokens_cycle(self):
        """Three-token cycle produces consistent prices."""
        # Fills: A->B (rate 2), B->C (rate 3), C->A (rate 1/6)
        # Expected: prices form consistent cycle

    def test_price_normalization_conservation(self):
        """Normalized prices satisfy conservation invariant."""
        # For each fill: sell_filled * sell_price ≈ buy_filled * buy_price

    def test_price_normalization_disconnected_tokens(self):
        """Disconnected token groups get independent prices."""
        # Fills: A<->B, C<->D (no connection)
        # Expected: both groups have prices, may be independent
```

---

### 2.3 Add Comprehensive Test Suite

**File:** `tests/unit/strategies/test_unified_cow.py` (new file)

```python
"""Tests for UnifiedCowStrategy."""

import pytest
from decimal import Decimal

from solver.models.auction import AuctionInstance, Order, Token
from solver.strategies.unified_cow import UnifiedCowStrategy
from tests.conftest import MockRouter, MockPoolRegistry


# === Test Fixtures ===

@pytest.fixture
def strategy():
    return UnifiedCowStrategy()

@pytest.fixture
def strategy_with_router(mock_router):
    return UnifiedCowStrategy(router=mock_router)


# === Fill-or-Kill Tests ===

class TestUnifiedCowFOK:
    def test_fok_order_fully_filled(self, strategy):
        """FOK order is fully filled when possible."""

    def test_fok_order_not_partially_filled(self, strategy):
        """FOK order is rejected rather than partially filled."""

    def test_fok_with_partial_counterparty(self, strategy):
        """FOK order can match with partially fillable counterparty."""

    def test_fok_in_cycle(self, strategy):
        """FOK orders in cycles are fully filled or excluded."""


# === Limit Price Tests ===

class TestUnifiedCowLimitPrice:
    def test_limit_price_satisfied_in_pair(self, strategy):
        """Pair matching satisfies both orders' limit prices."""

    def test_limit_price_satisfied_in_cycle(self, strategy):
        """Cycle settlement satisfies all orders' limit prices."""

    def test_limit_price_violation_rejects_match(self, strategy):
        """Match rejected if limit price would be violated."""

    def test_limit_price_boundary(self, strategy):
        """Match at exact limit price boundary is accepted."""


# === EBBO Tests ===

class TestUnifiedCowEBBO:
    def test_ebbo_validated_for_pairs(self, strategy_with_router):
        """EBBO is checked for bidirectional pair matches."""

    def test_ebbo_validated_for_cycles(self, strategy_with_router):
        """EBBO is checked for cycle settlements."""

    def test_ebbo_rejection_logged(self, strategy_with_router):
        """EBBO violations are logged with details."""

    def test_ebbo_zero_tolerance(self, strategy_with_router):
        """EBBO uses zero tolerance (no margin allowed)."""

    def test_ebbo_skipped_when_no_amm(self, strategy_with_router):
        """EBBO check skipped when no AMM liquidity."""


# === Uniform Price Tests ===

class TestUnifiedCowUniformPrice:
    def test_uniform_price_in_pair(self, strategy):
        """All orders in pair get same clearing price."""

    def test_uniform_price_in_cycle(self, strategy):
        """All orders in cycle get consistent prices."""

    def test_price_conservation_invariant(self, strategy):
        """Prices satisfy: sell_filled * sell_price = buy_filled * buy_price."""

    def test_multiple_pairs_independent_prices(self, strategy):
        """Non-overlapping pairs can have independent prices."""


# === Integration Tests ===

class TestUnifiedCowIntegration:
    def test_empty_auction(self, strategy):
        """Empty auction returns None."""

    def test_single_order(self, strategy):
        """Single order returns None (no CoW possible)."""

    def test_two_order_cow(self, strategy):
        """Two compatible orders match."""

    def test_three_order_cycle(self, strategy):
        """Three-order cycle is detected and settled."""

    def test_mixed_pairs_and_cycles(self, strategy):
        """Auction with both pairs and cycles is handled."""

    def test_overlapping_structures(self, strategy):
        """Overlapping pairs/cycles don't double-match orders."""
```

**Acceptance Criteria:**
- [ ] New test file created
- [ ] 20+ test cases covering all four constraints
- [ ] Tests for edge cases (empty, single order, boundary values)
- [ ] All tests pass
- [ ] Coverage > 80% for unified_cow.py

---

## Phase 3: Validation & Documentation

### 3.1 Add Integration Tests

**File:** `tests/integration/test_constraint_enforcement.py` (new file)

```python
"""Integration tests for constraint enforcement across strategies."""

import pytest
from pathlib import Path

from solver.solver import Solver
from solver.models.auction import AuctionInstance


class TestConstraintEnforcementIntegration:
    """Test that all constraints are enforced end-to-end."""

    def test_fok_enforced_through_solver(self):
        """FOK constraint enforced from API to solution."""

    def test_limit_price_enforced_through_solver(self):
        """Limit price constraint enforced from API to solution."""

    def test_ebbo_enforced_through_solver(self):
        """EBBO constraint enforced from API to solution."""

    def test_uniform_price_in_solution(self):
        """Final solution has uniform clearing prices."""

    def test_all_constraints_with_historical_auction(self):
        """All constraints satisfied on real auction data."""
```

### 3.2 Update Documentation

**Files to Update:**

1. **`CLAUDE.md`** - Add constraint enforcement section:
```markdown
### Constraint Enforcement

All strategies must enforce four constraints:

1. **Fill-or-Kill**: `partially_fillable=false` orders are fully filled or not at all
2. **Limit Price**: Actual rate >= limit rate (integer comparison)
3. **EBBO**: Clearing rate >= AMM rate (zero tolerance)
4. **Uniform Price**: All orders in a pair execute at same price

Safety nets in `base.py` and `solver.py` catch violations, but strategies should validate internally.
```

2. **`solver/strategies/README.md`** (new file) - Strategy constraint matrix

3. **Docstring updates** in each strategy file

---

## Implementation Order

### Week 1: Production Fixes (Critical)

| Day | Task | Files | Tests |
|-----|------|-------|-------|
| 1 | CowMatchStrategy EBBO | `cow_match.py` | 4 tests |
| 2 | AmmRoutingStrategy limit price | `amm_routing.py` | 4 tests |
| 3 | AmmRoutingStrategy price conflict | `amm_routing.py` | 3 tests |
| 4 | Integration tests | `test_constraint_enforcement.py` | 5 tests |
| 5 | Documentation updates | `CLAUDE.md`, docstrings | - |

### Week 2: UnifiedCowStrategy Fixes (Research)

| Day | Task | Files | Tests |
|-----|------|-------|-------|
| 1 | Fix EBBO formula | `unified_cow.py` | 4 tests |
| 2 | Fix price normalization | `unified_cow.py` | 4 tests |
| 3-4 | Comprehensive test suite | `test_unified_cow.py` | 20 tests |
| 5 | Code review & cleanup | All modified files | - |

---

## Acceptance Criteria Summary

### Phase 1 Complete When:
- [ ] CowMatchStrategy validates EBBO (4 tests)
- [ ] AmmRoutingStrategy validates limit prices (4 tests)
- [ ] AmmRoutingStrategy detects price conflicts (3 tests)
- [ ] All existing tests still pass
- [ ] Integration tests pass (5 tests)

### Phase 2 Complete When:
- [ ] UnifiedCowStrategy EBBO formula fixed (4 tests)
- [ ] UnifiedCowStrategy price normalization fixed (4 tests)
- [ ] Comprehensive test suite (20+ tests)
- [ ] All tests pass

### Phase 3 Complete When:
- [ ] Documentation updated
- [ ] All strategies have constraint docstrings
- [ ] CLAUDE.md includes constraint section

---

## Risk Mitigation

1. **Backward Compatibility**: All fixes are additive (new validation, not changed behavior)
2. **Safety Nets Remain**: Solver-level filters still catch any missed violations
3. **Incremental Rollout**: Fix production strategies first, research strategies second
4. **Test Coverage**: Every fix has dedicated tests before merge

---

## Success Metrics

After implementation:
- [ ] Zero EBBO violations in solver logs (currently some logged as ERROR)
- [ ] All 4 constraints documented in each strategy
- [ ] Test coverage > 90% for constraint-related code
- [ ] Benchmark shows no regression in match rate or performance
