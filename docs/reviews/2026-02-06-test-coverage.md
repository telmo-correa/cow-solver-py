# Test Coverage Gap Analysis

**Date:** 2026-02-06
**Reviewer:** Claude Opus 4.6
**Scope:** tests/unit/, tests/integration/, tests/conftest.py

---

## Remediation Status

**Tests:** 992 → 1023 (after two remediation passes; net reduction from deleting dead code tests)

**New test files added:**
- `tests/unit/math/test_bfp_safety.py` — 13 tests for Bfp underflow guards
- `tests/unit/api/test_request_limits.py` — 4 tests for request size limits
- `tests/unit/api/test_endpoints_timeout.py` — Timeout enforcement tests
- `tests/unit/strategies/test_settlement.py` — Settlement cycle tests
- `tests/unit/strategies/test_double_auction_rounding.py` — Rounding precision tests
- `tests/unit/strategies/test_multi_pair_cow.py` — Multi-pair CoW matching tests
- `tests/unit/test_ebbo.py` — EBBO validation tests

**Test files removed:**
- `tests/unit/strategies/test_unified_cow.py` — Deleted with UnifiedCowStrategy

**Remaining gaps** (lower priority): Comprehensive settlement bottleneck scaling edge cases, constraint enforcement integration tests across full strategy chain, multi-hop fee calculation edge cases.

---

## Test Statistics

- **992 tests passing** across 62 test files
- **Code-to-test ratio:** 1:1.25 (18,742 LOC : 23,510 test LOC)
- **Unit tests:** 49 files, 19,075 lines
- **Integration tests:** 8 files, 4,372 lines

---

## Modules With No Test Coverage

### CRITICAL: No dedicated tests

| Module | Lines | Risk |
|--------|-------|------|
| `solver/strategies/settlement.py` | 560 | **Critical** -- cycle settlement, bottleneck scaling, price extraction |
| `solver/strategies/components.py` | ~200 | Medium -- token component detection for multi-pair |
| `solver/strategies/graph.py` | ~150 | Medium -- spanning tree, union-find |
| `solver/routing/handlers/base.py` | ~120 | Low -- base handler template |
| `solver/routing/handlers/balancer.py` | ~100 | Medium -- Balancer routing handler |
| `solver/routing/handlers/limit_order.py` | ~80 | Medium -- limit order routing handler |
| `solver/routing/solution.py` | ~150 | Medium -- solution building from routing results |
| `solver/fees/price_estimation.py` | ~400 | Medium -- price estimation for fee calculation |
| `solver/strategies/unified_cow.py` | 1028 | Low -- appears to be dead code |

`settlement.py` is the most critical gap. It contains cycle settlement math, bottleneck scaling, and fill ratio calculations that are foundational to ring trade support.

---

## Missing Edge Case Tests

### 1. `OrderFill` methods untested

`OrderFill.from_order_with_amounts()`, `OrderFill.from_order_with_fill_ratio()`, and `OrderFill.to_trade()` have no dedicated tests. These are used in every strategy to create fills.

### 2. `convert_fill_ratios_to_fills` untested

This function in `settlement.py` converts fill ratios to actual `OrderFill` objects with integer amounts. No tests verify rounding behavior or edge cases.

### 3. `StrategyResult.combine` untested

Combining multiple strategy results is used in `Solver.solve()` but has no tests for overlapping fills, duplicate interactions, or price conflicts.

### 4. `_validate_fills_satisfy_limits` untested

The safety net in `base.py` that catches limit price violations has no dedicated test. Its behavior when detecting violations (logging and filtering) is unverified.

---

## Missing Negative / Error Path Tests

### 5. Solver error handling

No tests for:
- What happens when ALL strategies return None
- Strategy raising an exception mid-solve
- Invalid auction data reaching the solver (empty orders, empty liquidity)

### 6. Settlement edge cases

No tests for:
- `solve_cycle` with all fill-or-kill orders where bottleneck causes partial fills
- `calculate_cycle_settlement` with a cycle where one order has `sell_amount=0`
- Ring trades where the product of exchange rates is exactly 1.0

### 7. Routing failures

No tests for:
- `build_solution` receiving a `RoutingResult` with mismatched token addresses
- Multi-hop route where an intermediate hop fails but first hop succeeds

---

## Missing Boundary Condition Tests

### 8. Cycle settlement boundaries

- Cycle with exactly 2 tokens (degenerate cycle, should match `CowMatch`)
- Cycle with maximum tokens (4, per `max_cycle_length`)
- Bottleneck order with `sell_amount = 1` (minimum viable fill)

### 9. Fee calculation boundaries

- Gas price of 0 (should produce 0 fee for limit orders)
- Reference price of 0 (should avoid division by zero)
- Extremely large gas estimate (uint256 max range)

### 10. Uint256 validation boundaries

- Values at exactly `2^256 - 1` (max valid)
- Values at exactly `2^256` (first invalid)
- Negative values, empty strings, non-numeric strings

### 11. EBBO edge cases

- Clearing price exactly equal to EBBO price (boundary -- should pass)
- Clearing price 1 wei below EBBO (should fail)
- EBBO with no AMM liquidity for the pair (fallback behavior)

---

## Weak Assertions

### 12. Tests that don't assert enough

- Several `multi_pair` tests check `result is not None` and `len(result.fills) > 0` but don't verify the actual clearing prices or fill amounts satisfy constraints.
- Some AMM routing tests verify `amount_out > 0` without checking it matches the expected formula output.
- Integration tests for constraint enforcement verify that fills exist but don't verify the specific constraint values.

---

## Missing Integration Tests

### 13. Full strategy chain composition

No test runs `MultiPairCowStrategy` + `AmmRoutingStrategy` in sequence to verify:
- Orders matched by CoW are properly excluded from AMM routing
- Combined solution has consistent prices
- No duplicate fills across strategies

### 14. Settlement + EBBO interaction

No test verifies that cycle settlement fills pass EBBO validation end-to-end. The `_filter_ebbo_violations` safety net should never trigger for cycle fills, but this is unverified.

### 15. Multi-hop fee calculation

No test verifies that fee calculation for a multi-hop route correctly sums gas across hops.

---

## Test Quality Issues

### 16. Duplicate helper patterns

Test helper functions for creating mock orders are duplicated across test files rather than centralized in `conftest.py` or `tests/helpers/`.

### 17. Inconsistent token address constants

Some tests use `"0xtoken_a"`, others use `"0x" + "a" * 40`. No single canonical set of test token addresses.

### 18. Missing gas price in test fixtures

Several test fixtures for auctions don't include `gas_price` or `reference_token`, relying on defaults. This masks bugs in fee calculation paths that depend on these fields.

---

## Constraint Validation Gaps

### 19. Fill-or-kill

- No test for FOK with buy orders (only sell orders tested)
- No test for FOK enforcement in `MultiPairCowStrategy` cycle fills
- No test that FOK violation in cycle settlement is caught by any safety net

### 20. Limit price

- No test for limit price validation across `StrategyResult.combine`
- No test for limit price edge case where `sell_amount * buy_filled == buy_amount * sell_filled` (exact equality)

### 21. EBBO

- The integration test in `test_constraint_enforcement.py` for EBBO uses mock data that may not trigger the actual EBBO validation path
- No test for EBBO with multi-hop routes where intermediate hops have different rates

### 22. Uniform price

- No dedicated test that two orders in the same pair execute at the same clearing price
- The `verify_uniform_prices` function (if it exists) is not tested for actually detecting non-uniform prices
