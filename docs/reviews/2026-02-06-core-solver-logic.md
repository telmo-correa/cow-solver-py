# Core Solver Logic Review

**Date:** 2026-02-06
**Reviewer:** Claude Opus 4.6
**Scope:** solver/solver.py, solver/ebbo.py, solver/strategies/

---

## Remediation Status

| ID | Severity | Issue | Status |
|----|----------|-------|--------|
| C1 | Critical | EBBO filter keeps ALL interactions | **FIXED** — Token-pair matching for interaction filtering |
| C2 | Critical | Missing cycle prices in Phase 2 | **FIXED** — `_solve_cycles` returns `(fills, prices)` tuple |
| C3 | Critical | Fill-or-kill bypass in cycle settlement | **FIXED** — FOK validation added to `solve_cycle` |
| C4 | Critical | Rounding inconsistency between auction paths | **FIXED** — `_exact_price_key` replaces lossy sort key |
| H5 | High | `_price_ratio_sort_key` lossy scaling | **FIXED** — Exact cross-multiplication sort |
| H6 | High | `CowMatchStrategy` dead code | **FIXED** — Removed from default chain |
| H7 | High | No negative reserve guard | **FIXED** — Guard in `_create_updated_pool` |
| H8 | High | FOK-blind order selection for cycles | **FIXED** — FOK awareness added |
| H9 | High | Missing AttributeError handling in ebbo.py | **FIXED** — try/except with fallback |
| M10 | Medium | Mutable instance state in CowMatchStrategy | **DEFERRED** — Low risk, strategy not in default chain |
| M11 | Medium | `max()` for clearing prices | **FIXED** — ROUND_DOWN for clearing prices |
| M12 | Medium | Equal gas splitting | **DEFERRED** — Accept risk, minor inaccuracy |
| M13 | Medium | solve_fills_at_prices_v2 buy order handling | **REVIEWED** — Logic is correct; added clarifying comment |
| M14 | Medium | Token normalization inconsistency | **FIXED** — EBBO address normalization |
| M15 | Medium | _normalize_prices fallback | **FIXED** — Added clarifying comment |
| L16-L21 | Low | Various | See individual entries |

---

## CRITICAL Issues

### 1. EBBO violation filter keeps ALL interactions regardless of rejected fills

**File:** `solver/solver.py:447-453`

The code claims to filter interactions for valid fills, but unconditionally appends every interaction. When a fill is rejected for EBBO violation but its corresponding AMM interaction is retained, the on-chain settlement will execute a swap with no corresponding trade. The comment "we keep all since they may be shared" is incorrect for independently-routed AMM orders.

### 2. Missing cycle prices in Phase 2

**File:** `solver/strategies/multi_pair.py:163-170`

In `_solve_with_cycles`, fills from cycles are collected but their clearing prices are never added to `all_prices`. When `_build_result` is called, it receives `all_prices` which only contains Phase 1 prices. Cycle tokens may have **no clearing prices** in the solution, causing settlement failures.

### 3. Fill-or-kill bypass in cycle settlement

**File:** `solver/strategies/settlement.py:496-524`

`solve_cycle` skips fills where `sell_filled <= 0 or buy_filled <= 0` but does **not** check fill-or-kill constraints. If a non-partially-fillable order gets a partial fill due to bottleneck scaling, it passes through. There is **no downstream filter** for FOK on cycle fills.

### 4. Inconsistent rounding between hybrid and pure double auction

**Files:** `solver/strategies/double_auction/hybrid.py:309`, `solver/strategies/double_auction/core.py:241`

Hybrid auction uses ceiling division for `match_b` (favors seller). Pure double auction uses floor division (disfavors seller). The pure path silently skips valid matches that the ceiling-based approach would have matched.

---

## HIGH Issues

### 5. `_price_ratio_sort_key` uses lossy integer scaling

**File:** `solver/strategies/double_auction/core.py:88-96`

Converts rational price to fixed-point `num * 10**18 // denom`. Two close ratios can map to the same sort key, causing non-deterministic ordering. A `Fraction`-based comparator would be exact.

### 6. `CowMatchStrategy` is dead code in default configuration

**File:** `solver/strategies/cow_match.py:129`

Hard-coded to only work with exactly 2 orders. Not in the default strategy chain (`MultiPairCowStrategy` + `AmmRoutingStrategy`).

### 7. No negative reserve guard in AMM routing

**File:** `solver/strategies/amm_routing.py:269-271`

No check that reserves remain non-negative after `pool.reserve1 - amount_out`. A bug in routing could produce negative reserves, causing nonsensical results in subsequent pool usage.

### 8. FOK-blind order selection for cycles

**File:** `solver/strategies/multi_pair.py:427-433`

The "best" order per cycle edge is selected purely by rate. If this order is fill-or-kill and the settlement would only partially fill it, the settlement fails silently.

### 9. Missing AttributeError handling in ebbo.py

**File:** `solver/ebbo.py:261-266`

Unlike `ebbo_bounds.py` (which catches `AttributeError`), this code does not catch `AttributeError` if the router lacks `get_reference_price_ratio`. The sister module explicitly handles this case.

---

## MEDIUM Issues

### 10. Mutable instance state for auction in CowMatchStrategy

**File:** `solver/strategies/cow_match.py:107, 127`

`_current_auction` stored as instance state. Not safe for concurrent use. Fragile pattern even in single-threaded code.

### 11. `_normalize_prices` uses `max()` which is semantically incorrect

**File:** `solver/strategies/multi_pair.py:527-542`

Using `max(sells, buys)` for a clearing price does not correspond to any meaningful exchange rate and may violate the `price[sell] * executed = price[buy] * output` invariant.

### 12. Equal gas splitting across all fills

**File:** `solver/solver.py:305`

Gas is split equally across fills regardless of pool type (V2 vs V3 vs Balancer). Different pool types have dramatically different gas costs.

### 13. ROUND_HALF_UP for clearing prices can cause non-conservation

**File:** `solver/strategies/multi_pair.py:488`

Rounding a clearing price up with ROUND_HALF_UP can promise users more tokens than the solver has. ROUND_DOWN would be conservative.

### 14. `EBBOPrices.from_json` does not normalize token addresses

**File:** `solver/ebbo.py:67-78`

Token addresses from JSON stored as-is, but `get_price` normalizes lookup keys. Case mismatch causes lookup failures.

### 15. `solve_fills_at_prices_v2` does not handle buy orders correctly

**File:** `solver/strategies/pricing.py:549-556`

Only checks `current_ratio >= limit_price`, which is correct for sellers but inverted for buyers. Buy orders that should be eligible are excluded and vice versa.

---

## LOW Issues

- Empty `TYPE_CHECKING` block in `amm_routing.py:21-22`
- Duplicate `normalize_address` import in `solver.py:24, 267`
- Deprecated `get_limit_price` still exported in `__all__`
- `find_viable_cycle_direction` returns near-viable results (`.viable=False`) as non-None
- `SafeInt.__neg__` allows negative values despite "non-negative" documentation
- Inconsistent price naming conventions between `cow_match.py` and `solver.py`
