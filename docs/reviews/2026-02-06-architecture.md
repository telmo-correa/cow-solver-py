# Architecture & Design Review

**Date:** 2026-02-06
**Reviewer:** Claude Opus 4.6
**Scope:** Overall architecture, module design, dependencies, performance

---

## Remediation Status

| ID | Severity | Issue | Status |
|----|----------|-------|--------|
| H1 | High | Module-level singleton side effect | **FIXED** — Lazy `get_default_solver()` |
| M2 | Medium | AMM parameter convoy | **DEFERRED** — Large refactor, not correctness |
| M3 | Medium | Legacy code paths | **FIXED** — Legacy MultihopRouter dispatch removed |
| M4 | Medium | UnifiedCowStrategy dead code | **FIXED** — Deleted (~1028 lines) |
| M5 | Medium | Code duplication in strategies | **PARTIALLY FIXED** — Shared `decimal_utils.py` extracted |
| M6 | Medium | PoolRegistry encapsulation violation | **FIXED** — `get_all_token_pairs()` + simplified protocol |
| M7 | Medium | Pool registry rebuilt per auction | **DEFERRED** — Moderate refactor |
| M8 | Medium | Missing deadline enforcement | **FIXED** — 30s timeout in endpoint |
| M9 | Medium | build_solution complexity | **DEFERRED** — Decomposition is nice-to-have |
| M10 | Medium | scipy runtime dependency | **FIXED** — `/health` reports `scipy_available` |
| L11 | Low | Duck-typing with hasattr | **FIXED** — `isinstance(LiquidityInteraction)` in solver.py |
| L12 | Low | Excessive address normalization | **DEFERRED** — Performance optimization |
| L13 | Low | Wide `__init__.py` exports | **FIXED** — Trimmed to public API only |
| L14 | Low | No centralized configuration | **DEFERRED** — Architecture change |
| L15 | Low | BFS excessive temporary objects | **DEFERRED** — Performance optimization |
| L16 | Low | ProcessPoolExecutor overhead | **DEFERRED** — Performance optimization |
| L17 | Low | Logging inconsistency | **FIXED** — parse_limit_order failure logging added |
| L18 | Low | Unconditional /health endpoint | **FIXED** — Enhanced with scipy check |

---

## HIGH Issues

### 1. Module-level singleton side effect

**File:** `solver/solver.py:515`

```python
solver = _create_default_solver()
```

The module creates a singleton `Solver` at import time. This:
- Triggers `_create_default_solver()` on any import of `solver.solver`
- Potentially creates an RPC connection (if `RPC_URL` set) during import
- Cannot be reconfigured after startup
- Makes testing fragile (singleton created even for unrelated tests)
- Fails the entire import if RPC connection hangs

---

## MEDIUM Issues

### 2. AMM parameter convoy (repeated plumbing)

**Files:** `solver/solver.py:47-56`, `strategies/base_amm.py:45-53`, `routing/router.py:81-89`, `routing/multihop.py:30-36`, `strategies/multi_pair.py:90-101`

The same 5-parameter set (`amm`, `v3_amm`, `weighted_amm`, `stable_amm`, `limit_order_amm`) is passed through 6+ constructors. Adding a new AMM type requires updating every constructor. **Fix:** Bundle into an `AMMComponents` dataclass.

### 3. Legacy code paths still in production

**Files:** `solver/solver.py:50-55, 83-117`, `routing/router.py:89, 108-121, 243-246, 291-296`, `routing/multihop.py:343-439`

Extensive backward-compatibility code:
- `Solver.__init__` has a three-way branch (explicit strategies, legacy AMM params, defaults)
- `MultihopRouter` maintains complete legacy `isinstance`-based dispatch (~100 lines) alongside `HandlerRegistry`
- `SingleOrderRouter.route_order` falls back to legacy `pool_finder`

The `HandlerRegistry` pattern is the intended future. Legacy paths should be removed.

### 4. `UnifiedCowStrategy` is dead code (1028 lines)

**File:** `solver/strategies/unified_cow.py`

Not in default strategy chain, not imported by `__init__.py`, not referenced by `solver.py`. Contains near-complete duplication of logic from `MultiPairCowStrategy`, `pricing.py`, `components.py`, `graph.py`, and `ebbo_bounds.py`. Should be removed or clearly isolated as experimental.

### 5. Code duplication across strategy modules

**Files:** `strategies/pricing.py`, `strategies/ebbo_bounds.py`, `strategies/unified_cow.py`

- `_DECIMAL_HIGH_PREC_CONTEXT` and helper functions duplicated in 3 files
- `solve_fills_at_prices` and `solve_fills_at_prices_v2` are near-identical LP formulations
- `build_token_graph_from_groups` and `build_token_graph_from_orders` are near-identical graph builders

### 6. PoolRegistry encapsulation violation

**File:** `solver/routing/pathfinding.py:132-161`

The `PoolGraphSource` protocol exposes underscore-prefixed "private" attributes (`_pools`, `_v3_pools`, `_weighted_pools`, `_stable_pools`, `_limit_orders`). Any change to `PoolRegistry`'s internal storage breaks `TokenGraph._build_from_registry`. **Fix:** Expose a `get_all_token_pairs()` public method.

### 7. Pool registry rebuilt on every auction (twice)

**Files:** `strategies/amm_routing.py:153`, `strategies/multi_pair.py:129`

Both strategies call `build_registry_from_liquidity(auction.liquidity)` independently. For ~200 liquidity sources, this means parsing all sources twice per auction. **Fix:** Build once in `Solver.solve()` and pass to strategies.

### 8. Missing deadline enforcement

**Files:** `solver/api/endpoints.py:36-113`, `solver/solver.py:133-219`

`AuctionInstance.deadline` is never used. No timeout mechanism, no deadline checking, no early termination, no graceful degradation.

### 9. `StrategyResult.build_solution` is a 130-line mixed-concern method

**File:** `solver/strategies/base.py:276-406`

Mixes fee calculation, fee validation, price adjustment, trade building, and solution assembly. Should be decomposed.

### 10. scipy is dev-only but required at runtime

**File:** `pyproject.toml:27-35`

`scipy` is only in `dev` and `optimization` optional deps, but `MultiPairCowStrategy` (in the default production chain) uses `scipy.optimize.linprog`. Without scipy, multi-pair optimization silently degrades with no startup warning.

---

## LOW Issues

### 11. `Solver._build_separate_solutions` uses hasattr for duck typing

**File:** `solver/solver.py:274-283`

Uses `hasattr(interaction, "input_token")` instead of `isinstance(interaction, LiquidityInteraction)`.

### 12. Excessive address normalization

Throughout codebase. `normalize_address()` called multiple times for the same address in a single code path. **Fix:** Normalize once at the system boundary.

### 13. Wide `__init__.py` export in strategies

**File:** `solver/strategies/__init__.py`

Re-exports 22 symbols including internal utilities (`UnionFind`, `build_token_graph`, etc.). Makes refactoring internals risky.

### 14. No centralized configuration management

Configuration scattered across environment variables in 3 files, hard-coded constants, and strategy constructor parameters. No configuration object, no startup validation, no runtime tuning.

### 15. BFS pathfinding creates excessive temporary objects

**File:** `solver/routing/pathfinding.py:380-414`

Creates a new `frozenset` on every neighbor expansion. For high-degree hubs like WETH (~600 neighbors), this creates many temporary objects. Mutable set with backtracking would be more efficient.

### 16. ProcessPoolExecutor overhead for path pre-warming

**File:** `solver/routing/pathfinding.py:478-572`

Serializes the entire adjacency dict for each worker process. For typical auction sizes, the IPC overhead likely exceeds BFS computation time. Sequential fallback is probably faster.

### 17. Logging inconsistency

Failed routing in `amm_routing.py:113` logs at `debug` level. For production monitoring of failed orders, `info` or `warning` would be more appropriate.

### 18. `/health` endpoint is unconditional

**File:** `solver/api/main.py:24-26`

Returns `{"status": "ok"}` without checking solver creation, RPC connectivity, or scipy availability.

---

## Positive Observations

- **Strategy pattern** is well-designed and composable
- **Dependency injection** throughout enables isolated testing
- **SafeInt** wrapper prevents arithmetic errors in financial calculations
- **EBBO two-layer safety net** (strategy + solver) is robust
- **Handler registry** for routing is a clean abstraction
- **Clear separation** between API, models, strategies, routing, and AMM math
- **Comprehensive test suite** with 992 tests and 1:1.25 code-to-test ratio
