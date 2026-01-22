# Architecture Refactoring Plan

## Executive Summary

The codebase is well-structured overall, with clear separation of concerns and good test coverage (651 tests). However, several areas need refactoring, particularly the **massive balancer.py (1,814 LOC)** and some architectural inconsistencies across AMM implementations.

## Critical Issues

### 1. Monolithic `solver/amm/balancer.py` (1,814 LOC) - **HIGH PRIORITY**

This file contains 8 distinct concerns crammed into one module:

| Section | Lines | Content |
|---------|-------|---------|
| Exception classes | 83-131 | 8 error classes |
| Pool dataclasses | 143-254 | WeightedTokenReserve, BalancerWeightedPool, StableTokenReserve, BalancerStablePool |
| Scaling helpers | 256-315 | scale_up, scale_down_down, scale_down_up |
| Fee helpers | 317-371 | subtract_swap_fee_amount, add_swap_fee_amount |
| Weighted math | 373-522 | calc_out_given_in, calc_in_given_out |
| Stable math | 524-831 | calculate_invariant, get_token_balance, stable_calc_* |
| Parsing | 833-1367 | parse_weighted_pool, parse_stable_pool + 10 helpers |
| Binary search | 1369-1453 | _binary_search_max_sell_fill, _binary_search_max_buy_fill |
| AMM classes | 1455-1814 | BalancerWeightedAMM, BalancerStableAMM |

**Problem:** Too large to maintain, test, or understand. Math changes require touching parsing code.

### 2. `PoolRegistry` in Wrong Module - **MEDIUM PRIORITY**

`PoolRegistry` is defined in `uniswap_v2.py` (lines 460-830) but manages V2, V3, weighted, and stable pools:

```python
# Current location: solver/amm/uniswap_v2.py
class PoolRegistry:
    def add_pool(self, pool: UniswapV2Pool) -> None: ...
    def add_v3_pool(self, pool: UniswapV3Pool) -> None: ...
    def add_weighted_pool(self, pool: BalancerWeightedPool) -> None: ...
    def add_stable_pool(self, pool: BalancerStablePool) -> None: ...
```

**Problem:** Confusing location, name doesn't match capability.

### 3. Inconsistent AMM Interfaces - **MEDIUM PRIORITY**

The three AMM types have different method signatures:

```python
# UniswapV2 - takes reserves directly
UniswapV2.max_fill_sell_order(reserve_in, reserve_out, sell_amount, buy_amount, fee_multiplier)

# Balancer - takes pool object
BalancerWeightedAMM.max_fill_sell_order(pool, token_in, token_out, sell_amount, buy_amount)

# UniswapV3 - NO partial fill methods at all!
```

**Problem:** Router has to handle each AMM type differently. No V3 partial fill support.

### 4. Duplicated Router Logic - **MEDIUM PRIORITY**

`_route_through_weighted_pool` and `_route_through_stable_pool` in router.py are nearly identical (60+ lines each). The only difference is which AMM they call.

### 5. Type Alias as String - **LOW PRIORITY**

```python
# router.py line 53
AnyPool = "UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool"
```

This is a string, not a proper Union type, causing IDE issues.

---

## Refactoring Phases

### Phase 1: Split balancer.py into a Package

Create `solver/amm/balancer/` directory:

```
solver/amm/balancer/
├── __init__.py          # Re-export public API
├── errors.py            # Exception classes (~50 LOC)
├── pools.py             # Pool dataclasses (~120 LOC)
├── scaling.py           # scale_up, scale_down_* (~60 LOC)
├── weighted_math.py     # calc_out_given_in, calc_in_given_out (~180 LOC)
├── stable_math.py       # calculate_invariant, stable_calc_* (~350 LOC)
├── parsing.py           # parse_weighted_pool, parse_stable_pool (~480 LOC)
└── amm.py               # BalancerWeightedAMM, BalancerStableAMM (~400 LOC)
```

**Benefits:**
- Each file < 500 LOC (maintainable)
- Math changes don't touch parsing
- Easier to test in isolation
- Clear responsibilities

### Phase 2: Create pools Package

Move PoolRegistry to its own module:

```
solver/pools/
├── __init__.py
├── types.py             # AnyPool = Union[...] (proper type)
└── registry.py          # PoolRegistry class (~400 LOC)
```

Update imports:
```python
# Before
from solver.amm.uniswap_v2 import PoolRegistry

# After
from solver.pools import PoolRegistry, AnyPool
```

### Phase 3: Standardize AMM Interface

Create a Protocol that all AMMs must implement:

```python
# solver/amm/base.py
class SwapCalculator(Protocol):
    def simulate_swap(
        self, pool: Any, token_in: str, token_out: str, amount_in: int
    ) -> SwapResult | None: ...

    def simulate_swap_exact_output(
        self, pool: Any, token_in: str, token_out: str, amount_out: int
    ) -> SwapResult | None: ...

    def max_fill_sell_order(
        self, pool: Any, token_in: str, token_out: str, sell_amount: int, buy_amount: int
    ) -> int: ...

    def max_fill_buy_order(
        self, pool: Any, token_in: str, token_out: str, sell_amount: int, buy_amount: int
    ) -> int: ...
```

Then update all AMMs to conform.

### Phase 4: Simplify Router with Dispatch

Replace isinstance checks with a cleaner dispatch:

```python
def _get_amm_for_pool(self, pool: AnyPool) -> SwapCalculator | None:
    """Get the AMM that handles this pool type."""
    if isinstance(pool, UniswapV3Pool):
        return self.v3_amm
    elif isinstance(pool, BalancerWeightedPool):
        return self.weighted_amm
    elif isinstance(pool, BalancerStablePool):
        return self.stable_amm
    return self.amm  # V2 default
```

This allows consolidating `_route_through_weighted_pool` and `_route_through_stable_pool` into one method.

### Phase 5: Split Large Test Files

Split `test_balancer.py` (3,246 LOC) into:
- `test_balancer_scaling.py` - scaling/fee tests
- `test_balancer_weighted_math.py` - weighted pool math
- `test_balancer_stable_math.py` - stable pool math
- `test_balancer_parsing.py` - pool parsing tests
- `test_balancer_amm.py` - AMM class tests

---

## Implementation Order

Recommended sequence to minimize risk:

1. **Split balancer.py into package** (largest impact, most isolated)
2. **Create pools package** (clean dependency)
3. **Standardize AMM interface** (enables router simplification)
4. **Simplify router dispatch** (depends on #3)
5. **Split test files** (can be done anytime)

Each phase maintains all existing tests. No regressions.

---

## Success Criteria

### Overall
- [ ] All 651 tests pass (0 regressions)
- [ ] `mypy solver/` passes with no new errors
- [ ] `ruff check solver/` passes with no new errors
- [ ] Benchmarks still produce exact match with Rust baseline

### Phase 1: Split balancer.py
- [ ] `solver/amm/balancer.py` removed
- [ ] `solver/amm/balancer/` package created with 7 modules
- [ ] Each module < 500 LOC
- [ ] All imports work: `from solver.amm.balancer import BalancerWeightedPool, ...`
- [ ] No circular import errors
- [ ] All 651 tests pass

### Phase 2: Create pools Package
- [ ] `solver/pools/` package created
- [ ] `PoolRegistry` moved from `uniswap_v2.py`
- [ ] `AnyPool` is a proper Union type (not string)
- [ ] All imports updated throughout codebase
- [ ] All 651 tests pass

### Phase 3: Standardize AMM Interface
- [ ] `SwapCalculator` protocol defined in `solver/amm/base.py`
- [ ] All AMMs implement the protocol consistently
- [ ] `UniswapV3AMM` has `max_fill_*` methods (binary search fallback)
- [ ] All 651 tests pass

### Phase 4: Simplify Router
- [ ] `_get_amm_for_pool` dispatch method added
- [ ] `_route_through_weighted_pool` and `_route_through_stable_pool` consolidated
- [ ] Code duplication reduced
- [ ] All 651 tests pass

### Phase 5: Split Test Files
- [ ] `test_balancer.py` split into 5 focused files
- [ ] Each test file < 1000 LOC
- [ ] All 651 tests pass
- [ ] Test organization matches source organization

---

## File Changes Summary

### Files to Create
- `solver/amm/balancer/__init__.py`
- `solver/amm/balancer/errors.py`
- `solver/amm/balancer/pools.py`
- `solver/amm/balancer/scaling.py`
- `solver/amm/balancer/weighted_math.py`
- `solver/amm/balancer/stable_math.py`
- `solver/amm/balancer/parsing.py`
- `solver/amm/balancer/amm.py`
- `solver/pools/__init__.py`
- `solver/pools/types.py`
- `solver/pools/registry.py`
- `tests/unit/test_balancer_scaling.py`
- `tests/unit/test_balancer_weighted_math.py`
- `tests/unit/test_balancer_stable_math.py`
- `tests/unit/test_balancer_parsing.py`
- `tests/unit/test_balancer_amm.py`

### Files to Delete
- `solver/amm/balancer.py`
- `tests/unit/test_balancer.py`

### Files to Modify
- `solver/amm/__init__.py` - update exports
- `solver/amm/uniswap_v2.py` - remove PoolRegistry
- `solver/amm/uniswap_v3.py` - add max_fill methods
- `solver/amm/base.py` - add SwapCalculator protocol
- `solver/routing/router.py` - update imports, add dispatch
- `solver/solver.py` - update imports
- `tests/conftest.py` - update imports
- Various test files - update imports

---

## Rollback Plan

If issues arise during any phase:
1. Git revert to previous commit
2. All changes are atomic per phase
3. Each phase has its own commit for easy rollback
