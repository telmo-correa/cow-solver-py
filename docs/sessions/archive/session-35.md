# Session 35 - Architecture Review and Refactoring (Phases 1-5)

**Date:** 2026-01-22

## Summary

Comprehensive architecture review identified improvement opportunities, followed by systematic refactoring across 5 phases. Split large modules into focused packages, standardized AMM interfaces with protocols, and improved code organization.

## Architecture Review Findings

### Files Reviewed
- `solver/amm/balancer.py` (838 LOC) - Monolithic file with 4 distinct concerns
- `solver/pools/registry.py` (208 LOC) - Mixed with pools/__init__.py
- `solver/amm/uniswap_v2.py` - Missing protocol alignment
- `solver/amm/uniswap_v3.py` (1031 LOC) - Largest file, 6 distinct concerns
- `solver/routing/router.py` (509 LOC) - Multiple pool-type handlers mixed in
- `tests/unit/test_balancer.py` (1100 LOC) - Monolithic test file

### Key Recommendations Implemented
1. Split balancer.py into package (838 → 6 focused modules)
2. Create solver/pools package for PoolRegistry
3. Add SwapCalculator protocol for AMM interface standardization
4. Simplify router dispatch with proper AnyPool type
5. Split test_balancer.py into focused test files

## Refactoring Phases

### Phase 1: Balancer Package Split

Split `solver/amm/balancer.py` (838 LOC) into focused modules:

```
solver/amm/balancer/
├── __init__.py       # Re-exports for backward compatibility
├── types.py          # BalancerToken, PoolVersion (~30 LOC)
├── weighted.py       # BalancerWeightedPool, BalancerWeightedAMM (~200 LOC)
├── stable.py         # BalancerStablePool, BalancerStableAMM (~250 LOC)
├── parsing.py        # parse_balancer_liquidity (~150 LOC)
└── handler.py        # BalancerHandler for routing (~100 LOC)
```

All imports continue to work unchanged via `__init__.py` re-exports.

### Phase 2: Pools Package Creation

Created `solver/pools/` package for centralized pool management:

```
solver/pools/
├── __init__.py       # Re-exports: PoolRegistry, AnyPool
├── registry.py       # PoolRegistry class
└── types.py          # AnyPool type alias
```

Moved `AnyPool` type from scattered locations to centralized `solver/pools/types.py`.

### Phase 3: SwapCalculator Protocol

Added `SwapCalculator` protocol to `solver/amm/base.py`:

```python
class SwapCalculator(Protocol):
    """Protocol for AMM swap calculation."""

    def simulate_swap(
        self,
        pool: Any,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> SwapResult | None:
        """Simulate a swap through a pool."""
        ...

    def simulate_swap_exact_output(
        self,
        pool: Any,
        token_in: str,
        token_out: str,
        amount_out: int,
    ) -> SwapResult | None:
        """Simulate a swap for exact output amount."""
        ...
```

UniswapV2 now implements both patterns:
- `simulate_swap(pool, token_in, amount_in)` - V2-specific (2-token pools)
- `simulate_swap(pool, token_in, token_out, amount_in)` - Protocol-compliant

### Phase 4: Router Simplification

Simplified router dispatch using proper `AnyPool` type:

**Before:**
```python
from solver.amm.uniswap_v2 import UniswapV2Pool
from solver.amm.uniswap_v3 import UniswapV3Pool
from solver.amm.balancer import BalancerWeightedPool, BalancerStablePool

AnyPool = UniswapV2Pool | UniswapV3Pool | BalancerWeightedPool | BalancerStablePool
```

**After:**
```python
from solver.pools import AnyPool  # Single import from centralized location
```

### Phase 5: Test File Split

Split `tests/unit/test_balancer.py` (1100 LOC) into 5 focused files:

```
tests/unit/
├── test_balancer_types.py      # Token/pool type tests
├── test_balancer_weighted.py   # Weighted pool math tests
├── test_balancer_stable.py     # Stable pool math tests
├── test_balancer_parsing.py    # Liquidity parsing tests
└── test_balancer_handler.py    # Handler routing tests
```

### Phase 6: Router Handler Pattern

Split `solver/routing/router.py` into focused modules with handler pattern:

```
solver/routing/
├── __init__.py        # Re-exports
├── router.py          # SingleOrderRouter facade (~200 LOC)
├── types.py           # HopResult, RoutingResult dataclasses
├── solution.py        # Solution building from routing results
├── multihop.py        # Multi-hop routing logic
└── handlers/          # Pool-specific handlers
    ├── __init__.py
    ├── base.py        # PoolHandler protocol
    ├── v2.py          # UniswapV2Handler
    ├── v3.py          # UniswapV3Handler
    └── balancer.py    # BalancerHandler
```

Each handler implements the `PoolHandler` protocol:

```python
class PoolHandler(Protocol):
    """Protocol for pool-specific routing handlers."""

    def route(
        self,
        order: Order,
        pool: AnyPool,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route an order through a specific pool."""
        ...
```

## Test Results

**Total: 651 tests passing, 14 skipped**

All existing tests continue to pass after refactoring.

## Files Modified/Created

### Phase 1 (Balancer Split)
```
solver/amm/balancer/           (new package)
├── __init__.py
├── types.py
├── weighted.py
├── stable.py
├── parsing.py
└── handler.py
solver/amm/balancer.py         (deleted)
```

### Phase 2 (Pools Package)
```
solver/pools/                  (new package)
├── __init__.py
├── registry.py
└── types.py
```

### Phase 3 (SwapCalculator)
```
solver/amm/base.py             (SwapCalculator protocol added)
solver/amm/uniswap_v2.py       (protocol-compliant methods added)
```

### Phase 4 (Router Simplification)
```
solver/routing/router.py       (simplified imports)
solver/routing/multihop.py     (simplified imports)
```

### Phase 5 (Test Split)
```
tests/unit/test_balancer_types.py      (new)
tests/unit/test_balancer_weighted.py   (new)
tests/unit/test_balancer_stable.py     (new)
tests/unit/test_balancer_parsing.py    (new)
tests/unit/test_balancer_handler.py    (new)
tests/unit/test_balancer.py            (deleted)
```

### Phase 6 (Router Handlers)
```
solver/routing/handlers/       (new package)
├── __init__.py
├── base.py
├── v2.py
├── v3.py
└── balancer.py
solver/routing/router.py       (facade pattern)
solver/routing/solution.py     (extracted)
solver/routing/multihop.py     (extracted)
solver/routing/types.py        (dataclasses)
```

## Key Insights

### Module Size Guidelines
- Target: < 300 LOC per module
- Split when: 4+ distinct concerns or > 500 LOC
- Use `__init__.py` re-exports for backward compatibility

### Protocol-Based Design
- Protocols enable duck typing with type safety
- AMMs implement SwapCalculator for uniform interface
- Handlers implement PoolHandler for router dispatch

### Backward Compatibility
All imports continue to work:
```python
from solver.amm.balancer import BalancerWeightedPool  # Still works
from solver.pools import AnyPool, PoolRegistry         # Still works
```

## What's Next

- Further architecture improvements (V3 module split, handler registry)
- Consider Curve pools or multi-order optimization

## Commits

- `468b18f` refactor: split balancer.py into package (Phase 1)
- `084b748` refactor: create solver/pools package for PoolRegistry (Phase 2)
- `b425671` refactor: add SwapCalculator protocol and standardize AMM interfaces (Phase 3)
- `a103a13` refactor: simplify router dispatch and use proper AnyPool type (Phase 4)
- `28e532f` refactor: split test_balancer.py into 5 focused test files (Phase 5)
- `7c7af7c` chore: remove architecture refactoring plan (completed)
- `2cfed93` refactor: split router.py into focused modules with handler pattern
