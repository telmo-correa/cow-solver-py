# Session 37 - Test Coverage and Reorganization

**Date:** 2026-01-22

## Summary

Added test coverage for V3 + Balancer pool combinations, split large test file into focused modules, and reorganized test directory structure into logical subdirectories.

## Completed

### 1. V3 + Balancer Test Coverage

Added 11 new tests covering gaps in multi-source routing:

**TestV3MixedMultiHopRouting (7 tests):**
- `test_multihop_v2_then_v3` - V2 → V3 multi-hop
- `test_multihop_v3_then_v2` - V3 → V2 multi-hop
- `test_multihop_v3_then_weighted` - V3 → Balancer weighted
- `test_multihop_v3_then_stable` - V3 → Balancer stable
- `test_multihop_weighted_then_v3` - Balancer weighted → V3
- `test_multihop_v3_weighted_v3` - V3 → weighted → V3 (3 hops)
- `test_multihop_selects_best_v3_vs_weighted` - Best pool selection

**TestAllPoolTypesSelection (4 tests):**
- `test_selects_best_among_all_four_pool_types` - V2/V3/weighted/stable comparison
- `test_selects_v3_when_best_among_all_four` - V3 wins scenario
- `test_v3_vs_weighted_direct_comparison` - Direct V3 vs weighted
- `test_v3_vs_stable_direct_comparison` - Direct V3 vs stable (stablecoin pair)

### 2. Balancer Scaling Factor Fix

Fixed `scaling_factor` in test helpers for 6-decimal tokens (USDC):

```python
# Before (incorrect)
WeightedTokenReserve(
    token=self.USDC.lower(),
    balance=balance_usdc,
    weight=Decimal("0.5"),
    scaling_factor=1,  # Wrong for 6-decimal token
)

# After (correct)
WeightedTokenReserve(
    token=self.USDC.lower(),
    balance=balance_usdc,
    weight=Decimal("0.5"),
    scaling_factor=10**12,  # 6 decimals → 18 decimals
)
```

This ensures Balancer pools correctly scale 6-decimal inputs to 18-decimal internal math, then scale outputs back to native decimals.

### 3. Router Test File Split

Split `tests/unit/test_router.py` (2471 lines, 12 classes) into 6 focused files:

| File | Lines | Classes | Purpose |
|------|-------|---------|---------|
| `test_router_core.py` | 263 | 5 | Error handling, buy orders, build solution |
| `test_router_multihop.py` | 117 | 1 | Multi-hop routing |
| `test_router_di.py` | 137 | 1 | Dependency injection |
| `test_router_v3.py` | 269 | 1 | V3 integration |
| `test_router_balancer.py` | 357 | 1 | Balancer integration |
| `test_router_mixed.py` | 1160 | 3 | Mixed pool type tests |

### 4. Test Directory Reorganization

Reorganized `tests/unit/` from 19 flat files into 4 subdirectories:

```
tests/unit/
├── __init__.py
├── test_models.py
│
├── amm/                    (7 files)
│   ├── test_uniswap_v2.py  (renamed from test_amm.py)
│   ├── test_uniswap_v3.py
│   ├── test_balancer_amm.py
│   ├── test_balancer_parsing.py
│   ├── test_balancer_scaling.py
│   ├── test_balancer_stable_math.py
│   └── test_balancer_weighted_math.py
│
├── routing/                (6 files)
│   ├── test_core.py        (renamed from test_router_core.py)
│   ├── test_multihop.py
│   ├── test_di.py
│   ├── test_v3.py
│   ├── test_balancer.py
│   └── test_mixed.py
│
├── strategies/             (3 files)
│   ├── test_cow_match.py
│   ├── test_matching_rules.py
│   └── test_base.py        (renamed from test_strategy_base.py)
│
└── math/                   (3 files)
    ├── test_fixed_point.py
    ├── test_safe_int.py
    └── test_fee_calculator.py
```

**Benefits:**
- Groups related tests together
- Mirrors `solver/` package structure
- Removes redundant prefixes (e.g., `test_router_` → just `test_` in `routing/`)
- Easier navigation and discovery

## Test Results

**Total: 662 passed, 14 skipped**

All tests pass including the 11 new mixed-pool tests.

## Files Changed

### New Test Files
```
tests/unit/routing/test_core.py
tests/unit/routing/test_multihop.py
tests/unit/routing/test_di.py
tests/unit/routing/test_v3.py
tests/unit/routing/test_balancer.py
tests/unit/routing/test_mixed.py
```

### Moved/Renamed Files
```
test_amm.py → amm/test_uniswap_v2.py
test_uniswap_v3.py → amm/test_uniswap_v3.py
test_balancer_*.py → amm/test_balancer_*.py
test_router_*.py → routing/test_*.py
test_cow_match.py → strategies/test_cow_match.py
test_matching_rules.py → strategies/test_matching_rules.py
test_strategy_base.py → strategies/test_base.py
test_fixed_point.py → math/test_fixed_point.py
test_safe_int.py → math/test_safe_int.py
test_fee_calculator.py → math/test_fee_calculator.py
```

### Deleted Files
```
tests/unit/test_router.py (split into 6 files)
```

### Fixed Files
```
tests/unit/amm/test_uniswap_v3.py  # Updated fixture path after move
```

## Technical Notes

### Fixture Path Fix

After moving `test_uniswap_v3.py` to `amm/` subdirectory, the fixture path needed updating:

```python
# Before (in tests/unit/)
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "liquidity"

# After (in tests/unit/amm/)
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "liquidity"
```

### Scaling Factor Explanation

Balancer pools use `scaling_factor` to normalize all token amounts to 18 decimals for internal math:

- **18-decimal tokens (WETH, DAI):** `scaling_factor=1`
- **6-decimal tokens (USDC, USDT):** `scaling_factor=10**12`

The AMM's `scale_up()` multiplies by scaling_factor before math, and `scale_down_down()` divides after to return native decimals.

## What's Next

- All architecture improvements complete
- Test coverage comprehensive across all pool types
- Consider Phase 4: Multi-order CoW detection

## Commits

- (pending) test: add V3 + Balancer mixed routing tests
- (pending) refactor: split test_router.py into focused modules
- (pending) refactor: reorganize tests/unit into subdirectories
