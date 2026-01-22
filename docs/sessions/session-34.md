# Session 34 - Balancer Benchmarking (Slice 3.2.8)

**Date:** 2026-01-22

## Summary

Completed Slice 3.2.8: Ran Python vs Rust benchmarks on Balancer weighted and stable pool fixtures. All 5 Balancer tests produce exact match with Rust solver output. Fixed critical bug where default solver wasn't instantiating Balancer AMMs.

## Completed

### Bug Fix: Default Solver Missing Balancer AMMs

The default solver created by `_create_default_solver()` wasn't instantiating Balancer AMMs, causing all Balancer pool routing to fail silently (no solutions returned).

**Before (broken):**
```python
def _create_default_solver() -> Solver:
    rpc_url = os.environ.get("RPC_URL")
    if rpc_url:
        return Solver(v3_amm=v3_amm)  # No Balancer AMMs!
    else:
        return Solver()  # No Balancer AMMs!
```

**After (fixed):**
```python
def _create_default_solver() -> Solver:
    from solver.amm.balancer import BalancerWeightedAMM, BalancerStableAMM

    # Balancer AMMs are always enabled (local math, no RPC)
    weighted_amm = BalancerWeightedAMM()
    stable_amm = BalancerStableAMM()

    rpc_url = os.environ.get("RPC_URL")
    if rpc_url:
        return Solver(v3_amm=v3_amm, weighted_amm=weighted_amm, stable_amm=stable_amm)
    else:
        return Solver(weighted_amm=weighted_amm, stable_amm=stable_amm)
```

### Benchmark Results

| Test | Pool Type | Python | Rust | Status |
|------|-----------|--------|------|--------|
| weighted_gno_to_cow | V0 Weighted | 1657855325872947866705 | 1657855325872947866705 | ✅ Match |
| weighted_v3plus | V3Plus Weighted | 1663373703594405548696 | 1663373703594405548696 | ✅ Match |
| stable_dai_to_usdc | Stable (sell) | 9999475 | 9999475 | ✅ Match |
| stable_buy_order | Stable (buy) | 10000524328839166557 | 10000524328839166557 | ✅ Match |
| stable_composable | Composable Stable | 10029862202766050434 | 10029862202766050434 | ✅ Match |

**Full benchmark (18 test cases):**
- Matching: 15/18
- Improvements: 2/18 (Python partial fill exact calculation)
- Regressions: 0/18

### Documentation Updates

- Updated BENCHMARKS.md feature table: Balancer weighted and stable marked as ✅ for Python
- Updated benchmark results section with all 18 test cases including Balancer

## Test Results

**Total: 647 tests passing, 14 skipped**

## Files Modified

```
solver/solver.py             +12  # Add Balancer AMM instantiation to default solver
BENCHMARKS.md               +40  # Update feature table and benchmark results
docs/sessions/session-34.md      (new)
docs/sessions/README.md      +1  # Add session entry
```

## Key Insights

### Why Balancer AMMs Need Explicit Instantiation

Unlike V3 which requires RPC for quoting, Balancer pools use local math:
- Weighted pools: Fixed-point power functions with Bfp class
- Stable pools: StableSwap invariant with amplification parameter

Both can be instantiated without external dependencies, so they should **always** be enabled. The bug was that they were only passed through DI when explicitly provided, but the default solver creation didn't instantiate them.

### Benchmark Infrastructure

The benchmark comparison correctly identified the issue:
```
weighted_gno_to_cow:
  Python [○]: 6.1ms, solutions=0      <- No Balancer AMM = silent failure
  Rust   [✓]: 1.7ms, solutions=1
  Result [✗]: Python returned no solutions, Rust did
```

After fix:
```
weighted_gno_to_cow:
  Python [✓]: 6.7ms, solutions=1      <- Balancer AMM working
  Rust   [✓]: 2.4ms, solutions=1
  Result [✓]: Solutions match
```

## What's Next

- **Slice 3.2.8 Complete** - All Balancer benchmarks pass with exact Rust match
- Phase 3 Balancer integration is now complete (Slices 3.2.1-3.2.8)
- Consider Phase 4: Multi-order CoW detection or Curve pools

## Commits

- feat: enable Balancer AMMs in default solver and update benchmarks (Slice 3.2.8)
