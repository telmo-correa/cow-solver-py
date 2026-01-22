# Session 26: V3 Benchmarking and Documentation

**Date:** 2026-01-22
**Focus:** Run V3 benchmarks and document results permanently

## Summary

Completed the final slice of the V3 implementation plan (3.1.8) by running Python vs Rust benchmarks on V3 fixtures and documenting results in BENCHMARKS.md.

## Problem

Slice 3.1.8 required:
- Create benchmark fixtures with V3 liquidity (done in earlier sessions)
- Run benchmark: Python vs Rust on V3 auctions (not done)
- Document results (not done)

V3 fixtures existed but had never been benchmarked against Rust.

## Solution

### V3 Benchmark Setup

**Python Solver:**
```bash
RPC_URL="https://eth.llamarpc.com" python -m solver.api.main
```

**Rust Solver:** Using `baseline_v3.toml` with `uni-v3-node-url`:
```toml
chain-id = "1"
base-tokens = ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]
max-hops = 1
max-partial-attempts = 5
native-token-price-estimation-amount = "100000000000000000"
uni-v3-node-url = "https://eth.llamarpc.com"
```

### Benchmark Results

| Test Case | Result |
|-----------|--------|
| v3_weth_to_usdc | ✓ Match |
| v3_usdc_to_weth | ✓ Match |
| v3_buy_weth | ✓ Match |
| v2_v3_comparison | ✓ Match |

**Summary:** 13 total tests (9 V2 + 4 V3)
- Matching: 11/13
- Improvements: 2/13 (partial fill exact calculation)
- Regressions: 0/13

**Performance:**
- V2 swaps: Python ~1.5x slower than Rust (pure computation)
- V3 swaps: Python ~2.9x slower than Rust (both use RPC, different implementations)

## Files Changed

| File | Change |
|------|--------|
| `BENCHMARKS.md` | Updated feature table (V3: ❌→⚠️), added V3 setup instructions, added V3 benchmark results |

## Key Decisions

1. **Document in BENCHMARKS.md**: User feedback that session files are not permanent enough for benchmark instructions
2. **V3 Performance Note**: V3 ~3x slower due to RPC latency (300-900ms per QuoterV2 call)

## What's Next

V3 implementation is complete. Options:
- **Slice 3.2**: Balancer/Curve Integration
- Continue with code quality improvements from PLAN.md

## Commits

- `b2562a5` - docs: add V3 benchmark results and setup instructions
