# Session 52: Slice 4.6 - Multi-Pair Price Coordination & EBBO Integration

**Date:** 2026-01-24
**Slice:** 4.6 - Multi-Pair Price Coordination

## Summary

Completed multi-pair price coordination strategy (`MultiPairCowStrategy`) and integrated strict EBBO (Ethereum Best Bid/Offer) validation with zero tolerance across all CoW matching strategies.

## Completed

### Multi-Pair Price Coordination

- [x] Created `solver/strategies/multi_pair.py` (~900 lines)
  - Union-Find for connected component detection
  - `find_token_components()` - groups overlapping token pairs
  - `PriceCandidates` dataclass and price enumeration
  - `solve_fills_at_prices()` using scipy LP solver
  - `MultiPairCowStrategy` class integrating all pieces
- [x] Created `tests/unit/strategies/test_multi_pair.py` (28 tests)
- [x] Integrated into `solver/solver.py` (replaces HybridCowStrategy position)
- [x] Fixed Balancer parsing for decimal scaling factors

### EBBO Validation Integration

- [x] Changed `EBBO_TOLERANCE` from `0.001` to `0` (zero tolerance)
- [x] Added Solver-level EBBO filtering via `_filter_ebbo_violations()` method
- [x] Added EBBO checking to `HybridCowStrategy` (integer comparison for proper rounding)
- [x] Added EBBO checking to `MultiPairCowStrategy` (integer comparison)
- [x] Updated `RingTradeStrategy`:
  - Changed tolerance from 0.1% to 0%
  - Added ability to build router from auction liquidity for EBBO checks
- [x] Updated tests to work with zero EBBO tolerance

## Test Results

```
978 passed, 14 skipped
```

All strategy tests pass including:
- 191 strategy-specific tests
- 28 multi-pair tests
- 34 ring trade tests
- 7 hybrid cow strategy tests

## Benchmark Results (10 auctions, 56,289 orders)

| Strategy | Matched | Rate | EBBO Compliance | Avg Time |
|----------|---------|------|-----------------|----------|
| CowMatch | 0 | 0.00% | N/A | 0.00ms |
| HybridCow | 0 | 0.00% | N/A | 19.61ms |
| MultiPair | 40 | 0.07% | **100%** | 50.65ms |
| RingTrade | 70 | 0.12% | **100%** | 693.02ms |

### Key Observations

1. **EBBO Compliance**: Both MultiPair and RingTrade achieve 100% EBBO compliance with zero tolerance
2. **HybridCow at 0%**: With strict EBBO (zero tolerance), integer rounding in clearing calculations causes all HybridCow matches to fail EBBO validation - this is correct behavior
3. **RingTrade reduced**: Matches dropped from 97 to 70 after adding EBBO validation (27 violations filtered)
4. **Gap remains large**: Best strategy matches 0.12% vs 36.53% CoW potential

### EBBO Validation Approach

The EBBO check uses integer comparison to properly handle rounding:

```python
# Sellers of A receive total_cow_b tokens B
# AMM would give them: int(total_cow_a * amm_price)
# EBBO satisfied if: total_cow_b >= int(total_cow_a * amm_price)
amm_equivalent = int(Decimal(total_cow_a) * amm_price)
if total_cow_b < amm_equivalent:
    # Reject - EBBO violation
```

This approach:
- Uses integer arithmetic like the actual clearing
- Zero tolerance means any deficit rejects the match
- Properly accounts for rounding in the clearing calculation

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `solver/strategies/multi_pair.py` | Create | Multi-pair coordination strategy |
| `tests/unit/strategies/test_multi_pair.py` | Create | 28 unit tests |
| `solver/ebbo.py` | Modify | EBBO_TOLERANCE = 0 |
| `solver/solver.py` | Modify | Add EBBO filtering, ebbo_prices param |
| `solver/strategies/hybrid_cow.py` | Modify | Add EBBO validation |
| `solver/strategies/ring_trade.py` | Modify | Zero tolerance, build router from auction |
| `solver/amm/balancer/parsing.py` | Modify | Handle decimal scaling factors |
| `tests/unit/strategies/test_hybrid_cow_strategy.py` | Modify | Update for zero tolerance |

## Architecture Decisions

1. **EBBO at Multiple Levels**:
   - Strategy-level: Early rejection for efficiency
   - Solver-level: Universal filter for all strategies

2. **Integer Comparison for EBBO**:
   - Floating point comparison fails due to rounding
   - Integer comparison: `actual >= int(amount * rate)` handles rounding properly

3. **RingTrade Router Building**:
   - Builds router from auction liquidity when not injected
   - Enables EBBO checking even when called standalone

## Next Steps

1. Further optimization of MultiPairCowStrategy for larger components
2. Investigate why gap remains large (36.53% potential vs 0.12% matched)
3. Consider relaxing EBBO tolerance if needed (currently zero)
