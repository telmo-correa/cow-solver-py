# Slice 4.4 Ring Trade Detection - Benchmark Results

**Date:** 2026-01-23

## Summary

Ring trade detection was evaluated against 50 historical mainnet auctions.

| Metric | Value |
|--------|-------|
| Auctions analyzed | 50 |
| Total orders | 280,920 |
| 3-cycles detected | 17,186 |
| 4-cycles detected | 5,000 (capped at 100/auction) |
| **Orders matched** | **450** |
| **Match rate** | **0.16%** |

## Key Findings

### 1. Cycle Detection Works Well

Each auction has ~340 3-cycles and 100+ 4-cycles detected. The algorithm successfully finds potential ring trade opportunities.

### 2. Most Cycles Not Viable

Of ~340 cycles per auction:
- ~56 are viable (product ≤ 1)
- ~50 are near-viable (product 1.0-1.053)
- ~234 are not viable (product > 1.053)

This means only **16%** of detected cycles are immediately executable.

### 3. Token Overlap Constraint is the Bottleneck

The token overlap prevention (to avoid clearing price conflicts) severely limits ring selection:

| Selection Strategy | Orders Matched | Rate |
|--------------------|---------------|------|
| With token overlap (current) | 450 | 0.16% |
| Without token overlap | 2,007 | 0.71% |

Removing the constraint would provide **4.5x more matches**, but would require solving the clearing price consistency problem across shared tokens.

### 4. Near-Viable Cycles are Abundant

- **2,543 near-viable cycles** detected (product 1.0-1.053)
- These could potentially be closed with AMM assistance
- Would require the AMM to provide 1-5% price improvement

## Auction Characteristics

- ~5,600 orders per auction
- ~814 unique tokens per auction
- ~1,450 directed edges (token pairs with orders)

## Sample Viable Rings

From mainnet_11985000.json (selected by surplus):

| Tokens | Surplus | Product |
|--------|---------|---------|
| Token A → B → C | 67.2% | 0.328 |
| Token D → E → F | 49.7% | 0.503 |
| Token G → H → I | 30.3% | 0.697 |

## Comparison to Direct CoW

| Strategy | Match Rate | Notes |
|----------|------------|-------|
| Direct CoW (2-order) | 1.4% | Same token pair |
| Ring trades (current) | 0.16% | 3+ token cycles |
| Ring trades (no token overlap) | 0.71% | Needs price consistency |

## Recommendations

1. **Consider relaxing token overlap constraint** - With proper price consistency handling, could achieve 0.71% vs 0.16%

2. **Implement AMM-assisted rings** - 2,543 near-viable cycles per batch could be closed with AMM help

3. **Combine with direct CoW** - Ring trades complement direct CoW matching (different token patterns)

## Conclusion

Ring trade detection works correctly but provides modest value (0.16%) due to:
1. Most cycles have incompatible limit prices
2. Token overlap constraint limits non-overlapping selection
3. Real-world order flow doesn't often form viable multi-token cycles

The larger opportunity is in near-viable cycles (2,543) that could be closed with AMM assistance.
