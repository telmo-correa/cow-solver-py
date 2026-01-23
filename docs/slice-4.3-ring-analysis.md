# Slice 4.3: Ring Trade Potential Analysis

**Date:** 2026-01-23

## Context

After the initial Slice 4.3 evaluation showed direct CoW matching adds marginal value (1.4% match rate), we analyzed whether ring trades (A→B→C→A cycles) would improve results.

## Methodology

Script: `scripts/analyze_ring_potential.py`

1. Build directed graph from orders (sell_token → buy_token)
2. Find 3-node and 4-node cycles
3. Check economic viability: product of limit prices around cycle >= 1
4. Count **unique orders** in viable cycles (no double-counting)

## Results

| Metric | Value |
|--------|-------|
| Auctions analyzed | 10 |
| Total orders | 56,289 |
| Auctions with viable cycles | 100% |
| 3-node cycles found | 3,460 |
| Viable 3-node cycles | 2,000 (57.8%) |
| 4-node cycles found | 2,853 |
| Viable 4-node cycles | 735 (25.8%) |
| **Unique orders in viable cycles** | **3,046** |
| **Ring match rate** | **5.41%** |

## Comparison to Direct CoW

| Matching Type | Match Rate | Improvement |
|---------------|------------|-------------|
| Direct CoW (A↔B) | 1.40% | — |
| Ring trades (A→B→C→A) | 5.41% | **3.9x** |

## Per-Auction Breakdown

| Auction | Tokens | 3-cycles | Viable | 4-cycles | Viable | Unique Orders |
|---------|--------|----------|--------|----------|--------|---------------|
| 11985000 | 814 | 344 | 199 | 285 | 73 | 305 |
| 11985001 | 814 | 343 | 199 | 285 | 74 | 305 |
| 11985002 | 814 | 343 | 199 | 285 | 74 | 305 |
| 11985003 | 814 | 343 | 199 | 285 | 74 | 305 |
| 11985004 | 814 | 342 | 198 | 285 | 74 | 303 |
| 11985005 | 814 | 349 | 202 | 286 | 74 | 305 |
| 11985006 | 814 | 349 | 202 | 286 | 74 | 305 |
| 11985007 | 814 | 349 | 202 | 286 | 74 | 305 |
| 11985008 | 814 | 349 | 200 | 285 | 72 | 304 |
| 11985009 | 814 | 349 | 200 | 285 | 72 | 304 |

## Key Insights

1. **Cycles are abundant**: Every auction has hundreds of viable cycles
2. **3-node cycles dominate**: 57.8% viable vs 25.8% for 4-node
3. **Consistent across auctions**: ~305 unique orders per auction participate in viable cycles
4. **5.41% exceeds threshold**: Crosses the 5% "significant value" bar from Slice 4.3 criteria

## Decision

**Proceed to Ring Trade Implementation (Slice 4.4)**

Ring trades find matches that direct CoW misses by exploiting cyclic relationships. The 3.9x improvement justifies implementation effort.

## Next Steps

1. Implement cycle detection in `RingTradeStrategy`
2. Calculate optimal execution amounts around cycles
3. Handle partial fills when cycle volumes don't match
4. Benchmark against AMM-only baseline
