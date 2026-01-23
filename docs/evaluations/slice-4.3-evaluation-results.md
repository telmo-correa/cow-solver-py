# Slice 4.3: Hybrid Strategy Evaluation Results

**Date:** 2026-01-23
**Auctions evaluated:** 10
**Total orders:** 500

## Summary

| Metric | Value |
|--------|-------|
| Hybrid wins | 0 (0.0%) |
| AMM wins | 0 (0.0%) |
| Ties | 5 (50.0%) |
| Neither | 5 (50.0%) |
| **Contested win rate** | **0.0%** |
| CoW matches | 7 |
| CoW match rate | 1.40% |
| Total gas savings | 0 |

## Initial Decision

**Initial Recommendation:** Consider Split Routing or Performance Optimization

Hybrid CoW adds marginal value. Focus on split routing for large orders or performance.

## Revised Decision (after ring trade analysis)

**Revised Recommendation:** Proceed to Ring Trades (Slice 4.4)

Follow-up analysis (`scripts/analyze_ring_potential.py`) showed ring trades could match **5.41%** of orders (3.9x better than direct CoW's 1.4%). See [slice-4.3-ring-analysis.md](slice-4.3-ring-analysis.md) for details.
