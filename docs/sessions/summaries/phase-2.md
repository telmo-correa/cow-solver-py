# Phase 2: CoW Matching

**Sessions:** 10-15
**Dates:** 2026-01-21
**Status:** Complete

## Overview

Phase 2 implemented Coincidence of Wants (CoW) matching - the ability to settle orders peer-to-peer without AMM fees. This is a Python-only feature; the Rust baseline solver does not support CoW matching.

## Key Accomplishments

### Strategy Pattern (Session 10)
- `SolutionStrategy` protocol for extensible solution finding
- Strategy chain: CoW matching tried first, then AMM routing
- Clean separation of concerns

### CoW Matching (Sessions 10-11)
- Perfect match detection for all order type combinations:
  - sell-sell, sell-buy, buy-sell, buy-buy
- Uniform clearing price calculation
- Direct settlement (0 interactions, gas=0)

### Partial CoW + AMM Remainder (Session 12)
- `StrategyResult` and `OrderFill` dataclasses for composable results
- Remainder order generation with SHA-256 derived UIDs
- `original_uid` tracking for fill merging across strategies
- Fill-or-kill semantics enforcement
- Multi-order AMM routing with pool reserve updates

### AMM Partial Fills (Session 13)
- Exact calculation for optimal partial fills
- Outperforms Rust's binary search approach

### Data-Driven Matching Rules (Session 15)
- Constraint tables for auditability
- 62% code reduction in `cow_match.py` (703 → 267 lines)
- Self-documenting rule definitions

## Final Metrics

| Metric | Value |
|--------|-------|
| Tests | 202 passing |
| CoW Match Tests | 47 tests |
| Matching Rules Tests | 35 tests |

## Key Files Created

```
solver/strategies/
├── base.py            # SolutionStrategy, StrategyResult, OrderFill
├── cow_match.py       # CoW matching strategy
├── matching_rules.py  # Data-driven constraint tables
└── amm_routing.py     # AMM routing strategy
```

## Architecture Decisions

### Remainder Order UIDs
```python
hash_input = f"{original_uid}:remainder".encode()
new_uid = hashlib.sha256(hash_input).hexdigest()
```
Deterministic derivation prevents collisions while `original_uid` enables fill merging.

### Fill Merging
When combining strategy results, fills are merged by `original_uid`:
1. CoW fills order partially → creates remainder with `original_uid`
2. AMM fills remainder → uses remainder's UID
3. `combine()` merges both fills using `original_uid` as key

### Price Validation
Validates actual fill rates (not clearing prices):
```python
# Sell order: actual rate must be >= limit rate
if buy_filled * sell_amount < buy_amount * sell_filled:
    raise PriceWorsened(...)
```

### Solver Composition Flow
```python
for strategy in [CoW, AMM]:
    result = strategy.try_solve(current_auction)
    if result.has_fills:
        results.append(result)
        current_auction = sub_auction(result.remainder_orders)
combined = StrategyResult.combine(results)
```

## Sessions Index

| Session | Focus | Key Outcome |
|---------|-------|-------------|
| 10 | CoW Matching | Strategy pattern, 2-order matching |
| 11 | Buy Order Support | All order type combinations |
| 12 | Partial CoW + AMM | Composable strategies, remainder routing |
| 13 | AMM Partial Fills | Exact calculation (beats Rust) |
| 14 | Code Quality | Refactoring, error handling |
| 15 | Data-Driven Rules | Constraint tables, 62% code reduction |

See `archive/session-10.md` through `archive/session-15.md` for detailed session logs.
