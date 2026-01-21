# Session 12 - Partial CoW + AMM Remainder (Slice 2.2)
**Date:** 2026-01-21

## Completed
- [x] Designed composable strategy architecture (StrategyResult, OrderFill)
- [x] Implemented StrategyResult dataclass with fill merging via `original_uid`
- [x] Implemented OrderFill dataclass with remainder order generation
- [x] Added `original_uid` field to Order model for tracking remainder origins
- [x] Generate new UIDs for remainder orders (SHA-256 derived, deterministic)
- [x] Added PriceWorsened exception for limit price validation
- [x] Validate fill rates against order limits in StrategyResult.combine()
- [x] Updated SolutionStrategy protocol to return StrategyResult | None
- [x] Updated CowMatchStrategy with partial matching for ALL order types:
  - sell-sell, sell-buy, buy-sell, buy-buy combinations
- [x] Implemented fill-or-kill semantics (partiallyFillable=false enforcement)
- [x] Updated AmmRoutingStrategy to return StrategyResult
- [x] Added multi-order AMM routing with pool reserve updates
- [x] Refactored AmmRoutingStrategy with `_get_router()` and `_route_and_build()`
- [x] Updated Solver to compose StrategyResults via sub-auctions
- [x] Updated existing CoW match tests for new architecture
- [x] Added comprehensive partial matching tests
- [x] Added integration tests for partial CoW + AMM composition
- [x] Created benchmark fixtures for Python-only features

## Test Results
- Passing: 147/147
- Failing: 0

## Benchmark Results
| Metric | Python | Rust | Notes |
|--------|--------|------|-------|
| CoW matching | Supported | Not supported | Python-only feature |
| Partial CoW + AMM | Supported | Not supported | Python-only feature |

## Files Created
```
tests/fixtures/auctions/benchmark_python_only/partial_cow_amm.json   # Partial CoW + AMM
tests/fixtures/auctions/benchmark_python_only/fok_perfect_match.json # Fill-or-kill perfect match
tests/fixtures/auctions/benchmark_python_only/mixed_partial_fok.json # Mixed partial + FoK
```

## Files Modified
```
solver/models/auction.py         # Added original_uid field
solver/strategies/base.py        # Added OrderFill, StrategyResult, PriceWorsened
solver/strategies/cow_match.py   # Partial matching for all order types
solver/strategies/amm_routing.py # Multi-order routing, reserve updates, refactored
solver/strategies/__init__.py    # Export new types
solver/routing/router.py         # Solver composes StrategyResults via sub-auctions
tests/unit/test_cow_match.py     # Comprehensive partial matching tests
tests/integration/test_single_order.py  # Partial CoW + AMM integration tests
PLAN.md                          # Marked Slice 2.2 complete
CLAUDE.md                        # Updated status
BENCHMARKS.md                    # Added partial CoW fixture description
docs/sessions/README.md          # Added session 12
```

## Key Design Decisions

### Remainder Order UID Generation
Remainder orders get new UIDs derived via SHA-256:
```python
hash_input = f"{original_uid}:remainder".encode()
hash_bytes = hashlib.sha256(hash_input).digest()
# Extend to 56 bytes with second hash
```
This prevents UID collisions while `original_uid` field enables fill merging.

### Fill Merging via original_uid
When combining strategy results, fills are merged by `original_uid`:
- CoW fills order partially (creates remainder with new UID, tracks `original_uid`)
- AMM fills remainder (uses remainder's UID)
- `combine()` merges both fills using `original_uid` as key

### Price Validation
Validates actual fill rates, not clearing prices:
```python
# Sell order: buy_filled/sell_filled >= buy_amount/sell_amount
if fill.buy_filled * sell_amount < buy_amount * fill.sell_filled:
    raise PriceWorsened(...)
```

### Fill-or-Kill Semantics
Orders with `partiallyFillable=false` (default):
- Must be completely filled or not at all
- CAN participate in partial matches IF they get completely filled
- The OTHER order can be partially filled if it has `partiallyFillable=true`

### Partial Matching for All Order Types
- **sell-sell**: Check limit compatibility, fill constraining order completely
- **sell-buy**: A offers more than B wants → A partial, B complete
- **buy-sell**: B offers more than A wants → B partial, A complete
- **buy-buy**: One side can satisfy the other → that side complete

### Multi-Order AMM Routing
Routes each order independently, updating pool reserves after each swap:
```python
for order in orders:
    result = self._route_and_build(order, pool_registry)
    if result:
        self._update_reserves_after_swap(pool_registry, result.routing_result)
```

### Solver Composition Flow
```python
results = []
current_auction = auction
for strategy in strategies:
    result = strategy.try_solve(current_auction)
    if result and result.has_fills:
        results.append(result)
        if result.remainder_orders:
            current_auction = create_sub_auction(current_auction, result.remainder_orders)
        else:
            break
combined = StrategyResult.combine(results)
return SolverResponse(solutions=[combined.build_solution()])
```

## Code Quality
- mypy: No type errors
- ruff: All checks passed
- Refactored `AmmRoutingStrategy` to reduce code duplication

## Next Session
- Slice 2.3: Multi-Order CoW Detection
  - Build order flow graph (net demand per token pair)
  - Identify netting opportunities across N orders
  - Greedy matching algorithm
