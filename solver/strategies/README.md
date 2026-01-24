# Solver Strategies

This directory contains the strategy implementations for the CoW solver.

## Strategy Chain

The solver uses a strategy chain where each strategy processes remaining orders:

```python
strategies = [
    CowMatchStrategy(),      # 1. 2-order direct matching (fast path)
    MultiPairCowStrategy(),  # 2. N-order joint optimization (LP-based)
    AmmRoutingStrategy(),    # 3. AMM routing fallback
]
```

## Constraint Enforcement

All strategies must enforce four constraints:

| Constraint | Definition | Validation Method |
|------------|------------|-------------------|
| **Fill-or-Kill (FOK)** | `partially_fillable=false` orders fully filled or not at all | Check `fill_amount == order_amount` |
| **Limit Price** | Actual rate >= limit rate for the user | Integer cross-multiply: `buy_filled * sell_amount >= buy_amount * sell_filled` |
| **EBBO** | Clearing rate >= AMM rate (zero tolerance) | Integer comparison, no epsilon |
| **Uniform Price** | All orders in a pair execute at same clearing price | Single price per token in solution |

### Safety Nets

Even if a strategy misses a constraint, there are safety nets:

1. **`base.py`**: `_verify_limit_price()` checks fills before returning
2. **`solver.py`**: `EBBOValidator` filters EBBO violations from final solution
3. **Logging**: Violations logged as ERROR for monitoring

## Production Strategies

### CowMatchStrategy (`cow_match.py`)

Fast path for 2-order peer-to-peer matching.

- **Input**: Auction with orders
- **Output**: Direct CoW match (2 trades, 0 interactions)
- **Constraints**: All four enforced
- **EBBO**: Validated via router reference price

### MultiPairCowStrategy (`multi_pair.py`)

N-order joint optimization across overlapping pairs using LP solver.

- **Input**: Auction with multiple orders
- **Output**: Optimal fills across all pairs
- **Constraints**: All four enforced
- **EBBO**: Two-sided bounds from `get_ebbo_bounds()`

### AmmRoutingStrategy (`amm_routing.py`)

Single-order routing through AMM pools (V2, V3, Balancer, limit orders).

- **Input**: Single order
- **Output**: AMM swap interaction
- **Constraints**: Limit price enforced, EBBO implicit (uses AMM rate)

## Research Strategies

### UnifiedCowStrategy (`unified_cow.py`)

Experimental unified optimization for pairs and cycles.

- **Status**: Research only, not in default chain
- **Constraints**: All four enforced

### RingTradeStrategy (`ring_trade.py`)

Cyclic trade detection (A→B→C→A).

- **Status**: Research only, low ROI (~0.12% match rate)
- **Constraints**: Limit price enforced, EBBO validated

## Adding a New Strategy

1. Inherit from base or implement `try_solve(auction) -> StrategyResult | None`
2. Enforce all four constraints
3. Add tests in `tests/unit/strategies/test_<name>.py`
4. Add integration tests in `tests/integration/test_constraint_enforcement.py`
5. Document constraint handling in docstring
