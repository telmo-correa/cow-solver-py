# Pre-Phase 4 Architecture Refactoring

**Session:** 42
**Date:** 2026-01-23
**Status:** Complete

## Overview

Architecture refactoring to prepare the codebase for Phase 4 (Unified Optimization). Extracted pathfinding to a dedicated module, added order grouping utilities, improved type safety, and deduplicated handler code.

## Key Accomplishments

### 1. PathFinder Module (`solver/routing/pathfinding.py`)
- Extracted graph-building and pathfinding from `PoolRegistry` to dedicated `PathFinder` class
- `TokenGraph` class for adjacency list representation
- `PoolGraphSource` Protocol documenting the registry interface
- Lazy initialization with automatic cache invalidation
- Changed default `max_hops` from 2 to 3 (supports 3-hop routes)

### 2. OrderGroup Utility (`solver/models/order_groups.py`)
- `OrderGroup` dataclass for grouping orders by token pair
- `group_orders_by_pair()` function for canonical pair grouping
- `find_cow_opportunities()` for detecting N-order CoW potential
- Properties: `has_cow_potential`, `total_sell_a/b`, `total_buy_a/b`, `all_orders`
- Foundation for Phase 4's batch optimization

### 3. PoolRegistry Improvements
- Constructor now accepts `list[AnyPool]` (was `list[UniswapV2Pool]`)
- Added `add_any_pool()` method for type-dispatched pool addition
- Pathfinding methods delegate to `PathFinder`
- Cleaner separation: storage vs. graph operations

### 4. Balancer Handler Deduplication
- Extracted common logic into `_route_balancer_pool()`
- Added `_route_sell_order()` and `_route_buy_order()` helpers
- Removed duplicate `_try_partial_weighted()` and `_try_partial_stable()`
- Type aliases: `BalancerAMM`, `BalancerPool`
- ~40% code reduction in handler

### 5. Type-Safe Simulator Protocols
- Changed `SwapSimulator` and `ExactOutputSimulator` from Callable aliases to Protocols
- Added `GasEstimator` Protocol
- Detailed documentation explaining `type: ignore` necessity
- Better IDE support and self-documenting code

## Final Metrics

| Metric | Value |
|--------|-------|
| Tests | 799 passing, 14 skipped |
| New Source Files | 2 (`pathfinding.py`, `order_groups.py`) |
| New Test Files | 3 (`test_pathfinding.py`, `test_order_groups.py`, `test_registry.py`) |
| Lines Added | ~750 (source + tests) |
| Lines Removed | ~180 (deduplication) |

## Files Created

```
solver/models/order_groups.py      # OrderGroup, group_orders_by_pair, find_cow_opportunities
solver/routing/pathfinding.py      # TokenGraph, PathFinder, PoolGraphSource Protocol

tests/unit/models/test_order_groups.py    # 19 tests
tests/unit/pools/test_registry.py         # 11 tests
tests/unit/routing/test_pathfinding.py    # 18 tests
```

## Files Modified

```
solver/models/__init__.py          # Export OrderGroup utilities
solver/pools/registry.py           # AnyPool constructor, PathFinder delegation
solver/routing/__init__.py         # Export PathFinder, TokenGraph
solver/routing/handlers/balancer.py # Deduplicated routing logic
solver/routing/registry.py         # Protocol classes for simulators
```

## Architecture Decisions

### PathFinder Separation
The pathfinding logic was tightly coupled to `PoolRegistry`. Extracting it to `PathFinder`:
- Enables future graph algorithms (ring detection, split routing)
- Clearer responsibility: Registry stores pools, PathFinder finds routes
- Testable in isolation with mock registries

### OrderGroup for Batch Optimization
Phase 4 requires analyzing orders in aggregate. `OrderGroup` provides:
- Canonical token pair grouping (consistent regardless of order direction)
- Aggregate metrics (`total_sell_a`, `total_buy_b`, etc.)
- CoW opportunity detection (`has_cow_potential`)
- Foundation for N-order matching algorithms

### Protocol vs Callable
Using `Protocol` classes instead of `Callable` type aliases:
- Self-documenting parameter names
- IDE autocomplete for implementers
- Explicit documentation of the contract

## What's Next

Phase 4: Unified Optimization
- N-order CoW detection using `OrderGroup`
- Split routing across multiple paths
- Ring trade detection using `TokenGraph`
- Joint optimization across order groups
