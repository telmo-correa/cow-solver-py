# Session 20 - UniswapV3 Router Integration
**Date:** 2026-01-22

## Completed
- [x] Slice 3.1.5: Router Integration
  - Updated `PoolRegistry` to store V3 pools with multi-fee-tier support
  - Added `add_v3_pool()`, `get_v3_pools()`, `get_pools_for_pair()` methods
  - Updated `SingleOrderRouter` to accept V3 AMM via `v3_amm` parameter
  - Implemented best-quote selection across V2 and V3 pools
  - Created `_find_best_direct_route()` for multi-pool comparison
  - Created `_route_through_v3_pool()` for V3-specific routing
  - Updated `build_registry_from_liquidity()` to parse both V2 and V3 pools
  - Updated `AmmRoutingStrategy` to skip V3 pool reserve updates (V3 uses quoter)
  - Added 10 new unit tests for V3 router integration

## Test Results
- Passing: 264/264
- New V3 router tests: 10 in `TestV3RouterIntegration`

## Key Implementation Details

### PoolRegistry V3 Support
```python
class PoolRegistry:
    def __init__(self):
        self._pools: dict[frozenset[str], UniswapV2Pool] = {}
        # V3 pools keyed by (token0, token1, fee) for multiple fee tiers
        self._v3_pools: dict[tuple[str, str, int], UniswapV3Pool] = {}

    def add_v3_pool(self, pool: UniswapV3Pool) -> None:
        """Add a V3 pool with canonical token ordering."""

    def get_v3_pools(self, token_a: str, token_b: str) -> list[UniswapV3Pool]:
        """Get all V3 pools for a token pair (all fee tiers)."""

    def get_pools_for_pair(
        self, token_a: str, token_b: str
    ) -> list[UniswapV2Pool | UniswapV3Pool]:
        """Get all pools (V2 + V3) for a token pair."""
```

### SingleOrderRouter V3 Support
```python
class SingleOrderRouter:
    def __init__(
        self,
        amm: UniswapV2 | None = None,
        v3_amm: UniswapV3AMM | None = None,  # NEW
        pool_registry: PoolRegistry | None = None,
        pool_finder: Callable | None = None,  # Deprecated, backward compat
    ):
        self.amm = amm or uniswap_v2
        self.v3_amm = v3_amm  # V3 disabled if None

    def _find_best_direct_route(
        self, order, pools, sell_amount, buy_amount
    ) -> RoutingResult | None:
        """Get quotes from all pools, select best by output (sell) or input (buy)."""

    def _route_through_v3_pool(
        self, order, pool, sell_amount, buy_amount
    ) -> RoutingResult:
        """Route order through V3 pool using V3 AMM."""
```

### Best Quote Selection
- For sell orders: maximize output amount
- For buy orders: minimize input amount
- Falls back to V2 when V3 quoter is unavailable or fails
- Logs pool selection when multiple candidates exist

### Routing Flow
```
route_order()
    |
    v
get_pools_for_pair() --> [V2 pool, V3 pool 500, V3 pool 3000, ...]
    |
    v
_find_best_direct_route() --> Get quotes from all, select best
    |
    v
_route_sell_order() or _route_through_v3_pool()
    |
    v
RoutingResult with selected pool
```

## Files Modified
```
solver/amm/uniswap_v2.py       # Added V3 pool storage, get_pools_for_pair (~60 lines)
solver/routing/router.py       # Added V3 AMM support, best-quote selection (~130 lines)
solver/strategies/amm_routing.py # Skip V3 pool reserve updates (~10 lines)
tests/unit/test_router.py      # Added TestV3RouterIntegration (10 tests)
tests/integration/test_single_order.py # Updated swap_call count expectation
tests/integration/test_api.py  # Updated swap_call count expectation
```

## Code Review Issues Fixed
1. ruff UP037: Remove quotes from type annotations (14 fixes with `--fix`)
2. mypy: Bug in `min()` key function (was selecting pool, now selects input amount)
3. mypy: List variance issues - added `cast()` for V2-only pools in multihop
4. mypy: `_create_updated_pool` type mismatch - added `isinstance()` check for V2 pools
5. Tests: Router now calls `simulate_swap` twice (quote + route) - updated expectations

## Behavior Changes
- Router now calls `simulate_swap` twice for single-hop routes:
  1. During `_find_best_direct_route()` for quote comparison
  2. During `_route_sell_order()` or `_route_buy_order()` for routing
- This is acceptable overhead for enabling multi-pool comparison

## Next Session
- Slice 3.1.6: Integration Tests with Mock Quoter
  - Create V3 auction fixtures
  - Integration test: V3-only solve
  - Integration test: V2 vs V3 selection
  - Integration test: API endpoint with V3
