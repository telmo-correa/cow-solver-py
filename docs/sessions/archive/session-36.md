# Session 36 - Architecture Improvements (Handlers, V3 Split, Registry)

**Date:** 2026-01-22

## Summary

Implemented three architectural improvements: consolidated routing handlers with shared base class, split UniswapV3 monolithic module into focused package, and created centralized HandlerRegistry for extensible pool type dispatch.

## Completed

### 1. Routing Handler Consolidation

Extracted ~40-50 lines of duplicated code from handlers to shared `BaseHandler` class.

**solver/routing/handlers/base.py:**
```python
class BaseHandler:
    """Base class with shared handler utilities."""

    def _error_result(self, order: Order, error: str) -> RoutingResult:
        return RoutingResult(
            order=order, amount_in=0, amount_out=0,
            pool=None, success=False, error=error,
        )

    def _build_hop(
        self, pool: AnyPool, order: Order, amount_in: int, amount_out: int
    ) -> HopResult:
        return HopResult(
            pool=pool,
            input_token=normalize_address(order.sell_token),
            output_token=normalize_address(order.buy_token),
            amount_in=amount_in,
            amount_out=amount_out,
        )

    def _build_success_result(
        self, order: Order, pool: AnyPool,
        amount_in: int, amount_out: int, gas_estimate: int
    ) -> RoutingResult:
        hop = self._build_hop(pool, order, amount_in, amount_out)
        return RoutingResult(
            order=order, amount_in=amount_in, amount_out=amount_out,
            pool=pool, pools=[pool], hops=[hop],
            success=True, gas_estimate=gas_estimate,
        )
```

All handlers now inherit from `BaseHandler`:
- `UniswapV2Handler(BaseHandler)`
- `UniswapV3Handler(BaseHandler)`
- `BalancerHandler(BaseHandler)`

### 2. UniswapV3 Module Split

Split `solver/amm/uniswap_v3.py` (1031 LOC) into package with 6 focused modules:

```
solver/amm/uniswap_v3/
├── __init__.py      # Re-exports for backward compatibility (72 LOC)
├── constants.py     # Fee tiers, addresses (37 LOC)
├── pool.py          # UniswapV3Pool dataclass (69 LOC)
├── quoter.py        # Protocol, Mock, Web3 quoters (313 LOC)
├── encoding.py      # SwapRouter calldata (183 LOC)
├── amm.py           # UniswapV3AMM class (294 LOC)
└── parsing.py       # parse_v3_liquidity (210 LOC)
```

**Module Contents:**

| Module | LOC | Contents |
|--------|-----|----------|
| constants.py | 37 | Fee tiers, addresses, gas costs |
| pool.py | 69 | UniswapV3Pool dataclass |
| quoter.py | 313 | UniswapV3Quoter protocol, MockQuoter, Web3Quoter |
| encoding.py | 183 | SwapRouterV2 calldata encoding |
| amm.py | 294 | UniswapV3AMM class |
| parsing.py | 210 | parse_v3_liquidity function |

Largest module reduced from 1031 LOC to 313 LOC.

### 3. Handler Registry for Extensible Dispatch

Created `solver/routing/registry.py` with `HandlerRegistry` class that eliminates isinstance chains:

```python
class HandlerRegistry:
    """Registry for pool-specific routing handlers and simulators."""

    def register(
        self,
        pool_type: type,
        handler: PoolHandler,
        simulator: SwapSimulator,
        exact_output_simulator: ExactOutputSimulator | None = None,
        type_name: str = "unknown",
        gas_estimate: Callable[[AnyPool], int] | None = None,
    ) -> None:
        """Register a handler for a pool type."""

    def get_handler(self, pool: AnyPool) -> PoolHandler | None:
        """Get handler for a pool by its type."""

    def simulate_swap(
        self, pool: AnyPool, token_in: str, token_out: str, amount_in: int
    ) -> SwapResult | None:
        """Simulate swap using registered simulator."""

    def get_type_name(self, pool: AnyPool) -> str:
        """Get human-readable pool type name."""

    def get_gas_estimate(self, pool: AnyPool) -> int:
        """Get gas estimate for a pool."""

    def is_registered(self, pool: AnyPool) -> bool:
        """Check if a handler is registered for this pool type."""
```

**Usage in Router:**

```python
# Registration in SingleOrderRouter.__init__
self._handler_registry = self._build_handler_registry()

# _build_handler_registry() registers all pool types:
registry.register(
    UniswapV2Pool,
    handler=self._v2_handler,
    simulator=lambda p, ti, _to, ai: self.amm.simulate_swap(p, ti, ai),
    exact_output_simulator=lambda p, ti, _to, ao: self.amm.simulate_swap_exact_output(p, ti, ao),
    type_name="v2",
    gas_estimate=lambda _: POOL_SWAP_GAS_COST,
)

# Dispatch in routing:
def _route_through_pool(self, order, pool, sell_amount, buy_amount):
    handler = self._handler_registry.get_handler(pool)
    if handler is None:
        return self._error_result(order, f"No handler for {type(pool)}")
    return handler.route(order, pool, sell_amount, buy_amount)
```

**Benefits:**
- Adding new pool type: Only 2 file changes (pool class + register call)
- Previously: 8+ file changes with isinstance chains everywhere
- Centralized simulation dispatch for multihop routing
- Type-safe with proper protocols

## Test Results

**Total: 651 tests passing, 14 skipped**

All tests pass including:
- 60 router tests
- 159 Balancer tests
- 53 V3 tests

## Files Created/Modified

### Handler Consolidation
```
solver/routing/handlers/base.py     # Added BaseHandler class
solver/routing/handlers/v2.py       # Inherit BaseHandler
solver/routing/handlers/v3.py       # Inherit BaseHandler
solver/routing/handlers/balancer.py # Inherit BaseHandler
```

### V3 Module Split
```
solver/amm/uniswap_v3/              (new package)
├── __init__.py                     # Re-exports
├── constants.py                    # Fee tiers, addresses
├── pool.py                         # UniswapV3Pool
├── quoter.py                       # Quoter protocol and implementations
├── encoding.py                     # SwapRouter calldata
├── amm.py                          # UniswapV3AMM
└── parsing.py                      # parse_v3_liquidity
solver/amm/uniswap_v3.py            (deleted)
```

### Handler Registry
```
solver/routing/registry.py          (new - 176 LOC)
solver/routing/router.py            (use registry)
solver/routing/multihop.py          (use registry)
solver/routing/__init__.py          (export HandlerRegistry)
```

## Technical Notes

### Type Variance with Lambdas

The registry uses lambdas that accept `AnyPool` but call methods expecting specific types:

```python
simulator=lambda p, ti, _to, ai: self.amm.simulate_swap(p, ti, ai),  # type: ignore[arg-type]
```

The `type: ignore` is needed because:
- Registry callable signature: `(AnyPool, str, str, int) -> SwapResult`
- Actual method signature: `(UniswapV2Pool, str, int) -> SwapResult`

At runtime, the registry only calls the simulator with the correct pool type.

### Unused Lambda Parameters

V2/V3 simulators don't use `token_out` (derived from pool), so we use `_to`:

```python
lambda p, ti, _to, ai: self.amm.simulate_swap(p, ti, ai)
```

Balancer simulators need it:
```python
lambda p, ti, to, ai: weighted_amm.simulate_swap(p, ti, to, ai)
```

## What's Next

- Architecture refactoring complete
- Consider Phase 4: Multi-order CoW detection
- Consider additional liquidity sources (Curve pools)

## Commits

- `a7f569e` refactor: architecture improvements (handlers, V3 split, registry)
