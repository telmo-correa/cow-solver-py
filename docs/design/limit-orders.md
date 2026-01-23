# Design: 0x Limit Order Support (Slice 3.5)

## Overview

Add support for 0x Protocol foreign limit orders as a liquidity source, achieving full parity with the Rust baseline solver's 5 liquidity types.

## Background

### What are 0x Limit Orders?

0x Protocol allows market makers to post signed limit orders off-chain. These orders represent a commitment: "I'll give X tokens for Y tokens at this exact ratio." Unlike AMMs:

- **No slippage curve** — fixed exchange rate until filled
- **No on-chain state** — orders are signed messages, not contracts
- **Lower gas** — ~66k vs ~110k+ for AMM swaps
- **Fill-or-kill semantics** — orders execute fully or not at all (at solver level)

### Terminology

| Term | Description |
|------|-------------|
| **Maker** | Party who created the limit order (provides `makerToken`) |
| **Taker** | Party who fills the order (provides `takerToken`) |
| **makerAmount** | Amount of `makerToken` the maker will provide |
| **takerAmount** | Amount of `takerToken` the maker wants in return |
| **takerTokenFeeAmount** | Protocol fee paid in `takerToken` |

From the solver's perspective:
- **Input** = `takerToken` (what we're selling into the order)
- **Output** = `makerToken` (what we're getting from the order)

## Auction JSON Format

```json
{
  "kind": "limitOrder",
  "id": "0",
  "address": "0xdef1c0ded9bec7f1a1670819833240f027b25eff",
  "gasEstimate": "66358",
  "hash": "0x...",
  "makerToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
  "takerToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
  "makerAmount": "2500000000",
  "takerAmount": "1000000000000000000",
  "takerTokenFeeAmount": "2500000"
}
```

This example: Maker offers 2500 USDC for 1 WETH (+ 0.001% fee).

## Swap Math

The math is simple linear scaling — no AMM curves:

### Sell Order (get_amount_out)

Given input amount in `takerToken`, calculate output in `makerToken`:

```python
def get_amount_out(amount_in: int, maker_amount: int, taker_amount: int) -> int:
    """Calculate output for a sell order through limit order."""
    # amount_out = amount_in * maker_amount / taker_amount
    return (amount_in * maker_amount) // taker_amount
```

### Buy Order (get_amount_in)

Given desired output in `makerToken`, calculate required input in `takerToken`:

```python
def get_amount_in(amount_out: int, maker_amount: int, taker_amount: int) -> int:
    """Calculate required input for a buy order through limit order."""
    # amount_in = amount_out * taker_amount / maker_amount
    return (amount_out * taker_amount) // maker_amount
```

### Constraints

- `amount_in <= taker_amount` (cannot exceed order's taker limit)
- `amount_out <= maker_amount` (cannot exceed order's maker limit)
- Tokens must match exactly (no multi-hop through limit orders)

## Gas Estimate

Fixed constant from Dune analytics:

```python
GAS_PER_ZEROEX_ORDER = 66_358  # wei
```

Source: https://dune.com/queries/639669 (median gas per ZeroExInteraction)

## Implementation Plan

### 1. Data Model

**File:** `solver/pools/limit_order.py` (new)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LimitOrderPool:
    """A 0x Protocol foreign limit order used as liquidity."""

    id: str                    # Liquidity ID
    address: str               # 0x Exchange contract address
    maker_token: str           # Token the maker provides (output)
    taker_token: str           # Token the maker wants (input)
    maker_amount: int          # Max output available
    taker_amount: int          # Max input accepted
    taker_token_fee_amount: int  # Protocol fee in taker token
    gas_estimate: int          # ~66,358

    @property
    def token0(self) -> str:
        """Return taker_token (input token)."""
        return self.taker_token

    @property
    def token1(self) -> str:
        """Return maker_token (output token)."""
        return self.maker_token

    def supports_pair(self, token_in: str, token_out: str) -> bool:
        """Check if this order can route the given token pair."""
        return (
            token_in.lower() == self.taker_token.lower() and
            token_out.lower() == self.maker_token.lower()
        )
```

### 2. Parsing

**File:** `solver/pools/parsing.py` (modify)

Add parsing for `limitOrder` liquidity kind:

```python
def parse_limit_order(liquidity: dict) -> LimitOrderPool:
    """Parse a limit order from auction liquidity."""
    return LimitOrderPool(
        id=liquidity["id"],
        address=liquidity["address"],
        maker_token=liquidity["makerToken"],
        taker_token=liquidity["takerToken"],
        maker_amount=int(liquidity["makerAmount"]),
        taker_amount=int(liquidity["takerAmount"]),
        taker_token_fee_amount=int(liquidity["takerTokenFeeAmount"]),
        gas_estimate=int(liquidity.get("gasEstimate", 66358)),
    )
```

### 3. AMM Implementation

**File:** `solver/amm/limit_order.py` (new)

```python
from dataclasses import dataclass
from solver.amm.base import SwapResult
from solver.pools.limit_order import LimitOrderPool


@dataclass
class LimitOrderAMM:
    """Swap calculator for 0x limit orders."""

    def simulate_swap(
        self,
        pool: LimitOrderPool,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> SwapResult | None:
        """Simulate a swap through a limit order."""
        # Validate token pair
        if not pool.supports_pair(token_in, token_out):
            return None

        # Check input doesn't exceed order limit
        if amount_in > pool.taker_amount:
            return None

        # Calculate output: amount_in * maker_amount / taker_amount
        amount_out = (amount_in * pool.maker_amount) // pool.taker_amount

        # Check output doesn't exceed maker amount
        if amount_out > pool.maker_amount:
            return None

        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            gas_estimate=pool.gas_estimate,
        )

    def simulate_swap_exact_out(
        self,
        pool: LimitOrderPool,
        token_in: str,
        token_out: str,
        amount_out: int,
    ) -> SwapResult | None:
        """Simulate a swap with exact output (buy order)."""
        # Validate token pair
        if not pool.supports_pair(token_in, token_out):
            return None

        # Check output doesn't exceed order limit
        if amount_out > pool.maker_amount:
            return None

        # Calculate input: amount_out * taker_amount / maker_amount
        # Round up for buy orders
        amount_in = (amount_out * pool.taker_amount + pool.maker_amount - 1) // pool.maker_amount

        # Check input doesn't exceed taker amount
        if amount_in > pool.taker_amount:
            return None

        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            gas_estimate=pool.gas_estimate,
        )
```

### 4. Pool Registry Integration

**File:** `solver/pools/registry.py` (modify)

Add `LimitOrderPool` to the registry:

```python
from solver.pools.limit_order import LimitOrderPool

class PoolRegistry:
    def __init__(self):
        self._v2_pools: list[UniswapV2Pool] = []
        self._v3_pools: list[UniswapV3Pool] = []
        self._weighted_pools: list[BalancerWeightedPool] = []
        self._stable_pools: list[BalancerStablePool] = []
        self._limit_orders: list[LimitOrderPool] = []  # NEW

    def add_limit_order(self, order: LimitOrderPool) -> None:
        """Add a limit order to the registry."""
        self._limit_orders.append(order)

    def find_limit_orders(self, token_in: str, token_out: str) -> list[LimitOrderPool]:
        """Find limit orders that can route the given pair."""
        return [
            order for order in self._limit_orders
            if order.supports_pair(token_in, token_out)
        ]
```

### 5. Router Handler

**File:** `solver/routing/handlers/limit_order.py` (new)

```python
from solver.amm.limit_order import LimitOrderAMM
from solver.pools.limit_order import LimitOrderPool
from solver.routing.handlers.base import BaseHandler
from solver.routing.types import RoutingResult, HopResult
from solver.models.auction import Order


class LimitOrderHandler(BaseHandler):
    """Handler for routing through 0x limit orders."""

    def __init__(self, amm: LimitOrderAMM):
        self.amm = amm

    def route(
        self,
        order: Order,
        pool: LimitOrderPool,
        sell_amount: int,
        buy_amount: int,
    ) -> RoutingResult:
        """Route an order through a limit order."""
        if order.kind == "sell":
            result = self.amm.simulate_swap(
                pool, order.sell_token, order.buy_token, sell_amount
            )
            if result is None:
                return self._error_result(order, "Limit order swap failed")

            # Check limit price
            if result.amount_out < buy_amount:
                return self._error_result(order, "Output below limit price")

            return self._build_success_result(
                order, pool, result.amount_in, result.amount_out, result.gas_estimate
            )
        else:
            # Buy order
            result = self.amm.simulate_swap_exact_out(
                pool, order.sell_token, order.buy_token, buy_amount
            )
            if result is None:
                return self._error_result(order, "Limit order swap failed")

            # Check limit price
            if result.amount_in > sell_amount:
                return self._error_result(order, "Input exceeds limit price")

            return self._build_success_result(
                order, pool, result.amount_in, result.amount_out, result.gas_estimate
            )
```

### 6. Settlement Encoding

**File:** `solver/amm/limit_order.py` (add method)

The settlement interaction calls 0x's exchange contract. For the baseline solver comparison, we encode a simplified interaction:

```python
def encode_swap(
    self,
    pool: LimitOrderPool,
    token_in: str,
    amount_in: int,
) -> bytes:
    """Encode the swap interaction for settlement.

    The actual 0x fillOrKillLimitOrder call requires the full signed order,
    which the driver provides. The solver returns a placeholder that the
    driver fills with the complete order data.
    """
    # For baseline solver parity, we return a minimal encoded interaction
    # The driver layer handles the actual 0x order encoding
    from eth_abi import encode

    # Encode: (address takerToken, address makerToken, uint256 takerAmount)
    return encode(
        ["address", "address", "uint256"],
        [pool.taker_token, pool.maker_token, amount_in]
    )
```

Note: The full 0x `fillOrKillLimitOrder` encoding requires signature data that isn't provided in the auction DTO. The driver layer handles this. For solver benchmarking, we encode the essential swap parameters.

### 7. Types Update

**File:** `solver/pools/types.py` (modify)

Add `LimitOrderPool` to the `AnyPool` union:

```python
from solver.pools.limit_order import LimitOrderPool

AnyPool = (
    UniswapV2Pool
    | UniswapV3Pool
    | BalancerWeightedPool
    | BalancerStablePool
    | LimitOrderPool  # NEW
)
```

### 8. Router Integration

**File:** `solver/routing/router.py` (modify)

Add limit order support to the router:

```python
from solver.amm.limit_order import LimitOrderAMM
from solver.routing.handlers.limit_order import LimitOrderHandler

class SingleOrderRouter:
    def __init__(self, ...):
        # ... existing init ...
        self.limit_order_amm = LimitOrderAMM()
        self._limit_order_handler = LimitOrderHandler(self.limit_order_amm)
```

Register in the handler registry (if using Phase 3 registry pattern).

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `solver/pools/limit_order.py` | Create | `LimitOrderPool` dataclass |
| `solver/pools/types.py` | Modify | Add to `AnyPool` union |
| `solver/pools/parsing.py` | Modify | Add `parse_limit_order()` |
| `solver/pools/registry.py` | Modify | Add limit order storage |
| `solver/amm/limit_order.py` | Create | `LimitOrderAMM` with swap math |
| `solver/routing/handlers/limit_order.py` | Create | `LimitOrderHandler` |
| `solver/routing/router.py` | Modify | Wire up limit order handler |
| `solver/constants.py` | Modify | Add `GAS_PER_ZEROEX_ORDER = 66_358` |

## Test Plan

### Unit Tests

1. **Parsing tests** — Verify `LimitOrderPool` created correctly from JSON
2. **Swap math tests** — Verify `get_amount_out` and `get_amount_in` calculations
3. **Constraint tests** — Verify orders rejected when exceeding limits
4. **Handler tests** — Verify routing through limit orders

### Integration Tests

1. **Single limit order** — Route through one limit order
2. **Limit order vs AMM** — Verify best route selected
3. **Mixed liquidity** — Auction with V2, V3, Balancer, and limit orders

### Benchmark Fixtures

Create `tests/fixtures/auctions/benchmark/limit_order_*.json`:
- `limit_order_sell.json` — Sell order routed through limit order
- `limit_order_buy.json` — Buy order routed through limit order
- `limit_order_vs_v2.json` — Compare limit order vs V2 pricing

## Complexity Analysis

| Component | Complexity | Notes |
|-----------|------------|-------|
| Data model | Simple | Single dataclass |
| Swap math | Trivial | Two integer operations |
| Parsing | Simple | Direct field mapping |
| Handler | Simple | Follows existing pattern |
| Settlement | Medium | Basic encoding (driver handles full 0x encoding) |

**Estimated effort:** 1-2 sessions

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Missing test fixtures | Create from Rust solver test cases |
| Settlement encoding mismatch | Verify against Rust solver output |
| Fee handling edge cases | Match Rust's `taker_token_fee_amount` handling |

## Success Criteria

1. All 5 Rust liquidity types supported
2. Limit order benchmarks match Rust exactly
3. No regressions on existing tests (662 passing)
4. Full test coverage for new code

## References

- [0x Protocol Documentation](https://0x.org/docs)
- [0x LibMathV06 (swap math)](https://github.com/0xProject/protocol/blob/master/contracts/utils/contracts/src/v06/LibMathV06.sol)
- Rust implementation: `crates/solvers/src/boundary/liquidity/limit_order.rs`
- Gas estimate source: https://dune.com/queries/639669
