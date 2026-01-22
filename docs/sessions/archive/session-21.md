# Session 21 - UniswapV3 Integration Tests
**Date:** 2026-01-22

## Completed
- [x] Slice 3.1.6: Integration Tests with Mock Quoter
  - Created V3 auction fixtures directory: `tests/fixtures/auctions/v3/`
  - Created 3 V3 fixture files:
    - `v3_single_order.json` - V3-only auction
    - `v2_v3_mixed.json` - Mixed V2 and V3 liquidity
    - `v3_buy_order.json` - V3 buy order auction
  - Added `v3_amm` parameter to `AmmRoutingStrategy`
  - Added `v3_amm` parameter to `Solver` class
  - Created 12 integration tests in `tests/integration/test_uniswap_v3.py`:
    - `TestV3SingleOrderRouting` (3 tests): V3-only routing, buy orders, limit price
    - `TestV2V3PoolSelection` (4 tests): Best-quote selection, fallback behavior
    - `TestV3ApiEndpoint` (3 tests): API responses, V3 parsing, clearing prices
    - `TestV3GasEstimates` (2 tests): Gas estimates from pool config

## Test Results
- Passing: 276/276 (264 existing + 12 new V3 integration tests)
- All ruff and mypy checks pass

## Key Implementation Details

### V3 AMM Support in AmmRoutingStrategy
```python
class AmmRoutingStrategy:
    def __init__(
        self,
        amm: UniswapV2 | None = None,
        router: SingleOrderRouter | None = None,
        v3_amm: UniswapV3AMM | None = None,  # NEW
    ) -> None:
        self.amm = amm if amm is not None else uniswap_v2
        self._injected_router = router
        self.v3_amm = v3_amm  # NEW

    def _get_router(self, pool_registry: PoolRegistry) -> SingleOrderRouter:
        if self._injected_router is not None:
            return self._injected_router
        return SingleOrderRouter(
            amm=self.amm, pool_registry=pool_registry, v3_amm=self.v3_amm  # NEW
        )
```

### V3 AMM Support in Solver
```python
class Solver:
    def __init__(
        self,
        strategies: list[SolutionStrategy] | None = None,
        router: SingleOrderRouter | None = None,
        amm: UniswapV2 | None = None,
        v3_amm: UniswapV3AMM | None = None,  # NEW
    ) -> None:
        if strategies is not None:
            self.strategies = strategies
        elif router is not None or amm is not None or v3_amm is not None:
            self.strategies = [
                CowMatchStrategy(),
                AmmRoutingStrategy(amm=amm, router=router, v3_amm=v3_amm),  # NEW
            ]
        else:
            self.strategies = [CowMatchStrategy(), AmmRoutingStrategy()]
```

### V3 Fixture Format
```json
{
  "id": "v3_single_order",
  "liquidity": [
    {
      "id": "v3-weth-usdc-3000",
      "kind": "concentratedLiquidity",
      "address": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
      "tokens": ["0xC02...WETH", "0xA0b...USDC"],
      "fee": "0.003",
      "sqrtPrice": "1887339785326389816925594",
      "liquidity": "12000000000000000000",
      "tick": 201390,
      "liquidityNet": {...},
      "gasEstimate": "200000"
    }
  ]
}
```

### Integration Test Patterns
```python
# Override solver with mock V3 AMM
from solver.amm.uniswap_v3 import MockUniswapV3Quoter, UniswapV3AMM

quoter = MockUniswapV3Quoter(default_rate=2500e6 / 1e18)  # 2500 USDC/WETH
v3_amm = UniswapV3AMM(quoter=quoter)
solver = Solver(v3_amm=v3_amm)
app.dependency_overrides[get_solver] = lambda: solver
```

## Files Modified
```
solver/strategies/amm_routing.py   # Added v3_amm parameter (~15 lines)
solver/solver.py                   # Added v3_amm parameter (~10 lines)
```

## Files Created
```
tests/fixtures/auctions/v3/v3_single_order.json    # V3-only fixture
tests/fixtures/auctions/v3/v2_v3_mixed.json        # Mixed V2+V3 fixture
tests/fixtures/auctions/v3/v3_buy_order.json       # V3 buy order fixture
tests/integration/test_uniswap_v3.py               # 12 integration tests (~475 lines)
```

## Code Review Issues Fixed
1. ruff ARG002: Removed unused `mock_v3_amm` arguments from 2 test methods
2. ruff F401: Removed unused `QuoteKey` import
3. Test fixes: Added `availableBalance` to inline auction fixtures (Token model requires it)
4. Test fixes: Updated gas assertion to include settlement overhead (pool gas + 106391)

## Integration Test Coverage

### TestV3SingleOrderRouting
| Test | Description | Scenario |
|------|-------------|----------|
| test_v3_only_sell_order | V3 routing works | V3-only liquidity, sell order |
| test_v3_buy_order | Buy orders work | V3 with exact output quote |
| test_v3_order_respects_limit_price | Limit price enforced | Quote below limit rejected |

### TestV2V3PoolSelection
| Test | Description | Scenario |
|------|-------------|----------|
| test_selects_v3_when_better_quote | V3 selected for better rate | V3 returns 2600 vs V2's ~2500 |
| test_selects_v2_when_better_quote | V2 selected for better rate | V3 returns 2000 vs V2's ~2500 |
| test_falls_back_to_v2_when_v3_quoter_fails | Fallback works | V3 quoter returns None |
| test_v3_only_with_no_v2_liquidity | V3-only works | No V2 pools available |

### TestV3ApiEndpoint
| Test | Description | Scenario |
|------|-------------|----------|
| test_api_returns_v3_solution | Valid response | Full response validation |
| test_api_handles_v3_parsing | V3 parsing works | Inline auction with V3 pool |
| test_api_clearing_prices_with_v3 | Prices valid | Both tokens have positive prices |

### TestV3GasEstimates
| Test | Description | Scenario |
|------|-------------|----------|
| test_v3_solution_has_gas_estimate | Gas > 100k | Default V3 gas estimate |
| test_v3_gas_from_pool_config | Custom gas used | 200k pool + 106k overhead = 306k |

## Gas Calculation
V3 solution gas = pool gas estimate + settlement overhead
- Pool gas: Configurable per pool (default 150k)
- Settlement overhead: 106,391 (7365 + 44000 + 2×27513)
- Example: 200k pool gas → 306,391 total gas

## Next Session
- Slice 3.2: Real V3 Integration
  - Connect to Ethereum RPC for real quotes
  - Test against mainnet fork
  - Performance benchmarks
