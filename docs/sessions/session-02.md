# Session 2 - Single Order Routing (Slice 1.1 + 1.2)
**Date:** 2026-01-20

## Completed
- [x] **Slice 1.1:** Integration tests for /solve endpoint (5 tests)
- [x] **Slice 1.2:** UniswapV2 AMM math implementation
  - Constant product formula: `amount_out = (amount_in * 997 * reserve_out) / (reserve_in * 1000 + amount_in * 997)`
  - Reverse calculation for buy orders
  - Swap calldata encoding for UniswapV2 Router
- [x] **Slice 1.2:** Single order router
  - Routes sell orders through UniswapV2 pools
  - Builds complete Solution with trades and interactions
  - Validates limit prices
- [x] **Slice 1.2:** Integration tests for single order routing (6 tests)
- [x] Connected solver to API endpoint

## Test Results
- **39/39 passing**
  - 13 unit tests (models)
  - 15 unit tests (AMM math)
  - 11 integration tests (API + routing)
- Linting: clean (ruff)

## Benchmark Results
- N/A (Rust solver not yet configured for comparison)

## Files Created/Modified
```
solver/
├── amm/
│   ├── base.py              # NEW: AMM abstract base class
│   └── uniswap_v2.py        # NEW: UniswapV2 implementation
├── routing/
│   └── router.py            # NEW: Order routing and solution building
└── api/
    └── endpoints.py         # MODIFIED: Connected to solver

tests/
├── unit/
│   └── test_amm.py          # NEW: 15 AMM tests
└── integration/
    ├── test_api.py          # NEW: 5 API tests
    └── test_single_order.py # NEW: 6 routing tests
```

## Key Implementation Details
- UniswapV2 pools are hardcoded (WETH/USDC, WETH/USDT, WETH/DAI)
- Solver only handles single sell orders for now
- Solutions include proper clearing prices, trades, and swap interactions
- Limit price validation prevents unfillable orders

## Next Session
- **Slice 1.3:** Single buy order → UniswapV2 (inverse math)
- **Slice 1.4:** Multi-hop routing (A→B→C)
- Consider: Fetch real pool reserves from chain

## Open Questions
1. Should we fetch live pool reserves or continue with hardcoded?
2. Need to set up Rust solver for benchmarking
3. Buy order semantics may need adjustment
