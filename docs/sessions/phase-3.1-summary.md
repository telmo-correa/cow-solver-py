# Phase 3 Slice 3.1: UniswapV3 Integration

**Sessions:** 16-23
**Dates:** 2026-01-22
**Status:** Complete

## Overview

Slice 3.1 added UniswapV3 concentrated liquidity support. Unlike V2's constant product formula, V3 uses tick-based pricing that's too complex for local calculation. We use the on-chain QuoterV2 contract for swap quotes.

## Key Accomplishments

### V3 Data Structures (Session 16)
- `UniswapV3Pool` dataclass with tick, liquidity, sqrtPrice, liquidityNet
- Parse `concentratedLiquidity` pools from auction data
- Fee tier constants (100, 500, 3000, 10000)

### Quoter Interface (Session 17)
- `UniswapV3Quoter` protocol for swap quoting
- `MockV3Quoter` for deterministic testing
- `Web3V3Quoter` for real on-chain quotes via QuoterV2

### Settlement Encoding (Session 18)
- SwapRouterV2 calldata encoding
- `exactInputSingle` and `exactOutputSingle` functions
- Proper deadline, recipient, and fee tier handling

### V3 AMM Integration (Session 19)
- `UniswapV3AMM` class implementing `SwapCalculator` protocol
- Token ordering (token0/token1) handling
- `zeroForOne` direction detection

### Router Integration (Session 20)
- V3 pools in `PoolRegistry` alongside V2
- Best-quote selection between V2 and V3
- Mixed V2/V3 routing support

### Testing (Sessions 21-22)
- 12 V3 integration tests
- V3 benchmark fixtures
- Real RPC quoter tests with `pytest.mark.rpc`

### Limit Order Fees (Session 23)
- Fee calculation matching Rust baseline: `fee = gas_cost * 1e18 / reference_price`
- Overflow protection (reject if fee > executed_amount)
- Fixed V3 fixture reference prices

## Final Metrics

| Metric | Value |
|--------|-------|
| Tests | 288 passing, 14 skipped |
| V3 Unit Tests | 53 tests |
| V3 Integration Tests | 12 tests |
| Fee Tests | 12 tests |

## Key Files Created/Modified

```
solver/amm/
├── uniswap_v2.py      # Added V3 pool storage to PoolRegistry
└── uniswap_v3.py      # V3 pool, quoter, AMM, settlement encoding

solver/strategies/
└── base.py            # Added fee calculation for limit orders

tests/
├── unit/test_uniswap_v3.py       # V3 unit tests
├── unit/test_strategy_base.py    # Fee calculation tests
└── integration/test_v3_integration.py

tests/fixtures/auctions/benchmark/
├── v3_weth_to_usdc.json
├── v3_usdc_to_weth.json
├── v3_buy_weth.json
└── v2_v3_comparison.json
```

## Architecture Decisions

### Quoter-Based Approach
V3 tick-crossing simulation is complex. Instead of implementing local math:
- Use QuoterV2 contract for swap quotes (read-only, no gas)
- `MockV3Quoter` for offline testing
- `Web3V3Quoter` for real RPC calls

### Best-Quote Selection
```python
def _find_best_direct_route(order, pools):
    for pool in pools:
        if isinstance(pool, UniswapV3Pool):
            quote = v3_amm.get_quote(pool, amount)
        else:
            quote = v2_amm.simulate_swap(pool, amount)
    return best_quote  # max output (sell) or min input (buy)
```

### Limit Order Fee Formula
```python
# Matches Rust baseline solver
fee = (gas * gas_price * 1e18) // reference_price
```
- Market orders: No solver fee (protocol handles)
- Limit orders: Solver MUST calculate fee
- Overflow: If fee > executed_amount, trade rejected

## Known Limitations

1. **RPC Required**: V3 quotes need `RPC_URL` environment variable
2. **No Local Math**: Can't calculate V3 swaps offline (except via mock)
3. **Single-hop Only**: V3 multi-hop not yet implemented

## Sessions Index

| Session | Focus | Key Outcome |
|---------|-------|-------------|
| 16 | V3 Data Structures | UniswapV3Pool, parsing |
| 17 | Quoter Interface | Mock + Web3 quoter |
| 18 | Settlement Encoding | SwapRouterV2 calldata |
| 19 | V3 AMM Class | SwapCalculator implementation |
| 20 | Router Integration | Best-quote selection |
| 21 | Integration Tests | 12 V3 tests, fixtures |
| 22 | Real Quoter | RPC tests, pytest marker |
| 23 | Limit Order Fees | Fee calculation, overflow handling |

See `archive/session-16.md` through `archive/session-23.md` for detailed session logs.
