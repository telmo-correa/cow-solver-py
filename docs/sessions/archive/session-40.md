# Session 40 - V2 Gas Estimate Parity and Price Estimation

**Date:** 2026-01-23

## Completed

- [x] Code review of price estimation feature for fee calculation
- [x] Identified root cause of gas/fee differences: V2 pools used hardcoded 60,000 instead of auction's `gasEstimate`
- [x] Added `gas_estimate` field to `UniswapV2Pool` dataclass
- [x] Updated `parse_liquidity_to_pool()` to read `gasEstimate` from auction data
- [x] Updated `simulate_swap()` to use `pool.gas_estimate` instead of constant
- [x] Updated router handler registry to use `pool.gas_estimate`
- [x] Cleaned up `price_estimation.py`: extracted `get_token_info()` to module-level function
- [x] Removed unused `SimpleReferenceEstimator` class
- [x] Updated `calculator.py` to use shared `get_token_info()`
- [x] Simplified `amm_routing.py` `_create_fee_calculator()`
- [x] Added comprehensive tests for gas estimate and price estimation

## Test Results

- All tests passing (702+)

## Key Changes

### Root Cause Fix: V2 Gas Estimates

Python was using a hardcoded `POOL_SWAP_GAS_COST = 60,000` for V2 pools, while Rust reads `gasEstimate` from auction data (typically 88,892-110,000). This caused fee calculation differences.

**Fix:** Added `gas_estimate` field to `UniswapV2Pool` that defaults to `POOL_SWAP_GAS_COST` but is populated from auction data during parsing.

### Code Quality Improvements

- Extracted `get_token_info()` to shared module-level function (was duplicated)
- Removed unused `estimation_amount` and `registry` parameters from `PoolBasedPriceEstimator`
- Removed stale `# type: ignore` comments that were no longer needed

## Files Created

```
solver/fees/price_estimation.py    # Native token price estimation module
tests/unit/fees/__init__.py        # Test package init
tests/unit/fees/test_price_estimation.py  # Price estimation tests (16 tests)
```

## Files Modified

```
solver/amm/uniswap_v2.py          # Added gas_estimate field, updated parsing
solver/fees/__init__.py           # Export new price estimation symbols
solver/fees/calculator.py         # Use shared get_token_info
solver/routing/router.py          # Use pool.gas_estimate in handler registry
solver/strategies/amm_routing.py  # Simplified _create_fee_calculator
solver/strategies/base.py         # Minor type annotation cleanup
tests/unit/amm/test_uniswap_v2.py # Added gas estimate tests (8 tests)
```

## Key Learnings

- Gas estimate parity is critical for fee calculation matching between Python and Rust
- Auction data provides pool-specific gas estimates that vary significantly (60k vs 90-110k)
- Hardcoded constants should be avoided when auction data provides the real values

## Next Session

- Run full benchmark suite against Rust solver
- Investigate any remaining discrepancies in benchmark results
- Verify gas estimate fix resolves fee differences
