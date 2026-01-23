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

## Parity Test Results

Created `tests/integration/test_rust_parity.py` which runs 17 Rust baseline test fixtures.

**Results: 5 passed, 12 failed**

### Passing (5)
- bal_liquidity_composable_stable_v4
- direct_swap
- internalization_insufficient_balance
- internalization_trusted_token
- internalization_untrusted_sell_token

### Bugs Identified

1. **Multi-solution bug**: Python combines independent orders into single solution.
   - Rust returns separate solutions for each order, letting the driver choose.
   - Python incorrectly batches all orders into one solution.
   - Affects: `bal_liquidity_stable` (2 trades in 1 solution vs 2 solutions with 1 trade each)

2. **Router quote comparison bug**: Python uses direct path without comparing multi-hop quotes.
   - In `buy_order_rounding_balancer_weighted`, Rust finds better 2-hop path while Python uses suboptimal direct path.
   - Causes 6476037% price difference.

3. **Partial fill fee bug**: Python ignores limit order fee when calculating max partial fill.
   - Rust: `output / (input + fee) >= limit_price`
   - Python: `output / input >= limit_price`
   - Causes 30% difference in `partial_fill` test.

4. **Gas overhead discrepancy**: Python missing ~11,000 per-trade gas overhead.
   - Rust: 206,391 = 88,892 (pool) + 106,391 (overhead) + 11,108 (trade)
   - Python: 195,283 = 88,892 (pool) + 106,391 (overhead)

### Small Rounding Differences (Acceptable)
- `buy_order_rounding_uniswap`: 0.000000% output diff (428 wei)
- `buy_order_rounding_same_path`: 3 wei difference
- `buy_order_rounding_balancer_stable`: 1 wei difference

## Next Session

- Fix the 4 identified bugs for true Rust parity
- Most critical: multi-solution bug and partial fill fee bug
