# Session 47: Token Decimal Handling Fix

**Date:** 2026-01-23
**Focus:** Fix `get_reference_price` probe amount for non-18-decimal tokens

## Summary

Fixed a bug where `get_reference_price` was hardcoding the probe amount to `10^15`, which only worked correctly for 18-decimal tokens. For tokens like USDC (6 decimals), this meant probing with 1 billion USDC instead of 0.001 USDC, causing invalid AMM reference prices.

## Changes Made

### 1. `solver/routing/router.py`
- Added `token_in_decimals` parameter (default: 18 for backward compatibility)
- Probe amount now scales: `10 ** max(0, token_in_decimals - 3)` = 0.001 tokens
- Fixed docstring to correctly document raw price semantics

### 2. `solver/strategies/hybrid_cow.py`
- Added token decimal lookup using existing `get_token_info` helper
- Added warning log when token decimals not found (defaults to 18)

### 3. `tests/unit/routing/test_price_query.py`
- Added `TestGetReferencePriceNon18Decimals` class with 6 tests:
  - 6→18 decimal (USDC→WETH)
  - 18→6 decimal (WETH→USDC)
  - 6→6 decimal (USDC→USDT)
  - 8→18 decimal (WBTC→WETH)
  - Reverse direction mixed decimals
  - Small liquidity 6-decimal pools

### 4. Documentation
- Updated `docs/benchmark-slice4.2-plan.md` with resolution details
- Updated `docs/sessions/README.md` with new test count (848)

## Key Technical Details

### Raw Price Semantics
The returned price is in **raw units** (raw_token_out / raw_token_in), consistent with order limit prices:
- Pool: 2500 USDC per WETH
- WETH→USDC: `2500 * 10^(6-18) = 2.5e-9` raw price
- USDC→WETH: `0.0004 * 10^(18-6) = 4e8` raw price

### Probe Amount Scaling
| Token Decimals | Probe Amount | Human Value |
|----------------|--------------|-------------|
| 18 (WETH)      | 10^15        | 0.001 WETH  |
| 8 (WBTC)       | 10^5         | 0.001 WBTC  |
| 6 (USDC)       | 10^3         | 0.001 USDC  |

## Test Results

```
848 tests passing
HybridCow wins: 50.0% (9/18 CoW-eligible auctions)
Exit criteria: PASS (50% >= 20%)
```

## What's Next

Phase 4 continued - multi-order optimization
