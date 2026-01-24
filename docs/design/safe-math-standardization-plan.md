# Safe Math Standardization Plan

**Created:** 2026-01-24
**Status:** Planning

> **CRITICAL:** This is a financial application. NO tolerance/epsilon comparisons allowed. All financial calculations MUST use exact integer arithmetic.

---

## Problem Statement

The codebase has inconsistent math handling:
- **SafeInt** exists (`solver/safe_int.py`) but is underused (only 5 files)
- **26 unsafe patterns** identified across the codebase
- **Floating-point tolerance** used in settlement.py (forbidden!)
- **Truncation direction** not explicit - causes bugs like the double_auction limit price issue

### Root Cause of Double Auction Bug

```python
# In core.py line 148:
match_b = int(Decimal(match_a) * clearing_price)  # TRUNCATES DOWN

# When clearing_price == ask_limit exactly:
# - Seller should receive AT LEAST their limit
# - But int() truncates DOWN, giving them LESS
# - Then actual_price < ask_price check fails
```

**Correct fix:** Use ceiling when computing amount for the ask (seller must receive at least their limit).

---

## Existing Safe Math Infrastructure

### 1. SafeInt (`solver/safe_int.py`)

| Method | Behavior | Use Case |
|--------|----------|----------|
| `S(x) + S(y)` | Addition | Token amounts |
| `S(x) - S(y)` | Subtraction, raises `Underflow` if negative | Balance deduction |
| `S(x) * S(y)` | Multiplication | Price × amount |
| `S(x) // S(y)` | Floor division, raises `DivisionByZero` | Price computation |
| `S(x).ceiling_div(y)` | Ceiling division | When receiver needs minimum |
| `S(x).saturating_sub(y)` | Clamps to 0 instead of raising | Optional safety |
| `S(x).to_uint256()` | Validates fits in uint256 | On-chain submission |

### 2. Bfp (`solver/math/fixed_point.py`)

| Method | Behavior | Use Case |
|--------|----------|----------|
| `Bfp.mul_down(x, y)` | Multiply, round down | Favor protocol |
| `Bfp.mul_up(x, y)` | Multiply, round up | Favor user |
| `Bfp.div_down(x, y)` | Divide, round down | Favor protocol |
| `Bfp.div_up(x, y)` | Divide, round up | Favor user |

---

## Standardization Rules

### Rule 1: All Token Amounts Use SafeInt

```python
# BAD
amount_b = int(amount_a * price)

# GOOD
amount_b = (S(amount_a) * S(price_num)) // S(price_denom)
```

### Rule 2: Explicit Rounding Direction

| Party | Rounding Rule | Method |
|-------|---------------|--------|
| Seller receives B | Round UP | `ceiling_div` or `mul_up` |
| Buyer pays B | Round DOWN | `//` or `mul_down` |
| Protocol fee | Round UP | `ceiling_div` |

```python
# Seller receives at least their limit (round UP)
amount_b_for_seller = (S(amount_a) * S(price_num)).ceiling_div(price_denom)

# Buyer pays at most their limit (round DOWN)
amount_b_from_buyer = (S(amount_a) * S(price_num)) // S(price_denom)
```

### Rule 3: No Floating Point in Financial Paths

```python
# BAD - float fill_ratio
sell_filled = int(order.sell_amount_int * fill_ratio)

# GOOD - integer ratio
sell_filled = (S(order.sell_amount_int) * S(fill_numerator)) // S(fill_denominator)
```

### Rule 4: No Tolerance Comparisons

```python
# BAD - tolerance
SETTLEMENT_TOLERANCE = 0.001
if actual_rate < limit_rate * (1 - SETTLEMENT_TOLERANCE):

# GOOD - exact integer comparison
# actual_rate = amount_out / amount_in
# limit_rate = buy_amount / sell_amount
# actual >= limit means: amount_out * sell_amount >= buy_amount * amount_in
if S(amount_out) * S(sell_amount) >= S(buy_amount) * S(amount_in):
    # Limit satisfied
```

### Rule 5: Price Ratios as Integer Pairs

```python
# BAD - Decimal price
clearing_price = Decimal("333333333.3333333")
match_b = int(match_a * clearing_price)

# GOOD - price as num/denom pair
price_num = 1_000_000_000_000_000_000  # 1e18
price_denom = 3_000_000_000  # 3e9
match_b = (S(match_a) * S(price_num)) // S(price_denom)
```

---

## Files Requiring Changes

### Priority 1: Double Auction (Immediate - Fixes Current Bug)

| File | Issue | Fix |
|------|-------|-----|
| `solver/strategies/double_auction/core.py:148` | `int(Decimal(match_a) * clearing_price)` truncates | Use ceiling for ask, floor for bid |
| `solver/strategies/double_auction/core.py:123` | Division truncates toward zero | Use explicit floor |
| `solver/strategies/double_auction/core.py:141` | Same | Same |
| `solver/strategies/double_auction/hybrid.py:220,235,242` | Same patterns | Same fixes |

**Approach:** Represent clearing_price as `(numerator, denominator)` integer pair, not Decimal.

### Priority 2: Settlement Module (Forbidden Tolerance)

| File | Issue | Fix |
|------|-------|-----|
| `solver/strategies/settlement.py:23` | `SETTLEMENT_TOLERANCE = 0.001` | Remove, use exact comparison |
| `solver/strategies/settlement.py:251` | Float tolerance comparison | Integer cross-multiply |
| `solver/strategies/settlement.py:370` | Same | Same |
| `solver/strategies/settlement.py:235,354` | Float arithmetic | Integer ratios |

### Priority 3: LP Solver Fill Ratios

| File | Issue | Fix |
|------|-------|-----|
| `solver/strategies/unified_cow.py:700-701` | Float fill_ratio | Integer num/denom |
| `solver/strategies/pricing.py:408-409,609-610` | Same | Same |

### Priority 4: Fee and Config Parsing (Lower Risk)

| File | Issue | Fix |
|------|-------|-----|
| `solver/amm/uniswap_v2.py:589` | `int(fee_decimal * 10000)` | Parse directly as int |
| `solver/amm/uniswap_v3/parsing.py:156` | Float fee conversion | Same |
| `solver/amm/balancer/amm.py:591,656` | Float amp parameter | Same |

---

## Implementation Plan

### Phase 1: Fix Double Auction (Immediate)

**Goal:** Fix the current bug with exact integer arithmetic.

1. Change `clearing_price` from `Decimal` to `tuple[int, int]` (num, denom)
2. Use SafeInt for all amount calculations
3. Use ceiling_div when computing seller's receive amount
4. Use floor division when computing buyer's pay amount

```python
# New signature
def _execute_matches_at_price(
    clearing_price: tuple[int, int],  # (numerator, denominator)
    asks: list[tuple[Order, tuple[int, int], int]],  # limit as (num, denom)
    bids: list[tuple[Order, tuple[int, int], int]],
    ...
) -> _MatchingAtPriceResult:
    price_num, price_denom = clearing_price

    # For seller (ask): receives at least limit
    match_b_seller = (S(match_a) * S(price_num)).ceiling_div(price_denom)

    # For buyer (bid): pays at most limit
    match_b_buyer = (S(match_a) * S(price_num)) // S(price_denom)

    # Use the conservative value (min for seller, max for buyer)
    # In a CoW match both get the same, so use floor (buyer's perspective)
    # but verify seller's limit is still met
    match_b = (S(match_a) * S(price_num)) // S(price_denom)

    # Verify seller limit: match_b / match_a >= ask_num / ask_denom
    # Cross multiply: match_b * ask_denom >= ask_num * match_a
    ask_num, ask_denom = ask_price
    if S(match_b) * S(ask_denom) < S(ask_num) * S(match_a):
        # Seller limit violated
        ...
```

### Phase 2: Remove Settlement Tolerance

1. Delete `SETTLEMENT_TOLERANCE` constant
2. Replace float comparisons with integer cross-multiplication
3. Change rate tracking from float to integer pairs

### Phase 3: Fix LP Fill Ratios

1. scipy.optimize returns float - convert immediately to rational approximation
2. Or: scale fill ratios to integer (e.g., fill_ratio * 10^18)
3. All fill calculations use SafeInt

### Phase 4: Audit Remaining Files

1. Use grep to find remaining `int(` patterns
2. Review each for correctness
3. Add SafeInt imports where needed

---

## New SafeInt Methods (if needed)

Consider adding to `solver/safe_int.py`:

```python
def mul_ceiling_div(self, multiplier: int, divisor: int) -> SafeInt:
    """Compute (self * multiplier) / divisor, rounding up.

    Useful for: amount_b = (amount_a * price_num).ceiling_div(price_denom)
    Combined operation avoids intermediate overflow.
    """
    if divisor <= 0:
        raise DivisionByZero(f"mul_ceiling_div by {divisor}")
    product = self._value * multiplier
    return SafeInt((product + divisor - 1) // divisor)

def mul_floor_div(self, multiplier: int, divisor: int) -> SafeInt:
    """Compute (self * multiplier) / divisor, rounding down."""
    if divisor == 0:
        raise DivisionByZero("mul_floor_div by zero")
    return SafeInt((self._value * multiplier) // divisor)
```

---

## Testing Strategy

### Unit Tests

1. Test ceiling vs floor with exact boundary values
2. Test that seller always receives >= limit
3. Test that buyer always pays <= limit
4. Test integer overflow handling

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(
    amount_a=st.integers(1, 10**24),
    price_num=st.integers(1, 10**18),
    price_denom=st.integers(1, 10**18),
)
def test_seller_receives_at_least_limit(amount_a, price_num, price_denom):
    """Seller receives amount_b where amount_b/amount_a >= price_num/price_denom."""
    amount_b = ceiling_mul_div(amount_a, price_num, price_denom)
    # Verify: amount_b * price_denom >= price_num * amount_a
    assert amount_b * price_denom >= price_num * amount_a
```

---

## Success Criteria

1. **Zero tolerance comparisons** in production code
2. **All financial calculations** use SafeInt or Bfp
3. **Explicit rounding direction** at every division
4. **All existing tests pass**
5. **New property-based tests** verify invariants
6. **Double auction bug fixed** without tolerance

---

## Appendix: Files to Audit

```
solver/strategies/double_auction/core.py       - CRITICAL
solver/strategies/double_auction/hybrid.py     - CRITICAL
solver/strategies/settlement.py                - CRITICAL (has tolerance!)
solver/strategies/unified_cow.py               - HIGH
solver/strategies/pricing.py                   - HIGH
solver/strategies/multi_pair.py                - MEDIUM
solver/amm/uniswap_v2.py                       - LOW
solver/amm/uniswap_v3/quoter.py               - LOW
solver/amm/uniswap_v3/parsing.py              - LOW
solver/amm/balancer/amm.py                    - LOW
solver/amm/balancer/parsing.py                - LOW
```
