# Rust Parity Validation Design

## Problem Statement

The Python solver can legitimately produce **better** results than the Rust baseline solver in some cases. For example:

- **Partial fills**: Python calculates optimal fill (~65%) vs Rust's binary halving (50%)
- Both satisfy the limit price constraint, but Python fills more of the order

The current test approach requires exact field matching, which fails when Python is better. We need a validation approach that:

1. **Catches real bugs** (wrong fees, invalid amounts, limit price violations)
2. **Allows legitimate improvements** (better fills, same correctness)
3. **Documents when Python beats Rust**

## Design: Two-Phase Validation

### Phase 1: Correctness Invariants

These checks validate that the solution is **correct according to API constraints**, independent of what Rust produces. All must pass.

#### 1.1 Fee Calculation
```python
expected_fee = gas_estimate * gas_price * 1e18 // reference_price
assert trade.fee == expected_fee
```

**Catches**: Wrong fee formula, missing fee calculation

#### 1.2 Amount Consistency
```python
# For sell orders:
assert interaction.inputAmount == trade.executedAmount + trade.fee

# For buy orders:
assert interaction.outputAmount == trade.executedAmount
# (fee is deducted from sell side)
```

**Catches**: Forgot to add fee to input, inconsistent amounts

#### 1.3 AMM Simulation Verification
```python
# Simulate the swap with the claimed input
simulated_output = amm.get_amount_out(
    interaction.inputAmount,
    reserve_in,
    reserve_out,
    fee_multiplier
)
assert interaction.outputAmount == simulated_output
```

**Catches**: Impossible/fabricated amounts, wrong AMM math

#### 1.4 Limit Price Satisfaction
```python
# For sell orders: effective price >= limit price
# output/executed >= order.buyAmount/order.sellAmount
assert (interaction.outputAmount * order.sellAmount >=
        order.buyAmount * trade.executedAmount)

# For buy orders:
# executed/input >= order.buyAmount/order.sellAmount
assert (trade.executedAmount * order.sellAmount >=
        order.buyAmount * interaction.inputAmount)
```

**Catches**: Trades that violate user's limit price

#### 1.5 Clearing Price Consistency
```python
# Prices must reflect actual trade ratios
# price[buy] * input ≈ price[sell] * output
assert (prices[buyToken] * interaction.inputAmount ==
        prices[sellToken] * interaction.outputAmount)
```

**Catches**: Inconsistent clearing prices

### Phase 2: Rust Comparison

Only runs if Phase 1 passes. Compares Python's result to Rust baseline.

```python
python_fill = trade.executedAmount
rust_fill = rust_expected.trades[0].executedAmount

if python_fill > rust_fill:
    # Python is better - PASS with note
    log(f"Python fills more: {python_fill} vs {rust_fill}")
    return PASS

elif python_fill == rust_fill:
    # Same fill - validate exact field match
    validate_exact_match(python_solution, rust_expected)
    return PASS

else:
    # Python is worse - FAIL
    return FAIL(f"Python fills less: {python_fill} < {rust_fill}")
```

## Why This Catches Common Bugs

### Bug: "Forgot to account for fees"

**Scenario A**: Fee calculated as 0
- Fails 1.1 (fee calculation wrong)

**Scenario B**: Fee correct but not added to inputAmount
- Fails 1.2 (inputAmount != executedAmount + fee)

**Scenario C**: Claims better fill with wrong amounts
- Fails 1.3 (AMM simulation won't produce claimed output)

### Bug: "Limit price violation"

- Fails 1.4 (limit price check)

### Bug: "Wrong AMM math"

- Fails 1.3 (simulated output won't match claimed output)

### Bug: "Regression in fill amount"

- Fails Phase 2 (python_fill < rust_fill)

## Implementation

### File: `tests/integration/test_rust_parity.py`

```python
class SolutionValidator:
    """Validates solution correctness and compares to Rust baseline."""

    def __init__(self, auction: AuctionInstance):
        self.auction = auction
        self.amm = UniswapV2()
        # ... initialize pool registry from auction liquidity

    def validate(
        self,
        python_solution: Solution,
        rust_solution: dict,
        order: Order,
    ) -> ValidationResult:
        """Two-phase validation."""

        # Phase 1: Correctness invariants
        errors = []
        errors.extend(self._validate_fee(python_solution, order))
        errors.extend(self._validate_amounts(python_solution, order))
        errors.extend(self._validate_amm_simulation(python_solution))
        errors.extend(self._validate_limit_price(python_solution, order))
        errors.extend(self._validate_clearing_prices(python_solution))

        if errors:
            return ValidationResult(passed=False, errors=errors)

        # Phase 2: Rust comparison
        return self._compare_to_rust(python_solution, rust_solution, order)

    def _compare_to_rust(
        self,
        python_solution: Solution,
        rust_solution: dict,
        order: Order,
    ) -> ValidationResult:
        python_fill = self._get_executed_amount(python_solution)
        rust_fill = self._get_executed_amount_from_rust(rust_solution)

        if python_fill > rust_fill:
            return ValidationResult(
                passed=True,
                note=f"Python better: fills {python_fill} vs Rust {rust_fill}"
            )
        elif python_fill == rust_fill:
            # Exact match expected for all fields
            return self._validate_exact_match(python_solution, rust_solution)
        else:
            return ValidationResult(
                passed=False,
                errors=[f"Python worse: fills {python_fill} < Rust {rust_fill}"]
            )
```

### Handling Multiple Solutions

When Python returns multiple solutions (one per order, matching Rust), validate each:

```python
def validate_solutions(python_solutions, rust_solutions, orders):
    for py_sol, rust_sol, order in zip(python_solutions, rust_solutions, orders):
        result = validator.validate(py_sol, rust_sol, order)
        if not result.passed:
            return result
    return ValidationResult(passed=True)
```

### Handling Multi-Hop Routes

For multi-hop routes, validate each interaction in sequence:

```python
def _validate_amm_simulation(self, solution):
    """Validate each hop in a multi-hop route."""
    for interaction in solution.interactions:
        pool = self._get_pool(interaction.id)
        simulated = self._simulate_swap(pool, interaction)
        if simulated != interaction.outputAmount:
            return [f"Hop {i}: simulated {simulated} != claimed {interaction.outputAmount}"]
    return []
```

## Test Output

When Python beats Rust:
```
PASSED test_fixture_parity[partial_fill]
  Note: Python fills 65% (648245015255514214) vs Rust 50% (500000000000000000)
```

When Python matches Rust:
```
PASSED test_fixture_parity[direct_swap]
```

When Python has a bug:
```
FAILED test_fixture_parity[bad_fee]
  Error: Fee mismatch: calculated 2495865000000000, expected 2495865000000000
  Error: inputAmount 500000000000000000 != executedAmount 500000000000000000 + fee 2495865000000000
```

## Migration Plan

1. Create `SolutionValidator` class with Phase 1 checks
2. Update `test_fixture_parity` to use validator
3. Run tests - `partial_fill` should now pass with "better" note
4. All other tests should pass unchanged (exact match path)

## Future Considerations

- **CoW matches**: Need separate validation (no AMM interactions)
- **Multi-order solutions**: Validate uniform prices across orders
- **Gas estimates**: Allow some tolerance (estimates vary)
