# Balancer/Curve Integration Plan (Slice 3.2)

## Overview

Add support for Balancer weighted pools and Curve-style stable pools, matching the Rust baseline solver's functionality.

**Key insight from Rust analysis**: Unlike V3 which requires RPC, Balancer/Curve math can be computed locally using fixed-point arithmetic. This means we can achieve parity with Rust without external dependencies.

## Rust Implementation Summary

### Pool Types Supported

| Pool Type | Rust Support | Key Math | Gas Cost |
|-----------|--------------|----------|----------|
| Weighted Product (V0) | ✅ | Constant product with weights | 100,000 |
| Weighted Product (V3+) | ✅ | Optimized exponentiation for 50/50, 20/80 | 100,000 |
| Stable Pool | ✅ | StableSwap invariant (Newton-Raphson) | 183,520 |
| Composable Stable (V4) | ✅ | Same math, BPT token filtered | 183,520 |

### Auction Liquidity Format

**Weighted Pool:**
```json
{
  "kind": "weightedProduct",
  "tokens": {
    "0x...": {
      "balance": "11260752191375725565253",
      "scalingFactor": "1",
      "weight": "0.5"
    },
    "0x...": {
      "balance": "18764168403990393422000071",
      "scalingFactor": "1",
      "weight": "0.5"
    }
  },
  "fee": "0.005",
  "version": "v0",
  "balancerPoolId": "0x5c78d05b8ecf97507d1cf70646082c54faa4da950000000000000000000005ca",
  "address": "0x92762b42a06dcdddc5b7362cfb01e631c4d44b40",
  "gasEstimate": "88892"
}
```

**Stable Pool:**
```json
{
  "kind": "stable",
  "tokens": {
    "0x...DAI": {
      "balance": "505781036390938593206504",
      "scalingFactor": "1"
    },
    "0x...USDC": {
      "balance": "554894862074",
      "scalingFactor": "1000000000000"
    },
    "0x...USDT": {
      "balance": "1585576741011",
      "scalingFactor": "1000000000000"
    }
  },
  "amplificationParameter": "5000.0",
  "fee": "0.0001",
  "balancerPoolId": "0x5c78d05b8ecf97507d1cf70646082c54faa4da950000000000000000000005ca",
  "address": "0x06df3b2bbb68adc8b0e302443692037ed9f91b42",
  "gasEstimate": "183520"
}
```

### Key Differences from V2/V3

| Aspect | UniswapV2 | UniswapV3 | Balancer/Curve |
|--------|-----------|-----------|----------------|
| Math | Local (constant product) | RPC (QuoterV2) | Local (fixed-point) |
| Tokens | 2 | 2 | N (2+) |
| Settlement | Uniswap Router | SwapRouterV2 | Balancer Vault |
| Pool ID | Address | Address | 32-byte ID |

### Settlement Encoding

All Balancer swaps go through the **Balancer Vault** (same address on all chains):
- Address: `0xBA12222222228d8Ba445958a75a0704d566BF2C8`
- Function: `swap(SingleSwap, FundManagement, uint256 limit, uint256 deadline)`

```solidity
struct SingleSwap {
    bytes32 poolId;
    uint8 kind;        // 0 = GivenIn, 1 = GivenOut
    address assetIn;
    address assetOut;
    uint256 amount;
    bytes userData;    // Empty for standard swaps
}

struct FundManagement {
    address sender;              // Settlement contract
    bool fromInternalBalance;    // false
    address recipient;           // Settlement contract
    bool toInternalBalance;      // false
}
```

---

## Implementation Slices

### Slice 3.2.1: Fixed-Point Math Library

**Goal:** Implement 18-decimal fixed-point arithmetic matching Balancer's `Bfp`

**Files to create:**
- `solver/math/fixed_point.py` (new)

**Tasks:**

- [ ] Create `Bfp` (Balancer Fixed Point) class
  ```python
  class Bfp:
      """18-decimal fixed-point number stored as int."""
      ONE = 10**18

      def __init__(self, value: int):
          self.value = value

      @classmethod
      def from_wei(cls, wei: int) -> Bfp: ...

      def mul_down(self, other: Bfp) -> Bfp:
          """(a * b) // 10^18"""

      def mul_up(self, other: Bfp) -> Bfp:
          """ceil((a * b) / 10^18)"""

      def div_down(self, other: Bfp) -> Bfp:
          """(a * 10^18) // b"""

      def div_up(self, other: Bfp) -> Bfp:
          """ceil((a * 10^18) / b)"""

      def pow_up(self, exp: Bfp) -> Bfp:
          """Power with up rounding (for weighted pools)"""

      def complement(self) -> Bfp:
          """1 - self"""
  ```

- [ ] Implement power function for weighted pool exponentiation
  - Natural logarithm approximation
  - Exponential approximation
  - Handle edge cases (exp near 0, 1, 2)

- [ ] Add constants
  ```python
  MAX_IN_RATIO = Bfp.from_wei(3 * 10**17)   # 0.3
  MAX_OUT_RATIO = Bfp.from_wei(3 * 10**17)  # 0.3
  AMP_PRECISION = 1000
  ```

**Tests:**
- `tests/unit/test_fixed_point.py::test_mul_down_rounds_down`
- `tests/unit/test_fixed_point.py::test_mul_up_rounds_up`
- `tests/unit/test_fixed_point.py::test_div_precision`
- `tests/unit/test_fixed_point.py::test_pow_accuracy`

**No external dependencies:** ✅

---

### Slice 3.2.2: Weighted Pool Math

**Goal:** Implement Balancer weighted pool swap calculations

**Files to create:**
- `solver/amm/balancer_weighted.py` (new)

**Tasks:**

- [ ] Create `BalancerWeightedPool` dataclass
  ```python
  @dataclass
  class WeightedReserve:
      token: str
      balance: int
      weight: Decimal        # Normalized weight (sum to 1.0)
      scaling_factor: int    # 10^(18 - decimals)

  @dataclass
  class BalancerWeightedPool:
      address: str
      pool_id: str           # 32-byte hex
      tokens: list[WeightedReserve]  # Sorted by address
      fee: Decimal
      version: Literal["v0", "v3Plus"]
  ```

- [ ] Implement `calc_out_given_in()`
  ```python
  def calc_out_given_in(
      balance_in: Bfp,
      weight_in: Bfp,
      balance_out: Bfp,
      weight_out: Bfp,
      amount_in: Bfp,
  ) -> Bfp:
      """
      Formula:
        base = balance_in / (balance_in + amount_in)
        exponent = weight_in / weight_out
        power = base ^ exponent
        amount_out = balance_out * (1 - power)
      """
  ```

- [ ] Implement `calc_in_given_out()`
  ```python
  def calc_in_given_out(
      balance_in: Bfp,
      weight_in: Bfp,
      balance_out: Bfp,
      weight_out: Bfp,
      amount_out: Bfp,
  ) -> Bfp:
      """
      Formula:
        base = balance_out / (balance_out - amount_out)
        exponent = weight_out / weight_in
        power = base ^ exponent
        amount_in = balance_in * (power - 1)
      """
  ```

- [ ] Enforce ratio limits
  - `MAX_IN_RATIO = 0.3` (reject if input > 30% of balance)
  - `MAX_OUT_RATIO = 0.3` (reject if output > 30% of balance)

- [ ] Handle V0 vs V3Plus versions
  - V3Plus: Optimized for 50/50 and 20/80 weight ratios

**Tests:**
- `tests/unit/test_balancer_weighted.py::test_calc_out_given_in`
- `tests/unit/test_balancer_weighted.py::test_calc_in_given_out`
- `tests/unit/test_balancer_weighted.py::test_max_in_ratio_enforced`
- `tests/unit/test_balancer_weighted.py::test_fee_application`
- `tests/unit/test_balancer_weighted.py::test_matches_rust_output` (use Rust test vectors)

**Test vectors from Rust:**
```
Input: GNO→COW, 1 ETH, 50/50 pool, 0.5% fee
Expected output: 1657855325872947866705 COW
```

**No external dependencies:** ✅

---

### Slice 3.2.3: Stable Pool Math

**Goal:** Implement Curve StableSwap invariant calculations

**Files to create:**
- `solver/amm/balancer_stable.py` (new)

**Tasks:**

- [ ] Create `BalancerStablePool` dataclass
  ```python
  @dataclass
  class StableReserve:
      token: str
      balance: int
      scaling_factor: int    # 10^(18 - decimals)

  @dataclass
  class BalancerStablePool:
      address: str
      pool_id: str
      tokens: list[StableReserve]  # Sorted by address
      amplification_parameter: Decimal  # A parameter
      fee: Decimal
  ```

- [ ] Implement `calculate_invariant()` (Newton-Raphson)
  ```python
  def calculate_invariant(amp: int, balances: list[Bfp]) -> Bfp:
      """
      StableSwap invariant:
        A * n^n * sum(x_i) + D = A * D * n^n + D^(n+1) / (n^n * prod(x_i))

      Iteratively solve for D using Newton-Raphson.
      Max iterations: 255
      Convergence: |D_new - D_old| <= 1 wei
      """
  ```

- [ ] Implement `get_token_balance_given_invariant()`
  ```python
  def get_token_balance_given_invariant_and_all_other_balances(
      amp: int,
      balances: list[Bfp],
      invariant: Bfp,
      token_index: int,
  ) -> Bfp:
      """
      Given D and all other balances, solve for balance[token_index].
      Uses Newton-Raphson iteration.
      """
  ```

- [ ] Implement `calc_out_given_in()` and `calc_in_given_out()`
  - Apply fee before/after calculation
  - Subtract 1 wei from output for rounding protection

- [ ] Handle composable stable pools (V4)
  - Filter out BPT token from reserves
  - Same math otherwise

**Tests:**
- `tests/unit/test_balancer_stable.py::test_invariant_calculation`
- `tests/unit/test_balancer_stable.py::test_calc_out_given_in`
- `tests/unit/test_balancer_stable.py::test_calc_in_given_out`
- `tests/unit/test_balancer_stable.py::test_convergence_failure`
- `tests/unit/test_balancer_stable.py::test_matches_rust_output` (use Rust test vectors)

**Test vectors from Rust:**
```
Input: DAI→USDC, 10 DAI, 3-token stable pool, A=5000, 0.01% fee
Expected output: 9999475 USDC (6 decimals)
```

**No external dependencies:** ✅

---

### Slice 3.2.4: Pool Parsing and Registry

**Goal:** Parse Balancer/Curve pools from auction data

**Files to modify:**
- `solver/amm/uniswap_v2.py` (update PoolRegistry)

**Tasks:**

- [ ] Add `parse_weighted_liquidity()` function
  ```python
  def parse_weighted_liquidity(liquidity: Liquidity) -> BalancerWeightedPool | None:
      if liquidity.kind != "weightedProduct":
          return None
      # Parse tokens with weights and scaling factors
      # Sort by address
      # Return BalancerWeightedPool
  ```

- [ ] Add `parse_stable_liquidity()` function
  ```python
  def parse_stable_liquidity(liquidity: Liquidity) -> BalancerStablePool | None:
      if liquidity.kind != "stable":
          return None
      # Parse tokens with scaling factors
      # Parse amplification parameter
      # Sort by address
      # Handle composable pools (filter BPT)
      # Return BalancerStablePool
  ```

- [ ] Update `PoolRegistry` to store all pool types
  ```python
  class PoolRegistry:
      v2_pools: dict[tuple[str, str], UniswapV2Pool]
      v3_pools: dict[tuple[str, str, int], UniswapV3Pool]
      weighted_pools: dict[tuple[str, str], list[BalancerWeightedPool]]  # Multiple pools per pair
      stable_pools: dict[tuple[str, ...], list[BalancerStablePool]]      # N-token pools
  ```

- [ ] Update `get_pools_for_pair()` to return all pool types

**Tests:**
- `tests/unit/test_pool_registry.py::test_parse_weighted_pool`
- `tests/unit/test_pool_registry.py::test_parse_stable_pool`
- `tests/unit/test_pool_registry.py::test_parse_composable_stable`
- `tests/unit/test_pool_registry.py::test_get_pools_returns_all_types`

**No external dependencies:** ✅

---

### Slice 3.2.5: Balancer AMM Integration

**Goal:** Create AMM classes that implement the swap interface

**Files to create:**
- `solver/amm/balancer.py` (new, combines weighted + stable)

**Tasks:**

- [ ] Create `BalancerWeightedAMM` class
  ```python
  class BalancerWeightedAMM:
      def get_amount_out(
          self, pool: BalancerWeightedPool, amount_in: int, token_in: str
      ) -> SwapResult | None:
          # Scale amounts, apply fee, calculate, scale back

      def get_amount_in(
          self, pool: BalancerWeightedPool, amount_out: int, token_out: str
      ) -> SwapResult | None:
          # Scale amounts, calculate, apply fee, scale back
  ```

- [ ] Create `BalancerStableAMM` class
  ```python
  class BalancerStableAMM:
      def get_amount_out(
          self, pool: BalancerStablePool, amount_in: int, token_in: str
      ) -> SwapResult | None:

      def get_amount_in(
          self, pool: BalancerStablePool, amount_out: int, token_out: str
      ) -> SwapResult | None:
  ```

- [ ] Add gas cost constants
  ```python
  WEIGHTED_SWAP_GAS = 100_000
  STABLE_SWAP_GAS = 183_520
  ```

**Tests:**
- `tests/unit/test_balancer_amm.py::test_weighted_get_amount_out`
- `tests/unit/test_balancer_amm.py::test_weighted_get_amount_in`
- `tests/unit/test_balancer_amm.py::test_stable_get_amount_out`
- `tests/unit/test_balancer_amm.py::test_stable_get_amount_in`
- `tests/unit/test_balancer_amm.py::test_decimal_scaling`

**No external dependencies:** ✅

---

### Slice 3.2.6: Settlement Encoding

**Goal:** Encode Balancer Vault swap calldata

**Files to create:**
- `solver/amm/balancer.py` (add encoding functions)

**Tasks:**

- [ ] Add Balancer Vault constant
  ```python
  BALANCER_VAULT = "0xBA12222222228d8Ba445958a75a0704d566BF2C8"
  ```

- [ ] Implement `encode_swap()` for weighted pools
  ```python
  def encode_weighted_swap(
      pool: BalancerWeightedPool,
      token_in: str,
      token_out: str,
      amount_in: int,
      amount_out_min: int,
      settlement: str,
  ) -> bytes:
      """Encode Vault.swap() calldata for weighted pool."""
  ```

- [ ] Implement `encode_swap()` for stable pools
  ```python
  def encode_stable_swap(
      pool: BalancerStablePool,
      token_in: str,
      token_out: str,
      amount_in: int,
      amount_out_min: int,
      settlement: str,
  ) -> bytes:
      """Encode Vault.swap() calldata for stable pool."""
  ```

- [ ] Add Vault ABI (minimal, just `swap` function)
  ```python
  VAULT_SWAP_SELECTOR = "0x52bbbe29"  # swap(SingleSwap,FundManagement,uint256,uint256)
  ```

**Tests:**
- `tests/unit/test_balancer_encoding.py::test_encode_weighted_swap`
- `tests/unit/test_balancer_encoding.py::test_encode_stable_swap`
- `tests/unit/test_balancer_encoding.py::test_calldata_format`

**No external dependencies:** ✅

---

### Slice 3.2.7: Router Integration

**Goal:** Extend router to use Balancer/Curve pools

**Files to modify:**
- `solver/routing/router.py`

**Tasks:**

- [ ] Update `SingleOrderRouter` to accept Balancer AMMs
  ```python
  class SingleOrderRouter:
      def __init__(
          self,
          amm: UniswapV2AMM | None = None,
          v3_amm: UniswapV3AMM | None = None,
          weighted_amm: BalancerWeightedAMM | None = None,
          stable_amm: BalancerStableAMM | None = None,
          pool_finder: PoolFinder | None = None,
      ):
          ...
  ```

- [ ] Update `_find_best_direct_route()` to consider all pool types
  ```python
  def _find_best_direct_route(self, order, pools) -> RoutingResult | None:
      candidates = []
      for pool in pools:
          if isinstance(pool, UniswapV2Pool):
              result = self.amm.simulate_swap(...)
          elif isinstance(pool, UniswapV3Pool):
              result = self.v3_amm.get_amount_out(...)
          elif isinstance(pool, BalancerWeightedPool):
              result = self.weighted_amm.get_amount_out(...)
          elif isinstance(pool, BalancerStablePool):
              result = self.stable_amm.get_amount_out(...)
          if result:
              candidates.append((pool, result))

      # Select best by output amount (sell) or input amount (buy)
      return max/min(candidates, key=lambda x: x[1].amount_out/in)
  ```

- [ ] Build correct interaction based on pool type

**Tests:**
- `tests/unit/test_router.py::test_router_selects_best_across_all_types`
- `tests/unit/test_router.py::test_router_weighted_better_than_v2`
- `tests/unit/test_router.py::test_router_stable_better_than_v2`

**No external dependencies:** ✅

---

### Slice 3.2.8: Integration Tests

**Goal:** Full solve flow with Balancer/Curve pools

**Files to create:**
- `tests/integration/test_balancer_integration.py`
- `tests/fixtures/auctions/benchmark/weighted_*.json`
- `tests/fixtures/auctions/benchmark/stable_*.json`

**Tasks:**

- [ ] Create weighted pool fixtures (from Rust test cases)
  - `weighted_gno_to_cow.json` - 50/50 pool
  - `weighted_v3plus.json` - V3+ optimized pool

- [ ] Create stable pool fixtures (from Rust test cases)
  - `stable_dai_to_usdc.json` - 3-token stable pool
  - `stable_buy_order.json` - Buy order through stable pool
  - `stable_composable.json` - Composable stable pool

- [ ] Integration test: Weighted pool solve
  ```python
  def test_solve_with_weighted_pool():
      auction = load_fixture("weighted_gno_to_cow.json")
      result = solver.solve(auction)

      assert len(result.solutions) == 1
      # Verify output matches Rust: 1657855325872947866705
  ```

- [ ] Integration test: Stable pool solve
  ```python
  def test_solve_with_stable_pool():
      auction = load_fixture("stable_dai_to_usdc.json")
      result = solver.solve(auction)

      # Verify output matches Rust: 9999475
  ```

- [ ] Integration test: Best pool selection
  ```python
  def test_solver_selects_best_pool():
      # Auction with V2 + weighted + stable pools
      # Verify solver picks the one with best output
  ```

**No external dependencies:** ✅

---

### Slice 3.2.9: Benchmarking

**Goal:** Verify Python matches Rust on Balancer/Curve auctions

**Tasks:**

- [ ] Run Python vs Rust benchmarks on weighted fixtures
- [ ] Run Python vs Rust benchmarks on stable fixtures
- [ ] Document results in BENCHMARKS.md
- [ ] Update feature table

**Expected Results:**
- Weighted pools: Exact match with Rust (same math)
- Stable pools: Exact match with Rust (same iterative algorithm)
- Performance: Python ~1.5-2x slower (no RPC, just computation)

---

## Test Strategy Summary

| Slice | Test Type | External Deps |
|-------|-----------|---------------|
| 3.2.1 Fixed-Point | Unit | None |
| 3.2.2 Weighted Math | Unit | None |
| 3.2.3 Stable Math | Unit | None |
| 3.2.4 Pool Parsing | Unit | None |
| 3.2.5 AMM Integration | Unit | None |
| 3.2.6 Encoding | Unit | None |
| 3.2.7 Router | Unit | None |
| 3.2.8 Integration | Integration | None |
| 3.2.9 Benchmark | Benchmark | Rust solver |

**Key advantage:** All math is local (no RPC needed), so tests are fast and deterministic.

---

## Dependencies

### Python Dependencies
- None new required
- Use existing `eth-abi` for encoding

### Test Data
- Copy test vectors from Rust solver tests (`bal_liquidity.rs`)
- Ensures exact compatibility

---

## File Structure After Implementation

```
solver/
├── amm/
│   ├── __init__.py
│   ├── base.py                  # SwapResult, protocols
│   ├── uniswap_v2.py            # V2 + PoolRegistry
│   ├── uniswap_v3.py            # V3 quoter-based
│   ├── balancer_weighted.py     # NEW: Weighted pool math + AMM
│   ├── balancer_stable.py       # NEW: Stable pool math + AMM
│   └── balancer.py              # NEW: Settlement encoding
├── math/
│   └── fixed_point.py           # NEW: Bfp 18-decimal math
├── routing/
│   └── router.py                # Updated for all pool types
└── constants.py                 # Add Balancer Vault address

tests/
├── unit/
│   ├── test_fixed_point.py      # NEW
│   ├── test_balancer_weighted.py # NEW
│   ├── test_balancer_stable.py  # NEW
│   ├── test_balancer_amm.py     # NEW
│   └── test_balancer_encoding.py # NEW
├── integration/
│   └── test_balancer_integration.py # NEW
└── fixtures/auctions/benchmark/
    ├── weighted_gno_to_cow.json # NEW
    ├── weighted_v3plus.json     # NEW
    ├── stable_dai_to_usdc.json  # NEW
    └── stable_composable.json   # NEW
```

---

## Success Criteria

1. [ ] Fixed-point math matches Balancer's Bfp implementation
2. [ ] Weighted pool output matches Rust test vectors exactly
3. [ ] Stable pool output matches Rust test vectors exactly
4. [ ] Router selects best pool across V2, V3, weighted, and stable
5. [ ] All existing tests still pass
6. [ ] Benchmark shows exact match with Rust on Balancer/Curve auctions
7. [ ] mypy and ruff clean

---

## Estimated Effort

| Slice | Effort |
|-------|--------|
| 3.2.1 Fixed-Point Math | Medium |
| 3.2.2 Weighted Math | Medium |
| 3.2.3 Stable Math | Medium-Large (iterative algorithm) |
| 3.2.4 Pool Parsing | Small |
| 3.2.5 AMM Integration | Small |
| 3.2.6 Encoding | Small |
| 3.2.7 Router Integration | Small |
| 3.2.8 Integration Tests | Medium |
| 3.2.9 Benchmark | Small |

**Total:** ~2-3 sessions of focused work

---

## Key Implementation Notes

### Fixed-Point Precision
- All math uses 18-decimal fixed-point (`Bfp`)
- Rounding direction matters: use `mul_down`/`div_up` for conservative quotes
- Power function needs Taylor series approximation for non-integer exponents

### Stable Pool Convergence
- Newton-Raphson with max 255 iterations
- Convergence when `|D_new - D_old| <= 1`
- Must handle non-convergence gracefully (return None)

### Decimal Scaling
- Tokens have different decimals (DAI=18, USDC=6)
- `scalingFactor` in auction data normalizes to 18 decimals
- Scale up before math, scale down after

### Token Ordering
- Balancer sorts tokens by address (ascending)
- Pool ID encodes token order
- Must maintain consistent ordering
