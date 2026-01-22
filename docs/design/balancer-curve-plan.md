# Balancer/Curve Integration Plan (Slice 3.2)

## Overview

Add support for Balancer weighted pools and Curve-style stable pools, matching the Rust baseline solver's functionality.

**Key insight from Rust analysis**: Unlike V3 which requires RPC, Balancer/Curve math can be computed locally using fixed-point arithmetic. This means we can achieve parity with Rust without external dependencies.

## Rust Implementation Summary

### Pool Types Supported

| Pool Type | Rust Support | Key Math | Gas Cost |
|-----------|--------------|----------|----------|
| Weighted Product (V0) | ✅ | Constant product with weights | ~88,892 (from auction) |
| Weighted Product (V3+) | ✅ | Optimized power rounding for 50/50, 20/80 | ~88,892 (from auction) |
| Stable Pool | ✅ | StableSwap invariant (Newton-Raphson) | ~183,520 (from auction) |
| Composable Stable (V4) | ✅ | Same math, BPT token filtered | ~183,520 (from auction) |

**Note:** Gas estimates come from the auction data (`gasEstimate` field), not hardcoded constants.

### Auction Liquidity Format

**Weighted Pool:**
```json
{
  "kind": "weightedProduct",
  "tokens": {
    "0x6810e776880c02933d47db1b9fc05908e5386b96": {
      "balance": "11260752191375725565253",
      "scalingFactor": "1",
      "weight": "0.5"
    },
    "0xdef1ca1fb7fbcdc777520aa7f396b4e015f497ab": {
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
    "0x6b175474e89094c44da98b954eedeac495271d0f": {
      "balance": "505781036390938593206504",
      "scalingFactor": "1"
    },
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": {
      "balance": "554894862074",
      "scalingFactor": "1000000000000"
    },
    "0xdac17f958d2ee523a2206206994597c13d831ec7": {
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

**Key fields:**
- `scalingFactor`: Provided in auction data (NOT computed). For 6-decimal tokens like USDC: `10^12` to normalize to 18 decimals.
- `gasEstimate`: Use this value from auction data, not hardcoded constants.
- `balancerPoolId`: 32-byte pool identifier used for settlement encoding.

### Key Differences from V2/V3

| Aspect | UniswapV2 | UniswapV3 | Balancer/Curve |
|--------|-----------|-----------|----------------|
| Math | Local (constant product) | RPC (QuoterV2) | Local (fixed-point) |
| Tokens | 2 | 2 | N (2+) |
| Settlement | Uniswap Router | SwapRouterV2 | Balancer Vault |
| Pool ID | Address | Address | 32-byte ID |
| Multi-hop | Yes (via base tokens) | Yes | Yes (via base tokens) |

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

**Interaction format:** The solver returns `kind: "liquidity"` interactions (same as V2/V3), not raw calldata. The driver handles encoding.

```python
Interaction(
    kind="liquidity",
    internalize=False,
    id=pool.id,  # Pool ID from auction
    inputToken=token_in,
    outputToken=token_out,
    inputAmount=str(amount_in),
    outputAmount=str(amount_out),
)
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
      """18-decimal fixed-point number stored as int.

      All values are stored as integers scaled by 10^18.
      Example: 1.5 is stored as 1_500_000_000_000_000_000
      """
      ONE = 10**18

      def __init__(self, value: int):
          self.value = value

      @classmethod
      def from_wei(cls, wei: int) -> Bfp:
          """Create from raw wei value (already scaled)."""
          return cls(wei)

      @classmethod
      def from_decimal(cls, d: Decimal) -> Bfp:
          """Create from decimal (will be scaled by 10^18)."""
          return cls(int(d * cls.ONE))

      def mul_down(self, other: Bfp) -> Bfp:
          """Multiply with floor rounding: (a * b) // 10^18"""
          return Bfp((self.value * other.value) // self.ONE)

      def mul_up(self, other: Bfp) -> Bfp:
          """Multiply with ceiling rounding."""
          product = self.value * other.value
          return Bfp((product + self.ONE - 1) // self.ONE)

      def div_down(self, other: Bfp) -> Bfp:
          """Divide with floor rounding: (a * 10^18) // b"""
          if other.value == 0:
              raise ZeroDivisionError("Bfp division by zero")
          return Bfp((self.value * self.ONE) // other.value)

      def div_up(self, other: Bfp) -> Bfp:
          """Divide with ceiling rounding."""
          if other.value == 0:
              raise ZeroDivisionError("Bfp division by zero")
          numerator = self.value * self.ONE
          return Bfp((numerator + other.value - 1) // other.value)

      def complement(self) -> Bfp:
          """Return 1 - self. Requires self <= 1."""
          return Bfp(self.ONE - self.value)

      def add(self, other: Bfp) -> Bfp:
          return Bfp(self.value + other.value)

      def sub(self, other: Bfp) -> Bfp:
          return Bfp(self.value - other.value)
  ```

- [ ] Implement power function (complex - see detailed spec below)

- [ ] Add constants
  ```python
  MAX_IN_RATIO = Bfp.from_wei(3 * 10**17)   # 0.3 (30%)
  MAX_OUT_RATIO = Bfp.from_wei(3 * 10**17)  # 0.3 (30%)
  AMP_PRECISION = 1000
  ```

#### Power Function Specification

The power function `pow_up(base, exp)` is the most complex part. Balancer uses a Taylor series approximation with different strategies for different ranges:

```python
def pow_up(self, exp: Bfp) -> Bfp:
    """
    Compute self^exp with upward rounding.

    Strategy (matching Rust/Solidity):
    1. If exp == 0: return 1
    2. If exp == ONE (1.0): return self
    3. If exp == 2*ONE (2.0): return self * self (mul_up)
    4. If exp == ONE/2 (0.5): return sqrt(self)
    5. Otherwise: use natural log/exp approximation
       result = exp(exp * ln(self))

    The ln and exp functions use Taylor series:
    - ln(x) around x=1 for x in [0.8, 1.2]
    - For x outside this range, use: ln(x) = ln(x/2^k) + k*ln(2)

    V0 vs V3Plus difference:
    - V0: Standard rounding
    - V3Plus: For weights 0.5 and 0.2/0.8, uses optimized direct formulas
    """
```

**Implementation approach:**
1. Start with Python's `Decimal` for development/testing
2. Implement exact integer math version matching Rust
3. Verify against Rust test vectors

**Tests:**
- `tests/unit/test_fixed_point.py::test_mul_down_rounds_down`
- `tests/unit/test_fixed_point.py::test_mul_up_rounds_up`
- `tests/unit/test_fixed_point.py::test_div_precision`
- `tests/unit/test_fixed_point.py::test_pow_identity_cases` (exp=0, 1, 2)
- `tests/unit/test_fixed_point.py::test_pow_sqrt`
- `tests/unit/test_fixed_point.py::test_pow_matches_rust_vectors`

**No external dependencies:** ✅

---

### Slice 3.2.2: Weighted Pool Math

**Goal:** Implement Balancer weighted pool swap calculations

**Files to create:**
- `solver/amm/balancer.py` (new - consolidated file for all Balancer/Curve code)

**Tasks:**

- [ ] Create pool dataclasses
  ```python
  @dataclass
  class WeightedTokenReserve:
      token: str
      balance: int           # Raw balance from auction
      weight: Decimal        # Normalized weight (sum to 1.0)
      scaling_factor: int    # From auction data (e.g., 10^12 for USDC)

  @dataclass
  class BalancerWeightedPool:
      id: str                # Liquidity ID from auction (for Interaction)
      address: str
      pool_id: str           # balancerPoolId (32-byte hex for settlement)
      reserves: list[WeightedTokenReserve]  # Sorted by token address
      fee: Decimal
      version: Literal["v0", "v3Plus"]
      gas_estimate: int      # From auction gasEstimate field
  ```

- [ ] Implement `calc_out_given_in()`
  ```python
  def calc_out_given_in(
      balance_in: Bfp,
      weight_in: Bfp,
      balance_out: Bfp,
      weight_out: Bfp,
      amount_in: Bfp,
      version: str = "v0",
  ) -> Bfp | None:
      """
      Calculate output amount for a given input.

      Returns None if:
      - amount_in > balance_in * MAX_IN_RATIO (30%)

      Formula:
        base = balance_in / (balance_in + amount_in)
        exponent = weight_in / weight_out
        power = base ^ exponent
        amount_out = balance_out * (1 - power)

      Fee handling: Fee is subtracted from amount_in BEFORE this calculation.
      The caller must apply: amount_in_after_fee = amount_in * (1 - fee)
      """
      # Check ratio limit
      if amount_in.value > balance_in.mul_down(MAX_IN_RATIO).value:
          return None  # Exceeds 30% ratio limit

      denominator = balance_in.add(amount_in)
      base = balance_in.div_up(denominator)
      exponent = weight_in.div_down(weight_out)
      power = base.pow_up(exponent) if version == "v0" else base.pow_up_v3(exponent)
      return balance_out.mul_down(power.complement())
  ```

- [ ] Implement `calc_in_given_out()`
  ```python
  def calc_in_given_out(
      balance_in: Bfp,
      weight_in: Bfp,
      balance_out: Bfp,
      weight_out: Bfp,
      amount_out: Bfp,
      version: str = "v0",
  ) -> Bfp | None:
      """
      Calculate input amount for a given output.

      Returns None if:
      - amount_out > balance_out * MAX_OUT_RATIO (30%)

      Formula:
        base = balance_out / (balance_out - amount_out)
        exponent = weight_out / weight_in
        power = base ^ exponent
        amount_in = balance_in * (power - 1)

      Fee handling: Fee is added to result AFTER this calculation.
      The caller must apply: amount_in_with_fee = amount_in / (1 - fee)
      """
      # Check ratio limit
      if amount_out.value > balance_out.mul_down(MAX_OUT_RATIO).value:
          return None  # Exceeds 30% ratio limit

      denominator = balance_out.sub(amount_out)
      base = balance_out.div_up(denominator)
      exponent = weight_out.div_down(weight_in)
      power = base.pow_up(exponent) if version == "v0" else base.pow_up_v3(exponent)
      return balance_in.mul_up(power.sub(Bfp.ONE))
  ```

- [ ] Implement V0 vs V3Plus power variants
  ```python
  def pow_up_v3(self, exp: Bfp) -> Bfp:
      """V3+ optimized power for common weight ratios.

      For 50/50 pools (exp = 1.0): direct return
      For 80/20 pools (exp = 0.25 or 4.0): optimized formula
      Otherwise: falls back to standard pow_up
      """
  ```

**Tests:**
- `tests/unit/test_balancer.py::test_weighted_calc_out_given_in`
- `tests/unit/test_balancer.py::test_weighted_calc_in_given_out`
- `tests/unit/test_balancer.py::test_weighted_max_in_ratio_returns_none`
- `tests/unit/test_balancer.py::test_weighted_max_out_ratio_returns_none`
- `tests/unit/test_balancer.py::test_weighted_fee_application`
- `tests/unit/test_balancer.py::test_weighted_matches_rust_output`

**Test vector from Rust:**
```
Pool: GNO/COW 50/50, 0.5% fee
Input: 1 GNO (1000000000000000000 wei)
Expected output: 1657855325872947866705 COW
```

**No external dependencies:** ✅

---

### Slice 3.2.3: Stable Pool Math

**Goal:** Implement Curve StableSwap invariant calculations

**Files to modify:**
- `solver/amm/balancer.py` (add stable pool code)

**Tasks:**

- [ ] Create stable pool dataclass
  ```python
  @dataclass
  class StableTokenReserve:
      token: str
      balance: int           # Raw balance from auction
      scaling_factor: int    # From auction data

  @dataclass
  class BalancerStablePool:
      id: str                # Liquidity ID from auction
      address: str
      pool_id: str           # balancerPoolId (32-byte hex)
      reserves: list[StableTokenReserve]  # Sorted by token address
      amplification_parameter: Decimal  # A parameter
      fee: Decimal
      gas_estimate: int      # From auction gasEstimate field
  ```

- [ ] Implement `calculate_invariant()` (Newton-Raphson)
  ```python
  def calculate_invariant(amp: int, balances: list[Bfp]) -> Bfp | None:
      """
      Calculate StableSwap invariant D using Newton-Raphson iteration.

      Invariant equation:
        A * n^n * sum(x_i) + D = A * D * n^n + D^(n+1) / (n^n * prod(x_i))

      Where:
        A = amplification parameter (scaled by AMP_PRECISION=1000)
        n = number of tokens
        x_i = scaled token balance
        D = invariant to solve for

      Algorithm:
        1. Initial guess: D = sum(balances)
        2. Iterate until |D_new - D_old| <= 1 wei
        3. Max iterations: 255
        4. Return None if doesn't converge
      """
  ```

- [ ] Implement `get_token_balance_given_invariant()`
  ```python
  def get_token_balance_given_invariant_and_all_other_balances(
      amp: int,
      balances: list[Bfp],
      invariant: Bfp,
      token_index: int,
  ) -> Bfp | None:
      """
      Given D and all other balances, solve for balance[token_index].
      Uses Newton-Raphson iteration.

      Max iterations: 255
      Convergence: |y_new - y_old| <= 1 wei
      Returns None if doesn't converge.
      """
  ```

- [ ] Implement `calc_out_given_in()` and `calc_in_given_out()` for stable pools
  ```python
  def stable_calc_out_given_in(
      amp: int,
      balances: list[Bfp],
      token_index_in: int,
      token_index_out: int,
      amount_in: Bfp,
  ) -> Bfp | None:
      """
      Fee handling: Fee is subtracted from amount_in BEFORE this calculation.

      Algorithm:
        1. Calculate current invariant D
        2. Add amount_in to balances[token_index_in]
        3. Solve for new balances[token_index_out] given D
        4. Return: old_balance_out - new_balance_out - 1 (1 wei rounding protection)
      """

  def stable_calc_in_given_out(
      amp: int,
      balances: list[Bfp],
      token_index_in: int,
      token_index_out: int,
      amount_out: Bfp,
  ) -> Bfp | None:
      """
      Fee handling: Fee is added to result AFTER this calculation.

      Algorithm:
        1. Calculate current invariant D
        2. Subtract amount_out from balances[token_index_out]
        3. Solve for new balances[token_index_in] given D
        4. Return: new_balance_in - old_balance_in + 1 (1 wei rounding protection)
      """
  ```

- [ ] Handle composable stable pools (V4)
  ```python
  def filter_bpt_token(pool: BalancerStablePool) -> BalancerStablePool:
      """
      For composable stable pools, the pool's own BPT token is included
      in the reserves. Filter it out before calculations.

      Detection: BPT token address == pool address
      """
      filtered_reserves = [
          r for r in pool.reserves
          if r.token.lower() != pool.address.lower()
      ]
      return dataclasses.replace(pool, reserves=filtered_reserves)
  ```

**Tests:**
- `tests/unit/test_balancer.py::test_stable_invariant_calculation`
- `tests/unit/test_balancer.py::test_stable_calc_out_given_in`
- `tests/unit/test_balancer.py::test_stable_calc_in_given_out`
- `tests/unit/test_balancer.py::test_stable_convergence_failure_returns_none`
- `tests/unit/test_balancer.py::test_stable_bpt_filtering`
- `tests/unit/test_balancer.py::test_stable_matches_rust_output`

**Test vectors from Rust:**
```
Pool: DAI/USDC/USDT stable pool, A=5000, 0.01% fee
Input: 10 DAI (10000000000000000000 wei)
Expected output: 9999475 USDC (6 decimals)

Buy order:
Input: want 10 USDC (10000000 wei)
Expected input: 10000524328839166557 DAI
```

**No external dependencies:** ✅

---

### Slice 3.2.4: Pool Parsing and Registry

**Goal:** Parse Balancer/Curve pools from auction data and integrate with registry

**Files to modify:**
- `solver/amm/uniswap_v2.py` (update PoolRegistry)
- `solver/amm/balancer.py` (add parsing functions)

**Tasks:**

- [ ] Add parsing functions in `balancer.py`
  ```python
  def parse_weighted_pool(liquidity: Liquidity) -> BalancerWeightedPool | None:
      """Parse weightedProduct liquidity into BalancerWeightedPool."""
      if liquidity.kind != "weightedProduct":
          return None

      reserves = []
      for token_addr, token_data in liquidity.tokens.items():
          reserves.append(WeightedTokenReserve(
              token=token_addr,
              balance=int(token_data["balance"]),
              weight=Decimal(token_data["weight"]),
              scaling_factor=int(token_data.get("scalingFactor", "1")),
          ))

      # Sort by token address (required for Balancer)
      reserves.sort(key=lambda r: r.token.lower())

      return BalancerWeightedPool(
          id=liquidity.id,
          address=liquidity.address,
          pool_id=liquidity.balancerPoolId,
          reserves=reserves,
          fee=Decimal(liquidity.fee),
          version=liquidity.version or "v0",
          gas_estimate=int(liquidity.gasEstimate),
      )

  def parse_stable_pool(liquidity: Liquidity) -> BalancerStablePool | None:
      """Parse stable liquidity into BalancerStablePool."""
      if liquidity.kind != "stable":
          return None

      reserves = []
      for token_addr, token_data in liquidity.tokens.items():
          reserves.append(StableTokenReserve(
              token=token_addr,
              balance=int(token_data["balance"]),
              scaling_factor=int(token_data.get("scalingFactor", "1")),
          ))

      # Sort by token address
      reserves.sort(key=lambda r: r.token.lower())

      pool = BalancerStablePool(
          id=liquidity.id,
          address=liquidity.address,
          pool_id=liquidity.balancerPoolId,
          reserves=reserves,
          amplification_parameter=Decimal(liquidity.amplificationParameter),
          fee=Decimal(liquidity.fee),
          gas_estimate=int(liquidity.gasEstimate),
      )

      # Filter BPT for composable stable pools
      return filter_bpt_token(pool)
  ```

- [ ] Update `PoolRegistry` to store Balancer pools
  ```python
  class PoolRegistry:
      def __init__(self):
          self.v2_pools: dict[tuple[str, str], UniswapV2Pool] = {}
          self.v3_pools: dict[tuple[str, str, int], UniswapV3Pool] = {}
          # Index by token pair for O(1) lookup
          # A pool with N tokens creates N*(N-1)/2 pair entries
          self.weighted_pools: dict[tuple[str, str], list[BalancerWeightedPool]] = {}
          self.stable_pools: dict[tuple[str, str], list[BalancerStablePool]] = {}

      def add_weighted_pool(self, pool: BalancerWeightedPool) -> None:
          """Add weighted pool, indexing by all token pairs."""
          tokens = [r.token.lower() for r in pool.reserves]
          for i, t1 in enumerate(tokens):
              for t2 in tokens[i+1:]:
                  pair = (min(t1, t2), max(t1, t2))
                  self.weighted_pools.setdefault(pair, []).append(pool)

      def add_stable_pool(self, pool: BalancerStablePool) -> None:
          """Add stable pool, indexing by all token pairs."""
          tokens = [r.token.lower() for r in pool.reserves]
          for i, t1 in enumerate(tokens):
              for t2 in tokens[i+1:]:
                  pair = (min(t1, t2), max(t1, t2))
                  self.stable_pools.setdefault(pair, []).append(pool)

      def get_pools_for_pair(self, token_a: str, token_b: str) -> list[Pool]:
          """Get all pools (V2 + V3 + weighted + stable) for a token pair."""
          pair = (min(token_a.lower(), token_b.lower()),
                  max(token_a.lower(), token_b.lower()))
          pools = []
          if pair in self.v2_pools:
              pools.append(self.v2_pools[pair])
          pools.extend(self.v3_pools.get((pair[0], pair[1]), {}).values())
          pools.extend(self.weighted_pools.get(pair, []))
          pools.extend(self.stable_pools.get(pair, []))
          return pools
  ```

**Tests:**
- `tests/unit/test_balancer.py::test_parse_weighted_pool`
- `tests/unit/test_balancer.py::test_parse_stable_pool`
- `tests/unit/test_balancer.py::test_parse_composable_stable_filters_bpt`
- `tests/unit/test_balancer.py::test_registry_indexes_by_all_pairs`
- `tests/unit/test_balancer.py::test_registry_get_pools_returns_all_types`

**No external dependencies:** ✅

---

### Slice 3.2.5: AMM Integration

**Goal:** Create AMM classes that implement the swap interface

**Files to modify:**
- `solver/amm/balancer.py` (add AMM classes)

**Tasks:**

- [ ] Create `BalancerWeightedAMM` class
  ```python
  class BalancerWeightedAMM:
      """AMM for Balancer weighted pools."""

      def get_amount_out(
          self, pool: BalancerWeightedPool, amount_in: int, token_in: str
      ) -> SwapResult | None:
          """
          Calculate output amount for selling amount_in of token_in.

          Steps:
          1. Find token_in and token_out reserves
          2. Scale balances by scaling_factor
          3. Apply fee to input: amount_in_after_fee = amount_in * (1 - fee)
          4. Call calc_out_given_in()
          5. Scale output back to native decimals
          6. Return SwapResult with gas from pool.gas_estimate
          """

      def get_amount_in(
          self, pool: BalancerWeightedPool, amount_out: int, token_out: str
      ) -> SwapResult | None:
          """
          Calculate input amount for buying amount_out of token_out.

          Steps:
          1. Find token_in and token_out reserves
          2. Scale balances by scaling_factor
          3. Call calc_in_given_out()
          4. Apply fee to result: amount_in_with_fee = amount_in / (1 - fee)
          5. Scale input back to native decimals
          6. Return SwapResult with gas from pool.gas_estimate
          """
  ```

- [ ] Create `BalancerStableAMM` class
  ```python
  class BalancerStableAMM:
      """AMM for Balancer stable pools (Curve-style)."""

      def get_amount_out(
          self, pool: BalancerStablePool, amount_in: int, token_in: str
      ) -> SwapResult | None:
          """Same pattern as weighted, but uses stable math."""

      def get_amount_in(
          self, pool: BalancerStablePool, amount_out: int, token_out: str
      ) -> SwapResult | None:
          """Same pattern as weighted, but uses stable math."""
  ```

- [ ] Implement decimal scaling helpers
  ```python
  def scale_up(amount: int, scaling_factor: int) -> Bfp:
      """Scale token amount to 18 decimals for math."""
      return Bfp.from_wei(amount * scaling_factor)

  def scale_down(bfp: Bfp, scaling_factor: int) -> int:
      """Scale 18-decimal result back to token decimals."""
      return bfp.value // scaling_factor
  ```

**Tests:**
- `tests/unit/test_balancer.py::test_weighted_amm_get_amount_out`
- `tests/unit/test_balancer.py::test_weighted_amm_get_amount_in`
- `tests/unit/test_balancer.py::test_stable_amm_get_amount_out`
- `tests/unit/test_balancer.py::test_stable_amm_get_amount_in`
- `tests/unit/test_balancer.py::test_amm_decimal_scaling`
- `tests/unit/test_balancer.py::test_amm_returns_none_on_ratio_exceeded`

**No external dependencies:** ✅

---

### Slice 3.2.6: Router Integration

**Goal:** Extend router to use Balancer/Curve pools in direct and multi-hop routing

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
          self.amm = amm or UniswapV2AMM()
          self.v3_amm = v3_amm
          self.weighted_amm = weighted_amm or BalancerWeightedAMM()
          self.stable_amm = stable_amm or BalancerStableAMM()
          self.pool_finder = pool_finder
  ```

- [ ] Update `_find_best_direct_route()` to consider all pool types
  ```python
  def _get_quote(self, pool: Pool, order: Order) -> SwapResult | None:
      """Get quote from any pool type."""
      if isinstance(pool, UniswapV2Pool):
          return self.amm.simulate_swap(pool, ...)
      elif isinstance(pool, UniswapV3Pool):
          return self.v3_amm.get_amount_out(...) if self.v3_amm else None
      elif isinstance(pool, BalancerWeightedPool):
          return self.weighted_amm.get_amount_out(pool, ...)
      elif isinstance(pool, BalancerStablePool):
          return self.stable_amm.get_amount_out(pool, ...)
      return None

  def _find_best_direct_route(self, order, pools) -> RoutingResult | None:
      candidates = []
      for pool in pools:
          result = self._get_quote(pool, order)
          if result:
              candidates.append((pool, result))

      if not candidates:
          return None

      # Select best: max output for sell orders, min input for buy orders
      if order.is_sell_order:
          best = max(candidates, key=lambda x: x[1].amount_out)
      else:
          best = min(candidates, key=lambda x: x[1].amount_in)

      return self._build_routing_result(best[0], best[1], order)
  ```

- [ ] Support multi-hop routing through Balancer pools
  ```python
  # Balancer pools can be intermediate hops (e.g., A → WETH via weighted, WETH → B via stable)
  # The BFS pathfinding in _find_best_multihop_route() already handles this
  # if get_pools_for_pair() returns Balancer pools
  ```

- [ ] Build correct interaction based on pool type
  ```python
  def _build_interaction(self, pool: Pool, swap: SwapResult, order: Order) -> Interaction:
      """Build Interaction for any pool type.

      For all pool types, we return a 'liquidity' kind interaction.
      The driver handles the actual encoding.
      """
      return Interaction(
          kind="liquidity",
          internalize=False,
          id=pool.id,
          inputToken=swap.token_in,
          outputToken=swap.token_out,
          inputAmount=str(swap.amount_in),
          outputAmount=str(swap.amount_out),
      )
  ```

**Tests:**
- `tests/unit/test_router.py::test_router_selects_best_across_all_pool_types`
- `tests/unit/test_router.py::test_router_weighted_better_than_v2`
- `tests/unit/test_router.py::test_router_stable_better_than_v2`
- `tests/unit/test_router.py::test_router_multihop_through_weighted`
- `tests/unit/test_router.py::test_router_multihop_through_stable`

**No external dependencies:** ✅

---

### Slice 3.2.7: Integration Tests

**Goal:** Full solve flow with Balancer/Curve pools

**Files to create:**
- `tests/integration/test_balancer_integration.py`
- `tests/fixtures/auctions/benchmark/weighted_*.json`
- `tests/fixtures/auctions/benchmark/stable_*.json`

**Tasks:**

- [ ] Create weighted pool fixtures (from Rust test cases in `bal_liquidity.rs`)
  - `weighted_gno_to_cow.json` - 50/50 V0 pool
  - `weighted_v3plus.json` - V3+ optimized pool

- [ ] Create stable pool fixtures (from Rust test cases)
  - `stable_dai_to_usdc.json` - 3-token stable pool, sell order
  - `stable_buy_order.json` - 3-token stable pool, buy order
  - `stable_composable.json` - Composable stable pool with BPT

- [ ] Integration test: Weighted pool solve
  ```python
  def test_solve_with_weighted_pool():
      auction = load_fixture("weighted_gno_to_cow.json")
      result = solver.solve(auction)

      assert len(result.solutions) == 1
      solution = result.solutions[0]

      # Verify output matches Rust exactly
      interaction = solution.interactions[0]
      assert interaction.outputAmount == "1657855325872947866705"
  ```

- [ ] Integration test: Stable pool solve
  ```python
  def test_solve_with_stable_pool_sell():
      auction = load_fixture("stable_dai_to_usdc.json")
      result = solver.solve(auction)

      assert len(result.solutions) == 1
      interaction = result.solutions[0].interactions[0]
      assert interaction.outputAmount == "9999475"

  def test_solve_with_stable_pool_buy():
      auction = load_fixture("stable_buy_order.json")
      result = solver.solve(auction)

      assert len(result.solutions) == 1
      interaction = result.solutions[0].interactions[0]
      assert interaction.inputAmount == "10000524328839166557"
  ```

- [ ] Integration test: Best pool selection across types
  ```python
  def test_solver_selects_best_pool_across_types():
      # Auction with V2 + weighted + stable pools for same pair
      # Verify solver picks the one with best output
  ```

- [ ] Integration test: Composable stable pool
  ```python
  def test_solve_with_composable_stable():
      auction = load_fixture("stable_composable.json")
      result = solver.solve(auction)

      # Verify BPT token was filtered and swap works
      assert len(result.solutions) == 1
  ```

**No external dependencies:** ✅

---

### Slice 3.2.8: Benchmarking

**Goal:** Verify Python matches Rust on Balancer/Curve auctions

**Tasks:**

- [ ] Run Python vs Rust benchmarks on weighted fixtures
  ```bash
  python scripts/run_benchmarks.py \
      --python-url http://localhost:8000 \
      --rust-url http://localhost:8080 \
      --auctions tests/fixtures/auctions/benchmark
  ```

- [ ] Run Python vs Rust benchmarks on stable fixtures

- [ ] Document results in BENCHMARKS.md
  - Update feature table: Balancer weighted ❌ → ✅
  - Update feature table: Balancer/Curve stable ❌ → ✅
  - Add benchmark results section

- [ ] Verify exact match with Rust
  - Weighted pools: Should be exact (same math, no RPC)
  - Stable pools: Should be exact (same iterative algorithm)

**Expected Results:**
- Weighted pools: Exact match with Rust
- Stable pools: Exact match with Rust
- Performance: Python ~1.5-2x slower (pure computation, no RPC)

---

## Test Strategy Summary

| Slice | Test Type | External Deps |
|-------|-----------|---------------|
| 3.2.1 Fixed-Point | Unit | None |
| 3.2.2 Weighted Math | Unit | None |
| 3.2.3 Stable Math | Unit | None |
| 3.2.4 Pool Parsing | Unit | None |
| 3.2.5 AMM Integration | Unit | None |
| 3.2.6 Router | Unit | None |
| 3.2.7 Integration | Integration | None |
| 3.2.8 Benchmark | Benchmark | Rust solver |

**Key advantage:** All math is local (no RPC needed), so tests are fast and deterministic.

---

## Dependencies

### Python Dependencies
- None new required
- Use existing `eth-abi` for any encoding needs

### Test Data
- Copy test vectors from Rust solver tests (`bal_liquidity.rs`)
- Ensures exact compatibility

---

## File Structure After Implementation

```
solver/
├── amm/
│   ├── __init__.py
│   ├── base.py              # SwapResult, protocols
│   ├── uniswap_v2.py        # V2 + PoolRegistry (updated)
│   ├── uniswap_v3.py        # V3 quoter-based
│   └── balancer.py          # NEW: All Balancer/Curve code
│                            #   - Weighted pool dataclass + math
│                            #   - Stable pool dataclass + math
│                            #   - Parsing functions
│                            #   - AMM classes
├── math/
│   └── fixed_point.py       # NEW: Bfp 18-decimal math
├── routing/
│   └── router.py            # Updated for all pool types
└── constants.py             # Add Balancer Vault address

tests/
├── unit/
│   ├── test_fixed_point.py  # NEW
│   └── test_balancer.py     # NEW (all Balancer tests in one file)
├── integration/
│   └── test_balancer_integration.py # NEW
└── fixtures/auctions/benchmark/
    ├── weighted_gno_to_cow.json  # NEW
    ├── weighted_v3plus.json      # NEW
    ├── stable_dai_to_usdc.json   # NEW
    ├── stable_buy_order.json     # NEW
    └── stable_composable.json    # NEW
```

---

## Success Criteria

1. [ ] Fixed-point math matches Balancer's Bfp implementation
2. [ ] Weighted pool output matches Rust test vectors exactly
3. [ ] Stable pool output matches Rust test vectors exactly
4. [ ] Composable stable pools work (BPT filtered correctly)
5. [ ] Router selects best pool across V2, V3, weighted, and stable
6. [ ] Multi-hop routing works through Balancer pools
7. [ ] All existing tests still pass
8. [ ] Benchmark shows exact match with Rust on Balancer/Curve auctions
9. [ ] mypy and ruff clean

---

## Estimated Effort

| Slice | Effort |
|-------|--------|
| 3.2.1 Fixed-Point Math | Medium-Large (power function is complex) |
| 3.2.2 Weighted Math | Medium |
| 3.2.3 Stable Math | Medium-Large (iterative algorithm) |
| 3.2.4 Pool Parsing | Small |
| 3.2.5 AMM Integration | Small |
| 3.2.6 Router Integration | Small-Medium |
| 3.2.7 Integration Tests | Medium |
| 3.2.8 Benchmark | Small |

**Total:** ~3-4 sessions of focused work

---

## Key Implementation Notes

### Fixed-Point Precision
- All math uses 18-decimal fixed-point (`Bfp`)
- Rounding direction matters:
  - `mul_down`/`div_down`: Conservative (less output)
  - `mul_up`/`div_up`: Aggressive (more input)
- Power function is the hardest part - consider using Python Decimal initially

### Fee Application Order
- **Sell order (calc_out_given_in):**
  1. Subtract fee from input: `amount_in_after_fee = amount_in * (1 - fee)`
  2. Calculate output using fee-adjusted input
- **Buy order (calc_in_given_out):**
  1. Calculate raw input amount
  2. Add fee to result: `amount_in_with_fee = amount_in / (1 - fee)`

### Stable Pool Convergence
- Newton-Raphson with max 255 iterations
- Convergence when `|new - old| <= 1` wei
- Return `None` on non-convergence (don't raise exception)

### Decimal Scaling
- `scalingFactor` is provided in auction data
- Scale up before math: `scaled = amount * scalingFactor`
- Scale down after math: `result = scaled_result // scalingFactor`

### Token Ordering
- Balancer requires tokens sorted by address (ascending)
- Always normalize to lowercase before sorting/comparing

### Ratio Limits
- `MAX_IN_RATIO = 0.3` - Cannot swap more than 30% of input reserve
- `MAX_OUT_RATIO = 0.3` - Cannot receive more than 30% of output reserve
- Return `None` when exceeded (not an error, just can't fill)

### Composable Stable Pools
- BPT token is included in reserves but must be filtered out
- Detection: `bpt_address == pool_address`
