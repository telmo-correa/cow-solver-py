# UniswapV3 Implementation Plan

## Overview

Add UniswapV3 support matching the Rust solver's approach:
- **V2:** Local math (existing)
- **V3:** QuoterV2 contract for quotes, SwapRouterV2 for settlement

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Auction Data                              │
│  liquidity: [{ kind: "constantProduct", ... },                  │
│              { kind: "concentratedLiquidity", ... }]            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       PoolRegistry                               │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ V2 Pools        │    │ V3 Pools        │                     │
│  │ (local math)    │    │ (quoter-based)  │                     │
│  └─────────────────┘    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SingleOrderRouter                           │
│  1. Find candidate pools (V2 + V3)                              │
│  2. Get quotes from each (local for V2, RPC for V3)             │
│  3. Select best quote                                            │
│  4. Build solution with appropriate encoding                     │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Slices

---

### Slice 3.1.1: V3 Data Structures & Parsing

**Goal:** Parse V3 pools from auction liquidity data

**Files to create/modify:**
- `solver/amm/uniswap_v3.py` (new)
- `solver/amm/pool_registry.py` (new, extract from router.py)

**Tasks:**

- [ ] Create `UniswapV3Pool` dataclass
  ```python
  @dataclass
  class UniswapV3Pool:
      address: str
      router: str
      token0: str
      token1: str
      fee: int                      # 500, 3000, 10000
      sqrt_price_x96: int
      liquidity: int
      tick: int
      liquidity_net: dict[int, int]  # tick -> net liquidity change
  ```

- [ ] Create `parse_v3_liquidity()` function
  ```python
  def parse_v3_liquidity(liquidity_data: dict) -> UniswapV3Pool | None:
      if liquidity_data.get("kind") != "concentratedLiquidity":
          return None
      ...
  ```

- [ ] Extract `PoolRegistry` from router.py (refactor)
  - Move V2 pool parsing to new module
  - Add V3 pool storage
  - Unified interface for getting pools by token pair

- [ ] Add fee tier constants
  ```python
  V3_FEE_TIERS = [100, 500, 3000, 10000]  # 0.01%, 0.05%, 0.3%, 1%
  ```

**Tests:**
- `tests/unit/test_uniswap_v3.py::test_parse_v3_pool_from_auction`
- `tests/unit/test_uniswap_v3.py::test_parse_v3_pool_with_liquidity_net`
- `tests/unit/test_uniswap_v3.py::test_parse_ignores_non_v3_liquidity`

**Fixtures needed:**
- `tests/fixtures/liquidity/v3_pool_basic.json`
- `tests/fixtures/liquidity/v3_pool_with_ticks.json`

**No RPC required:** ✅

---

### Slice 3.1.2: V3 Quoter Interface

**Goal:** Create quoter abstraction with dependency injection for testing

**Files to create/modify:**
- `solver/amm/uniswap_v3.py` (add quoter)

**Tasks:**

- [ ] Define `UniswapV3Quoter` protocol
  ```python
  class UniswapV3Quoter(Protocol):
      def quote_exact_input(
          self, token_in: str, token_out: str, fee: int, amount_in: int
      ) -> int | None: ...

      def quote_exact_output(
          self, token_in: str, token_out: str, fee: int, amount_out: int
      ) -> int | None: ...
  ```

- [ ] Implement `Web3UniswapV3Quoter` (real RPC)
  ```python
  class Web3UniswapV3Quoter:
      QUOTER_ADDRESS = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"

      def __init__(self, web3_provider: str):
          self.w3 = Web3(Web3.HTTPProvider(web3_provider))
          ...
  ```

- [ ] Implement `MockUniswapV3Quoter` for testing
  ```python
  class MockUniswapV3Quoter:
      def __init__(self, quotes: dict[tuple, int]):
          self.quotes = quotes  # (token_in, token_out, fee, amount) -> result
          self.calls: list[tuple] = []  # Track calls for assertions
  ```

- [ ] Add QuoterV2 ABI (minimal, just the functions we need)
  ```python
  QUOTER_V2_ABI = [
      {
          "name": "quoteExactInputSingle",
          "type": "function",
          "inputs": [...],
          "outputs": [...]
      },
      ...
  ]
  ```

**Tests:**
- `tests/unit/test_uniswap_v3.py::test_mock_quoter_returns_configured_value`
- `tests/unit/test_uniswap_v3.py::test_mock_quoter_tracks_calls`
- `tests/unit/test_uniswap_v3.py::test_mock_quoter_returns_none_for_unknown`

**No RPC required:** ✅ (mock quoter for unit tests)

---

### Slice 3.1.3: V3 Settlement Encoding

**Goal:** Encode SwapRouterV2 calldata for V3 swaps

**Files to create/modify:**
- `solver/amm/uniswap_v3.py` (add encoder)

**Tasks:**

- [ ] Add SwapRouterV2 address constant
  ```python
  SWAP_ROUTER_V2 = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"
  ```

- [ ] Implement `encode_exact_input_single()`
  ```python
  def encode_exact_input_single(
      token_in: str,
      token_out: str,
      fee: int,
      recipient: str,
      amount_in: int,
      amount_out_minimum: int,
  ) -> bytes:
      """Encode SwapRouterV2.exactInputSingle call."""
      ...
  ```

- [ ] Implement `encode_exact_output_single()`
  ```python
  def encode_exact_output_single(
      token_in: str,
      token_out: str,
      fee: int,
      recipient: str,
      amount_out: int,
      amount_in_maximum: int,
  ) -> bytes:
      """Encode SwapRouterV2.exactOutputSingle call."""
      ...
  ```

- [ ] Add function selectors and ABI encoding
  ```python
  # exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))
  EXACT_INPUT_SINGLE_SELECTOR = "0x04e45aaf"

  # exactOutputSingle((address,address,uint24,address,uint256,uint256,uint160))
  EXACT_OUTPUT_SINGLE_SELECTOR = "0x5023b4df"
  ```

**Tests:**
- `tests/unit/test_uniswap_v3.py::test_encode_exact_input_single`
- `tests/unit/test_uniswap_v3.py::test_encode_exact_output_single`
- `tests/unit/test_uniswap_v3.py::test_encoding_matches_expected_calldata`

**No RPC required:** ✅

---

### Slice 3.1.4: V3 AMM Integration

**Goal:** Create UniswapV3AMM class that implements the AMM interface

**Files to create/modify:**
- `solver/amm/uniswap_v3.py` (add AMM class)
- `solver/amm/base.py` (ensure protocol is compatible)

**Tasks:**

- [ ] Create `UniswapV3AMM` class
  ```python
  class UniswapV3AMM:
      """UniswapV3 AMM using QuoterV2 for quotes."""

      def __init__(self, quoter: UniswapV3Quoter | None = None):
          self.quoter = quoter

      def get_amount_out(
          self, pool: UniswapV3Pool, amount_in: int, token_in: str
      ) -> SwapResult | None:
          if self.quoter is None:
              return None  # V3 disabled
          ...

      def get_amount_in(
          self, pool: UniswapV3Pool, amount_out: int, token_out: str
      ) -> SwapResult | None:
          ...

      def encode_swap(
          self, pool: UniswapV3Pool, swap_result: SwapResult, ...
      ) -> bytes:
          ...
  ```

- [ ] Handle quoter failures gracefully (return None, log warning)

- [ ] Add gas estimation constant
  ```python
  V3_SWAP_GAS_COST = 106_000  # From Rust solver
  ```

**Tests:**
- `tests/unit/test_uniswap_v3.py::test_v3_amm_returns_none_when_no_quoter`
- `tests/unit/test_uniswap_v3.py::test_v3_amm_uses_quoter_for_amount_out`
- `tests/unit/test_uniswap_v3.py::test_v3_amm_uses_quoter_for_amount_in`
- `tests/unit/test_uniswap_v3.py::test_v3_amm_returns_correct_gas_estimate`

**No RPC required:** ✅ (mock quoter)

---

### Slice 3.1.5: Router Integration

**Goal:** Extend router to use V3 pools alongside V2

**Files to create/modify:**
- `solver/routing/router.py`
- `solver/amm/pool_registry.py`

**Tasks:**

- [ ] Update `PoolRegistry` to store V3 pools
  ```python
  class PoolRegistry:
      def __init__(self):
          self.v2_pools: dict[tuple[str, str], UniswapV2Pool] = {}
          self.v3_pools: dict[tuple[str, str, int], UniswapV3Pool] = {}

      def get_pools_for_pair(self, token_a: str, token_b: str) -> list[Pool]:
          """Get all pools (V2 + V3) for a token pair."""
          ...
  ```

- [ ] Update `SingleOrderRouter` to accept V3 AMM
  ```python
  class SingleOrderRouter:
      def __init__(
          self,
          amm: UniswapV2AMM | None = None,
          v3_amm: UniswapV3AMM | None = None,
          pool_finder: PoolFinder | None = None,
      ):
          ...
  ```

- [ ] Implement best-quote selection across V2 and V3
  ```python
  def _find_best_route(self, order, pools) -> RoutingResult | None:
      candidates = []
      for pool in pools:
          if isinstance(pool, UniswapV2Pool):
              result = self.amm.simulate_swap(...)
          elif isinstance(pool, UniswapV3Pool):
              result = self.v3_amm.get_amount_out(...)
          if result:
              candidates.append((pool, result))

      # Select best by output amount
      return max(candidates, key=lambda x: x[1].amount_out)
  ```

- [ ] Build correct interaction based on pool type

**Tests:**
- `tests/unit/test_router.py::test_router_uses_v2_when_no_v3`
- `tests/unit/test_router.py::test_router_uses_v3_when_better_quote`
- `tests/unit/test_router.py::test_router_uses_v2_when_better_quote`
- `tests/unit/test_router.py::test_router_skips_v3_when_quoter_unavailable`

**No RPC required:** ✅ (mock quoter)

---

### Slice 3.1.6: Integration Tests with Mock Quoter

**Goal:** Full solve flow with V3 pools using mock quoter

**Files to create/modify:**
- `tests/integration/test_uniswap_v3.py` (new)
- `tests/fixtures/auctions/v3/` (new directory)

**Tasks:**

- [ ] Create V3 auction fixtures
  - `v3_single_order.json` - Single order, V3 pool only
  - `v2_v3_mixed.json` - Both V2 and V3 pools, V3 is better
  - `v2_better_than_v3.json` - Both pools, V2 is better

- [ ] Integration test: V3-only solve
  ```python
  def test_solve_with_v3_pool():
      mock_quoter = MockUniswapV3Quoter({...})
      solver = Solver(v3_amm=UniswapV3AMM(quoter=mock_quoter))

      auction = load_fixture("v3_single_order.json")
      result = solver.solve(auction)

      assert len(result.solutions) == 1
      assert result.solutions[0].interactions[0].target == SWAP_ROUTER_V2
  ```

- [ ] Integration test: V2 vs V3 selection
  ```python
  def test_solver_selects_better_pool():
      # Configure mock so V3 gives better output
      mock_quoter = MockUniswapV3Quoter({...})
      ...
  ```

- [ ] Integration test: API endpoint with V3
  ```python
  def test_api_returns_v3_solution():
      # Inject mock quoter via dependency override
      ...
  ```

**No RPC required:** ✅ (mock quoter)

---

### Slice 3.1.7: Real Quoter Integration (Optional)

**Goal:** Test with real QuoterV2 contract via RPC

**Files to create/modify:**
- `tests/integration/test_v3_quoter_real.py` (new)

**Tasks:**

- [ ] Add pytest marker for RPC tests
  ```python
  # conftest.py
  def pytest_configure(config):
      config.addinivalue_line("markers", "requires_rpc: test requires RPC connection")
  ```

- [ ] Skip RPC tests by default
  ```python
  @pytest.mark.requires_rpc
  @pytest.mark.skipif(not os.environ.get("RPC_URL"), reason="RPC_URL not set")
  def test_real_quoter():
      ...
  ```

- [ ] Test real quoter returns sensible values
  ```python
  def test_quoter_weth_usdc():
      quoter = Web3UniswapV3Quoter(os.environ["RPC_URL"])

      # Quote 1 WETH -> USDC
      amount_out = quoter.quote_exact_input(
          WETH, USDC, 3000, 10**18
      )

      # Should be roughly $2000-4000 USDC (sanity check)
      assert 2000 * 10**6 < amount_out < 4000 * 10**6
  ```

**Requires RPC:** ⚠️ (optional, skipped by default)

---

### Slice 3.1.8: Benchmarking

**Goal:** Compare Python V3 performance with Rust baseline

**Files to create/modify:**
- `tests/fixtures/auctions/benchmark/` (add V3 fixtures)
- `scripts/run_benchmarks.py` (if needed)

**Tasks:**

- [ ] Create benchmark fixtures with V3 liquidity
  - Copy from real auctions that have `concentratedLiquidity`

- [ ] Run benchmark: Python vs Rust on V3 auctions
  ```bash
  python scripts/run_benchmarks.py \
      --python-url http://localhost:8000 \
      --rust-url http://localhost:8080 \
      --auctions tests/fixtures/auctions/benchmark
  ```

- [ ] Document results in session file

**Requires RPC:** ⚠️ (for real quotes, or mock for latency testing)

---

## Test Strategy Summary

| Slice | Test Type | RPC Required |
|-------|-----------|--------------|
| 3.1.1 Parsing | Unit | No |
| 3.1.2 Quoter Interface | Unit | No (mock) |
| 3.1.3 Encoding | Unit | No |
| 3.1.4 V3 AMM | Unit | No (mock) |
| 3.1.5 Router | Unit | No (mock) |
| 3.1.6 Integration | Integration | No (mock) |
| 3.1.7 Real Quoter | Integration | Yes (optional) |
| 3.1.8 Benchmark | Benchmark | Yes (optional) |

## Dependencies

### New Dependencies

None required. We'll use existing `web3` and `eth-abi` packages.

### Contract ABIs

Minimal ABIs (just the functions we need):
- QuoterV2: `quoteExactInputSingle`, `quoteExactOutputSingle`
- SwapRouterV2: `exactInputSingle`, `exactOutputSingle`

## File Structure After Implementation

```
solver/
├── amm/
│   ├── __init__.py
│   ├── base.py              # SwapResult, protocols
│   ├── uniswap_v2.py        # Existing V2 implementation
│   ├── uniswap_v3.py        # NEW: V3 pool, quoter, encoder, AMM
│   └── pool_registry.py     # NEW: Unified pool registry
├── routing/
│   └── router.py            # Updated to use V3
└── constants.py             # Add V3 addresses

tests/
├── unit/
│   ├── test_uniswap_v3.py   # NEW: V3 unit tests
│   └── ...
├── integration/
│   ├── test_uniswap_v3.py   # NEW: V3 integration tests
│   └── ...
└── fixtures/
    ├── auctions/
    │   └── v3/              # NEW: V3 auction fixtures
    └── liquidity/
        └── v3_pool_*.json   # NEW: V3 pool fixtures
```

## Success Criteria

1. ✅ V3 pools parsed correctly from auction data
2. ✅ Mock quoter enables full unit/integration testing without RPC
3. ✅ Router selects best pool across V2 and V3
4. ✅ Solutions use correct encoding (SwapRouterV2 for V3)
5. ✅ All existing tests still pass
6. ✅ Benchmark shows parity with Rust on V3 auctions

## Estimated Effort

| Slice | Effort |
|-------|--------|
| 3.1.1 Parsing | Small |
| 3.1.2 Quoter Interface | Small |
| 3.1.3 Encoding | Small |
| 3.1.4 V3 AMM | Medium |
| 3.1.5 Router Integration | Medium |
| 3.1.6 Integration Tests | Medium |
| 3.1.7 Real Quoter | Small (optional) |
| 3.1.8 Benchmark | Small |

**Total:** ~1-2 sessions of focused work
