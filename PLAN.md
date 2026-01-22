# CoW Solver Python Implementation Plan

## Project Goals
1. Build a working CoW Protocol solver in Python
2. Benchmark against the reference Rust implementation
3. Document performance gaps and optimization opportunities
4. Demonstrate AI-assisted development workflow

## Approach: Test-Driven Vertical Slices

Each slice delivers end-to-end functionality for a specific auction type, with tests and benchmarks.

---

## Phase 0: Infrastructure ✅ COMPLETE

### 0.1 Project Skeleton
- [x] `pyproject.toml` with dependencies (FastAPI, Pydantic, web3, httpx, pytest)
- [x] Directory structure
- [x] CI setup (GitHub Actions)
- [x] Pre-commit hooks (ruff, mypy)

### 0.2 Pydantic Models
- [x] `AuctionInstance` - incoming auction request
- [x] `Solution` - solver response
- [x] `Order`, `Token`, `Trade`, `Interaction` models
- [x] Validation against OpenAPI spec

### 0.3 Benchmark Harness
- [x] Rust solver runner (subprocess wrapper)
- [x] Python solver interface (protocol/ABC)
- [x] Timing and scoring comparison
- [x] Results output (JSON, markdown table)

### 0.4 Historical Data Collection
- [x] Auction fetcher from CoW API / driver logs
- [x] Categorization script (by order count, token pairs, complexity)
- [x] Store as JSON fixtures in `tests/fixtures/auctions/`

**Exit Criteria:** Can run `pytest benchmarks/` and see Rust solver results on historical data.
**Status:** Infrastructure complete. Tests not yet run (need `pip install -e ".[dev]"`).

---

## Phase 1: Single Order via DEX

### Slice 1.1: Passthrough (No-op Solver) ✅ COMPLETE
**Goal:** Accept auction, return empty solution (valid but scores 0)

- [x] FastAPI `/solve` endpoint
- [x] Request parsing into Pydantic models
- [x] Return `{"solutions": []}`
- [x] Test: endpoint accepts real auction JSON without error

### Slice 1.2: Single Sell Order → UniswapV2 ✅ COMPLETE
**Goal:** Route one sell order through a single UniV2 pool

- [x] UniswapV2 AMM math (constant product)
- [x] Hardcoded pool addresses for mainnet (WETH/USDC, etc.)
- [x] Calculate output amount given input
- [x] Build `Interaction` with swap calldata
- [x] Calculate clearing prices
- [x] Test: solution scores > 0 on single-order auctions
- [x] Benchmark: compare score and time vs Rust baseline (Python 1.74x slower, 4/4 solutions)

### Slice 1.3: Single Buy Order → UniswapV2 ✅ COMPLETE
**Goal:** Handle buy orders (fixed output, variable input)

- [x] Inverse AMM math (input given output)
- [x] Adjust solution encoding for buy semantics
- [x] Test: buy order auctions pass
- [x] Benchmark: Python 1.95x slower, 5/5 solutions (including buy order)

### Slice 1.4: Multi-hop Routing (A→B→C) ✅ COMPLETE
**Goal:** Route through intermediate tokens when direct pool doesn't exist

- [x] PoolRegistry class for dynamic pool management from auction liquidity
- [x] Token graph construction (adjacency list)
- [x] BFS pathfinding for shortest route (max 2 hops)
- [x] Chain multiple swap interactions via UniswapV2 router path encoding
- [x] Test: multi-hop sell/buy orders (USDC→WETH→DAI and reverse)
- [x] Benchmark: Python 7/7 solutions, Rust 7/7 solutions (both match, ~2x slower)
- [x] Refactored solver to use auction-provided liquidity instead of hardcoded pools
- [x] Configured Rust baseline solver for multi-hop routing (max-hops=1, base-tokens=[WETH])
- [x] Gas estimation: 60k per hop (`POOL_SWAP_GAS_COST`) + 106k settlement overhead

**Exit Criteria:** Solver handles any single-order auction with UniV2 liquidity. ✅

---

## Phase 2: Coincidence of Wants (CoW)

> **Note:** The Rust baseline solver does NOT support CoW matching. It only routes
> individual orders through AMM liquidity. CoW matching is Python-only functionality.
> See `BENCHMARKS.md` for details on benchmark categories.

### Slice 2.1: Perfect CoW Match ✅ COMPLETE
**Goal:** Two orders that exactly offset (A sells X for Y, B sells Y for X)

- [x] Strategy pattern for solution finding (SolutionStrategy protocol)
- [x] CowMatchStrategy for 2-order auctions
- [x] Order pair detection (opposite directions, same tokens)
- [x] Direct settlement without AMM (0 interactions, gas=0)
- [x] Uniform clearing price calculation
- [x] Test: 20 unit tests for CoW matching (all order type combinations)
- [x] Benchmark: Python-only (Rust baseline doesn't support CoW matching)

### Slice 2.2: Partial CoW + AMM Remainder ✅ COMPLETE
**Goal:** Match what we can peer-to-peer, route remainder through AMM

- [x] Composable strategy architecture (StrategyResult, OrderFill)
- [x] Partial matching for ALL order type combinations (sell-sell, sell-buy, buy-sell, buy-buy)
- [x] Fill-or-kill semantics (partiallyFillable=false enforcement)
- [x] New UID generation for remainder orders (SHA-256 derived)
- [x] original_uid tracking for fill merging across strategies
- [x] PriceWorsened exception for limit price validation
- [x] Multi-order AMM routing with pool reserve updates
- [x] Split order execution (CoW portion + AMM portion)
- [x] Merged fills from multiple strategies into single trades
- [x] Test: 47 CoW match tests (including partial scenarios, all order types)
- [x] Test: Integration tests for partial CoW + AMM composition
- [x] Benchmark fixtures: `partial_cow_amm.json`, `fok_perfect_match.json`, `mixed_partial_fok.json`
- [x] Data-driven matching rules (matching_rules.py) for auditability

**Exit Criteria:** Solver handles 2-order CoW matching with all order type combinations. ✅

> **Note on Multi-Order CoW (formerly Slice 2.3):**
> Multi-order CoW detection has been deferred to Phase 4 (Unified Optimization).
>
> The optimal solution for N-order matching isn't just "better CoW detection" — it's a
> joint optimization across ALL mechanisms (CoW + multiple AMM sources). Solving CoW
> in isolation would create technical debt that needs refactoring when we add more
> liquidity sources.
>
> We'll first expand liquidity sources (Phase 3) to understand the full problem space,
> then design a unified optimizer that handles CoW matching, AMM routing, and their
> interactions together.

---

## Phase 3: Liquidity Expansion ⬅️ CURRENT FOCUS

> **Why this is next:** Before designing a unified optimizer for multi-order CoW + AMM,
> we need to understand the full liquidity landscape. Each AMM type has different
> price curves (constant product, concentrated liquidity, weighted pools) that affect
> the optimization problem. Adding these sources first lets us:
> 1. Achieve feature parity with the Rust baseline
> 2. Understand the real complexity of multi-source routing
> 3. Measure where our sequential approach leaves surplus on the table
> 4. Design the unified optimizer with full context

### Slice 3.1: UniswapV3 Integration
**Goal:** Add concentrated liquidity support (very different from V2)

- [ ] Concentrated liquidity math (liquidity within tick ranges)
- [ ] Tick-based price calculation
- [ ] Parse UniswapV3 liquidity from auction data
- [ ] Handle tick crossing during swaps
- [ ] Test: auctions where V3 beats V2
- [ ] Benchmark: compare with Rust baseline on V3 pools

### Slice 3.2: Balancer/Curve Integration
**Goal:** Add weighted and stable pool support

- [ ] Weighted pool math (Balancer)
- [ ] Stable pool math (Curve/Balancer)
- [ ] Parse pool parameters from auction data
- [ ] Test: auctions with Balancer/Curve liquidity
- [ ] Benchmark: compare with Rust baseline

### Slice 3.3: Multi-Source Routing
**Goal:** Route through best available liquidity

- [ ] Quote comparison across DEXs (V2, V3, Balancer)
- [ ] Select best execution venue per order
- [ ] Test: auctions with multiple liquidity options
- [ ] Benchmark: measure improvement over single-source

### Slice 3.4: Split Routing (Optional)
**Goal:** Split orders across multiple venues for better execution

- [ ] Identify when splitting improves execution
- [ ] Calculate optimal split ratios
- [ ] Test: large orders that benefit from splitting
- [ ] Benchmark: this approaches unified optimization territory

**Exit Criteria:** Solver uses multiple liquidity sources, matching Rust baseline capabilities.

---

## Phase 4: Unified Optimization

> **The Big Picture:** The optimal solution isn't "CoW matching" OR "AMM routing" —
> it's the joint optimization across all mechanisms. This phase designs a unified
> optimizer that considers:
> - Multi-order CoW matching (N orders, not just pairs)
> - Multiple AMM sources with different price curves
> - Ring trades (A→B→C→A cycles)
> - Partial fills across mechanisms
> - Gas costs in the objective function

### Slice 4.1: Problem Formulation
**Goal:** Define the optimization problem precisely

- [ ] Formalize as constraint optimization (variables, constraints, objective)
- [ ] Identify which constraints are linear vs non-linear
- [ ] Analyze problem structure (decomposition opportunities)
- [ ] Document complexity and tractability
- [ ] Decide on solver approach (custom algorithm vs LP/MIP vs heuristic)

### Slice 4.2: Multi-Order CoW Detection
**Goal:** Find CoW opportunities across N orders (moved from Phase 2)

- [ ] Token pair decomposition (independent subproblems)
- [ ] Double auction clearing for optimal pairwise matching
- [ ] Handle fill-or-kill constraints
- [ ] Test: 5+ order auctions with CoW potential
- [ ] Benchmark: measure surplus vs 2-order matching

### Slice 4.3: Unified Solver
**Goal:** Joint optimization across CoW + all AMM sources

- [ ] Implement chosen optimization approach
- [ ] Handle non-linear AMM price curves
- [ ] Compare against sequential composition baseline
- [ ] Test: auctions where joint optimization beats sequential
- [ ] Benchmark: measure surplus improvement and latency

### Slice 4.4: Ring Trade Detection (Optional)
**Goal:** Find cyclic trading opportunities (A→B→C→A)

- [ ] Build token graph from orders
- [ ] Find short cycles (3-4 tokens max)
- [ ] Calculate ring trade feasibility and surplus
- [ ] Test: auctions with ring trade potential
- [ ] Benchmark: measure frequency and value of ring trades

**Exit Criteria:** Solver finds globally optimal (or near-optimal) solutions across all mechanisms.

---

## Phase 5: Performance Optimization

### Slice 5.1: Profiling
- [ ] Profile on 100 historical auctions
- [ ] Identify top 5 hotspots
- [ ] Document findings

### Slice 5.2: Cython Hot Paths
- [ ] Convert identified hotspots to Cython
- [ ] Benchmark before/after
- [ ] Document speedup

### Slice 5.3: Algorithmic Improvements
- [ ] Parallel solution evaluation
- [ ] Caching for repeated calculations
- [ ] Early termination heuristics

**Exit Criteria:** Documented optimization journey with metrics.

---

## Phase 6: Production Readiness (Optional)

### Slice 6.1: Error Handling & Resilience
- [ ] Timeout handling
- [ ] Malformed input handling
- [ ] Graceful degradation

### Slice 6.2: Observability
- [ ] Structured logging
- [ ] Metrics (solution time, score distribution)
- [ ] Health endpoint

### Slice 6.3: Shadow Mode Testing
- [ ] Run against live auctions (shadow competition)
- [ ] Compare with winning solutions
- [ ] Identify gaps

---

## Vertical Slice Template

Each slice follows this workflow:

```
1. SELECT target auction type
2. COLLECT 5-10 fixture auctions of that type
3. WRITE failing tests
4. IMPLEMENT minimum code to pass
5. BENCHMARK against Rust
6. DOCUMENT results
7. OPTIMIZE if gap > 10x (time) or < 80% (score)
```

---

## Session Handoff Format

End each development session with:

```markdown
## Session N Summary
**Date:** YYYY-MM-DD
**Slice:** X.Y - Description

### Completed
- [x] Task 1
- [x] Task 2

### Test Results
- Passing: X/Y
- Failing: Z (reason)

### Benchmark Results
| Metric | Python | Rust | Ratio |
|--------|--------|------|-------|
| Time (ms) | 150 | 12 | 12.5x |
| Score | 1.2e18 | 1.3e18 | 92% |

### Next Session
- Slice X.Z: Description
- Open questions: ...

### Files Changed
- `solver/amm/uniswap_v2.py` - Added swap math
- `tests/test_single_order.py` - New test cases
```

---

## Directory Structure

```
cow-solver-py/
├── pyproject.toml
├── README.md
├── PLAN.md                    # This file
├── BENCHMARKS.md              # Benchmarking guide
├── docs/sessions/             # Session handoff logs
│
├── solver/
│   ├── __init__.py
│   ├── constants.py           # Centralized constants
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app
│   │   └── endpoints.py       # /solve endpoint (with DI)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── auction.py         # AuctionInstance, Order, Token
│   │   ├── solution.py        # Solution, Trade, Interaction
│   │   └── types.py           # Shared types (HexAddress, etc.)
│   │
│   ├── amm/
│   │   ├── __init__.py
│   │   ├── base.py            # SwapResult dataclass
│   │   └── uniswap_v2.py      # UniswapV2 implementation
│   │
│   └── routing/
│       ├── __init__.py
│       └── router.py          # Order routing logic (with DI)
│
├── benchmarks/
│   ├── __init__.py
│   ├── harness.py             # Main benchmark runner
│   ├── rust_runner.py         # Subprocess wrapper for Rust solver
│   ├── metrics.py             # Comparison calculations
│   └── report.py              # Output formatting
│
├── scripts/
│   ├── collect_auctions.py    # Fetch historical data
│   └── run_benchmarks.py      # HTTP benchmark runner
│
├── tests/
│   ├── conftest.py            # Fixtures + mock classes for DI
│   ├── fixtures/
│   │   └── auctions/          # Historical auction JSON files
│   │       ├── single_order/
│   │       ├── cow_pairs/
│   │       ├── benchmark/              # Shared: Python vs Rust comparison
│   │       └── benchmark_python_only/  # Python-only features (CoW matching)
│   │
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_amm.py
│   │   └── test_router.py
│   │
│   └── integration/
│       ├── test_api.py
│       └── test_single_order.py
│
└── cython_modules/            # Added in Phase 5
    ├── setup.py
    └── fast_amm.pyx
```

---

## Dependencies

### Core
- `fastapi` - API framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `httpx` - Async HTTP client

### Web3
- `web3` - Ethereum interaction
- `eth-abi` - ABI encoding

### Optimization (Phase 5)
- `cython` - Performance optimization
- `numpy` - Numerical operations

### Testing
- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `pytest-benchmark` - Timing

### Dev
- `ruff` - Linting
- `mypy` - Type checking

---

## Success Metrics

### Correctness
- All solutions pass driver validation
- No negative scores (invalid solutions)

### Performance
- Response time < 5 seconds for 95% of auctions
- Score within 80% of Rust baseline for 90% of auctions

### Portfolio Value
- Clear documentation of approach
- Quantified performance comparisons
- Demonstrated optimization methodology
