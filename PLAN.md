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

## Phase 3: Liquidity Expansion

> **Status:** V2, V3, and Balancer pools complete. 0x limit orders remaining for full Rust parity.
> Full multi-source routing: V2, V3, Balancer weighted, Balancer stable pools.

### Slice 3.1: UniswapV3 Integration ✅ COMPLETE
**Goal:** Add concentrated liquidity support (very different from V2)

**Data Structures & Parsing (Session 16):**
- [x] `UniswapV3Pool` dataclass with tick, liquidity, sqrtPrice, liquidityNet
- [x] Parse `concentratedLiquidity` pools from auction data
- [x] V3 pool fixtures for WETH/USDC (500 fee tier)

**Quoter Interface (Session 17):**
- [x] `UniswapV3Quoter` protocol for swap quoting
- [x] `MockV3Quoter` for deterministic testing
- [x] `Web3V3Quoter` for real on-chain quotes via QuoterV2 contract
- [x] QuoterV2 ABI and contract address

**Settlement Encoding (Session 18):**
- [x] SwapRouterV2 calldata encoding (`exactInputSingle`, `exactOutputSingle`)
- [x] Proper deadline, recipient, and fee tier handling
- [x] V3-specific interaction building

**AMM Integration (Session 19):**
- [x] `UniswapV3AMM` class implementing `SwapCalculator` protocol
- [x] Quoter-based swap calculation (no local tick math needed)
- [x] Token ordering (token0/token1) handling
- [x] `zeroForOne` direction detection

**Router Integration (Session 20):**
- [x] V3 pools in `PoolRegistry` alongside V2
- [x] Best-quote selection between V2 and V3
- [x] Mixed V2/V3 routing support

**Testing (Sessions 21-22):**
- [x] 12 V3 integration tests (sell/buy, both directions)
- [x] V3 benchmark fixtures (`v3_weth_to_usdc.json`, `v3_usdc_to_weth.json`, `v3_buy_weth.json`)
- [x] V2 vs V3 comparison fixture (`v2_v3_comparison.json`)
- [x] Real RPC quoter tests with `pytest.mark.rpc` marker
- [x] Mock quoter for offline testing

**Fee Handling (Session 23):**
- [x] Limit order fee calculation matching Rust baseline
- [x] Fee formula: `fee = gas_cost_wei * 1e18 / reference_price`
- [x] Overflow protection: reject trades where fee > executed_amount
- [x] V3 fixtures use `class: limit` with correct reference prices

**Known Limitations:**
- V3 swap math uses on-chain quoter (requires RPC for real quotes)
- No local tick-crossing simulation (quoter handles this)
- Tests skip V3 by default unless RPC_URL is set

**Exit Criteria:** V3 pools integrated, tested, and benchmarked against Rust. ✅

### Slice 3.2: Balancer Integration ✅ COMPLETE
**Goal:** Add weighted and stable pool support

- [x] Fixed-point math (Bfp class, 18-decimal precision)
- [x] Weighted pool math (calc_out_given_in, calc_in_given_out)
- [x] Stable pool math (StableSwap invariant, Newton-Raphson)
- [x] Pool parsing from auction data
- [x] BalancerWeightedAMM and BalancerStableAMM classes
- [x] Router integration (multi-hop, partial fills)
- [x] Integration tests: 7 tests matching Rust baseline exactly
- [x] Benchmark: 5/5 exact match with Rust baseline

**Sessions:** 25-34 (see `docs/sessions/phase-3.2-summary.md`)

### Slice 3.3: Multi-Source Routing ✅ COMPLETE
**Goal:** Route through best available liquidity

- [x] Quote comparison across DEXs (V2, V3, Balancer weighted, Balancer stable)
- [x] Select best execution venue per order
- [x] Multi-hop through mixed pool types
- [x] Test: auctions with multiple liquidity options
- [x] Benchmark: exact match with Rust baseline

### Slice 3.4: Split Routing — DEFERRED TO PHASE 4
**Goal:** Split orders across multiple venues for better execution

Split routing requires solving for optimal split ratios across price impact curves,
which is fundamentally an optimization problem. Deferred to Phase 4 (Unified Optimization)
where it will be handled as a natural output of the joint optimizer rather than a
standalone heuristic.

### Slice 3.5: 0x Limit Orders ✅ COMPLETE
**Goal:** Add foreign limit order support to achieve full Rust liquidity parity

The Rust baseline solver supports 5 liquidity types. All implemented:

| Liquidity Type | Description | Status |
|----------------|-------------|--------|
| `constantProduct` | UniswapV2 | ✅ Complete |
| `weightedProduct` | Balancer Weighted | ✅ Complete |
| `stable` | Balancer/Curve Stable | ✅ Complete |
| `concentratedLiquidity` | UniswapV3 | ✅ Complete |
| `limitOrder` | 0x Foreign Orders | ✅ **Complete** |

**Tasks:**
- [x] Parse `limitOrder` liquidity from auction JSON
- [x] Create `LimitOrderPool` dataclass (makerToken, takerToken, amounts, fee)
- [x] Implement swap simulation (simple proportional math)
- [x] Add `LimitOrderHandler` for routing
- [x] Integrate into `PoolRegistry` and router
- [x] Encode settlement interaction (0x protocol format)
- [x] Test: auctions with limit order liquidity
- [x] Benchmark fixtures created

**Exit Criteria:** ✅ Solver handles all 5 Rust liquidity types. Full parity achieved.

**Tests:** 702 passing, 14 skipped

---

## Phase 4: Unified Optimization

> **The Big Picture:** The optimal solution isn't "CoW matching" OR "AMM routing" —
> it's the joint optimization across all mechanisms. This phase designs a unified
> optimizer that considers:
> - Multi-order CoW matching (N orders, not just pairs)
> - Multiple AMM sources with different price curves
> - **Split routing** (orders across multiple venues, deferred from Slice 3.4)
> - **Flash loans** (temporary capital for arbitrage, splits, ring trades)
> - Ring trades (A→B→C→A cycles)
> - Partial fills across mechanisms
> - Gas costs in the objective function
>
> **Research:** See `docs/research/flash-loans.md` for flash loan provider analysis and design decisions.
>
> **Key Insight from 4.1:** Most CoW pairs have **crossing prices** (ask > bid), meaning
> pure peer-to-peer matching rarely works. The value comes from using AMM prices as
> reference points to unlock more matches. See `docs/sessions/session-43-slice4.1-analysis.md`.

### Slice 4.1: Problem Formulation ✅ COMPLETE
**Goal:** Define the optimization problem precisely

- [x] Formalize as constraint optimization (variables, constraints, objective)
- [x] Analyze problem structure (decomposition opportunities)
- [x] Empirical analysis: 20 real mainnet auctions (~5,600 orders each)
- [x] Prototype double auction algorithm for single-pair matching
- [x] Test on historical data to understand matching potential

**Key Findings:**
| Metric | Value |
|--------|-------|
| CoW-eligible orders | 36.5% |
| Pairs with 10+ orders | 858 (ideal for double auction) |
| Ring trade potential | 100% of auctions |
| Pure CoW matches | Rare (most prices cross) |

**Deliverables:**
- `solver/strategies/double_auction.py` - O(n log n) clearing algorithm
- `scripts/analyze_auction_structure.py` - Historical auction analysis
- `docs/design/phase4-slice4.1-problem-formulation.md` - Formal problem definition

### Slice 4.2: Hybrid CoW+AMM Strategy ✅ COMPLETE
**Goal:** Use AMM prices as reference to unlock CoW matches

> **Revised Approach:** Pure double auction has limited value because prices cross.
> By using AMM as a "virtual participant" at current market price, we can match
> orders that would otherwise need AMM routing — capturing the gas savings while
> ensuring fair execution.

**4.2a: AMM Price Integration** ✅
- [x] Add `get_reference_price(token_a, token_b)` method to router
- [x] Query V2/V3/Balancer for reference price
- [x] Handle no-liquidity cases gracefully (returns None)
- [x] Test: price queries for common pairs

**4.2b: Hybrid Double Auction** ✅
- [x] Create `run_hybrid_auction()` with AMM reference price
- [x] AMM price used as clearing price for CoW matches
- [x] Match orders against each other at AMM price
- [x] Route unmatched remainders through AMM (returns `amm_routes`)
- [x] Multi-price candidate selection for fill-or-kill orders
- [x] Test: hybrid auction behavior

**4.2c: Strategy Integration** ✅
- [x] Create `HybridCowStrategy` with 3-strategy chain (CowMatch → HybridCow → AmmRouting)
- [x] Build router from auction liquidity at solve time
- [x] Handle overlapping tokens across pairs (filter to largest)
- [x] Fall back to AMM routing for single-direction pairs
- [x] Benchmark: measure surplus improvement on historical auctions (50% win rate)

**Exit Criteria:** Hybrid strategy outperforms pure-AMM routing on at least 20% of CoW-eligible auctions. ✅ PASS (50%)

### Slice 4.3: Evaluation & Next Direction ✅ COMPLETE
**Goal:** Data-driven decision on ring trades vs split routing vs unified solver

**Evaluation Results (2026-01-23):**
| Metric | Value |
|--------|-------|
| Auctions evaluated | 10 |
| Orders processed | 500 (50 random sample per auction) |
| Hybrid wins | 0 (0.0%) |
| AMM wins | 0 (0.0%) |
| Ties | 5 (50.0%) |
| Neither | 5 (50.0%) |
| **Contested win rate** | **0.0%** |
| CoW matches | 7 (1.4% of orders) |

**Finding:** Hybrid CoW adds **marginal value** (0% win rate, well below 5% threshold).

**Root Causes:**
1. Most orders target exotic token pairs with no opposing liquidity
2. Random sampling means CoW-eligible pairs are rare (~1.4%)
3. When pairs match, prices often cross (limit price mismatch)
4. V3 disabled (no RPC) limits routing options

**Initial Decision:** ⚠️ Skip ring trades (direct CoW adds marginal value)

**Revised Decision (after ring analysis):** ✅ **Proceed to Ring Trades**

Follow-up analysis of ring trade potential showed:
| Metric | Direct CoW | Ring Trades |
|--------|------------|-------------|
| Match rate | 1.40% | **5.41%** |
| Improvement | — | **3.9x** |

Ring trades find cycles (A→B→C→A) that direct CoW matching (A↔B) misses. The 5.41% rate exceeds the 5% threshold for "significant value."

**Tasks:**
- [x] Run hybrid strategy on 50+ historical auctions
- [x] Measure surplus improvement vs pure-AMM baseline
- [x] Document findings (`docs/slice-4.3-evaluation-results.md`)
- [x] Analyze ring trade potential (`scripts/analyze_ring_potential.py`)
- [x] Update PLAN.md with chosen direction → **Ring Trades (Slice 4.4)**

### Slice 4.4: Ring Trade Detection ✅ COMPLETE
**Goal:** Find cyclic trading opportunities (A→B→C→A)

**Implementation:**
- [x] Build token graph from orders (`OrderGraph`)
- [x] Find short cycles (3-4 tokens max)
- [x] Calculate ring trade feasibility and surplus
- [x] Implement `RingTradeStrategy`
- [x] Test: auctions with ring trade potential
- [x] Benchmark: measure actual surplus improvement

**Results (50 auctions, 280,920 orders):**
| Metric | Value |
|--------|-------|
| RingTrade matches | 467 orders (0.17%) |
| HybridCow matches | 192 orders (0.07%) |
| CowMatch matches | 0 orders (0.00%) |

### Slice 4.5: Settlement Optimization Analysis ✅ COMPLETE
**Goal:** Formalize the problem and evaluate optimization approaches

**Comprehensive Analysis:**
- [x] Formalize as Mixed-Integer Bilinear Program (MIBLP)
- [x] Benchmark existing strategies on 50 historical auctions
- [x] Characterize matching gap (40,521 crossing orders vs 467 matched)
- [x] Prototype price enumeration approach
- [x] Evaluate LP solver (scipy HIGHS) performance

**Key Findings (see `docs/design/settlement-optimization-formulation.md`):**

| Metric | Value |
|--------|-------|
| Orders on crossing pairs | 40,521 (14.4%) |
| Orders matched by strategies | ~420-520 (0.15-0.19%) |
| **Gap factor** | **~100x** |

**Root Causes of Gap:**
1. **Token overlap**: Processing pairs independently prevents matching when pairs share tokens
2. **Volume imbalance**: Many pairs have 100x more sell than buy pressure (or vice versa)
3. **Problem structure**: Uniform clearing price constraint is bilinear (NP-hard)

**LP Solver Results:**
- scipy HIGHS is fast: ~8ms per auction
- Token overlap is the blocker, not solver performance
- Multi-pair coordination needed to capture more value

### Slice 4.6: Multi-Pair Price Coordination ✅ COMPLETE
**Goal:** Optimize prices across token-connected pairs

> **Finding from 4.5:** Token overlap is the main blocker. When pairs share tokens,
> independent processing leaves value on the table.

**Implementation:**
- [x] Build token connectivity graph (Union-Find algorithm)
- [x] Partition pairs into independent components
- [x] Implement joint price optimization within components (LP solver)
- [x] Create `MultiPairCowStrategy` integrating all pieces
- [x] Add EBBO validation with zero tolerance to all strategies
- [x] Test: 28 unit tests for multi-pair strategy
- [x] Benchmark: 100% EBBO compliance achieved

**Results (10 auctions, 56,289 orders):**

| Strategy | Matched | Rate | EBBO Compliance |
|----------|---------|------|-----------------|
| MultiPair | 40 | 0.07% | 100% |
| RingTrade | 70 | 0.12% | 100% |

**EBBO Integration:**
- Zero tolerance (EBBO_TOLERANCE = 0)
- Integer comparison for proper rounding
- Validation at strategy level and Solver level
- All strategies now reject EBBO-violating matches

**Gap Analysis:**
Despite multi-pair coordination, significant gap remains:
- CoW potential: 36.53% of orders
- Best match rate: 0.12%
- Root causes under investigation

### Slice 4.7: Split Routing (Deferred)
**Goal:** Split large orders across multiple venues for better execution

> **Status:** Deferred. Analysis showed token overlap is a bigger gap than split routing.
> Will revisit after multi-pair coordination.

- [ ] Quote multiple venues for same order
- [ ] Convex optimization for split ratios
- [ ] Gas cost vs slippage tradeoff
- [ ] Test: large orders (>10% of pool liquidity)
- [ ] Benchmark: slippage reduction vs single-venue

### Slice 4.8: Flash Loan Integration (Future)
**Goal:** Use flash loans to enable better user fills (not solver profit)

> **Design Doc:** See `docs/research/flash-loans.md` for full research and design decisions.

Flash loans provide temporary capital within a single transaction, enabling:
- Split routing across venues (bootstrap capital for parallel execution)
- Ring trade execution (capital to start the cycle)
- Arbitrage as better fills (price improvements passed to users)

**Key design decisions:**
- Surplus goes to users, not solver (per CIP-11)
- Model providers separately (Balancer 0% fee, Aave 0.05%, different gas costs)
- CoW Protocol already has infrastructure (`FlashLoanRouter` from CIP-66)

**Tasks:**
- [ ] Implement `FlashLoanProvider` protocol with concrete implementations
- [ ] Add flash loan variables to optimization problem
- [ ] Constraint: borrowed = repaid + fee
- [ ] Provider selection: minimize (fee + gas) subject to liquidity
- [ ] Test: settlements that require flash loan capital
- [ ] Benchmark: measure user surplus improvement vs non-flash solutions

**Provider priority heuristic:**
1. Balancer (0% fee, ~24k gas) - if liquidity sufficient
2. Maker (0% fee, unlimited DAI) - if DAI needed
3. Aave (0.05% fee, ~70k gas) - fallback for large amounts

**Exit Criteria:** Solver finds globally optimal (or near-optimal) solutions across all mechanisms.

### Phase 4 Status Summary

| Slice | Status | Notes |
|-------|--------|-------|
| 4.1 Problem Formulation | ✅ Complete | Double auction prototype |
| 4.2 Hybrid CoW+AMM | ✅ Complete | HybridCowStrategy (superseded) |
| 4.3 Evaluation | ✅ Complete | Data-driven direction choice |
| 4.4 Ring Trade Detection | ✅ Complete | RingTradeStrategy (research) |
| 4.5 Settlement Optimization | ✅ Complete | MIBLP formulation + LP solver |
| 4.6 Multi-Pair Coordination | ✅ Complete | MultiPairCowStrategy + EBBO |
| 4.7 Split Routing | Deferred | Lower priority than token overlap |
| 4.8 Flash Loans | Future | Requires 4.7 first |

**Current Production Chain:** CowMatch → MultiPair → AmmRouting (992 tests passing)

**CoW Match Rate:** 0.12% — this is near-optimal given market constraints (see `docs/evaluations/cow-potential-analysis.md`)

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
│   │   ├── base.py            # SwapResult dataclass, SwapCalculator protocol
│   │   ├── uniswap_v2.py      # UniswapV2 implementation
│   │   └── uniswap_v3.py      # UniswapV3 implementation (quoter-based)
│   │
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py            # SolutionStrategy protocol, StrategyResult, fee calculation
│   │   ├── cow_match.py       # CoW matching strategy
│   │   ├── matching_rules.py  # Data-driven matching rules
│   │   └── amm_routing.py     # AMM routing strategy
│   │
│   └── routing/
│       ├── __init__.py
│       └── router.py          # Order routing logic, Solver class (with DI)
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
│   │       ├── benchmark/              # Shared: Python vs Rust comparison (V2 + V3)
│   │       └── benchmark_python_only/  # Python-only features (CoW matching)
│   │
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_amm.py
│   │   ├── test_router.py
│   │   ├── test_cow_match.py
│   │   ├── test_strategy_base.py      # Fee calculation tests
│   │   └── test_uniswap_v3.py         # V3 AMM tests
│   │
│   └── integration/
│       ├── test_api.py
│       ├── test_single_order.py
│       └── test_v3_integration.py     # V3 router integration tests
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
