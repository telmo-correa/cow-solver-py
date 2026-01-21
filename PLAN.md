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

### Slice 1.4: Multi-hop Routing (A→B→C)
**Goal:** Route through intermediate tokens when direct pool doesn't exist

- [ ] Token graph construction
- [ ] Shortest path (by output amount, not hops)
- [ ] Chain multiple interactions
- [ ] Test: auctions requiring 2-hop routes
- [ ] Benchmark

**Exit Criteria:** Solver handles any single-order auction with UniV2 liquidity.

---

## Phase 2: Coincidence of Wants (CoW)

### Slice 2.1: Perfect CoW Match
**Goal:** Two orders that exactly offset (A sells X for Y, B sells Y for X)

- [ ] Order pair detection (opposite directions, same tokens)
- [ ] Direct settlement without AMM
- [ ] Uniform clearing price calculation
- [ ] Test: synthetic perfect-match auctions
- [ ] Benchmark: should beat Rust on these (no AMM overhead)

### Slice 2.2: Partial CoW + AMM Remainder
**Goal:** Match what we can peer-to-peer, route remainder through AMM

- [ ] Calculate matchable volume
- [ ] Split order execution (CoW portion + AMM portion)
- [ ] Combined solution with mixed trades
- [ ] Test: auctions with partial overlap
- [ ] Benchmark

### Slice 2.3: Multi-Order CoW Detection
**Goal:** Find CoW opportunities across N orders

- [ ] Build order flow graph (net demand per token pair)
- [ ] Identify netting opportunities
- [ ] Greedy matching algorithm
- [ ] Test: 5+ order auctions with CoW potential
- [ ] Benchmark

**Exit Criteria:** Solver finds and exploits CoW opportunities.

---

## Phase 3: Liquidity Expansion

### Slice 3.1: UniswapV3 Integration
- [ ] Concentrated liquidity math
- [ ] Tick-based price calculation
- [ ] Pool state fetching
- [ ] Test: auctions where V3 beats V2

### Slice 3.2: Balancer Integration
- [ ] Weighted pool math
- [ ] Stable pool math
- [ ] Test: auctions with Balancer liquidity

### Slice 3.3: Multi-Source Routing
- [ ] Compare quotes across DEXs
- [ ] Select best execution venue
- [ ] Split orders across venues
- [ ] Test: complex auctions
- [ ] Benchmark: this is where Rust likely pulls ahead

**Exit Criteria:** Solver uses multiple liquidity sources intelligently.

---

## Phase 4: Optimization & Cython

### Slice 4.1: Profiling
- [ ] Profile on 100 historical auctions
- [ ] Identify top 5 hotspots
- [ ] Document findings

### Slice 4.2: Cython Hot Paths
- [ ] Convert identified hotspots to Cython
- [ ] Benchmark before/after
- [ ] Document speedup

### Slice 4.3: Algorithmic Improvements
- [ ] Parallel solution evaluation
- [ ] Caching for repeated calculations
- [ ] Early termination heuristics

**Exit Criteria:** Documented optimization journey with metrics.

---

## Phase 5: Production Readiness (Optional)

### Slice 5.1: Error Handling & Resilience
- [ ] Timeout handling
- [ ] Malformed input handling
- [ ] Graceful degradation

### Slice 5.2: Observability
- [ ] Structured logging
- [ ] Metrics (solution time, score distribution)
- [ ] Health endpoint

### Slice 5.3: Shadow Mode Testing
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
├── SESSIONS.md                # Session handoff log
├── BENCHMARKS.md              # Benchmarking guide
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
│   │       ├── multi_hop/
│   │       └── benchmark/
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
└── cython_modules/            # Added in Phase 4
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

### Optimization (Phase 4)
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
