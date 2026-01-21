# CoW Solver Python - AI Assistant Context

This file provides context for AI assistants working on this project. Read this first.

## Project Overview

**What:** A Python implementation of a CoW Protocol solver, built as a learning project to explore AI-assisted development and benchmark Python vs Rust performance.

**Why:**
1. Learn the CoW Protocol solver problem domain
2. Demonstrate AI-assisted development workflow
3. Create portfolio material with quantified performance comparisons
4. Explore where Python can compete with Rust and where it can't

**Status:** Phase 1 (Single Order via DEX) complete. Ready for Phase 2 (Coincidence of Wants).

## What is CoW Protocol?

CoW Protocol is a decentralized exchange that uses batch auctions. Key concepts:

- **Solvers** compete to find optimal trade settlements for batches of orders
- **Coincidence of Wants (CoW):** When traders want opposite sides of a trade, they can be matched directly without AMM fees
- **Batch Auction:** Orders are collected, solvers propose solutions, best solution wins
- **Driver:** Intermediary that sends auctions to solvers and validates responses

### Solver API Contract

Solvers expose a `POST /{environment}/{network}` endpoint that:
- Receives: `AuctionInstance` (orders, tokens, liquidity, deadline)
- Returns: `SolverResponse` (list of solutions with prices, trades, interactions)

## Project Structure

```
cow-solver-py/
├── CLAUDE.md              # THIS FILE - read first
├── PLAN.md                # Detailed implementation plan with slices
├── SESSIONS.md            # Session handoff log
├── BENCHMARKS.md          # Benchmarking guide
├── pyproject.toml         # Dependencies and config
│
├── solver/                # Main package
│   ├── api/               # FastAPI app
│   │   ├── main.py        # App entry point
│   │   └── endpoints.py   # POST /solve endpoint with DI support
│   ├── models/            # Pydantic schemas
│   │   ├── auction.py     # AuctionInstance, Order, Token
│   │   └── solution.py    # Solution, Trade, Interaction
│   ├── amm/               # AMM math
│   │   ├── base.py        # SwapResult dataclass
│   │   └── uniswap_v2.py  # UniswapV2 implementation
│   ├── routing/           # Order routing
│   │   └── router.py      # SingleOrderRouter, Solver (with DI)
│   └── constants.py       # Centralized constants (addresses, etc.)
│
├── benchmarks/            # Performance comparison
│   ├── harness.py         # Main runner
│   ├── rust_runner.py     # Subprocess wrapper for Rust solver
│   ├── metrics.py         # Statistics
│   └── report.py          # Output formatting
│
├── scripts/
│   ├── collect_auctions.py  # Fetch historical auctions from CoW API
│   └── run_benchmarks.py    # HTTP benchmark runner
│
└── tests/
    ├── conftest.py        # Fixtures + mock classes for DI testing
    ├── unit/              # Unit tests
    ├── integration/       # Integration tests
    └── fixtures/auctions/ # JSON test data
        ├── single_order/  # Single order fixtures
        ├── cow_pairs/     # CoW pair fixtures
        └── benchmark/     # Benchmark fixtures
```

## Current State

### What's Done (Phase 0 + Phase 1 Complete)
- ✅ Project skeleton with pyproject.toml
- ✅ Pydantic models matching CoW OpenAPI spec
- ✅ FastAPI endpoint that accepts auctions
- ✅ Benchmark harness for Python vs Rust comparison
- ✅ Auction collector script
- ✅ Sample fixture auctions
- ✅ **UniswapV2 AMM math** (constant product formula)
- ✅ **Single order router** (routes sell/buy orders through UniV2)
- ✅ **Solution builder** (trades, interactions, clearing prices)
- ✅ **Buy order support** (exact output swaps)
- ✅ **Multi-hop routing** (A→B→C via BFS pathfinding)
- ✅ **PoolRegistry** for dynamic pool management from auction liquidity
- ✅ **Dependency injection** for testability (router, solver, AMM)
- ✅ **Mock fixtures** for isolated testing (MockAMM, MockPoolFinder, MockRouter)
- ✅ Centralized constants (`solver/constants.py`)
- ✅ Server management script (`scripts/servers.sh`)

**Total: 85 passing tests** (unit + integration)

### What's Next (Phase 2: Coincidence of Wants)
See `PLAN.md` for full details. Next slices:

**Slice 2.1: Perfect CoW Match**
- [ ] Detect two orders that exactly offset (A sells X for Y, B sells Y for X)
- [ ] Direct settlement without AMM
- [ ] Uniform clearing price calculation

**Slice 2.2: Partial CoW + AMM Remainder**
- [ ] Match what we can peer-to-peer
- [ ] Route remainder through AMM

## Key Files to Know

| File | Purpose |
|------|---------|
| `solver/api/endpoints.py` | The `/solve` endpoint with DI support |
| `solver/models/auction.py` | Input data structures |
| `solver/models/solution.py` | Output data structures |
| `solver/amm/uniswap_v2.py` | UniswapV2 AMM math and encoding |
| `solver/routing/router.py` | Order routing and solution building (with DI) |
| `solver/constants.py` | Centralized addresses and constants |
| `tests/conftest.py` | Mock fixtures for DI testing |
| `benchmarks/harness.py` | Run both solvers and compare |
| `PLAN.md` | Detailed slice breakdown |
| `BENCHMARKS.md` | How to run benchmarks |

## Dependency Injection for Testing

The solver supports dependency injection for isolated testing:

```python
# Inject mock AMM and pool finder into router
from tests.conftest import MockAMM, MockPoolFinder, MockSwapConfig

mock_amm = MockAMM(MockSwapConfig(fixed_output=3000_000_000))
mock_finder = MockPoolFinder(default_pool=my_pool)
router = SingleOrderRouter(amm=mock_amm, pool_finder=mock_finder)

# Inject mock router into solver
solver = Solver(router=mock_router)

# Override FastAPI dependency for API tests
from solver.api.endpoints import get_solver
app.dependency_overrides[get_solver] = lambda: my_mock_solver
```

Available mock classes in `tests/conftest.py`:
- `MockAMM` - Configurable swap outputs with call tracking
- `MockPoolFinder` - Returns configured pools for token pairs
- `MockRouter` - Returns configured routing results
- `MockSwapConfig` - Configure fixed output, multiplier, or default rate

## Development Workflow

### Setup (if not done)
```bash
cd /Users/telmo/project/cow
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run Tests
```bash
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests (when they exist)
```

### Run Solver
```bash
python -m solver.api.main       # Starts on localhost:8000
```

### Run Benchmarks
```bash
python -m benchmarks.harness    # Compare Python vs Rust
```

## How to Continue Development

When the user says "do the next step" or similar:

1. **Read `SESSIONS.md`** to see what was done last
2. **Check `PLAN.md`** to find the current slice
3. **Follow the slice workflow:**
   - Write failing tests first
   - Implement minimum code to pass
   - Run benchmarks if applicable
   - Update `SESSIONS.md` with results

### Slice Workflow
```
1. SELECT target auction type from PLAN.md
2. WRITE failing tests using fixtures
3. IMPLEMENT minimum code to pass
4. BENCHMARK against Rust (if configured)
5. DOCUMENT results in SESSIONS.md
```

## Important Context

### CoW Solution Format
A valid solution needs:
- `prices`: Uniform clearing prices for all tokens involved
- `trades`: Which orders are being filled and by how much
- `interactions`: On-chain calls (AMM swaps) to execute

### UniswapV2 Math (for Slice 1.2)
```
output = (input * 997 * reserve_out) / (reserve_in * 1000 + input * 997)
```
- 0.3% fee (997/1000)
- Constant product: `reserve_in * reserve_out = k`

### Scoring
Solutions are scored by surplus generated for users. Higher is better.
The driver validates solutions and rejects invalid ones.

## Session Handoff

After each session, update `SESSIONS.md` with:
- What was completed
- Test results
- Benchmark results (if any)
- What's next
- Open questions

This ensures the next session (human or AI) can pick up seamlessly.

## Commands Reference

```bash
# Development
pip install -e ".[dev]"         # Install with dev deps
pytest                          # Run all tests
pytest -v tests/unit/           # Run unit tests verbose
ruff check .                    # Lint
mypy solver/                    # Type check

# Running
python -m solver.api.main       # Start API server
python -m scripts.collect_auctions --count 50  # Fetch auctions

# Benchmarking
python -m benchmarks.harness    # Run comparison
```

## Notes for AI Assistants

1. **Always update SESSIONS.md** after making progress
2. **Write tests first** — this is test-driven development
3. **Keep changes minimal** — each slice should be small and focused
4. **Run tests after changes** — verify nothing broke
5. **Check PLAN.md** for what's next if unsure
6. **Use DI for testing** — inject mocks via `tests/conftest.py` fixtures
7. **Run benchmarks** — see `BENCHMARKS.md` for setup (Rust solver at `/Users/telmo/project/cow-services`)
