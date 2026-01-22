# CoW Solver Python - AI Assistant Context

This file provides context for AI assistants working on this project. Read this first.

## Project Overview

**What:** A Python implementation of a CoW Protocol solver, built as a learning project to explore AI-assisted development and benchmark Python vs Rust performance.

**Why:**
1. Learn the CoW Protocol solver problem domain
2. Demonstrate AI-assisted development workflow
3. Create portfolio material with quantified performance comparisons
4. Explore where Python can compete with Rust and where it can't

**Status:** Phase 1 complete. Phase 2 complete (2-order CoW matching). Phase 3 Slice 3.1 complete (UniswapV3).

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
├── BENCHMARKS.md          # Benchmarking guide
├── docs/sessions/         # Session handoff logs (one file per session)
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
│   │   ├── base.py        # SwapResult dataclass, SwapCalculator protocol
│   │   ├── uniswap_v2.py  # UniswapV2 implementation
│   │   └── uniswap_v3.py  # UniswapV3 implementation (quoter-based)
│   ├── strategies/        # Solution strategies
│   │   ├── base.py        # SolutionStrategy protocol, fee calculation
│   │   ├── cow_match.py   # CoW matching strategy
│   │   ├── matching_rules.py # Data-driven matching rules
│   │   └── amm_routing.py # AMM routing strategy
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
        ├── benchmark/     # Shared benchmark fixtures (Python vs Rust)
        └── benchmark_python_only/  # Python-only benchmarks
```

## Current State

### What's Done (Phase 0 + Phase 1 + Phase 2 + Phase 3 Slice 3.1)
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
- ✅ **Strategy pattern** for solution finding (SolutionStrategy protocol)
- ✅ **CoW matching** (2-order peer-to-peer settlement, all order type combinations)
- ✅ **Composable strategies** (StrategyResult, OrderFill for partial matching)
- ✅ **Partial CoW + AMM** (partial CoW match with AMM remainder routing)
- ✅ **AMM partial fills** (exact calculation outperforms Rust's binary search)
- ✅ **Data-driven matching rules** (matching_rules.py for auditability)
- ✅ **UniswapV3 integration** (quoter-based swap calculation)
- ✅ **V3 Quoter interface** (MockV3Quoter for tests, Web3V3Quoter for RPC)
- ✅ **V3 settlement encoding** (SwapRouterV2 calldata)
- ✅ **Best-quote selection** (V2 vs V3 comparison)
- ✅ **Limit order fee calculation** (matching Rust baseline behavior)

**Total: 288 passing tests** (unit + integration)

### Rust Baseline Solver Limitations

The Rust "baseline" solver from [cowprotocol/services](https://github.com/cowprotocol/services) is a **single-order AMM router only**:
- ✅ Routes individual orders through AMM liquidity
- ✅ Supports multi-hop paths
- ❌ Does NOT support CoW matching
- ❌ Does NOT optimize across multiple orders

See `BENCHMARKS.md` for details. Benchmarks are split into:
- `benchmark/` - Shared functionality (Python vs Rust comparison)
- `benchmark_python_only/` - Python-only features (CoW matching)

### What's Next (Phase 3: Liquidity Expansion)
See `PLAN.md` for full details.

> **Why Liquidity First:** Multi-order CoW detection has been deferred to Phase 4.
> The optimal solution requires joint optimization across CoW + all AMM sources.
> We're adding liquidity sources first to understand the full problem space before
> designing the unified optimizer.

**Slice 3.1: UniswapV3 Integration** ✅ COMPLETE
- [x] V3 pool parsing and quoter interface
- [x] Best-quote selection between V2 and V3
- [x] Limit order fee calculation

**Slice 3.2: Balancer/Curve Integration** ⬅️ NEXT
- [ ] Weighted pool math (Balancer)
- [ ] Stable pool math (Curve/Balancer)

## Key Files to Know

| File | Purpose |
|------|---------|
| `solver/api/endpoints.py` | The `/solve` endpoint with DI support |
| `solver/models/auction.py` | Input data structures (Order, Token, AuctionInstance) |
| `solver/models/solution.py` | Output data structures (Solution, Trade, Interaction) |
| `solver/amm/uniswap_v2.py` | UniswapV2 AMM math and encoding |
| `solver/amm/uniswap_v3.py` | UniswapV3 AMM with quoter-based swap calculation |
| `solver/strategies/base.py` | SolutionStrategy protocol, StrategyResult, fee calculation |
| `solver/strategies/matching_rules.py` | Data-driven matching rules (constraint tables) |
| `solver/strategies/cow_match.py` | CoW matching (perfect + partial) |
| `solver/strategies/amm_routing.py` | AMM routing strategy |
| `solver/routing/router.py` | Order routing, Solver (composes strategies) |
| `solver/constants.py` | Centralized addresses and constants |
| `tests/conftest.py` | Mock fixtures for DI testing (MockAMM, MockV3Quoter) |
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

# Inject custom strategies into solver
from solver.strategies import CowMatchStrategy, AmmRoutingStrategy
solver = Solver(strategies=[CowMatchStrategy(), AmmRoutingStrategy()])

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

1. **Read `docs/sessions/`** to see what was done last (check README.md for index)
2. **Check `PLAN.md`** to find the current slice
3. **Follow the slice workflow:**
   - Write failing tests first
   - Implement minimum code to pass
   - Run benchmarks if applicable
   - Create new session file in `docs/sessions/`

### Slice Workflow
```
1. SELECT target auction type from PLAN.md
2. WRITE failing tests using fixtures
3. IMPLEMENT minimum code to pass
4. BENCHMARK against Rust (if applicable - some features are Python-only)
5. DOCUMENT results in docs/sessions/session-NN.md
```

## Important Context

### CoW Solution Format
A valid solution needs:
- `prices`: Uniform clearing prices for all tokens involved
- `trades`: Which orders are being filled and by how much
- `interactions`: On-chain calls (AMM swaps) to execute

### UniswapV2 Math
```
output = (input * 997 * reserve_out) / (reserve_in * 1000 + input * 997)
```
- 0.3% fee (997/1000)
- Constant product: `reserve_in * reserve_out = k`

### CoW Matching
For a valid CoW match:
- Order A sells token X, wants token Y
- Order B sells token Y, wants token X
- Both orders' limit prices must be satisfied
- Solution has 2 trades, 0 interactions, gas=0

### Scoring
Solutions are scored by surplus generated for users. Higher is better.
The driver validates solutions and rejects invalid ones.

### Limit Order Fees
For limit orders (`class: limit`), the solver MUST calculate a fee:
```
fee = gas_cost_wei * 1e18 / reference_price
```
- Market orders (`class: market`): No solver fee needed
- Limit orders: Fee is mandatory, deducted from executed amount
- If fee > executed_amount, trade is rejected (overflow protection)

### UniswapV3
V3 uses concentrated liquidity (different from V2's constant product):
- Swap calculation via on-chain QuoterV2 contract (requires RPC)
- Settlement via SwapRouterV2 (`exactInputSingle`/`exactOutputSingle`)
- Tests use MockV3Quoter for offline testing
- Real V3 tests require `RPC_URL` env var and `pytest -m rpc`

## Session Handoff

After each session, create a new file in `docs/sessions/` (e.g., `session-10.md`) with:
- What was completed
- Test results
- Benchmark results (if any)
- What's next
- Open questions

Then update `docs/sessions/README.md` with the new session entry.

See `docs/sessions/session-template.md` for the format.

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

# Benchmarking (shared functionality)
python scripts/run_benchmarks.py --python-url http://localhost:8000 --rust-url http://localhost:8080

# Benchmarking (Python-only features)
python scripts/run_benchmarks.py --python-url http://localhost:8000 \
    --auctions tests/fixtures/auctions/benchmark_python_only
```

## Notes for AI Assistants

1. **Always create a new session file in `docs/sessions/`** after making progress
2. **Write tests first** — this is test-driven development
3. **Keep changes minimal** — each slice should be small and focused
4. **Run tests after changes** — verify nothing broke
5. **Check PLAN.md** for what's next if unsure
6. **Use DI for testing** — inject mocks via `tests/conftest.py` fixtures
7. **Run benchmarks** — see `BENCHMARKS.md` for setup (Rust solver at `/Users/telmo/project/cow-services`)
8. **Know Rust limitations** — Some features (CoW matching) are Python-only; can't compare with Rust baseline
