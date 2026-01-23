# CoW Solver Python - AI Assistant Context

This file provides context for AI assistants working on this project. Read this first.

## Project Overview

**What:** A Python implementation of a CoW Protocol solver, built as a learning project to explore AI-assisted development and benchmark Python vs Rust performance.

**Status:** Phase 1-3 complete. Full liquidity parity with Rust baseline (V2, V3, Balancer weighted/stable, 0x limit orders). 750 tests passing.

## What is CoW Protocol?

CoW Protocol is a decentralized exchange that uses batch auctions:

- **Solvers** compete to find optimal trade settlements for batches of orders
- **Coincidence of Wants (CoW):** Direct peer-to-peer matching without AMM fees
- **Batch Auction:** Orders collected, solvers propose solutions, best solution wins
- **Driver:** Intermediary that sends auctions to solvers and validates responses

### Solver API Contract

Solvers expose `POST /{environment}/{network}`:
- **Receives:** `AuctionInstance` (orders, tokens, liquidity, deadline)
- **Returns:** `SolverResponse` (list of solutions with prices, trades, interactions)

## Project Structure

```
solver/                # Main package
├── api/               # FastAPI app (main.py, endpoints.py)
├── models/            # Pydantic schemas (auction.py, solution.py)
├── amm/               # AMM math
│   ├── uniswap_v2.py  # V2 constant product
│   ├── uniswap_v3/    # V3 concentrated liquidity (6 modules)
│   ├── balancer/      # Weighted + stable pools (5 modules)
│   └── limit_order.py # 0x foreign limit orders
├── pools/             # Pool registry and types
├── strategies/        # CoW matching + AMM routing
├── routing/           # Order routing
│   ├── router.py      # SingleOrderRouter facade
│   ├── registry.py    # HandlerRegistry for dispatch
│   ├── handlers/      # Pool-specific handlers (v2, v3, balancer, limit_order)
│   └── multihop.py    # Multi-hop routing
├── math/              # Fixed-point arithmetic, SafeInt
└── constants.py       # Centralized addresses

tests/
├── unit/              # Unit tests by module
├── integration/       # Rust parity tests
└── fixtures/          # JSON test data
    ├── benchmark/     # Python vs Rust comparison
    └── benchmark_python_only/  # CoW matching (Python-only)

docs/sessions/         # Session handoff logs
```

## Current State

**Phase 3 Complete:** All Rust parity tests pass. Python matches or exceeds Rust on all benchmarks.

**Capabilities:**
- All 5 Rust liquidity types (V2, V3, Balancer weighted/stable, 0x limit orders)
- CoW matching (Python-only feature, not in Rust baseline)
- Partial fills with solver fee in limit price validation
- Per-pool gas estimates from auction data
- Handler registry pattern for extensible pool dispatch

**Rust Baseline Limitations:** The Rust solver is single-order AMM routing only. It does NOT support CoW matching or multi-order optimization.

**Next:** Phase 4 - Multi-order CoW detection and unified optimization.

## Key Files

| File | Purpose |
|------|---------|
| `solver/api/endpoints.py` | POST /solve endpoint |
| `solver/models/auction.py` | AuctionInstance, Order, Token |
| `solver/models/solution.py` | Solution, Trade, Interaction |
| `solver/amm/uniswap_v2.py` | V2 AMM math and encoding |
| `solver/amm/uniswap_v3/` | V3 package (quoter, encoding, etc.) |
| `solver/amm/balancer/` | Balancer package (weighted, stable) |
| `solver/amm/limit_order.py` | 0x limit order AMM |
| `solver/routing/router.py` | SingleOrderRouter (delegates to handlers) |
| `solver/routing/registry.py` | HandlerRegistry for pool dispatch |
| `solver/strategies/cow_match.py` | CoW matching strategy |
| `solver/strategies/amm_routing.py` | AMM routing strategy |
| `tests/conftest.py` | Mock fixtures (MockAMM, MockRouter, etc.) |
| `PLAN.md` | Implementation roadmap |
| `BENCHMARKS.md` | Benchmarking guide |

## Development

### Commands
```bash
# Setup
pip install -e ".[dev]"

# Test
pytest                          # All tests
pytest tests/unit/ -v           # Unit tests verbose
ruff check .                    # Lint
mypy solver/                    # Type check

# Run
python -m solver.api.main       # Start solver on :8000

# Benchmark (requires Rust solver on :8080)
python scripts/run_benchmarks.py --python-url http://localhost:8000 --rust-url http://localhost:8080
```

### Dependency Injection

The solver supports DI for isolated testing:
```python
from tests.conftest import MockAMM, MockPoolFinder
router = SingleOrderRouter(amm=MockAMM(...), pool_finder=MockPoolFinder(...))
solver = Solver(router=mock_router)
```

Mock classes: `MockAMM`, `MockPoolFinder`, `MockRouter`, `MockSwapConfig`

## Important Context

### Limit Order Fees
For `class: limit` orders, solvers MUST calculate a fee:
```
fee = gas_estimate * gas_price * 1e18 // reference_price
```
Market orders (`class: market`) have no solver fee.

### CoW Matching
Valid CoW match requires:
- Order A sells X for Y, Order B sells Y for X
- Both limit prices satisfied
- Result: 2 trades, 0 interactions, gas=0

### Solution Format
A valid solution needs:
- `prices`: Uniform clearing prices for all tokens
- `trades`: Order fills with executedAmount and fee
- `interactions`: On-chain calls (AMM swaps)

### UniswapV3
Requires RPC for QuoterV2 contract calls. Set `RPC_URL` env var. Tests use `MockV3Quoter` for offline testing.

## Session Workflow

After each session, create `docs/sessions/session-NN.md` with:
- What was completed
- Test results
- What's next

See `docs/sessions/session-template.md` for format. Update `docs/sessions/README.md` index.

## Notes for AI Assistants

1. **Write tests first** — TDD approach
2. **Keep changes minimal** — small, focused slices
3. **Run tests after changes** — verify nothing broke
4. **Check PLAN.md** for what's next
5. **Use DI for testing** — inject mocks via `tests/conftest.py`
6. **Create session files** — document progress in `docs/sessions/`
7. **Know Rust limitations** — CoW matching is Python-only
