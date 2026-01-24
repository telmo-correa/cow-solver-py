# CoW Solver Python - AI Assistant Context

This file provides context for AI assistants working on this project. Read this first.

> **CRITICAL - FINANCIAL APPLICATION:** This is a financial solver handling real money. NEVER use tolerance/epsilon comparisons for price or amount calculations. All financial comparisons MUST use exact integer arithmetic. Tolerance-based comparisons can lead to value extraction, incorrect settlements, or protocol violations. When integer truncation causes issues, fix the root cause (use ceiling/floor appropriately, adjust formulas) rather than adding tolerance.

> **IMPORTANT:** NEVER run `git push` without an explicit request from the user. Always wait for the user to ask you to push.

## Project Overview

**What:** A Python implementation of a CoW Protocol solver, built as a learning project to explore AI-assisted development and benchmark Python vs Rust performance.

**Status:** Phase 1-3 + Phase 4 (Slices 4.1-4.6) complete. Full liquidity parity with Rust baseline. 992 tests passing.

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
├── models/            # Pydantic schemas
│   ├── auction.py     # AuctionInstance, Order, Token
│   ├── solution.py    # Solution, Trade, Interaction
│   └── order_groups.py # OrderGroup for batch optimization
├── amm/               # AMM math
│   ├── uniswap_v2.py  # V2 constant product
│   ├── uniswap_v3/    # V3 concentrated liquidity (6 modules)
│   ├── balancer/      # Weighted + stable pools (7 modules)
│   └── limit_order.py # 0x foreign limit orders
├── pools/             # Pool registry and types
├── strategies/        # CoW matching + AMM routing (see Strategy Chain below)
├── routing/           # Order routing
│   ├── router.py      # SingleOrderRouter facade
│   ├── registry.py    # HandlerRegistry for dispatch
│   ├── pathfinding.py # TokenGraph and PathFinder
│   ├── handlers/      # Pool-specific handlers (v2, v3, balancer, limit_order)
│   └── multihop.py    # Multi-hop routing
├── fees/              # Fee calculation and price estimation
├── math/              # Fixed-point arithmetic (Bfp)
├── ebbo.py            # EBBO validation (zero tolerance)
├── safe_int.py        # Safe integer arithmetic
└── constants.py       # Centralized addresses

tests/
├── unit/              # Unit tests by module (712 tests)
├── integration/       # Integration tests (106 tests)
└── fixtures/          # JSON test data
    ├── benchmark/     # Python vs Rust comparison
    └── benchmark_python_only/  # CoW matching (Python-only)

docs/
├── sessions/          # Session handoff logs (52 sessions)
├── design/            # Architecture and algorithm designs
├── evaluations/       # Benchmark analysis
└── research/          # Future explorations (flash loans)
```

## Current State

**Phase 4 Complete:** Multi-order CoW optimization with EBBO validation.

**Capabilities:**
- All 5 Rust liquidity types (V2, V3, Balancer weighted/stable, 0x limit orders)
- Multi-order CoW matching with joint price optimization
- Ring trade detection (3-4 token cycles)
- EBBO validation with zero tolerance
- Partial fills with solver fee in limit price validation
- Per-pool gas estimates from auction data

**Rust Baseline Limitations:** The Rust solver is single-order AMM routing only. It does NOT support CoW matching or multi-order optimization.

**Current Focus:** Gap analysis - understanding why 36.53% CoW potential yields only 0.12% actual matches.

## Strategy Chain

The solver uses a **strategy chain** where each strategy processes remaining orders:

```python
# Default chain in solver/solver.py
strategies = [
    CowMatchStrategy(),      # 1. 2-order direct matching (fast path)
    MultiPairCowStrategy(),  # 2. N-order joint optimization (LP-based)
    AmmRoutingStrategy(),    # 3. AMM routing fallback
]
```

### Production Strategies

| Strategy | File | Purpose |
|----------|------|---------|
| `CowMatchStrategy` | `strategies/cow_match.py` | 2-order peer-to-peer matching |
| `MultiPairCowStrategy` | `strategies/multi_pair.py` | N-order CoW with joint price optimization across overlapping pairs |
| `AmmRoutingStrategy` | `strategies/amm_routing.py` | Single-order AMM routing through V2/V3/Balancer/limit orders |

### Research/Experimental Strategies (not in default chain)

| Strategy | File | Purpose | Notes |
|----------|------|---------|-------|
| `HybridCowStrategy` | `strategies/hybrid_cow.py` | N-order with AMM reference price | **Superseded** by MultiPairCowStrategy |
| `RingTradeStrategy` | `strategies/ring_trade.py` | Cyclic trades (A→B→C→A) | Low ROI (0.12% match rate) |

## Key Files

| File | Purpose |
|------|---------|
| `solver/solver.py` | Main Solver class, strategy orchestration, EBBO filtering |
| `solver/api/endpoints.py` | POST /solve endpoint |
| `solver/models/auction.py` | AuctionInstance, Order, Token |
| `solver/models/solution.py` | Solution, Trade, Interaction |
| `solver/models/order_groups.py` | OrderGroup for batch optimization |
| `solver/ebbo.py` | EBBO validation (EBBOPrices, EBBOValidator) |
| `solver/strategies/cow_match.py` | 2-order CoW matching |
| `solver/strategies/multi_pair.py` | N-order joint optimization (Slice 4.6) |
| `solver/strategies/amm_routing.py` | AMM routing strategy |
| `solver/strategies/double_auction.py` | Double auction algorithm (used by multi_pair) |
| `solver/strategies/ebbo_bounds.py` | Two-sided EBBO bounds calculation helper |
| `solver/strategies/ring_trade.py` | Ring trade detection (research) |
| `solver/amm/uniswap_v2.py` | V2 AMM math and encoding |
| `solver/amm/uniswap_v3/` | V3 package (quoter, encoding, etc.) |
| `solver/amm/balancer/` | Balancer package (weighted, stable) |
| `solver/pools/registry.py` | PoolRegistry (storage + PathFinder delegation) |
| `solver/routing/router.py` | SingleOrderRouter (delegates to handlers) |
| `tests/conftest.py` | Mock fixtures (MockAMM, MockRouter, etc.) |
| `PLAN.md` | Implementation roadmap |
| `BENCHMARKS.md` | Benchmarking guide |

## Development

### Commands
```bash
# Setup
pip install -e ".[dev]"

# Test
pytest                          # All tests (992)
pytest tests/unit/ -v           # Unit tests verbose
ruff check .                    # Lint
mypy solver/                    # Type check

# Run
python -m solver.api.main       # Start solver on :8000

# Benchmark
python scripts/benchmark_strategies.py --check-ebbo --limit 50
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

### EBBO Validation
EBBO (Ethereum Best Bid/Offer) ensures users get at least as good execution as AMMs:
- Zero tolerance enforced (Slice 4.6)
- Two-sided validation: `ebbo_min` protects sellers, `ebbo_max` protects buyers
- Validation at strategy level AND solver level (safety net with error logging)
- Uses `get_ebbo_bounds()` helper for consistent bounds calculation

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
- EBBO compliance (clearing rate >= AMM rate)
- Result: 2 trades, 0 interactions, gas=0

### Solution Format
A valid solution needs:
- `prices`: Uniform clearing prices for all tokens
- `trades`: Order fills with executedAmount and fee
- `interactions`: On-chain calls (AMM swaps)

### UniswapV3
Requires RPC for QuoterV2 contract calls. Set `RPC_URL` env var. Tests use `MockV3Quoter` for offline testing.

## Session Workflow

After each session, create `docs/sessions/archive/session-NN.md` with:
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
8. **EBBO is mandatory** — zero tolerance, all strategies must comply
