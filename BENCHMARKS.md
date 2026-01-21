# Benchmarking Guide

This document describes how to run benchmark comparisons between the Python solver and the reference Rust solver from CoW Protocol.

## Rust Baseline Solver Capabilities

The Rust "baseline" solver is part of the [CoW Protocol Services repository](https://github.com/cowprotocol/services) (`crates/solvers/`). It is a **single-order AMM routing engine**. Understanding its capabilities is important for interpreting benchmark results.

### Feature Summary

| Category | Feature | Rust | Python | Notes |
|----------|---------|------|--------|-------|
| **Order Types** | Sell orders | ✅ | ✅ | Fixed sell amount |
| | Buy orders | ✅ | ✅ | Fixed buy amount, rounding protection |
| | Limit orders | ✅ | ✅ | With order class support |
| | Market orders | ✅ | ✅ | Protocol fees only |
| | Partially fillable | ✅ | ❌ | Binary search (configurable attempts) |
| **Liquidity** | UniswapV2 (constant product) | ✅ | ✅ | 0.3% fee |
| | UniswapV3 (concentrated) | ⚠️ | ❌ | Requires RPC config |
| | Balancer V2 weighted | ✅ | ❌ | V0 and V3+ versions |
| | Balancer/Curve stable | ✅ | ❌ | With amplification |
| | 0x limit orders | ✅ | ❌ | As liquidity source |
| **Routing** | Direct swaps (A→B) | ✅ | ✅ | Single pool |
| | Multi-hop (A→B→C) | ✅ | ✅ | Configurable max hops |
| | Order splitting | ❌ | ❌ | Single path per order |
| **Multi-Order** | CoW matching | ❌ | ✅ | Peer-to-peer settlement |
| | Batch optimization | ❌ | ❌ | Independent per order |
| | JIT liquidity | ❌ | ❌ | No dynamic provision |

### Order Types Supported (Rust)

**Fully Supported:**
- **Sell Orders** - Fixed sell amount, minimum buy amount
- **Buy Orders** - Fixed buy amount, maximum sell amount (with rounding protection)
- **Order Classes** - Both `market` and `limit` classes
- **Partially Fillable** - Binary search algorithm (1/1, 1/2, 1/4, 1/8...) with configurable `max-partial-attempts` (default: 5)

### Liquidity Sources (Rust)

The Rust solver supports 5 liquidity types:

| Type | Description | Gas Estimate |
|------|-------------|--------------|
| **Constant Product** | UniswapV2-style pools | ~110,000 |
| **Weighted Product** | Balancer V2 weighted pools (V0 and V3+) | ~88,892 |
| **Stable Pools** | Curve-style with amplification | ~183,520 |
| **Concentrated** | UniswapV3 (requires `uni-v3-node-url` config) | ~106,000 |
| **Limit Orders** | 0x Protocol-style | Variable |

### Routing Configuration (Rust)

Multi-hop routing is configurable via TOML config:

```toml
# Example: crates/solvers/config/example.baseline.toml
max-hops = 1                    # 0=direct only, 1=one intermediate, 2=two
base-tokens = ["0xC02..."]      # Intermediate routing tokens (WETH auto-included)
solution-gas-offset = 106391    # Settlement overhead
```

### What the Rust Baseline Does NOT Support

| Feature | Notes |
|---------|-------|
| **CoW matching** | Does not match orders against each other |
| **Multi-order optimization** | Processes each order independently in a loop |
| **Order splitting** | Finds single best path, doesn't split across routes |
| **JIT liquidity** | No dynamic liquidity provision |
| **Cross-order batching** | Solutions don't combine multiple orders |

When the Rust baseline solver receives a multi-order auction, it processes each order independently and returns separate solutions. It does **not** attempt to find matching opportunities between orders.

## Benchmark Categories

Due to the Rust solver's limitations, benchmarks are split into two categories:

### Shared Functionality Benchmarks (`benchmark/`)

Location: `tests/fixtures/auctions/benchmark/`

These test features supported by both Python and Rust solvers:
- Single sell orders via AMM
- Single buy orders via AMM
- Multi-hop routing (A -> WETH -> B)

**Use for**: Python vs Rust comparison, verifying parity

### Python-Only Benchmarks (`benchmark_python_only/`)

Location: `tests/fixtures/auctions/benchmark_python_only/`

These test Python-only features not available in the Rust baseline:
- CoW matching (2-order peer-to-peer settlement)
  - sell-sell matches (`cow_pair_basic.json`)
  - sell-buy matches (`cow_pair_sell_buy.json`)
  - buy-buy matches (`cow_pair_buy_buy.json`)
- Partial CoW + AMM remainder
  - partial match with AMM routing for remainder (`partial_cow_amm.json`)
- Future: multi-order batching

**Use for**: Python-only validation (no Rust comparison possible)

## Quick Start

Both solvers run as HTTP servers and accept the same auction JSON format.

### Step 1: Start Both Solvers

**Terminal 1 - Python Solver (port 8000):**
```bash
python -m solver.api.main
```

**Terminal 2 - Rust Solver (port 8080):**
```bash
# From the cowprotocol/services repo
./target/release/solvers --addr 127.0.0.1:8080 baseline --config crates/solvers/config/example.baseline.toml
```

### Step 2: Run Benchmarks

**Shared functionality (Python vs Rust):**
```bash
python scripts/run_benchmarks.py --python-url http://localhost:8000 --rust-url http://localhost:8080
```

**Python-only features:**
```bash
python scripts/run_benchmarks.py --python-url http://localhost:8000 \
    --auctions tests/fixtures/auctions/benchmark_python_only
```

## Latest Benchmark Results

### Shared Functionality (Python vs Rust)

```
============================================================
CoW Protocol Solver Benchmark
============================================================
Auctions directory: tests/fixtures/auctions/benchmark
Python solver: http://localhost:8000
Rust solver:   http://localhost:8080

Total auctions: 7
Successful: 7/7

Python found solutions: 7/7
Rust found solutions:   7/7
Solutions match: 7/7 (100%)

Individual Results:
------------------------------------------------------------
  buy_usdc_with_weth (BUY ORDER):
    Python: OK, Rust: OK, Match: YES
  usdc_to_weth:
    Python: OK, Rust: OK, Match: YES
  weth_to_dai:
    Python: OK, Rust: OK, Match: YES
  weth_to_usdc:
    Python: OK, Rust: OK, Match: YES
  large_weth_to_usdc:
    Python: OK, Rust: OK, Match: YES
  usdc_to_dai_multihop (2-hop):
    Python: OK, Rust: OK, Match: YES
  dai_to_usdc_multihop (2-hop):
    Python: OK, Rust: OK, Match: YES
```

**Summary**: Both solvers produce identical solutions for all 7 test cases, including buy orders and multi-hop routes. Python is approximately 2x slower than Rust.

### Python-Only Features

```
============================================================
Python-Only Benchmarks
============================================================
Auctions directory: tests/fixtures/auctions/benchmark_python_only

Total auctions: 1
Python found solutions: 1/1

Individual Results:
------------------------------------------------------------
  cow_pair_basic (CoW Match):
    Python: OK (2 trades, 0 interactions, gas=0)
    Rust: N/A (not supported by baseline solver)
```

**Summary**: Python CoW matching produces mathematically correct solutions. Order A (sell 1 WETH, want 2500 USDC) matches with Order B (sell 2600 USDC, want 1 WETH). Both parties receive at least their limit prices, with Order A receiving 100 USDC surplus.

## Setting Up the Rust Solver

The reference Rust solver is part of the [CoW Protocol Services repository](https://github.com/cowprotocol/services).

### Clone and Build

```bash
# Clone the CoW Protocol services repository
git clone --depth 1 https://github.com/cowprotocol/services.git cow-services
cd cow-services

# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build the solver in release mode
cargo build --release -p solvers
```

### Start the Rust Solver

```bash
./target/release/solvers --addr 127.0.0.1:8080 baseline --config crates/solvers/config/example.baseline.toml
```

### Verify It's Running

```bash
curl -X POST http://127.0.0.1:8080/solve \
    -H "Content-Type: application/json" \
    -d '{"id":"1","tokens":{},"orders":[],"liquidity":[],"effectiveGasPrice":"1","deadline":"2030-01-01T00:00:00Z","surplusCapturingJitOrderOwners":[]}'
# Expected: {"solutions":[]}
```

## Benchmark CLI Options

```
usage: run_benchmarks.py [-h] [--auctions AUCTIONS] [--python-url URL]
                         [--rust-url URL] [--output OUTPUT]
                         [--format {markdown,json,both}] [--verbose]

Options:
  --auctions AUCTIONS   Directory containing auction JSON fixtures
                        (default: tests/fixtures/auctions/benchmark)

  --python-url URL      URL of Python solver HTTP server
                        (e.g., http://localhost:8000)

  --rust-url URL        URL of Rust solver HTTP server
                        (e.g., http://localhost:8080)

  --output OUTPUT       Directory for output reports
                        (default: benchmarks/results)

  --format FORMAT       Output format: markdown, json, or both
                        (default: both)

  --verbose, -v         Enable verbose logging
```

## Auction Fixture Format

Both solvers accept the same JSON auction format. Fixtures must include:
- Token information with `referencePrice`, `availableBalance`, and `trusted` fields
- Orders with full CoW Protocol fields including `fullSellAmount`, `fullBuyAmount`, `feePolicies`, etc.
- Liquidity data describing available AMM pools (for AMM routing benchmarks)

Example fixture (see `tests/fixtures/auctions/benchmark/` for full examples):

```json
{
  "id": "1",
  "tokens": {
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {
      "decimals": 18,
      "symbol": "WETH",
      "referencePrice": "1000000000000000000",
      "availableBalance": "1000000000000000000",
      "trusted": true
    },
    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {
      "decimals": 6,
      "symbol": "USDC",
      "referencePrice": "400000000000000",
      "availableBalance": "10000000000",
      "trusted": true
    }
  },
  "orders": [
    {
      "uid": "0x...",
      "sellToken": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
      "buyToken": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
      "sellAmount": "1000000000000000000",
      "fullSellAmount": "1000000000000000000",
      "buyAmount": "2400000000",
      "fullBuyAmount": "2400000000",
      "feePolicies": [],
      "validTo": 0,
      "kind": "sell",
      "owner": "0x5b1e2c2762667331bc91648052f646d1b0d35984",
      "partiallyFillable": false,
      "preInteractions": [],
      "postInteractions": [],
      "sellTokenSource": "erc20",
      "buyTokenDestination": "erc20",
      "class": "market",
      "appData": "0x0000000000000000000000000000000000000000000000000000000000000000",
      "signingScheme": "presign",
      "signature": "0x"
    }
  ],
  "liquidity": [
    {
      "kind": "constantProduct",
      "tokens": {
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"balance": "20000000000000000000000"},
        "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {"balance": "50000000000000"}
      },
      "fee": "0.003",
      "id": "0",
      "address": "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
      "router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
      "gasEstimate": "110000"
    }
  ],
  "effectiveGasPrice": "15000000000",
  "deadline": "2106-01-01T00:00:00.000Z",
  "surplusCapturingJitOrderOwners": []
}
```

## API Endpoints

| Solver | Endpoint | Method |
|--------|----------|--------|
| Python | `/{environment}/{network}` (e.g., `/benchmark/mainnet`) | POST |
| Rust | `/solve` | POST |

Both solvers accept the same auction JSON body and return the same response format:

```json
{
  "solutions": [
    {
      "id": 0,
      "prices": {"0x...": "...", "0x...": "..."},
      "trades": [{"kind": "fulfillment", "order": "0x...", "executedAmount": "..."}],
      "preInteractions": [],
      "interactions": [...],
      "postInteractions": [],
      "gas": 166391
    }
  ]
}
```

## Understanding the Results

### Time Ratio
- `< 1.0` = Python is faster
- `= 1.0` = Equal performance
- `> 1.0` = Python is slower (e.g., 2.0x means Python took twice as long)

### Success Categories
- **Both succeeded**: Both solvers found valid solutions
- **Python only**: Python found a solution, Rust didn't
- **Rust only**: Rust found a solution, Python didn't
- **Both failed**: Neither solver found a solution

## Report Output

Benchmarks generate reports in `benchmarks/results/`:
- `benchmark_YYYYMMDD_HHMMSS.md` - Human-readable markdown
- `benchmark_YYYYMMDD_HHMMSS.json` - Machine-readable JSON
