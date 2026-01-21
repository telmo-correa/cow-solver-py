# Benchmarking Guide

This document describes how to run benchmark comparisons between the Python solver and the reference Rust solver from CoW Protocol.

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

```bash
python scripts/run_benchmarks.py --python-url http://localhost:8000 --rust-url http://localhost:8080
```

## Latest Benchmark Results

```
============================================================
CoW Protocol Solver Benchmark (HTTP)
============================================================
Auctions directory: tests/fixtures/auctions/benchmark
Python solver: http://localhost:8000
Rust solver:   http://localhost:8080

Total auctions: 5
Successful: 5

Python found solutions: 5/5
Rust found solutions:   5/5

Time Comparison (Python / Rust):
  Mean:   1.95x
  Median: 1.67x
  Range:  1.49x - 3.15x
  Faster: Rust

Individual Results:
------------------------------------------------------------
  buy_usdc_with_weth (BUY ORDER):
    Python: 6.4ms, solutions=1
    Rust:   2.0ms, solutions=1
  usdc_to_weth:
    Python: 1.5ms, solutions=1
    Rust:   0.9ms, solutions=1
  weth_to_dai:
    Python: 1.4ms, solutions=1
    Rust:   0.8ms, solutions=1
  weth_to_usdc:
    Python: 1.3ms, solutions=1
    Rust:   0.8ms, solutions=1
  large_weth_to_usdc:
    Python: 1.1ms, solutions=1
    Rust:   0.8ms, solutions=1
```

**Summary**: The Rust solver is approximately 2x faster than the Python solver. Both solvers find valid solutions for all test cases, including buy orders.

## Setting Up the Rust Solver

The reference Rust solver is part of the CoW Protocol services repository.

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
- Liquidity data describing available AMM pools

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
