# CoW Solver (Python)

A Python implementation of a CoW Protocol solver for learning and benchmarking.

## Overview

This project implements a solver for [CoW Protocol](https://cow.fi/) batch auctions in Python, with the goal of:

1. Learning the CoW Protocol solver problem domain
2. Benchmarking Python performance against the reference Rust implementation
3. Exploring optimization techniques (Cython, algorithmic improvements)
4. Demonstrating AI-assisted development workflow

## Status

✅ **Phase 1 Complete** — Single order routing via UniswapV2 (direct + multi-hop)

See [PLAN.md](PLAN.md) for the detailed implementation roadmap.

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Usage

### Run the Solver API

```bash
python -m solver.api.main
# Server starts at http://localhost:8000
```

### Run Tests

```bash
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
```

### Run Benchmarks

```bash
python -m benchmarks.harness
```

### Collect Historical Auctions

```bash
python -m scripts.collect_auctions --count 50
```

## Project Structure

```
solver/          # Main solver package
  api/           # FastAPI endpoints
  models/        # Pydantic schemas
  amm/           # AMM math (UniswapV2)
  routing/       # Order routing and solution building
  constants.py   # Centralized constants

benchmarks/      # Performance comparison harness
scripts/         # Utility scripts
tests/           # Test suite with mock fixtures for DI
```

## Documentation

- [PLAN.md](PLAN.md) — Implementation roadmap
- [SESSIONS.md](SESSIONS.md) — Development session log
- [CLAUDE.md](CLAUDE.md) — AI assistant context

## License

MIT
