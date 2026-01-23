# Documentation

This directory contains project documentation organized by purpose.

## Structure

```
docs/
├── design/          # Design documents and problem formulations
├── evaluations/     # Benchmarks, analysis, and evaluation results
├── research/        # Research spikes and exploratory docs
├── reviews/         # Architecture and code reviews
└── sessions/        # Development session logs
    ├── summaries/   # Phase summaries
    └── archive/     # Individual session logs
```

## Quick Links

### Current Phase: Phase 4 (Unified Optimization)

- [Slice 4.1 Problem Formulation](design/phase4-slice4.1-problem-formulation.md) - Formal optimization problem definition
- [Slice 4.3 Evaluation Results](evaluations/slice-4.3-evaluation-results.md) - Hybrid CoW+AMM evaluation
- [Slice 4.3 Ring Analysis](evaluations/slice-4.3-ring-analysis.md) - Ring trade potential (5.41% match rate)

### Design Documents

| Document | Description |
|----------|-------------|
| [Phase 4 Problem Formulation](design/phase4-slice4.1-problem-formulation.md) | Constraint optimization for multi-order matching |

### Evaluations & Benchmarks

| Document | Description |
|----------|-------------|
| [Slice 4.2 Benchmark Plan](evaluations/slice-4.2-benchmark-plan.md) | Hybrid strategy benchmark methodology |
| [Slice 4.3 Evaluation](evaluations/slice-4.3-evaluation-results.md) | Direct CoW match rate: 1.4% |
| [Slice 4.3 Ring Analysis](evaluations/slice-4.3-ring-analysis.md) | Ring trade match rate: 5.41% (3.9x improvement) |

### Research

| Document | Description |
|----------|-------------|
| [Flash Loans](research/flash-loans.md) | Flash loan providers and integration design |

### Reviews

| Document | Description |
|----------|-------------|
| [2026-01-23 Architecture Review](reviews/2026-01-23-architecture.md) | Uint256 validation, Balancer overflow, API hardening |

### Session Logs

See [sessions/README.md](sessions/README.md) for the full development history.

**Recent:**
- Sessions 43-47: Phase 4 (Slice 4.1-4.3)
- Sessions 38-42: Phase 3.5 (0x Limit Orders) + Pre-Phase 4
- Sessions 25-37: Phase 3.2 (Balancer Integration)

## Contributing

When adding documentation:

1. **Design docs** → `design/` - Problem formulations, architecture decisions
2. **Evaluations** → `evaluations/` - Benchmarks, analysis, metrics
3. **Research** → `research/` - Exploratory research, spikes
4. **Reviews** → `reviews/` - Architecture/code reviews (name: `YYYY-MM-DD-topic.md`)
5. **Session logs** → `sessions/` - Daily progress (see [template](sessions/session-template.md))
