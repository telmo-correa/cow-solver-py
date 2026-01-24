# Slice 4.4: Ring Trade Detection - Design Document

## Overview

**Goal:** Implement ring trade detection and execution to match orders in cyclic patterns (A→B→C→A).

**Why Ring Trades Matter:**
- Direct CoW matching (A↔B) only achieves 1.4% match rate
- Ring trades achieve 5.41% match rate (3.9x improvement)
- 100% of historical auctions contain viable ring trade cycles
- Zero AMM interaction required = maximum gas savings

**Deliverables:**
- `solver/strategies/ring_trade.py` - Ring trade detection and execution
- Unit tests for cycle detection and settlement
- Integration tests on historical auction data
- Benchmark comparing ring trades vs AMM-only baseline

---

## Background

### What is a Ring Trade?

A ring trade settles multiple orders in a cycle without AMM interaction:

```
Order 1: Alice sells ETH for USDC
Order 2: Bob sells USDC for DAI
Order 3: Carol sells DAI for ETH

Settlement: Alice's ETH → Carol, Carol's DAI → Bob, Bob's USDC → Alice
Result: All three orders filled, zero AMM fees, zero slippage
```

### When is a Ring Trade Viable?

A cycle is economically viable when the product of exchange rates >= 1:

```
rate_1 × rate_2 × rate_3 >= 1

where rate_i = buy_amount_i / sell_amount_i (what each order is willing to pay)
```

If the product > 1, there's **surplus** to distribute among participants.
If the product = 1, it's break-even (still saves gas vs AMM).
If the product < 1, the cycle is not viable.

### Analysis Results (from Slice 4.3)

| Metric | Value |
|--------|-------|
| Auctions with viable cycles | 100% |
| 3-node cycles (viable) | 2,000 (57.8%) |
| 4-node cycles (viable) | 735 (25.8%) |
| Unique orders in cycles | 3,046 |
| Ring match rate | 5.41% |

**Key insight:** 3-node cycles are more common and have higher viability rate. Focus implementation on 3-node cycles first.

---

## Algorithm Design

### Phase 1: Graph Construction

Build a directed graph from orders:

```python
# Node: token address
# Edge: (sell_token → buy_token) with list of orders

def build_order_graph(orders: list[Order]) -> dict[Token, dict[Token, list[Order]]]:
    """
    Returns: graph[sell_token][buy_token] = [order1, order2, ...]
    """
```

**Complexity:** O(n) where n = number of orders

### Phase 2: Cycle Detection

Find all 3-node cycles in the graph:

```python
def find_3_cycles(graph) -> list[tuple[Token, Token, Token]]:
    """
    Find cycles A → B → C → A

    Algorithm:
    1. For each token A in graph
    2.   For each neighbor B of A
    3.     For each neighbor C of B (where C != A, C != B)
    4.       If A is neighbor of C, found cycle (A, B, C)
    5. Deduplicate by sorting tokens in each cycle
    """
```

**Complexity:** O(V × E²) worst case, but sparse in practice (~300 cycles per auction)

### Phase 3: Viability Check

For each cycle, check if it can be settled profitably:

```python
def is_cycle_viable(cycle: tuple[Token, ...], graph) -> tuple[bool, list[Order]]:
    """
    Check if product of best rates around cycle >= 1

    Returns: (is_viable, list of orders to use)
    """
    product = 1.0
    orders_used = []

    for i in range(len(cycle)):
        from_token = cycle[i]
        to_token = cycle[(i + 1) % len(cycle)]

        # Find order with best rate on this edge
        best_order = max(
            graph[from_token][to_token],
            key=lambda o: o.buy_amount / o.sell_amount
        )
        product *= best_order.buy_amount / best_order.sell_amount
        orders_used.append(best_order)

    return product >= 1.0, orders_used
```

### Phase 4: Amount Calculation

Determine how much to fill each order in the cycle:

```python
def calculate_ring_amounts(orders: list[Order]) -> list[int]:
    """
    Calculate fill amounts that balance the cycle.

    Constraint: What flows out of each order must equal what flows into the next.

    The bottleneck is the order with smallest available amount (when normalized).
    Scale all fills proportionally to this bottleneck.
    """
```

**Key challenge:** Orders have different amounts. The cycle is limited by the smallest order (normalized to a common unit).

Example:
```
Order 1: Sell 10 ETH for 20,000 USDC (rate: 2000 USDC/ETH)
Order 2: Sell 25,000 USDC for 25,000 DAI (rate: 1 DAI/USDC)
Order 3: Sell 30,000 DAI for 12 ETH (rate: 0.0004 ETH/DAI)

Bottleneck: Order 1 (10 ETH)
Fill amounts:
  - Order 1: 10 ETH → 20,000 USDC
  - Order 2: 20,000 USDC → 20,000 DAI
  - Order 3: 20,000 DAI → 8 ETH (partial fill)

Wait - this doesn't balance! Alice gives 10 ETH but Carol only gives 8 ETH.
```

**Solution:** Use clearing price that satisfies all limit prices:

```python
def find_clearing_amounts(orders: list[Order]) -> list[tuple[int, int]]:
    """
    Find (sell_fill, buy_fill) for each order such that:
    1. All limit prices satisfied: buy_fill >= sell_fill * limit_rate
    2. Conservation: sum of each token in = sum of each token out
    3. No order overfilled: sell_fill <= sell_amount

    Returns list of (sell_filled, buy_filled) tuples, or None if infeasible.
    """
```

### Phase 5: Settlement Encoding

Convert ring trade to Solution format:

```python
def build_ring_solution(
    orders: list[Order],
    fills: list[tuple[int, int]],
    clearing_prices: dict[Token, int]
) -> Solution:
    """
    Build solution with:
    - prices: uniform clearing prices for all tokens in cycle
    - trades: one trade per order with executed amounts
    - interactions: empty (no AMM calls needed)
    - gas: 0 (peer-to-peer settlement)
    """
```

---

## Implementation Plan

### File Structure

```
solver/strategies/
├── ring_trade.py          # Main strategy
│   ├── RingTradeStrategy  # SolutionStrategy implementation
│   ├── OrderGraph         # Graph construction
│   ├── CycleFinder        # Cycle detection
│   └── RingSettlement     # Amount calculation & encoding
└── __init__.py            # Export RingTradeStrategy
```

### Class Design

```python
@dataclass
class OrderGraph:
    """Directed graph of orders by token pair."""
    edges: dict[str, dict[str, list[Order]]]  # sell_token -> buy_token -> orders

    @classmethod
    def from_orders(cls, orders: Iterable[Order]) -> "OrderGraph":
        ...

    def find_3_cycles(self) -> list[tuple[str, str, str]]:
        ...

    def find_4_cycles(self, limit: int = 50) -> list[tuple[str, str, str, str]]:
        ...


@dataclass
class RingTrade:
    """A viable ring trade ready for settlement."""
    cycle: tuple[str, ...]           # Token addresses in order
    orders: list[Order]              # One order per edge
    fills: list[tuple[int, int]]     # (sell_filled, buy_filled) per order
    clearing_prices: dict[str, int]  # Uniform prices
    surplus: int                     # Total surplus generated

    def to_strategy_result(self) -> StrategyResult:
        ...


class RingTradeStrategy(SolutionStrategy):
    """Find and execute ring trades."""

    def solve(self, auction: AuctionInstance) -> StrategyResult:
        # 1. Build order graph
        graph = OrderGraph.from_orders(auction.orders)

        # 2. Find cycles (3-node first, then 4-node)
        cycles = graph.find_3_cycles() + graph.find_4_cycles()

        # 3. For each cycle, check viability and calculate settlement
        ring_trades = []
        for cycle in cycles:
            ring = self._try_settle_cycle(cycle, graph)
            if ring:
                ring_trades.append(ring)

        # 4. Select best non-overlapping rings (greedy by surplus)
        selected = self._select_rings(ring_trades)

        # 5. Combine into single StrategyResult
        return self._combine_rings(selected)
```

### Key Algorithms

#### Cycle Selection (Greedy)

Multiple cycles may share orders. Select non-overlapping cycles by surplus:

```python
def _select_rings(self, rings: list[RingTrade]) -> list[RingTrade]:
    """Greedy selection of non-overlapping rings by surplus."""
    rings.sort(key=lambda r: r.surplus, reverse=True)

    selected = []
    used_orders = set()

    for ring in rings:
        order_uids = {o.uid for o in ring.orders}
        if not order_uids & used_orders:  # No overlap
            selected.append(ring)
            used_orders.update(order_uids)

    return selected
```

#### Clearing Price Calculation

For a 3-node cycle with tokens A, B, C:

```python
def calculate_clearing_prices(
    cycle: tuple[str, str, str],
    orders: list[Order]
) -> dict[str, int]:
    """
    Set prices such that all trades execute at uniform clearing price.

    Convention: Price token A in terms of reference token (e.g., first token).
    price[A] = 1e18 (reference)
    price[B] = price[A] * rate_AB (from order A→B)
    price[C] = price[B] * rate_BC (from order B→C)

    Verify: price[A] ~= price[C] * rate_CA (cycle closes)
    """
```

---

## Testing Strategy

### Unit Tests

#### 1. Graph Construction (`test_order_graph.py`)

```python
class TestOrderGraph:
    def test_empty_orders(self):
        """Empty order list produces empty graph."""

    def test_single_order(self):
        """Single order creates one edge."""

    def test_multiple_orders_same_pair(self):
        """Multiple orders on same pair grouped together."""

    def test_bidirectional_orders(self):
        """Orders A→B and B→A create separate edges."""
```

#### 2. Cycle Detection (`test_cycle_finder.py`)

```python
class TestCycleFinder:
    def test_no_cycles(self):
        """Graph with no cycles returns empty list."""

    def test_simple_3_cycle(self):
        """A→B→C→A detected as one cycle."""

    def test_multiple_3_cycles(self):
        """Multiple independent cycles all found."""

    def test_shared_edge_cycles(self):
        """Cycles sharing an edge both detected."""

    def test_4_cycle(self):
        """A→B→C→D→A detected."""

    def test_cycle_deduplication(self):
        """Same cycle not reported multiple times."""
```

#### 3. Viability Check (`test_cycle_viability.py`)

```python
class TestCycleViability:
    def test_viable_cycle_product_above_1(self):
        """Cycle with rate product > 1 is viable."""

    def test_viable_cycle_product_equals_1(self):
        """Cycle with rate product = 1 is viable (break-even)."""

    def test_not_viable_product_below_1(self):
        """Cycle with rate product < 1 is not viable."""

    def test_missing_edge_not_viable(self):
        """Cycle with missing order on one edge is not viable."""
```

#### 4. Amount Calculation (`test_ring_amounts.py`)

```python
class TestRingAmounts:
    def test_equal_amounts(self):
        """Orders with matching amounts fill completely."""

    def test_bottleneck_order(self):
        """Smallest order limits the cycle."""

    def test_partial_fills(self):
        """Larger orders partially filled."""

    def test_limit_price_respected(self):
        """All fills respect order limit prices."""

    def test_token_conservation(self):
        """Total of each token in = total out."""
```

#### 5. Settlement Encoding (`test_ring_settlement.py`)

```python
class TestRingSettlement:
    def test_solution_format(self):
        """Solution has correct structure."""

    def test_clearing_prices(self):
        """Prices are uniform and consistent."""

    def test_no_interactions(self):
        """Ring trades have zero AMM interactions."""

    def test_gas_is_zero(self):
        """Gas estimate is 0 for peer-to-peer."""
```

### Integration Tests

#### 1. Strategy Integration (`test_ring_trade_strategy.py`)

```python
class TestRingTradeStrategy:
    def test_finds_ring_in_simple_auction(self):
        """Strategy finds and settles obvious ring trade."""

    def test_no_ring_when_none_viable(self):
        """Returns empty when no viable cycles exist."""

    def test_selects_best_rings(self):
        """Higher surplus rings preferred."""

    def test_no_order_overlap(self):
        """Selected rings don't share orders."""

    def test_combines_with_amm_routing(self):
        """Remainder orders routed through AMM."""
```

#### 2. Historical Auction Tests (`test_ring_trade_historical.py`)

```python
class TestRingTradeHistorical:
    @pytest.mark.parametrize("auction_file", HISTORICAL_AUCTIONS[:10])
    def test_finds_rings_in_historical(self, auction_file):
        """Ring trade strategy finds cycles in real auctions."""

    def test_match_rate_above_threshold(self):
        """Aggregate match rate >= 5% on historical data."""

    def test_all_settlements_valid(self):
        """All ring settlements satisfy constraints."""
```

### Benchmark Tests

```python
class TestRingTradeBenchmark:
    def test_ring_vs_amm_surplus(self):
        """Ring trades produce higher surplus than AMM-only."""

    def test_ring_gas_savings(self):
        """Ring trades save gas vs equivalent AMM routes."""

    def test_combined_strategy_improvement(self):
        """Ring + AMM beats AMM-only on historical auctions."""
```

---

## Success Criteria

### Functional Requirements

| Requirement | Test | Target |
|-------------|------|--------|
| Detect 3-node cycles | Unit test | 100% of viable cycles found |
| Detect 4-node cycles | Unit test | 100% of viable cycles found |
| Viability check correct | Unit test | No false positives/negatives |
| Amount calculation correct | Unit test | Token conservation verified |
| Settlement valid | Integration test | All constraints satisfied |

### Performance Requirements

| Metric | Target | Rationale |
|--------|--------|-----------|
| Ring match rate | >= 5% | Analysis showed 5.41% potential |
| Cycle detection time | < 100ms | Must fit in solver deadline |
| Strategy overhead | < 10% of total solve time | Don't slow down AMM routing |

### Quality Requirements

| Requirement | How to Verify |
|-------------|---------------|
| No duplicate fills | Order can only be in one ring |
| Limit prices respected | All fills checked against limits |
| Gas savings realized | Interactions = 0 for ring trades |

---

## Open Questions

### Q1: How to handle partial ring fills?

If a cycle has orders with mismatched amounts, we can either:
- **Option A:** Fill the minimum, leave remainders for AMM routing
- **Option B:** Try to combine multiple orders on same edge

**Recommendation:** Start with Option A (simpler), measure how much surplus is left on table.

### Q2: Priority between 3-node and 4-node cycles?

3-node cycles are more common and viable. Options:
- **Option A:** Process 3-node first, then 4-node with remaining orders
- **Option B:** Rank all cycles by surplus, select greedily

**Recommendation:** Option B (globally optimal selection).

### Q3: How to integrate with existing strategies?

Options:
- **Option A:** Run before CowMatchStrategy (ring trades are superset)
- **Option B:** Run after CowMatchStrategy (catch what direct CoW missed)
- **Option C:** Replace CowMatchStrategy entirely

**Recommendation:** Option A - ring trades subsume direct CoW matches.

### Q4: What about rings that need AMM to balance?

Sometimes a ring is "almost" viable but needs small AMM trade to close. This is out of scope for Slice 4.4 but noted for future work.

---

## Timeline

| Task | Estimate |
|------|----------|
| Graph construction + cycle detection | 1 session |
| Viability check + amount calculation | 1 session |
| Settlement encoding + strategy integration | 1 session |
| Unit tests | 1 session |
| Integration tests + benchmarks | 1 session |
| **Total** | **5 sessions** |

---

## References

- [Slice 4.3 Ring Analysis](../evaluations/slice-4.3-ring-analysis.md) - Empirical analysis
- [Slice 4.1 Problem Formulation](phase4-slice4.1-problem-formulation.md) - Optimization model
- [CoW Protocol Docs](https://docs.cow.fi/) - Settlement semantics
- `scripts/analyze_ring_potential.py` - Analysis script (basis for implementation)
