# CoW Protocol Settlement Optimization: Problem Formulation

This document formulates the CoW Protocol batch settlement problem as a mathematical optimization problem. The goal is to understand the problem structure before choosing a solution approach.

## 1. Problem Overview

Given a batch of orders, find:
1. Which orders to fill (and by how much)
2. At what clearing prices
3. Which AMM interactions to include

Such that all constraints are satisfied and surplus is maximized.

---

## 2. Input Data

### 2.1 Orders

Each order $i \in O$ has:
- $s_i$: sell token address
- $b_i$: buy token address
- $S_i$: maximum sell amount
- $B_i$: minimum buy amount (defines limit price)
- $k_i \in \{\text{sell}, \text{buy}\}$: order kind
- $f_i \in \{\text{partial}, \text{fok}\}$: fill type (partially fillable or fill-or-kill)
- $c_i \in \{\text{market}, \text{limit}\}$: order class

The **limit price** (minimum acceptable rate) is:
$$r_i^{limit} = \frac{B_i}{S_i}$$

For a sell order, this means: "I'm selling $S_i$ tokens and I must receive at least $B_i$ tokens."

### 2.2 Tokens

Set of tokens $T$ appearing in orders. For each token $t \in T$:
- $p_t$: reference price (ETH value of 1e18 tokens, used for fee calculation)
- $d_t$: decimals (typically 18)

### 2.3 Liquidity (AMMs)

Set of AMM pools $L$. Each pool $\ell \in L$:
- Pair of tokens $(t_\ell^{in}, t_\ell^{out})$
- Exchange function $\phi_\ell(x)$: input $x$ of $t_\ell^{in}$ yields $\phi_\ell(x)$ of $t_\ell^{out}$
- Gas cost $g_\ell$

Note: AMM functions are generally non-linear (constant product, concentrated liquidity, etc.).

### 2.4 Auction Parameters

- $G$: effective gas price (wei per gas unit)
- Native token reference price for fee denomination

---

## 3. Decision Variables

### 3.1 Fill Amounts

For each order $i$:
- $x_i \in [0, S_i]$: amount of sell token filled

For fill-or-kill orders, this becomes a binary constraint:
$$x_i \in \{0, S_i\} \quad \text{if } f_i = \text{fok}$$

### 3.2 Clearing Prices

For each token $t \in T$:
- $P_t > 0$: uniform clearing price

### 3.3 AMM Usage

For each pool $\ell \in L$:
- $a_\ell \geq 0$: amount of input token sent to this pool

---

## 4. Constraints

### 4.1 Uniform Clearing Price Constraint (CRITICAL)

**All orders on the same directed pair must execute at the same rate.**

For order $i$ with sell token $s_i$ and buy token $b_i$:
$$\frac{y_i}{x_i} = \frac{P_{s_i}}{P_{b_i}} \quad \text{for all } i \text{ with } x_i > 0$$

Where $y_i$ is the buy amount received by order $i$.

Equivalently:
$$y_i \cdot P_{b_i} = x_i \cdot P_{s_i}$$

**This constraint is bilinear** because it involves products of:
- $y_i$ (derived from $x_i$) with $P_{b_i}$
- $x_i$ with $P_{s_i}$

If we substitute $y_i = x_i \cdot \frac{P_{s_i}}{P_{b_i}}$, then:
$$y_i = x_i \cdot \frac{P_{s_i}}{P_{b_i}}$$

This defines the buy amount as a function of the fill and price ratio.

### 4.2 Limit Price Satisfaction

Order $i$ must receive at least their limit price:
$$\frac{y_i}{x_i} \geq \frac{B_i}{S_i} \quad \text{for } x_i > 0$$

Or equivalently:
$$y_i \cdot S_i \geq x_i \cdot B_i$$

Substituting $y_i = x_i \cdot \frac{P_{s_i}}{P_{b_i}}$:
$$x_i \cdot \frac{P_{s_i}}{P_{b_i}} \cdot S_i \geq x_i \cdot B_i$$

For $x_i > 0$:
$$\frac{P_{s_i}}{P_{b_i}} \geq \frac{B_i}{S_i}$$

**Key insight**: This is a constraint purely on prices, independent of fill amounts.

### 4.3 Token Conservation (with Fee Sink)

For each token $t$, the sum of outflows must equal the sum of inflows:

$$\sum_{i: b_i = t} y_i + \sum_{\ell: t_\ell^{in} = t} a_\ell = \sum_{i: s_i = t} (x_i - \text{fee}_i) + \sum_{\ell: t_\ell^{out} = t} \phi_\ell(a_\ell)$$

Where:
- Left side: tokens received by orders + tokens sent to AMMs
- Right side: tokens sold by orders (minus fees) + tokens received from AMMs

### 4.4 Fee Calculation

For limit orders ($c_i = \text{limit}$):
$$\text{fee}_i = \frac{g_i \cdot G \cdot 10^{18}}{p_{s_i}}$$

Where:
- $g_i$ is the gas cost attributed to order $i$
- $G$ is the effective gas price
- $p_{s_i}$ is the reference price of the sell token

For market orders: $\text{fee}_i = 0$ (protocol handles separately).

**Note**: For pure CoW (no AMM), gas cost is minimal (just transfers), so fees are small. For AMM routing, each interaction adds gas cost.

### 4.5 Fill Bounds

For partially fillable orders:
$$0 \leq x_i \leq S_i$$

For fill-or-kill orders:
$$x_i \in \{0, S_i\}$$

The fill-or-kill constraint makes this a **Mixed-Integer** problem.

### 4.6 Non-Negativity

- $x_i \geq 0$ for all orders
- $P_t > 0$ for all tokens
- $a_\ell \geq 0$ for all AMM interactions
- $\text{fee}_i \geq 0$ for all orders

---

## 5. Objective Function

Maximize total **surplus** (benefit to users beyond their minimum requirements):

$$\text{maximize} \sum_{i \in O} \left( y_i - x_i \cdot \frac{B_i}{S_i} \right)$$

This measures how much better users do compared to their limit prices.

Alternatively, in a **cost minimization** formulation:
$$\text{minimize} \sum_{\ell \in L} g_\ell \cdot [a_\ell > 0]$$

Minimize gas costs from AMM interactions (prefer CoW matching).

---

## 6. Problem Structure Analysis

### 6.1 Classification

The problem is:
1. **Bilinear** due to $x_i \cdot P_t$ terms in uniform price constraint
2. **Mixed-Integer** due to fill-or-kill orders
3. **Non-Convex** due to both above factors
4. **Non-Linear** if AMM functions are included

This is a **Mixed-Integer Bilinear Program (MIBLP)**.

### 6.2 Hardness

General MIBLP is NP-hard. However, we can exploit structure:

**Key Observation**: If prices $P_t$ are fixed, the problem becomes:
- Linear in fill amounts $x_i$
- Mixed-integer only due to fill-or-kill orders

**Key Observation**: If fills $x_i$ are fixed, the prices $P_t$ are determined by:
- Linear constraints from limit price satisfaction
- Consistency with the exchange rates achieved

### 6.3 Special Cases

**Pure CoW (no AMMs)**:
- No AMM constraints
- Conservation becomes: $\sum_{i: b_i = t} y_i = \sum_{i: s_i = t} x_i$ for each token
- Fees are negligible (only transfer gas)

**Two-token pair**:
- Only two prices $P_A, P_B$
- Price ratio $P_A/P_B$ is the single degree of freedom
- Reduces to double auction (current implementation)

**Ring trade (n-token cycle)**:
- Each token appears exactly once as sell and once as buy
- Conservation forces specific relationships
- Feasibility check is product of rates $\leq 1$

---

## 7. Decomposition Approaches

Given the bilinear structure, natural decompositions exist:

### 7.1 Price-First, Then Fills

1. Enumerate candidate price points (or price relationships)
2. For each price set, solve LP for optimal fills
3. Select best solution

**Candidate prices** come from:
- Order limit prices (critical points)
- AMM spot prices
- Geometric mean of crossing orders

### 7.2 Fills-First, Then Prices

1. Find matching structure (which orders match with which)
2. Compute consistent prices from the matching
3. Verify limit constraints

This is essentially what current CoW/ring strategies do.

### 7.3 Alternating Optimization

1. Initialize prices (e.g., from AMM spot prices)
2. Solve for optimal fills given prices
3. Update prices to maximize surplus
4. Repeat until convergence

---

## 8. Relationship to Existing Strategies

### 8.1 CowMatchStrategy (2-order)

Handles the special case:
- Exactly 2 orders
- Opposite directions on same pair
- Prices derived from order limit prices
- Fill amounts computed to satisfy conservation

### 8.2 HybridCowStrategy (N-order, single pair)

Handles:
- Multiple orders on same token pair
- Uses AMM price as reference (fixes price ratio)
- Given price, solves LP for fills (double auction)

### 8.3 RingTradeStrategy (N-token cycle)

Handles:
- Orders forming a cycle A→B→C→...→A
- Each token has exactly one in-edge and one out-edge
- Feasibility: product of limit rates $\leq 1$
- Prices derived from the cycle structure

### 8.4 Gap Analysis

Current strategies miss:
1. **Multiple disconnected pairs** processed independently (could share surplus)
2. **Partial AMM fill** where CoW fills part, AMM fills rest
3. **Cross-pair optimization** where price on one pair affects another
4. **Multi-cycle selection** beyond greedy

---

## 9. Computational Considerations

### 9.1 Scale

Typical auction:
- 10-1000 orders
- 50-500 tokens
- 100-5000 liquidity sources

### 9.2 Time Budget

Solvers have ~2-5 seconds to respond. Solution approach must be:
- Fast enough for real-time use
- Parallelizable if needed
- Have good anytime behavior (produce valid solution quickly, improve if time allows)

### 9.3 Integer Variables

Fill-or-kill orders create integer constraints. In practice:
- Most orders are fill-or-kill
- Could be hundreds of binary variables
- Full branch-and-bound may be too slow

**Practical approach**: Relax integers, solve continuous, round carefully.

---

## 10. Summary

### Problem Type
**Mixed-Integer Bilinear Program (MIBLP)**

### Key Constraints
1. Uniform clearing prices (bilinear)
2. Limit price satisfaction (linear in prices given fills)
3. Token conservation with fee sink
4. Fill-or-kill (integer)

### Exploitable Structure
- Price-fixing linearizes the fill problem
- Fill-fixing determines prices
- Special cases (pairs, rings) have efficient algorithms
- Limit prices create natural price breakpoints

### Open Questions
1. How to efficiently search the price space?
2. Can we enumerate "relevant" price combinations?
3. What's the best rounding strategy for relaxed solutions?
4. How to handle AMM non-linearity practically?

---

## 11. Benchmark Results (Historical Data)

Benchmarked on 50 mainnet auctions (280,920 orders):

### 11.1 Strategy Performance

| Strategy | Orders Matched | Match Rate | Auctions w/ Matches |
|----------|----------------|------------|---------------------|
| CowMatch | 0 | 0.00% | 0/50 |
| HybridCow | 192 | 0.07% | 42/50 |
| RingTrade | 467 | 0.17% | 50/50 |

### 11.2 Theoretical Potential

| Metric | Count | Percentage |
|--------|-------|------------|
| Orders on bidirectional pairs | 102,373 | 36.4% |
| Orders on crossing pairs (ask ≤ bid) | 40,521 | 14.4% |
| Orders matched by best strategy | 467 | 0.17% |

### 11.3 Gap Analysis

**Primary Bottleneck: Price Crossing**
- Only 39.6% of CoW-potential orders have crossing prices
- 60.4% cannot match (ask > bid)

**Crossing Pairs Breakdown:**
- 611 pairs have crossing prices
- 53 are 1v1 (single order each side)
- 558 are multi-order (need aggregation)
- Most limited by A-sellers (sell pressure)

**Fill-or-Kill Impact:**
- 94.9% of CoW-potential orders are partially fillable
- Only 5.1% are fill-or-kill (not a major bottleneck)

### 11.4 Key Insight

The gap between **40,521 crossing orders** and **467 matched** (87x difference) comes from:

1. **Strategy limitations:**
   - CowMatch only handles 2-order auctions
   - HybridCow needs AMM reference price
   - RingTrade needs exact cycles

2. **Volume imbalance:** Unequal buy/sell pressure on pairs

3. **Token overlap:** Multiple pairs share tokens, causing clearing price conflicts

---

## 12. Prototype Results

### 12.1 Price Enumeration

Tested on 50 auctions (280,920 orders):

| Approach | Orders Matched | Rate | Time/auction |
|----------|----------------|------|--------------|
| RingTrade | 467 | 0.17% | 56ms |
| Price Enumeration (greedy) | 522 | 0.19% | 149ms |
| Price Enumeration + LP | 422 | 0.15% | 180ms |

**Observation**: Greedy enumeration matches more orders but may not enforce
strict token conservation. LP-based approach is more correct but slower.

### 12.2 LP Solver Evaluation

scipy.optimize.linprog (HIGHS solver):
- Fast: 8ms total LP solve time per auction
- Correct: Enforces exact token conservation
- Tractable: Handles ~200 price candidates × ~500 orders

**Key insight**: LP solve time is negligible (~8ms per auction).
The bottleneck is price enumeration overhead (~180ms) and token overlap handling.

### 12.3 Remaining Gap

With price enumeration + LP, we match ~420 orders out of ~40,000 crossing orders.
The 100x gap comes from:

1. **Token overlap**: Processing pairs independently prevents matching when
   pairs share tokens (would need multi-pair price coordination)

2. **Volume imbalance**: Many pairs have 100x more sell pressure than buy
   pressure (or vice versa), limiting matchable volume

3. **Price candidate selection**: Current approach only uses limit prices;
   intermediate prices might unlock more matches

---

## 13. Conclusions

### What We Learned

1. **Problem is tractable for per-pair optimization**: LP solve time is <10ms
   per auction, well within the 2-5 second deadline.

2. **Token overlap is the main blocker**: Independent pair processing leaves
   significant value on the table. Multi-pair coordination is needed.

3. **Current strategies are close to per-pair optimal**: RingTrade (467)
   vs LP-per-pair (422) shows current strategies capture most per-pair value.

4. **The 100x gap requires multi-pair optimization**: To capture more of the
   40,000 crossing orders, we need:
   - Joint price optimization across pairs
   - Token-aware pair selection
   - Possibly bilinear/MILP solver for full problem

### Recommended Path Forward

1. **Short-term**: Improve current strategies incrementally
   - Better pair prioritization (volume-weighted)
   - Process more pairs before hitting token overlap

2. **Medium-term**: Implement multi-pair price coordination
   - Group pairs by connected component (shared tokens)
   - Optimize prices jointly within each component

3. **Long-term**: Full MIBLP solver for optimal matching
   - Requires specialized solver (Gurobi, CPLEX, or custom)
   - May be too slow for real-time use

---

## 14. Next Steps

~~1. Benchmark existing strategies on historical data to establish baseline~~ ✓
~~2. Characterize the gap between optimal and current strategies~~ ✓
~~3. Prototype price enumeration to test if discrete price search is tractable~~ ✓
~~4. Evaluate solvers (scipy) for the continuous relaxation~~ ✓
5. **Design multi-pair coordination** for connected token components
6. **Implement improved HybridCowStrategy** with pair prioritization
