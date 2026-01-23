# Flash Loans Research & Design Decisions

**Date:** 2026-01-23
**Status:** Research complete, implementation deferred to Phase 4

## Overview

Flash loans are uncollateralized loans that must be borrowed and repaid within a single atomic transaction. They can enable better user fills by providing temporary capital for arbitrage, split routing, and ring trades.

## Key Finding: CoW Protocol Already Has Flash Loan Infrastructure

CoW Protocol introduced flash loan support in [CIP-66](https://github.com/cowprotocol/flash-loan-wrapper-solver) with the following contracts (same address on all networks):

| Contract | Address |
|----------|---------|
| FlashLoanRouter | `0x9da8B48441583a2b93e2eF8213aAD0EC0b392C69` |
| AaveBorrower | `0x7d9C4DeE56933151Bc5C909cfe09DEf0d315CB4A` |
| ERC3156Borrower | `0x47d71b4B3336AB2729436186C216955F3C27cD04` |

**How it works:**
1. Solver calls `flashLoanAndSettle(loans[], settleCalldata)` instead of `settle` directly
2. Router borrows from specified providers
3. Router calls `settle` with borrowed funds available
4. Settlement executes (all existing interactions work unchanged)
5. Router repays loans from settlement proceeds
6. If repayment fails → entire transaction reverts

This means all existing liquidity interactions (V2, V3, Balancer, limit orders) are already batchable into a single transaction with flash loans.

## Flash Loan Providers Comparison

| Provider | Fee | Min Gas | Liquidity | Notes |
|----------|-----|---------|-----------|-------|
| **Balancer** | 0% | ~24k | Pool-dependent | Most gas efficient, zero fee |
| **Euler** | 0% | ~19k | Limited | Lowest gas, limited assets |
| **Aave V3** | 0.05% | ~70k | $5-10B+ | Most liquid, highest gas |
| **Uniswap V3** | 0.05-1% | ~23k | Billions | Fee varies by pool |
| **Maker** | 0% | ~20k | Unlimited DAI | DAI only |

Source: [Jeiwan/flash-loans-comparison](https://github.com/Jeiwan/flash-loans-comparison)

### Provider Selection Matters

- **Gas costs differ 3x**: Balancer ~24k vs Aave ~70k
- **Fees differ**: 0% (Balancer, Maker) vs 0.05% (Aave)
- **Liquidity differs**: Maker unlimited DAI, Aave deep multi-asset, Balancer pool-limited
- **Token coverage differs**: Maker is DAI-only

## Solver Profit Rules (CIP-11)

[CIP-11](https://snapshot.org/#/cow.eth/proposal/0x16d8c681d52b24f1ccd854084e07a99fce6a7af1e25fd21ddae6534b411df870) and the [solver competition rules](https://docs.cow.fi/cow-protocol/reference/core/auctions/competition-rules) establish clear guidelines:

**Solvers are scored by surplus generated for users:**
```
score = surplus_to_users + protocol_fees
```

**Prohibited behaviors:**
- Illegal surplus shifts between orders
- Systematic buffer trading for solver profit
- Score inflation using tokens

**Business model:** Solvers are paid in COW tokens by the protocol for winning auctions. They should NOT accumulate tokens or extract surplus from settlements.

## Design Decisions

### Decision 1: Flash Loans Enable Better User Fills, Not Solver Profit

**Decision:** Any efficiency gained from flash loans manifests as better user fills (more surplus), not solver accumulation.

**Rationale:**
- Aligns with CIP-11 and solver competition rules
- Solvers are scored by user surplus, not their own profit
- Solver revenue comes from COW token rewards, not arbitrage extraction

### Decision 2: Model Providers Separately with Unified Interface

**Decision:** Implement concrete provider classes behind a common interface.

```python
class FlashLoanProvider(Protocol):
    """Abstract interface for flash loan providers."""

    def available_liquidity(self, token: str) -> int:
        """Max amount borrowable for token."""
        ...

    def fee_bps(self, token: str) -> int:
        """Fee in basis points (100 = 1%)."""
        ...

    def gas_overhead(self) -> int:
        """Estimated gas cost of flash loan mechanics."""
        ...
```

**Rationale:**
- Optimal provider depends on situation (token needed, amount, gas price)
- Provider selection can be part of optimization objective
- Future-proofing for new providers (Uniswap V4 zero-fee flash)

### Decision 3: Defer Implementation to Phase 4

**Decision:** Implement flash loans as part of unified optimization, not as a standalone feature.

**Rationale:**
- Flash loans are optimization enablers, not a liquidity source themselves
- Natural fit with Phase 4's joint optimization across CoW + AMM + splits
- Avoids two-phase approach that would be suboptimal
- CoW Protocol infrastructure already exists; no urgency

### Decision 4: Provider Priority Heuristic

**Decision:** For initial implementation, use this priority order:

1. **Balancer** (0% fee, low gas) - if liquidity sufficient
2. **Maker** (0% fee, unlimited) - if DAI needed
3. **Aave** (0.05% fee, high liquidity) - fallback for large amounts

**Rationale:**
- Minimizes cost (fee + gas) while ensuring liquidity availability
- Simple heuristic that can be refined with real-world data
- Covers most common scenarios (ETH, stablecoins, large amounts)

## Use Cases for Better User Fills

### 1. Arbitrage as Better Price

```
User sells 100 ETH, wants USDC
AMM A: 100 ETH → 250,000 USDC
AMM B: 100 ETH → 248,000 USDC (mispriced)

With flash loan:
1. Flash borrow tokens to exploit A/B price difference
2. Arbitrage profit flows to user as better execution price
3. User receives MORE than 250,000 USDC
```

### 2. Split Routing Enablement

```
User sells 1000 ETH (large order)
Single pool: significant price impact

With flash loan:
1. Borrow intermediate capital
2. Execute across multiple venues simultaneously
3. User gets better average price
```

### 3. Ring Trade Execution

```
Order A: ETH → USDC (no direct liquidity)
Order B: USDC → DAI
Order C: DAI → ETH

Flash loan provides bootstrap capital to execute the ring.
All three users get filled when they otherwise couldn't.
```

## Phase 4 Integration

Flash loans will be integrated into the unified optimizer as:

**Variables:**
- `flash_loan[provider, token]` = amount to borrow

**Constraints:**
- `borrowed_amount <= available_liquidity[provider, token]`
- `repayment = borrowed_amount + fee`
- All borrows repaid within settlement

**Objective contribution:**
- Flash loan costs (fees + gas) reduce solution score
- Benefits (better fills, enabled trades) increase user surplus

## References

- [CoW Protocol Flash Loan Router](https://github.com/cowprotocol/flash-loan-wrapper-solver)
- [CoW Protocol Flash Loan Docs](https://docs.cow.fi/cow-protocol/tutorials/cow-swap/flash-loans)
- [Solver Competition Rules](https://docs.cow.fi/cow-protocol/reference/core/auctions/competition-rules)
- [Flash Loan Gas Comparison](https://github.com/Jeiwan/flash-loans-comparison)
- [Instadapp Flash Loan Aggregator](https://docs.instadapp.io/flashloan/docs)
- [CIP-11 Social Consensus Rules](https://snapshot.org/#/cow.eth/proposal/0x16d8c681d52b24f1ccd854084e07a99fce6a7af1e25fd21ddae6534b411df870)
