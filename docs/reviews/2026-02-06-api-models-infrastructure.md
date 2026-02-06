# API, Models & Infrastructure Review

**Date:** 2026-02-06
**Reviewer:** Claude Opus 4.6
**Scope:** solver/api/, solver/models/, solver/pools/, solver/routing/, solver/fees/, solver/constants.py

---

## Remediation Status

| ID | Severity | Issue | Status |
|----|----------|-------|--------|
| H1 | High | No timeout on solver execution | **FIXED** — 30s timeout via `asyncio.wait_for` |
| M2 | Medium | No request size limit | **FIXED** — 10MB limit middleware |
| M3 | Medium | No rate limiting | **DOCUMENTED** — Infrastructure-layer concern |
| M4 | Medium | Liquidity model unbounded extra fields | **DEFERRED** — Required by downstream parsers |
| M5 | Medium | No sell_token != buy_token validation | **FIXED** — Pydantic `model_validator` |
| M6 | Medium | No non-zero amount validation | **FIXED** — Pydantic `field_validator` |
| M7 | Medium | LimitOrderPool no amount validation | **FIXED** — `__post_init__` positive check |
| M8 | Medium | PriceEstimator rebuilds registry | **DEFERRED** — Low-traffic path |
| M9 | Medium | pool_count only counts V2 | **DEFERRED** — Acceptable for current usage |
| L10 | Low | normalize_address validation | **DEFERRED** — Performance concern |
| L11 | Low | RPC URL API key in logs | **FIXED** — Hostname-only logging |
| L12 | Low | Uint256 string form | **DEFERRED** — Would break backward compat |
| L13 | Low | Missing enum validation in FeePolicy.kind | **DEFERRED** — Protocol may add new kinds |
| L14 | Low | Interaction discriminator defaults | **DEFERRED** — Protocol design choice |
| L15 | Low | OrderGroup cached_property | **DEFERRED** — Lists never mutated |
| L16 | Low | Multi-hop forward verification | **DEFERRED** — 1 wei difference |
| L17 | Low | TokenGraph bidirectional edges | **DEFERRED** — Filtered later |
| L18 | Low | Fee calc multi-hop gas | **DEFERRED** — Caller responsibility |
| L19 | Low | Legacy pool_finder fallback | **DEFERRED** — Too many test dependencies |
| L20 | Low | Missing pool validation on add | **FIXED** — V2 replacement logging added |
| L21 | Low | Constants mainnet-only | **DEFERRED** — Multi-chain separate feature |
| — | — | /health endpoint | **FIXED** — Returns `scipy_available` |

---

## HIGH Issues

### 1. No timeout on solver execution

**File:** `solver/api/endpoints.py:93`

`solver_instance.solve(auction)` runs with no timeout. The `AuctionInstance.deadline` field is never used to enforce a timeout. A pathological auction input could cause the solver to spin indefinitely, blocking the endpoint.

---

## MEDIUM Issues

### 2. No request size limit (DoS vector)

**File:** `solver/api/main.py:15-21`

No request body size limit configured. An attacker could send an enormous JSON payload that consumes all server memory before Pydantic validation runs.

### 3. No rate limiting

**File:** `solver/api/endpoints.py:36-113`

The `/solve` endpoint has no rate limiting. Solving is CPU-intensive; a single client could exhaust resources.

### 4. `Liquidity` model unbounded extra fields

**File:** `solver/models/auction.py:232`

`extra="allow"` means any arbitrary JSON keys are stored. Thousands of extra fields with long values = memory exhaustion.

### 5. No validation that `sell_token != buy_token`

**File:** `solver/models/auction.py:114`

An order where `sell_token == buy_token` passes all validation. Downstream grouping would place it inconsistently.

### 6. No validation that amounts are non-zero

**File:** `solver/models/auction.py:120-121`

`sell_amount="0"` and `buy_amount="0"` are valid Uint256. Causes division by zero in `limit_price`, zero ratios in EBBO checks.

### 7. `LimitOrderPool` has no amount validation

**File:** `solver/pools/limit_order.py:15-45`

`maker_amount` and `taker_amount` can be zero or negative. `taker_amount=0` causes division by zero in swap simulation.

### 8. `PoolBasedPriceEstimator` rebuilds registry every call

**File:** `solver/fees/price_estimation.py:231`

`build_registry_from_liquidity(auction.liquidity)` is called on every `_estimate_via_pools` invocation. Should be built once per auction.

### 9. `pool_count` only counts V2 pools

**File:** `solver/fees/price_estimation.py:233`

`registry.pool_count` only counts V2 pools. If the auction only has V3 or Balancer liquidity, the estimator bails out even though usable liquidity exists.

### 10. `normalize_address` without `validate=True` accepts any string

**File:** `solver/models/types.py:82-106`

`normalize_address("hello world")` returns `"0xhello world"` without error. Internal code calling `normalize_address` on non-validated strings could produce garbage addresses that silently propagate.

---

## LOW Issues

### 11. RPC URL partially logged (API key leak)

**File:** `solver/solver.py:497`

`logger.info("v3_support_enabled", rpc_url=rpc_url[:50] + "...")` logs first 50 chars of the RPC URL. If it contains an API key (common for Infura/Alchemy), this leaks to logs.

### 12. `Uint256` preserves original string form

**File:** `solver/models/types.py:53`

`"0123"` remains `"0123"` rather than being normalized to `"123"`. Since Uint256 values are used as dict keys in `prices`, different string representations create separate entries.

### 13. `FeePolicy.kind` has no enum validation

**File:** `solver/models/auction.py:91`

Any string value is accepted for fee policy kind. Invalid kinds pass silently.

### 14. `Interaction` discriminator defaults to "custom"

**File:** `solver/models/solution.py:103-107`

A malformed dict without a `kind` field is silently parsed as `CustomInteraction`.

### 15. `OrderGroup` uses `cached_property` on mutable dataclass

**File:** `solver/models/order_groups.py:51-79`

Cached properties become stale if underlying lists are mutated. No enforcement prevents this.

### 16. Legacy `pool_finder` code path may bypass registry

**File:** `solver/routing/router.py:291-296`

After pathfinding finds no paths, fallback to `pool_finder` returns pools unknown to the registry and cache.

### 17. Multi-hop buy order forward verification inconsistency

**File:** `solver/routing/multihop.py:206-224`

`RoutingResult.amount_in` uses backwards-computed `amounts[0]`, but hops use forward-verified `actual_amounts`. If rounding causes difference, the trade's `amount_in` won't match the first hop.

### 18. `TokenGraph` adds bidirectional edges for unidirectional limit orders

**File:** `solver/routing/pathfinding.py:211-215`

Limit orders are unidirectional but the graph adds bidirectional edges. Invalid directions filtered later but wastes computation.

### 19. Fee calculation doesn't account for multi-hop gas correctly

**File:** `solver/fees/calculator.py:184`

Caller is responsible for providing correct gas estimate. No validation that the estimate is reasonable.

### 20. Constants are mainnet-only

**File:** `solver/constants.py:58-61`

Token addresses hardcoded for mainnet. No per-chain mapping exists.

### 21. `parse_limit_order` silently returns None on parse failure

**File:** `solver/pools/limit_order.py:120`

Broad `except` catches all parse errors silently with no logging. Makes it difficult to diagnose missing liquidity.

### 22. `PoolRegistry` silently replaces V2 pools for same token pair

**File:** `solver/pools/registry.py:124`

No logging or warning when a V2 pool is replaced. In an auction with multiple V2 pools for the same pair, only the last survives.
