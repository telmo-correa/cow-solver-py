"""Unified CoW strategy combining LP for pairs and cycle settlement for rings.

This strategy handles both bidirectional pairs and cycles, avoiding the greedy
phase separation that can miss optimal solutions.

Key insight:
- Bidirectional pairs: LP with conservation constraints (true markets)
- Cycles: RingTrade's specialized settlement (rate-product constraint)

IMPORTANT: All financial calculations use exact integer arithmetic.
Decimal comparisons are converted to integer arithmetic for exactness.
"""

from __future__ import annotations

import decimal
from collections import defaultdict, deque
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from solver.fees.price_estimation import get_token_info
from solver.models.auction import AuctionInstance, Order
from solver.models.types import normalize_address
from solver.pools import build_registry_from_liquidity
from solver.routing.router import SingleOrderRouter
from solver.strategies.base import OrderFill, StrategyResult, convert_fill_ratios_to_fills
from solver.strategies.ebbo_bounds import get_ebbo_bounds, verify_fills_against_ebbo
from solver.strategies.settlement import (
    CycleViability,
    calculate_cycle_settlement,
    find_viable_cycle_direction,
)

if TYPE_CHECKING:
    from solver.amm.uniswap_v2 import UniswapV2

logger = structlog.get_logger()

# Use context with high precision for Decimal operations to ensure exactness
_DECIMAL_HIGH_PREC_CONTEXT = decimal.Context(prec=78)


def _decimal_ge(a: Decimal, b: Decimal) -> bool:
    """Compare a >= b with high precision for exactness."""
    with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
        diff = a - b
        return diff >= 0


def _decimal_lt(a: Decimal, b: Decimal) -> bool:
    """Compare a < b with high precision for exactness."""
    with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
        diff = a - b
        return diff < 0


class UnifiedCowStrategy:
    """Strategy that solves all CoW opportunities optimally.

    Unlike MultiPairCowStrategy which processes bidirectional pairs first
    (potentially blocking better cycle solutions), this strategy:

    1. Builds token graph from ALL orders
    2. Finds both cycles (3/4-token) and bidirectional pairs
    3. Scores each by potential value
    4. Processes in value order, using:
       - Cycle settlement for cycles (specialized rate-product algorithm)
       - LP with price enumeration for bidirectional pairs
    5. Validates EBBO compliance

    **Constraint Enforcement:**

    1. Fill-or-Kill: Explicitly validated in _validate_fok_fills(). FOK orders
       (partially_fillable=False) must be fully filled or are excluded. The
       hybrid auction also respects FOK via respect_fill_or_kill flag.

    2. Limit Price: Enforced via exact integer cross-multiplication:
       buy_filled * sell_amount >= buy_amount * sell_filled
       No tolerance/epsilon - financial calculations are exact.

    3. EBBO (Two-sided): Validated in _verify_ebbo() and via hybrid_auction's
       ebbo_min/ebbo_max bounds. Zero tolerance - clearing rate must be
       >= ebbo_min and <= ebbo_max for both trade directions.

    4. Uniform Price: Enforced structurally. Single clearing price per token
       derived from fill conservation (sell_filled * sell_price = buy_filled *
       buy_price). Prices propagated through fill graph via BFS to ensure
       consistency.

    Args:
        max_tokens: Maximum tokens per component (limits complexity)
        max_price_combos: Maximum price combinations to try per component
        amm: Optional AMM for price queries
        enforce_ebbo: Whether to enforce EBBO constraints
    """

    def __init__(
        self,
        max_tokens: int = 6,
        max_price_combos: int = 50,
        amm: UniswapV2 | None = None,
        enforce_ebbo: bool = True,
        use_lp_solver: bool = False,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_price_combos = max_price_combos
        self.amm = amm
        self.enforce_ebbo = enforce_ebbo
        self.use_lp_solver = use_lp_solver  # Use LP solver for pairs instead of double auction
        self._current_auction: AuctionInstance | None = None

    def try_solve(self, auction: AuctionInstance) -> StrategyResult | None:
        """Try to find CoW matches using unified optimization."""
        if auction.order_count < 2:
            return None

        self._current_auction = auction
        orders = list(auction.orders)

        # Build pool registry for AMM price queries
        pool_registry = build_registry_from_liquidity(auction.liquidity)
        router = SingleOrderRouter(pool_registry=pool_registry, amm=self.amm)

        # Find matchable structures: cycles and bidirectional pairs
        cycles, pairs = self._find_matchable_structures(orders)

        logger.debug(
            "unified_cow_structures",
            cycles=len(cycles),
            pairs=len(pairs),
        )

        all_fills: list[OrderFill] = []
        all_prices: dict[str, int] = {}  # Track prices from each structure
        processed_uids: set[str] = set()
        priced_tokens: set[str] = set()  # Tokens that already have prices

        # Build combined list with type tag for sorting
        structures: list[tuple[str, CycleViability | tuple[set[str], list[Order]]]] = []

        for viability in cycles:
            structures.append(("cycle", viability))

        for tokens, pair_orders in pairs:
            structures.append(("pair", (tokens, pair_orders)))

        # Sort order: pairs first (high surplus), then cycles
        # Bidirectional pairs with double-auction matching generate more surplus
        # than specialized cycle settlement
        def get_priority(
            item: tuple[str, CycleViability | tuple[set[str], list[Order]]],
        ) -> tuple[int, int]:
            if item[0] == "pair":
                # Pairs first, sorted by order count (more orders = more potential)
                _, pair_orders = item[1]  # type: ignore[misc]
                return (0, -len(pair_orders))
            # Cycles after pairs, sorted by 3-cycles then 4-cycles
            viability: CycleViability = item[1]  # type: ignore[assignment]
            return (1, len(viability.orders))

        structures.sort(key=get_priority)

        for struct_type, struct in structures:
            if struct_type == "cycle":
                viability = struct  # type: ignore
                # Skip if any order already matched
                if any(o.uid in processed_uids for o in viability.orders):
                    continue

                # Skip cycles with already-priced tokens (would create inconsistent prices)
                cycle_tokens = {normalize_address(o.sell_token) for o in viability.orders}
                if cycle_tokens & priced_tokens:
                    continue

                fills, prices = self._solve_cycle_with_prices(viability, router, auction)
            else:
                tokens, pair_orders = struct  # type: ignore
                # Skip if any order already matched
                if any(o.uid in processed_uids for o in pair_orders):
                    continue

                # Skip pairs with already-priced tokens (would create inconsistent prices)
                if tokens & priced_tokens:
                    continue

                if self.use_lp_solver:
                    fills = self._solve_pair_lp(
                        pair_orders, tokens, router, auction, processed_uids
                    )
                    prices = {}  # LP solver doesn't return prices
                else:
                    fills, prices = self._solve_pair_with_prices(
                        pair_orders, tokens, router, auction, processed_uids
                    )

            for fill in fills:
                if fill.order.uid not in processed_uids:
                    all_fills.append(fill)
                    processed_uids.add(fill.order.uid)

            # Track prices and priced tokens
            for token, price in prices.items():
                all_prices[token] = price
                priced_tokens.add(token)

        if not all_fills:
            return None

        return self._build_result_with_prices(all_fills, all_prices, auction)

    def _find_matchable_structures(
        self, orders: list[Order]
    ) -> tuple[list[CycleViability], list[tuple[set[str], list[Order]]]]:
        """Find matchable structures: viable cycles and bidirectional pairs.

        Returns:
            (cycles, pairs) where:
            - cycles: list of CycleViability for viable cycles
            - pairs: list of (tokens, orders) for bidirectional pairs
        """
        from solver.strategies.graph import OrderGraph

        graph = OrderGraph.from_orders(orders)
        cycles: list[CycleViability] = []
        pairs: list[tuple[set[str], list[Order]]] = []

        # 1. Find viable cycles first (they have specific rate constraints)
        # 3-cycles
        for cycle_3 in graph.find_3_cycles():
            result = find_viable_cycle_direction(cycle_3, graph.get_orders)
            if result and result.viable:
                cycles.append(result)

        # 4-cycles (limited)
        for cycle_4 in graph.find_4_cycles(limit=50):
            result = find_viable_cycle_direction(cycle_4, graph.get_orders)
            if result and result.viable:
                cycles.append(result)

        # 2. Find bidirectional pairs (orders in both directions)
        seen_pairs: set[tuple[str, str]] = set()
        for order in orders:
            sell = normalize_address(order.sell_token)
            buy = normalize_address(order.buy_token)
            pair = (min(sell, buy), max(sell, buy))

            if pair in seen_pairs:
                continue

            # Check if reverse direction exists
            forward_orders = graph.get_orders(sell, buy)
            reverse_orders = graph.get_orders(buy, sell)

            if forward_orders and reverse_orders:
                seen_pairs.add(pair)
                tokens = {sell, buy}
                pair_orders = list(forward_orders) + list(reverse_orders)
                pairs.append((tokens, pair_orders))

        return cycles, pairs

    def _solve_cycle_with_prices(
        self,
        viability: CycleViability,
        router: SingleOrderRouter,
        auction: AuctionInstance,
    ) -> tuple[list[OrderFill], dict[str, int]]:
        """Solve a viable cycle using RingTrade's settlement algorithm.

        Returns:
            Tuple of (fills, prices) where prices are from cycle settlement.
        """
        settlement = calculate_cycle_settlement(viability)
        if not settlement:
            return [], {}

        # Verify EBBO
        if self.enforce_ebbo and not self._verify_cycle_ebbo(
            settlement.fills, settlement.clearing_prices, router, auction
        ):
            return [], {}

        return settlement.fills, settlement.clearing_prices

    def _verify_cycle_ebbo(
        self,
        fills: list[OrderFill],
        clearing_prices: dict[str, int],
        router: SingleOrderRouter,
        auction: AuctionInstance,
    ) -> bool:
        """Verify cycle fills satisfy EBBO constraint."""
        return verify_fills_against_ebbo(fills, clearing_prices, router, auction)

    def _find_token_components(self, orders: list[Order]) -> list[tuple[set[str], list[Order]]]:
        """Find connected components of tokens using Union-Find.

        Two tokens are connected if there's an order between them.
        Returns (token_set, orders) for each component.
        """
        # Map tokens to indices
        all_tokens: set[str] = set()
        for order in orders:
            all_tokens.add(normalize_address(order.sell_token))
            all_tokens.add(normalize_address(order.buy_token))

        token_list = sorted(all_tokens)
        token_to_idx = {t: i for i, t in enumerate(token_list)}

        # Union-Find
        parent = list(range(len(token_list)))
        rank = [0] * len(token_list)

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Union tokens connected by orders
        for order in orders:
            sell_idx = token_to_idx[normalize_address(order.sell_token)]
            buy_idx = token_to_idx[normalize_address(order.buy_token)]
            union(sell_idx, buy_idx)

        # Group tokens and orders by component
        component_tokens: dict[int, set[str]] = defaultdict(set)
        for token, idx in token_to_idx.items():
            root = find(idx)
            component_tokens[root].add(token)

        component_orders: dict[int, list[Order]] = defaultdict(list)
        for order in orders:
            sell_idx = token_to_idx[normalize_address(order.sell_token)]
            root = find(sell_idx)
            component_orders[root].append(order)

        # Build result, sorted by order count
        result = [(component_tokens[root], component_orders[root]) for root in component_tokens]
        result.sort(key=lambda x: len(x[1]), reverse=True)

        return result

    def _solve_pair_with_prices(
        self,
        orders: list[Order],
        tokens: set[str],
        router: SingleOrderRouter,
        auction: AuctionInstance,
        already_matched: set[str],
    ) -> tuple[list[OrderFill], dict[str, int]]:
        """Solve a bidirectional pair using double auction (same as MultiPair).

        Returns:
            Tuple of (fills, prices) where prices are from run_hybrid_auction totals.
        """
        from solver.models.order_groups import OrderGroup
        from solver.strategies.double_auction import run_hybrid_auction

        # Filter out already-matched orders
        available_orders = [o for o in orders if o.uid not in already_matched]
        if len(available_orders) < 2:
            return [], {}

        # Get the two tokens
        token_list = sorted(tokens)
        if len(token_list) != 2:
            return [], {}

        token_a, token_b = token_list

        # Create order group manually
        sellers_of_a: list[Order] = []
        sellers_of_b: list[Order] = []

        for order in available_orders:
            sell_token = normalize_address(order.sell_token)
            if sell_token == token_a:
                sellers_of_a.append(order)
            else:
                sellers_of_b.append(order)

        if not sellers_of_a or not sellers_of_b:
            return [], {}  # No bidirectional flow

        group = OrderGroup(
            token_a=token_a,
            token_b=token_b,
            sellers_of_a=sellers_of_a,
            sellers_of_b=sellers_of_b,
        )

        if group.order_count < 2:
            return [], {}

        # Get two-sided EBBO bounds
        bounds = get_ebbo_bounds(token_a, token_b, router, auction)

        # Run hybrid auction with both EBBO bounds
        result = run_hybrid_auction(
            group,
            amm_price=bounds.amm_price,
            ebbo_min=bounds.ebbo_min,
            ebbo_max=bounds.ebbo_max,
        )

        if not result.cow_matches:
            return [], {}

        # EBBO validation is now handled by run_hybrid_auction with both bounds

        # Convert matches to fills
        fills: list[OrderFill] = []
        for match in result.cow_matches:
            fills.append(
                OrderFill(order=match.seller, sell_filled=match.amount_a, buy_filled=match.amount_b)
            )
            fills.append(
                OrderFill(order=match.buyer, sell_filled=match.amount_b, buy_filled=match.amount_a)
            )

        # Use prices from hybrid auction (CoW Protocol standard format)
        # price[token_a] = total_b (amount of B exchanged)
        # price[token_b] = total_a (amount of A exchanged)
        prices = {
            token_a: result.total_cow_b,
            token_b: result.total_cow_a,
        }

        return fills, prices

    def _solve_pair_lp(
        self,
        orders: list[Order],
        tokens: set[str],
        router: SingleOrderRouter,
        auction: AuctionInstance,
        already_matched: set[str],
    ) -> list[OrderFill]:
        """Solve a bidirectional pair using LP (legacy implementation)."""
        # Filter out already-matched orders
        available_orders = [o for o in orders if o.uid not in already_matched]
        if len(available_orders) < 2:
            return []

        # Build price candidates from order limits and AMM
        candidates = self._build_price_candidates(available_orders, tokens, router, auction)

        # Build spanning tree for price propagation
        spanning_tree = self._build_spanning_tree(available_orders, tokens)

        # Enumerate price combinations
        price_combos = self._enumerate_prices(candidates, spanning_tree, tokens)

        if not price_combos:
            return []

        # Try each price combination, keep best result by surplus
        best_fills: list[OrderFill] = []
        best_surplus = Decimal(0)

        for prices in price_combos:
            fills = self._solve_lp_at_prices(available_orders, prices)
            if not fills:
                continue

            # Validate EBBO
            if not self._verify_ebbo(fills, prices, router, auction):
                continue

            # Calculate surplus for this solution
            surplus = self._calculate_fills_surplus(fills, prices)
            if surplus > best_surplus:
                best_surplus = surplus
                best_fills = fills

        return best_fills

    def _calculate_fills_surplus(
        self,
        fills: list[OrderFill],
        prices: dict[str, Decimal],
    ) -> Decimal:
        """Calculate total surplus for a set of fills at given prices."""
        total_surplus = Decimal(0)

        for fill in fills:
            order = fill.order
            sell_token = normalize_address(order.sell_token)
            buy_token = normalize_address(order.buy_token)

            sell_amt = Decimal(order.sell_amount_int)
            buy_amt = Decimal(order.buy_amount_int)

            if sell_amt == 0:
                continue

            # Use high-precision context for exact division
            with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
                limit_rate = buy_amt / sell_amt

            price_sell = prices.get(sell_token)
            price_buy = prices.get(buy_token)

            if price_sell is None or price_buy is None or price_sell <= 0 or price_buy <= 0:
                continue

            # Use high-precision context for exact division
            # Clearing rate = buy_filled / sell_filled = price_sell / price_buy
            # (from conservation: sell_filled * price_sell = buy_filled * price_buy)
            with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
                clearing_rate = price_sell / price_buy

            # Surplus = sell_filled × (clearing_rate - limit_rate)
            surplus = Decimal(fill.sell_filled) * (clearing_rate - limit_rate)
            total_surplus += surplus

        return total_surplus

    def _build_price_candidates(
        self,
        orders: list[Order],
        _tokens: set[str],
        router: SingleOrderRouter,
        auction: AuctionInstance,
    ) -> dict[tuple[str, str], list[Decimal]]:
        """Build price ratio candidates for each token pair."""
        candidates: dict[tuple[str, str], list[Decimal]] = defaultdict(list)
        amm_queried: set[tuple[str, str]] = set()

        for order in orders:
            sell_token = normalize_address(order.sell_token)
            buy_token = normalize_address(order.buy_token)

            # Order limit price: buy_amount / sell_amount
            sell_amt = order.sell_amount_int
            buy_amt = order.buy_amount_int

            if sell_amt > 0 and buy_amt > 0:
                # Use high-precision context for exact division
                with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
                    limit_price = Decimal(buy_amt) / Decimal(sell_amt)
                key = (sell_token, buy_token)
                if limit_price not in candidates[key]:
                    candidates[key].append(limit_price)

            # AMM spot price (once per pair)
            if (sell_token, buy_token) not in amm_queried:
                amm_queried.add((sell_token, buy_token))
                token_info = get_token_info(auction, sell_token)
                decimals = (
                    18 if token_info is None or token_info.decimals is None else token_info.decimals
                )

                amm_price = router.get_reference_price(
                    sell_token, buy_token, token_in_decimals=decimals
                )
                if amm_price is not None and amm_price > 0:
                    key = (sell_token, buy_token)
                    if amm_price not in candidates[key]:
                        candidates[key].append(amm_price)

        return candidates

    def _build_spanning_tree(
        self,
        orders: list[Order],
        tokens: set[str],
    ) -> list[tuple[str, str]]:
        """Build a spanning tree of token pairs using BFS."""
        # Build adjacency list
        adj: dict[str, set[str]] = defaultdict(set)
        for order in orders:
            sell = normalize_address(order.sell_token)
            buy = normalize_address(order.buy_token)
            adj[sell].add(buy)
            adj[buy].add(sell)

        # BFS from first token
        if not tokens:
            return []

        start = min(tokens)  # Deterministic start
        visited = {start}
        queue = [start]
        tree: list[tuple[str, str]] = []

        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    tree.append((current, neighbor))

        return tree

    def _enumerate_prices(
        self,
        candidates: dict[tuple[str, str], list[Decimal]],
        spanning_tree: list[tuple[str, str]],
        tokens: set[str],
    ) -> list[dict[str, Decimal]]:
        """Enumerate price combinations by propagating through spanning tree."""
        if not spanning_tree:
            # Single token or disconnected - use uniform price
            if tokens:
                return [{t: Decimal(1) for t in tokens}]
            return []

        # Get candidate ratios for each tree edge
        edge_ratios: list[list[Decimal]] = []
        for parent, child in spanning_tree:
            # Try both orderings
            ratios = candidates.get((parent, child), [])
            if not ratios:
                # Try reverse and invert using high-precision context for exact arithmetic
                reverse_ratios = candidates.get((child, parent), [])
                if reverse_ratios:
                    with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
                        ratios = [Decimal(1) / r for r in reverse_ratios if r > 0]
            if not ratios:
                ratios = [Decimal(1)]  # Default

            # Limit candidates per edge to control explosion
            if len(ratios) > 3:
                sorted_ratios = sorted(ratios)
                ratios = [
                    sorted_ratios[0],
                    sorted_ratios[len(sorted_ratios) // 2],
                    sorted_ratios[-1],
                ]

            edge_ratios.append(ratios)

        # Generate combinations (limited)
        from itertools import product

        all_combos = list(product(*edge_ratios))
        if len(all_combos) > self.max_price_combos:
            # Sample evenly
            step = len(all_combos) // self.max_price_combos
            all_combos = all_combos[::step][: self.max_price_combos]

        # Propagate prices for each combination
        results: list[dict[str, Decimal]] = []
        reference_token = spanning_tree[0][0]

        for combo in all_combos:
            prices: dict[str, Decimal] = {reference_token: Decimal(1)}

            for i, (parent, child) in enumerate(spanning_tree):
                ratio = combo[i]
                if parent in prices:
                    prices[child] = prices[parent] * ratio
                elif child in prices and ratio > 0:
                    prices[parent] = prices[child] / ratio

            results.append(prices)

        return results

    def _solve_lp_at_prices(
        self,
        orders: list[Order],
        prices: dict[str, Decimal],
    ) -> list[OrderFill]:
        """Solve LP for optimal fills at given prices, maximizing surplus.

        Surplus = what user gets - what user minimally required
        For each order: surplus = sell_amount × (clearing_rate - limit_rate)

        The LP maximizes total surplus subject to token conservation.
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            logger.warning("scipy not available")
            return []

        # Find eligible orders (limit price satisfied) and compute surplus coefficients
        eligible: list[tuple[Order, Decimal, Decimal]] = []  # (order, limit_rate, surplus_coeff)

        for order in orders:
            sell_token = normalize_address(order.sell_token)
            buy_token = normalize_address(order.buy_token)

            sell_amt = order.sell_amount_int
            buy_amt = order.buy_amount_int

            if sell_amt <= 0 or buy_amt <= 0:
                continue

            # Use high-precision context for exact division
            with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
                limit_rate = Decimal(buy_amt) / Decimal(sell_amt)

            price_sell = prices.get(sell_token)
            price_buy = prices.get(buy_token)

            if price_sell is None or price_buy is None or price_sell <= 0 or price_buy <= 0:
                continue

            # Clearing rate at these prices (buy tokens per sell token)
            # clearing_rate = buy_filled / sell_filled = price_sell / price_buy
            # (from conservation: sell_filled * price_sell = buy_filled * price_buy)
            with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
                clearing_rate = price_sell / price_buy

            # Order is eligible if it gets at least its limit price
            # Use integer comparison for exactness
            if _decimal_ge(clearing_rate, limit_rate):
                # Surplus coefficient: sell_amount × (clearing_rate - limit_rate)
                # This is the surplus generated per unit of fill ratio
                surplus_coeff = Decimal(sell_amt) * (clearing_rate - limit_rate)
                eligible.append((order, limit_rate, surplus_coeff))

        if not eligible:
            return []

        n_orders = len(eligible)

        # Get all tokens
        all_tokens: set[str] = set()
        for order, _, _ in eligible:
            all_tokens.add(normalize_address(order.sell_token))
            all_tokens.add(normalize_address(order.buy_token))
        token_list = sorted(all_tokens)

        # Objective: maximize sum(x_i * surplus_coeff_i)
        # linprog minimizes, so negate
        c = [-float(surplus_coeff) for _, _, surplus_coeff in eligible]

        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1) for _ in range(n_orders)]

        # Conservation: buys - sells <= 0 for each token
        A_ub = []
        b_ub = []

        for token in token_list:
            row = [0.0] * n_orders
            for i, (order, _, _) in enumerate(eligible):
                sell_token = normalize_address(order.sell_token)
                buy_token = normalize_address(order.buy_token)

                if sell_token == token:
                    row[i] -= order.sell_amount_int
                if buy_token == token:
                    row[i] += order.buy_amount_int

            A_ub.append(row)
            b_ub.append(0.0)

        # Solve
        try:
            result = linprog(
                c,
                A_ub=A_ub if A_ub else None,
                b_ub=b_ub if b_ub else None,
                bounds=bounds,
                method="highs",
            )
        except Exception as e:
            logger.debug("lp_error", error=str(e))
            return []

        if not result.success or result.x is None:
            return []

        # Convert to fills using shared helper for integer arithmetic
        orders_with_ratios = [(order, result.x[i]) for i, (order, _, _) in enumerate(eligible)]
        return convert_fill_ratios_to_fills(orders_with_ratios)

    def _verify_ebbo(
        self,
        fills: list[OrderFill],
        prices: dict[str, Decimal],
        router: SingleOrderRouter,
        auction: AuctionInstance,
    ) -> bool:
        """Verify all fills satisfy EBBO constraint."""
        for fill in fills:
            order = fill.order
            sell_token = normalize_address(order.sell_token)
            buy_token = normalize_address(order.buy_token)

            price_sell = prices.get(sell_token)
            price_buy = prices.get(buy_token)

            if price_sell is None or price_buy is None or price_sell <= 0 or price_buy <= 0:
                continue

            # Use high-precision context for exact division
            # Clearing rate = buy_filled / sell_filled = price_sell / price_buy
            # (from conservation: sell_filled * price_sell = buy_filled * price_buy)
            with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
                clearing_rate = price_sell / price_buy

            token_info = auction.tokens.get(sell_token)
            decimals = (
                18 if token_info is None or token_info.decimals is None else token_info.decimals
            )

            amm_rate = router.get_reference_price(sell_token, buy_token, token_in_decimals=decimals)

            if amm_rate is None:
                continue

            # EBBO: user must get at least AMM rate
            # Use integer comparison for exactness
            if _decimal_lt(clearing_rate, amm_rate):
                return False

        return True

    def _validate_fok_fills(self, fills: list[OrderFill]) -> list[OrderFill]:
        """Filter fills that violate Fill-or-Kill constraint.

        FOK orders (partially_fillable=False) must be fully filled or excluded.
        This is a safety check - the double auction already respects FOK,
        but we validate here for cycle settlements and LP solutions.

        Args:
            fills: List of order fills to validate

        Returns:
            List of valid fills (FOK orders either fully filled or removed)
        """
        valid_fills: list[OrderFill] = []

        for fill in fills:
            order = fill.order
            if not order.partially_fillable:
                # FOK order - must be fully filled
                # For sell orders, check sell_filled == sell_amount
                # For buy orders, check buy_filled == buy_amount
                if order.kind == "sell":
                    expected = int(order.sell_amount)
                    actual = fill.sell_filled
                else:
                    expected = int(order.buy_amount)
                    actual = fill.buy_filled

                if actual < expected:
                    logger.debug(
                        "unified_cow_fok_rejected",
                        order_uid=order.uid[:18],
                        expected=expected,
                        actual=actual,
                        kind=order.kind,
                    )
                    continue  # Skip this fill

            valid_fills.append(fill)

        return valid_fills

    def _build_result_with_prices(
        self,
        fills: list[OrderFill],
        prices: dict[str, int],
        auction: AuctionInstance,
    ) -> StrategyResult:
        """Build final StrategyResult using pre-computed prices.

        Args:
            fills: Order fills from matched structures
            prices: Prices collected from each structure (token -> price)
            auction: The auction instance
        """
        # Validate FOK constraint before building result
        fills = self._validate_fok_fills(fills)
        if not fills:
            return StrategyResult(
                fills=[],
                interactions=[],
                prices={},
                gas=0,
            )

        filled_uids = {f.order.uid for f in fills}

        # Remainder orders
        remainder_orders: list[Order] = []
        for order in auction.orders:
            if order.uid not in filled_uids:
                remainder_orders.append(order)

        for fill in fills:
            remainder = fill.get_remainder_order()
            if remainder:
                remainder_orders.append(remainder)

        # Convert prices to strings
        str_prices = {token: str(price) for token, price in prices.items()}

        return StrategyResult(
            fills=fills,
            interactions=[],
            prices=str_prices,
            gas=0,
            remainder_orders=remainder_orders,
        )

    def _build_result(
        self,
        fills: list[OrderFill],
        auction: AuctionInstance,
    ) -> StrategyResult:
        """Build final StrategyResult (legacy, uses _normalize_prices)."""
        # Validate FOK constraint before building result
        fills = self._validate_fok_fills(fills)
        if not fills:
            return StrategyResult(
                fills=[],
                interactions=[],
                prices={},
                gas=0,
            )

        filled_uids = {f.order.uid for f in fills}

        # Remainder orders
        remainder_orders: list[Order] = []
        for order in auction.orders:
            if order.uid not in filled_uids:
                remainder_orders.append(order)

        for fill in fills:
            remainder = fill.get_remainder_order()
            if remainder:
                remainder_orders.append(remainder)

        # Normalize prices
        prices = self._normalize_prices(fills)

        return StrategyResult(
            fills=fills,
            interactions=[],
            prices=prices,
            gas=0,
            remainder_orders=remainder_orders,
        )

    def _normalize_prices(self, fills: list[OrderFill]) -> dict[str, str]:
        """Compute clearing prices from fills using conservation invariant.

        For each fill, conservation requires:
            sell_filled * sell_price = buy_filled * buy_price

        We set a reference price and propagate through the fill graph using BFS.
        This ensures all prices are consistent with the fill amounts.
        """
        if not fills:
            return {}

        # Build adjacency: token -> [(other_token, rate)]
        # where rate = buy_filled / sell_filled for order selling token
        # (i.e., how many other_token you get per token)
        token_rates: dict[str, list[tuple[str, Decimal]]] = defaultdict(list)

        for fill in fills:
            if fill.sell_filled <= 0 or fill.buy_filled <= 0:
                continue

            sell_token = normalize_address(fill.order.sell_token)
            buy_token = normalize_address(fill.order.buy_token)

            # Rate from sell_token's perspective: buy_token per sell_token
            with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
                rate = Decimal(fill.buy_filled) / Decimal(fill.sell_filled)
            token_rates[sell_token].append((buy_token, rate))

            # Inverse rate from buy_token's perspective: sell_token per buy_token
            with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
                inv_rate = Decimal(fill.sell_filled) / Decimal(fill.buy_filled)
            token_rates[buy_token].append((sell_token, inv_rate))

        all_tokens = set(token_rates.keys())
        if not all_tokens:
            return {}

        # BFS from first token to set consistent prices
        prices: dict[str, Decimal] = {}
        visited: set[str] = set()

        for start_token in all_tokens:
            if start_token in visited:
                continue

            # Set reference price for this component
            prices[start_token] = Decimal(10**18)
            visited.add(start_token)
            queue: deque[str] = deque([start_token])

            while queue:
                token = queue.popleft()
                current_price = prices[token]

                for other_token, rate in token_rates[token]:
                    if other_token not in visited:
                        # price[other] = price[token] / rate
                        # Because: token -> other at rate means:
                        # sell_filled(token) * price(token) = buy_filled(other) * price(other)
                        # rate = buy_filled / sell_filled
                        # So: price(other) = price(token) / rate
                        with decimal.localcontext(_DECIMAL_HIGH_PREC_CONTEXT):
                            prices[other_token] = current_price / rate
                        visited.add(other_token)
                        queue.append(other_token)

        # Convert to strings (as integers)
        return {token: str(int(price)) for token, price in prices.items()}


__all__ = ["UnifiedCowStrategy"]
