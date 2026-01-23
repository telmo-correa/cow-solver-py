"""Base protocol and data structures for solution strategies.

Uses SafeInt for arithmetic operations to prevent:
- Underflow when subtracting fees from executed amounts
- Division by zero in fill ratio calculations
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import structlog

from solver.fees import DEFAULT_FEE_CALCULATOR, FeeCalculator
from solver.models.auction import AuctionInstance, Order
from solver.models.solution import Interaction, Solution, Trade, TradeKind
from solver.models.types import normalize_address
from solver.safe_int import S, Underflow

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class PriceWorsened(ValueError):
    """Raised when a price change violates an order's limit price.

    When combining strategy results, the final clearing prices must satisfy
    all filled orders' limit constraints. A fill is valid if the clearing
    price gives at least as good a rate as the order's limit price.
    """

    pass


@dataclass
class OrderFill:
    """Record of how much of an order was filled by a strategy.

    Tracks the amounts filled on both sides of the trade, which enables:
    - Computing executed amounts for trades
    - Creating remainder orders for unfilled portions
    - Combining fills from multiple strategies

    Attributes:
        order: The original order being filled
        sell_filled: Amount of sell token used/transferred
        buy_filled: Amount of buy token received
    """

    order: Order
    sell_filled: int
    buy_filled: int

    @property
    def is_complete(self) -> bool:
        """Check if the order is completely filled.

        For sell orders: complete when sell_filled == sell_amount
        For buy orders: complete when buy_filled == buy_amount
        """
        if self.order.is_sell_order:
            return self.sell_filled >= self.order.sell_amount_int
        else:
            return self.buy_filled >= self.order.buy_amount_int

    @property
    def executed_amount(self) -> int:
        """Get the executed amount for the trade.

        Per CoW Protocol:
        - For sell orders: executed_amount = sell_filled
        - For buy orders: executed_amount = buy_filled
        """
        if self.order.is_sell_order:
            return self.sell_filled
        else:
            return self.buy_filled

    @property
    def fill_ratio(self) -> float:
        """Get the fill ratio as a decimal (0.0 to 1.0).

        For sell orders: sell_filled / sell_amount
        For buy orders: buy_filled / buy_amount
        """
        if self.order.is_sell_order:
            total = self.order.sell_amount_int
            return self.sell_filled / total if total > 0 else 0.0
        else:
            total = self.order.buy_amount_int
            return self.buy_filled / total if total > 0 else 0.0

    def get_remainder_order(self) -> Order | None:
        """Create a synthetic order for the unfilled portion.

        Returns None if the order is completely filled.

        The remainder order has:
        - New derived UID (hash of original UID + ":remainder")
        - Reduced sell_amount/buy_amount based on what's left
        - Adjusted buy_amount to ensure TOTAL requirement is met
        - original_uid field tracking the parent order for fill merging

        Note: CoW Protocol uses total-based limits, not per-unit limits.
        If original order wanted 5000 USDC for 2 WETH and got 3000 USDC for 1 WETH,
        the remainder needs 2000 USDC for 1 WETH (to meet 5000 total).
        """
        if self.is_complete:
            return None

        original_sell = self.order.sell_amount_int
        original_buy = self.order.buy_amount_int

        remaining_sell = original_sell - self.sell_filled
        remaining_buy = original_buy - self.buy_filled

        if remaining_sell <= 0 or remaining_buy <= 0:
            return None

        # Generate a new UID for the remainder order
        # Derived deterministically from original UID so it's reproducible
        remainder_uid = self._derive_remainder_uid(self.order.uid)

        # Track the original UID for fill merging
        # If this order is already a remainder, preserve the root original_uid
        original_uid = self.order.original_uid or self.order.uid

        # Create a copy with reduced amounts, new UID, and original_uid tracking
        # We use model_copy to preserve all other fields
        return self.order.model_copy(
            update={
                "uid": remainder_uid,
                "sell_amount": str(remaining_sell),
                "buy_amount": str(remaining_buy),
                "full_sell_amount": str(remaining_sell),
                "full_buy_amount": str(remaining_buy),
                "original_uid": original_uid,
            }
        )

    @staticmethod
    def _derive_remainder_uid(original_uid: str) -> str:
        """Derive a new UID for a remainder order.

        Uses SHA-256 hash of original UID + ":remainder" to create a
        deterministic but unique UID for the remainder portion.

        Args:
            original_uid: The original order's UID (hex string with 0x prefix)

        Returns:
            A new UID in the same format (0x + 112 hex chars = 56 bytes)
        """
        # Hash the original UID with a suffix to derive the remainder UID
        hash_input = f"{original_uid}:remainder".encode()
        hash_bytes = hashlib.sha256(hash_input).digest()

        # UID is 56 bytes (112 hex chars), SHA-256 gives 32 bytes
        # Extend by hashing again with different suffix
        hash_input2 = f"{original_uid}:remainder:ext".encode()
        hash_bytes2 = hashlib.sha256(hash_input2).digest()

        # Combine first 32 bytes + first 24 bytes of second hash = 56 bytes
        combined = hash_bytes + hash_bytes2[:24]
        return "0x" + combined.hex()


@dataclass
class StrategyResult:
    """Result of a strategy's attempt to solve an auction (partial or complete).

    Strategies return what they filled and what's left unfilled, enabling
    composition of partial results from multiple strategies.

    For example, CowMatchStrategy might partially fill two orders via CoW,
    then AmmRoutingStrategy fills the remainder through AMM pools.

    Attributes:
        fills: List of order fills (partial or complete)
        interactions: AMM interactions needed for the fills
        prices: Clearing prices for tokens involved in the fills
        gas: Gas estimate for the interactions
        remainder_orders: Synthetic orders representing unfilled portions
        fee_calculator: Optional custom fee calculator for this result.
                       Used for pool-based price estimation when reference prices are missing.
    """

    fills: list[OrderFill] = field(default_factory=list)
    interactions: list[Interaction] = field(default_factory=list)
    prices: dict[str, str] = field(default_factory=dict)
    gas: int = 0
    remainder_orders: list[Order] = field(default_factory=list)
    fee_calculator: FeeCalculator | None = None

    @property
    def is_complete(self) -> bool:
        """True if all input orders were completely filled."""
        return len(self.remainder_orders) == 0

    @property
    def has_fills(self) -> bool:
        """True if at least one order was (partially) filled."""
        return len(self.fills) > 0

    def build_solution(
        self,
        solution_id: int = 0,
        auction: AuctionInstance | None = None,
        fee_calculator: FeeCalculator | None = None,
    ) -> Solution:
        """Convert this result to a Solution for the solver response.

        For limit orders, calculates the solver-determined fee based on gas cost.
        The fee is taken from the sell token and must be included in the trade.

        Fee calculation (matching Rust baseline solver):
            fee = gas_cost_wei * 1e18 / reference_price

        Args:
            solution_id: ID to assign to the solution
            auction: The auction instance (needed for fee calculation on limit orders)
            fee_calculator: Optional custom fee calculator. Uses default if not provided.
                           If not provided, uses self.fee_calculator if set.

        Returns:
            A Solution object ready for the SolverResponse
        """
        # Priority: explicit parameter > stored in result > default
        calculator = fee_calculator or self.fee_calculator or DEFAULT_FEE_CALCULATOR
        trades = []
        # For limit orders with fees, we need to adjust prices by the fee ratio
        # Rust scales both clearing prices so they're based on executed amount, not swap amount
        adjusted_prices = dict(self.prices)

        for fill in self.fills:
            order = fill.order
            order_uid = getattr(order, "original_uid", None) or order.uid

            # Calculate fee for limit orders using the fee calculator
            fee_result = calculator.calculate_solver_fee(order, self.gas, auction)

            # Handle fee calculation errors
            if fee_result.is_error:
                logger.warning(
                    "fee_calculation_error_skipping_trade",
                    order_uid=order_uid[:18] + "...",
                    error=fee_result.error.value if fee_result.error else "unknown",
                    detail=fee_result.error_detail,
                )
                continue  # Skip this trade

            # If fee is required, validate and apply it
            if fee_result.requires_fee:
                fee = fee_result.fee
                assert fee is not None  # Guaranteed by requires_fee

                # Validate fee doesn't exceed executed amount (for sell orders)
                if order.is_sell_order:
                    validation = calculator.validate_fee_against_amount(
                        fee, fill.executed_amount, is_sell_order=True
                    )
                    if validation.is_error:
                        logger.warning(
                            "fee_validation_error_skipping_trade",
                            order_uid=order_uid[:18] + "...",
                            fee=fee,
                            executed=fill.executed_amount,
                            error=validation.error.value if validation.error else "unknown",
                        )
                        continue  # Skip this trade
                    # Use validated fee (may be capped if config allows)
                    fee = validation.fee
                    assert fee is not None
                    # Use SafeInt for explicit underflow protection
                    # Note: validate_fee_against_amount should prevent this, so if it
                    # triggers, there's a bug in the validation logic
                    try:
                        executed = (S(fill.executed_amount) - S(fee)).value
                    except Underflow:
                        logger.critical(
                            "fee_subtraction_underflow_validation_bug",
                            order_uid=order_uid[:18] + "...",
                            fee=fee,
                            executed_amount=fill.executed_amount,
                            reason="This indicates a bug in validate_fee_against_amount",
                        )
                        continue  # Skip this trade

                    # Adjust clearing prices by fee ratio to match Rust behavior.
                    # Rust scales both prices so they're based on executedAmount,
                    # not the full swap amount. This maintains the exchange rate
                    # but uses fee-adjusted values.
                    # price[buy_token] = executedAmount (for sell orders)
                    # price[sell_token] = outputAmount * (executedAmount / inputAmount)
                    original_amount = fill.executed_amount
                    if original_amount > 0:
                        sell_token = normalize_address(order.sell_token)
                        buy_token = normalize_address(order.buy_token)
                        if buy_token in adjusted_prices and sell_token in adjusted_prices:
                            # Scale both prices by executed/original ratio
                            sell_price = int(adjusted_prices[sell_token])
                            # New prices maintain the same ratio but scaled down
                            adjusted_prices[buy_token] = str(executed)
                            adjusted_prices[sell_token] = str(
                                sell_price * executed // original_amount
                            )
                else:
                    # For buy orders, fee is added to the sell side
                    executed = fill.executed_amount

                trades.append(
                    Trade(
                        kind=TradeKind.FULFILLMENT,
                        order=order_uid,
                        executedAmount=str(executed),
                        fee=str(fee),
                    )
                )
            else:
                # Market orders or zero fee: no fee in trade
                trades.append(
                    Trade(
                        kind=TradeKind.FULFILLMENT,
                        order=order_uid,
                        executedAmount=str(fill.executed_amount),
                    )
                )

        return Solution(
            id=solution_id,
            prices=adjusted_prices,
            trades=trades,
            interactions=self.interactions,
            gas=self.gas,
        )

    @staticmethod
    def combine(results: list[StrategyResult]) -> StrategyResult:
        """Combine multiple strategy results into one.

        Merges fills, interactions, and prices from all results.
        Fills for the same order are merged into a single fill.
        The remainder_orders come from the last result (representing
        what's still unfilled after all strategies).

        Price handling:
        - Later strategies' prices override earlier ones
        - After combining, validates all fills satisfy their order's limit price
        - Raises PriceWorsened if any fill violates its limit at final prices

        Args:
            results: List of StrategyResults to combine

        Returns:
            A combined StrategyResult

        Raises:
            PriceWorsened: If final prices violate any order's limit
        """
        if not results:
            return StrategyResult()

        # Merge fills by order UID
        fills_by_uid: dict[str, OrderFill] = {}
        combined_interactions: list[Interaction] = []
        combined_prices: dict[str, str] = {}
        total_gas = 0

        for result in results:
            for fill in result.fills:
                # Use original_uid for merging if this is a remainder order fill
                # This ensures fills for the same logical order are combined
                uid = fill.order.original_uid or fill.order.uid
                if uid in fills_by_uid:
                    # Merge with existing fill - add amounts
                    existing = fills_by_uid[uid]
                    fills_by_uid[uid] = OrderFill(
                        order=existing.order,  # Keep original order (not remainder)
                        sell_filled=existing.sell_filled + fill.sell_filled,
                        buy_filled=existing.buy_filled + fill.buy_filled,
                    )
                else:
                    fills_by_uid[uid] = fill

            combined_interactions.extend(result.interactions)

            # Merge prices - later strategies override earlier ones
            for token, price in result.prices.items():
                if token in combined_prices and combined_prices[token] != price:
                    logger.debug(
                        "price_override_in_combine",
                        token=token[-8:],
                        old_price=combined_prices[token],
                        new_price=price,
                    )
                combined_prices[token] = price

            total_gas += result.gas

        # Validate all fills satisfy their limit prices
        StrategyResult._validate_fills_satisfy_limits(list(fills_by_uid.values()))

        # Remainder orders come from the last result
        remainder_orders = results[-1].remainder_orders if results else []

        # Fee calculator: use the last result's calculator (if any)
        # This ensures pool-based price estimation is available for final solution
        fee_calc = None
        for result in reversed(results):
            if result.fee_calculator is not None:
                fee_calc = result.fee_calculator
                break

        return StrategyResult(
            fills=list(fills_by_uid.values()),
            interactions=combined_interactions,
            prices=combined_prices,
            gas=total_gas,
            remainder_orders=remainder_orders,
            fee_calculator=fee_calc,
        )

    @staticmethod
    def _validate_fills_satisfy_limits(fills: list[OrderFill]) -> None:
        """Validate that all fills satisfy their order's limit price.

        For each fill, checks that the ACTUAL fill rate (buy_filled/sell_filled)
        is at least as good as the order's limit price.

        For sell orders: buy_filled/sell_filled >= buy_amount/sell_amount
        For buy orders: sell_filled/buy_filled <= sell_amount/buy_amount

        This validates the actual execution, not the clearing prices. Clearing
        prices are for accounting; the fill amounts are what users actually get.

        Args:
            fills: List of order fills to validate

        Raises:
            PriceWorsened: If any fill violates its order's limit
        """
        for fill in fills:
            order = fill.order

            # Order's limit amounts
            sell_amount = order.sell_amount_int
            buy_amount = order.buy_amount_int

            if sell_amount == 0 or fill.sell_filled == 0:
                continue

            # For a sell order: user sells X, wants at least Y
            # Actual rate: buy_filled / sell_filled
            # Limit rate: buy_amount / sell_amount
            # Constraint: actual >= limit (get at least as much as requested per unit)
            # Rearranged: buy_filled * sell_amount >= buy_amount * sell_filled

            # For a buy order: user wants Y, pays at most X
            # Actual rate: sell_filled / buy_filled
            # Limit rate: sell_amount / buy_amount
            # Constraint: actual <= limit (pay at most as requested per unit)
            # Rearranged: sell_filled * buy_amount <= sell_amount * buy_filled

            if order.is_sell_order:
                # Sell order: actual rate must be >= limit rate
                # buy_filled * sell_amount >= buy_amount * sell_filled
                if fill.buy_filled * sell_amount < buy_amount * fill.sell_filled:
                    actual_rate = fill.buy_filled / fill.sell_filled
                    limit_rate = buy_amount / sell_amount
                    logger.error(
                        "fill_violates_limit",
                        order_uid=order.uid[:18] + "...",
                        order_kind="sell",
                        sell_filled=fill.sell_filled,
                        buy_filled=fill.buy_filled,
                        limit_rate=limit_rate,
                        actual_rate=actual_rate,
                    )
                    raise PriceWorsened(
                        f"Sell order {order.uid[:18]}... limit violated: "
                        f"actual rate {actual_rate:.6f} < limit {limit_rate:.6f}"
                    )
            else:
                # Buy order: actual rate must be <= limit rate
                # sell_filled * buy_amount <= sell_amount * buy_filled
                if (
                    buy_amount > 0
                    and fill.buy_filled > 0
                    and fill.sell_filled * buy_amount > sell_amount * fill.buy_filled
                ):
                    actual_rate = fill.sell_filled / fill.buy_filled
                    limit_rate = sell_amount / buy_amount
                    logger.error(
                        "fill_violates_limit",
                        order_uid=order.uid[:18] + "...",
                        order_kind="buy",
                        sell_filled=fill.sell_filled,
                        buy_filled=fill.buy_filled,
                        limit_rate=limit_rate,
                        actual_rate=actual_rate,
                    )
                    raise PriceWorsened(
                        f"Buy order {order.uid[:18]}... limit violated: "
                        f"actual rate {actual_rate:.6f} > limit {limit_rate:.6f}"
                    )


class SolutionStrategy(Protocol):
    """Protocol for solution strategies.

    Each strategy attempts to solve an auction (or part of it). Strategies
    return a StrategyResult containing:
    - fills: What was matched/filled
    - remainder_orders: What's left for subsequent strategies

    This enables composition: CowMatchStrategy might partially fill orders,
    then AmmRoutingStrategy handles the remainder.

    Strategies are tried in priority order (e.g., CoW matching before AMM
    routing) since CoW matches are better for users (no AMM fees).
    """

    def try_solve(self, auction: AuctionInstance) -> StrategyResult | None:
        """Attempt to solve the auction.

        Args:
            auction: The auction to solve

        Returns:
            A StrategyResult if this strategy can handle the auction
            (even partially), None otherwise.
        """
        ...
