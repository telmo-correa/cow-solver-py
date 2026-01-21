"""Data-driven matching rules for CoW order matching.

This module defines the constraints and execution formulas for matching
different combinations of order types (sell/buy × sell/buy).

The rules are expressed as data structures rather than code, making them:
- Easy to audit and verify
- Self-documenting
- Testable in isolation
- Extensible for new order types
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import structlog

logger = structlog.get_logger()


class OrderAmounts(NamedTuple):
    """Parsed order amounts for CoW matching calculations.

    Attributes:
        sell_a: Order A's sell amount
        buy_a: Order A's minimum buy amount
        sell_b: Order B's sell amount
        buy_b: Order B's minimum buy amount
        a_is_sell: True if A is a sell order
        b_is_sell: True if B is a sell order
    """

    sell_a: int
    buy_a: int
    sell_b: int
    buy_b: int
    a_is_sell: bool
    b_is_sell: bool


class MatchType(Enum):
    """Order type combinations for matching."""

    SELL_SELL = (True, True)
    SELL_BUY = (True, False)
    BUY_SELL = (False, True)
    BUY_BUY = (False, False)

    @classmethod
    def from_orders(cls, a_is_sell: bool, b_is_sell: bool) -> "MatchType":
        """Get match type from order flags."""
        return cls((a_is_sell, b_is_sell))


@dataclass(frozen=True)
class Constraint:
    """A constraint that must hold for a match to be valid.

    Attributes:
        description: Human-readable description for logging
        check: Function that returns True if constraint is satisfied
    """

    description: str
    check: Callable[[OrderAmounts], bool]

    def is_satisfied(self, amounts: OrderAmounts) -> bool:
        """Check if this constraint is satisfied."""
        return self.check(amounts)


@dataclass(frozen=True)
class ExecutionAmounts:
    """Execution amounts for a matched pair.

    These are the actual amounts that flow in the settlement:
    - exec_sell_a: Amount A sells (token X)
    - exec_buy_a: Amount A receives (token Y)
    - exec_sell_b: Amount B sells (token Y)
    - exec_buy_b: Amount B receives (token X)
    """

    exec_sell_a: int
    exec_buy_a: int
    exec_sell_b: int
    exec_buy_b: int


# Type alias for execution formula
ExecutionFormula = Callable[[OrderAmounts], ExecutionAmounts]


@dataclass(frozen=True)
class PerfectMatchRule:
    """Rule for matching two orders perfectly (both fully filled).

    Attributes:
        match_type: The order type combination this rule handles
        constraints: Constraints that must all be satisfied
        compute_execution: Function to compute execution amounts
    """

    match_type: MatchType
    constraints: tuple[Constraint, ...]
    compute_execution: ExecutionFormula


@dataclass(frozen=True)
class PartialMatchRule:
    """Rule for partial matching (one order partially filled).

    Partial matching is more complex than perfect matching because:
    1. We must determine which order gets partially filled
    2. Fill-or-kill orders cannot be partially filled
    3. Proportional amounts must satisfy limit prices

    Attributes:
        match_type: The order type combination this rule handles
        can_partial_match: Check if partial matching is possible
        compute_partial: Compute partial execution (returns None if not possible)
    """

    match_type: MatchType
    can_partial_match: Callable[[OrderAmounts], bool]
    # Returns (exec_amounts, a_is_partial, b_is_partial) or None
    compute_partial: Callable[[OrderAmounts], tuple[ExecutionAmounts, bool, bool] | None]


# =============================================================================
# Perfect Match Rules
# =============================================================================


def _exec_sell_sell(a: OrderAmounts) -> ExecutionAmounts:
    """Both orders fully sell their amounts."""
    return ExecutionAmounts(
        exec_sell_a=a.sell_a,
        exec_buy_a=a.sell_b,
        exec_sell_b=a.sell_b,
        exec_buy_b=a.sell_a,
    )


def _exec_sell_buy(a: OrderAmounts) -> ExecutionAmounts:
    """A sells exact, B buys exact."""
    return ExecutionAmounts(
        exec_sell_a=a.sell_a,
        exec_buy_a=a.sell_b,
        exec_sell_b=a.sell_b,
        exec_buy_b=a.buy_b,
    )


def _exec_buy_sell(a: OrderAmounts) -> ExecutionAmounts:
    """A buys exact, B sells exact."""
    return ExecutionAmounts(
        exec_sell_a=a.sell_a,
        exec_buy_a=a.buy_a,
        exec_sell_b=a.sell_b,
        exec_buy_b=a.sell_a,
    )


def _exec_buy_buy(a: OrderAmounts) -> ExecutionAmounts:
    """Both orders get exactly what they want."""
    return ExecutionAmounts(
        exec_sell_a=a.buy_b,
        exec_buy_a=a.buy_a,
        exec_sell_b=a.buy_a,
        exec_buy_b=a.buy_b,
    )


PERFECT_MATCH_RULES: dict[MatchType, PerfectMatchRule] = {
    MatchType.SELL_SELL: PerfectMatchRule(
        match_type=MatchType.SELL_SELL,
        constraints=(
            Constraint(
                description="A receives enough: sell_b >= buy_a",
                check=lambda a: a.sell_b >= a.buy_a,
            ),
            Constraint(
                description="B receives enough: sell_a >= buy_b",
                check=lambda a: a.sell_a >= a.buy_b,
            ),
        ),
        compute_execution=_exec_sell_sell,
    ),
    MatchType.SELL_BUY: PerfectMatchRule(
        match_type=MatchType.SELL_BUY,
        constraints=(
            Constraint(
                description="Amounts match: sell_a == buy_b",
                check=lambda a: a.sell_a == a.buy_b,
            ),
            Constraint(
                description="A receives enough: sell_b >= buy_a",
                check=lambda a: a.sell_b >= a.buy_a,
            ),
        ),
        compute_execution=_exec_sell_buy,
    ),
    MatchType.BUY_SELL: PerfectMatchRule(
        match_type=MatchType.BUY_SELL,
        constraints=(
            Constraint(
                description="Amounts match: buy_a == sell_b",
                check=lambda a: a.buy_a == a.sell_b,
            ),
            Constraint(
                description="B receives enough: sell_a >= buy_b",
                check=lambda a: a.sell_a >= a.buy_b,
            ),
        ),
        compute_execution=_exec_buy_sell,
    ),
    MatchType.BUY_BUY: PerfectMatchRule(
        match_type=MatchType.BUY_BUY,
        constraints=(
            Constraint(
                description="A can afford B's want: sell_a >= buy_b",
                check=lambda a: a.sell_a >= a.buy_b,
            ),
            Constraint(
                description="B can afford A's want: sell_b >= buy_a",
                check=lambda a: a.sell_b >= a.buy_a,
            ),
        ),
        compute_execution=_exec_buy_buy,
    ),
}


# =============================================================================
# Partial Match Rules
# =============================================================================


def _partial_sell_sell(a: OrderAmounts) -> tuple[ExecutionAmounts, bool, bool] | None:
    """Partial match for two sell orders.

    Logic:
    - Check limits are compatible: buy_a * buy_b <= sell_a * sell_b
    - cow_x = min(sell_a, buy_b) - the amount of X transferred
    - If cow_x equals both, it's a perfect match (return None)
    - Calculate cow_y proportionally
    - Verify both limits are satisfied
    """
    # Check limits compatibility
    if a.buy_a * a.buy_b > a.sell_a * a.sell_b:
        logger.debug("cow_partial_no_match_limits", reason="Limit prices not compatible")
        return None

    cow_x = min(a.sell_a, a.buy_b)

    # Perfect match case - handled elsewhere
    if cow_x == a.sell_a and cow_x == a.buy_b:
        return None

    # Determine which order is partial
    a_is_partial = cow_x == a.buy_b  # B fully filled, A partial
    b_is_partial = cow_x == a.sell_a  # A fully filled, B partial

    # Calculate cow_y (keep if-else for clarity with explanatory comments)
    if cow_x == a.buy_b:  # noqa: SIM108
        cow_y = a.sell_b  # B gives all their Y
    else:
        cow_y = (cow_x * a.sell_b) // a.buy_b  # Proportional using B's rate

    # Verify A's limit (ceiling division)
    a_min_receive = (cow_x * a.buy_a + a.sell_a - 1) // a.sell_a
    if cow_y < a_min_receive:
        logger.debug(
            "cow_partial_no_match_a_limit",
            reason="Partial fill doesn't satisfy A's limit",
            cow_y=cow_y,
            a_min_receive=a_min_receive,
        )
        return None

    # Verify B's limit (ceiling division)
    b_min_receive = (cow_y * a.buy_b + a.sell_b - 1) // a.sell_b
    if cow_x < b_min_receive:
        logger.debug(
            "cow_partial_no_match_b_limit",
            reason="Partial fill doesn't satisfy B's limit",
            cow_x=cow_x,
            b_min_receive=b_min_receive,
        )
        return None

    return (
        ExecutionAmounts(
            exec_sell_a=cow_x,
            exec_buy_a=cow_y,
            exec_sell_b=cow_y,
            exec_buy_b=cow_x,
        ),
        a_is_partial,
        b_is_partial,
    )


def _partial_sell_buy(a: OrderAmounts) -> tuple[ExecutionAmounts, bool, bool] | None:
    """Partial match for sell order A and buy order B.

    Partial when A offers more X than B wants.
    A is partially filled, B is fully filled.
    """
    # Only partial if A offers MORE than B wants
    if a.sell_a <= a.buy_b:
        return None

    # Check A's limit can be satisfied by B's payment
    if a.sell_b * a.sell_a < a.buy_b * a.buy_a:
        logger.debug(
            "cow_partial_sell_buy_limit_not_met",
            reason="A's limit not satisfied by B's max payment",
        )
        return None

    cow_x = a.buy_b  # X transferred (A→B)
    cow_y = a.sell_b  # Y transferred (B→A)

    return (
        ExecutionAmounts(
            exec_sell_a=cow_x,
            exec_buy_a=cow_y,
            exec_sell_b=cow_y,
            exec_buy_b=cow_x,
        ),
        True,  # A is partial
        False,  # B is fully filled
    )


def _partial_buy_sell(a: OrderAmounts) -> tuple[ExecutionAmounts, bool, bool] | None:
    """Partial match for buy order A and sell order B.

    Partial when B offers more Y than A wants.
    B is partially filled, A is fully filled.
    """
    # Only partial if B offers MORE than A wants
    if a.sell_b <= a.buy_a:
        return None

    # Check B's limit can be satisfied by A's payment
    if a.sell_a * a.sell_b < a.buy_a * a.buy_b:
        logger.debug(
            "cow_partial_buy_sell_limit_not_met",
            reason="B's limit not satisfied by A's max payment",
        )
        return None

    cow_x = a.sell_a  # X transferred (A→B)
    cow_y = a.buy_a  # Y transferred (B→A)

    return (
        ExecutionAmounts(
            exec_sell_a=cow_x,
            exec_buy_a=cow_y,
            exec_sell_b=cow_y,
            exec_buy_b=cow_x,
        ),
        False,  # A is fully filled
        True,  # B is partial
    )


def _partial_buy_buy(a: OrderAmounts) -> tuple[ExecutionAmounts, bool, bool] | None:
    """Partial match for two buy orders.

    Partial when exactly one can satisfy the other.
    """
    a_can_satisfy_b = a.sell_a >= a.buy_b
    b_can_satisfy_a = a.sell_b >= a.buy_a

    # Both can satisfy or neither can - no partial match
    if a_can_satisfy_b == b_can_satisfy_a:
        return None

    if a_can_satisfy_b:
        # B gets complete fill, A gets partial
        cow_x = a.buy_b
        cow_y = a.sell_b
        a_is_partial = True
        b_is_partial = False
    else:
        # A gets complete fill, B gets partial
        cow_x = a.sell_a
        cow_y = a.buy_a
        a_is_partial = False
        b_is_partial = True

    return (
        ExecutionAmounts(
            exec_sell_a=cow_x,
            exec_buy_a=cow_y,
            exec_sell_b=cow_y,
            exec_buy_b=cow_x,
        ),
        a_is_partial,
        b_is_partial,
    )


PARTIAL_MATCH_RULES: dict[MatchType, PartialMatchRule] = {
    MatchType.SELL_SELL: PartialMatchRule(
        match_type=MatchType.SELL_SELL,
        can_partial_match=lambda a: a.buy_a * a.buy_b <= a.sell_a * a.sell_b,
        compute_partial=_partial_sell_sell,
    ),
    MatchType.SELL_BUY: PartialMatchRule(
        match_type=MatchType.SELL_BUY,
        can_partial_match=lambda a: a.sell_a > a.buy_b,
        compute_partial=_partial_sell_buy,
    ),
    MatchType.BUY_SELL: PartialMatchRule(
        match_type=MatchType.BUY_SELL,
        can_partial_match=lambda a: a.sell_b > a.buy_a,
        compute_partial=_partial_buy_sell,
    ),
    MatchType.BUY_BUY: PartialMatchRule(
        match_type=MatchType.BUY_BUY,
        can_partial_match=lambda a: (a.sell_a >= a.buy_b) != (a.sell_b >= a.buy_a),
        compute_partial=_partial_buy_buy,
    ),
}


# =============================================================================
# Rule Evaluation Functions
# =============================================================================


def evaluate_perfect_match(amounts: OrderAmounts) -> ExecutionAmounts | None:
    """Evaluate perfect match rules for given order amounts.

    Args:
        amounts: The parsed order amounts

    Returns:
        ExecutionAmounts if a perfect match is possible, None otherwise
    """
    match_type = MatchType.from_orders(amounts.a_is_sell, amounts.b_is_sell)
    rule = PERFECT_MATCH_RULES[match_type]

    # Check all constraints
    for constraint in rule.constraints:
        if not constraint.is_satisfied(amounts):
            logger.debug(
                "cow_perfect_match_constraint_failed",
                match_type=match_type.name,
                constraint=constraint.description,
            )
            return None

    return rule.compute_execution(amounts)


def evaluate_partial_match(
    amounts: OrderAmounts,
) -> tuple[ExecutionAmounts, bool, bool] | None:
    """Evaluate partial match rules for given order amounts.

    Args:
        amounts: The parsed order amounts

    Returns:
        Tuple of (execution_amounts, a_is_partial, b_is_partial) or None
    """
    match_type = MatchType.from_orders(amounts.a_is_sell, amounts.b_is_sell)
    rule = PARTIAL_MATCH_RULES[match_type]

    # Quick check if partial matching is possible
    if not rule.can_partial_match(amounts):
        return None

    return rule.compute_partial(amounts)
