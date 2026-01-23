"""Tests for the data-driven matching rules.

These tests verify the matching rules in isolation, without involving
the full CowMatchStrategy or Order objects.
"""

from dataclasses import FrozenInstanceError

import pytest

from solver.strategies.matching_rules import (
    PARTIAL_MATCH_RULES,
    PERFECT_MATCH_RULES,
    Constraint,
    ExecutionAmounts,
    MatchType,
    OrderAmounts,
    evaluate_partial_match,
    evaluate_perfect_match,
)


class TestMatchType:
    """Tests for MatchType enum."""

    def test_from_orders_sell_sell(self):
        assert MatchType.from_orders(True, True) == MatchType.SELL_SELL

    def test_from_orders_sell_buy(self):
        assert MatchType.from_orders(True, False) == MatchType.SELL_BUY

    def test_from_orders_buy_sell(self):
        assert MatchType.from_orders(False, True) == MatchType.BUY_SELL

    def test_from_orders_buy_buy(self):
        assert MatchType.from_orders(False, False) == MatchType.BUY_BUY

    def test_all_match_types_have_perfect_rules(self):
        """Every match type should have a perfect match rule."""
        for match_type in MatchType:
            assert match_type in PERFECT_MATCH_RULES

    def test_all_match_types_have_partial_rules(self):
        """Every match type should have a partial match rule."""
        for match_type in MatchType:
            assert match_type in PARTIAL_MATCH_RULES


class TestConstraint:
    """Tests for Constraint dataclass."""

    def test_constraint_satisfied(self):
        constraint = Constraint(
            description="A >= B",
            check=lambda a: a.sell_a >= a.buy_a,
        )
        amounts = OrderAmounts(
            sell_a=100, buy_a=50, sell_b=0, buy_b=0, a_is_sell=True, b_is_sell=True
        )
        assert constraint.is_satisfied(amounts) is True

    def test_constraint_not_satisfied(self):
        constraint = Constraint(
            description="A >= B",
            check=lambda a: a.sell_a >= a.buy_a,
        )
        amounts = OrderAmounts(
            sell_a=50, buy_a=100, sell_b=0, buy_b=0, a_is_sell=True, b_is_sell=True
        )
        assert constraint.is_satisfied(amounts) is False


class TestPerfectMatchRules:
    """Tests for perfect match rule evaluation."""

    def test_sell_sell_perfect_match(self):
        """Two sell orders that perfectly satisfy each other's limits."""
        amounts = OrderAmounts(
            sell_a=100,  # A sells 100 X
            buy_a=90,  # A wants at least 90 Y
            sell_b=95,  # B sells 95 Y
            buy_b=100,  # B wants at least 100 X
            a_is_sell=True,
            b_is_sell=True,
        )
        result = evaluate_perfect_match(amounts)
        assert result is not None
        assert result.exec_sell_a == 100  # A sells all X
        assert result.exec_buy_a == 95  # A receives all B's Y
        assert result.exec_sell_b == 95  # B sells all Y
        assert result.exec_buy_b == 100  # B receives all A's X

    def test_sell_sell_limit_a_not_met(self):
        """A's limit not satisfied - B offers less than A's minimum."""
        amounts = OrderAmounts(
            sell_a=100,
            buy_a=100,  # A wants at least 100 Y
            sell_b=90,  # B only offers 90 Y
            buy_b=100,
            a_is_sell=True,
            b_is_sell=True,
        )
        result = evaluate_perfect_match(amounts)
        assert result is None

    def test_sell_sell_limit_b_not_met(self):
        """B's limit not satisfied - A offers less than B's minimum."""
        amounts = OrderAmounts(
            sell_a=90,  # A only offers 90 X
            buy_a=90,
            sell_b=100,
            buy_b=100,  # B wants at least 100 X
            a_is_sell=True,
            b_is_sell=True,
        )
        result = evaluate_perfect_match(amounts)
        assert result is None

    def test_sell_buy_perfect_match(self):
        """Sell order A and buy order B that perfectly match."""
        amounts = OrderAmounts(
            sell_a=100,  # A sells 100 X
            buy_a=90,  # A wants at least 90 Y
            sell_b=95,  # B willing to pay up to 95 Y
            buy_b=100,  # B wants exactly 100 X
            a_is_sell=True,
            b_is_sell=False,
        )
        result = evaluate_perfect_match(amounts)
        assert result is not None
        assert result.exec_sell_a == 100
        assert result.exec_buy_a == 95
        assert result.exec_sell_b == 95
        assert result.exec_buy_b == 100

    def test_sell_buy_amounts_dont_match(self):
        """Sell-buy: A's sell doesn't equal B's want."""
        amounts = OrderAmounts(
            sell_a=100,
            buy_a=90,
            sell_b=95,
            buy_b=90,  # B wants 90, not 100
            a_is_sell=True,
            b_is_sell=False,
        )
        result = evaluate_perfect_match(amounts)
        assert result is None

    def test_buy_sell_perfect_match(self):
        """Buy order A and sell order B that perfectly match."""
        amounts = OrderAmounts(
            sell_a=100,  # A willing to pay up to 100 X
            buy_a=95,  # A wants exactly 95 Y
            sell_b=95,  # B sells 95 Y
            buy_b=90,  # B wants at least 90 X
            a_is_sell=False,
            b_is_sell=True,
        )
        result = evaluate_perfect_match(amounts)
        assert result is not None
        assert result.exec_sell_a == 100
        assert result.exec_buy_a == 95
        assert result.exec_sell_b == 95
        assert result.exec_buy_b == 100

    def test_buy_buy_perfect_match(self):
        """Two buy orders that can satisfy each other."""
        amounts = OrderAmounts(
            sell_a=100,  # A willing to pay up to 100 X
            buy_a=90,  # A wants exactly 90 Y
            sell_b=95,  # B willing to pay up to 95 Y
            buy_b=80,  # B wants exactly 80 X
            a_is_sell=False,
            b_is_sell=False,
        )
        result = evaluate_perfect_match(amounts)
        assert result is not None
        # A pays what B wants, receives what A wants
        assert result.exec_sell_a == 80  # A pays B's want
        assert result.exec_buy_a == 90  # A gets what they want
        assert result.exec_sell_b == 90  # B pays A's want
        assert result.exec_buy_b == 80  # B gets what they want

    def test_buy_buy_a_cant_afford(self):
        """Buy-buy: A can't afford what B wants."""
        amounts = OrderAmounts(
            sell_a=50,  # A can only pay 50 X
            buy_a=90,
            sell_b=95,
            buy_b=80,  # B wants 80 X
            a_is_sell=False,
            b_is_sell=False,
        )
        result = evaluate_perfect_match(amounts)
        assert result is None


class TestPartialMatchRules:
    """Tests for partial match rule evaluation."""

    def test_sell_sell_partial_a_has_remainder(self):
        """Sell-sell: A offers more than B wants, B fully filled."""
        amounts = OrderAmounts(
            sell_a=100,  # A sells 100 X
            buy_a=90,  # A wants at least 90 Y (ratio: 0.9 Y/X)
            sell_b=50,  # B sells 50 Y
            buy_b=50,  # B wants 50 X (ratio: 1.0 Y/X)
            a_is_sell=True,
            b_is_sell=True,
        )
        result = evaluate_partial_match(amounts)
        assert result is not None
        exec, a_partial, b_partial = result
        assert a_partial is True  # A has remainder
        assert b_partial is False  # B fully filled
        assert exec.exec_sell_a == 50  # A sells what B wants
        assert exec.exec_buy_a == 50  # A receives all B's Y
        assert exec.exec_sell_b == 50  # B sells all Y
        assert exec.exec_buy_b == 50  # B receives what they want

    def test_sell_sell_partial_b_has_remainder(self):
        """Sell-sell: A offers less than B wants, A fully filled, B has remainder."""
        amounts = OrderAmounts(
            sell_a=40,  # A sells 40 X
            buy_a=35,  # A wants at least 35 Y
            sell_b=100,  # B sells 100 Y
            buy_b=50,  # B wants 50 X (but A only has 40)
            a_is_sell=True,
            b_is_sell=True,
        )
        # cow_x = min(sell_a=40, buy_b=50) = 40
        # cow_x == sell_a, so A is fully filled, B is partial
        result = evaluate_partial_match(amounts)
        assert result is not None
        exec, a_partial, b_partial = result
        assert a_partial is False  # A fully filled (sold all 40 X)
        assert b_partial is True  # B has remainder (wanted 50, got 40)

    def test_sell_sell_limits_incompatible(self):
        """Sell-sell: Limits can't be satisfied simultaneously."""
        amounts = OrderAmounts(
            sell_a=100,
            buy_a=200,  # A wants 2.0 Y/X
            sell_b=100,
            buy_b=200,  # B wants 2.0 X/Y (i.e., 0.5 Y/X)
            a_is_sell=True,
            b_is_sell=True,
        )
        # buy_a * buy_b (200*200=40000) > sell_a * sell_b (100*100=10000)
        result = evaluate_partial_match(amounts)
        assert result is None

    def test_sell_buy_partial_match(self):
        """Sell-buy: A offers more than B wants."""
        amounts = OrderAmounts(
            sell_a=100,  # A sells 100 X
            buy_a=80,  # A wants at least 80 Y
            sell_b=90,  # B pays up to 90 Y
            buy_b=50,  # B wants exactly 50 X
            a_is_sell=True,
            b_is_sell=False,
        )
        result = evaluate_partial_match(amounts)
        assert result is not None
        exec, a_partial, b_partial = result
        assert a_partial is True
        assert b_partial is False
        assert exec.exec_sell_a == 50  # A sells what B wants
        assert exec.exec_buy_a == 90  # A receives B's max payment

    def test_sell_buy_no_partial_when_a_offers_less(self):
        """Sell-buy: A offers less than B wants - no partial possible."""
        amounts = OrderAmounts(
            sell_a=40,  # A only sells 40 X
            buy_a=30,
            sell_b=50,
            buy_b=50,  # B wants 50 X
            a_is_sell=True,
            b_is_sell=False,
        )
        result = evaluate_partial_match(amounts)
        assert result is None

    def test_buy_sell_partial_match(self):
        """Buy-sell: B offers more than A wants."""
        amounts = OrderAmounts(
            sell_a=100,  # A pays up to 100 X
            buy_a=50,  # A wants exactly 50 Y
            sell_b=100,  # B sells 100 Y
            buy_b=80,  # B wants at least 80 X
            a_is_sell=False,
            b_is_sell=True,
        )
        result = evaluate_partial_match(amounts)
        assert result is not None
        exec, a_partial, b_partial = result
        assert a_partial is False
        assert b_partial is True
        assert exec.exec_sell_a == 100  # A pays max
        assert exec.exec_buy_a == 50  # A gets what they want

    def test_buy_buy_partial_b_complete(self):
        """Buy-buy: A can satisfy B, B can't satisfy A."""
        amounts = OrderAmounts(
            sell_a=100,  # A can pay up to 100 X
            buy_a=90,  # A wants 90 Y
            sell_b=50,  # B can only pay up to 50 Y
            buy_b=80,  # B wants 80 X
            a_is_sell=False,
            b_is_sell=False,
        )
        # A can pay 80 (B's want) <= 100 (A's max): True
        # B can pay 90 (A's want) <= 50 (B's max): False
        result = evaluate_partial_match(amounts)
        assert result is not None
        exec, a_partial, b_partial = result
        assert a_partial is True  # A gets partial (only 50 Y, wanted 90)
        assert b_partial is False  # B fully filled

    def test_buy_buy_partial_a_complete(self):
        """Buy-buy: B can satisfy A, A can't satisfy B."""
        amounts = OrderAmounts(
            sell_a=50,  # A can only pay up to 50 X
            buy_a=90,  # A wants 90 Y
            sell_b=100,  # B can pay up to 100 Y
            buy_b=80,  # B wants 80 X
            a_is_sell=False,
            b_is_sell=False,
        )
        # A can pay 80 (B's want) <= 50 (A's max): False
        # B can pay 90 (A's want) <= 100 (B's max): True
        result = evaluate_partial_match(amounts)
        assert result is not None
        exec, a_partial, b_partial = result
        assert a_partial is False  # A fully filled
        assert b_partial is True  # B gets partial (only 50 X, wanted 80)

    def test_buy_buy_no_partial_both_can_satisfy(self):
        """Buy-buy: Both can satisfy - should be perfect match."""
        amounts = OrderAmounts(
            sell_a=100,
            buy_a=90,
            sell_b=100,
            buy_b=80,
            a_is_sell=False,
            b_is_sell=False,
        )
        # Both can satisfy each other - this is a perfect match case
        result = evaluate_partial_match(amounts)
        assert result is None

    def test_buy_buy_no_partial_neither_can_satisfy(self):
        """Buy-buy: Neither can satisfy the other."""
        amounts = OrderAmounts(
            sell_a=50,
            buy_a=90,
            sell_b=50,
            buy_b=80,
            a_is_sell=False,
            b_is_sell=False,
        )
        result = evaluate_partial_match(amounts)
        assert result is None


class TestExecutionAmounts:
    """Tests for ExecutionAmounts dataclass."""

    def test_execution_amounts_immutable(self):
        """ExecutionAmounts should be immutable (frozen dataclass)."""
        exec = ExecutionAmounts(exec_sell_a=100, exec_buy_a=90, exec_sell_b=90, exec_buy_b=100)
        with pytest.raises(FrozenInstanceError):
            exec.exec_sell_a = 200

    def test_execution_amounts_equality(self):
        """Two ExecutionAmounts with same values should be equal."""
        exec1 = ExecutionAmounts(100, 90, 90, 100)
        exec2 = ExecutionAmounts(100, 90, 90, 100)
        assert exec1 == exec2


class TestRuleCompleteness:
    """Tests to ensure all rules are properly defined."""

    def test_perfect_rules_have_constraints(self):
        """All perfect match rules should have at least one constraint."""
        for match_type, rule in PERFECT_MATCH_RULES.items():
            assert len(rule.constraints) > 0, f"{match_type} has no constraints"

    def test_perfect_rules_have_execution_formula(self):
        """All perfect match rules should have an execution formula."""
        for match_type, rule in PERFECT_MATCH_RULES.items():
            assert rule.compute_execution is not None, f"{match_type} has no formula"

    def test_partial_rules_have_check_function(self):
        """All partial match rules should have a can_partial_match function."""
        for _match_type, rule in PARTIAL_MATCH_RULES.items():
            assert rule.can_partial_match is not None

    def test_partial_rules_have_compute_function(self):
        """All partial match rules should have a compute_partial function."""
        for _match_type, rule in PARTIAL_MATCH_RULES.items():
            assert rule.compute_partial is not None


class TestEdgeCases:
    """Edge case tests for matching rules."""

    def test_zero_amounts_handled(self):
        """Rules should handle zero amounts without crashing."""
        amounts = OrderAmounts(sell_a=0, buy_a=0, sell_b=0, buy_b=0, a_is_sell=True, b_is_sell=True)
        # Should return None, not crash
        result = evaluate_perfect_match(amounts)
        # Division by zero protection in constraints
        assert result is None or isinstance(result, ExecutionAmounts)

    def test_equal_amounts_perfect_match(self):
        """When all amounts equal, should be a perfect match for sell-sell."""
        amounts = OrderAmounts(
            sell_a=100,
            buy_a=100,
            sell_b=100,
            buy_b=100,
            a_is_sell=True,
            b_is_sell=True,
        )
        result = evaluate_perfect_match(amounts)
        assert result is not None
        assert result.exec_sell_a == 100
        assert result.exec_buy_a == 100

    def test_large_amounts_no_overflow(self):
        """Rules should handle large amounts without overflow."""
        # Typical token amounts can be 10^18 or larger
        large = 10**18
        amounts = OrderAmounts(
            sell_a=large,
            buy_a=large - 1,
            sell_b=large,
            buy_b=large - 1,
            a_is_sell=True,
            b_is_sell=True,
        )
        result = evaluate_perfect_match(amounts)
        assert result is not None
        assert result.exec_sell_a == large
