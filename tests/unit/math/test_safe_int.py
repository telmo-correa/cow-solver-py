"""Tests for SafeInt safe arithmetic wrapper."""

import pytest

from solver.safe_int import (
    UINT256_MAX,
    DivisionByZero,
    S,
    SafeInt,
    SafeIntError,
    Uint256Overflow,
    Underflow,
)


class TestSafeIntConstruction:
    """Tests for SafeInt construction."""

    def test_from_int(self):
        """SafeInt can be constructed from int."""
        s = SafeInt(42)
        assert s.value == 42

    def test_from_safeint(self):
        """SafeInt can be constructed from another SafeInt."""
        s1 = SafeInt(42)
        s2 = SafeInt(s1)
        assert s2.value == 42

    def test_from_negative(self):
        """SafeInt can hold negative values (validated on conversion)."""
        s = SafeInt(-10)
        assert s.value == -10

    def test_from_zero(self):
        """SafeInt can hold zero."""
        s = SafeInt(0)
        assert s.value == 0

    def test_from_large(self):
        """SafeInt can hold large values."""
        s = SafeInt(10**50)
        assert s.value == 10**50

    def test_from_invalid_type_raises(self):
        """SafeInt rejects invalid types."""
        with pytest.raises(TypeError):
            SafeInt("42")  # type: ignore
        with pytest.raises(TypeError):
            SafeInt(3.14)  # type: ignore

    def test_alias_s(self):
        """S is an alias for SafeInt."""
        assert S is SafeInt
        s = S(42)
        assert isinstance(s, SafeInt)

    def test_zero_constructor(self):
        """SafeInt.zero() creates zero value."""
        s = SafeInt.zero()
        assert s.value == 0

    def test_from_str(self):
        """SafeInt.from_str parses strings."""
        s = SafeInt.from_str("12345")
        assert s.value == 12345

    def test_from_str_invalid_raises(self):
        """SafeInt.from_str raises on invalid input."""
        with pytest.raises(ValueError):
            SafeInt.from_str("not a number")


class TestSafeIntArithmetic:
    """Tests for SafeInt arithmetic operations."""

    def test_add(self):
        """Addition works correctly."""
        assert (S(10) + S(5)).value == 15
        assert (S(10) + 5).value == 15
        assert (5 + S(10)).value == 15

    def test_add_negative_result(self):
        """Addition can produce negative results."""
        assert (S(-10) + S(5)).value == -5

    def test_sub_positive_result(self):
        """Subtraction with positive result works."""
        assert (S(10) - S(3)).value == 7
        assert (S(10) - 3).value == 7
        assert (10 - S(3)).value == 7

    def test_sub_zero_result(self):
        """Subtraction resulting in zero works."""
        assert (S(5) - S(5)).value == 0

    def test_sub_underflow_raises(self):
        """Subtraction underflow raises Underflow."""
        with pytest.raises(Underflow) as exc_info:
            S(5) - S(10)
        assert "Underflow" in str(exc_info.value)
        assert "5 - 10" in str(exc_info.value)

    def test_rsub_underflow_raises(self):
        """Reverse subtraction underflow raises Underflow."""
        with pytest.raises(Underflow):
            5 - S(10)

    def test_mul(self):
        """Multiplication works correctly."""
        assert (S(6) * S(7)).value == 42
        assert (S(6) * 7).value == 42
        assert (6 * S(7)).value == 42

    def test_mul_large(self):
        """Multiplication handles large values."""
        big = 10**30
        result = S(big) * S(big)
        assert result.value == big * big

    def test_floordiv(self):
        """Floor division works correctly."""
        assert (S(10) // S(3)).value == 3
        assert (S(10) // 3).value == 3
        assert (10 // S(3)).value == 3

    def test_floordiv_by_zero_raises(self):
        """Division by zero raises DivisionByZero."""
        with pytest.raises(DivisionByZero) as exc_info:
            S(10) // S(0)
        assert "Division by zero" in str(exc_info.value)

    def test_rfloordiv_by_zero_raises(self):
        """Reverse division by zero raises DivisionByZero."""
        with pytest.raises(DivisionByZero):
            10 // S(0)

    def test_mod(self):
        """Modulo works correctly."""
        assert (S(10) % S(3)).value == 1
        assert (S(10) % 3).value == 1
        assert (10 % S(3)).value == 1

    def test_mod_by_zero_raises(self):
        """Modulo by zero raises DivisionByZero."""
        with pytest.raises(DivisionByZero):
            S(10) % S(0)

    def test_truediv_raises_typeerror(self):
        """True division raises TypeError to prevent float results."""
        with pytest.raises(TypeError) as exc_info:
            S(10) / S(3)
        assert "floor division" in str(exc_info.value)

    def test_rtruediv_raises_typeerror(self):
        """Reverse true division raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            10 / S(3)
        assert "floor division" in str(exc_info.value)

    def test_neg(self):
        """Negation works."""
        assert (-S(5)).value == -5
        assert (-S(-5)).value == 5

    def test_pos(self):
        """Unary positive returns self."""
        s = S(5)
        assert (+s).value == 5

    def test_abs(self):
        """Absolute value works."""
        assert abs(S(-5)).value == 5
        assert abs(S(5)).value == 5


class TestSafeIntComparison:
    """Tests for SafeInt comparison operations."""

    def test_eq_safeint(self):
        """Equality with SafeInt."""
        assert S(5) == S(5)
        assert S(5) != S(6)

    def test_eq_int(self):
        """Equality with int."""
        assert S(5) == 5
        assert S(5) != 6

    def test_ne(self):
        """Inequality works."""
        assert S(5) != S(6)
        assert S(5) != 6

    def test_lt(self):
        """Less than works."""
        assert S(5) < S(6)
        assert S(5) < 6
        assert not (S(6) < S(5))

    def test_le(self):
        """Less than or equal works."""
        assert S(5) <= S(6)
        assert S(5) <= S(5)
        assert S(5) <= 6

    def test_gt(self):
        """Greater than works."""
        assert S(6) > S(5)
        assert S(6) > 5
        assert not (S(5) > S(6))

    def test_ge(self):
        """Greater than or equal works."""
        assert S(6) >= S(5)
        assert S(5) >= S(5)
        assert S(6) >= 5


class TestSafeIntConversion:
    """Tests for SafeInt conversion operations."""

    def test_int(self):
        """int() conversion works."""
        assert int(S(42)) == 42

    def test_bool_nonzero(self):
        """bool() is True for non-zero."""
        assert bool(S(1)) is True
        assert bool(S(-1)) is True

    def test_bool_zero(self):
        """bool() is False for zero."""
        assert bool(S(0)) is False

    def test_str(self):
        """str() conversion works."""
        assert str(S(42)) == "42"

    def test_repr(self):
        """repr() shows SafeInt wrapper."""
        assert repr(S(42)) == "SafeInt(42)"

    def test_hash(self):
        """SafeInt is hashable."""
        s = S(42)
        d = {s: "value"}
        assert d[S(42)] == "value"

    def test_index(self):
        """SafeInt can be used as index."""
        lst = [0, 1, 2, 3, 4]
        assert lst[S(2)] == 2


class TestSafeIntNamedOps:
    """Tests for SafeInt named operations."""

    def test_ceiling_div(self):
        """Ceiling division rounds up."""
        assert S(10).ceiling_div(3).value == 4  # ceil(10/3) = 4
        assert S(9).ceiling_div(3).value == 3  # ceil(9/3) = 3
        assert S(1).ceiling_div(2).value == 1  # ceil(1/2) = 1

    def test_ceiling_div_by_zero_raises(self):
        """Ceiling division by zero raises."""
        with pytest.raises(DivisionByZero):
            S(10).ceiling_div(0)

    def test_ceiling_div_negative_raises(self):
        """Ceiling division by negative raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            S(10).ceiling_div(-3)
        assert "positive divisor" in str(exc_info.value)

    def test_min(self):
        """min() returns smaller value."""
        assert S(10).min(5).value == 5
        assert S(5).min(10).value == 5
        assert S(5).min(S(10)).value == 5

    def test_max(self):
        """max() returns larger value."""
        assert S(10).max(5).value == 10
        assert S(5).max(10).value == 10

    def test_clamp(self):
        """clamp() restricts to range."""
        assert S(5).clamp(0, 10).value == 5
        assert S(-5).clamp(0, 10).value == 0
        assert S(15).clamp(0, 10).value == 10

    def test_saturating_sub(self):
        """saturating_sub clamps to zero."""
        assert S(10).saturating_sub(3).value == 7
        assert S(10).saturating_sub(10).value == 0
        assert S(10).saturating_sub(15).value == 0  # Clamped, no exception

    def test_checked_sub_success(self):
        """checked_sub returns SafeInt on success."""
        result = S(10).checked_sub(3)
        assert result is not None
        assert result.value == 7

    def test_checked_sub_underflow(self):
        """checked_sub returns None on underflow."""
        result = S(5).checked_sub(10)
        assert result is None

    def test_checked_div_success(self):
        """checked_div returns SafeInt on success."""
        result = S(10).checked_div(3)
        assert result is not None
        assert result.value == 3

    def test_checked_div_by_zero(self):
        """checked_div returns None on zero."""
        result = S(10).checked_div(0)
        assert result is None

    def test_checked_ceiling_div_success(self):
        """checked_ceiling_div returns SafeInt on success."""
        result = S(10).checked_ceiling_div(3)
        assert result is not None
        assert result.value == 4

    def test_checked_ceiling_div_by_zero(self):
        """checked_ceiling_div returns None on zero."""
        result = S(10).checked_ceiling_div(0)
        assert result is None

    def test_checked_ceiling_div_negative(self):
        """checked_ceiling_div returns None on negative divisor."""
        result = S(10).checked_ceiling_div(-3)
        assert result is None


class TestSafeIntUint256:
    """Tests for uint256 validation."""

    def test_to_uint256_valid(self):
        """to_uint256 returns value when valid."""
        assert S(0).to_uint256() == 0
        assert S(1000).to_uint256() == 1000
        assert S(UINT256_MAX).to_uint256() == UINT256_MAX

    def test_to_uint256_negative_raises(self):
        """to_uint256 raises on negative value."""
        with pytest.raises(Uint256Overflow) as exc_info:
            S(-1).to_uint256()
        assert "Negative" in str(exc_info.value)

    def test_to_uint256_overflow_raises(self):
        """to_uint256 raises on overflow."""
        with pytest.raises(Uint256Overflow) as exc_info:
            S(UINT256_MAX + 1).to_uint256()
        assert "exceeds uint256" in str(exc_info.value)

    def test_is_uint256_valid(self):
        """is_uint256 returns True for valid values."""
        assert S(0).is_uint256() is True
        assert S(1000).is_uint256() is True
        assert S(UINT256_MAX).is_uint256() is True

    def test_is_uint256_invalid(self):
        """is_uint256 returns False for invalid values."""
        assert S(-1).is_uint256() is False
        assert S(UINT256_MAX + 1).is_uint256() is False


class TestSafeIntExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_division_by_zero_is_safeint_error(self):
        """DivisionByZero inherits from SafeIntError."""
        assert issubclass(DivisionByZero, SafeIntError)
        assert issubclass(DivisionByZero, ArithmeticError)

    def test_underflow_is_safeint_error(self):
        """Underflow inherits from SafeIntError."""
        assert issubclass(Underflow, SafeIntError)
        assert issubclass(Underflow, ArithmeticError)

    def test_uint256_overflow_is_safeint_error(self):
        """Uint256Overflow inherits from SafeIntError."""
        assert issubclass(Uint256Overflow, SafeIntError)
        assert issubclass(Uint256Overflow, ArithmeticError)

    def test_can_catch_all_with_safeint_error(self):
        """All SafeInt exceptions can be caught with SafeIntError."""
        caught = []
        try:
            S(5) - S(10)
        except SafeIntError:
            caught.append("underflow")

        try:
            S(10) // S(0)
        except SafeIntError:
            caught.append("division")

        try:
            S(-1).to_uint256()
        except SafeIntError:
            caught.append("overflow")

        assert caught == ["underflow", "division", "overflow"]


class TestSafeIntChaining:
    """Tests for chained operations."""

    def test_chain_operations(self):
        """Operations can be chained naturally."""
        # (10 + 5) * 3 // 2 = 22
        result = ((S(10) + S(5)) * S(3)) // S(2)
        assert result.value == 22

    def test_chain_with_int(self):
        """Chaining works with int operands."""
        result = (S(100) - 50) * 2 // 10
        assert result.value == 10

    def test_chain_ceiling_div(self):
        """ceiling_div chains with other operations."""
        # (10 * 3).ceiling_div(4) = ceil(30/4) = 8
        result = (S(10) * 3).ceiling_div(4)
        assert result.value == 8


class TestSafeIntEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_divided_by_nonzero(self):
        """Zero divided by non-zero is zero."""
        assert (S(0) // S(5)).value == 0

    def test_subtract_from_zero_raises(self):
        """Subtracting from zero raises if result would be negative."""
        with pytest.raises(Underflow):
            S(0) - S(1)

    def test_multiply_by_zero(self):
        """Multiplying by zero produces zero."""
        assert (S(100) * S(0)).value == 0
        assert (S(0) * S(100)).value == 0

    def test_large_multiplication_no_overflow(self):
        """Python arbitrary precision handles large multiplications."""
        big = 10**50
        result = S(big) * S(big)
        assert result.value == big * big
        # But it won't fit in uint256
        assert result.is_uint256() is False

    def test_precise_division(self):
        """Floor division truncates correctly."""
        assert (S(7) // S(3)).value == 2
        assert (S(8) // S(3)).value == 2
        assert (S(9) // S(3)).value == 3

    def test_ceiling_div_edge_cases(self):
        """Ceiling division handles edge cases."""
        assert S(0).ceiling_div(5).value == 0
        assert S(1).ceiling_div(1).value == 1
        assert S(UINT256_MAX).ceiling_div(UINT256_MAX).value == 1
