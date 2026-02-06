"""Tests for Bfp safety guards (underflow, negative inputs).

Verifies that Bfp complement(), sub(), and from_decimal() handle
edge cases safely without producing negative values.
"""

from decimal import Decimal

import pytest

from solver.math.fixed_point import Bfp


class TestBfpComplement:
    """Fix 6: complement() underflow guard."""

    def test_complement_of_zero(self):
        """complement(0) = 1."""
        assert Bfp(0).complement() == Bfp(Bfp.ONE)

    def test_complement_of_one(self):
        """complement(1) = 0."""
        assert Bfp(Bfp.ONE).complement() == Bfp(0)

    def test_complement_of_half(self):
        """complement(0.5) = 0.5."""
        half = Bfp(Bfp.ONE // 2)
        result = half.complement()
        assert result.value == Bfp.ONE - Bfp.ONE // 2

    def test_complement_greater_than_one_clamps_to_zero(self):
        """complement(1.5) should clamp to 0 instead of going negative."""
        one_and_half = Bfp(Bfp.ONE * 3 // 2)
        result = one_and_half.complement()
        assert result.value == 0

    def test_complement_of_two_clamps_to_zero(self):
        """complement(2) should clamp to 0."""
        two = Bfp(Bfp.ONE * 2)
        result = two.complement()
        assert result.value == 0


class TestBfpSub:
    """Fix 7: sub() negative guard."""

    def test_normal_subtraction(self):
        """5 - 3 = 2."""
        a = Bfp(5 * Bfp.ONE)
        b = Bfp(3 * Bfp.ONE)
        result = a.sub(b)
        assert result.value == 2 * Bfp.ONE

    def test_subtract_equal_values(self):
        """5 - 5 = 0."""
        a = Bfp(5 * Bfp.ONE)
        result = a.sub(a)
        assert result.value == 0

    def test_subtract_larger_clamps_to_zero(self):
        """3 - 5 should clamp to 0 instead of going negative."""
        a = Bfp(3 * Bfp.ONE)
        b = Bfp(5 * Bfp.ONE)
        result = a.sub(b)
        assert result.value == 0

    def test_subtract_zero(self):
        """5 - 0 = 5."""
        a = Bfp(5 * Bfp.ONE)
        result = a.sub(Bfp(0))
        assert result.value == 5 * Bfp.ONE


class TestBfpFromDecimal:
    """Fix 8: from_decimal() negative guard."""

    def test_positive_decimal(self):
        """Positive decimal is accepted."""
        result = Bfp.from_decimal(Decimal("1.5"))
        assert result.value == Bfp.ONE * 3 // 2

    def test_zero_decimal(self):
        """Zero is accepted."""
        result = Bfp.from_decimal(Decimal("0"))
        assert result.value == 0

    def test_negative_decimal_raises(self):
        """Negative decimal raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Bfp.from_decimal(Decimal("-1"))

    def test_negative_fraction_raises(self):
        """Negative fraction raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Bfp.from_decimal(Decimal("-0.001"))
