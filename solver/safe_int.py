"""Safe integer wrapper for arithmetic on token amounts.

This module provides SafeInt, a lightweight wrapper that makes arithmetic
operations safe by default:
- Division by zero raises ArithmeticError
- Subtraction underflow raises ArithmeticError
- uint256 overflow is caught on conversion

Usage pattern:
    from solver.safe_int import SafeInt, S

    def calculate(a: int, b: int, c: int) -> int:
        # Wrap at entry
        sa, sb, sc = S(a), S(b), S(c)

        # Natural arithmetic - automatically safe
        result = (sa * sb) // sc  # Raises if sc == 0
        remainder = sa - sb       # Raises if sb > sa

        # Unwrap at exit
        return result.value
"""

from __future__ import annotations

UINT256_MAX = 2**256 - 1


class SafeIntError(ArithmeticError):
    """Base class for SafeInt arithmetic errors."""

    pass


class DivisionByZero(SafeIntError):
    """Division or modulo by zero."""

    pass


class Underflow(SafeIntError):
    """Subtraction would produce negative result."""

    pass


class Uint256Overflow(SafeIntError):
    """Value exceeds uint256 maximum."""

    pass


class SafeInt:
    """Integer with safe arithmetic operations.

    Wraps an integer and provides arithmetic operators that raise
    descriptive errors instead of producing invalid results:
    - Division by zero raises DivisionByZero
    - Negative results from subtraction raise Underflow
    - Values exceeding uint256 raise Uint256Overflow on to_uint256()

    SafeInt is designed for calculations on token amounts, prices,
    and other values that must remain non-negative and eventually
    fit in Ethereum's uint256.

    Attributes:
        value: The underlying integer value (read-only)
    """

    __slots__ = ("_value",)
    _value: int

    def __init__(self, value: int | SafeInt) -> None:
        """Create a SafeInt from an integer or another SafeInt.

        Args:
            value: Integer value to wrap, or SafeInt to copy

        Raises:
            TypeError: If value is not an int or SafeInt
        """
        if isinstance(value, SafeInt):
            self._value = value._value
        elif isinstance(value, int):
            self._value = value
        else:
            raise TypeError(f"SafeInt requires int, got {type(value).__name__}")

    @property
    def value(self) -> int:
        """The underlying integer value."""
        return self._value

    def __repr__(self) -> str:
        return f"SafeInt({self._value})"

    def __str__(self) -> str:
        return str(self._value)

    def __hash__(self) -> int:
        return hash(self._value)

    # --- Arithmetic operations ---

    def __add__(self, other: SafeInt | int) -> SafeInt:
        """Add two values. Result may be negative (no check)."""
        other_val = _extract_value(other)
        return SafeInt(self._value + other_val)

    def __radd__(self, other: int) -> SafeInt:
        return SafeInt(other + self._value)

    def __sub__(self, other: SafeInt | int) -> SafeInt:
        """Subtract other from self.

        Raises:
            Underflow: If result would be negative
        """
        other_val = _extract_value(other)
        result = self._value - other_val
        if result < 0:
            raise Underflow(f"Underflow: {self._value} - {other_val} = {result}")
        return SafeInt(result)

    def __rsub__(self, other: int) -> SafeInt:
        """Subtract self from other (other - self).

        Raises:
            Underflow: If result would be negative
        """
        result = other - self._value
        if result < 0:
            raise Underflow(f"Underflow: {other} - {self._value} = {result}")
        return SafeInt(result)

    def __mul__(self, other: SafeInt | int) -> SafeInt:
        """Multiply two values."""
        other_val = _extract_value(other)
        return SafeInt(self._value * other_val)

    def __rmul__(self, other: int) -> SafeInt:
        return SafeInt(other * self._value)

    def __floordiv__(self, other: SafeInt | int) -> SafeInt:
        """Integer division.

        Raises:
            DivisionByZero: If other is zero
        """
        other_val = _extract_value(other)
        if other_val == 0:
            raise DivisionByZero(f"Division by zero: {self._value} // 0")
        return SafeInt(self._value // other_val)

    def __rfloordiv__(self, other: int) -> SafeInt:
        """Integer division (other // self).

        Raises:
            DivisionByZero: If self is zero
        """
        if self._value == 0:
            raise DivisionByZero(f"Division by zero: {other} // 0")
        return SafeInt(other // self._value)

    def __mod__(self, other: SafeInt | int) -> SafeInt:
        """Modulo operation.

        Raises:
            DivisionByZero: If other is zero
        """
        other_val = _extract_value(other)
        if other_val == 0:
            raise DivisionByZero(f"Modulo by zero: {self._value} % 0")
        return SafeInt(self._value % other_val)

    def __rmod__(self, other: int) -> SafeInt:
        """Modulo operation (other % self).

        Raises:
            DivisionByZero: If self is zero
        """
        if self._value == 0:
            raise DivisionByZero(f"Modulo by zero: {other} % 0")
        return SafeInt(other % self._value)

    def __neg__(self) -> SafeInt:
        """Negate the value. Result may be negative."""
        return SafeInt(-self._value)

    def __pos__(self) -> SafeInt:
        """Unary positive (returns self)."""
        return self

    def __abs__(self) -> SafeInt:
        """Absolute value."""
        return SafeInt(abs(self._value))

    # --- Comparison operations ---

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SafeInt):
            return self._value == other._value
        if isinstance(other, int):
            return self._value == other
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __lt__(self, other: SafeInt | int) -> bool:
        other_val = _extract_value(other)
        return self._value < other_val

    def __le__(self, other: SafeInt | int) -> bool:
        other_val = _extract_value(other)
        return self._value <= other_val

    def __gt__(self, other: SafeInt | int) -> bool:
        other_val = _extract_value(other)
        return self._value > other_val

    def __ge__(self, other: SafeInt | int) -> bool:
        other_val = _extract_value(other)
        return self._value >= other_val

    # --- Conversion ---

    def __int__(self) -> int:
        """Convert to int."""
        return self._value

    def __bool__(self) -> bool:
        """True if non-zero."""
        return self._value != 0

    def __index__(self) -> int:
        """Support use in slices and as array index."""
        return self._value

    # --- Named operations ---

    def ceiling_div(self, other: SafeInt | int) -> SafeInt:
        """Ceiling division (rounds up).

        Equivalent to: (self + other - 1) // other

        Raises:
            DivisionByZero: If other is zero
        """
        other_val = _extract_value(other)
        if other_val == 0:
            raise DivisionByZero(f"Ceiling division by zero: {self._value}")
        return SafeInt((self._value + other_val - 1) // other_val)

    def min(self, other: SafeInt | int) -> SafeInt:
        """Return minimum of self and other."""
        other_val = _extract_value(other)
        return SafeInt(min(self._value, other_val))

    def max(self, other: SafeInt | int) -> SafeInt:
        """Return maximum of self and other."""
        other_val = _extract_value(other)
        return SafeInt(max(self._value, other_val))

    def clamp(self, min_val: int, max_val: int) -> SafeInt:
        """Clamp value to range [min_val, max_val]."""
        return SafeInt(max(min_val, min(self._value, max_val)))

    def saturating_sub(self, other: SafeInt | int) -> SafeInt:
        """Subtract, clamping result to zero instead of raising.

        Unlike __sub__, this never raises Underflow.
        """
        other_val = _extract_value(other)
        return SafeInt(max(0, self._value - other_val))

    def checked_sub(self, other: SafeInt | int) -> SafeInt | None:
        """Subtract, returning None on underflow instead of raising.

        Unlike __sub__, this returns None instead of raising Underflow.
        """
        other_val = _extract_value(other)
        result = self._value - other_val
        if result < 0:
            return None
        return SafeInt(result)

    def checked_div(self, other: SafeInt | int) -> SafeInt | None:
        """Divide, returning None on zero instead of raising.

        Unlike __floordiv__, this returns None instead of raising DivisionByZero.
        """
        other_val = _extract_value(other)
        if other_val == 0:
            return None
        return SafeInt(self._value // other_val)

    def checked_ceiling_div(self, other: SafeInt | int) -> SafeInt | None:
        """Ceiling division, returning None on zero instead of raising."""
        other_val = _extract_value(other)
        if other_val == 0:
            return None
        return SafeInt((self._value + other_val - 1) // other_val)

    def to_uint256(self) -> int:
        """Convert to int, validating uint256 bounds.

        Raises:
            Uint256Overflow: If value is negative or exceeds 2^256-1
        """
        if self._value < 0:
            raise Uint256Overflow(f"Negative value cannot be uint256: {self._value}")
        if self._value > UINT256_MAX:
            raise Uint256Overflow(f"Value exceeds uint256 max: {self._value}")
        return self._value

    def is_uint256(self) -> bool:
        """Check if value fits in uint256 without raising."""
        return 0 <= self._value <= UINT256_MAX

    @classmethod
    def zero(cls) -> SafeInt:
        """Create a SafeInt with value 0."""
        return cls(0)

    @classmethod
    def from_str(cls, s: str) -> SafeInt:
        """Parse SafeInt from string.

        Raises:
            ValueError: If string is not a valid integer
        """
        return cls(int(s))


def _extract_value(x: SafeInt | int) -> int:
    """Extract integer value from SafeInt or int."""
    if isinstance(x, SafeInt):
        return x._value
    return x


# Convenience alias for concise code
S = SafeInt
