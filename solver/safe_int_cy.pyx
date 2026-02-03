# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""Cython-optimized safe integer wrapper for arithmetic on token amounts.

This module provides SafeIntCy, a lightweight wrapper that makes arithmetic
operations safe by default with minimal overhead.
"""

cdef object UINT256_MAX = 2**256 - 1


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


cdef inline object _extract_value_fast(object x):
    """Extract integer value from SafeIntCy or int."""
    if isinstance(x, SafeIntCy):
        return (<SafeIntCy>x)._value
    return x


cdef class SafeIntCy:
    """Cython-optimized integer with safe arithmetic operations."""

    cdef public object _value

    def __init__(self, value):
        if isinstance(value, SafeIntCy):
            self._value = (<SafeIntCy>value)._value
        elif isinstance(value, int):
            self._value = value
        else:
            raise TypeError(f"SafeIntCy requires int, got {type(value).__name__}")

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return f"SafeIntCy({self._value})"

    def __str__(self):
        return str(self._value)

    def __hash__(self):
        return hash(self._value)

    # --- Arithmetic operations ---

    def __add__(self, other):
        cdef object other_val = _extract_value_fast(other)
        return SafeIntCy(self._value + other_val)

    def __radd__(self, other):
        return SafeIntCy(other + self._value)

    def __sub__(self, other):
        cdef object other_val = _extract_value_fast(other)
        cdef object result = self._value - other_val
        if result < 0:
            raise Underflow(f"Underflow: {self._value} - {other_val} = {result}")
        return SafeIntCy(result)

    def __rsub__(self, other):
        cdef object result = other - self._value
        if result < 0:
            raise Underflow(f"Underflow: {other} - {self._value} = {result}")
        return SafeIntCy(result)

    def __mul__(self, other):
        cdef object other_val = _extract_value_fast(other)
        return SafeIntCy(self._value * other_val)

    def __rmul__(self, other):
        return SafeIntCy(other * self._value)

    def __floordiv__(self, other):
        cdef object other_val = _extract_value_fast(other)
        if other_val == 0:
            raise DivisionByZero(f"Division by zero: {self._value} // 0")
        return SafeIntCy(self._value // other_val)

    def __rfloordiv__(self, other):
        if self._value == 0:
            raise DivisionByZero(f"Division by zero: {other} // 0")
        return SafeIntCy(other // self._value)

    def __mod__(self, other):
        cdef object other_val = _extract_value_fast(other)
        if other_val == 0:
            raise DivisionByZero(f"Modulo by zero: {self._value} % 0")
        return SafeIntCy(self._value % other_val)

    def __rmod__(self, other):
        if self._value == 0:
            raise DivisionByZero(f"Modulo by zero: {other} % 0")
        return SafeIntCy(other % self._value)

    def __truediv__(self, other):
        raise TypeError(
            "SafeIntCy does not support true division (/). "
            "Use floor division (//) for integer division."
        )

    def __rtruediv__(self, other):
        raise TypeError(
            "SafeIntCy does not support true division (/). "
            "Use floor division (//) for integer division."
        )

    def __neg__(self):
        return SafeIntCy(-self._value)

    def __pos__(self):
        return self

    def __abs__(self):
        return SafeIntCy(abs(self._value))

    # --- Comparison operations ---

    def __eq__(self, other):
        if isinstance(other, SafeIntCy):
            return self._value == (<SafeIntCy>other)._value
        if isinstance(other, int):
            return self._value == other
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __lt__(self, other):
        cdef object other_val = _extract_value_fast(other)
        return self._value < other_val

    def __le__(self, other):
        cdef object other_val = _extract_value_fast(other)
        return self._value <= other_val

    def __gt__(self, other):
        cdef object other_val = _extract_value_fast(other)
        return self._value > other_val

    def __ge__(self, other):
        cdef object other_val = _extract_value_fast(other)
        return self._value >= other_val

    # --- Conversion ---

    def __int__(self):
        return self._value

    def __bool__(self):
        return self._value != 0

    def __index__(self):
        return self._value

    # --- Named operations ---

    cpdef SafeIntCy ceiling_div(self, other):
        """Ceiling division (rounds up)."""
        cdef object other_val = _extract_value_fast(other)
        if other_val == 0:
            raise DivisionByZero(f"Ceiling division by zero: {self._value}")
        if other_val < 0:
            raise ValueError(f"ceiling_div requires positive divisor, got {other_val}")
        return SafeIntCy((self._value + other_val - 1) // other_val)

    cpdef SafeIntCy min(self, other):
        """Return minimum of self and other."""
        cdef object other_val = _extract_value_fast(other)
        return SafeIntCy(min(self._value, other_val))

    cpdef SafeIntCy max(self, other):
        """Return maximum of self and other."""
        cdef object other_val = _extract_value_fast(other)
        return SafeIntCy(max(self._value, other_val))

    cpdef SafeIntCy clamp(self, object min_val, object max_val):
        """Clamp value to range [min_val, max_val]."""
        return SafeIntCy(max(min_val, min(self._value, max_val)))

    cpdef SafeIntCy saturating_sub(self, other):
        """Subtract, clamping result to zero instead of raising."""
        cdef object other_val = _extract_value_fast(other)
        return SafeIntCy(max(0, self._value - other_val))

    def checked_sub(self, other):
        """Subtract, returning None on underflow instead of raising."""
        cdef object other_val = _extract_value_fast(other)
        cdef object result = self._value - other_val
        if result < 0:
            return None
        return SafeIntCy(result)

    def checked_div(self, other):
        """Divide, returning None on zero instead of raising."""
        cdef object other_val = _extract_value_fast(other)
        if other_val == 0:
            return None
        return SafeIntCy(self._value // other_val)

    def checked_ceiling_div(self, other):
        """Ceiling division, returning None on zero or negative."""
        cdef object other_val = _extract_value_fast(other)
        if other_val <= 0:
            return None
        return SafeIntCy((self._value + other_val - 1) // other_val)

    cpdef object to_uint256(self):
        """Convert to int, validating uint256 bounds."""
        if self._value < 0:
            raise Uint256Overflow(f"Negative value cannot be uint256: {self._value}")
        if self._value > UINT256_MAX:
            raise Uint256Overflow(f"Value exceeds uint256 max: {self._value}")
        return self._value

    cpdef bint is_uint256(self):
        """Check if value fits in uint256 without raising."""
        return 0 <= self._value <= UINT256_MAX

    @classmethod
    def zero(cls):
        """Create a SafeIntCy with value 0."""
        return cls(0)

    @classmethod
    def from_str(cls, s):
        """Parse SafeIntCy from string."""
        return cls(int(s))


# Convenience alias
SCy = SafeIntCy
