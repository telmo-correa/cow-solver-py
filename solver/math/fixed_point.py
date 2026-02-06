"""Balancer Fixed Point (Bfp) math library.

This module implements 18-decimal fixed-point arithmetic matching Balancer's
on-chain LogExpMath.sol exactly. The implementation is a direct port of:
https://github.com/balancer-labs/balancer-v2-monorepo/blob/6c9e24e22d0c46cca6dd15861d3d33da61a60b98/pkg/solidity-utils/contracts/math/LogExpMath.sol

All values are stored as integers scaled by 10^18.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import ClassVar

__all__ = [
    # Classes
    "Bfp",
    # Errors
    "LogExpMathError",
    "XOutOfBounds",
    "YOutOfBounds",
    "ProductOutOfBounds",
    "InvalidExponent",
    # Functions
    "pow_raw",
    "exp",
    # Constants
    "ONE_18",
    "ONE_20",
    "ONE_36",
    "MAX_IN_RATIO",
    "MAX_OUT_RATIO",
    "AMP_PRECISION",
]

# =============================================================================
# Constants (matching Solidity/Rust exactly)
# =============================================================================

ONE_18 = 10**18
ONE_20 = 10**20
ONE_36 = 10**36

MAX_NATURAL_EXPONENT = 130 * ONE_18  # e^130 is the max we can handle
MIN_NATURAL_EXPONENT = -41 * ONE_18  # e^-41 is close to zero

LN_36_LOWER_BOUND = ONE_18 - 10**17  # 0.9 in fixed-point
LN_36_UPPER_BOUND = ONE_18 + 10**17  # 1.1 in fixed-point

# 2^254 / ONE_20 - bounds the exponent to prevent overflow
MILD_EXPONENT_BOUND = (1 << 254) // ONE_20

# Pre-computed constants for digit extraction
# x values represent exponents (powers of 2), a values are e^x

# 18-decimal precision constants (for large values)
X_18 = {
    0: 128 * ONE_18,  # 2^7
    1: 64 * ONE_18,  # 2^6
}
A_18 = {
    0: 38877084059945950922200000000000000000000000000000000000,  # e^128
    1: 6235149080811616882910000000,  # e^64
}

# 20-decimal precision constants (for medium values)
X_20 = {
    2: 3_200_000_000_000_000_000_000,  # 32 * ONE_20 = 2^5
    3: 1_600_000_000_000_000_000_000,  # 16 * ONE_20 = 2^4
    4: 800_000_000_000_000_000_000,  # 8 * ONE_20 = 2^3
    5: 400_000_000_000_000_000_000,  # 4 * ONE_20 = 2^2
    6: 200_000_000_000_000_000_000,  # 2 * ONE_20 = 2^1
    7: 100_000_000_000_000_000_000,  # 1 * ONE_20 = 2^0
    8: 50_000_000_000_000_000_000,  # 0.5 * ONE_20 = 2^-1
    9: 25_000_000_000_000_000_000,  # 0.25 * ONE_20 = 2^-2
    10: 12_500_000_000_000_000_000,  # 0.125 * ONE_20 = 2^-3
    11: 6_250_000_000_000_000_000,  # 0.0625 * ONE_20 = 2^-4
}
A_20 = {
    2: 7_896_296_018_268_069_516_100_000_000_000_000,  # e^32
    3: 888_611_052_050_787_263_676_000_000,  # e^16
    4: 298_095_798_704_172_827_474_000,  # e^8
    5: 5_459_815_003_314_423_907_810,  # e^4
    6: 738_905_609_893_065_022_723,  # e^2
    7: 271_828_182_845_904_523_536,  # e^1
    8: 164_872_127_070_012_814_685,  # e^0.5
    9: 128_402_541_668_774_148_407,  # e^0.25
    10: 113_314_845_306_682_631_683,  # e^0.125
    11: 106_449_445_891_785_942_956,  # e^0.0625
}


# =============================================================================
# Error classes
# =============================================================================


class LogExpMathError(Exception):
    """Base error for LogExpMath operations."""

    pass


class XOutOfBounds(LogExpMathError):
    """Error 006: Base x is out of valid range."""

    pass


class YOutOfBounds(LogExpMathError):
    """Error 007: Exponent y exceeds MILD_EXPONENT_BOUND."""

    pass


class ProductOutOfBounds(LogExpMathError):
    """Error 008: Result of y * ln(x) is outside valid range for exp."""

    pass


class InvalidExponent(LogExpMathError):
    """Error 009: Exponent is out of valid range after reduction."""

    pass


# =============================================================================
# Core math functions (matching Solidity/Rust)
# =============================================================================


def _div_trunc(a: int, b: int) -> int:
    """Integer division with truncation toward zero (matching Solidity/Rust).

    Python's // operator rounds toward negative infinity, but Solidity and Rust
    truncate toward zero. This matters for negative numbers.

    Args:
        a: Dividend (can be positive or negative)
        b: Divisor (must be non-zero)

    Returns:
        a / b truncated toward zero

    Raises:
        ZeroDivisionError: If b is zero

    Examples:
        Python: -7 // 3 = -3 (rounds toward -inf)
        Solidity/Rust: -7 / 3 = -2 (truncates toward zero)
    """
    if b == 0:
        raise ZeroDivisionError("Division by zero in _div_trunc")

    # Python's // floors toward -inf, but for same-sign operands the result
    # is non-negative, so floor and truncate give the same answer.
    # For different signs, we need to explicitly truncate toward zero.
    if (a >= 0) == (b >= 0):
        # Same sign: result is non-negative, // gives correct truncation
        return a // b
    else:
        # Different signs: use -(abs(a) // abs(b)) to truncate toward zero
        return -(abs(a) // abs(b))


def _ln(a: int) -> int:
    """Compute natural logarithm of a (18-decimal fixed-point).

    Uses digit extraction and Taylor series for arctanh.

    Args:
        a: Input value, must be positive (a > 0). Zero raises ZeroDivisionError.

    Returns:
        ln(a) as 18-decimal fixed-point integer.

    Note: This function uses Python's // operator directly (not _div_trunc)
    because after the a < ONE_18 check and recursive call, all subsequent
    operations involve positive values only, where // behaves identically
    to Solidity/Rust truncation.
    """
    if a < ONE_18:
        # ln(a) = -ln(1/a)
        return -_ln((ONE_18 * ONE_18) // a)

    sum_val = 0

    # Extract large powers of e (18-decimal precision)
    for i in range(2):
        if a >= A_18[i] * ONE_18:
            a //= A_18[i]
            sum_val += X_18[i]

    # Scale up to 20-decimal precision
    sum_val *= 100
    a *= 100

    # Extract medium powers of e (20-decimal precision)
    for i in range(2, 12):
        if a >= A_20[i]:
            a = (a * ONE_20) // A_20[i]
            sum_val += X_20[i]

    # Taylor series for remaining fraction using arctanh
    # ln(a) = 2 * arctanh((a-1)/(a+1)) = 2 * (z + z^3/3 + z^5/5 + ...)
    z = ((a - ONE_20) * ONE_20) // (a + ONE_20)
    z_squared = (z * z) // ONE_20

    num = z
    series_sum = num

    # 6 terms total: z + z^3/3 + z^5/5 + z^7/7 + z^9/9 + z^11/11
    for i in range(3, 12, 2):  # i = 3, 5, 7, 9, 11
        num = (num * z_squared) // ONE_20
        series_sum += num // i

    series_sum *= 2

    return (sum_val + series_sum) // 100


def _ln_36(x: int) -> int:
    """Compute natural logarithm with 36-decimal precision.

    Used when x is close to 1 for better accuracy.
    Uses truncation division to match Solidity/Rust behavior for x < 1.

    Args:
        x: Input value in 18-decimal fixed-point, should be close to ONE_18.

    Returns:
        ln(x) as 36-decimal fixed-point integer.
    """
    x *= ONE_18  # Scale to 36 decimals

    # z can be negative when original input < 1, use truncation division
    z = _div_trunc((x - ONE_36) * ONE_36, x + ONE_36)
    z_squared = _div_trunc(z * z, ONE_36)

    num = z
    series_sum = num

    # 8 terms total: z + z^3/3 + z^5/5 + z^7/7 + z^9/9 + z^11/11 + z^13/13 + z^15/15
    for i in range(3, 16, 2):  # i = 3, 5, 7, 9, 11, 13, 15
        num = _div_trunc(num * z_squared, ONE_36)
        series_sum += _div_trunc(num, i)

    return series_sum * 2


def exp(x: int) -> int:
    """Compute e^x where x is 18-decimal fixed-point.

    Args:
        x: Exponent in 18-decimal fixed-point (can be negative).

    Returns:
        e^x as 18-decimal fixed-point integer.

    Raises:
        InvalidExponent: If x is outside [MIN_NATURAL_EXPONENT, MAX_NATURAL_EXPONENT]
    """
    if not (MIN_NATURAL_EXPONENT <= x <= MAX_NATURAL_EXPONENT):
        raise InvalidExponent(f"Exponent {x} outside valid range")

    if x < 0:
        # e^-x = 1 / e^x
        return (ONE_18 * ONE_18) // exp(-x)

    # Extract large powers of e (18-decimal)
    if x >= X_18[0]:
        x -= X_18[0]
        first_an = A_18[0]
    elif x >= X_18[1]:
        x -= X_18[1]
        first_an = A_18[1]
    else:
        first_an = 1

    # Scale to 20-decimal precision
    x *= 100

    # Extract medium powers of e (20-decimal)
    product = ONE_20
    for i in range(2, 10):
        if x >= X_20[i]:
            x -= X_20[i]
            product = (product * A_20[i]) // ONE_20

    # Taylor series: e^x = 1 + x + x^2/2! + x^3/3! + ... + x^12/12!
    series_sum = ONE_20
    term = x
    series_sum += term

    for i in range(2, 13):
        term = ((term * x) // ONE_20) // i
        series_sum += term

    return (((product * series_sum) // ONE_20) * first_an) // 100


def pow_raw(x: int, y: int) -> int:
    """Compute x^y where both are 18-decimal fixed-point (non-negative).

    This is the raw power function matching Balancer's LogExpMath.pow().

    Args:
        x: Base (non-negative, 18-decimal fixed-point)
        y: Exponent (non-negative, 18-decimal fixed-point)

    Returns:
        x^y as 18-decimal fixed-point

    Raises:
        XOutOfBounds: If x is too large
        YOutOfBounds: If y exceeds MILD_EXPONENT_BOUND
        ProductOutOfBounds: If y * ln(x) is outside valid range
    """
    if y == 0:
        return ONE_18
    if x == 0:
        return 0

    # Check x fits in signed 256-bit (for ln calculation)
    if x >= (1 << 255):
        raise XOutOfBounds(f"Base {x} too large")

    # Check y is below mild bound
    if y >= MILD_EXPONENT_BOUND:
        raise YOutOfBounds(f"Exponent {y} exceeds bound")

    # Compute ln(x) * y
    # Use higher precision _ln_36 when x is close to 1
    # Note: All divisions must use truncation toward zero (like Rust/Solidity)
    if LN_36_LOWER_BOUND < x < LN_36_UPPER_BOUND:
        ln_36_x = _ln_36(x)
        # Rust-style division and remainder (truncate toward zero)
        div1 = _div_trunc(ln_36_x, ONE_18)
        rem1 = ln_36_x - div1 * ONE_18  # Rust remainder
        logx_times_y = div1 * y + _div_trunc(rem1 * y, ONE_18)
    else:
        logx_times_y = _ln(x) * y

    logx_times_y = _div_trunc(logx_times_y, ONE_18)

    if not (MIN_NATURAL_EXPONENT <= logx_times_y <= MAX_NATURAL_EXPONENT):
        raise ProductOutOfBounds(f"Product {logx_times_y} outside valid range")

    return exp(logx_times_y)


# =============================================================================
# Bfp class (wrapper for convenient usage)
# =============================================================================


class Bfp:
    """18-decimal fixed-point number stored as int.

    All values are stored as integers scaled by 10^18.
    Example: 1.5 is stored as 1_500_000_000_000_000_000
    """

    ONE: ClassVar[int] = ONE_18
    MAX_POW_RELATIVE_ERROR: ClassVar[int] = 10000  # 10^-14 relative error

    __slots__ = ("value",)
    __hash__ = None  # type: ignore[assignment]  # Unhashable since we define __eq__

    def __init__(self, value: int) -> None:
        """Create Bfp from raw scaled value."""
        self.value = value

    @classmethod
    def from_wei(cls, wei: int) -> Bfp:
        """Create from raw wei value (already scaled to 18 decimals)."""
        return cls(wei)

    @classmethod
    def from_decimal(cls, d: Decimal) -> Bfp:
        """Create from decimal (will be scaled by 10^18).

        Uses ROUND_HALF_UP for consistent rounding behavior.
        Requires non-negative input (matches Solidity unsigned semantics).
        """
        if d < 0:
            raise ValueError(f"Bfp.from_decimal requires non-negative input, got {d}")
        scaled = (d * cls.ONE).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return cls(int(scaled))

    @classmethod
    def from_int(cls, i: int) -> Bfp:
        """Create from integer (will be scaled by 10^18)."""
        return cls(i * cls.ONE)

    def to_decimal(self) -> Decimal:
        """Convert to Decimal for display."""
        return Decimal(self.value) / Decimal(self.ONE)

    def mul_down(self, other: Bfp) -> Bfp:
        """Multiply with floor rounding: (a * b) // 10^18"""
        return Bfp((self.value * other.value) // self.ONE)

    def mul_up(self, other: Bfp) -> Bfp:
        """Multiply with ceiling rounding."""
        product = self.value * other.value
        if product == 0:
            return Bfp(0)
        return Bfp((product - 1) // self.ONE + 1)

    def div_down(self, other: Bfp) -> Bfp:
        """Divide with floor rounding: (a * 10^18) // b"""
        if other.value == 0:
            raise ZeroDivisionError("Bfp division by zero")
        return Bfp((self.value * self.ONE) // other.value)

    def div_up(self, other: Bfp) -> Bfp:
        """Divide with ceiling rounding."""
        if other.value == 0:
            raise ZeroDivisionError("Bfp division by zero")
        numerator = self.value * self.ONE
        if numerator == 0:
            return Bfp(0)
        return Bfp((numerator - 1) // other.value + 1)

    def complement(self) -> Bfp:
        """Return 1 - self. Clamps to 0 if self > 1."""
        return Bfp(max(0, self.ONE - self.value))

    def add(self, other: Bfp) -> Bfp:
        """Add two Bfp values."""
        return Bfp(self.value + other.value)

    def sub(self, other: Bfp) -> Bfp:
        """Subtract other from self. Clamps to 0 if result would be negative."""
        result = self.value - other.value
        if result < 0:
            return Bfp(0)
        return Bfp(result)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bfp):
            return NotImplemented
        return self.value == other.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Bfp):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Bfp):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Bfp):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Bfp):
            return NotImplemented
        return self.value >= other.value

    def __repr__(self) -> str:
        return f"Bfp({self.value})"

    def __str__(self) -> str:
        return str(self.to_decimal())

    def pow_down(self, exp: Bfp) -> Bfp:
        """Compute self^exp with downward rounding."""
        raw = pow_raw(self.value, exp.value)
        # max_error = mul_up(raw, MAX_POW_RELATIVE_ERROR) + 1
        # Inline mul_up: ((product - 1) // ONE) + 1 when product > 0, else 0
        product = raw * self.MAX_POW_RELATIVE_ERROR
        mul_up_result = ((product - 1) // self.ONE + 1) if product > 0 else 0
        max_error = mul_up_result + 1
        if raw < max_error:
            return Bfp(0)
        return Bfp(raw - max_error)

    def pow_up(self, exp: Bfp) -> Bfp:
        """Compute self^exp with upward rounding."""
        raw = pow_raw(self.value, exp.value)
        # max_error = mul_up(raw, MAX_POW_RELATIVE_ERROR) + 1
        # Inline mul_up: ((product - 1) // ONE) + 1 when product > 0, else 0
        product = raw * self.MAX_POW_RELATIVE_ERROR
        mul_up_result = ((product - 1) // self.ONE + 1) if product > 0 else 0
        max_error = mul_up_result + 1
        return Bfp(raw + max_error)

    def pow_up_v3(self, exp: Bfp) -> Bfp:
        """Compute self^exp with upward rounding (V3Plus variant).

        V3Plus pools use optimized power calculations for common exponents
        to avoid precision loss in the ln/exp round-trip:
        - exponent == 1: return self directly
        - exponent == 2: return self * self (mul_up)
        - exponent == 4: return (self * self) * (self * self) (mul_up)

        For other exponents, falls back to pow_up.
        """
        ONE = Bfp(self.ONE)
        TWO = Bfp(2 * self.ONE)
        FOUR = Bfp(4 * self.ONE)

        if exp == ONE:
            return self
        elif exp == TWO:
            return self.mul_up(self)
        elif exp == FOUR:
            square = self.mul_up(self)
            return square.mul_up(square)
        else:
            return self.pow_up(exp)


# =============================================================================
# Module-level constants for Balancer math
# =============================================================================

MAX_IN_RATIO = Bfp.from_wei(3 * 10**17)  # 0.3 (30%)
MAX_OUT_RATIO = Bfp.from_wei(3 * 10**17)  # 0.3 (30%)
AMP_PRECISION = 1000
