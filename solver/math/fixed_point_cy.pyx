# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""Cython-optimized Balancer Fixed Point (Bfp) math library.

This module provides Cython-accelerated 18-decimal fixed-point arithmetic
matching Balancer's on-chain LogExpMath.sol exactly.

NOTE: Most values use Python objects (arbitrary precision ints) since the
values exceed C's 64-bit range. The speedup comes from reduced interpreter
overhead and type checking.
"""

# Constants stored as Python objects (arbitrary precision)
ONE_18 = 10**18
ONE_20 = 10**20
ONE_36 = 10**36

MAX_NATURAL_EXPONENT = 130 * ONE_18
MIN_NATURAL_EXPONENT = -41 * ONE_18

LN_36_LOWER_BOUND = ONE_18 - 10**17
LN_36_UPPER_BOUND = ONE_18 + 10**17

# Pre-computed constants for digit extraction (18-decimal precision)
X_18_0 = 128 * ONE_18
X_18_1 = 64 * ONE_18

# e^128 and e^64 (huge Python ints)
A_18_0 = 38877084059945950922200000000000000000000000000000000000
A_18_1 = 6235149080811616882910000000

# 20-decimal precision constants
X_20_2 = 3_200_000_000_000_000_000_000
X_20_3 = 1_600_000_000_000_000_000_000
X_20_4 = 800_000_000_000_000_000_000
X_20_5 = 400_000_000_000_000_000_000
X_20_6 = 200_000_000_000_000_000_000
X_20_7 = 100_000_000_000_000_000_000
X_20_8 = 50_000_000_000_000_000_000
X_20_9 = 25_000_000_000_000_000_000
X_20_10 = 12_500_000_000_000_000_000
X_20_11 = 6_250_000_000_000_000_000

# A values for 20-decimal
A_20_2 = 7_896_296_018_268_069_516_100_000_000_000_000
A_20_3 = 888_611_052_050_787_263_676_000_000
A_20_4 = 298_095_798_704_172_827_474_000
A_20_5 = 5_459_815_003_314_423_907_810
A_20_6 = 738_905_609_893_065_022_723
A_20_7 = 271_828_182_845_904_523_536
A_20_8 = 164_872_127_070_012_814_685
A_20_9 = 128_402_541_668_774_148_407
A_20_10 = 113_314_845_306_682_631_683
A_20_11 = 106_449_445_891_785_942_956


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


MILD_EXPONENT_BOUND = (1 << 254) // ONE_20
MAX_POW_RELATIVE_ERROR = 10000


cdef inline object _div_trunc_py(object a, object b):
    """Integer division with truncation toward zero (Python objects)."""
    if b == 0:
        raise ZeroDivisionError("Division by zero in _div_trunc")
    if (a >= 0) == (b >= 0):
        return a // b
    else:
        return -(abs(a) // abs(b))


cpdef object _ln(object a):
    """Compute natural logarithm of a (18-decimal fixed-point).

    Uses digit extraction and Taylor series for arctanh.
    """
    cdef object sum_val, z, z_squared, num, series_sum

    if a < ONE_18:
        return -_ln((ONE_18 * ONE_18) // a)

    sum_val = 0

    # Extract large powers of e (18-decimal precision)
    if a >= A_18_0 * ONE_18:
        a = a // A_18_0
        sum_val = sum_val + X_18_0
    if a >= A_18_1 * ONE_18:
        a = a // A_18_1
        sum_val = sum_val + X_18_1

    # Scale up to 20-decimal precision
    sum_val = sum_val * 100
    a = a * 100

    # Extract medium powers of e (20-decimal precision) - unrolled for speed
    if a >= A_20_2:
        a = (a * ONE_20) // A_20_2
        sum_val = sum_val + X_20_2
    if a >= A_20_3:
        a = (a * ONE_20) // A_20_3
        sum_val = sum_val + X_20_3
    if a >= A_20_4:
        a = (a * ONE_20) // A_20_4
        sum_val = sum_val + X_20_4
    if a >= A_20_5:
        a = (a * ONE_20) // A_20_5
        sum_val = sum_val + X_20_5
    if a >= A_20_6:
        a = (a * ONE_20) // A_20_6
        sum_val = sum_val + X_20_6
    if a >= A_20_7:
        a = (a * ONE_20) // A_20_7
        sum_val = sum_val + X_20_7
    if a >= A_20_8:
        a = (a * ONE_20) // A_20_8
        sum_val = sum_val + X_20_8
    if a >= A_20_9:
        a = (a * ONE_20) // A_20_9
        sum_val = sum_val + X_20_9
    if a >= A_20_10:
        a = (a * ONE_20) // A_20_10
        sum_val = sum_val + X_20_10
    if a >= A_20_11:
        a = (a * ONE_20) // A_20_11
        sum_val = sum_val + X_20_11

    # Taylor series for remaining fraction using arctanh
    z = ((a - ONE_20) * ONE_20) // (a + ONE_20)
    z_squared = (z * z) // ONE_20

    num = z
    series_sum = num

    # 6 terms: z + z^3/3 + z^5/5 + z^7/7 + z^9/9 + z^11/11 (unrolled)
    num = (num * z_squared) // ONE_20
    series_sum = series_sum + num // 3
    num = (num * z_squared) // ONE_20
    series_sum = series_sum + num // 5
    num = (num * z_squared) // ONE_20
    series_sum = series_sum + num // 7
    num = (num * z_squared) // ONE_20
    series_sum = series_sum + num // 9
    num = (num * z_squared) // ONE_20
    series_sum = series_sum + num // 11

    series_sum = series_sum * 2

    return (sum_val + series_sum) // 100


cpdef object _ln_36(object x):
    """Compute natural logarithm with 36-decimal precision."""
    cdef object z, z_squared, num, series_sum

    x = x * ONE_18  # Scale to 36 decimals

    z = _div_trunc_py((x - ONE_36) * ONE_36, x + ONE_36)
    z_squared = _div_trunc_py(z * z, ONE_36)

    num = z
    series_sum = num

    # 8 terms (unrolled)
    num = _div_trunc_py(num * z_squared, ONE_36)
    series_sum = series_sum + _div_trunc_py(num, 3)
    num = _div_trunc_py(num * z_squared, ONE_36)
    series_sum = series_sum + _div_trunc_py(num, 5)
    num = _div_trunc_py(num * z_squared, ONE_36)
    series_sum = series_sum + _div_trunc_py(num, 7)
    num = _div_trunc_py(num * z_squared, ONE_36)
    series_sum = series_sum + _div_trunc_py(num, 9)
    num = _div_trunc_py(num * z_squared, ONE_36)
    series_sum = series_sum + _div_trunc_py(num, 11)
    num = _div_trunc_py(num * z_squared, ONE_36)
    series_sum = series_sum + _div_trunc_py(num, 13)
    num = _div_trunc_py(num * z_squared, ONE_36)
    series_sum = series_sum + _div_trunc_py(num, 15)

    return series_sum * 2


cpdef object exp(object x):
    """Compute e^x where x is 18-decimal fixed-point."""
    cdef object first_an, product, series_sum, term
    cdef int i

    if not (MIN_NATURAL_EXPONENT <= x <= MAX_NATURAL_EXPONENT):
        raise InvalidExponent(f"Exponent {x} outside valid range")

    if x < 0:
        return (ONE_18 * ONE_18) // exp(-x)

    # Extract large powers of e (18-decimal)
    if x >= X_18_0:
        x = x - X_18_0
        first_an = A_18_0
    elif x >= X_18_1:
        x = x - X_18_1
        first_an = A_18_1
    else:
        first_an = 1

    # Scale to 20-decimal precision
    x = x * 100

    # Extract medium powers of e (20-decimal) - unrolled
    product = ONE_20
    if x >= X_20_2:
        x = x - X_20_2
        product = (product * A_20_2) // ONE_20
    if x >= X_20_3:
        x = x - X_20_3
        product = (product * A_20_3) // ONE_20
    if x >= X_20_4:
        x = x - X_20_4
        product = (product * A_20_4) // ONE_20
    if x >= X_20_5:
        x = x - X_20_5
        product = (product * A_20_5) // ONE_20
    if x >= X_20_6:
        x = x - X_20_6
        product = (product * A_20_6) // ONE_20
    if x >= X_20_7:
        x = x - X_20_7
        product = (product * A_20_7) // ONE_20
    if x >= X_20_8:
        x = x - X_20_8
        product = (product * A_20_8) // ONE_20
    if x >= X_20_9:
        x = x - X_20_9
        product = (product * A_20_9) // ONE_20

    # Taylor series: e^x = 1 + x + x^2/2! + ... + x^12/12! (unrolled)
    series_sum = ONE_20
    term = x
    series_sum = series_sum + term

    term = ((term * x) // ONE_20) // 2
    series_sum = series_sum + term
    term = ((term * x) // ONE_20) // 3
    series_sum = series_sum + term
    term = ((term * x) // ONE_20) // 4
    series_sum = series_sum + term
    term = ((term * x) // ONE_20) // 5
    series_sum = series_sum + term
    term = ((term * x) // ONE_20) // 6
    series_sum = series_sum + term
    term = ((term * x) // ONE_20) // 7
    series_sum = series_sum + term
    term = ((term * x) // ONE_20) // 8
    series_sum = series_sum + term
    term = ((term * x) // ONE_20) // 9
    series_sum = series_sum + term
    term = ((term * x) // ONE_20) // 10
    series_sum = series_sum + term
    term = ((term * x) // ONE_20) // 11
    series_sum = series_sum + term
    term = ((term * x) // ONE_20) // 12
    series_sum = series_sum + term

    return (((product * series_sum) // ONE_20) * first_an) // 100


cpdef object pow_raw(object x, object y):
    """Compute x^y where both are 18-decimal fixed-point (non-negative)."""
    cdef object logx_times_y, ln_36_x, div1, rem1

    if y == 0:
        return ONE_18
    if x == 0:
        return 0

    if x >= (1 << 255):
        raise XOutOfBounds(f"Base {x} too large")

    if y >= MILD_EXPONENT_BOUND:
        raise YOutOfBounds(f"Exponent {y} exceeds bound")

    # Compute ln(x) * y
    if LN_36_LOWER_BOUND < x < LN_36_UPPER_BOUND:
        ln_36_x = _ln_36(x)
        div1 = _div_trunc_py(ln_36_x, ONE_18)
        rem1 = ln_36_x - div1 * ONE_18
        logx_times_y = div1 * y + _div_trunc_py(rem1 * y, ONE_18)
    else:
        logx_times_y = _ln(x) * y

    logx_times_y = _div_trunc_py(logx_times_y, ONE_18)

    if not (MIN_NATURAL_EXPONENT <= logx_times_y <= MAX_NATURAL_EXPONENT):
        raise ProductOutOfBounds(f"Product {logx_times_y} outside valid range")

    return exp(logx_times_y)


cdef inline object _get_bfp_value(obj):
    """Extract value from any Bfp-like object."""
    return obj.value


cdef class BfpCy:
    """Cython-optimized 18-decimal fixed-point number."""

    cdef public object value

    def __init__(self, object value):
        self.value = value

    @staticmethod
    def from_wei(object wei):
        return BfpCy(wei)

    @staticmethod
    def from_decimal(dec):
        """Create from decimal (will be scaled by 10^18)."""
        from decimal import ROUND_HALF_UP, Decimal
        scaled = (dec * ONE_18).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return BfpCy(int(scaled))

    @staticmethod
    def from_int(i):
        """Create from integer (will be scaled by 10^18)."""
        return BfpCy(i * ONE_18)

    def to_decimal(self):
        """Convert to Decimal for display."""
        from decimal import Decimal
        return Decimal(self.value) / Decimal(ONE_18)

    def mul_down(self, other):
        """Multiply with floor rounding: (a * b) // 10^18"""
        cdef object other_val = _get_bfp_value(other)
        return BfpCy((self.value * other_val) // ONE_18)

    def mul_up(self, other):
        """Multiply with ceiling rounding."""
        cdef object other_val = _get_bfp_value(other)
        cdef object product = self.value * other_val
        if product == 0:
            return BfpCy(0)
        return BfpCy((product - 1) // ONE_18 + 1)

    def div_down(self, other):
        """Divide with floor rounding: (a * 10^18) // b"""
        cdef object other_val = _get_bfp_value(other)
        if other_val == 0:
            raise ZeroDivisionError("Bfp division by zero")
        return BfpCy((self.value * ONE_18) // other_val)

    def div_up(self, other):
        """Divide with ceiling rounding."""
        cdef object other_val = _get_bfp_value(other)
        if other_val == 0:
            raise ZeroDivisionError("Bfp division by zero")
        cdef object numerator = self.value * ONE_18
        if numerator == 0:
            return BfpCy(0)
        return BfpCy((numerator - 1) // other_val + 1)

    def complement(self):
        """Return 1 - self."""
        return BfpCy(ONE_18 - self.value)

    def add(self, other):
        """Add two Bfp values."""
        cdef object other_val = _get_bfp_value(other)
        return BfpCy(self.value + other_val)

    def sub(self, other):
        """Subtract other from self."""
        cdef object other_val = _get_bfp_value(other)
        return BfpCy(self.value - other_val)

    def __eq__(self, other):
        if not hasattr(other, 'value'):
            return NotImplemented
        return self.value == other.value

    def __lt__(self, other):
        if not hasattr(other, 'value'):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other):
        if not hasattr(other, 'value'):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other):
        if not hasattr(other, 'value'):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other):
        if not hasattr(other, 'value'):
            return NotImplemented
        return self.value >= other.value

    def __repr__(self):
        return f"BfpCy({self.value})"

    def pow_down(self, exp_val):
        """Compute self^exp with downward rounding."""
        cdef object exp_v = _get_bfp_value(exp_val)
        cdef object raw = pow_raw(self.value, exp_v)
        cdef object product = raw * MAX_POW_RELATIVE_ERROR
        cdef object mul_up_result = ((product - 1) // ONE_18 + 1) if product > 0 else 0
        cdef object max_error = mul_up_result + 1
        if raw < max_error:
            return BfpCy(0)
        return BfpCy(raw - max_error)

    def pow_up(self, exp_val):
        """Compute self^exp with upward rounding."""
        cdef object exp_v = _get_bfp_value(exp_val)
        cdef object raw = pow_raw(self.value, exp_v)
        cdef object product = raw * MAX_POW_RELATIVE_ERROR
        cdef object mul_up_result = ((product - 1) // ONE_18 + 1) if product > 0 else 0
        cdef object max_error = mul_up_result + 1
        return BfpCy(raw + max_error)

    def pow_up_v3(self, exp_val):
        """Compute self^exp with upward rounding (V3Plus variant)."""
        cdef object exp_v = _get_bfp_value(exp_val)
        if exp_v == ONE_18:
            return self
        elif exp_v == 2 * ONE_18:
            return self.mul_up(self)
        elif exp_v == 4 * ONE_18:
            square = self.mul_up(self)
            return square.mul_up(square)
        else:
            return self.pow_up(exp_val)
