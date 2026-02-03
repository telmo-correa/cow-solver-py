"""Fast fixed-point math - uses Cython if available, falls back to pure Python.

This module provides the same API as fixed_point.py but uses Cython-accelerated
implementations when available.

Usage:
    from solver.math.fixed_point_fast import Bfp, pow_raw, exp, ONE_18
"""

try:
    # Try to import Cython-optimized version
    from solver.math.fixed_point_cy import (
        ONE_18,
        ONE_20,
        ONE_36,
        InvalidExponent,
        LogExpMathError,
        ProductOutOfBounds,
        XOutOfBounds,
        YOutOfBounds,
        _ln,
        _ln_36,
        exp,
        pow_raw,
    )
    from solver.math.fixed_point_cy import (
        BfpCy as Bfp,
    )

    _USING_CYTHON = True
except ImportError:
    # Fall back to pure Python
    from solver.math.fixed_point import (
        ONE_18,
        ONE_20,
        ONE_36,
        Bfp,
        InvalidExponent,
        LogExpMathError,
        ProductOutOfBounds,
        XOutOfBounds,
        YOutOfBounds,
        _ln,
        _ln_36,
        exp,
        pow_raw,
    )

    _USING_CYTHON = False

# Re-export constants that might be needed
from solver.math.fixed_point import (
    AMP_PRECISION,
    MAX_IN_RATIO,
    MAX_OUT_RATIO,
)

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
    "_ln",
    "_ln_36",
    # Constants
    "ONE_18",
    "ONE_20",
    "ONE_36",
    "MAX_IN_RATIO",
    "MAX_OUT_RATIO",
    "AMP_PRECISION",
    # Flag
    "_USING_CYTHON",
]
