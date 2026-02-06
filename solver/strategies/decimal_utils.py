"""Shared high-precision Decimal utilities for financial calculations.

All financial Decimal comparisons must use high-precision context to avoid
rounding artifacts with very large values (up to 10^77 for uint256).
"""

from __future__ import annotations

import decimal
from decimal import Decimal

# 78 digits of precision — enough for uint256 values (up to ~10^77)
DECIMAL_HIGH_PREC_CONTEXT = decimal.Context(prec=78)


def decimal_lt(a: Decimal, b: Decimal) -> bool:
    """Compare a < b with high precision for exactness."""
    with decimal.localcontext(DECIMAL_HIGH_PREC_CONTEXT):
        return (a - b) < 0


def decimal_le(a: Decimal, b: Decimal) -> bool:
    """Compare a <= b with high precision for exactness."""
    with decimal.localcontext(DECIMAL_HIGH_PREC_CONTEXT):
        return (a - b) <= 0


def decimal_gt(a: Decimal, b: Decimal) -> bool:
    """Compare a > b with high precision for exactness."""
    with decimal.localcontext(DECIMAL_HIGH_PREC_CONTEXT):
        return (a - b) > 0


def decimal_ge(a: Decimal, b: Decimal) -> bool:
    """Compare a >= b with high precision for exactness."""
    with decimal.localcontext(DECIMAL_HIGH_PREC_CONTEXT):
        return (a - b) >= 0


__all__ = [
    "DECIMAL_HIGH_PREC_CONTEXT",
    "decimal_lt",
    "decimal_le",
    "decimal_gt",
    "decimal_ge",
]
