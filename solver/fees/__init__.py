"""Fee calculation module for the CoW solver.

This module provides centralized fee handling including:
- Solver fee calculation for limit orders
- Fee validation (overflow checking)
- Configurable behavior for edge cases

Usage:
    from solver.fees import FeeCalculator, FeeResult, FeeConfig

    calculator = FeeCalculator()
    result = calculator.calculate_solver_fee(order, gas_estimate, auction)

    if result.is_valid:
        fee = result.fee
    else:
        handle_error(result.error)
"""

from solver.fees.calculator import (
    DEFAULT_FEE_CALCULATOR,
    DefaultFeeCalculator,
    FeeCalculator,
)
from solver.fees.config import DEFAULT_FEE_CONFIG, FeeConfig
from solver.fees.result import FeeError, FeeResult

__all__ = [
    # Calculator
    "FeeCalculator",
    "DefaultFeeCalculator",
    "DEFAULT_FEE_CALCULATOR",
    # Config
    "FeeConfig",
    "DEFAULT_FEE_CONFIG",
    # Result
    "FeeResult",
    "FeeError",
]
