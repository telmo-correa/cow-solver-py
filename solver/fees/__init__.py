"""Fee calculation module for the CoW solver.

This module provides centralized fee handling including:
- Solver fee calculation for limit orders
- Fee validation (overflow checking)
- Configurable behavior for edge cases
- Native token price estimation via pool routing

Usage:
    from solver.fees import FeeCalculator, FeeResult, FeeConfig

    calculator = FeeCalculator()
    result = calculator.calculate_solver_fee(order, gas_estimate, auction)

    if result.is_valid:
        fee = result.fee
    else:
        handle_error(result.error)

For pool-based price estimation:
    from solver.fees import PoolBasedPriceEstimator, DefaultFeeCalculator

    estimator = PoolBasedPriceEstimator(router=router, registry=registry)
    calculator = DefaultFeeCalculator(price_estimator=estimator)
"""

from solver.fees.calculator import (
    DEFAULT_FEE_CALCULATOR,
    DefaultFeeCalculator,
    FeeCalculator,
)
from solver.fees.config import DEFAULT_FEE_CONFIG, FeeConfig
from solver.fees.price_estimation import (
    DEFAULT_ESTIMATION_AMOUNT,
    DEFAULT_NATIVE_TOKEN,
    DEFAULT_NATIVE_TOKENS,
    U256_MAX,
    WETH_MAINNET,
    WXDAI_GNOSIS,
    PoolBasedPriceEstimator,
    PriceEstimate,
    PriceEstimator,
    get_token_info,
)
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
    # Price estimation
    "PriceEstimator",
    "PriceEstimate",
    "PoolBasedPriceEstimator",
    "get_token_info",
    "DEFAULT_NATIVE_TOKEN",
    "DEFAULT_NATIVE_TOKENS",
    "DEFAULT_ESTIMATION_AMOUNT",
    "U256_MAX",
    "WETH_MAINNET",
    "WXDAI_GNOSIS",
]
