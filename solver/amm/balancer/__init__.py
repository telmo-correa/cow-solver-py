"""Balancer weighted and stable pool implementations.

This package provides swap calculations for Balancer V2 pools, matching the
Rust baseline solver's implementation exactly.

Pool types supported:
- Weighted Product (V0 and V3Plus)
- Stable pools (StableSwap / Curve-style)
"""

# AMM classes
from .amm import BalancerStableAMM, BalancerWeightedAMM

# Errors
from .errors import (
    BalancerError,
    InvalidFeeError,
    InvalidScalingFactorError,
    MaxInRatioError,
    MaxOutRatioError,
    StableGetBalanceDidNotConverge,
    StableInvariantDidNotConverge,
    ZeroBalanceError,
    ZeroWeightError,
)

# Pool parsing
from .parsing import parse_stable_pool, parse_weighted_pool

# Pool dataclasses
from .pools import (
    BalancerStablePool,
    BalancerWeightedPool,
    StableTokenReserve,
    WeightedTokenReserve,
)

# Scaling helpers
from .scaling import (
    add_swap_fee_amount,
    scale_down_down,
    scale_down_up,
    scale_up,
    subtract_swap_fee_amount,
)

# Stable math
from .stable_math import (
    calculate_invariant,
    filter_bpt_token,
    get_token_balance_given_invariant_and_all_other_balances,
    stable_calc_in_given_out,
    stable_calc_out_given_in,
)

# Weighted math
from .weighted_math import calc_in_given_out, calc_out_given_in

__all__ = [
    # Weighted pool dataclasses
    "WeightedTokenReserve",
    "BalancerWeightedPool",
    # Stable pool dataclasses
    "StableTokenReserve",
    "BalancerStablePool",
    # AMM classes
    "BalancerWeightedAMM",
    "BalancerStableAMM",
    # Weighted pool math functions
    "calc_out_given_in",
    "calc_in_given_out",
    # Stable pool math functions
    "calculate_invariant",
    "get_token_balance_given_invariant_and_all_other_balances",
    "stable_calc_out_given_in",
    "stable_calc_in_given_out",
    # Stable pool helpers
    "filter_bpt_token",
    # Pool parsing
    "parse_weighted_pool",
    "parse_stable_pool",
    # Scaling helpers
    "scale_up",
    "scale_down_down",
    "scale_down_up",
    # Fee helpers
    "subtract_swap_fee_amount",
    "add_swap_fee_amount",
    # Errors
    "BalancerError",
    "MaxInRatioError",
    "MaxOutRatioError",
    "InvalidFeeError",
    "InvalidScalingFactorError",
    "ZeroWeightError",
    "ZeroBalanceError",
    "StableInvariantDidNotConverge",
    "StableGetBalanceDidNotConverge",
]
