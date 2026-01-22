"""Balancer error classes.

These errors map to Balancer V2 protocol error codes.
"""


class BalancerError(Exception):
    """Base error for Balancer operations."""

    pass


class MaxInRatioError(BalancerError):
    """Error 304: Input amount exceeds 30% of balance_in."""

    pass


class MaxOutRatioError(BalancerError):
    """Error 305: Output amount exceeds 30% of balance_out."""

    pass


class InvalidFeeError(BalancerError):
    """Swap fee must be in range [0, 1)."""

    pass


class InvalidScalingFactorError(BalancerError):
    """Scaling factor must be positive."""

    pass


class ZeroWeightError(BalancerError):
    """Token weight must be positive."""

    pass


class ZeroBalanceError(BalancerError):
    """Token balance must be positive for swaps."""

    pass


class StableInvariantDidNotConverge(BalancerError):
    """Newton-Raphson iteration for stable invariant D did not converge."""

    pass


class StableGetBalanceDidNotConverge(BalancerError):
    """Newton-Raphson iteration for stable balance Y did not converge."""

    pass
