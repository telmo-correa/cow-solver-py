"""Balancer pool parsing.

Functions to parse pool data from auction liquidity sources.
"""

from __future__ import annotations

import contextlib
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Literal

import structlog

from .pools import (
    BalancerStablePool,
    BalancerWeightedPool,
    StableTokenReserve,
    WeightedTokenReserve,
)

if TYPE_CHECKING:
    from solver.models.auction import Liquidity, TokenBalance

logger = structlog.get_logger()


def _get_liquidity_extra(liquidity: Liquidity, key: str, default: Any = None) -> Any:
    """Get extra field from Liquidity model.

    Pydantic v2 stores extra fields in model_extra (dict).
    """
    # Try model_extra first (Pydantic v2)
    if hasattr(liquidity, "model_extra") and liquidity.model_extra and key in liquidity.model_extra:
        return liquidity.model_extra[key]

    # Fall back to direct attribute access
    return getattr(liquidity, key, default)


def _normalize_dict_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Normalize dictionary keys to lowercase for case-insensitive lookup."""
    return {k.lower(): v for k, v in d.items()}


def _parse_balance(
    token_data: TokenBalance,
    liquidity_id: str,
    token_addr: str,
    pool_type: str,
) -> int | None:
    """Parse and validate balance from token data.

    Returns None if balance is invalid or non-positive.
    """
    balance_raw = token_data.get("balance", "0")
    try:
        balance = int(balance_raw)
    except (ValueError, TypeError):
        logger.warning(
            f"{pool_type}_invalid_balance",
            liquidity_id=liquidity_id,
            token=token_addr,
            raw_balance=balance_raw,
        )
        return None

    if balance <= 0:
        logger.debug(
            f"{pool_type}_zero_balance",
            liquidity_id=liquidity_id,
            token=token_addr,
            balance=balance,
        )
        return None

    return balance


def _parse_scaling_factor(
    scaling_factors_dict: dict[str, Any],
    token_addr: str,
    liquidity_id: str,
    pool_type: str,
    token_data: TokenBalance | None = None,
) -> int:
    """Parse scaling factor with fallback to default of 1.

    Looks up scaling factor in two places:
    1. Top-level scaling_factors_dict (Python auction format)
    2. Per-token data with "scalingFactor" key (Rust auction format)

    Falls back to 1 if not found or invalid.
    """
    token_addr_lower = token_addr.lower()
    scaling_raw: Any = scaling_factors_dict.get(token_addr_lower)
    if scaling_raw is None and token_data is not None:
        # Try per-token data (Rust auction format uses "scalingFactor")
        scaling_raw = token_data.get("scalingFactor", "1")
    if scaling_raw is None:
        scaling_raw = "1"
    try:
        return int(scaling_raw)
    except (ValueError, TypeError):
        logger.warning(
            f"{pool_type}_invalid_scaling",
            liquidity_id=liquidity_id,
            token=token_addr,
            raw_scaling=scaling_raw,
        )
        return 1


def _parse_fee(
    liquidity_fee: str | None,
    default_fee: str,
    liquidity_id: str,
    pool_type: str,
) -> Decimal:
    """Parse fee with fallback to default."""
    if liquidity_fee:
        try:
            return Decimal(str(liquidity_fee))
        except (InvalidOperation, TypeError):
            logger.warning(
                f"{pool_type}_invalid_fee",
                liquidity_id=liquidity_id,
                raw_fee=liquidity_fee,
                using_default=default_fee,
            )
    return Decimal(default_fee)


def _parse_gas_estimate(gas_estimate_raw: str | None, default: int) -> int:
    """Parse gas estimate with fallback to default."""
    if gas_estimate_raw:
        with contextlib.suppress(ValueError, TypeError):
            return int(gas_estimate_raw)
    return default


def parse_weighted_pool(liquidity: Liquidity) -> BalancerWeightedPool | None:
    """Parse weightedProduct liquidity into BalancerWeightedPool.

    Args:
        liquidity: Liquidity source from the auction

    Returns:
        BalancerWeightedPool if liquidity is a weighted pool, None otherwise
    """
    if liquidity.kind != "weightedProduct":
        return None

    if liquidity.address is None:
        logger.debug("weighted_pool_missing_address", liquidity_id=liquidity.id)
        return None

    # tokens must be a dict with balance info
    if not isinstance(liquidity.tokens, dict):
        logger.debug("weighted_pool_invalid_tokens", liquidity_id=liquidity.id)
        return None

    # Get balancerPoolId from extra fields
    pool_id = _get_liquidity_extra(liquidity, "balancerPoolId")
    if pool_id is None:
        logger.debug("weighted_pool_missing_pool_id", liquidity_id=liquidity.id)
        return None

    # Try to get weights from top-level dict first (Python format)
    weights_dict_raw = _get_liquidity_extra(liquidity, "weights")
    weights_dict: dict[str, str] = {}
    if weights_dict_raw is not None and isinstance(weights_dict_raw, dict):
        weights_dict = _normalize_dict_keys(weights_dict_raw)

    # Get scaling factors dict (may be top-level or per-token)
    scaling_factors_raw = _get_liquidity_extra(liquidity, "scalingFactors", {})
    scaling_factors_dict = (
        _normalize_dict_keys(scaling_factors_raw) if isinstance(scaling_factors_raw, dict) else {}
    )

    # Get version (defaults to v0)
    version_raw = _get_liquidity_extra(liquidity, "version", "v0")
    version: Literal["v0", "v3Plus"] = "v3Plus" if version_raw == "v3Plus" else "v0"

    # Parse token reserves
    reserves: list[WeightedTokenReserve] = []
    for token_addr, token_data in liquidity.tokens.items():
        if not isinstance(token_data, dict):
            logger.debug(
                "weighted_pool_invalid_token_data",
                liquidity_id=liquidity.id,
                token=token_addr,
            )
            return None

        token_addr_lower = token_addr.lower()

        # Get weight: first try top-level dict, then try per-token data (Rust format)
        weight_raw: str | None = weights_dict.get(token_addr_lower)
        if weight_raw is None:
            # Try to get weight from token data (Rust auction format)
            weight_from_data = token_data.get("weight")
            weight_raw = str(weight_from_data) if weight_from_data is not None else None
        if weight_raw is None:
            logger.debug(
                "weighted_pool_missing_weight",
                liquidity_id=liquidity.id,
                token=token_addr,
            )
            return None

        try:
            weight = Decimal(str(weight_raw))
        except (InvalidOperation, TypeError):
            logger.warning(
                "weighted_pool_invalid_weight",
                liquidity_id=liquidity.id,
                token=token_addr,
                raw_weight=weight_raw,
            )
            return None

        # Validate weight is positive
        if weight <= 0:
            logger.warning(
                "weighted_pool_invalid_weight_value",
                liquidity_id=liquidity.id,
                token=token_addr,
                weight=str(weight),
            )
            return None

        # Parse and validate balance
        balance = _parse_balance(token_data, liquidity.id, token_addr, "weighted_pool")
        if balance is None:
            return None

        # Get scaling factor (supports both top-level dict and per-token data)
        scaling_factor = _parse_scaling_factor(
            scaling_factors_dict, token_addr, liquidity.id, "weighted_pool", token_data
        )

        reserves.append(
            WeightedTokenReserve(
                token=token_addr,
                balance=balance,
                weight=weight,
                scaling_factor=scaling_factor,
            )
        )

    if len(reserves) < 2:
        logger.debug(
            "weighted_pool_insufficient_tokens",
            liquidity_id=liquidity.id,
            token_count=len(reserves),
        )
        return None

    # Validate weight sum is approximately 1.0 (allow small tolerance for rounding)
    total_weight = sum(r.weight for r in reserves)
    if not (Decimal("0.99") <= total_weight <= Decimal("1.01")):
        logger.warning(
            "weighted_pool_invalid_weight_sum",
            liquidity_id=liquidity.id,
            total_weight=str(total_weight),
        )
        return None

    # Sort reserves by token address (required for Balancer)
    reserves.sort(key=lambda r: r.token.lower())

    # Parse fee (default 0.3%)
    fee = _parse_fee(liquidity.fee, "0.003", liquidity.id, "weighted_pool")

    # Get gas estimate
    gas_estimate = _parse_gas_estimate(liquidity.gas_estimate, 88892)

    return BalancerWeightedPool(
        id=liquidity.id,
        address=liquidity.address,
        pool_id=pool_id,
        reserves=tuple(reserves),
        fee=fee,
        version=version,
        gas_estimate=gas_estimate,
    )


def parse_stable_pool(liquidity: Liquidity) -> BalancerStablePool | None:
    """Parse stable liquidity into BalancerStablePool.

    For composable stable pools, automatically filters out the BPT token
    (token address == pool address) during parsing.

    Args:
        liquidity: Liquidity source from the auction

    Returns:
        BalancerStablePool if liquidity is a stable pool, None otherwise
    """
    if liquidity.kind != "stable":
        return None

    if liquidity.address is None:
        logger.debug("stable_pool_missing_address", liquidity_id=liquidity.id)
        return None

    # tokens must be a dict with balance info
    if not isinstance(liquidity.tokens, dict):
        logger.debug("stable_pool_invalid_tokens", liquidity_id=liquidity.id)
        return None

    # Get balancerPoolId from extra fields
    pool_id = _get_liquidity_extra(liquidity, "balancerPoolId")
    if pool_id is None:
        logger.debug("stable_pool_missing_pool_id", liquidity_id=liquidity.id)
        return None

    # Get amplification parameter (required for stable pools)
    amp_raw = _get_liquidity_extra(liquidity, "amplificationParameter")
    if amp_raw is None:
        logger.debug("stable_pool_missing_amp", liquidity_id=liquidity.id)
        return None

    try:
        amplification_parameter = Decimal(str(amp_raw))
    except (InvalidOperation, TypeError):
        logger.warning(
            "stable_pool_invalid_amp",
            liquidity_id=liquidity.id,
            raw_amp=amp_raw,
        )
        return None

    # Validate amplification parameter is positive
    if amplification_parameter <= 0:
        logger.warning(
            "stable_pool_invalid_amp_value",
            liquidity_id=liquidity.id,
            amp=str(amplification_parameter),
        )
        return None

    # Get scaling factors dict (normalized for case-insensitive lookup)
    scaling_factors_raw = _get_liquidity_extra(liquidity, "scalingFactors", {})
    scaling_factors_dict = (
        _normalize_dict_keys(scaling_factors_raw) if isinstance(scaling_factors_raw, dict) else {}
    )

    # Pool address (lowercase) for BPT filtering
    pool_address_lower = liquidity.address.lower()

    # Parse token reserves
    reserves: list[StableTokenReserve] = []
    for token_addr, token_data in liquidity.tokens.items():
        # Skip BPT token (pool's own token in composable stable pools)
        if token_addr.lower() == pool_address_lower:
            continue

        if not isinstance(token_data, dict):
            logger.debug(
                "stable_pool_invalid_token_data",
                liquidity_id=liquidity.id,
                token=token_addr,
            )
            return None

        # Parse and validate balance
        balance = _parse_balance(token_data, liquidity.id, token_addr, "stable_pool")
        if balance is None:
            return None

        # Get scaling factor (supports both top-level dict and per-token data)
        scaling_factor = _parse_scaling_factor(
            scaling_factors_dict, token_addr, liquidity.id, "stable_pool", token_data
        )

        reserves.append(
            StableTokenReserve(
                token=token_addr,
                balance=balance,
                scaling_factor=scaling_factor,
            )
        )

    if len(reserves) < 2:
        logger.debug(
            "stable_pool_insufficient_tokens",
            liquidity_id=liquidity.id,
            token_count=len(reserves),
        )
        return None

    # Sort reserves by token address (required for Balancer)
    reserves.sort(key=lambda r: r.token.lower())

    # Parse fee (default 0.01% for stable pools)
    fee = _parse_fee(liquidity.fee, "0.0001", liquidity.id, "stable_pool")

    # Get gas estimate
    gas_estimate = _parse_gas_estimate(liquidity.gas_estimate, 183520)

    return BalancerStablePool(
        id=liquidity.id,
        address=liquidity.address,
        pool_id=pool_id,
        reserves=tuple(reserves),
        amplification_parameter=amplification_parameter,
        fee=fee,
        gas_estimate=gas_estimate,
    )
