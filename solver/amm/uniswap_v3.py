"""UniswapV3 AMM implementation.

UniswapV3 uses concentrated liquidity with tick-based price ranges.
Unlike V2, we use the QuoterV2 contract for swap simulation rather than
implementing the complex tick-crossing math locally.
"""

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import structlog

from solver.models.types import normalize_address

if TYPE_CHECKING:
    from solver.models.auction import Liquidity

logger = structlog.get_logger()


# V3 Fee tiers in Uniswap units (hundredths of a basis point)
# Fee = units / 1,000,000 (e.g., 3000 = 0.3%)
V3_FEE_LOWEST = 100  # 0.01% - stable pairs
V3_FEE_LOW = 500  # 0.05% - stable pairs
V3_FEE_MEDIUM = 3000  # 0.30% - most pairs
V3_FEE_HIGH = 10000  # 1.00% - exotic pairs

V3_FEE_TIERS = [V3_FEE_LOWEST, V3_FEE_LOW, V3_FEE_MEDIUM, V3_FEE_HIGH]

# Tick spacing per fee tier
V3_TICK_SPACING = {
    V3_FEE_LOWEST: 1,
    V3_FEE_LOW: 10,
    V3_FEE_MEDIUM: 60,
    V3_FEE_HIGH: 200,
}

# Gas cost for V3 swaps (from Rust solver)
V3_SWAP_GAS_COST = 106_000

# Contract addresses (mainnet)
QUOTER_V2_ADDRESS = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
SWAP_ROUTER_V2_ADDRESS = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"


@dataclass
class UniswapV3Pool:
    """Represents a UniswapV3 concentrated liquidity pool.

    UniswapV3 pools have liquidity concentrated in price ranges (ticks).
    The pool state includes:
    - Current price (as sqrtPriceX96)
    - Current tick
    - Active liquidity at current tick
    - Net liquidity changes at each initialized tick

    Note: We store the full state for completeness, but swap simulation
    is delegated to the QuoterV2 contract rather than done locally.
    """

    address: str
    token0: str
    token1: str
    fee: int  # Fee in Uniswap units (e.g., 3000 for 0.3%)
    sqrt_price_x96: int  # Current sqrt(price) * 2^96
    liquidity: int  # Current active liquidity
    tick: int  # Current tick index
    liquidity_net: dict[int, int] = field(default_factory=dict)  # tick -> net liquidity
    router: str = SWAP_ROUTER_V2_ADDRESS
    liquidity_id: str | None = None  # ID from auction for LiquidityInteraction
    gas_estimate: int = V3_SWAP_GAS_COST

    @property
    def fee_percent(self) -> float:
        """Fee as percentage (e.g., 0.3 for 0.3%)."""
        return self.fee / 10000

    @property
    def fee_decimal(self) -> float:
        """Fee as decimal (e.g., 0.003 for 0.3%)."""
        return self.fee / 1_000_000

    @property
    def tick_spacing(self) -> int:
        """Get tick spacing for this pool's fee tier."""
        return V3_TICK_SPACING.get(self.fee, 60)  # Default to medium

    def get_token_out(self, token_in: str) -> str:
        """Get the output token for a given input token."""
        token_in_norm = normalize_address(token_in)
        if token_in_norm == normalize_address(self.token0):
            return self.token1
        elif token_in_norm == normalize_address(self.token1):
            return self.token0
        else:
            raise ValueError(f"Token {token_in} not in pool")

    def is_token0(self, token: str) -> bool:
        """Check if token is token0 (determines swap direction)."""
        return normalize_address(token) == normalize_address(self.token0)


class UniswapV3Quoter(Protocol):
    """Protocol for UniswapV3 quoter implementations.

    This allows swapping between real RPC-based quoter and mock quoter for testing.
    """

    def quote_exact_input(
        self,
        token_in: str,
        token_out: str,
        fee: int,
        amount_in: int,
    ) -> int | None:
        """Get output amount for exact input.

        Args:
            token_in: Input token address
            token_out: Output token address
            fee: Pool fee tier (e.g., 3000)
            amount_in: Input amount

        Returns:
            Output amount, or None if quote fails
        """
        ...

    def quote_exact_output(
        self,
        token_in: str,
        token_out: str,
        fee: int,
        amount_out: int,
    ) -> int | None:
        """Get input amount for exact output.

        Args:
            token_in: Input token address
            token_out: Output token address
            fee: Pool fee tier (e.g., 3000)
            amount_out: Desired output amount

        Returns:
            Required input amount, or None if quote fails
        """
        ...


def parse_v3_liquidity(liquidity: "Liquidity") -> UniswapV3Pool | None:
    """Parse UniswapV3 pool from auction liquidity data.

    Args:
        liquidity: Liquidity source from the auction

    Returns:
        UniswapV3Pool if liquidity is concentrated liquidity, None otherwise
    """
    # Only handle concentrated liquidity (UniswapV3-style) pools
    if liquidity.kind != "concentratedLiquidity":
        return None

    if liquidity.address is None:
        logger.debug("v3_pool_missing_address", liquidity_id=liquidity.id)
        return None

    # Get token addresses
    # V3 liquidity has tokens as a list, not dict
    if isinstance(liquidity.tokens, dict):
        token_addresses = list(liquidity.tokens.keys())
    elif isinstance(liquidity.tokens, list):
        token_addresses = liquidity.tokens
    else:
        logger.debug("v3_pool_invalid_tokens", liquidity_id=liquidity.id)
        return None

    if len(token_addresses) != 2:
        logger.debug(
            "v3_pool_wrong_token_count",
            liquidity_id=liquidity.id,
            count=len(token_addresses),
        )
        return None

    # Normalize and order tokens (V3 orders by address like V2)
    token0 = normalize_address(token_addresses[0])
    token1 = normalize_address(token_addresses[1])

    token0_bytes = bytes.fromhex(token0[2:])
    token1_bytes = bytes.fromhex(token1[2:])

    if token0_bytes > token1_bytes:
        token0, token1 = token1, token0

    # Parse V3-specific fields from extra data
    # These come through as additional fields on the Liquidity model
    sqrt_price_x96 = _parse_int_field(liquidity, "sqrtPrice", 0)
    pool_liquidity = _parse_int_field(liquidity, "liquidity", 0)
    tick = _parse_int_field(liquidity, "tick", 0)

    # Parse fee - comes as decimal string (e.g., "0.003" for 0.3%)
    fee = _parse_fee(liquidity)

    # Parse liquidityNet - map of tick index to net liquidity change
    liquidity_net = _parse_liquidity_net(liquidity)

    # Get router address if provided
    router = SWAP_ROUTER_V2_ADDRESS
    if liquidity.router:
        router = normalize_address(liquidity.router)

    # Get gas estimate if provided
    gas_estimate = V3_SWAP_GAS_COST
    if liquidity.gas_estimate:
        with contextlib.suppress(ValueError, TypeError):
            gas_estimate = int(liquidity.gas_estimate)

    return UniswapV3Pool(
        address=normalize_address(liquidity.address),
        token0=token0,
        token1=token1,
        fee=fee,
        sqrt_price_x96=sqrt_price_x96,
        liquidity=pool_liquidity,
        tick=tick,
        liquidity_net=liquidity_net,
        router=router,
        liquidity_id=liquidity.id,
        gas_estimate=gas_estimate,
    )


def _parse_int_field(liquidity: "Liquidity", field_name: str, default: int) -> int:
    """Parse an integer field from liquidity extra data.

    Pydantic's extra="allow" stores unknown fields in model_extra and makes
    them accessible via getattr.
    """
    # Check direct attribute (Pydantic extra="allow" adds fields to model)
    value = getattr(liquidity, field_name, None)

    # Also check model_extra dict (Pydantic stores extras here too)
    if value is None and hasattr(liquidity, "model_extra") and liquidity.model_extra:
        value = liquidity.model_extra.get(field_name)

    if value is None:
        return default

    try:
        return int(value)
    except (ValueError, TypeError):
        logger.debug(
            "v3_pool_parse_int_failed",
            liquidity_id=liquidity.id,
            field=field_name,
            value=value,
        )
        return default


def _parse_fee(liquidity: "Liquidity") -> int:
    """Parse fee from liquidity data.

    Fee can come as:
    - Decimal string "0.003" (0.3%) -> multiply by 1,000,000 -> 3000
    - Integer string "3000" -> use directly

    Returns:
        Fee in Uniswap units (e.g., 3000 for 0.3%)
    """
    # Check the explicit fee field first
    fee_value = liquidity.fee

    # Also check model_extra (for Pydantic extra="allow")
    if fee_value is None and hasattr(liquidity, "model_extra") and liquidity.model_extra:
        fee_value = liquidity.model_extra.get("fee")

    if fee_value is None:
        return V3_FEE_MEDIUM  # Default to 0.3%

    try:
        fee_float = float(fee_value)
        # If it looks like a decimal (< 1), convert to Uniswap units
        if fee_float < 1:
            return int(fee_float * 1_000_000)
        # Otherwise assume it's already in Uniswap units
        return int(fee_float)
    except (ValueError, TypeError):
        logger.debug(
            "v3_pool_parse_fee_failed",
            liquidity_id=liquidity.id,
            fee=fee_value,
        )
        return V3_FEE_MEDIUM


def _parse_liquidity_net(liquidity: "Liquidity") -> dict[int, int]:
    """Parse liquidityNet mapping from liquidity data.

    The liquidityNet maps tick indices to net liquidity changes.
    Format in JSON: {"tick_index": "liquidity_value", ...}
    """
    # Check direct attribute first (Pydantic extra="allow")
    liquidity_net_raw = getattr(liquidity, "liquidityNet", None)

    # Also check model_extra
    if liquidity_net_raw is None and hasattr(liquidity, "model_extra") and liquidity.model_extra:
        liquidity_net_raw = liquidity.model_extra.get("liquidityNet")

    if liquidity_net_raw is None:
        return {}

    if not isinstance(liquidity_net_raw, dict):
        logger.debug(
            "v3_pool_invalid_liquidity_net",
            liquidity_id=liquidity.id,
            type=type(liquidity_net_raw).__name__,
        )
        return {}

    result = {}
    for tick_str, net_str in liquidity_net_raw.items():
        try:
            tick = int(tick_str)
            net = int(net_str)
            result[tick] = net
        except (ValueError, TypeError):
            logger.debug(
                "v3_pool_parse_tick_failed",
                liquidity_id=liquidity.id,
                tick=tick_str,
                net=net_str,
            )
            continue

    return result
