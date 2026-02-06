"""0x Protocol foreign limit order pool representation.

Limit orders are external orders from 0x Protocol that can be used as
liquidity sources. Unlike AMM pools, they have a fixed exchange rate
(no slippage curve) until filled.
"""

from dataclasses import dataclass
from typing import Any

import structlog

from solver.constants import GAS_PER_ZEROEX_ORDER
from solver.models.types import normalize_address

logger = structlog.get_logger()


@dataclass(frozen=True)
class LimitOrderPool:
    """A 0x Protocol foreign limit order used as liquidity.

    In 0x terminology:
    - Maker: The party who created the order (provides makerToken)
    - Taker: The party who fills the order (provides takerToken)

    From the solver's perspective:
    - takerToken is the input token (what we sell into the order)
    - makerToken is the output token (what we receive from the order)

    Attributes:
        id: Liquidity ID from the auction
        address: 0x Exchange contract address
        maker_token: Token the maker provides (output for solver)
        taker_token: Token the maker wants (input for solver)
        maker_amount: Maximum output available from this order
        taker_amount: Maximum input accepted by this order
        taker_token_fee_amount: Protocol fee in taker token
        gas_estimate: Estimated gas cost (~66,358)
    """

    id: str
    address: str
    maker_token: str
    taker_token: str
    maker_amount: int
    taker_amount: int
    taker_token_fee_amount: int
    gas_estimate: int

    def __post_init__(self) -> None:
        """Validate amounts are positive."""
        if self.taker_amount <= 0:
            raise ValueError(f"taker_amount must be positive, got {self.taker_amount}")
        if self.maker_amount <= 0:
            raise ValueError(f"maker_amount must be positive, got {self.maker_amount}")

    @property
    def token0(self) -> str:
        """Return taker_token (input token) normalized."""
        return normalize_address(self.taker_token)

    @property
    def token1(self) -> str:
        """Return maker_token (output token) normalized."""
        return normalize_address(self.maker_token)

    def supports_pair(self, token_in: str, token_out: str) -> bool:
        """Check if this order can route the given token pair.

        Limit orders are unidirectional: taker_token -> maker_token only.

        Args:
            token_in: Input token address
            token_out: Output token address

        Returns:
            True if this order supports the given direction
        """
        return normalize_address(token_in) == normalize_address(
            self.taker_token
        ) and normalize_address(token_out) == normalize_address(self.maker_token)

    @property
    def liquidity_id(self) -> str:
        """Alias for id field, for compatibility with LiquidityInteraction."""
        return self.id


def parse_limit_order(liquidity: Any) -> LimitOrderPool | None:
    """Parse a limit order from auction liquidity.

    Args:
        liquidity: Liquidity object from auction with kind="limitOrder"

    Returns:
        LimitOrderPool if valid, None otherwise
    """
    # Handle both Liquidity objects and dicts
    kind = liquidity.kind if hasattr(liquidity, "kind") else liquidity.get("kind")

    if kind != "limitOrder":
        return None

    try:
        # Access fields - Pydantic models with extra="allow" store extra fields as attributes
        if hasattr(liquidity, "id"):
            # Liquidity object
            return LimitOrderPool(
                id=liquidity.id,
                address=getattr(liquidity, "address", None) or "",
                maker_token=getattr(liquidity, "makerToken", ""),
                taker_token=getattr(liquidity, "takerToken", ""),
                maker_amount=int(getattr(liquidity, "makerAmount", 0)),
                taker_amount=int(getattr(liquidity, "takerAmount", 0)),
                taker_token_fee_amount=int(getattr(liquidity, "takerTokenFeeAmount", 0)),
                gas_estimate=int(getattr(liquidity, "gasEstimate", GAS_PER_ZEROEX_ORDER)),
            )
        else:
            # Dict
            return LimitOrderPool(
                id=liquidity["id"],
                address=liquidity["address"],
                maker_token=liquidity["makerToken"],
                taker_token=liquidity["takerToken"],
                maker_amount=int(liquidity["makerAmount"]),
                taker_amount=int(liquidity["takerAmount"]),
                taker_token_fee_amount=int(liquidity.get("takerTokenFeeAmount", "0")),
                gas_estimate=int(liquidity.get("gasEstimate", str(GAS_PER_ZEROEX_ORDER))),
            )
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.debug("parse_limit_order_failed", error=str(e))
        return None


__all__ = ["LimitOrderPool", "parse_limit_order"]
