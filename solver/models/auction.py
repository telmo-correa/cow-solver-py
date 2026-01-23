"""Pydantic models for CoW Protocol auction data structures.

Based on the OpenAPI spec at:
https://github.com/cowprotocol/services/blob/main/crates/solvers/openapi.yml
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from solver.models.types import Address, Bytes, OrderUid, Uint256, normalize_address


class TokenBalance(TypedDict):
    """Token balance information in liquidity pools."""

    balance: str


class OrderKind(str, Enum):
    """Whether the order is a sell or buy order."""

    SELL = "sell"
    BUY = "buy"


class OrderClass(str, Enum):
    """Classification of the order."""

    MARKET = "market"
    LIMIT = "limit"
    LIQUIDITY = "liquidity"


class SigningScheme(str, Enum):
    """How the order was signed."""

    EIP712 = "eip712"
    ETHSIGN = "ethsign"
    PRESIGN = "presign"
    EIP1271 = "eip1271"


class SellTokenBalance(str, Enum):
    """Where to source the sell token balance."""

    ERC20 = "erc20"
    INTERNAL = "internal"
    EXTERNAL = "external"


class BuyTokenBalance(str, Enum):
    """Where to send the buy token."""

    ERC20 = "erc20"
    INTERNAL = "internal"


class Token(BaseModel):
    """Token metadata included in the auction."""

    # Note: Most tokens use 18 decimals, but some use different values (USDC=6, WBTC=8)
    # Some exotic tokens may use more than 18 decimals, so we allow up to 77 (max for uint256)
    decimals: int | None = Field(default=None, ge=0, le=77)
    symbol: str | None = None
    reference_price: Uint256 | None = Field(
        default=None,
        alias="referencePrice",
        description="Price of this token in the native asset (ETH), as a fraction.",
    )
    available_balance: Uint256 = Field(
        alias="availableBalance",
        description="Available balance for this token across all orders.",
    )
    trusted: bool = Field(
        default=False,
        description="Whether the token is trusted (on the allow list).",
    )

    model_config = {"populate_by_name": True}


class FeePolicy(BaseModel):
    """Fee policy for an order."""

    kind: str
    surplus_factor: float | None = Field(default=None, alias="surplusFactor")
    surplus_max_volume_factor: float | None = Field(default=None, alias="surplusMaxVolumeFactor")
    volume_factor: float | None = Field(default=None, alias="volumeFactor")
    price_improvement_factor: float | None = Field(default=None, alias="priceImprovementFactor")
    price_improvement_max_volume_factor: float | None = Field(
        default=None, alias="priceImprovementMaxVolumeFactor"
    )

    model_config = {"populate_by_name": True}


class Signature(BaseModel):
    """Order signature data.

    Note: This model is defined for API completeness but the Order model
    currently uses `signature: Bytes | None` for simplicity. In a full
    implementation, orders would include structured signature data.
    """

    scheme: SigningScheme
    data: Bytes


class Order(BaseModel):
    """An order in the auction batch."""

    uid: OrderUid = Field(description="Unique identifier for the order.")
    sell_token: Address = Field(alias="sellToken")
    buy_token: Address = Field(alias="buyToken")
    sell_amount: Uint256 = Field(alias="sellAmount")
    buy_amount: Uint256 = Field(alias="buyAmount")
    full_sell_amount: Uint256 | None = Field(default=None, alias="fullSellAmount")
    full_buy_amount: Uint256 | None = Field(default=None, alias="fullBuyAmount")
    fee_amount: Uint256 = Field(default="0", alias="feeAmount")
    kind: OrderKind
    partially_fillable: bool = Field(default=False, alias="partiallyFillable")
    class_: OrderClass = Field(alias="class")
    # For remainder orders: tracks the original order's UID for fill merging
    original_uid: OrderUid | None = Field(
        default=None,
        description="For remainder orders, the UID of the original order. Used for fill merging.",
    )

    # Optional fields
    app_data: Bytes | None = Field(default=None, alias="appData")
    signing_scheme: SigningScheme | None = Field(default=None, alias="signingScheme")
    signature: Bytes | None = None
    receiver: Address | None = None
    owner: Address | None = None
    valid_to: int | None = Field(default=None, alias="validTo")
    sell_token_balance: SellTokenBalance = Field(
        default=SellTokenBalance.ERC20, alias="sellTokenBalance"
    )
    buy_token_balance: BuyTokenBalance = Field(
        default=BuyTokenBalance.ERC20, alias="buyTokenBalance"
    )
    pre_interactions: list[dict[str, Any]] = Field(default_factory=list, alias="preInteractions")
    post_interactions: list[dict[str, Any]] = Field(default_factory=list, alias="postInteractions")
    fee_policies: list[FeePolicy] | None = Field(default=None, alias="feePolicies")

    model_config = {"populate_by_name": True}

    @property
    def sell_amount_int(self) -> int:
        """Sell amount as integer for calculations."""
        return int(self.sell_amount)

    @property
    def buy_amount_int(self) -> int:
        """Buy amount as integer for calculations."""
        return int(self.buy_amount)

    @property
    def full_sell_amount_int(self) -> int:
        """Full sell amount as integer (falls back to sell_amount if not set)."""
        if self.full_sell_amount is not None:
            return int(self.full_sell_amount)
        return int(self.sell_amount)

    @property
    def full_buy_amount_int(self) -> int:
        """Full buy amount as integer (falls back to buy_amount if not set)."""
        if self.full_buy_amount is not None:
            return int(self.full_buy_amount)
        return int(self.buy_amount)

    @property
    def is_sell_order(self) -> bool:
        """Return True if this is a sell order."""
        return self.kind == OrderKind.SELL

    @property
    def is_buy_order(self) -> bool:
        """Return True if this is a buy order."""
        return self.kind == OrderKind.BUY

    @property
    def limit_price(self) -> float:
        """Return the limit price (buy_amount / sell_amount for sell orders)."""
        sell = self.sell_amount_int
        buy = self.buy_amount_int
        if sell == 0:
            return float("inf")
        return buy / sell


class Liquidity(BaseModel):
    """On-chain liquidity source available for routing.

    Supports both simplified format (tokens as list) and full Rust solver format
    (tokens as dict with balances).

    For constantProduct pools, the full format includes:
    - address: Pool contract address
    - router: DEX router address
    - gasEstimate: Estimated gas for swap
    - tokens: Dict mapping token address to {balance: amount}
    - fee: Fee as decimal string (e.g., "0.003" for 0.3%)
    """

    id: str
    kind: str
    # Tokens: list of addresses OR dict mapping address to balance info
    # Note: For limit orders, tokens are specified via makerToken/takerToken instead
    tokens: list[Address] | dict[Address, TokenBalance] | None = None
    # Optional fields for full format
    address: Address | None = None
    router: Address | None = None
    # Note: gas_estimate uses str (not Uint256) to allow fallback to default
    # when invalid values are provided. Parsing code handles the conversion.
    gas_estimate: str | None = Field(default=None, alias="gasEstimate")
    fee: str | None = None
    # Additional fields depend on the AMM type
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow", "populate_by_name": True}


class AuctionInstance(BaseModel):
    """The auction instance sent to the solver.

    Contains all orders in the batch and available liquidity sources.
    """

    id: str | None = Field(
        default=None,
        description="Auction ID. Null for quote requests.",
    )
    tokens: dict[Address, Token] = Field(
        default_factory=dict,
        description="Token metadata keyed by address.",
    )
    orders: list[Order] = Field(
        default_factory=list,
        description="Orders to be settled in this auction.",
    )
    liquidity: list[Liquidity] = Field(
        default_factory=list,
        description="Available on-chain liquidity.",
    )
    effective_gas_price: Uint256 | None = Field(
        default=None,
        alias="effectiveGasPrice",
        description="Current gas price in wei.",
    )
    deadline: datetime | None = Field(
        default=None,
        description="Deadline for solution submission.",
    )
    surplus_capturing_jit_order_owners: list[Address] = Field(
        default_factory=list,
        alias="surplusCapturingJitOrderOwners",
    )

    model_config = {"populate_by_name": True}

    @property
    def order_count(self) -> int:
        """Return the number of orders in this auction."""
        return len(self.orders)

    @property
    def token_pairs(self) -> set[tuple[str, str]]:
        """Return the set of unique token pairs being traded.

        Pairs are normalized to lowercase and sorted for consistency.
        """
        pairs: set[tuple[str, str]] = set()
        for order in self.orders:
            # Normalize addresses to lowercase and sort for consistent ordering
            sell_norm = normalize_address(order.sell_token)
            buy_norm = normalize_address(order.buy_token)
            pair = tuple(sorted([sell_norm, buy_norm]))
            pairs.add(pair)  # type: ignore
        return pairs

    def orders_for_pair(self, token_a: Address, token_b: Address) -> list[Order]:
        """Return all orders trading between two tokens.

        Args:
            token_a: First token address (any case)
            token_b: Second token address (any case)

        Returns:
            List of orders trading between these tokens (in either direction)
        """
        pair = {normalize_address(token_a), normalize_address(token_b)}
        return [
            order
            for order in self.orders
            if {normalize_address(order.sell_token), normalize_address(order.buy_token)} == pair
        ]
