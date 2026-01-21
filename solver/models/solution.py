"""Pydantic models for CoW Protocol solution/response data structures.

Based on the OpenAPI spec at:
https://github.com/cowprotocol/services/blob/main/crates/solvers/openapi.yml
"""

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, Tag

from solver.models.types import Address, Bytes, OrderUid, Uint256


class TokenAmount(BaseModel):
    """A token and its associated amount.

    Used to represent input/output token flows in interactions.
    """

    token: Address = Field(description="Token address (lowercase, 0x-prefixed)")
    amount: Uint256 = Field(description="Token amount as decimal string")


class TradeKind(str, Enum):
    """The kind of trade execution."""

    FULFILLMENT = "fulfillment"  # Regular order fulfillment
    JIT = "jit"  # Just-in-time liquidity


class Trade(BaseModel):
    """A trade execution within a solution.

    Represents a (partial) fill of an order.
    """

    kind: TradeKind = TradeKind.FULFILLMENT
    order: OrderUid = Field(description="UID of the order being filled.")
    executed_amount: Uint256 = Field(
        alias="executedAmount",
        description="Amount executed. For sell orders: sell amount. For buy orders: buy amount.",
    )
    fee: Uint256 | None = Field(
        default=None,
        description="Fee amount taken from the sell token.",
    )

    model_config = {"populate_by_name": True}


class LiquidityInteraction(BaseModel):
    """Interaction referencing liquidity provided in the auction.

    Used by the Rust solver to reference AMM pools by their auction ID
    rather than encoding the actual swap calldata.
    """

    kind: Literal["liquidity"] = "liquidity"
    internalize: bool = Field(
        default=False,
        description="Whether to use internal buffers instead of on-chain (CIP-2).",
    )
    id: str | int = Field(description="Liquidity source ID from auction.")
    input_token: Address = Field(alias="inputToken", description="Input token address.")
    output_token: Address = Field(alias="outputToken", description="Output token address.")
    input_amount: Uint256 = Field(alias="inputAmount", description="Input amount.")
    output_amount: Uint256 = Field(alias="outputAmount", description="Output amount.")

    model_config = {"populate_by_name": True}


class CustomInteraction(BaseModel):
    """Custom on-chain interaction with encoded calldata.

    Used by the Python solver to encode actual swap calls to DEX routers.
    """

    kind: Literal["custom"] = "custom"
    internalize: bool = Field(
        default=False,
        description="Whether to use internal buffers instead of on-chain (CIP-2).",
    )
    target: Address = Field(description="Contract address to call.")
    value: Uint256 = Field(default="0", description="ETH value to send.")
    call_data: Bytes = Field(alias="callData", description="Encoded function call.")
    allowances: list[dict[str, str]] = Field(
        default_factory=list,
        description="Token approvals needed for this interaction.",
    )
    inputs: list[TokenAmount] = Field(
        default_factory=list,
        description="Input token amounts consumed by this interaction.",
    )
    outputs: list[TokenAmount] = Field(
        default_factory=list,
        description="Output token amounts produced by this interaction.",
    )

    model_config = {"populate_by_name": True}


def _get_interaction_kind(v: dict[str, Any] | LiquidityInteraction | CustomInteraction) -> str:
    """Discriminator function for Interaction union type."""
    if isinstance(v, dict):
        return str(v.get("kind", "custom"))
    return v.kind


# Discriminated union: Pydantic will use the 'kind' field to determine the type
Interaction = Annotated[
    Annotated[LiquidityInteraction, Tag("liquidity")] | Annotated[CustomInteraction, Tag("custom")],
    Discriminator(_get_interaction_kind),
]


class Call(BaseModel):
    """A simple contract call for pre/post interactions."""

    target: Address
    value: Uint256 = Field(default="0")
    call_data: Bytes = Field(alias="callData")

    model_config = {"populate_by_name": True}


class Solution(BaseModel):
    """A proposed solution to the auction.

    Contains clearing prices, trades, and on-chain interactions.
    """

    id: int = Field(description="Unique solution ID within this response.")
    prices: dict[Address, Uint256] = Field(
        default_factory=dict,
        description="Uniform clearing prices. Maps token address to price in sell token terms.",
    )
    trades: list[Trade] = Field(
        default_factory=list,
        description="Order executions in this solution.",
    )
    pre_interactions: list[Call] = Field(
        default_factory=list,
        alias="preInteractions",
        description="Interactions to execute before settlement.",
    )
    interactions: list[Interaction] = Field(
        default_factory=list,
        description="On-chain interactions (AMM swaps, etc.).",
    )
    post_interactions: list[Call] = Field(
        default_factory=list,
        alias="postInteractions",
        description="Interactions to execute after settlement.",
    )
    score: Uint256 | None = Field(
        default=None,
        description="Optional pre-computed score for this solution.",
    )
    gas: int | None = Field(
        default=None,
        description="Estimated gas usage for this solution.",
    )

    model_config = {"populate_by_name": True}

    @classmethod
    def empty(cls, solution_id: int = 0) -> "Solution":
        """Create an empty solution (no trades, valid but scores 0)."""
        return cls(id=solution_id, prices={}, trades=[], interactions=[])


class SolverResponse(BaseModel):
    """The response from the solver's /solve endpoint."""

    solutions: list[Solution] = Field(
        default_factory=list,
        description="List of proposed solutions, ordered by preference.",
    )

    model_config = {"populate_by_name": True}

    @classmethod
    def empty(cls) -> "SolverResponse":
        """Create an empty response (no solutions)."""
        return cls(solutions=[])

    @classmethod
    def with_empty_solution(cls) -> "SolverResponse":
        """Create a response with a single empty solution."""
        return cls(solutions=[Solution.empty()])
