"""Pydantic models for CoW Protocol solution/response data structures.

Based on the OpenAPI spec at:
https://github.com/cowprotocol/services/blob/main/crates/solvers/openapi.yml
"""

from enum import Enum

from pydantic import BaseModel, Field

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


class InteractionKind(str, Enum):
    """The kind of interaction."""

    LIQUIDITY = "liquidity"  # AMM swap
    CUSTOM = "custom"  # Custom interaction


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


class JitTrade(BaseModel):
    """A just-in-time (JIT) liquidity trade.

    JIT trades allow solvers to provide liquidity on-the-fly by creating
    orders that exist only for the duration of a single settlement.

    This is an advanced feature used by sophisticated solvers to:
    1. Provide better prices by temporarily adding liquidity
    2. Capture arbitrage opportunities
    3. Fill orders that can't be matched through existing AMM pools

    Note: This model is defined for API completeness but not yet used
    in the current solver implementation. JIT liquidity requires careful
    economic analysis and is planned for a future phase.
    """

    kind: TradeKind = TradeKind.JIT
    order: dict = Field(description="The JIT order details.")
    executed_amount: Uint256 = Field(alias="executedAmount")

    model_config = {"populate_by_name": True}


class Interaction(BaseModel):
    """An on-chain interaction (contract call) within a solution.

    Typically represents AMM swaps or other DeFi operations.
    """

    kind: InteractionKind = InteractionKind.CUSTOM
    internalize: bool = Field(
        default=False,
        description="Whether to use internal buffers instead of executing on-chain.",
    )
    target: Address = Field(description="Contract address to call.")
    value: Uint256 = Field(default="0", description="ETH value to send.")
    call_data: Bytes = Field(alias="callData", description="Encoded function call.")

    # Optional: link to liquidity source for liquidity interactions
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
    interactions: list[Interaction] = Field(
        default_factory=list,
        description="On-chain interactions (AMM swaps, etc.).",
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
