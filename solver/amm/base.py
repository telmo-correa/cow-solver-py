"""Base classes for AMM implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from solver.constants import POOL_SWAP_GAS_COST


@dataclass
class SwapResult:
    """Result of simulating a swap through an AMM."""

    amount_in: int
    amount_out: int
    pool_address: str
    token_in: str
    token_out: str
    # Gas estimate for this swap (uses constant for consistency)
    gas_estimate: int = POOL_SWAP_GAS_COST


class AMM(ABC):
    """Abstract base class for AMM implementations.

    Implementations may extend the base method signatures with additional
    optional parameters. For example, UniswapV2 adds a fee_multiplier
    parameter to get_amount_out() and get_amount_in() to support pools
    with different fee tiers.
    """

    @abstractmethod
    def get_amount_out(
        self,
        amount_in: int,
        reserve_in: int,
        reserve_out: int,
    ) -> int:
        """Calculate output amount for a given input.

        Args:
            amount_in: Input token amount
            reserve_in: Reserve of input token in pool
            reserve_out: Reserve of output token in pool

        Returns:
            Output token amount

        Note:
            Implementations may add optional parameters (e.g., fee_multiplier)
            to customize the calculation for different pool configurations.
        """
        ...

    @abstractmethod
    def get_amount_in(
        self,
        amount_out: int,
        reserve_in: int,
        reserve_out: int,
    ) -> int:
        """Calculate required input for a desired output.

        Args:
            amount_out: Desired output token amount
            reserve_in: Reserve of input token in pool
            reserve_out: Reserve of output token in pool

        Returns:
            Required input token amount

        Note:
            Implementations may add optional parameters (e.g., fee_multiplier)
            to customize the calculation for different pool configurations.
        """
        ...

    @abstractmethod
    def encode_swap(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        amount_out_min: int,
        recipient: str,
    ) -> tuple[str, str]:
        """Encode a swap as calldata.

        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Amount of input token
            amount_out_min: Minimum output (slippage protection)
            recipient: Address to receive output tokens

        Returns:
            Tuple of (target_address, calldata)
        """
        ...


@runtime_checkable
class SwapCalculator(Protocol):
    """Protocol for AMM swap calculators.

    This defines the standard interface for simulating swaps and calculating
    partial fills across all pool types (V2, V3, Balancer weighted/stable).

    All methods take a pool object and token addresses, making the interface
    uniform regardless of the underlying pool type.
    """

    def simulate_swap(
        self,
        pool: Any,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> SwapResult | None:
        """Simulate a swap through a pool (exact input).

        Args:
            pool: The liquidity pool
            token_in: Input token address
            token_out: Output token address
            amount_in: Amount of input token

        Returns:
            SwapResult with amounts and pool info, or None if swap fails
        """
        ...

    def simulate_swap_exact_output(
        self,
        pool: Any,
        token_in: str,
        token_out: str,
        amount_out: int,
    ) -> SwapResult | None:
        """Simulate a swap to get exact output amount.

        Args:
            pool: The liquidity pool
            token_in: Input token address
            token_out: Output token address
            amount_out: Desired output amount

        Returns:
            SwapResult with required input and desired output, or None if swap fails
        """
        ...

    def max_fill_sell_order(
        self,
        pool: Any,
        token_in: str,
        token_out: str,
        sell_amount: int,
        buy_amount: int,
    ) -> int:
        """Calculate maximum input for a sell order that satisfies the limit price.

        For a sell order, the user wants to sell tokens and receive at least
        a minimum amount. This calculates the maximum input amount where the
        output rate still satisfies the limit: output/input >= buy_amount/sell_amount.

        Args:
            pool: The liquidity pool
            token_in: Input token address
            token_out: Output token address
            sell_amount: Order's sell amount (used for limit rate)
            buy_amount: Order's minimum buy amount (used for limit rate)

        Returns:
            Maximum input amount that satisfies the limit, or 0 if impossible
        """
        ...

    def max_fill_buy_order(
        self,
        pool: Any,
        token_in: str,
        token_out: str,
        sell_amount: int,
        buy_amount: int,
    ) -> int:
        """Calculate maximum output for a buy order that satisfies the limit price.

        For a buy order, the user wants to receive a specific amount and is
        willing to pay up to a maximum. This calculates the maximum output
        where the input rate still satisfies the limit: input/output <= sell_amount/buy_amount.

        Args:
            pool: The liquidity pool
            token_in: Input token address
            token_out: Output token address
            sell_amount: Order's maximum sell amount (used for limit rate)
            buy_amount: Order's desired buy amount (used for limit rate)

        Returns:
            Maximum output amount that satisfies the limit, or 0 if impossible
        """
        ...
