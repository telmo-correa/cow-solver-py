"""Base classes for AMM implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
