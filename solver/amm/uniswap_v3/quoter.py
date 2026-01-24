"""UniswapV3 quoter implementations for swap simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import structlog

from solver.models.types import normalize_address

from .constants import QUOTER_V2_ADDRESS

logger = structlog.get_logger()


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


@dataclass
class QuoteKey:
    """Key for looking up quotes in MockUniswapV3Quoter."""

    token_in: str
    token_out: str
    fee: int
    amount: int
    is_exact_input: bool

    def __hash__(self) -> int:
        return hash(
            (
                normalize_address(self.token_in),
                normalize_address(self.token_out),
                self.fee,
                self.amount,
                self.is_exact_input,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuoteKey):
            return False
        return (
            normalize_address(self.token_in) == normalize_address(other.token_in)
            and normalize_address(self.token_out) == normalize_address(other.token_out)
            and self.fee == other.fee
            and self.amount == other.amount
            and self.is_exact_input == other.is_exact_input
        )


class MockUniswapV3Quoter:
    """Mock quoter for testing without RPC calls.

    Configure with expected quotes, and track calls for assertions.
    """

    def __init__(
        self,
        quotes: dict[QuoteKey, int] | None = None,
        default_rate: tuple[int, int] | None = None,
    ):
        """Initialize mock quoter.

        Args:
            quotes: Mapping of QuoteKey -> result amount for specific quotes
            default_rate: If set, (numerator, denominator) ratio for any unconfigured quote.
                         For exact_input: amount_out = amount_in * num // denom
                         For exact_output: amount_in = (amount_out * denom + num - 1) // num
                         Example: (1, 1) for 1:1 rate, (3, 2) for 1.5x rate
        """
        self.quotes = quotes or {}
        self.default_rate = default_rate
        self.calls: list[tuple[str, str, str, int, int]] = []  # (method, in, out, fee, amount)

    def quote_exact_input(
        self,
        token_in: str,
        token_out: str,
        fee: int,
        amount_in: int,
    ) -> int | None:
        """Get output amount for exact input."""
        self.calls.append(("exact_input", token_in, token_out, fee, amount_in))

        key = QuoteKey(token_in, token_out, fee, amount_in, is_exact_input=True)
        if key in self.quotes:
            return self.quotes[key]

        if self.default_rate is not None:
            num, denom = self.default_rate
            # Floor division for output amount (conservative for receiver)
            return amount_in * num // denom

        return None

    def quote_exact_output(
        self,
        token_in: str,
        token_out: str,
        fee: int,
        amount_out: int,
    ) -> int | None:
        """Get input amount for exact output."""
        self.calls.append(("exact_output", token_in, token_out, fee, amount_out))

        key = QuoteKey(token_in, token_out, fee, amount_out, is_exact_input=False)
        if key in self.quotes:
            return self.quotes[key]

        if self.default_rate is not None:
            num, denom = self.default_rate
            if num > 0:
                # Ceiling division for input amount (conservative for payer)
                return (amount_out * denom + num - 1) // num

        return None


# QuoterV2 ABI - minimal, just the functions we need
QUOTER_V2_ABI = [
    {
        "name": "quoteExactInputSingle",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {
                "name": "params",
                "type": "tuple",
                "components": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
            }
        ],
        "outputs": [
            {"name": "amountOut", "type": "uint256"},
            {"name": "sqrtPriceX96After", "type": "uint160"},
            {"name": "initializedTicksCrossed", "type": "uint32"},
            {"name": "gasEstimate", "type": "uint256"},
        ],
    },
    {
        "name": "quoteExactOutputSingle",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {
                "name": "params",
                "type": "tuple",
                "components": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
            }
        ],
        "outputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "sqrtPriceX96After", "type": "uint160"},
            {"name": "initializedTicksCrossed", "type": "uint32"},
            {"name": "gasEstimate", "type": "uint256"},
        ],
    },
]


class Web3UniswapV3Quoter:
    """Real quoter that calls QuoterV2 contract via RPC.

    This makes actual eth_call requests to the QuoterV2 contract.
    """

    def __init__(self, web3_provider: str, quoter_address: str = QUOTER_V2_ADDRESS):
        """Initialize quoter with web3 provider.

        Args:
            web3_provider: HTTP RPC URL (e.g., "https://eth.llamarpc.com")
            quoter_address: QuoterV2 contract address
        """
        try:
            from web3 import Web3
        except ImportError as e:
            raise ImportError(
                "web3 package required for Web3UniswapV3Quoter. Install with: pip install web3"
            ) from e

        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.quoter = self.w3.eth.contract(
            address=Web3.to_checksum_address(quoter_address),
            abi=QUOTER_V2_ABI,
        )

    def quote_exact_input(
        self,
        token_in: str,
        token_out: str,
        fee: int,
        amount_in: int,
    ) -> int | None:
        """Get output amount for exact input via RPC call."""
        try:
            from web3 import Web3

            result = self.quoter.functions.quoteExactInputSingle(
                (
                    Web3.to_checksum_address(token_in),
                    Web3.to_checksum_address(token_out),
                    amount_in,
                    fee,
                    0,  # sqrtPriceLimitX96 = 0 means no limit
                )
            ).call()

            # Result is (amountOut, sqrtPriceX96After, initializedTicksCrossed, gasEstimate)
            return int(result[0])
        except Exception as e:
            logger.warning(
                "v3_quote_exact_input_failed",
                token_in=token_in,
                token_out=token_out,
                fee=fee,
                amount_in=amount_in,
                error=str(e),
            )
            return None

    def quote_exact_output(
        self,
        token_in: str,
        token_out: str,
        fee: int,
        amount_out: int,
    ) -> int | None:
        """Get input amount for exact output via RPC call."""
        try:
            from web3 import Web3

            result = self.quoter.functions.quoteExactOutputSingle(
                (
                    Web3.to_checksum_address(token_in),
                    Web3.to_checksum_address(token_out),
                    amount_out,
                    fee,
                    0,  # sqrtPriceLimitX96 = 0 means no limit
                )
            ).call()

            # Result is (amountIn, sqrtPriceX96After, initializedTicksCrossed, gasEstimate)
            return int(result[0])
        except Exception as e:
            logger.warning(
                "v3_quote_exact_output_failed",
                token_in=token_in,
                token_out=token_out,
                fee=fee,
                amount_out=amount_out,
                error=str(e),
            )
            return None


__all__ = [
    "UniswapV3Quoter",
    "QuoteKey",
    "MockUniswapV3Quoter",
    "Web3UniswapV3Quoter",
    "QUOTER_V2_ABI",
]
