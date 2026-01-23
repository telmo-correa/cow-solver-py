"""UniswapV2 AMM implementation.

UniswapV2 uses the constant product formula: x * y = k
With a 0.3% fee on input amounts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import structlog
from eth_abi import encode  # type: ignore[attr-defined]

from solver.amm.base import AMM, SwapResult
from solver.constants import POOL_SWAP_GAS_COST
from solver.models.types import is_valid_address, normalize_address

if TYPE_CHECKING:
    from solver.models.auction import Liquidity

logger = structlog.get_logger()


@dataclass
class UniswapV2Pool:
    """Represents a UniswapV2 liquidity pool."""

    address: str
    token0: str
    token1: str
    reserve0: int
    reserve1: int
    # Fee in basis points (30 = 0.3%)
    # Standard UniswapV2 is 30 bps, but some forks use different fees
    fee_bps: int = 30
    # Liquidity ID from the auction (for LiquidityInteraction)
    liquidity_id: str | None = None

    @property
    def fee_multiplier(self) -> int:
        """Fee multiplier for AMM math (1000 - fee_bps/10).

        For 30 bps (0.3%), this returns 997.
        Used in the formula: amount_in_with_fee = amount_in * fee_multiplier
        """
        return 1000 - (self.fee_bps // 10)

    def get_reserves(self, token_in: str) -> tuple[int, int]:
        """Get reserves ordered as (reserve_in, reserve_out)."""
        if token_in.lower() == self.token0.lower():
            return self.reserve0, self.reserve1
        elif token_in.lower() == self.token1.lower():
            return self.reserve1, self.reserve0
        else:
            raise ValueError(f"Token {token_in} not in pool")

    def get_token_out(self, token_in: str) -> str:
        """Get the output token for a given input token."""
        if token_in.lower() == self.token0.lower():
            return self.token1
        elif token_in.lower() == self.token1.lower():
            return self.token0
        else:
            raise ValueError(f"Token {token_in} not in pool")


class UniswapV2(AMM):
    """UniswapV2 AMM math and encoding.

    Formula: amount_out = (amount_in * 997 * reserve_out) / (reserve_in * 1000 + amount_in * 997)

    The 997/1000 factor accounts for the 0.3% fee.
    """

    # UniswapV2 Router02 address on mainnet (lowercase for consistency)
    ROUTER_ADDRESS: ClassVar[str] = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d"

    # Function selectors
    SWAP_EXACT_TOKENS_SELECTOR: ClassVar[str] = "0x38ed1739"  # swapExactTokensForTokens
    SWAP_TOKENS_FOR_EXACT_SELECTOR: ClassVar[str] = "0x8803dbee"  # swapTokensForExactTokens

    def get_amount_out(
        self,
        amount_in: int,
        reserve_in: int,
        reserve_out: int,
        fee_multiplier: int = 997,
    ) -> int:
        """Calculate output amount using constant product formula.

        Formula: amount_out = (in * fee * res_out) / (res_in * 1000 + in * fee)

        Args:
            amount_in: Input token amount
            reserve_in: Reserve of input token in pool
            reserve_out: Reserve of output token in pool
            fee_multiplier: Fee multiplier (default 997 for 0.3% fee)

        Returns:
            Output token amount
        """
        if amount_in <= 0:
            return 0
        if reserve_in <= 0 or reserve_out <= 0:
            return 0

        amount_in_with_fee = amount_in * fee_multiplier
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in * 1000 + amount_in_with_fee

        return numerator // denominator

    def get_amount_in(
        self,
        amount_out: int,
        reserve_in: int,
        reserve_out: int,
        fee_multiplier: int = 997,
    ) -> int:
        """Calculate required input for desired output.

        Formula: amount_in = (res_in * out * 1000) / ((res_out - out) * fee) + 1

        Args:
            amount_out: Desired output token amount
            reserve_in: Reserve of input token in pool
            reserve_out: Reserve of output token in pool
            fee_multiplier: Fee multiplier (default 997 for 0.3% fee)

        Returns:
            Required input token amount
        """
        if amount_out <= 0:
            return 0
        if reserve_in <= 0 or reserve_out <= 0:
            return 0
        if amount_out >= reserve_out:
            # Can't extract more than the reserve
            return 2**256 - 1  # Max uint256

        numerator = reserve_in * amount_out * 1000
        denominator = (reserve_out - amount_out) * fee_multiplier

        return (numerator // denominator) + 1

    def max_fill_sell_order(
        self,
        reserve_in: int,
        reserve_out: int,
        sell_amount: int,
        buy_amount: int,
        fee_multiplier: int = 997,
    ) -> int:
        """Calculate maximum input for a sell order that satisfies the limit price.

        For a sell order, the user wants to sell tokens and receive at least
        a minimum amount. This calculates the maximum input amount where the
        output rate still satisfies the limit: output/input >= buy_amount/sell_amount.

        The formula is derived from:
        - output(x) = (x * fee * R_out) / (R_in * 1000 + x * fee)
        - Constraint: output(x) / x >= buy_amount / sell_amount
        - Solving: x <= R_out * sell_amount / buy_amount - R_in * 1000 / fee

        Args:
            reserve_in: Pool reserve of input token
            reserve_out: Pool reserve of output token
            sell_amount: Order's sell amount (used for limit rate)
            buy_amount: Order's minimum buy amount (used for limit rate)
            fee_multiplier: Fee multiplier (default 997 for 0.3% fee)

        Returns:
            Maximum input amount that satisfies the limit, or 0 if impossible
        """
        if reserve_in <= 0 or reserve_out <= 0:
            return 0
        if buy_amount <= 0:
            return sell_amount  # No limit constraint, can sell full amount
        if sell_amount <= 0:
            return 0

        # max_input = R_out * sell_amount / buy_amount - R_in * 1000 / fee
        # Use integer math to avoid precision loss
        # max_input = (R_out * sell_amount * fee - R_in * 1000 * buy_amount) / (buy_amount * fee)
        numerator = reserve_out * sell_amount * fee_multiplier - reserve_in * 1000 * buy_amount
        denominator = buy_amount * fee_multiplier

        if numerator <= 0:
            return 0  # Pool rate is worse than limit rate

        max_input = numerator // denominator
        max_input = min(max_input, sell_amount)

        # Verify and adjust for integer rounding in get_amount_out
        # Use binary search to find the exact maximum that satisfies the constraint
        if max_input > 0:
            actual_output = self.get_amount_out(max_input, reserve_in, reserve_out, fee_multiplier)
            # Check: actual_output * sell_amount >= buy_amount * max_input
            if actual_output * sell_amount < buy_amount * max_input:
                # Binary search for the largest valid input
                lo, hi = 0, max_input
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    mid_output = self.get_amount_out(mid, reserve_in, reserve_out, fee_multiplier)
                    if mid_output * sell_amount >= buy_amount * mid:
                        lo = mid
                    else:
                        hi = mid - 1
                max_input = lo

        return max_input

    def max_fill_buy_order(
        self,
        reserve_in: int,
        reserve_out: int,
        sell_amount: int,
        buy_amount: int,
        fee_multiplier: int = 997,
    ) -> int:
        """Calculate maximum output for a buy order that satisfies the limit price.

        For a buy order, the user wants to receive a specific amount and is
        willing to pay up to a maximum. This calculates the maximum output
        where the input rate still satisfies the limit: input/output <= sell_amount/buy_amount.

        The formula is derived from:
        - input(y) = (y * R_in * 1000) / (fee * (R_out - y))
        - Constraint: input(y) / y <= sell_amount / buy_amount
        - Solving: y <= R_out - R_in * 1000 * buy_amount / (fee * sell_amount)

        Args:
            reserve_in: Pool reserve of input token
            reserve_out: Pool reserve of output token
            sell_amount: Order's maximum sell amount (used for limit rate)
            buy_amount: Order's desired buy amount (used for limit rate)
            fee_multiplier: Fee multiplier (default 997 for 0.3% fee)

        Returns:
            Maximum output amount that satisfies the limit, or 0 if impossible
        """
        if reserve_in <= 0 or reserve_out <= 0:
            return 0
        if sell_amount <= 0:
            return 0  # No budget to spend
        if buy_amount <= 0:
            return buy_amount  # No output desired

        # max_output = R_out - R_in * 1000 * buy_amount / (fee * sell_amount)
        # Use integer math: max_output = (R_out * fee * sell_amount - R_in * 1000 * buy_amount) / (fee * sell_amount)
        numerator = reserve_out * fee_multiplier * sell_amount - reserve_in * 1000 * buy_amount
        denominator = fee_multiplier * sell_amount

        if numerator <= 0:
            return 0  # Pool rate is worse than limit rate

        max_output = numerator // denominator
        max_output = min(max_output, buy_amount)

        # Verify and adjust for integer rounding in get_amount_in
        # Use binary search to find the exact maximum that satisfies the constraint
        if max_output > 0:
            actual_input = self.get_amount_in(max_output, reserve_in, reserve_out, fee_multiplier)
            # Check: actual_input * buy_amount <= sell_amount * max_output
            if actual_input * buy_amount > sell_amount * max_output:
                # Binary search for the largest valid output
                lo, hi = 0, max_output
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    mid_input = self.get_amount_in(mid, reserve_in, reserve_out, fee_multiplier)
                    if mid_input * buy_amount <= sell_amount * mid:
                        lo = mid
                    else:
                        hi = mid - 1
                max_output = lo

        return max_output

    def simulate_swap(
        self,
        pool: UniswapV2Pool,
        token_in: str,
        amount_in: int,
    ) -> SwapResult:
        """Simulate a swap through a pool (exact input).

        Args:
            pool: The liquidity pool
            token_in: Input token address
            amount_in: Amount to swap

        Returns:
            SwapResult with amounts and pool info
        """
        reserve_in, reserve_out = pool.get_reserves(token_in)
        token_out = pool.get_token_out(token_in)
        amount_out = self.get_amount_out(amount_in, reserve_in, reserve_out, pool.fee_multiplier)

        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            pool_address=pool.address,
            token_in=token_in,
            token_out=token_out,
            gas_estimate=POOL_SWAP_GAS_COST,
        )

    def simulate_swap_exact_output(
        self,
        pool: UniswapV2Pool,
        token_in: str,
        amount_out: int,
    ) -> SwapResult:
        """Simulate a swap to get exact output amount.

        Args:
            pool: The liquidity pool
            token_in: Input token address
            amount_out: Desired output amount

        Returns:
            SwapResult with required input and desired output
        """
        reserve_in, reserve_out = pool.get_reserves(token_in)
        token_out = pool.get_token_out(token_in)
        amount_in = self.get_amount_in(amount_out, reserve_in, reserve_out, pool.fee_multiplier)

        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            pool_address=pool.address,
            token_in=token_in,
            token_out=token_out,
            gas_estimate=POOL_SWAP_GAS_COST,
        )

    def pool_max_fill_sell_order(
        self,
        pool: UniswapV2Pool,
        token_in: str,
        _token_out: str,
        sell_amount: int,
        buy_amount: int,
    ) -> int:
        """Calculate maximum input for a sell order that satisfies the limit price.

        Pool-based wrapper for max_fill_sell_order that matches SwapCalculator protocol.

        Args:
            pool: The liquidity pool
            token_in: Input token address
            token_out: Output token address (unused, derived from pool)
            sell_amount: Order's sell amount (search range upper bound)
            buy_amount: Order's minimum buy amount (for limit check)

        Returns:
            Maximum input amount that satisfies the limit, or 0 if impossible
        """
        reserve_in, reserve_out = pool.get_reserves(token_in)
        return self.max_fill_sell_order(
            reserve_in=reserve_in,
            reserve_out=reserve_out,
            sell_amount=sell_amount,
            buy_amount=buy_amount,
            fee_multiplier=pool.fee_multiplier,
        )

    def pool_max_fill_buy_order(
        self,
        pool: UniswapV2Pool,
        token_in: str,
        _token_out: str,
        sell_amount: int,
        buy_amount: int,
    ) -> int:
        """Calculate maximum output for a buy order that satisfies the limit price.

        Pool-based wrapper for max_fill_buy_order that matches SwapCalculator protocol.

        Args:
            pool: The liquidity pool
            token_in: Input token address
            token_out: Output token address (unused, derived from pool)
            sell_amount: Order's maximum sell amount (for limit check)
            buy_amount: Order's desired buy amount (search range upper bound)

        Returns:
            Maximum output amount that satisfies the limit, or 0 if impossible
        """
        reserve_in, reserve_out = pool.get_reserves(token_in)
        return self.max_fill_buy_order(
            reserve_in=reserve_in,
            reserve_out=reserve_out,
            sell_amount=sell_amount,
            buy_amount=buy_amount,
            fee_multiplier=pool.fee_multiplier,
        )

    def _prepare_swap_encoding(
        self,
        token_in: str,
        token_out: str,
        recipient: str,
        path: list[str] | None = None,
    ) -> tuple[list[bytes], bytes, int]:
        """Prepare common encoding components for swap calls.

        Args:
            token_in: Input token address
            token_out: Output token address
            recipient: Address to receive output tokens
            path: Optional full path. Defaults to [token_in, token_out].

        Returns:
            Tuple of (path_bytes, recipient_bytes, deadline)

        Raises:
            ValueError: If any address is invalid
        """
        if path is None:
            path = [token_in, token_out]

        for i, addr in enumerate(path):
            if not is_valid_address(addr):
                raise ValueError(f"Invalid address in path[{i}]: {addr}")

        if not is_valid_address(recipient):
            raise ValueError(f"Invalid recipient address: {recipient}")

        path_bytes = [bytes.fromhex(addr[2:]) for addr in path]
        recipient_bytes = bytes.fromhex(recipient[2:])
        deadline = 2**32 - 1  # Far future, replaced by driver

        return path_bytes, recipient_bytes, deadline

    def encode_swap(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        amount_out_min: int,
        recipient: str,
        path: list[str] | None = None,
    ) -> tuple[str, str]:
        """Encode a swap as calldata for UniswapV2 Router.

        Uses swapExactTokensForTokens(uint256,uint256,address[],address,uint256)

        Args:
            token_in: Input token address (0x-prefixed hex)
            token_out: Output token address (0x-prefixed hex)
            amount_in: Amount of input token
            amount_out_min: Minimum output (slippage protection)
            recipient: Address to receive output tokens
            path: Optional full path for multi-hop swaps. If not provided,
                  defaults to [token_in, token_out] for direct swaps.

        Returns:
            Tuple of (router_address, calldata)

        Raises:
            ValueError: If any address is invalid
        """
        path_bytes, recipient_bytes, deadline = self._prepare_swap_encoding(
            token_in, token_out, recipient, path
        )

        encoded_args = encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [amount_in, amount_out_min, path_bytes, recipient_bytes, deadline],
        )

        return self.ROUTER_ADDRESS, self.SWAP_EXACT_TOKENS_SELECTOR + encoded_args.hex()

    def encode_swap_exact_output(
        self,
        token_in: str,
        token_out: str,
        amount_out: int,
        amount_in_max: int,
        recipient: str,
        path: list[str] | None = None,
    ) -> tuple[str, str]:
        """Encode a swap for exact output as calldata for UniswapV2 Router.

        Uses swapTokensForExactTokens(uint256,uint256,address[],address,uint256)

        Args:
            token_in: Input token address (0x-prefixed hex)
            token_out: Output token address (0x-prefixed hex)
            amount_out: Exact amount of output token desired
            amount_in_max: Maximum input amount (slippage protection)
            recipient: Address to receive output tokens
            path: Optional full path for multi-hop swaps. If not provided,
                  defaults to [token_in, token_out] for direct swaps.

        Returns:
            Tuple of (router_address, calldata)

        Raises:
            ValueError: If any address is invalid
        """
        path_bytes, recipient_bytes, deadline = self._prepare_swap_encoding(
            token_in, token_out, recipient, path
        )

        encoded_args = encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [amount_out, amount_in_max, path_bytes, recipient_bytes, deadline],
        )

        return self.ROUTER_ADDRESS, self.SWAP_TOKENS_FOR_EXACT_SELECTOR + encoded_args.hex()


# Singleton instance
uniswap_v2 = UniswapV2()


def parse_liquidity_to_pool(liquidity: Liquidity) -> UniswapV2Pool | None:
    """Convert auction Liquidity to UniswapV2Pool.

    Args:
        liquidity: Liquidity source from the auction

    Returns:
        UniswapV2Pool if liquidity is a constant product pool, None otherwise
    """
    # Only handle constant product (UniswapV2-style) pools
    if liquidity.kind != "constantProduct":
        return None

    if liquidity.address is None:
        return None

    # tokens must be a dict with balance info
    if not isinstance(liquidity.tokens, dict):
        return None

    token_addresses = list(liquidity.tokens.keys())
    if len(token_addresses) != 2:
        return None

    token0 = normalize_address(token_addresses[0])
    token1 = normalize_address(token_addresses[1])
    balance0 = int(liquidity.tokens[token_addresses[0]]["balance"])
    balance1 = int(liquidity.tokens[token_addresses[1]]["balance"])

    # Determine token order (UniswapV2 sorts by address bytes)
    token0_bytes = bytes.fromhex(token0[2:])
    token1_bytes = bytes.fromhex(token1[2:])

    if token0_bytes > token1_bytes:
        # Swap to maintain canonical order
        token0, token1 = token1, token0
        balance0, balance1 = balance1, balance0

    # Parse fee (default 0.3% = 30 bps)
    fee_bps = 30
    if liquidity.fee:
        try:
            fee_decimal = float(liquidity.fee)
            fee_bps = int(fee_decimal * 10000)
        except ValueError:
            logger.warning(
                "fee_parse_failed",
                pool_id=liquidity.id,
                raw_fee=liquidity.fee,
                using_default="0.003 (30 bps)",
            )

    return UniswapV2Pool(
        address=normalize_address(liquidity.address),
        token0=token0,
        token1=token1,
        reserve0=balance0,
        reserve1=balance1,
        fee_bps=fee_bps,
        liquidity_id=liquidity.id,
    )


__all__ = [
    "UniswapV2Pool",
    "UniswapV2",
    "uniswap_v2",
    "parse_liquidity_to_pool",
]
