"""UniswapV2 AMM implementation.

UniswapV2 uses the constant product formula: x * y = k
With a 0.3% fee on input amounts.
"""

from dataclasses import dataclass
from typing import ClassVar

from eth_abi import encode  # type: ignore[attr-defined]

from solver.amm.base import AMM, SwapResult
from solver.models.types import is_valid_address, normalize_address


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

    # Gas estimates
    SWAP_GAS: ClassVar[int] = 150_000

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

    def simulate_swap(
        self,
        pool: UniswapV2Pool,
        token_in: str,
        amount_in: int,
    ) -> SwapResult:
        """Simulate a swap through a pool.

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
            gas_estimate=self.SWAP_GAS,
        )

    def encode_swap(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        amount_out_min: int,
        recipient: str,
    ) -> tuple[str, str]:
        """Encode a swap as calldata for UniswapV2 Router.

        Uses swapExactTokensForTokens(uint256,uint256,address[],address,uint256)

        Args:
            token_in: Input token address (0x-prefixed hex)
            token_out: Output token address (0x-prefixed hex)
            amount_in: Amount of input token
            amount_out_min: Minimum output (slippage protection)
            recipient: Address to receive output tokens

        Returns:
            Tuple of (router_address, calldata)

        Raises:
            ValueError: If any address is invalid
        """
        # Validate addresses (is_valid_address ensures they're valid hex)
        for name, addr in [
            ("token_in", token_in),
            ("token_out", token_out),
            ("recipient", recipient),
        ]:
            if not is_valid_address(addr):
                raise ValueError(f"Invalid {name} address: {addr}")

        # Path is [token_in, token_out]
        # Note: bytes.fromhex() is safe here since is_valid_address already verified hex format
        path = [
            bytes.fromhex(token_in[2:]),
            bytes.fromhex(token_out[2:]),
        ]
        recipient_bytes = bytes.fromhex(recipient[2:])

        # Deadline far in the future (will be replaced by driver)
        deadline = 2**32 - 1

        # Encode the function call
        encoded_args = encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [amount_in, amount_out_min, path, recipient_bytes, deadline],
        )

        calldata = self.SWAP_EXACT_TOKENS_SELECTOR + encoded_args.hex()

        return self.ROUTER_ADDRESS, calldata

    def encode_swap_direct(
        self,
        pool_address: str,
        token_in: str,
        token_out: str,
        _amount_in: int,  # Not used in direct swap encoding
        amount_out: int,
        recipient: str,
    ) -> tuple[str, str]:
        """Encode a direct swap on the pair contract.

        Uses swap(uint256,uint256,address,bytes) on the pair directly.
        This is more gas efficient than going through the router.

        Note: This method is currently not used but kept for future optimization.
        Direct pool swaps save ~20k gas compared to router swaps.

        Args:
            pool_address: UniswapV2 pair contract address
            token_in: Input token address (0x-prefixed hex)
            token_out: Output token address (0x-prefixed hex)
            _amount_in: Input amount (unused - pool infers from balance change)
            amount_out: Expected output amount
            recipient: Address to receive output tokens

        Returns:
            Tuple of (pool_address, calldata)

        Raises:
            ValueError: If any address is invalid
        """
        # Validate addresses (is_valid_address ensures they're valid hex)
        for name, addr in [
            ("pool_address", pool_address),
            ("token_in", token_in),
            ("token_out", token_out),
            ("recipient", recipient),
        ]:
            if not is_valid_address(addr):
                raise ValueError(f"Invalid {name} address: {addr}")

        # Determine which amount goes where based on token order
        # In UniswapV2, token0 < token1 (sorted by address bytes, not string)
        # Note: bytes.fromhex() is safe here since is_valid_address already verified hex format
        token_in_bytes = bytes.fromhex(token_in[2:].lower())
        token_out_bytes = bytes.fromhex(token_out[2:].lower())
        recipient_bytes = bytes.fromhex(recipient[2:])

        if token_in_bytes < token_out_bytes:
            # token_in is token0, so we're swapping token0 for token1
            amount0_out = 0
            amount1_out = amount_out
        else:
            # token_in is token1, so we're swapping token1 for token0
            amount0_out = amount_out
            amount1_out = 0

        # swap(uint amount0Out, uint amount1Out, address to, bytes calldata data)
        selector = "0x022c0d9f"
        encoded_args = encode(
            ["uint256", "uint256", "address", "bytes"],
            [amount0_out, amount1_out, recipient_bytes, b""],
        )

        calldata = selector + encoded_args.hex()

        return pool_address, calldata


# Singleton instance
uniswap_v2 = UniswapV2()


def _validate_token_address(name: str, address: str) -> str:
    """Validate and return a token address.

    Args:
        name: Name of the token (for error messages)
        address: The address to validate

    Returns:
        The validated address

    Raises:
        ValueError: If the address is invalid
    """
    if not is_valid_address(address):
        raise ValueError(f"Invalid {name} address: {address} (must be 0x + 40 hex chars)")
    return address


# Well-known token addresses (lowercase for consistency)
# All addresses are validated at import time to catch typos early
WETH = _validate_token_address("WETH", "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2")
USDC = _validate_token_address("USDC", "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")
USDT = _validate_token_address("USDT", "0xdac17f958d2ee523a2206206994597c13d831ec7")
DAI = _validate_token_address("DAI", "0x6b175474e89094c44da98b954eecdecb5f6f8fa0")

# Well-known pools on mainnet (reserve values are examples - would be fetched on-chain)
# All addresses are stored in lowercase for consistent comparison
# Using frozenset keys for O(1) lookup regardless of token order
MAINNET_POOLS: dict[frozenset[str], UniswapV2Pool] = {
    # WETH/USDC
    frozenset([WETH, USDC]): UniswapV2Pool(
        address="0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
        token0=USDC,  # USDC is token0 (lower address bytes)
        token1=WETH,  # WETH is token1
        reserve0=50_000_000 * 10**6,  # 50M USDC
        reserve1=20_000 * 10**18,  # 20K WETH
    ),
    # WETH/USDT
    frozenset([WETH, USDT]): UniswapV2Pool(
        address="0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852",
        token0=WETH,  # WETH is token0 (lower address bytes)
        token1=USDT,  # USDT is token1
        reserve0=15_000 * 10**18,  # 15K WETH
        reserve1=37_500_000 * 10**6,  # 37.5M USDT
    ),
    # WETH/DAI
    frozenset([WETH, DAI]): UniswapV2Pool(
        address="0xa478c2975ab1ea89e8196811f51a7b7ade33eb11",
        token0=DAI,  # DAI is token0 (lower address bytes)
        token1=WETH,  # WETH is token1
        reserve0=30_000_000 * 10**18,  # 30M DAI
        reserve1=12_000 * 10**18,  # 12K WETH
    ),
}


def get_pool(token_a: str, token_b: str) -> UniswapV2Pool | None:
    """Get a pool for a token pair (order independent).

    Args:
        token_a: First token address (any case)
        token_b: Second token address (any case)

    Returns:
        UniswapV2Pool if found, None otherwise

    Note:
        Uses frozenset keys for O(1) lookup regardless of token order.
    """
    # Normalize to lowercase for comparison
    token_a_norm = normalize_address(token_a)
    token_b_norm = normalize_address(token_b)
    pair_key = frozenset([token_a_norm, token_b_norm])

    return MAINNET_POOLS.get(pair_key)
