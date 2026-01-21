"""UniswapV2 AMM implementation.

UniswapV2 uses the constant product formula: x * y = k
With a 0.3% fee on input amounts.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from eth_abi import encode  # type: ignore[attr-defined]

from solver.amm.base import AMM, SwapResult
from solver.models.types import is_valid_address, normalize_address

if TYPE_CHECKING:
    from solver.models.auction import Liquidity


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
            gas_estimate=self.SWAP_GAS,
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
            gas_estimate=self.SWAP_GAS,
        )

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
        # Build path if not provided
        if path is None:
            path = [token_in, token_out]

        # Validate all addresses in path
        for i, addr in enumerate(path):
            if not is_valid_address(addr):
                raise ValueError(f"Invalid address in path[{i}]: {addr}")

        if not is_valid_address(recipient):
            raise ValueError(f"Invalid recipient address: {recipient}")

        # Convert path to bytes
        path_bytes = [bytes.fromhex(addr[2:]) for addr in path]
        recipient_bytes = bytes.fromhex(recipient[2:])

        # Deadline far in the future (will be replaced by driver)
        deadline = 2**32 - 1

        # Encode the function call
        encoded_args = encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [amount_in, amount_out_min, path_bytes, recipient_bytes, deadline],
        )

        calldata = self.SWAP_EXACT_TOKENS_SELECTOR + encoded_args.hex()

        return self.ROUTER_ADDRESS, calldata

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
        # Build path if not provided
        if path is None:
            path = [token_in, token_out]

        # Validate all addresses in path
        for i, addr in enumerate(path):
            if not is_valid_address(addr):
                raise ValueError(f"Invalid address in path[{i}]: {addr}")

        if not is_valid_address(recipient):
            raise ValueError(f"Invalid recipient address: {recipient}")

        # Convert path to bytes
        path_bytes = [bytes.fromhex(addr[2:]) for addr in path]
        recipient_bytes = bytes.fromhex(recipient[2:])

        # Deadline far in the future (will be replaced by driver)
        deadline = 2**32 - 1

        # Encode the function call
        # swapTokensForExactTokens(amountOut, amountInMax, path, to, deadline)
        encoded_args = encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [amount_out, amount_in_max, path_bytes, recipient_bytes, deadline],
        )

        calldata = self.SWAP_TOKENS_FOR_EXACT_SELECTOR + encoded_args.hex()

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


class PoolRegistry:
    """Registry of liquidity pools for routing.

    This class manages a collection of pools and provides methods for:
    - Looking up pools by token pair
    - Building token graphs for pathfinding
    - Finding multi-hop paths through available liquidity
    """

    def __init__(self, pools: list[UniswapV2Pool] | None = None) -> None:
        """Initialize the registry with optional pools.

        Args:
            pools: Initial list of pools. If None, starts empty.
        """
        self._pools: dict[frozenset[str], UniswapV2Pool] = {}
        self._graph: dict[str, set[str]] | None = None  # Cached graph

        if pools:
            for pool in pools:
                self.add_pool(pool)

    def add_pool(self, pool: UniswapV2Pool) -> None:
        """Add a pool to the registry.

        Args:
            pool: The pool to add. If a pool for this token pair already exists,
                  it will be replaced.
        """
        token0_norm = normalize_address(pool.token0)
        token1_norm = normalize_address(pool.token1)
        pair_key = frozenset([token0_norm, token1_norm])
        self._pools[pair_key] = pool
        self._graph = None  # Invalidate cached graph

    def get_pool(self, token_a: str, token_b: str) -> UniswapV2Pool | None:
        """Get a pool for a token pair (order independent).

        Args:
            token_a: First token address (any case)
            token_b: Second token address (any case)

        Returns:
            UniswapV2Pool if found, None otherwise
        """
        token_a_norm = normalize_address(token_a)
        token_b_norm = normalize_address(token_b)
        pair_key = frozenset([token_a_norm, token_b_norm])
        return self._pools.get(pair_key)

    def _build_graph(self) -> dict[str, set[str]]:
        """Build adjacency list of tokens connected by pools."""
        graph: dict[str, set[str]] = {}

        for token_pair in self._pools:
            tokens = list(token_pair)
            token_a, token_b = tokens[0], tokens[1]

            if token_a not in graph:
                graph[token_a] = set()
            if token_b not in graph:
                graph[token_b] = set()

            graph[token_a].add(token_b)
            graph[token_b].add(token_a)

        return graph

    def find_path(
        self,
        token_in: str,
        token_out: str,
        max_hops: int = 2,
    ) -> list[str] | None:
        """Find a path from token_in to token_out through available pools.

        Uses BFS to find the shortest path (by number of hops).

        Args:
            token_in: Starting token address
            token_out: Target token address
            max_hops: Maximum number of swaps allowed (default 2)

        Returns:
            List of token addresses representing the path, or None if no path found.
        """
        from collections import deque

        token_in_norm = normalize_address(token_in)
        token_out_norm = normalize_address(token_out)

        if token_in_norm == token_out_norm:
            return [token_in_norm]

        # Use cached graph or rebuild
        if self._graph is None:
            self._graph = self._build_graph()
        graph = self._graph

        if token_in_norm not in graph or token_out_norm not in graph:
            return None

        queue: deque[list[str]] = deque([[token_in_norm]])
        visited = {token_in_norm}

        while queue:
            path = queue.popleft()
            if len(path) > max_hops + 1:
                continue

            current = path[-1]
            for neighbor in graph.get(current, set()):
                if neighbor == token_out_norm:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None

    def get_all_pools_on_path(self, path: list[str]) -> list[UniswapV2Pool]:
        """Get all pools needed to execute a multi-hop swap path.

        Args:
            path: List of token addresses forming the swap path

        Returns:
            List of pools for each hop in the path

        Raises:
            ValueError: If any pool in the path is not found
        """
        pools = []
        for i in range(len(path) - 1):
            pool = self.get_pool(path[i], path[i + 1])
            if pool is None:
                raise ValueError(f"No pool found for {path[i]} -> {path[i + 1]}")
            pools.append(pool)
        return pools

    @property
    def pool_count(self) -> int:
        """Return the number of pools in the registry."""
        return len(self._pools)


def parse_liquidity_to_pool(liquidity: "Liquidity") -> UniswapV2Pool | None:
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
            pass

    return UniswapV2Pool(
        address=normalize_address(liquidity.address),
        token0=token0,
        token1=token1,
        reserve0=balance0,
        reserve1=balance1,
        fee_bps=fee_bps,
        liquidity_id=liquidity.id,
    )


def build_registry_from_liquidity(liquidity_list: list["Liquidity"]) -> PoolRegistry:
    """Build a PoolRegistry from auction liquidity sources.

    Args:
        liquidity_list: List of Liquidity objects from the auction

    Returns:
        PoolRegistry populated with parsed pools
    """
    registry = PoolRegistry()
    for liq in liquidity_list:
        pool = parse_liquidity_to_pool(liq)
        if pool is not None:
            registry.add_pool(pool)
    return registry
