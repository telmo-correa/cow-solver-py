"""Native token price estimation for fee calculation.

When reference prices are missing, estimate token prices by simulating
swaps through available pools to the native token (WETH/wxDAI).

This matches the Rust baseline solver's behavior:
1. If reference_price exists, use it
2. If token is native, price = 1e18
3. Otherwise, route through pools to estimate price
4. If no route found, default to U256_MAX (fee=0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import structlog

from solver.constants import WETH
from solver.models.types import normalize_address

if TYPE_CHECKING:
    from solver.models.auction import AuctionInstance, Token
    from solver.pools import AnyPool
    from solver.routing.router import SingleOrderRouter

logger = structlog.get_logger()

# Native tokens by chain
# Mainnet: WETH
# Gnosis: wxDAI
WETH_MAINNET = WETH
WXDAI_GNOSIS = "0xe91d153e0b41518a2ce8dd3d7944fa863463a97d"

# Try these native tokens in order
DEFAULT_NATIVE_TOKENS = [WETH_MAINNET, WXDAI_GNOSIS]

# Default native token (WETH on mainnet, for backwards compatibility)
DEFAULT_NATIVE_TOKEN = WETH

# Default amount to use for price estimation (1 token with 18 decimals)
DEFAULT_ESTIMATION_AMOUNT = 10**18

# U256 max - signals "infinite price" which results in fee=0
U256_MAX = 2**256 - 1


def get_token_info(auction: AuctionInstance, token_address: str) -> Token | None:
    """Get token info with case-insensitive address lookup.

    Token addresses may have different case in orders vs tokens dict.
    This function normalizes the lookup to handle both cases.

    Args:
        auction: The auction instance
        token_address: Token address to look up

    Returns:
        Token info if found, None otherwise
    """
    token_norm = normalize_address(token_address)

    # Try direct lookup first (most common case)
    token_info = auction.tokens.get(token_address)
    if token_info is not None:
        return token_info

    # Try case-insensitive lookup
    for addr, info in auction.tokens.items():
        if normalize_address(addr) == token_norm:
            return info

    return None


@dataclass
class PriceEstimate:
    """Result of a price estimation.

    Attributes:
        token: The token being priced
        price: Price in wei to buy 1e18 of the token (reference price format)
        source: Where the price came from ('reference', 'native', 'pool', 'default')
    """

    token: str
    price: int
    source: str


class PriceEstimator(Protocol):
    """Protocol for token price estimation.

    Implementations can use different sources to estimate prices:
    - Reference prices from auction
    - Direct pool routing
    - Oracle data
    """

    def estimate_price(
        self,
        token: str,
        auction: AuctionInstance,
    ) -> PriceEstimate:
        """Estimate the price of a token in native token units.

        Args:
            token: Token address to price
            auction: Auction with reference prices and liquidity

        Returns:
            PriceEstimate with the price and source
        """
        ...


class PoolBasedPriceEstimator:
    """Estimate token prices using pool routing when reference prices are missing.

    Matches Rust baseline solver behavior:
    1. Check auction reference price
    2. If token is native, return 1e18
    3. Try to route through pools to native token
    4. If no route, return U256_MAX (means fee=0)
    """

    def __init__(
        self,
        router: SingleOrderRouter | None = None,
        native_tokens: list[str] | None = None,
    ) -> None:
        """Initialize the price estimator.

        Args:
            router: Router for simulating swaps (required for pool routing)
            native_tokens: List of native token addresses to try (in order).
                          Defaults to [WETH, wxDAI] for multi-chain support.
        """
        self._router = router
        self._native_tokens = [
            normalize_address(t) for t in (native_tokens or DEFAULT_NATIVE_TOKENS)
        ]

    def estimate_price(
        self,
        token: str,
        auction: AuctionInstance,
    ) -> PriceEstimate:
        """Estimate the price of a token.

        Priority:
        1. Use reference price from auction if available
        2. If token is native, return 1e18
        3. Try pool-based estimation (against multiple native tokens)
        4. Default to U256_MAX (fee=0)
        """
        token_normalized = normalize_address(token)

        # 1. Check reference price from auction
        token_info = get_token_info(auction, token_normalized)
        if token_info is not None and token_info.reference_price is not None:
            ref_price = int(token_info.reference_price)
            if ref_price > 0:
                return PriceEstimate(
                    token=token_normalized,
                    price=ref_price,
                    source="reference",
                )

        # 2. If token is native, price is 1e18
        if token_normalized in self._native_tokens:
            return PriceEstimate(
                token=token_normalized,
                price=10**18,
                source="native",
            )

        # 3. Try pool-based estimation (try each native token in order)
        if self._router is not None:
            for native_token in self._native_tokens:
                estimated = self._estimate_via_pools(token_normalized, auction, native_token)
                if estimated is not None:
                    return PriceEstimate(
                        token=token_normalized,
                        price=estimated,
                        source="pool",
                    )

        # 4. Default: U256_MAX (infinite price = fee of 0)
        logger.debug(
            "price_estimation_defaulting",
            token=token_normalized[-8:],
            reason="no reference price and no pool route to native",
        )
        return PriceEstimate(
            token=token_normalized,
            price=U256_MAX,
            source="default",
        )

    def _estimate_via_pools(
        self,
        token: str,
        auction: AuctionInstance,
        native_token: str,
    ) -> int | None:
        """Estimate price by routing through pools to native token.

        Matches Rust baseline behavior exactly:
        1. Create a BUY order for `native_token_price_estimation_amount` (1e18) of native
        2. Route through pools to find required input amount
        3. Price = native_token_price_estimation_amount / input_amount * 1e18

        See cow-services/crates/solvers/src/domain/solver.rs:312-338 for Rust impl.

        Args:
            token: Token to price
            auction: Auction with liquidity
            native_token: Native token address to route to

        Returns:
            Estimated reference price, or None if routing fails
        """
        if self._router is None:
            return None

        # Build pool registry from auction liquidity
        from solver.pools import build_registry_from_liquidity

        registry = build_registry_from_liquidity(auction.liquidity)

        if registry.pool_count == 0:
            return None

        # Find a path from token to native
        path = registry.find_path(token, native_token, max_hops=2)
        if path is None:
            logger.debug(
                "price_estimation_no_path",
                token=token[-8:],
                native=native_token[-8:],
            )
            return None

        # Get pools for this path
        try:
            pools = registry.get_all_pools_on_path(path)
        except ValueError:
            return None

        # Rust approach: simulate a BUY order for 1e18 native token
        # to find the required input amount of the sell token.
        # See cow-services/crates/solvers/src/domain/solver.rs:312-338
        native_amount = 10**18  # Amount of native token to buy

        # Simulate exact output swap: how much token do we need to get 1e18 native?
        input_amount = self._simulate_path_exact_output(pools, path, native_amount)
        if input_amount is None or input_amount == 0:
            # Fallback to forward simulation if exact output fails.
            # This uses a different formula and may produce different prices.
            logger.warning(
                "price_estimation_exact_output_failed_using_forward_fallback",
                token=token[-8:],
                native=native_token[-8:],
            )
            output = self._simulate_path_output(pools, path, 10**18)
            if output is None or output == 0:
                return None
            price = output
        else:
            # Price = native_amount / input_amount * 1e18
            # This matches Rust: f64(native_amount) / f64(input_amount) then * 1e18
            price = native_amount * 10**18 // input_amount

        logger.debug(
            "price_estimation_via_pools",
            token=token[-8:],
            native=native_token[-8:],
            native_amount=native_amount,
            input_amount=input_amount if input_amount else "forward_fallback",
            estimated_price=price,
        )

        return price

    def _simulate_path_output(
        self,
        pools: list[AnyPool],
        path: list[str],
        amount_in: int,
    ) -> int | None:
        """Simulate swap through path and return output amount.

        Args:
            pools: List of pools in the path
            path: Token addresses in path order
            amount_in: Input amount

        Returns:
            Output amount, or None if simulation fails
        """
        if self._router is None:
            return None

        handler_registry = self._router._handler_registry

        # Simulate forward through path
        current_amount = amount_in
        for i, pool in enumerate(pools):
            result = handler_registry.simulate_swap(pool, path[i], path[i + 1], current_amount)
            if result is None or result.amount_out == 0:
                return None
            current_amount = result.amount_out

        return current_amount

    def _simulate_path_exact_output(
        self,
        pools: list[AnyPool],
        path: list[str],
        amount_out: int,
    ) -> int | None:
        """Simulate exact output swap through path and return required input amount.

        Works backwards through the path to calculate the required input for
        a desired output amount. This matches Rust's buy order routing.

        Args:
            pools: List of pools in the path
            path: Token addresses in path order
            amount_out: Desired output amount

        Returns:
            Required input amount, or None if simulation fails
        """
        if self._router is None:
            return None

        handler_registry = self._router._handler_registry

        # Work backwards through path to find required input
        # Start with desired output and calculate required input at each step
        current_amount = amount_out
        for i in range(len(pools) - 1, -1, -1):
            pool = pools[i]
            token_in = path[i]
            token_out = path[i + 1]

            # Use exact output simulation via handler registry
            result = handler_registry.simulate_swap_exact_output(
                pool, token_in, token_out, current_amount
            )
            if result is None:
                return None
            if result.amount_in == 0:
                return None
            current_amount = result.amount_in

        return current_amount


__all__ = [
    "PriceEstimate",
    "PriceEstimator",
    "PoolBasedPriceEstimator",
    "get_token_info",
    "DEFAULT_NATIVE_TOKEN",
    "DEFAULT_NATIVE_TOKENS",
    "DEFAULT_ESTIMATION_AMOUNT",
    "U256_MAX",
    "WETH_MAINNET",
    "WXDAI_GNOSIS",
]
