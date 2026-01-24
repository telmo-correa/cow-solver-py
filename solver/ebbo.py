"""EBBO (Ethereum Best Bid and Offer) validation utilities.

EBBO is a social consensus rule (CIP-11) that requires solvers to provide
execution prices at least as good as what users could get from base protocols
(Uniswap, Balancer, etc.).

This module provides:
- EBBOPrices: Container for precomputed EBBO prices
- EBBOValidator: Validates clearing prices against EBBO
- load_ebbo_prices: Load EBBO data from JSON file

References:
- https://docs.cow.fi/cow-protocol/reference/core/auctions/ebbo-rules
- https://docs.cow.fi/cow-protocol/reference/core/auctions/competition-rules
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from solver.models.types import normalize_address

if TYPE_CHECKING:
    from solver.models.auction import AuctionInstance, Order
    from solver.routing.router import SingleOrderRouter

logger = structlog.get_logger()

# Tolerance for EBBO comparison (0.1% = 0.001)
EBBO_TOLERANCE = Decimal("0.001")


@dataclass
class EBBOPrices:
    """Container for EBBO reference prices.

    Prices are stored as Decimal for precision in comparisons.
    """

    auction_id: str
    prices: dict[str, dict[str, Decimal]]

    def get_price(self, sell_token: str, buy_token: str) -> Decimal | None:
        """Get EBBO price for a token pair.

        Args:
            sell_token: Sell token address (will be normalized)
            buy_token: Buy token address (will be normalized)

        Returns:
            EBBO price (sell/buy ratio), or None if not available
        """
        sell = normalize_address(sell_token)
        buy = normalize_address(buy_token)
        return self.prices.get(sell, {}).get(buy)

    @classmethod
    def from_json(cls, data: dict[str, object]) -> EBBOPrices:
        """Load from JSON data."""
        prices: dict[str, dict[str, Decimal]] = {}
        ebbo_prices = data.get("ebbo_prices", {})
        if isinstance(ebbo_prices, dict):
            for sell_token, buy_prices in ebbo_prices.items():
                if isinstance(buy_prices, dict):
                    prices[str(sell_token)] = {}
                    for buy_token, price_str in buy_prices.items():
                        prices[str(sell_token)][str(buy_token)] = Decimal(str(price_str))
        auction_id = data.get("auction_id", "")
        return cls(auction_id=str(auction_id) if auction_id else "", prices=prices)

    @classmethod
    def from_file(cls, path: Path) -> EBBOPrices:
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_json(json.load(f))

    @classmethod
    def compute_from_auction(
        cls,
        auction: AuctionInstance,
        router: SingleOrderRouter,
    ) -> EBBOPrices:
        """Compute EBBO prices from auction liquidity.

        Args:
            auction: Auction with liquidity data
            router: Router for price queries

        Returns:
            EBBOPrices with computed prices
        """
        prices: dict[str, dict[str, Decimal]] = {}

        # Collect unique pairs from orders
        pairs: set[tuple[str, str]] = set()
        for order in auction.orders:
            sell = normalize_address(order.sell_token)
            buy = normalize_address(order.buy_token)
            pairs.add((sell, buy))

        # Query prices for each pair
        for sell_token, buy_token in pairs:
            token_info = auction.tokens.get(sell_token)
            decimals = token_info.decimals if token_info and token_info.decimals else 18

            price = router.get_reference_price(sell_token, buy_token, token_in_decimals=decimals)

            if price is not None:
                if sell_token not in prices:
                    prices[sell_token] = {}
                prices[sell_token][buy_token] = price

        return cls(auction_id=auction.id or "", prices=prices)


@dataclass
class EBBOViolation:
    """Record of an EBBO violation."""

    sell_token: str
    buy_token: str
    clearing_rate: Decimal
    ebbo_rate: Decimal
    deficit_pct: float  # How much worse than EBBO (percentage)

    def __str__(self) -> str:
        return (
            f"EBBO violation: {self.sell_token[-8:]}->{self.buy_token[-8:]} "
            f"clearing={float(self.clearing_rate):.6f} vs "
            f"EBBO={float(self.ebbo_rate):.6f} ({self.deficit_pct:.2f}% worse)"
        )


class EBBOValidator:
    """Validates clearing prices against EBBO reference prices."""

    def __init__(
        self,
        ebbo_prices: EBBOPrices | None = None,
        router: SingleOrderRouter | None = None,
        tolerance: Decimal = EBBO_TOLERANCE,
    ):
        """Initialize validator.

        Args:
            ebbo_prices: Precomputed EBBO prices (preferred)
            router: Router for computing prices on-the-fly (fallback)
            tolerance: Allowed deviation from EBBO (default 0.1%)
        """
        self.ebbo_prices = ebbo_prices
        self.router = router
        self.tolerance = tolerance

    def check_clearing_prices(
        self,
        clearing_prices: dict[str, int],
        orders: list[Order],
        auction: AuctionInstance | None = None,
    ) -> list[EBBOViolation]:
        """Check clearing prices against EBBO.

        Args:
            clearing_prices: Token -> clearing price (int)
            orders: Orders being settled
            auction: Auction context (for computing EBBO if needed)

        Returns:
            List of EBBO violations (empty if compliant)
        """
        violations: list[EBBOViolation] = []

        for order in orders:
            sell = normalize_address(order.sell_token)
            buy = normalize_address(order.buy_token)

            sell_price = clearing_prices.get(sell, 0)
            buy_price = clearing_prices.get(buy, 0)

            if sell_price == 0 or buy_price == 0:
                continue

            clearing_rate = Decimal(sell_price) / Decimal(buy_price)

            # Get EBBO rate
            ebbo_rate = self._get_ebbo_rate(sell, buy, auction)
            if ebbo_rate is None:
                continue

            # Check if clearing rate is at least as good as EBBO
            min_acceptable = ebbo_rate * (1 - self.tolerance)
            if clearing_rate < min_acceptable:
                deficit_pct = float((ebbo_rate - clearing_rate) / ebbo_rate * 100)
                violations.append(
                    EBBOViolation(
                        sell_token=sell,
                        buy_token=buy,
                        clearing_rate=clearing_rate,
                        ebbo_rate=ebbo_rate,
                        deficit_pct=deficit_pct,
                    )
                )

        return violations

    def _get_ebbo_rate(
        self,
        sell_token: str,
        buy_token: str,
        auction: AuctionInstance | None,
    ) -> Decimal | None:
        """Get EBBO rate from precomputed prices or router."""
        # Try precomputed prices first
        if self.ebbo_prices is not None:
            rate = self.ebbo_prices.get_price(sell_token, buy_token)
            if rate is not None:
                return rate

        # Fall back to router if available
        if self.router is not None and auction is not None:
            token_info = auction.tokens.get(sell_token)
            decimals = token_info.decimals if token_info and token_info.decimals else 18
            return self.router.get_reference_price(
                sell_token, buy_token, token_in_decimals=decimals
            )

        return None


def load_ebbo_prices(auction_file: Path) -> EBBOPrices | None:
    """Load EBBO prices for an auction file.

    Looks for a companion file named {auction_stem}_ebbo.json.

    Args:
        auction_file: Path to auction JSON file

    Returns:
        EBBOPrices if companion file exists, None otherwise
    """
    ebbo_file = auction_file.parent / f"{auction_file.stem}_ebbo.json"
    if ebbo_file.exists():
        return EBBOPrices.from_file(ebbo_file)
    return None
