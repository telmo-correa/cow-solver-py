"""Tests for EBBO validator (H9: AttributeError handling).

Verifies that EBBOValidator._get_ebbo_rate_ratio handles routers
that don't have the get_reference_price_ratio method.
"""

from unittest.mock import MagicMock, Mock

from solver.ebbo import EBBO_TOLERANCE, EBBOValidator
from solver.models.auction import AuctionInstance, Token


def make_auction() -> AuctionInstance:
    """Create minimal auction for EBBO tests."""
    return AuctionInstance(
        id="test",
        orders=[],
        tokens={
            "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa": Token(decimals=18, available_balance="0"),
            "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB": Token(decimals=18, available_balance="0"),
        },
    )


class TestEBBOValidatorAttributeError:
    """H9: Router without get_reference_price_ratio should not crash."""

    def test_router_without_method_returns_none(self) -> None:
        """When router doesn't have get_reference_price_ratio, return None gracefully."""
        # Create a simple object without the method
        mock_router = Mock(spec=[])  # No methods at all

        validator = EBBOValidator(tolerance=EBBO_TOLERANCE)
        validator.router = mock_router

        auction = make_auction()
        result = validator._get_ebbo_rate_ratio(
            "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa",
            "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB",
            auction,
        )

        assert result is None

    def test_router_with_method_works_normally(self) -> None:
        """When router has get_reference_price_ratio, use it normally."""
        mock_router = MagicMock()
        mock_router.get_reference_price_ratio.return_value = (3, 7)

        validator = EBBOValidator(tolerance=EBBO_TOLERANCE)
        validator.router = mock_router

        auction = make_auction()
        result = validator._get_ebbo_rate_ratio(
            "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa",
            "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB",
            auction,
        )

        assert result == (3, 7)

    def test_no_router_returns_none(self) -> None:
        """When no router is configured, return None."""
        validator = EBBOValidator(tolerance=EBBO_TOLERANCE)
        validator.router = None

        auction = make_auction()
        result = validator._get_ebbo_rate_ratio(
            "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa",
            "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB",
            auction,
        )

        assert result is None
