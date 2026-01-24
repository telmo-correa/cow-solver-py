"""Tests for EBBO bounds helper."""

from decimal import Decimal
from unittest.mock import MagicMock

from solver.models.auction import AuctionInstance, Token
from solver.strategies.ebbo_bounds import EBBOBounds, get_ebbo_bounds

TOKEN_A = "0xaAaAaAaaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"
TOKEN_B = "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB"


def make_auction_with_tokens(
    token_a_decimals: int | None = 18,
    token_b_decimals: int | None = 18,
) -> AuctionInstance:
    """Create an auction with specified token decimals."""
    tokens = {}
    if token_a_decimals is not None:
        tokens[TOKEN_A.lower()] = Token(
            decimals=token_a_decimals,
            symbol="TKA",
            referencePrice="1000000000000000000",
            availableBalance="1000000000000000000000",
        )
    if token_b_decimals is not None:
        tokens[TOKEN_B.lower()] = Token(
            decimals=token_b_decimals,
            symbol="TKB",
            referencePrice="1000000000000000000",
            availableBalance="1000000000000000000000",
        )
    return AuctionInstance(id="test", orders=[], tokens=tokens)


class TestEBBOBoundsDataclass:
    """Tests for EBBOBounds dataclass."""

    def test_ebbo_bounds_with_values(self) -> None:
        """EBBOBounds stores all values correctly."""
        bounds = EBBOBounds(
            ebbo_min=Decimal("2.5"),
            ebbo_max=Decimal("3.5"),
            amm_price=Decimal("2.5"),
        )
        assert bounds.ebbo_min == Decimal("2.5")
        assert bounds.ebbo_max == Decimal("3.5")
        assert bounds.amm_price == Decimal("2.5")

    def test_ebbo_bounds_with_none(self) -> None:
        """EBBOBounds handles None values."""
        bounds = EBBOBounds(
            ebbo_min=None,
            ebbo_max=None,
            amm_price=None,
        )
        assert bounds.ebbo_min is None
        assert bounds.ebbo_max is None
        assert bounds.amm_price is None


class TestGetEBBOBounds:
    """Tests for get_ebbo_bounds function."""

    def test_returns_ebbo_min_from_a_to_b_price(self) -> None:
        """ebbo_min should be the A→B AMM rate."""
        router = MagicMock()
        router.get_reference_price.side_effect = [
            Decimal("2.5"),  # A→B
            Decimal("0.4"),  # B→A
        ]
        auction = make_auction_with_tokens()

        bounds = get_ebbo_bounds(TOKEN_A, TOKEN_B, router, auction)

        assert bounds.ebbo_min == Decimal("2.5")

    def test_returns_ebbo_max_from_inverse_b_to_a_price(self) -> None:
        """ebbo_max should be 1 / (B→A AMM rate)."""
        router = MagicMock()
        router.get_reference_price.side_effect = [
            Decimal("2.5"),  # A→B
            Decimal("0.4"),  # B→A (inverse = 2.5)
        ]
        auction = make_auction_with_tokens()

        bounds = get_ebbo_bounds(TOKEN_A, TOKEN_B, router, auction)

        assert bounds.ebbo_max == Decimal(1) / Decimal("0.4")

    def test_amm_price_equals_ebbo_min(self) -> None:
        """amm_price should equal ebbo_min (standard approach)."""
        router = MagicMock()
        router.get_reference_price.side_effect = [
            Decimal("2.5"),  # A→B
            Decimal("0.4"),  # B→A
        ]
        auction = make_auction_with_tokens()

        bounds = get_ebbo_bounds(TOKEN_A, TOKEN_B, router, auction)

        assert bounds.amm_price == bounds.ebbo_min

    def test_handles_none_a_to_b_price(self) -> None:
        """Should handle None A→B price gracefully."""
        router = MagicMock()
        router.get_reference_price.side_effect = [
            None,  # A→B
            Decimal("0.4"),  # B→A
        ]
        auction = make_auction_with_tokens()

        bounds = get_ebbo_bounds(TOKEN_A, TOKEN_B, router, auction)

        assert bounds.ebbo_min is None
        assert bounds.amm_price is None
        # ebbo_max still computed
        assert bounds.ebbo_max == Decimal(1) / Decimal("0.4")

    def test_handles_none_b_to_a_price(self) -> None:
        """Should handle None B→A price gracefully."""
        router = MagicMock()
        router.get_reference_price.side_effect = [
            Decimal("2.5"),  # A→B
            None,  # B→A
        ]
        auction = make_auction_with_tokens()

        bounds = get_ebbo_bounds(TOKEN_A, TOKEN_B, router, auction)

        assert bounds.ebbo_min == Decimal("2.5")
        assert bounds.ebbo_max is None

    def test_handles_zero_b_to_a_price(self) -> None:
        """Should handle zero B→A price (avoid division by zero)."""
        router = MagicMock()
        router.get_reference_price.side_effect = [
            Decimal("2.5"),  # A→B
            Decimal("0"),  # B→A (zero)
        ]
        auction = make_auction_with_tokens()

        bounds = get_ebbo_bounds(TOKEN_A, TOKEN_B, router, auction)

        assert bounds.ebbo_min == Decimal("2.5")
        assert bounds.ebbo_max is None  # Avoid division by zero

    def test_uses_token_decimals_from_auction(self) -> None:
        """Should pass correct decimals to router."""
        router = MagicMock()
        router.get_reference_price.return_value = Decimal("2.5")
        auction = make_auction_with_tokens(token_a_decimals=6, token_b_decimals=8)

        get_ebbo_bounds(TOKEN_A, TOKEN_B, router, auction)

        # First call: A→B with token_a decimals
        first_call = router.get_reference_price.call_args_list[0]
        assert first_call[1]["token_in_decimals"] == 6

        # Second call: B→A with token_b decimals
        second_call = router.get_reference_price.call_args_list[1]
        assert second_call[1]["token_in_decimals"] == 8

    def test_defaults_to_18_decimals_when_missing(self) -> None:
        """Should default to 18 decimals for unknown tokens."""
        router = MagicMock()
        router.get_reference_price.return_value = Decimal("2.5")
        # Auction with no token info
        auction = AuctionInstance(id="test", orders=[], tokens={})

        get_ebbo_bounds(TOKEN_A, TOKEN_B, router, auction)

        # Both calls should use default 18 decimals
        first_call = router.get_reference_price.call_args_list[0]
        assert first_call[1]["token_in_decimals"] == 18

        second_call = router.get_reference_price.call_args_list[1]
        assert second_call[1]["token_in_decimals"] == 18
