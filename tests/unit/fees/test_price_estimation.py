"""Tests for price estimation module."""

from solver.fees.price_estimation import (
    DEFAULT_NATIVE_TOKENS,
    U256_MAX,
    WETH_MAINNET,
    WXDAI_GNOSIS,
    PoolBasedPriceEstimator,
    PriceEstimate,
    get_token_info,
)
from solver.models.auction import AuctionInstance, Order, Token

# --- Test Fixtures ---


def make_auction(
    *,
    tokens: dict[str, Token] | None = None,
    orders: list[Order] | None = None,
    liquidity: list | None = None,
) -> AuctionInstance:
    """Create a test auction with customizable tokens."""
    default_tokens = {
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": Token(
            decimals=6,
            symbol="USDC",
            referencePrice="450000000000000000000000000",
            availableBalance="10000000000",
            trusted=True,
        ),
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": Token(
            decimals=18,
            symbol="WETH",
            referencePrice="1000000000000000000",
            availableBalance="10000000000000000000",
            trusted=True,
        ),
    }
    return AuctionInstance(
        id="test-auction",
        tokens=tokens or default_tokens,
        orders=orders or [],
        liquidity=liquidity or [],
        effectiveGasPrice="15000000000",
    )


# --- get_token_info Tests ---


class TestGetTokenInfo:
    """Tests for get_token_info function."""

    def test_direct_lookup(self):
        """Test direct token address lookup."""
        auction = make_auction()
        token_info = get_token_info(auction, "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")

        assert token_info is not None
        assert token_info.symbol == "USDC"
        assert token_info.decimals == 6

    def test_case_insensitive_lookup(self):
        """Test case-insensitive address lookup."""
        auction = make_auction()

        # Uppercase lookup
        token_info = get_token_info(auction, "0xA0B86991C6218B36C1D19D4A2E9EB0CE3606EB48")
        assert token_info is not None
        assert token_info.symbol == "USDC"

        # Mixed case lookup
        token_info = get_token_info(auction, "0xA0b86991c6218b36c1d19d4a2e9Eb0cE3606eB48")
        assert token_info is not None
        assert token_info.symbol == "USDC"

    def test_not_found_returns_none(self):
        """Test that non-existent token returns None."""
        auction = make_auction()
        token_info = get_token_info(auction, "0x0000000000000000000000000000000000000000")

        assert token_info is None

    def test_empty_tokens(self):
        """Test lookup with empty tokens dict."""
        auction = AuctionInstance(
            id="test-auction",
            tokens={},
            orders=[],
            liquidity=[],
            effectiveGasPrice="15000000000",
        )
        token_info = get_token_info(auction, "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")

        assert token_info is None


# --- PriceEstimate Tests ---


class TestPriceEstimate:
    """Tests for PriceEstimate dataclass."""

    def test_create_estimate(self):
        """Test creating a price estimate."""
        estimate = PriceEstimate(
            token="0xtoken",
            price=1000000000000000000,
            source="reference",
        )

        assert estimate.token == "0xtoken"
        assert estimate.price == 10**18
        assert estimate.source == "reference"

    def test_estimate_sources(self):
        """Test different estimate sources."""
        for source in ["reference", "native", "pool", "default"]:
            estimate = PriceEstimate(token="0x", price=1, source=source)
            assert estimate.source == source


# --- PoolBasedPriceEstimator Tests ---


class TestPoolBasedPriceEstimator:
    """Tests for PoolBasedPriceEstimator class."""

    def test_uses_reference_price_when_available(self):
        """Test that reference price is used when available in auction."""
        auction = make_auction()
        estimator = PoolBasedPriceEstimator()

        estimate = estimator.estimate_price(
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
            auction,
        )

        assert estimate.source == "reference"
        assert estimate.price == 450000000000000000000000000

    def test_native_token_returns_1e18(self):
        """Test that native token (WETH) returns 1e18."""
        # Auction without reference price for WETH
        auction = make_auction(
            tokens={
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": Token(
                    decimals=18,
                    symbol="WETH",
                    referencePrice=None,  # No reference price
                    availableBalance="10000000000000000000",
                    trusted=True,
                ),
            }
        )
        estimator = PoolBasedPriceEstimator()

        estimate = estimator.estimate_price(
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
            auction,
        )

        assert estimate.source == "native"
        assert estimate.price == 10**18

    def test_wxdai_is_native_token(self):
        """Test that wxDAI is recognized as native token on Gnosis."""
        auction = make_auction(
            tokens={
                WXDAI_GNOSIS: Token(
                    decimals=18,
                    symbol="wxDAI",
                    referencePrice=None,
                    availableBalance="10000000000000000000",
                    trusted=True,
                ),
            }
        )
        estimator = PoolBasedPriceEstimator()

        estimate = estimator.estimate_price(WXDAI_GNOSIS, auction)

        assert estimate.source == "native"
        assert estimate.price == 10**18

    def test_no_reference_no_router_returns_default(self):
        """Test that missing price without router returns U256_MAX."""
        # Token without reference price
        auction = make_auction(
            tokens={
                "0x1234567890123456789012345678901234567890": Token(
                    decimals=18,
                    symbol="UNKNOWN",
                    referencePrice=None,
                    availableBalance="10000000000000000000",
                    trusted=True,
                ),
            }
        )
        estimator = PoolBasedPriceEstimator()  # No router

        estimate = estimator.estimate_price(
            "0x1234567890123456789012345678901234567890",
            auction,
        )

        assert estimate.source == "default"
        assert estimate.price == U256_MAX

    def test_zero_reference_price_is_ignored(self):
        """Test that zero reference price falls through to other sources."""
        auction = make_auction(
            tokens={
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": Token(
                    decimals=18,
                    symbol="WETH",
                    referencePrice="0",  # Zero reference price
                    availableBalance="10000000000000000000",
                    trusted=True,
                ),
            }
        )
        estimator = PoolBasedPriceEstimator()

        # WETH should fall back to native since ref price is 0
        estimate = estimator.estimate_price(
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            auction,
        )

        assert estimate.source == "native"
        assert estimate.price == 10**18

    def test_custom_native_tokens(self):
        """Test using custom native token list."""
        custom_native = "0x1111111111111111111111111111111111111111"
        auction = make_auction(
            tokens={
                custom_native: Token(
                    decimals=18,
                    symbol="CUSTOM",
                    referencePrice=None,
                    availableBalance="10000000000000000000",
                    trusted=True,
                ),
            }
        )
        estimator = PoolBasedPriceEstimator(native_tokens=[custom_native])

        estimate = estimator.estimate_price(custom_native, auction)

        assert estimate.source == "native"
        assert estimate.price == 10**18


# --- Constants Tests ---


class TestConstants:
    """Tests for price estimation constants."""

    def test_weth_mainnet_address(self):
        """Test WETH mainnet address is correct."""
        assert WETH_MAINNET == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

    def test_wxdai_gnosis_address(self):
        """Test wxDAI Gnosis address is correct."""
        assert WXDAI_GNOSIS == "0xe91d153e0b41518a2ce8dd3d7944fa863463a97d"

    def test_default_native_tokens(self):
        """Test default native tokens include both WETH and wxDAI."""
        assert WETH_MAINNET in DEFAULT_NATIVE_TOKENS
        assert WXDAI_GNOSIS in DEFAULT_NATIVE_TOKENS
        assert len(DEFAULT_NATIVE_TOKENS) == 2

    def test_u256_max(self):
        """Test U256_MAX is correct."""
        assert U256_MAX == 2**256 - 1
