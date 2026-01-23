"""Tests for the fee calculator service."""

import pytest

from solver.fees import (
    DEFAULT_FEE_CALCULATOR,
    DefaultFeeCalculator,
    FeeConfig,
    FeeError,
    FeeResult,
)
from solver.models.auction import (
    AuctionInstance,
    Order,
    OrderClass,
    OrderKind,
    Token,
)

# --- Test Fixtures ---


def make_order(
    *,
    sell_token: str = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    buy_token: str = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    sell_amount: str = "1000000000",  # 1000 USDC
    buy_amount: str = "400000000000000000",  # 0.4 WETH
    kind: OrderKind = OrderKind.SELL,
    order_class: OrderClass = OrderClass.MARKET,
    uid: str = "0xaa01020304050607080910111213141516171819202122232425262728293031333435363738394041424344454647484950515253545556",
) -> Order:
    """Create a test order with sensible defaults."""
    return Order(
        uid=uid,
        sellToken=sell_token,
        buyToken=buy_token,
        sellAmount=sell_amount,
        buyAmount=buy_amount,
        fullSellAmount=sell_amount,
        fullBuyAmount=buy_amount,
        kind=kind,
        **{"class": order_class},
        partiallyFillable=False,
    )


def make_auction(
    orders: list[Order],
    gas_price: str = "15000000000",  # 15 gwei
    usdc_ref_price: str = "450000000000000000000000000",  # ~4.5e26
    weth_ref_price: str = "1000000000000000000",  # 1e18
) -> AuctionInstance:
    """Create a test auction with token reference prices."""
    return AuctionInstance(
        id="test-auction",
        tokens={
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": Token(
                decimals=6,
                symbol="USDC",
                referencePrice=usdc_ref_price,
                availableBalance="10000000000",
                trusted=True,
            ),
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": Token(
                decimals=18,
                symbol="WETH",
                referencePrice=weth_ref_price,
                availableBalance="10000000000000000000",
                trusted=True,
            ),
        },
        orders=orders,
        liquidity=[],
        effectiveGasPrice=gas_price,
    )


# --- FeeResult Tests ---


class TestFeeResult:
    """Tests for FeeResult dataclass."""

    def test_no_fee_is_valid(self):
        """FeeResult.no_fee() should be valid with None fee."""
        result = FeeResult.no_fee()
        assert result.is_valid
        assert not result.is_error
        assert not result.requires_fee
        assert result.fee is None

    def test_zero_fee_is_valid(self):
        """FeeResult.zero_fee() should be valid with 0 fee."""
        result = FeeResult.zero_fee()
        assert result.is_valid
        assert not result.is_error
        assert not result.requires_fee  # 0 fee doesn't require application
        assert result.fee == 0

    def test_with_fee_is_valid(self):
        """FeeResult.with_fee() should be valid and require fee."""
        result = FeeResult.with_fee(5000000)
        assert result.is_valid
        assert not result.is_error
        assert result.requires_fee
        assert result.fee == 5000000

    def test_with_error_is_invalid(self):
        """FeeResult.with_error() should be invalid."""
        result = FeeResult.with_error(FeeError.MISSING_REFERENCE_PRICE, "No price")
        assert not result.is_valid
        assert result.is_error
        assert not result.requires_fee
        assert result.error == FeeError.MISSING_REFERENCE_PRICE
        assert result.error_detail == "No price"

    def test_frozen(self):
        """FeeResult should be immutable."""
        result = FeeResult.with_fee(100)
        with pytest.raises(AttributeError):
            result.fee = 200  # type: ignore


# --- FeeConfig Tests ---


class TestFeeConfig:
    """Tests for FeeConfig dataclass."""

    def test_default_values(self):
        """Default config should have expected values."""
        config = FeeConfig()
        assert config.swap_gas_cost == 60_000
        assert config.settlement_overhead == 106_391
        assert config.fee_base == 10**18
        assert config.reject_on_missing_reference_price is True
        assert config.reject_on_fee_overflow is True

    def test_custom_values(self):
        """Config should accept custom values."""
        config = FeeConfig(
            swap_gas_cost=80_000,
            reject_on_missing_reference_price=False,
        )
        assert config.swap_gas_cost == 80_000
        assert config.reject_on_missing_reference_price is False

    def test_frozen(self):
        """FeeConfig should be immutable."""
        config = FeeConfig()
        with pytest.raises(AttributeError):
            config.swap_gas_cost = 100_000  # type: ignore


# --- DefaultFeeCalculator Tests ---


class TestDefaultFeeCalculator:
    """Tests for DefaultFeeCalculator."""

    def test_market_order_no_fee(self):
        """Market orders should return no fee."""
        calculator = DefaultFeeCalculator()
        order = make_order(order_class=OrderClass.MARKET)
        auction = make_auction([order])

        result = calculator.calculate_solver_fee(order, 150000, auction)

        assert result.is_valid
        assert result.fee is None
        assert not result.requires_fee

    def test_limit_order_has_fee(self):
        """Limit orders should return a calculated fee."""
        calculator = DefaultFeeCalculator()
        order = make_order(order_class=OrderClass.LIMIT)
        auction = make_auction([order])

        result = calculator.calculate_solver_fee(order, 150000, auction)

        assert result.is_valid
        assert result.requires_fee
        assert result.fee is not None
        assert result.fee > 0

    def test_fee_formula_matches_rust(self):
        """Fee should match Rust formula: gas_cost * 1e18 / reference_price."""
        calculator = DefaultFeeCalculator()
        order = make_order(order_class=OrderClass.LIMIT)

        gas = 150000
        gas_price = 15000000000  # 15 gwei
        usdc_ref_price = 450000000000000000000000000  # 4.5e26

        auction = make_auction(
            [order],
            gas_price=str(gas_price),
            usdc_ref_price=str(usdc_ref_price),
        )

        result = calculator.calculate_solver_fee(order, gas, auction)

        # Expected: gas * gas_price * 1e18 / ref_price
        gas_cost_wei = gas * gas_price
        expected_fee = (gas_cost_wei * 10**18) // usdc_ref_price

        assert result.is_valid
        assert result.fee == expected_fee

    def test_missing_auction_returns_error(self):
        """Missing auction should return error."""
        calculator = DefaultFeeCalculator()
        order = make_order(order_class=OrderClass.LIMIT)

        result = calculator.calculate_solver_fee(order, 150000, None)

        assert result.is_error
        assert result.error == FeeError.MISSING_AUCTION

    def test_missing_reference_price_returns_error(self):
        """Missing reference price should return error with default config."""
        calculator = DefaultFeeCalculator()
        # Use a token not in the auction
        order = make_order(
            order_class=OrderClass.LIMIT,
            sell_token="0x1111111111111111111111111111111111111111",
        )
        auction = make_auction([order])

        result = calculator.calculate_solver_fee(order, 150000, auction)

        assert result.is_error
        assert result.error == FeeError.MISSING_REFERENCE_PRICE

    def test_missing_reference_price_returns_zero_when_configured(self):
        """Missing reference price should return 0 when configured to not reject."""
        config = FeeConfig(reject_on_missing_reference_price=False)
        calculator = DefaultFeeCalculator(config)
        order = make_order(
            order_class=OrderClass.LIMIT,
            sell_token="0x1111111111111111111111111111111111111111",
        )
        auction = make_auction([order])

        result = calculator.calculate_solver_fee(order, 150000, auction)

        assert result.is_valid
        assert result.fee == 0

    def test_zero_gas_price_returns_zero_fee(self):
        """Zero gas price should return 0 fee."""
        calculator = DefaultFeeCalculator()
        order = make_order(order_class=OrderClass.LIMIT)
        auction = make_auction([order], gas_price="0")

        result = calculator.calculate_solver_fee(order, 150000, auction)

        assert result.is_valid
        assert result.fee == 0

    def test_zero_reference_price_returns_error(self):
        """Zero reference price should return error."""
        calculator = DefaultFeeCalculator()
        order = make_order(order_class=OrderClass.LIMIT)
        auction = make_auction([order], usdc_ref_price="0")

        result = calculator.calculate_solver_fee(order, 150000, auction)

        assert result.is_error
        assert result.error == FeeError.ZERO_REFERENCE_PRICE

    def test_liquidity_order_no_fee(self):
        """Liquidity orders should return no fee."""
        calculator = DefaultFeeCalculator()
        order = make_order(order_class=OrderClass.LIQUIDITY)
        auction = make_auction([order])

        result = calculator.calculate_solver_fee(order, 150000, auction)

        assert result.is_valid
        assert result.fee is None


class TestFeeValidation:
    """Tests for fee validation against executed amount."""

    def test_valid_fee_passes(self):
        """Fee less than executed amount should pass."""
        calculator = DefaultFeeCalculator()

        result = calculator.validate_fee_against_amount(
            fee=100,
            executed_amount=1000,
            is_sell_order=True,
        )

        assert result.is_valid
        assert result.fee == 100

    def test_fee_equals_amount_passes(self):
        """Fee equal to executed amount should pass (edge case)."""
        calculator = DefaultFeeCalculator()

        result = calculator.validate_fee_against_amount(
            fee=1000,
            executed_amount=1000,
            is_sell_order=True,
        )

        assert result.is_valid
        assert result.fee == 1000

    def test_fee_exceeds_amount_returns_error(self):
        """Fee exceeding executed amount should return error with default config."""
        calculator = DefaultFeeCalculator()

        result = calculator.validate_fee_against_amount(
            fee=2000,
            executed_amount=1000,
            is_sell_order=True,
        )

        assert result.is_error
        assert result.error == FeeError.FEE_EXCEEDS_AMOUNT

    def test_fee_exceeds_amount_caps_when_configured(self):
        """Fee exceeding amount should be capped when configured."""
        config = FeeConfig(reject_on_fee_overflow=False)
        calculator = DefaultFeeCalculator(config)

        result = calculator.validate_fee_against_amount(
            fee=2000,
            executed_amount=1000,
            is_sell_order=True,
        )

        assert result.is_valid
        assert result.fee == 1000  # Capped at executed amount

    def test_buy_order_no_overflow_check(self):
        """Buy orders should not have overflow check (fee added to sell side)."""
        calculator = DefaultFeeCalculator()

        result = calculator.validate_fee_against_amount(
            fee=2000,
            executed_amount=1000,
            is_sell_order=False,  # Buy order
        )

        # Buy orders don't fail on fee > executed because fee is on sell side
        assert result.is_valid
        assert result.fee == 2000


class TestDefaultCalculatorInstance:
    """Tests for the default calculator instance."""

    def test_default_instance_exists(self):
        """DEFAULT_FEE_CALCULATOR should be available."""
        assert DEFAULT_FEE_CALCULATOR is not None
        assert isinstance(DEFAULT_FEE_CALCULATOR, DefaultFeeCalculator)

    def test_default_instance_uses_default_config(self):
        """Default instance should use default config."""
        calculator = DEFAULT_FEE_CALCULATOR
        assert calculator.config.swap_gas_cost == 60_000
        assert calculator.config.reject_on_missing_reference_price is True
