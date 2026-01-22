"""Fee calculator service for the solver."""

from typing import Protocol

import structlog

from solver.fees.config import DEFAULT_FEE_CONFIG, FeeConfig
from solver.fees.result import FeeError, FeeResult
from solver.models.auction import AuctionInstance, Order, OrderClass, Token

logger = structlog.get_logger()


class FeeCalculator(Protocol):
    """Protocol for fee calculation.

    This protocol defines the interface for calculating solver fees.
    Different implementations can be used for testing or for different
    fee calculation strategies.
    """

    def calculate_solver_fee(
        self,
        order: Order,
        gas_estimate: int,
        auction: AuctionInstance | None,
    ) -> FeeResult:
        """Calculate the solver fee for an order.

        Args:
            order: The order to calculate fee for
            gas_estimate: Estimated gas cost for executing this order
            auction: The auction instance with gas price and reference prices

        Returns:
            FeeResult with the calculated fee or error information
        """
        ...

    def validate_fee_against_amount(
        self,
        fee: int,
        executed_amount: int,
        is_sell_order: bool,
    ) -> FeeResult:
        """Validate that the fee doesn't exceed the executed amount.

        For sell orders, the fee is deducted from the executed amount,
        so fee must be <= executed_amount.

        Args:
            fee: The calculated fee amount
            executed_amount: The amount being executed
            is_sell_order: Whether this is a sell order

        Returns:
            FeeResult with the validated fee or error if overflow
        """
        ...


class DefaultFeeCalculator:
    """Default implementation of fee calculation.

    Calculates solver fees for limit orders matching the Rust baseline:
        fee = gas_cost_wei * 1e18 / reference_price

    Market orders return no fee (protocol handles them).

    Attributes:
        config: Fee configuration settings
    """

    def __init__(self, config: FeeConfig | None = None):
        """Initialize with optional configuration.

        Args:
            config: Fee configuration. Uses DEFAULT_FEE_CONFIG if not provided.
        """
        self.config = config or DEFAULT_FEE_CONFIG

    def calculate_solver_fee(
        self,
        order: Order,
        gas_estimate: int,
        auction: AuctionInstance | None,
    ) -> FeeResult:
        """Calculate the solver fee for an order.

        For limit orders, calculates fee based on gas cost and reference price.
        For market orders, returns no fee (protocol determines fee).

        Fee formula (matching Rust baseline):
            fee = gas_cost_wei * 1e18 / reference_price

        Where:
            - gas_cost_wei = gas_estimate * auction.effective_gas_price
            - reference_price = price in wei to buy 1e18 of the sell token

        Args:
            order: The order to calculate fee for
            gas_estimate: Estimated gas cost for executing this order
            auction: The auction instance with gas price and reference prices

        Returns:
            FeeResult with the calculated fee or error information
        """
        # Only limit orders require solver-determined fee
        if order.class_ != OrderClass.LIMIT:
            return FeeResult.no_fee()

        # Validate auction data
        if auction is None:
            logger.warning(
                "fee_calculation_missing_auction",
                order_uid=order.uid[:18] + "...",
            )
            return FeeResult.with_error(
                FeeError.MISSING_AUCTION,
                "Cannot calculate fee without auction data",
            )

        # Get gas price from auction
        gas_price = int(auction.effective_gas_price) if auction.effective_gas_price else 0
        if gas_price == 0:
            logger.debug(
                "fee_calculation_zero_gas_price",
                order_uid=order.uid[:18] + "...",
            )
            return FeeResult.zero_fee()

        # Get reference price for sell token (case-insensitive lookup)
        sell_token = order.sell_token.lower()
        token_info = self._get_token_info(auction, sell_token)

        if token_info is None or token_info.reference_price is None:
            detail = f"No reference price for token {sell_token[-8:]}"
            logger.warning(
                "fee_calculation_missing_reference_price",
                order_uid=order.uid[:18] + "...",
                sell_token=sell_token[-8:],
            )
            if self.config.reject_on_missing_reference_price:
                return FeeResult.with_error(FeeError.MISSING_REFERENCE_PRICE, detail)
            return FeeResult.zero_fee()

        reference_price = int(token_info.reference_price)
        if reference_price == 0:
            logger.warning(
                "fee_calculation_zero_reference_price",
                order_uid=order.uid[:18] + "...",
                sell_token=sell_token[-8:],
            )
            if self.config.reject_on_missing_reference_price:
                return FeeResult.with_error(
                    FeeError.ZERO_REFERENCE_PRICE,
                    f"Reference price is zero for {sell_token[-8:]}",
                )
            return FeeResult.zero_fee()

        # Calculate fee: gas_cost * 1e18 / reference_price
        gas_cost_wei = gas_estimate * gas_price
        fee = (gas_cost_wei * self.config.fee_base) // reference_price

        logger.debug(
            "fee_calculated",
            order_uid=order.uid[:18] + "...",
            gas_estimate=gas_estimate,
            gas_price=gas_price,
            reference_price=reference_price,
            fee=fee,
        )

        return FeeResult.with_fee(fee)

    def validate_fee_against_amount(
        self,
        fee: int,
        executed_amount: int,
        is_sell_order: bool,
    ) -> FeeResult:
        """Validate that the fee doesn't exceed the executed amount.

        For sell orders, the fee is deducted from the executed amount.
        If fee > executed_amount, this is an overflow condition.

        The behavior depends on config.reject_on_fee_overflow:
        - If True: Return error (matches Rust's checked_sub behavior)
        - If False: Cap fee at executed_amount

        Args:
            fee: The calculated fee amount
            executed_amount: The amount being executed
            is_sell_order: Whether this is a sell order

        Returns:
            FeeResult with the validated fee or error if overflow
        """
        if not is_sell_order:
            # Buy orders: fee is added to sell side, no overflow check here
            return FeeResult.with_fee(fee)

        if fee <= executed_amount:
            return FeeResult.with_fee(fee)

        # Fee exceeds executed amount - this is an overflow
        if self.config.reject_on_fee_overflow:
            logger.warning(
                "fee_exceeds_executed_amount",
                fee=fee,
                executed_amount=executed_amount,
                reason="Fee overflow - trade rejected (matches Rust behavior)",
            )
            return FeeResult.with_error(
                FeeError.FEE_EXCEEDS_AMOUNT,
                f"Fee {fee} exceeds executed amount {executed_amount}",
            )

        # Cap fee at executed amount
        logger.warning(
            "fee_capped_at_executed_amount",
            original_fee=fee,
            capped_fee=executed_amount,
        )
        return FeeResult.with_fee(executed_amount)

    @staticmethod
    def _get_token_info(auction: AuctionInstance, token_address: str) -> Token | None:
        """Get token info with case-insensitive address lookup.

        Token addresses may have different case in orders vs tokens dict.
        This method normalizes the lookup to handle both cases.

        Args:
            auction: The auction instance
            token_address: Token address (should be lowercase)

        Returns:
            Token info if found, None otherwise
        """
        # Try direct lookup first (most common case)
        token_info = auction.tokens.get(token_address)
        if token_info is not None:
            return token_info

        # Try case-insensitive lookup
        token_lower = token_address.lower()
        for addr, info in auction.tokens.items():
            if addr.lower() == token_lower:
                return info

        return None


# Default calculator instance
DEFAULT_FEE_CALCULATOR = DefaultFeeCalculator()
