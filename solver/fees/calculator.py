"""Fee calculator service for the solver.

Uses SafeInt for arithmetic operations to prevent:
- Division by zero (caught by early returns, SafeInt adds explicit protection)
- uint256 overflow (validated before returning fee)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import structlog

from solver.fees.config import DEFAULT_FEE_CONFIG, FeeConfig
from solver.fees.price_estimation import get_token_info
from solver.fees.result import FeeError, FeeResult
from solver.models.auction import AuctionInstance, Order, OrderClass
from solver.models.types import normalize_address
from solver.safe_int import S, Uint256Overflow

if TYPE_CHECKING:
    from solver.fees.price_estimation import PriceEstimator

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

    When reference prices are missing, can use a PriceEstimator to estimate
    prices via pool routing (matching Rust baseline behavior).

    Attributes:
        config: Fee configuration settings
        price_estimator: Optional estimator for when reference prices are missing
    """

    def __init__(
        self,
        config: FeeConfig | None = None,
        price_estimator: PriceEstimator | None = None,
    ):
        """Initialize with optional configuration and price estimator.

        Args:
            config: Fee configuration. Uses DEFAULT_FEE_CONFIG if not provided.
            price_estimator: Optional price estimator for missing reference prices.
                            If provided, enables pool-based price estimation.
        """
        self.config = config or DEFAULT_FEE_CONFIG
        self._price_estimator = price_estimator

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

        # Get reference price for sell token
        sell_token = normalize_address(order.sell_token)
        reference_price, price_source = self._get_reference_price(sell_token, auction)

        if reference_price is None:
            detail = f"No reference price for token {sell_token[-8:]}"
            logger.warning(
                "fee_calculation_missing_reference_price",
                order_uid=order.uid[:18] + "...",
                sell_token=sell_token[-8:],
            )
            if self.config.reject_on_missing_reference_price:
                return FeeResult.with_error(FeeError.MISSING_REFERENCE_PRICE, detail)
            return FeeResult.zero_fee()

        if reference_price == 0:
            detail = f"Reference price is zero for {sell_token[-8:]}"
            logger.warning(
                "fee_calculation_zero_reference_price",
                order_uid=order.uid[:18] + "...",
                sell_token=sell_token[-8:],
            )
            if self.config.reject_on_missing_reference_price:
                return FeeResult.with_error(FeeError.ZERO_REFERENCE_PRICE, detail)
            return FeeResult.zero_fee()

        # Calculate fee using SafeInt: gas_cost * 1e18 / reference_price
        # Note: reference_price is validated non-zero above
        gas_cost_wei = S(gas_estimate) * S(gas_price)
        fee = (gas_cost_wei * self.config.fee_base) // reference_price

        # Validate fee fits in uint256 before returning
        try:
            fee_uint256 = fee.to_uint256()
        except Uint256Overflow:
            logger.warning(
                "fee_calculation_overflow",
                order_uid=order.uid[:18] + "...",
                fee_value=str(fee.value),
                reason="Calculated fee exceeds uint256",
            )
            return FeeResult.with_error(
                FeeError.FEE_EXCEEDS_AMOUNT,
                f"Calculated fee {fee.value} exceeds uint256 max",
            )

        logger.debug(
            "fee_calculated",
            order_uid=order.uid[:18] + "...",
            gas_estimate=gas_estimate,
            gas_price=gas_price,
            reference_price=reference_price,
            fee=fee_uint256,
        )

        return FeeResult.with_fee(fee_uint256)

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

    def _get_reference_price(
        self, token_address: str, auction: AuctionInstance
    ) -> tuple[int | None, str]:
        """Get reference price for a token.

        Priority:
        1. Use reference price from auction if available
        2. If price_estimator is set, use it to estimate price
        3. Return None if no price available

        Args:
            token_address: Token address (should be lowercase)
            auction: The auction instance

        Returns:
            Tuple of (price, source) where:
            - price: Reference price, 0 if explicitly zero, None if not available
            - source: Where the price came from ('reference', 'pool', 'none')
        """
        # Try direct lookup first
        token_info = get_token_info(auction, token_address)
        if token_info is not None and token_info.reference_price is not None:
            price = int(token_info.reference_price)
            # Return 0 explicitly if set to 0 (different from missing)
            return price, "reference"

        # Try price estimator if available
        if self._price_estimator is not None:
            from solver.fees.price_estimation import U256_MAX

            estimate = self._price_estimator.estimate_price(token_address, auction)
            if estimate.source != "none":
                if estimate.price > 0 and estimate.price < U256_MAX:
                    logger.debug(
                        "fee_calculation_using_estimated_price",
                        token=token_address[-8:],
                        price=estimate.price,
                        source=estimate.source,
                    )
                    return estimate.price, estimate.source
                elif estimate.price == U256_MAX:
                    # U256_MAX means "no route found, use fee=0"
                    # Return a very high price so fee calculation gives ~0
                    logger.debug(
                        "fee_calculation_no_route_defaulting_zero",
                        token=token_address[-8:],
                        source=estimate.source,
                    )
                    return U256_MAX, estimate.source

        return None, "none"


# Default calculator instance
DEFAULT_FEE_CALCULATOR = DefaultFeeCalculator()
