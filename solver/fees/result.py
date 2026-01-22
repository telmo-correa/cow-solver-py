"""Fee calculation result types."""

from dataclasses import dataclass
from enum import Enum


class FeeError(Enum):
    """Types of fee calculation errors."""

    MISSING_AUCTION = "missing_auction"
    MISSING_REFERENCE_PRICE = "missing_reference_price"
    ZERO_REFERENCE_PRICE = "zero_reference_price"
    ZERO_GAS_PRICE = "zero_gas_price"
    FEE_EXCEEDS_AMOUNT = "fee_exceeds_amount"


@dataclass(frozen=True)
class FeeResult:
    """Result of a fee calculation.

    This dataclass provides explicit success/failure handling for fee
    calculations, avoiding silent None returns that can mask errors.

    Attributes:
        fee: The calculated fee amount in sell token units, or None if
            no fee should be charged (e.g., market orders).
        error: If calculation failed, the type of error that occurred.
        error_detail: Optional human-readable detail about the error.

    Examples:
        # Successful calculation
        result = FeeResult(fee=5000000)
        assert result.is_valid
        assert result.fee == 5000000

        # Market order (no fee needed)
        result = FeeResult(fee=None)
        assert result.is_valid
        assert result.fee is None

        # Error case
        result = FeeResult(fee=None, error=FeeError.MISSING_REFERENCE_PRICE)
        assert not result.is_valid
    """

    fee: int | None
    error: FeeError | None = None
    error_detail: str | None = None

    @property
    def is_valid(self) -> bool:
        """True if calculation succeeded (even if fee is None/0)."""
        return self.error is None

    @property
    def is_error(self) -> bool:
        """True if calculation failed with an error."""
        return self.error is not None

    @property
    def requires_fee(self) -> bool:
        """True if a non-zero fee should be applied."""
        return self.is_valid and self.fee is not None and self.fee > 0

    @classmethod
    def no_fee(cls) -> "FeeResult":
        """Create a result indicating no fee is needed (e.g., market orders)."""
        return cls(fee=None)

    @classmethod
    def zero_fee(cls) -> "FeeResult":
        """Create a result with zero fee."""
        return cls(fee=0)

    @classmethod
    def with_fee(cls, amount: int) -> "FeeResult":
        """Create a successful result with a fee amount."""
        return cls(fee=amount)

    @classmethod
    def with_error(cls, error: FeeError, detail: str | None = None) -> "FeeResult":
        """Create an error result."""
        return cls(fee=None, error=error, error_detail=detail)
