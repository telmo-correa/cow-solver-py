"""Fee configuration for the solver."""

from dataclasses import dataclass

from solver.constants import POOL_SWAP_GAS_COST, SETTLEMENT_OVERHEAD


@dataclass(frozen=True)
class FeeConfig:
    """Centralized configuration for fee calculation.

    This dataclass holds all fee-related constants and behavior flags,
    making it easy to test with different configurations and ensuring
    consistency across the codebase.

    Attributes:
        swap_gas_cost: Gas cost per swap hop (default: 60,000)
        settlement_overhead: Fixed gas overhead for settlement (default: 106,391)
        fee_base: Base unit for fee calculation (1e18)
        reject_on_missing_reference_price: If True, return error when reference
            price is missing. If False, return fee=0.
        reject_on_fee_overflow: If True, return error when fee > executed amount.
            If False, cap fee at executed amount.
    """

    # Gas costs (from constants.py)
    swap_gas_cost: int = POOL_SWAP_GAS_COST
    settlement_overhead: int = SETTLEMENT_OVERHEAD

    # Fee calculation base unit (1e18)
    fee_base: int = 10**18

    # Behavior flags
    reject_on_missing_reference_price: bool = True
    reject_on_fee_overflow: bool = True


# Default configuration instance
DEFAULT_FEE_CONFIG = FeeConfig()
