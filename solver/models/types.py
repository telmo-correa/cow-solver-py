"""Shared type definitions for CoW Protocol models.

These types are used across auction and solution models.
"""

from typing import Annotated, Any

from pydantic import BeforeValidator, Field

# Maximum uint256 value
UINT256_MAX = 2**256 - 1


def validate_uint256(value: Any) -> str:
    """Validate that a value is a valid uint256 decimal string.

    Args:
        value: Value to validate (string or int)

    Returns:
        Valid uint256 as decimal string

    Raises:
        ValueError: If value is not a valid non-negative integer within uint256 range
    """
    # Accept int directly
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"Uint256 cannot be negative: {value}")
        if value > UINT256_MAX:
            raise ValueError(f"Uint256 overflow: {value} > 2^256-1")
        return str(value)

    # Must be string
    if not isinstance(value, str):
        raise ValueError(f"Uint256 must be string or int, got {type(value).__name__}")

    # Validate it's a valid decimal integer
    try:
        int_value = int(value)
    except ValueError as err:
        raise ValueError(f"Uint256 must be a decimal integer string: '{value}'") from err

    # Check non-negative
    if int_value < 0:
        raise ValueError(f"Uint256 cannot be negative: {value}")

    # Check uint256 range
    if int_value > UINT256_MAX:
        raise ValueError(f"Uint256 overflow: {value} > 2^256-1")

    return value


# Ethereum address (40 hex chars after 0x prefix)
Address = Annotated[str, Field(pattern=r"^0x[a-fA-F0-9]{40}$")]

# 256-bit unsigned integer as decimal string (validated)
Uint256 = Annotated[
    str,
    BeforeValidator(validate_uint256),
    Field(description="256-bit unsigned integer as decimal string"),
]

# Arbitrary hex bytes
Bytes = Annotated[str, Field(pattern=r"^0x[a-fA-F0-9]*$")]

# Order UID (56 bytes = 112 hex chars)
OrderUid = Annotated[str, Field(pattern=r"^0x[a-fA-F0-9]{112}$")]


def normalize_address(address: str, *, validate: bool = False) -> str:
    """Normalize an Ethereum address to lowercase.

    Args:
        address: An Ethereum address (with or without 0x prefix)
        validate: If True, raises ValueError for invalid addresses.
                  If False (default), returns normalized form without validation.

    Returns:
        Lowercase address with 0x prefix

    Raises:
        ValueError: If validate=True and address is not a valid Ethereum address

    Note:
        When validate=False, this function does NOT check if the input is a valid
        address. It simply lowercases and adds 0x prefix. Use is_valid_address()
        to check validity, or pass validate=True for combined normalization+validation.
    """
    addr = address.lower()
    if not addr.startswith("0x"):
        addr = "0x" + addr

    if validate and not is_valid_address(addr):
        raise ValueError(f"Invalid address: {address}")

    return addr


def is_valid_address(address: str) -> bool:
    """Check if a string is a valid Ethereum address.

    Args:
        address: String to validate

    Returns:
        True if valid Ethereum address format
    """
    if not isinstance(address, str):
        return False
    if not address.startswith("0x"):
        return False
    if len(address) != 42:
        return False
    try:
        int(address, 16)
        return True
    except ValueError:
        return False
