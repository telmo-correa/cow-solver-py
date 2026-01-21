"""Shared type definitions for CoW Protocol models.

These types are used across auction and solution models.
"""

from typing import Annotated

from pydantic import Field

# Ethereum address (40 hex chars after 0x prefix)
Address = Annotated[str, Field(pattern=r"^0x[a-fA-F0-9]{40}$")]

# 256-bit unsigned integer as decimal string
Uint256 = Annotated[str, Field(description="256-bit unsigned integer as decimal string")]

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
