"""Factory functions for creating test objects.

Usage:
    from tests.helpers import make_order
    # or
    from tests.helpers.factories import make_order, make_named_order

    order = make_order(sell_token=WETH, buy_token=USDC)
"""

from solver.models.auction import Order, OrderClass, OrderKind
from tests.helpers.constants import USDC, WETH

# Global counter for unique UIDs in named orders
_uid_counter = 0


def make_order(
    sell_token: str = WETH,
    buy_token: str = USDC,
    sell_amount: str | int = "1000000000000000000",  # 1 WETH
    buy_amount: str | int = "2000000000",  # 2000 USDC
    kind: OrderKind | str = OrderKind.SELL,
    order_class: OrderClass | str = OrderClass.LIMIT,
    uid: str | None = None,
    partially_fillable: bool = False,
) -> Order:
    """Create a test order with sensible defaults.

    This is the primary factory function for creating orders in tests.
    Supports both string and int amounts, and both enum and string kinds.

    Args:
        sell_token: Token to sell (default: WETH)
        buy_token: Token to buy (default: USDC)
        sell_amount: Amount to sell as string or int (default: 1 WETH)
        buy_amount: Amount to buy as string or int (default: 2000 USDC)
        kind: Order kind - "sell"/"buy" or OrderKind enum (default: SELL)
        order_class: Order class - "market"/"limit" or OrderClass enum (default: LIMIT)
        uid: Order UID (default: auto-generated)
        partially_fillable: Whether order can be partially filled (default: False)

    Returns:
        Order instance ready for testing
    """
    if uid is None:
        uid = "0x" + "01" * 56  # Default test UID

    # Convert amounts to strings if they're ints
    if isinstance(sell_amount, int):
        sell_amount = str(sell_amount)
    if isinstance(buy_amount, int):
        buy_amount = str(buy_amount)

    # Convert kind to string if it's an enum
    kind_str = kind.value if isinstance(kind, OrderKind) else kind

    # Convert order_class to OrderClass enum if it's a string
    if isinstance(order_class, str):
        order_class = OrderClass(order_class)

    return Order(
        uid=uid,
        sellToken=sell_token,
        buyToken=buy_token,
        sellAmount=sell_amount,
        buyAmount=buy_amount,
        kind=kind_str,
        partiallyFillable=partially_fillable,
        **{"class": order_class},
    )


def make_named_order(
    name: str,  # noqa: ARG001 - used for documentation only
    sell_token: str,
    buy_token: str,
    sell_amount: int,
    buy_amount: int,
    partially_fillable: bool = False,
    kind: str = "sell",
    order_class: str = "limit",
) -> Order:
    """Create a test order with auto-generated UID for named scenarios.

    This variant is useful for tests that create multiple orders and need
    unique UIDs for each. The name parameter is for documentation/debugging
    only and is not stored in the order.

    Args:
        name: Descriptive name for the order (for documentation only)
        sell_token: Token to sell
        buy_token: Token to buy
        sell_amount: Amount to sell (as int, converted to string)
        buy_amount: Amount to buy (as int, converted to string)
        partially_fillable: Whether order can be partially filled (default: False)
        kind: Order kind (default: "sell")
        order_class: Order class (default: "limit")

    Returns:
        Order instance with unique UID
    """
    global _uid_counter
    _uid_counter += 1

    # Generate unique UID based on counter
    uid = f"0x{_uid_counter:0112x}"

    return make_order(
        sell_token=sell_token,
        buy_token=buy_token,
        sell_amount=sell_amount,
        buy_amount=buy_amount,
        kind=kind,
        order_class=order_class,
        uid=uid,
        partially_fillable=partially_fillable,
    )


def reset_uid_counter() -> None:
    """Reset the UID counter (useful for test isolation)."""
    global _uid_counter
    _uid_counter = 0


__all__ = ["make_order", "make_named_order", "reset_uid_counter"]
