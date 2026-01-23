"""Pydantic models for CoW Protocol data structures."""

from solver.models.auction import AuctionInstance, Order, Token
from solver.models.order_groups import OrderGroup, find_cow_opportunities, group_orders_by_pair
from solver.models.solution import Interaction, Solution, SolverResponse, TokenAmount, Trade
from solver.models.types import Address, Bytes, OrderUid, Uint256

__all__ = [
    # Types
    "Address",
    "Bytes",
    "OrderUid",
    "Uint256",
    # Auction models
    "AuctionInstance",
    "Order",
    "Token",
    # Order grouping (for Phase 4 optimization)
    "OrderGroup",
    "group_orders_by_pair",
    "find_cow_opportunities",
    # Solution models
    "Solution",
    "SolverResponse",
    "Trade",
    "Interaction",
    "TokenAmount",
]
