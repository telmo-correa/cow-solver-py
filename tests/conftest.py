"""Pytest configuration and fixtures."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from solver.models import AuctionInstance

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUCTIONS_DIR = FIXTURES_DIR / "auctions"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the fixtures directory path."""
    return FIXTURES_DIR


@pytest.fixture
def auctions_dir() -> Path:
    """Return the auctions fixtures directory path."""
    return AUCTIONS_DIR


def load_auction_fixture(name: str) -> AuctionInstance:
    """Load an auction fixture by name.

    Args:
        name: Fixture name (e.g., "single_order/basic_sell")

    Returns:
        Parsed AuctionInstance
    """
    path = AUCTIONS_DIR / f"{name}.json"
    with open(path) as f:
        data = json.load(f)
    return AuctionInstance.model_validate(data)


def iter_auction_fixtures(category: str | None = None) -> Iterator[tuple[str, AuctionInstance]]:
    """Iterate over auction fixtures, optionally filtered by category.

    Args:
        category: Optional subdirectory to filter (e.g., "single_order")

    Yields:
        Tuples of (fixture_name, AuctionInstance)
    """
    search_dir = AUCTIONS_DIR / category if category else AUCTIONS_DIR
    if not search_dir.exists():
        return

    for path in search_dir.rglob("*.json"):
        rel_path = path.relative_to(AUCTIONS_DIR)
        name = str(rel_path.with_suffix(""))
        yield name, load_auction_fixture(name)


@pytest.fixture
def single_order_auctions() -> list[tuple[str, AuctionInstance]]:
    """Load all single-order auction fixtures."""
    return list(iter_auction_fixtures("single_order"))


@pytest.fixture
def cow_pair_auctions() -> list[tuple[str, AuctionInstance]]:
    """Load all CoW pair auction fixtures."""
    return list(iter_auction_fixtures("cow_pairs"))


@pytest.fixture
def multi_hop_auctions() -> list[tuple[str, AuctionInstance]]:
    """Load all multi-hop auction fixtures."""
    return list(iter_auction_fixtures("multi_hop"))
