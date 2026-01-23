"""Tests for 0x limit order AMM and pool."""

import pytest

from solver.amm.limit_order import LimitOrderAMM
from solver.pools.limit_order import LimitOrderPool, parse_limit_order


class TestLimitOrderPool:
    """Tests for LimitOrderPool dataclass."""

    def test_create_pool(self):
        """Test creating a limit order pool."""
        pool = LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
            taker_token="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
            maker_amount=2500_000_000,  # 2500 USDC
            taker_amount=1_000_000_000_000_000_000,  # 1 WETH
            taker_token_fee_amount=2_500_000,  # 0.1% fee
            gas_estimate=66_358,
        )

        assert pool.id == "0"
        assert pool.maker_amount == 2500_000_000
        assert pool.taker_amount == 1_000_000_000_000_000_000
        assert pool.gas_estimate == 66_358

    def test_liquidity_id_property(self):
        """Test that liquidity_id returns id."""
        pool = LimitOrderPool(
            id="test-123",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            taker_token="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            maker_amount=1000,
            taker_amount=1000,
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

        assert pool.liquidity_id == "test-123"

    def test_supports_pair_valid_direction(self):
        """Test supports_pair returns True for valid direction."""
        pool = LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC (output)
            taker_token="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH (input)
            maker_amount=2500_000_000,
            taker_amount=1_000_000_000_000_000_000,
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

        # Valid: WETH -> USDC
        assert pool.supports_pair(
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        )

    def test_supports_pair_invalid_direction(self):
        """Test supports_pair returns False for invalid direction."""
        pool = LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC (output)
            taker_token="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH (input)
            maker_amount=2500_000_000,
            taker_amount=1_000_000_000_000_000_000,
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

        # Invalid: USDC -> WETH (reverse direction)
        assert not pool.supports_pair(
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
        )

    def test_supports_pair_wrong_tokens(self):
        """Test supports_pair returns False for wrong tokens."""
        pool = LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            taker_token="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            maker_amount=2500_000_000,
            taker_amount=1_000_000_000_000_000_000,
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

        # Wrong tokens entirely
        assert not pool.supports_pair(
            "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI
            "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
        )

    def test_supports_pair_case_insensitive(self):
        """Test supports_pair handles case differences."""
        pool = LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            taker_token="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            maker_amount=2500_000_000,
            taker_amount=1_000_000_000_000_000_000,
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

        # Mixed case should work
        assert pool.supports_pair(
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH mixed case
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC mixed case
        )


class TestParseLimitOrder:
    """Tests for parse_limit_order function."""

    def test_parse_valid_limit_order(self):
        """Test parsing a valid limit order."""
        liquidity = {
            "kind": "limitOrder",
            "id": "0",
            "address": "0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            "gasEstimate": "66358",
            "hash": "0x0000000000000000000000000000000000000000000000000000000000000000",
            "makerToken": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            "takerToken": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            "makerAmount": "2500000000",
            "takerAmount": "1000000000000000000",
            "takerTokenFeeAmount": "2500000",
        }

        pool = parse_limit_order(liquidity)

        assert pool is not None
        assert pool.id == "0"
        assert pool.address == "0xdef1c0ded9bec7f1a1670819833240f027b25eff"
        assert pool.maker_token == "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        assert pool.taker_token == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
        assert pool.maker_amount == 2500_000_000
        assert pool.taker_amount == 1_000_000_000_000_000_000
        assert pool.taker_token_fee_amount == 2_500_000
        assert pool.gas_estimate == 66_358

    def test_parse_wrong_kind(self):
        """Test parsing returns None for wrong kind."""
        liquidity = {
            "kind": "constantProduct",
            "id": "0",
            "address": "0xdef1c0ded9bec7f1a1670819833240f027b25eff",
        }

        pool = parse_limit_order(liquidity)
        assert pool is None

    def test_parse_missing_fields(self):
        """Test parsing returns None for missing required fields."""
        liquidity = {
            "kind": "limitOrder",
            "id": "0",
            # Missing address, makerToken, etc.
        }

        pool = parse_limit_order(liquidity)
        assert pool is None

    def test_parse_default_gas_estimate(self):
        """Test parsing uses default gas estimate when not provided."""
        liquidity = {
            "kind": "limitOrder",
            "id": "0",
            "address": "0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            "makerToken": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            "takerToken": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            "makerAmount": "1000",
            "takerAmount": "1000",
            # No gasEstimate - should use default
        }

        pool = parse_limit_order(liquidity)

        assert pool is not None
        assert pool.gas_estimate == 66_358  # Default value


class TestLimitOrderAMM:
    """Tests for LimitOrderAMM swap calculations."""

    WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

    @pytest.fixture
    def amm(self):
        """Create a LimitOrderAMM instance."""
        return LimitOrderAMM()

    @pytest.fixture
    def pool(self):
        """Create a test limit order pool.

        Offers 2500 USDC for 1 WETH (rate: 2500 USDC/WETH)
        """
        return LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token=self.USDC,  # Output
            taker_token=self.WETH,  # Input
            maker_amount=2500_000_000,  # 2500 USDC (6 decimals)
            taker_amount=1_000_000_000_000_000_000,  # 1 WETH (18 decimals)
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

    def test_simulate_swap_full_amount(self, amm, pool):
        """Test swapping the full taker amount."""
        result = amm.simulate_swap(
            pool,
            token_in=self.WETH,
            token_out=self.USDC,
            amount_in=1_000_000_000_000_000_000,  # 1 WETH
        )

        assert result is not None
        assert result.amount_in == 1_000_000_000_000_000_000
        assert result.amount_out == 2500_000_000  # 2500 USDC
        assert result.gas_estimate == 66_358

    def test_simulate_swap_partial_amount(self, amm, pool):
        """Test swapping a partial amount."""
        result = amm.simulate_swap(
            pool,
            token_in=self.WETH,
            token_out=self.USDC,
            amount_in=500_000_000_000_000_000,  # 0.5 WETH
        )

        assert result is not None
        assert result.amount_in == 500_000_000_000_000_000
        assert result.amount_out == 1250_000_000  # 1250 USDC (half of 2500)

    def test_simulate_swap_exceeds_limit(self, amm, pool):
        """Test that swapping more than taker_amount caps at the limit."""
        result = amm.simulate_swap(
            pool,
            token_in=self.WETH,
            token_out=self.USDC,
            amount_in=2_000_000_000_000_000_000,  # 2 WETH (exceeds 1 WETH limit)
        )

        # Should cap at pool's maximum capacity (1 WETH -> 2500 USDC)
        assert result is not None
        assert result.amount_in == 1_000_000_000_000_000_000  # Capped to 1 WETH
        assert result.amount_out == 2500_000_000  # Full 2500 USDC

    def test_simulate_swap_wrong_direction(self, amm, pool):
        """Test that swapping in wrong direction fails."""
        result = amm.simulate_swap(
            pool,
            token_in=self.USDC,  # Wrong: USDC is maker (output)
            token_out=self.WETH,  # Wrong: WETH is taker (input)
            amount_in=1000_000_000,
        )

        assert result is None

    def test_simulate_swap_wrong_tokens(self, amm, pool):
        """Test that swapping with wrong tokens fails."""
        result = amm.simulate_swap(
            pool,
            token_in="0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI
            token_out="0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
            amount_in=1000,
        )

        assert result is None

    def test_simulate_swap_exact_output_full(self, amm, pool):
        """Test exact output swap for full amount."""
        result = amm.simulate_swap_exact_output(
            pool,
            token_in=self.WETH,
            token_out=self.USDC,
            amount_out=2500_000_000,  # Full 2500 USDC
        )

        assert result is not None
        assert result.amount_out == 2500_000_000
        assert result.amount_in == 1_000_000_000_000_000_000  # 1 WETH

    def test_simulate_swap_exact_output_partial(self, amm, pool):
        """Test exact output swap for partial amount."""
        result = amm.simulate_swap_exact_output(
            pool,
            token_in=self.WETH,
            token_out=self.USDC,
            amount_out=1250_000_000,  # 1250 USDC (half)
        )

        assert result is not None
        assert result.amount_out == 1250_000_000
        # input = 1250 * 1e18 / 2500 = 0.5 WETH
        assert result.amount_in == 500_000_000_000_000_000

    def test_simulate_swap_exact_output_exceeds_limit(self, amm, pool):
        """Test that requesting more than maker_amount fails."""
        result = amm.simulate_swap_exact_output(
            pool,
            token_in=self.WETH,
            token_out=self.USDC,
            amount_out=5000_000_000,  # 5000 USDC (exceeds 2500 limit)
        )

        assert result is None

    def test_simulate_swap_exact_output_rounds_up(self, amm, pool):
        """Test that exact output rounds up input for buyer protection."""
        # Create a pool where rounding matters
        pool = LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token=self.USDC,
            taker_token=self.WETH,
            maker_amount=1000,  # 1000 wei USDC
            taker_amount=3,  # 3 wei WETH (awkward ratio for rounding)
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

        result = amm.simulate_swap_exact_output(
            pool,
            token_in=self.WETH,
            token_out=self.USDC,
            amount_out=1,  # Want 1 wei USDC
        )

        assert result is not None
        # input = ceil(1 * 3 / 1000) = ceil(0.003) = 1 (rounds up)
        assert result.amount_in == 1  # Rounded up

    def test_linear_pricing_no_slippage(self, amm, pool):
        """Test that limit orders have linear pricing (no slippage curve)."""
        # Small swap
        result_small = amm.simulate_swap(
            pool,
            token_in=self.WETH,
            token_out=self.USDC,
            amount_in=100_000_000_000_000_000,  # 0.1 WETH
        )

        # Large swap (10x)
        result_large = amm.simulate_swap(
            pool,
            token_in=self.WETH,
            token_out=self.USDC,
            amount_in=1_000_000_000_000_000_000,  # 1 WETH (full amount)
        )

        assert result_small is not None
        assert result_large is not None

        # Verify linear scaling: output should be exactly 10x
        assert result_large.amount_out == result_small.amount_out * 10


class TestLimitOrderRegistry:
    """Tests for limit orders in PoolRegistry."""

    WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

    def test_add_and_get_limit_orders(self):
        """Test adding and retrieving limit orders."""
        from solver.pools.registry import PoolRegistry

        registry = PoolRegistry()

        order = LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token=self.USDC,
            taker_token=self.WETH,
            maker_amount=2500_000_000,
            taker_amount=1_000_000_000_000_000_000,
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

        registry.add_limit_order(order)

        # Get in correct direction
        orders = registry.get_limit_orders(self.WETH, self.USDC)
        assert len(orders) == 1
        assert orders[0].id == "0"

        # Get in wrong direction (should be empty)
        orders_reverse = registry.get_limit_orders(self.USDC, self.WETH)
        assert len(orders_reverse) == 0

    def test_limit_order_in_pools_for_pair(self):
        """Test that limit orders appear in get_pools_for_pair."""
        from solver.pools.registry import PoolRegistry

        registry = PoolRegistry()

        order = LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token=self.USDC,
            taker_token=self.WETH,
            maker_amount=2500_000_000,
            taker_amount=1_000_000_000_000_000_000,
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

        registry.add_limit_order(order)

        # Should appear when querying in correct direction
        pools = registry.get_pools_for_pair(self.WETH, self.USDC)
        assert len(pools) == 1
        assert isinstance(pools[0], LimitOrderPool)

    def test_duplicate_limit_orders_ignored(self):
        """Test that duplicate orders (same ID) are ignored."""
        from solver.pools.registry import PoolRegistry

        registry = PoolRegistry()

        order1 = LimitOrderPool(
            id="0",
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token=self.USDC,
            taker_token=self.WETH,
            maker_amount=2500_000_000,
            taker_amount=1_000_000_000_000_000_000,
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

        order2 = LimitOrderPool(
            id="0",  # Same ID
            address="0xdef1c0ded9bec7f1a1670819833240f027b25eff",
            maker_token=self.USDC,
            taker_token=self.WETH,
            maker_amount=5000_000_000,  # Different amount
            taker_amount=2_000_000_000_000_000_000,
            taker_token_fee_amount=0,
            gas_estimate=66_358,
        )

        registry.add_limit_order(order1)
        registry.add_limit_order(order2)

        assert registry.limit_order_count == 1

    def test_build_registry_from_liquidity_with_limit_orders(self):
        """Test that build_registry_from_liquidity parses limit orders."""
        from solver.pools.registry import build_registry_from_liquidity

        liquidity = [
            {
                "kind": "limitOrder",
                "id": "0",
                "address": "0xdef1c0ded9bec7f1a1670819833240f027b25eff",
                "gasEstimate": "66358",
                "hash": "0x0000000000000000000000000000000000000000000000000000000000000000",
                "makerToken": self.USDC,
                "takerToken": self.WETH,
                "makerAmount": "2500000000",
                "takerAmount": "1000000000000000000",
                "takerTokenFeeAmount": "0",
            }
        ]

        # Convert dicts to Liquidity objects
        from solver.models.auction import Liquidity

        liq_objects = [Liquidity.model_validate(liq) for liq in liquidity]
        registry = build_registry_from_liquidity(liq_objects)

        assert registry.limit_order_count == 1
        orders = registry.get_limit_orders(self.WETH, self.USDC)
        assert len(orders) == 1
