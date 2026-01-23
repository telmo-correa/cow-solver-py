"""Tests for Balancer pool parsing and registry.

This module tests:
- parse_weighted_pool function
- parse_stable_pool function
- PoolRegistry Balancer pool support
- build_registry_from_liquidity with Balancer pools
"""

from decimal import Decimal

from solver.amm.balancer import (
    BalancerStablePool,
    BalancerWeightedPool,
    StableTokenReserve,
    WeightedTokenReserve,
    parse_stable_pool,
    parse_weighted_pool,
)
from solver.models.auction import Liquidity


class TestParseWeightedPool:
    """Tests for parse_weighted_pool function."""

    # Test addresses (proper 40-char hex)
    TOKEN_A = "0x" + "a" * 40
    TOKEN_B = "0x" + "b" * 40
    TOKEN_C = "0x" + "c" * 40
    POOL_ADDR = "0x" + "1" * 40
    # 64-char hex for balancerPoolId
    POOL_ID = "0x" + "1" * 64

    def test_parse_weighted_pool_basic(self) -> None:
        """Parse a basic 2-token weighted pool."""
        liq = Liquidity(
            id="pool-1",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "1000000000000000000000"},
                self.TOKEN_B: {"balance": "2000000000000000000000"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1000000000000",
            },
            weights={
                self.TOKEN_A: "0.5",
                self.TOKEN_B: "0.5",
            },
        )

        pool = parse_weighted_pool(liq)

        assert pool is not None
        assert pool.id == "pool-1"  # id is the liquidity id
        assert pool.pool_id == self.POOL_ID  # pool_id is the balancerPoolId
        assert pool.address == self.POOL_ADDR
        assert len(pool.reserves) == 2  # field is 'reserves', not 'tokens'
        assert pool.fee == Decimal("0.003")

        # Check reserves
        token_a = next(r for r in pool.reserves if r.token == self.TOKEN_A)
        token_b = next(r for r in pool.reserves if r.token == self.TOKEN_B)

        assert token_a.balance == 1000000000000000000000
        assert token_a.weight == Decimal("0.5")
        assert token_a.scaling_factor == 1

        assert token_b.balance == 2000000000000000000000
        assert token_b.weight == Decimal("0.5")
        assert token_b.scaling_factor == 1000000000000

    def test_parse_weighted_pool_three_tokens(self) -> None:
        """Parse a 3-token weighted pool."""
        liq = Liquidity(
            id="pool-3tok",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
                self.TOKEN_C: {"balance": "300"},
            },
            address=self.POOL_ADDR,
            fee="0.01",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
                self.TOKEN_C: "1",
            },
            weights={
                self.TOKEN_A: "0.33",
                self.TOKEN_B: "0.33",
                self.TOKEN_C: "0.34",
            },
        )

        pool = parse_weighted_pool(liq)

        assert pool is not None
        assert len(pool.reserves) == 3

    def test_parse_weighted_pool_wrong_kind(self) -> None:
        """Return None for non-weighted pool."""
        liq = Liquidity(
            id="pool-v2",
            kind="constantProduct",
            tokens=[self.TOKEN_A, self.TOKEN_B],
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_missing_weights(self) -> None:
        """Return None if weights are missing."""
        liq = Liquidity(
            id="pool-no-weights",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
            },
            # No weights field
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_default_fee(self) -> None:
        """Use default fee when not provided."""
        liq = Liquidity(
            id="pool-no-fee",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            # No fee field
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
            },
            weights={
                self.TOKEN_A: "0.5",
                self.TOKEN_B: "0.5",
            },
        )

        pool = parse_weighted_pool(liq)

        assert pool is not None
        assert pool.fee == Decimal("0.003")  # Default fee

    def test_parse_weighted_pool_default_scaling_factors(self) -> None:
        """Use default scaling factors when not provided."""
        liq = Liquidity(
            id="pool-no-scaling",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            # No scalingFactors field
            weights={
                self.TOKEN_A: "0.5",
                self.TOKEN_B: "0.5",
            },
        )

        pool = parse_weighted_pool(liq)

        assert pool is not None
        # Default scaling factor is 1 - field is 'reserves' not 'tokens'
        for reserve in pool.reserves:
            assert reserve.scaling_factor == 1

    def test_parse_weighted_pool_zero_balance(self) -> None:
        """Return None if any token has zero balance."""
        liq = Liquidity(
            id="pool-zero-balance",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "0"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.5", self.TOKEN_B: "0.5"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_negative_balance(self) -> None:
        """Return None if any token has negative balance."""
        liq = Liquidity(
            id="pool-negative-balance",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "-100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.5", self.TOKEN_B: "0.5"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_invalid_weight_sum(self) -> None:
        """Return None if weight sum is not approximately 1.0."""
        liq = Liquidity(
            id="pool-bad-weight-sum",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.3", self.TOKEN_B: "0.3"},  # Sum = 0.6, not 1.0
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_zero_weight(self) -> None:
        """Return None if any token has zero weight."""
        liq = Liquidity(
            id="pool-zero-weight",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "100"},
                self.TOKEN_B: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0", self.TOKEN_B: "1.0"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_case_insensitive_lookup(self) -> None:
        """Weights and scalingFactors should be looked up case-insensitively."""
        # Use mixed case in tokens keys
        token_a_mixed = "0x" + "A" * 40  # Uppercase
        token_b_mixed = "0x" + "B" * 40  # Uppercase

        liq = Liquidity(
            id="pool-case-insensitive",
            kind="weightedProduct",
            tokens={
                token_a_mixed: {"balance": "100"},
                token_b_mixed: {"balance": "200"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            # Use lowercase in weights/scalingFactors
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.5", self.TOKEN_B: "0.5"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is not None
        assert len(pool.reserves) == 2

    def test_parse_weighted_pool_reserves_sorted(self) -> None:
        """Reserves should be sorted by token address (lowercase)."""
        # TOKEN_C < TOKEN_A < TOKEN_B when sorted by lowercase
        # 0xcc... < 0xaa... is False, 0xaa... < 0xbb... < 0xcc... alphabetically
        liq = Liquidity(
            id="pool-sorting",
            kind="weightedProduct",
            tokens={
                self.TOKEN_C: {"balance": "300"},  # 0xcc...
                self.TOKEN_A: {"balance": "100"},  # 0xaa...
                self.TOKEN_B: {"balance": "200"},  # 0xbb...
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
                self.TOKEN_C: "1",
            },
            weights={
                self.TOKEN_A: "0.33",
                self.TOKEN_B: "0.33",
                self.TOKEN_C: "0.34",
            },
        )

        pool = parse_weighted_pool(liq)
        assert pool is not None
        # Should be sorted: TOKEN_A < TOKEN_B < TOKEN_C
        assert pool.reserves[0].token == self.TOKEN_A
        assert pool.reserves[1].token == self.TOKEN_B
        assert pool.reserves[2].token == self.TOKEN_C

    def test_parse_weighted_pool_invalid_token_data_type(self) -> None:
        """Return None if token_data is not a dict (e.g., tokens is a list)."""
        # This simulates the simplified format where tokens is a list of addresses
        liq = Liquidity(
            id="weighted-invalid-token-data",
            kind="weightedProduct",
            tokens=[self.TOKEN_A, self.TOKEN_B],  # List instead of dict
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.5", self.TOKEN_B: "0.5"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is None

    def test_parse_weighted_pool_single_token(self) -> None:
        """Return None if pool has only one valid token."""
        liq = Liquidity(
            id="weighted-single-token",
            kind="weightedProduct",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                # TOKEN_B has zero balance - will be filtered out
                self.TOKEN_B: {"balance": "0"},
            },
            address=self.POOL_ADDR,
            fee="0.003",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            weights={self.TOKEN_A: "0.5", self.TOKEN_B: "0.5"},
        )

        pool = parse_weighted_pool(liq)
        assert pool is None  # Insufficient tokens (need at least 2)


class TestParseStablePool:
    """Tests for parse_stable_pool function."""

    # Test addresses (proper 40-char hex)
    TOKEN_A = "0x" + "a" * 40
    TOKEN_B = "0x" + "b" * 40
    TOKEN_C = "0x" + "c" * 40
    POOL_ADDR = "0x" + "2" * 40
    DAI_ADDR = "0x" + "d" * 40
    USDC_ADDR = "0x" + "e" * 40
    USDT_ADDR = "0x" + "f" * 40
    # 64-char hex for balancerPoolId
    POOL_ID = "0x" + "2" * 64

    def test_parse_stable_pool_basic(self) -> None:
        """Parse a basic 2-token stable pool."""
        liq = Liquidity(
            id="stable-1",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000000000000000000000"},
                self.TOKEN_B: {"balance": "1000000000000000000000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1000000000000",
            },
            amplificationParameter="200",
        )

        pool = parse_stable_pool(liq)

        assert pool is not None
        assert pool.pool_id == self.POOL_ID  # pool_id is balancerPoolId, not id
        assert pool.address == self.POOL_ADDR
        assert len(pool.reserves) == 2  # Field is 'reserves' not 'tokens'
        assert pool.fee == Decimal("0.0004")
        assert pool.amplification_parameter == 200

    def test_parse_stable_pool_three_tokens(self) -> None:
        """Parse a 3-token stable pool."""
        liq = Liquidity(
            id="stable-3",
            kind="stable",
            tokens={
                self.DAI_ADDR: {"balance": "1000000000000000000000000"},
                self.USDC_ADDR: {"balance": "1000000000000"},
                self.USDT_ADDR: {"balance": "1000000000000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.DAI_ADDR: "1",
                self.USDC_ADDR: "1000000000000",
                self.USDT_ADDR: "1000000000000",
            },
            amplificationParameter="2000",
        )

        pool = parse_stable_pool(liq)

        assert pool is not None
        assert len(pool.reserves) == 3
        assert pool.amplification_parameter == 2000

    def test_parse_stable_pool_filters_bpt(self) -> None:
        """BPT token should be filtered out (composable stable pools)."""
        pool_address = self.POOL_ADDR
        liq = Liquidity(
            id="composable-stable",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                self.TOKEN_B: {"balance": "2000"},
                pool_address: {"balance": "1000000"},  # BPT token
            },
            address=pool_address,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
                pool_address: "1",
            },
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)

        assert pool is not None
        # BPT should be filtered out
        assert len(pool.reserves) == 2
        assert all(r.token != pool_address for r in pool.reserves)

    def test_parse_stable_pool_wrong_kind(self) -> None:
        """Return None for non-stable pool."""
        liq = Liquidity(
            id="pool-v2",
            kind="constantProduct",
            tokens=[self.TOKEN_A, self.TOKEN_B],
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_missing_amp(self) -> None:
        """Return None if amplification parameter is missing."""
        liq = Liquidity(
            id="stable-no-amp",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
            },
            # No amplificationParameter field
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_default_fee(self) -> None:
        """Use default fee when not provided."""
        liq = Liquidity(
            id="stable-no-fee",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            # No fee field
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
            },
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)

        assert pool is not None
        assert pool.fee == Decimal("0.0001")  # Default stable fee is 0.01%

    def test_parse_stable_pool_zero_balance(self) -> None:
        """Return None if any token has zero balance."""
        liq = Liquidity(
            id="stable-zero-balance",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "0"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_negative_balance(self) -> None:
        """Return None if any token has negative balance."""
        liq = Liquidity(
            id="stable-negative-balance",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "-1000"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_zero_amp(self) -> None:
        """Return None if amplification parameter is zero."""
        liq = Liquidity(
            id="stable-zero-amp",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="0",
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_negative_amp(self) -> None:
        """Return None if amplification parameter is negative."""
        liq = Liquidity(
            id="stable-negative-amp",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                self.TOKEN_B: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="-100",
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_case_insensitive_lookup(self) -> None:
        """scalingFactors should be looked up case-insensitively."""
        # Use mixed case in tokens keys
        token_a_mixed = "0x" + "A" * 40  # Uppercase
        token_b_mixed = "0x" + "B" * 40  # Uppercase

        liq = Liquidity(
            id="stable-case-insensitive",
            kind="stable",
            tokens={
                token_a_mixed: {"balance": "1000"},
                token_b_mixed: {"balance": "2000"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            # Use lowercase in scalingFactors
            scalingFactors={self.TOKEN_A: "1000000000000", self.TOKEN_B: "1"},
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is not None
        assert len(pool.reserves) == 2

    def test_parse_stable_pool_reserves_sorted(self) -> None:
        """Reserves should be sorted by token address (lowercase)."""
        liq = Liquidity(
            id="stable-sorting",
            kind="stable",
            tokens={
                self.TOKEN_C: {"balance": "3000"},  # 0xcc...
                self.TOKEN_A: {"balance": "1000"},  # 0xaa...
                self.TOKEN_B: {"balance": "2000"},  # 0xbb...
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={
                self.TOKEN_A: "1",
                self.TOKEN_B: "1",
                self.TOKEN_C: "1",
            },
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is not None
        # Should be sorted: TOKEN_A < TOKEN_B < TOKEN_C
        assert pool.reserves[0].token == self.TOKEN_A
        assert pool.reserves[1].token == self.TOKEN_B
        assert pool.reserves[2].token == self.TOKEN_C

    def test_parse_stable_pool_invalid_token_data_type(self) -> None:
        """Return None if token_data is not a dict (e.g., tokens is a list)."""
        liq = Liquidity(
            id="stable-invalid-token-data",
            kind="stable",
            tokens=[self.TOKEN_A, self.TOKEN_B],  # List instead of dict
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is None

    def test_parse_stable_pool_single_token(self) -> None:
        """Return None if pool has only one valid token after filtering."""
        liq = Liquidity(
            id="stable-single-token",
            kind="stable",
            tokens={
                self.TOKEN_A: {"balance": "1000"},
                # TOKEN_B has zero balance - will be filtered out
                self.TOKEN_B: {"balance": "0"},
            },
            address=self.POOL_ADDR,
            fee="0.0004",
            balancerPoolId=self.POOL_ID,
            scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
            amplificationParameter="100",
        )

        pool = parse_stable_pool(liq)
        assert pool is None  # Insufficient tokens (need at least 2)


class TestPoolRegistryBalancer:
    """Tests for PoolRegistry Balancer pool support."""

    # Test addresses (proper 40-char hex)
    TOKEN_A = "0x" + "a" * 40
    TOKEN_B = "0x" + "b" * 40
    TOKEN_C = "0x" + "c" * 40
    TOKEN_D = "0x" + "d" * 40
    POOL_1 = "0x" + "1" * 40
    POOL_2 = "0x" + "2" * 40
    POOL_3 = "0x" + "3" * 40
    # 64-char hex for balancerPoolId
    POOL_ID_1 = "0x" + "1" * 64
    POOL_ID_2 = "0x" + "2" * 64

    def test_add_and_get_weighted_pool(self) -> None:
        """Add and retrieve weighted pool."""
        from solver.pools import PoolRegistry

        registry = PoolRegistry()
        pool = BalancerWeightedPool(
            id="weighted-1",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A,
                    balance=1000,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B,
                    balance=2000,
                    weight=Decimal("0.5"),
                    scaling_factor=1,
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=88892,
        )

        registry.add_weighted_pool(pool)

        # Should find by token pair
        pools = registry.get_weighted_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1
        assert pools[0].id == "weighted-1"

        # Also find with reversed order
        pools = registry.get_weighted_pools(self.TOKEN_B, self.TOKEN_A)
        assert len(pools) == 1

    def test_add_and_get_stable_pool(self) -> None:
        """Add and retrieve stable pool."""
        from solver.pools import PoolRegistry

        registry = PoolRegistry()
        pool = BalancerStablePool(
            id="stable-1",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                StableTokenReserve(
                    token=self.TOKEN_A,
                    balance=1000000000000000000000,
                    scaling_factor=1,
                ),
                StableTokenReserve(
                    token=self.TOKEN_B,
                    balance=1000000000,
                    scaling_factor=1000000000000,
                ),
            ),
            fee=Decimal("0.0004"),
            amplification_parameter=Decimal("200"),
            gas_estimate=183520,
        )

        registry.add_stable_pool(pool)

        # Should find by token pair
        pools = registry.get_stable_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1
        assert pools[0].id == "stable-1"

    def test_weighted_pool_multi_token_indexing(self) -> None:
        """3-token weighted pool indexed by all pairs."""
        from solver.pools import PoolRegistry

        registry = PoolRegistry()
        pool = BalancerWeightedPool(
            id="weighted-3tok",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A, balance=100, weight=Decimal("0.33"), scaling_factor=1
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B, balance=200, weight=Decimal("0.33"), scaling_factor=1
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_C, balance=300, weight=Decimal("0.34"), scaling_factor=1
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=88892,
        )

        registry.add_weighted_pool(pool)

        # Should find by all 3 token pairs
        assert len(registry.get_weighted_pools(self.TOKEN_A, self.TOKEN_B)) == 1
        assert len(registry.get_weighted_pools(self.TOKEN_A, self.TOKEN_C)) == 1
        assert len(registry.get_weighted_pools(self.TOKEN_B, self.TOKEN_C)) == 1

    def test_get_pools_for_pair_includes_balancer(self) -> None:
        """get_pools_for_pair returns all pool types."""
        from solver.pools import PoolRegistry, UniswapV2Pool

        registry = PoolRegistry()

        # Add a V2 pool
        v2_pool = UniswapV2Pool(
            address=self.POOL_1,
            token0=self.TOKEN_A,
            token1=self.TOKEN_B,
            reserve0=1000,
            reserve1=2000,
            fee_bps=30,  # 0.3%
        )
        registry.add_pool(v2_pool)

        # Add a weighted pool
        weighted_pool = BalancerWeightedPool(
            id="weighted-1",
            address=self.POOL_2,
            pool_id=self.POOL_ID_1,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A, balance=500, weight=Decimal("0.5"), scaling_factor=1
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B, balance=1000, weight=Decimal("0.5"), scaling_factor=1
                ),
            ),
            fee=Decimal("0.002"),
            version="v0",
            gas_estimate=88892,
        )
        registry.add_weighted_pool(weighted_pool)

        # get_pools_for_pair should return both
        pools = registry.get_pools_for_pair(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 2

    def test_pool_counts(self) -> None:
        """Pool count properties work correctly."""
        from solver.pools import PoolRegistry

        registry = PoolRegistry()

        weighted = BalancerWeightedPool(
            id="w1",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A, balance=100, weight=Decimal("0.5"), scaling_factor=1
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B, balance=100, weight=Decimal("0.5"), scaling_factor=1
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=88892,
        )
        stable = BalancerStablePool(
            id="s1",
            address=self.POOL_2,
            pool_id=self.POOL_ID_2,
            reserves=(
                StableTokenReserve(token=self.TOKEN_C, balance=100, scaling_factor=1),
                StableTokenReserve(token=self.TOKEN_D, balance=100, scaling_factor=1),
            ),
            fee=Decimal("0.0004"),
            amplification_parameter=Decimal("100"),
            gas_estimate=183520,
        )

        assert registry.weighted_pool_count == 0
        assert registry.stable_pool_count == 0

        registry.add_weighted_pool(weighted)
        assert registry.weighted_pool_count == 1

        registry.add_stable_pool(stable)
        assert registry.stable_pool_count == 1

    def test_duplicate_weighted_pool_ignored(self) -> None:
        """Adding the same weighted pool twice is a no-op."""
        from solver.pools import PoolRegistry

        registry = PoolRegistry()

        weighted = BalancerWeightedPool(
            id="w1",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                WeightedTokenReserve(
                    token=self.TOKEN_A, balance=100, weight=Decimal("0.5"), scaling_factor=1
                ),
                WeightedTokenReserve(
                    token=self.TOKEN_B, balance=100, weight=Decimal("0.5"), scaling_factor=1
                ),
            ),
            fee=Decimal("0.003"),
            version="v0",
            gas_estimate=88892,
        )

        registry.add_weighted_pool(weighted)
        assert registry.weighted_pool_count == 1

        # Adding same pool again should be ignored
        registry.add_weighted_pool(weighted)
        assert registry.weighted_pool_count == 1

        # get_weighted_pools should return only one
        pools = registry.get_weighted_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1

    def test_duplicate_stable_pool_ignored(self) -> None:
        """Adding the same stable pool twice is a no-op."""
        from solver.pools import PoolRegistry

        registry = PoolRegistry()

        stable = BalancerStablePool(
            id="s1",
            address=self.POOL_1,
            pool_id=self.POOL_ID_1,
            reserves=(
                StableTokenReserve(token=self.TOKEN_A, balance=100, scaling_factor=1),
                StableTokenReserve(token=self.TOKEN_B, balance=100, scaling_factor=1),
            ),
            fee=Decimal("0.0004"),
            amplification_parameter=Decimal("100"),
            gas_estimate=183520,
        )

        registry.add_stable_pool(stable)
        assert registry.stable_pool_count == 1

        # Adding same pool again should be ignored
        registry.add_stable_pool(stable)
        assert registry.stable_pool_count == 1

        # get_stable_pools should return only one
        pools = registry.get_stable_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1


class TestBuildRegistryWithBalancer:
    """Tests for build_registry_from_liquidity with Balancer pools."""

    # Test addresses (proper 40-char hex)
    TOKEN_A = "0x" + "a" * 40
    TOKEN_B = "0x" + "b" * 40
    TOKEN_C = "0x" + "c" * 40
    TOKEN_D = "0x" + "d" * 40
    TOKEN_E = "0x" + "e" * 40
    TOKEN_F = "0x" + "f" * 40
    POOL_1 = "0x" + "1" * 40
    POOL_2 = "0x" + "2" * 40
    POOL_3 = "0x" + "3" * 40
    # 64-char hex for balancerPoolId
    POOL_ID_1 = "0x" + "1" * 64
    POOL_ID_2 = "0x" + "2" * 64
    POOL_ID_3 = "0x" + "3" * 64

    def test_parse_weighted_pool_from_liquidity(self) -> None:
        """build_registry_from_liquidity parses weighted pools."""
        from solver.pools import build_registry_from_liquidity

        liquidity_list = [
            Liquidity(
                id="weighted-pool",
                kind="weightedProduct",
                tokens={
                    self.TOKEN_A: {"balance": "10000000000000000000"},
                    self.TOKEN_B: {"balance": "100000000000000000000"},
                },
                address=self.POOL_1,
                fee="0.003",
                balancerPoolId=self.POOL_ID_1,
                scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1"},
                weights={self.TOKEN_A: "0.8", self.TOKEN_B: "0.2"},
            )
        ]

        registry = build_registry_from_liquidity(liquidity_list)

        assert registry.weighted_pool_count == 1
        pools = registry.get_weighted_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1
        assert pools[0].id == "weighted-pool"  # id is the liquidity id

    def test_parse_stable_pool_from_liquidity(self) -> None:
        """build_registry_from_liquidity parses stable pools."""
        from solver.pools import build_registry_from_liquidity

        liquidity_list = [
            Liquidity(
                id="stable-pool",
                kind="stable",
                tokens={
                    self.TOKEN_A: {"balance": "1000000000000000000000000"},
                    self.TOKEN_B: {"balance": "1000000000000"},
                },
                address=self.POOL_1,
                fee="0.0004",
                balancerPoolId=self.POOL_ID_1,
                scalingFactors={self.TOKEN_A: "1", self.TOKEN_B: "1000000000000"},
                amplificationParameter="200",
            )
        ]

        registry = build_registry_from_liquidity(liquidity_list)

        assert registry.stable_pool_count == 1
        pools = registry.get_stable_pools(self.TOKEN_A, self.TOKEN_B)
        assert len(pools) == 1
        assert pools[0].id == "stable-pool"  # id is the liquidity id

    def test_mixed_pool_types(self) -> None:
        """build_registry_from_liquidity handles mixed pool types."""
        from solver.pools import build_registry_from_liquidity

        liquidity_list = [
            # V2 pool
            Liquidity(
                id="v2-pool",
                kind="constantProduct",
                tokens={
                    self.TOKEN_A: {"balance": "1000"},
                    self.TOKEN_B: {"balance": "2000"},
                },
                address=self.POOL_1,
                fee="0.003",
            ),
            # Weighted pool
            Liquidity(
                id="weighted-pool",
                kind="weightedProduct",
                tokens={
                    self.TOKEN_C: {"balance": "500"},
                    self.TOKEN_D: {"balance": "1500"},
                },
                address=self.POOL_2,
                fee="0.002",
                balancerPoolId=self.POOL_ID_2,
                scalingFactors={self.TOKEN_C: "1", self.TOKEN_D: "1"},
                weights={self.TOKEN_C: "0.5", self.TOKEN_D: "0.5"},
            ),
            # Stable pool
            Liquidity(
                id="stable-pool",
                kind="stable",
                tokens={
                    self.TOKEN_E: {"balance": "1000"},
                    self.TOKEN_F: {"balance": "1000"},
                },
                address=self.POOL_3,
                fee="0.0004",
                balancerPoolId=self.POOL_ID_3,
                scalingFactors={self.TOKEN_E: "1", self.TOKEN_F: "1"},
                amplificationParameter="100",
            ),
        ]

        registry = build_registry_from_liquidity(liquidity_list)

        assert registry.pool_count == 1  # V2
        assert registry.weighted_pool_count == 1
        assert registry.stable_pool_count == 1
