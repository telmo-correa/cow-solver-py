"""Tests for AMM implementations."""

import pytest

from solver.amm.uniswap_v2 import UniswapV2, UniswapV2Pool, uniswap_v2


class TestUniswapV2Math:
    """Tests for UniswapV2 constant product math."""

    def test_get_amount_out_basic(self):
        """Basic swap calculation."""
        # 1 ETH in, with reserves of 100 ETH and 250,000 USDC
        # Expected: roughly 2475 USDC (accounting for 0.3% fee)
        amm = UniswapV2()
        amount_in = 1 * 10**18  # 1 ETH
        reserve_in = 100 * 10**18  # 100 ETH
        reserve_out = 250_000 * 10**6  # 250K USDC

        amount_out = amm.get_amount_out(amount_in, reserve_in, reserve_out)

        # With fee: (1 * 997 * 250000) / (100 * 1000 + 1 * 997) = 2467.58...
        # In 6 decimals: ~2467580000
        assert amount_out > 0
        assert amount_out < reserve_out  # Can't get more than reserve
        # Approximate check (within 1% of expected)
        expected = 2467 * 10**6
        assert abs(amount_out - expected) < expected * 0.01

    def test_get_amount_out_zero_input(self):
        """Zero input returns zero output."""
        amm = UniswapV2()
        assert amm.get_amount_out(0, 100, 100) == 0

    def test_get_amount_out_zero_reserves(self):
        """Zero reserves returns zero."""
        amm = UniswapV2()
        assert amm.get_amount_out(100, 0, 100) == 0
        assert amm.get_amount_out(100, 100, 0) == 0

    def test_get_amount_in_basic(self):
        """Calculate input needed for desired output."""
        amm = UniswapV2()
        amount_out = 2467 * 10**6  # Want ~2467 USDC
        reserve_in = 100 * 10**18
        reserve_out = 250_000 * 10**6

        amount_in = amm.get_amount_in(amount_out, reserve_in, reserve_out)

        # Should need roughly 1 ETH
        expected = 1 * 10**18
        assert abs(amount_in - expected) < expected * 0.01

    def test_get_amount_in_exceeds_reserve(self):
        """Requesting more than reserve returns max uint."""
        amm = UniswapV2()
        amount_out = 300_000 * 10**6  # More than reserve
        reserve_in = 100 * 10**18
        reserve_out = 250_000 * 10**6

        amount_in = amm.get_amount_in(amount_out, reserve_in, reserve_out)
        assert amount_in == 2**256 - 1

    def test_round_trip_consistency(self):
        """get_amount_out followed by get_amount_in should recover input."""
        amm = UniswapV2()
        reserve_in = 100 * 10**18
        reserve_out = 250_000 * 10**6

        # Start with desired output, calculate required input
        desired_output = 2467 * 10**6
        required_input = amm.get_amount_in(desired_output, reserve_in, reserve_out)

        # Now verify: if we input that amount, do we get at least desired output?
        actual_output = amm.get_amount_out(required_input, reserve_in, reserve_out)

        # get_amount_in rounds up, so actual output should be >= desired
        assert actual_output >= desired_output
        # But not by too much (within 0.01%)
        assert actual_output < desired_output * 1.0001


class TestUniswapV2Pool:
    """Tests for UniswapV2Pool dataclass."""

    def test_get_reserves_token0_in(self):
        """Get reserves when swapping token0 for token1."""
        pool = UniswapV2Pool(
            address="0x1234",
            token0="0xAAAA",
            token1="0xBBBB",
            reserve0=100,
            reserve1=200,
        )
        res_in, res_out = pool.get_reserves("0xAAAA")
        assert res_in == 100
        assert res_out == 200

    def test_get_reserves_token1_in(self):
        """Get reserves when swapping token1 for token0."""
        pool = UniswapV2Pool(
            address="0x1234",
            token0="0xAAAA",
            token1="0xBBBB",
            reserve0=100,
            reserve1=200,
        )
        res_in, res_out = pool.get_reserves("0xBBBB")
        assert res_in == 200
        assert res_out == 100

    def test_get_reserves_case_insensitive(self):
        """Token matching should be case insensitive."""
        pool = UniswapV2Pool(
            address="0x1234",
            token0="0xAAAA",
            token1="0xBBBB",
            reserve0=100,
            reserve1=200,
        )
        res_in, res_out = pool.get_reserves("0xaaaa")
        assert res_in == 100

    def test_get_token_out(self):
        """Get output token."""
        pool = UniswapV2Pool(
            address="0x1234",
            token0="0xAAAA",
            token1="0xBBBB",
            reserve0=100,
            reserve1=200,
        )
        assert pool.get_token_out("0xAAAA") == "0xBBBB"
        assert pool.get_token_out("0xBBBB") == "0xAAAA"


class TestSwapSimulation:
    """Tests for full swap simulation."""

    def test_simulate_swap(self):
        """Simulate a swap through a pool."""
        pool = UniswapV2Pool(
            address="0xPoolAddress",
            token0="0xTokenA",
            token1="0xTokenB",
            reserve0=100 * 10**18,
            reserve1=200 * 10**18,
        )

        result = uniswap_v2.simulate_swap(
            pool=pool,
            token_in="0xTokenA",
            amount_in=1 * 10**18,
        )

        assert result.amount_in == 1 * 10**18
        assert result.amount_out > 0
        assert result.pool_address == "0xPoolAddress"
        assert result.token_in == "0xTokenA"
        assert result.token_out == "0xTokenB"

    def test_simulate_swap_exact_output(self):
        """Simulate a swap for exact output amount."""
        pool = UniswapV2Pool(
            address="0xPoolAddress",
            token0="0xTokenA",
            token1="0xTokenB",
            reserve0=100 * 10**18,
            reserve1=200 * 10**18,
        )

        result = uniswap_v2.simulate_swap_exact_output(
            pool=pool,
            token_in="0xTokenA",
            amount_out=1 * 10**18,
        )

        assert result.amount_out == 1 * 10**18
        assert result.amount_in > 0
        assert result.pool_address == "0xPoolAddress"
        assert result.token_in == "0xTokenA"
        assert result.token_out == "0xTokenB"

    def test_simulate_swap_exact_output_consistency(self):
        """Exact output simulation should be consistent with exact input."""
        pool = UniswapV2Pool(
            address="0xPoolAddress",
            token0="0xTokenA",
            token1="0xTokenB",
            reserve0=100 * 10**18,
            reserve1=200 * 10**18,
        )

        # Get required input for 1 token output
        exact_out_result = uniswap_v2.simulate_swap_exact_output(
            pool=pool,
            token_in="0xTokenA",
            amount_out=1 * 10**18,
        )

        # Use that input amount and verify output
        exact_in_result = uniswap_v2.simulate_swap(
            pool=pool,
            token_in="0xTokenA",
            amount_in=exact_out_result.amount_in,
        )

        # Should get at least the desired output (get_amount_in rounds up)
        assert exact_in_result.amount_out >= exact_out_result.amount_out


class TestPoolRegistry:
    """Tests for PoolRegistry."""

    def test_get_weth_usdc_pool(self, test_pool_registry):
        """Can find WETH/USDC pool in registry."""
        # Addresses can be any case - get_pool normalizes them
        weth = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        usdc = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"

        pool = test_pool_registry.get_pool(weth, usdc)
        assert pool is not None
        # Pool addresses are stored lowercase for consistency
        assert pool.address == "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc"

    def test_get_pool_order_independent(self, test_pool_registry):
        """Pool lookup works regardless of token order."""
        weth = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        usdc = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"

        pool1 = test_pool_registry.get_pool(weth, usdc)
        pool2 = test_pool_registry.get_pool(usdc, weth)
        assert pool1 == pool2

    def test_get_nonexistent_pool(self, test_pool_registry):
        """Returns None for unknown token pairs."""
        pool = test_pool_registry.get_pool("0xUnknown1", "0xUnknown2")
        assert pool is None

    def test_find_path_direct(self, test_pool_registry):
        """Find direct path when pool exists."""
        weth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
        usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

        path = test_pool_registry.find_path(weth, usdc)
        assert path is not None
        assert len(path) == 2  # Direct path: [WETH, USDC]

    def test_find_path_multihop(self, test_pool_registry):
        """Find multi-hop path through intermediate token."""
        usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        dai = "0x6b175474e89094c44da98b954eedeac495271d0f"

        path = test_pool_registry.find_path(usdc, dai)
        assert path is not None
        assert len(path) == 3  # Multi-hop: [USDC, WETH, DAI]


class TestSwapEncoding:
    """Tests for swap calldata encoding."""

    def test_encode_swap_returns_valid_calldata(self):
        """Encoded swap has correct format."""
        amm = UniswapV2()
        target, calldata = amm.encode_swap(
            token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            amount_in=1 * 10**18,
            amount_out_min=2400 * 10**6,
            recipient="0x9008D19f58AAbD9eD0D60971565AA8510560ab41",  # CoW settlement
        )

        assert target == UniswapV2.ROUTER_ADDRESS
        assert calldata.startswith("0x38ed1739")  # swapExactTokensForTokens selector
        assert len(calldata) > 10  # Has encoded args

    def test_encode_swap_encodes_correct_values(self):
        """Verify the encoded calldata contains correct values."""
        from eth_abi import decode

        amm = UniswapV2()
        weth = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        usdc = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
        recipient = "0x9008D19f58AAbD9eD0D60971565AA8510560ab41"
        amount_in = 1 * 10**18
        amount_out_min = 2400 * 10**6

        _, calldata = amm.encode_swap(
            token_in=weth,
            token_out=usdc,
            amount_in=amount_in,
            amount_out_min=amount_out_min,
            recipient=recipient,
        )

        # Remove selector (first 4 bytes = 8 hex chars + "0x")
        encoded_args = bytes.fromhex(calldata[10:])

        # Decode the arguments
        # swapExactTokensForTokens(uint256,uint256,address[],address,uint256)
        decoded = decode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            encoded_args,
        )

        decoded_amount_in, decoded_amount_out_min, path, decoded_recipient, deadline = decoded

        assert decoded_amount_in == amount_in
        assert decoded_amount_out_min == amount_out_min
        assert len(path) == 2
        assert path[0].lower() == weth.lower()
        assert path[1].lower() == usdc.lower()
        assert decoded_recipient.lower() == recipient.lower()
        assert deadline == 2**32 - 1  # Max uint32 as deadline

    def test_encode_swap_validates_addresses(self):
        """encode_swap should reject invalid addresses."""
        amm = UniswapV2()

        # Invalid token_in (validated as part of path)
        with pytest.raises(ValueError, match="Invalid address"):
            amm.encode_swap(
                token_in="not-an-address",
                token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                amount_in=1000,
                amount_out_min=900,
                recipient="0x9008D19f58AAbD9eD0D60971565AA8510560ab41",
            )

        # Invalid token_out (validated as part of path)
        with pytest.raises(ValueError, match="Invalid address"):
            amm.encode_swap(
                token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                token_out="0x123",  # Too short
                amount_in=1000,
                amount_out_min=900,
                recipient="0x9008D19f58AAbD9eD0D60971565AA8510560ab41",
            )

        # Invalid recipient
        with pytest.raises(ValueError, match="Invalid recipient"):
            amm.encode_swap(
                token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                amount_in=1000,
                amount_out_min=900,
                recipient="invalid",
            )


class TestSwapExactOutputEncoding:
    """Tests for exact output swap calldata encoding (buy orders)."""

    def test_encode_swap_exact_output_returns_valid_calldata(self):
        """Encoded exact output swap has correct format."""
        amm = UniswapV2()
        target, calldata = amm.encode_swap_exact_output(
            token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            amount_out=2400 * 10**6,
            amount_in_max=1 * 10**18,
            recipient="0x9008D19f58AAbD9eD0D60971565AA8510560ab41",
        )

        assert target == UniswapV2.ROUTER_ADDRESS
        assert calldata.startswith("0x8803dbee")  # swapTokensForExactTokens selector
        assert len(calldata) > 10

    def test_encode_swap_exact_output_encodes_correct_values(self):
        """Verify the encoded calldata contains correct values."""
        from eth_abi import decode

        amm = UniswapV2()
        weth = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        usdc = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
        recipient = "0x9008D19f58AAbD9eD0D60971565AA8510560ab41"
        amount_out = 2400 * 10**6
        amount_in_max = 1 * 10**18

        _, calldata = amm.encode_swap_exact_output(
            token_in=weth,
            token_out=usdc,
            amount_out=amount_out,
            amount_in_max=amount_in_max,
            recipient=recipient,
        )

        # Remove selector (first 4 bytes = 8 hex chars + "0x")
        encoded_args = bytes.fromhex(calldata[10:])

        # Decode the arguments
        # swapTokensForExactTokens(uint256,uint256,address[],address,uint256)
        decoded = decode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            encoded_args,
        )

        decoded_amount_out, decoded_amount_in_max, path, decoded_recipient, deadline = decoded

        assert decoded_amount_out == amount_out
        assert decoded_amount_in_max == amount_in_max
        assert len(path) == 2
        assert path[0].lower() == weth.lower()
        assert path[1].lower() == usdc.lower()
        assert decoded_recipient.lower() == recipient.lower()
        assert deadline == 2**32 - 1

    def test_encode_swap_exact_output_validates_addresses(self):
        """encode_swap_exact_output should reject invalid addresses."""
        amm = UniswapV2()

        # Invalid token_in (validated as part of path)
        with pytest.raises(ValueError, match="Invalid address"):
            amm.encode_swap_exact_output(
                token_in="invalid",
                token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                amount_out=1000,
                amount_in_max=2000,
                recipient="0x9008D19f58AAbD9eD0D60971565AA8510560ab41",
            )


class TestSwapDirectEncoding:
    """Tests for direct pool swap encoding (encode_swap_direct)."""

    def test_encode_swap_direct_token0_to_token1(self):
        """Direct swap from token0 to token1 sets amount1_out."""
        amm = UniswapV2()
        # USDC < WETH in bytes comparison, so USDC is token0
        usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        weth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
        pool = "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc"
        recipient = "0x9008d19f58aabd9ed0d60971565aa8510560ab41"

        target, calldata = amm.encode_swap_direct(
            pool_address=pool,
            token_in=usdc,  # token0
            token_out=weth,  # token1
            _amount_in=2500 * 10**6,
            amount_out=1 * 10**18,
            recipient=recipient,
        )

        assert target == pool
        assert calldata.startswith("0x022c0d9f")  # swap selector

    def test_encode_swap_direct_token1_to_token0(self):
        """Direct swap from token1 to token0 sets amount0_out."""
        amm = UniswapV2()
        usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        weth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
        pool = "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc"
        recipient = "0x9008d19f58aabd9ed0d60971565aa8510560ab41"

        target, calldata = amm.encode_swap_direct(
            pool_address=pool,
            token_in=weth,  # token1
            token_out=usdc,  # token0
            _amount_in=1 * 10**18,
            amount_out=2500 * 10**6,
            recipient=recipient,
        )

        assert target == pool
        assert calldata.startswith("0x022c0d9f")  # swap selector

    def test_encode_swap_direct_validates_addresses(self):
        """encode_swap_direct should reject invalid addresses."""
        amm = UniswapV2()

        # Invalid pool address
        with pytest.raises(ValueError, match="Invalid pool_address"):
            amm.encode_swap_direct(
                pool_address="not-valid",
                token_in="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                token_out="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
                _amount_in=1000,
                amount_out=900,
                recipient="0x9008d19f58aabd9ed0d60971565aa8510560ab41",
            )
