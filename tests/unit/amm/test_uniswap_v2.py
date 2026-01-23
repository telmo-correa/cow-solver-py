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

        # Output is >= requested due to get_amount_in rounding up
        assert result.amount_out >= 1 * 10**18
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


class TestPartialFillCalculation:
    """Tests for partial fill calculation methods."""

    def test_max_fill_sell_order_full_fill_possible(self):
        """When full fill is possible, return full sell amount."""
        amm = UniswapV2()
        # Pool: 100 ETH / 250k USDC
        reserve_in = 100 * 10**18
        reserve_out = 250_000 * 10**6

        # Order: sell 1 ETH, want at least 2400 USDC
        sell_amount = 1 * 10**18
        buy_amount = 2400 * 10**6

        max_input = amm.max_fill_sell_order(reserve_in, reserve_out, sell_amount, buy_amount)

        # Pool can provide ~2467 USDC for 1 ETH, so full fill is possible
        assert max_input == sell_amount

    def test_max_fill_sell_order_partial_fill(self):
        """When full fill impossible, return maximum partial that satisfies limit."""
        amm = UniswapV2()
        # Small pool: 10 ETH / 25k USDC (same rate as large pool)
        reserve_in = 10 * 10**18
        reserve_out = 25_000 * 10**6

        # Order: sell 5 ETH, want at least 2400 USDC per ETH (12000 USDC total)
        # At 5 ETH input, slippage makes rate < 2400/ETH
        sell_amount = 5 * 10**18
        buy_amount = 12_000 * 10**6

        max_input = amm.max_fill_sell_order(reserve_in, reserve_out, sell_amount, buy_amount)

        # Should return partial fill amount
        assert max_input > 0
        assert max_input < sell_amount

        # Verify the partial fill satisfies the limit rate
        actual_output = amm.get_amount_out(max_input, reserve_in, reserve_out)
        # output/input >= buy_amount/sell_amount
        assert actual_output * sell_amount >= buy_amount * max_input

    def test_max_fill_sell_order_impossible(self):
        """When pool rate is worse than limit, return 0."""
        amm = UniswapV2()
        # Pool: 100 ETH / 200k USDC (rate: 2000 USDC/ETH)
        reserve_in = 100 * 10**18
        reserve_out = 200_000 * 10**6

        # Order: sell 1 ETH, want at least 2500 USDC (limit rate > pool rate)
        sell_amount = 1 * 10**18
        buy_amount = 2500 * 10**6

        max_input = amm.max_fill_sell_order(reserve_in, reserve_out, sell_amount, buy_amount)

        # Pool can only provide ~1970 USDC/ETH after fee, can't meet 2500
        assert max_input == 0

    def test_max_fill_sell_order_edge_cases(self):
        """Edge cases for max fill sell order."""
        amm = UniswapV2()

        # Zero reserves
        assert amm.max_fill_sell_order(0, 100, 100, 50) == 0
        assert amm.max_fill_sell_order(100, 0, 100, 50) == 0

        # Zero buy amount (no limit) returns full sell amount
        assert amm.max_fill_sell_order(100, 100, 100, 0) == 100

        # Zero sell amount
        assert amm.max_fill_sell_order(100, 100, 0, 50) == 0

    def test_max_fill_buy_order_full_fill_possible(self):
        """When full fill is possible, return full buy amount."""
        amm = UniswapV2()
        # Pool: 100 ETH / 250k USDC
        reserve_in = 250_000 * 10**6
        reserve_out = 100 * 10**18

        # Order: buy 1 ETH, willing to pay up to 2600 USDC
        buy_amount = 1 * 10**18
        sell_amount = 2600 * 10**6

        max_output = amm.max_fill_buy_order(reserve_in, reserve_out, sell_amount, buy_amount)

        # Pool requires ~2533 USDC for 1 ETH, which is under 2600 limit
        assert max_output == buy_amount

    def test_max_fill_buy_order_partial_fill(self):
        """When full fill impossible, return maximum partial that satisfies limit."""
        amm = UniswapV2()
        # Small pool: 25k USDC / 10 ETH
        reserve_in = 25_000 * 10**6
        reserve_out = 10 * 10**18

        # Order: buy 5 ETH, willing to pay up to 2600 USDC per ETH (13000 USDC total)
        # At 5 ETH output, slippage makes rate > 2600/ETH
        buy_amount = 5 * 10**18
        sell_amount = 13_000 * 10**6

        max_output = amm.max_fill_buy_order(reserve_in, reserve_out, sell_amount, buy_amount)

        # Should return partial fill amount
        assert max_output > 0
        assert max_output < buy_amount

        # Verify the partial fill satisfies the limit rate
        actual_input = amm.get_amount_in(max_output, reserve_in, reserve_out)
        # input/output <= sell_amount/buy_amount
        assert actual_input * buy_amount <= sell_amount * max_output

    def test_max_fill_buy_order_impossible(self):
        """When pool rate is worse than limit, return 0."""
        amm = UniswapV2()
        # Pool: 200k USDC / 100 ETH (rate: 2000 USDC/ETH)
        reserve_in = 200_000 * 10**6
        reserve_out = 100 * 10**18

        # Order: buy 1 ETH, willing to pay up to 1900 USDC (limit rate < pool rate)
        buy_amount = 1 * 10**18
        sell_amount = 1900 * 10**6

        max_output = amm.max_fill_buy_order(reserve_in, reserve_out, sell_amount, buy_amount)

        # Pool requires ~2030 USDC/ETH after fee, can't meet 1900 limit
        assert max_output == 0

    def test_max_fill_buy_order_edge_cases(self):
        """Edge cases for max fill buy order."""
        amm = UniswapV2()

        # Zero reserves
        assert amm.max_fill_buy_order(0, 100, 100, 50) == 0
        assert amm.max_fill_buy_order(100, 0, 100, 50) == 0

        # Zero sell amount (no budget)
        assert amm.max_fill_buy_order(100, 100, 0, 50) == 0

    def test_partial_fill_consistency_sell(self):
        """Partial fill for sell order should produce valid output."""
        amm = UniswapV2()
        reserve_in = 50 * 10**18
        reserve_out = 125_000 * 10**6

        sell_amount = 10 * 10**18
        buy_amount = 24_000 * 10**6  # 2400 USDC/ETH limit

        max_input = amm.max_fill_sell_order(reserve_in, reserve_out, sell_amount, buy_amount)

        if max_input > 0:
            output = amm.get_amount_out(max_input, reserve_in, reserve_out)
            # The actual rate should be at or above the limit rate
            actual_rate = output / max_input if max_input > 0 else 0
            limit_rate = buy_amount / sell_amount
            assert actual_rate >= limit_rate * 0.9999  # Allow tiny rounding

    def test_partial_fill_consistency_buy(self):
        """Partial fill for buy order should produce valid input."""
        amm = UniswapV2()
        reserve_in = 125_000 * 10**6
        reserve_out = 50 * 10**18

        sell_amount = 26_000 * 10**6  # Max willing to pay
        buy_amount = 10 * 10**18  # Want 10 ETH

        max_output = amm.max_fill_buy_order(reserve_in, reserve_out, sell_amount, buy_amount)

        if max_output > 0:
            input_required = amm.get_amount_in(max_output, reserve_in, reserve_out)
            # The actual rate should be at or below the limit rate
            actual_rate = input_required / max_output if max_output > 0 else float("inf")
            limit_rate = sell_amount / buy_amount
            assert actual_rate <= limit_rate * 1.0001  # Allow tiny rounding


class TestFeeParsingFallback:
    """Tests for fee parsing error handling in parse_liquidity_to_pool."""

    def test_valid_fee_parsing(self):
        """Test parsing a valid fee string."""
        from solver.amm.uniswap_v2 import parse_liquidity_to_pool
        from solver.models.auction import Liquidity

        liquidity = Liquidity(
            id="test-pool",
            kind="constantProduct",
            tokens={
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"balance": "1000000000000000000"},
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {"balance": "2500000000"},
            },
            address="0x1234567890123456789012345678901234567890",
            fee="0.003",  # 0.3%
        )

        pool = parse_liquidity_to_pool(liquidity)
        assert pool is not None
        assert pool.fee_bps == 30  # 0.3% = 30 bps

    def test_custom_fee_parsing(self):
        """Test parsing a custom fee (e.g., 0.05% = 5 bps)."""
        from solver.amm.uniswap_v2 import parse_liquidity_to_pool
        from solver.models.auction import Liquidity

        liquidity = Liquidity(
            id="test-pool",
            kind="constantProduct",
            tokens={
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"balance": "1000000000000000000"},
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {"balance": "2500000000"},
            },
            address="0x1234567890123456789012345678901234567890",
            fee="0.0005",  # 0.05%
        )

        pool = parse_liquidity_to_pool(liquidity)
        assert pool is not None
        assert pool.fee_bps == 5  # 0.05% = 5 bps

    def test_invalid_fee_falls_back_to_default(self):
        """Test that invalid fee string falls back to default 30 bps."""
        from solver.amm.uniswap_v2 import parse_liquidity_to_pool
        from solver.models.auction import Liquidity

        liquidity = Liquidity(
            id="test-pool",
            kind="constantProduct",
            tokens={
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"balance": "1000000000000000000"},
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {"balance": "2500000000"},
            },
            address="0x1234567890123456789012345678901234567890",
            fee="not-a-number",  # Invalid fee string
        )

        pool = parse_liquidity_to_pool(liquidity)
        assert pool is not None
        assert pool.fee_bps == 30  # Falls back to default 30 bps

    def test_no_fee_uses_default(self):
        """Test that missing fee uses default 30 bps."""
        from solver.amm.uniswap_v2 import parse_liquidity_to_pool
        from solver.models.auction import Liquidity

        liquidity = Liquidity(
            id="test-pool",
            kind="constantProduct",
            tokens={
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"balance": "1000000000000000000"},
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {"balance": "2500000000"},
            },
            address="0x1234567890123456789012345678901234567890",
            # No fee field
        )

        pool = parse_liquidity_to_pool(liquidity)
        assert pool is not None
        assert pool.fee_bps == 30  # Default 30 bps


class TestParseNonConstantProduct:
    """Tests for parsing non-constantProduct liquidity."""

    def test_non_constant_product_returns_none(self):
        """Test that non-constantProduct liquidity returns None."""
        from solver.amm.uniswap_v2 import parse_liquidity_to_pool
        from solver.models.auction import Liquidity

        liquidity = Liquidity(
            id="test-pool",
            kind="weightedProduct",  # Not constantProduct
            tokens=[
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            ],
            address="0x1234567890123456789012345678901234567890",
        )

        pool = parse_liquidity_to_pool(liquidity)
        assert pool is None  # Should return None for non-constantProduct

    def test_three_token_pool_returns_none(self):
        """Test that pools with != 2 tokens return None."""
        from solver.amm.uniswap_v2 import parse_liquidity_to_pool
        from solver.models.auction import Liquidity

        liquidity = Liquidity(
            id="test-pool",
            kind="constantProduct",
            tokens={
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"balance": "1000"},
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {"balance": "1000"},
                "0x6B175474E89094C44Da98b954EeDeAC495271d0F": {"balance": "1000"},  # 3 tokens
            },
            address="0x1234567890123456789012345678901234567890",
        )

        pool = parse_liquidity_to_pool(liquidity)
        assert pool is None  # Should return None for non-2-token pools


class TestGasEstimate:
    """Tests for gas estimate handling in V2 pools."""

    def test_pool_default_gas_estimate(self):
        """Test that UniswapV2Pool has default gas_estimate from constant."""
        from solver.constants import POOL_SWAP_GAS_COST

        pool = UniswapV2Pool(
            address="0xPoolAddress",
            token0="0xTokenA",
            token1="0xTokenB",
            reserve0=100 * 10**18,
            reserve1=200 * 10**18,
        )

        # Default gas estimate should be POOL_SWAP_GAS_COST
        assert pool.gas_estimate == POOL_SWAP_GAS_COST
        assert pool.gas_estimate == 60_000

    def test_pool_custom_gas_estimate(self):
        """Test that UniswapV2Pool can have custom gas_estimate."""
        pool = UniswapV2Pool(
            address="0xPoolAddress",
            token0="0xTokenA",
            token1="0xTokenB",
            reserve0=100 * 10**18,
            reserve1=200 * 10**18,
            gas_estimate=110_000,
        )

        assert pool.gas_estimate == 110_000

    def test_simulate_swap_returns_pool_gas_estimate(self):
        """Test that simulate_swap returns pool's gas_estimate, not constant."""
        pool = UniswapV2Pool(
            address="0xPoolAddress",
            token0="0xTokenA",
            token1="0xTokenB",
            reserve0=100 * 10**18,
            reserve1=200 * 10**18,
            gas_estimate=88_892,  # Custom gas estimate from auction
        )

        result = uniswap_v2.simulate_swap(
            pool=pool,
            token_in="0xTokenA",
            amount_in=1 * 10**18,
        )

        # Gas estimate in result should match pool's gas_estimate
        assert result.gas_estimate == 88_892

    def test_simulate_swap_exact_output_returns_pool_gas_estimate(self):
        """Test that simulate_swap_exact_output returns pool's gas_estimate."""
        pool = UniswapV2Pool(
            address="0xPoolAddress",
            token0="0xTokenA",
            token1="0xTokenB",
            reserve0=100 * 10**18,
            reserve1=200 * 10**18,
            gas_estimate=110_000,
        )

        result = uniswap_v2.simulate_swap_exact_output(
            pool=pool,
            token_in="0xTokenA",
            amount_out=1 * 10**18,
        )

        assert result.gas_estimate == 110_000


class TestGasEstimateParsing:
    """Tests for gas estimate parsing in parse_liquidity_to_pool."""

    def test_parse_gas_estimate_from_liquidity(self):
        """Test parsing gasEstimate from liquidity data."""
        from solver.amm.uniswap_v2 import parse_liquidity_to_pool
        from solver.models.auction import Liquidity

        liquidity = Liquidity(
            id="test-pool",
            kind="constantProduct",
            tokens={
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"balance": "1000000000000000000"},
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {"balance": "2500000000"},
            },
            address="0x1234567890123456789012345678901234567890",
            fee="0.003",
            gas_estimate="110000",  # Use alias
        )

        pool = parse_liquidity_to_pool(liquidity)
        assert pool is not None
        assert pool.gas_estimate == 110_000

    def test_parse_gas_estimate_missing_uses_default(self):
        """Test that missing gasEstimate uses default constant."""
        from solver.amm.uniswap_v2 import parse_liquidity_to_pool
        from solver.constants import POOL_SWAP_GAS_COST
        from solver.models.auction import Liquidity

        liquidity = Liquidity(
            id="test-pool",
            kind="constantProduct",
            tokens={
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"balance": "1000000000000000000"},
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {"balance": "2500000000"},
            },
            address="0x1234567890123456789012345678901234567890",
            # No gas_estimate field
        )

        pool = parse_liquidity_to_pool(liquidity)
        assert pool is not None
        assert pool.gas_estimate == POOL_SWAP_GAS_COST

    def test_parse_gas_estimate_invalid_uses_default(self):
        """Test that invalid gasEstimate falls back to default."""
        from solver.amm.uniswap_v2 import parse_liquidity_to_pool
        from solver.constants import POOL_SWAP_GAS_COST
        from solver.models.auction import Liquidity

        liquidity = Liquidity(
            id="test-pool",
            kind="constantProduct",
            tokens={
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"balance": "1000000000000000000"},
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {"balance": "2500000000"},
            },
            address="0x1234567890123456789012345678901234567890",
            gas_estimate="not-a-number",  # Invalid
        )

        pool = parse_liquidity_to_pool(liquidity)
        assert pool is not None
        assert pool.gas_estimate == POOL_SWAP_GAS_COST

    def test_parse_gas_estimate_realistic_value(self):
        """Test parsing realistic gas estimate from Rust solver (88892)."""
        from solver.amm.uniswap_v2 import parse_liquidity_to_pool
        from solver.models.auction import Liquidity

        # This is the value the Rust solver typically provides
        liquidity = Liquidity(
            id="test-pool",
            kind="constantProduct",
            tokens={
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"balance": "1000000000000000000"},
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {"balance": "2500000000"},
            },
            address="0x1234567890123456789012345678901234567890",
            gas_estimate="88892",
        )

        pool = parse_liquidity_to_pool(liquidity)
        assert pool is not None
        assert pool.gas_estimate == 88_892
