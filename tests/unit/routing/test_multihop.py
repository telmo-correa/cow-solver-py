"""Unit tests for multi-hop routing (A→B→C)."""

from solver.models.auction import OrderKind
from tests.conftest import make_order


class TestMultiHopRouting:
    """Tests for multi-hop routing (A→B→C)."""

    def test_multihop_sell_order_usdc_to_dai(self, router):
        """Multi-hop sell order: USDC → WETH → DAI (no direct pool)."""
        # USDC and DAI have no direct pool, must go via WETH
        from solver.constants import DAI, USDC

        order = make_order(
            sell_token=USDC,
            buy_token=DAI,
            sell_amount="1000000000",  # 1000 USDC (6 decimals)
            buy_amount="900000000000000000000",  # 900 DAI min (18 decimals)
            kind=OrderKind.SELL,
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.is_multihop is True
        assert result.path is not None
        assert len(result.path) == 3  # USDC → WETH → DAI
        assert result.pools is not None
        assert len(result.pools) == 2
        # 2 swaps * 60k gas per swap = 120k (before settlement overhead)
        assert result.gas_estimate == 120_000

    def test_multihop_buy_order_usdc_to_dai(self, router):
        """Multi-hop buy order: USDC → WETH → DAI."""
        from solver.constants import DAI, USDC

        order = make_order(
            sell_token=USDC,
            buy_token=DAI,
            sell_amount="2000000000",  # 2000 USDC max
            buy_amount="1000000000000000000000",  # Want exactly 1000 DAI
            kind=OrderKind.BUY,
        )
        result = router.route_order(order)

        assert result.success is True
        assert result.is_multihop is True
        assert result.amount_out == 1000000000000000000000  # Exact output

    def test_multihop_direct_pool_preferred(self, router):
        """Direct pool is preferred over multi-hop when available."""
        # WETH/USDC has a direct pool
        order = make_order(
            sell_amount="1000000000000000000",  # 1 WETH
            buy_amount="2000000000",  # 2000 USDC min
            kind=OrderKind.SELL,
        )
        result = router.route_order(order)

        assert result.success is True
        # Should use direct routing, not multi-hop
        assert result.is_multihop is False
        assert result.path is None  # No path for direct routing

    def test_multihop_solution_has_correct_gas(self, router):
        """Multi-hop solution includes gas for all hops."""
        from solver.constants import DAI, USDC

        order = make_order(
            sell_token=USDC,
            buy_token=DAI,
            sell_amount="1000000000",
            buy_amount="1",  # Very low min for test
            kind=OrderKind.SELL,
        )
        result = router.route_order(order)
        solution = router.build_solution(result)

        assert solution is not None
        # Gas = 2 swaps * 60k + 106391 settlement overhead = 226391
        assert solution.gas == 226_391

    def test_multihop_solution_encodes_full_path(self, router):
        """Multi-hop solution encodes the full path in calldata."""
        from solver.constants import DAI, USDC, WETH

        order = make_order(
            sell_token=USDC,
            buy_token=DAI,
            sell_amount="1000000000",
            buy_amount="1",
            kind=OrderKind.SELL,
        )
        result = router.route_order(order)
        solution = router.build_solution(result)

        assert solution is not None
        # Multi-hop routes have one LiquidityInteraction per hop
        assert len(solution.interactions) == 2

        # First hop: USDC → WETH
        hop1 = solution.interactions[0]
        assert hop1.kind == "liquidity"
        assert hop1.input_token.lower() == USDC.lower()
        assert hop1.output_token.lower() == WETH.lower()

        # Second hop: WETH → DAI
        hop2 = solution.interactions[1]
        assert hop2.kind == "liquidity"
        assert hop2.input_token.lower() == WETH.lower()
        assert hop2.output_token.lower() == DAI.lower()

        # Amounts should chain correctly: hop1.output = hop2.input
        assert hop1.output_amount == hop2.input_amount
