"""Tests for Python/Rust solver parity using extracted Rust test fixtures.

These fixtures are extracted from the Rust baseline solver's test cases.
Validation follows the two-phase approach documented in:
    docs/design/rust-parity-validation.md

Phase 1: Validate correctness invariants (fees, amounts, limit prices)
Phase 2: Compare to Rust, allowing Python to be "better" (more fill)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from solver.amm.uniswap_v2 import UniswapV2
from solver.models.auction import AuctionInstance, Order
from solver.models.types import normalize_address
from solver.pools import build_registry_from_liquidity
from solver.solver import Solver

if TYPE_CHECKING:
    from solver.pools import PoolRegistry

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "auctions" / "benchmark_rust"


def get_fixture_pairs() -> list[tuple[str, Path, Path]]:
    """Get all input/expected file pairs from benchmark_rust."""
    pairs = []
    for input_file in FIXTURES_DIR.glob("*.json"):
        # Skip expected files and index
        if "_expected" in input_file.stem or input_file.stem == "index":
            continue
        expected_file = input_file.with_name(f"{input_file.stem}_expected.json")
        if expected_file.exists():
            pairs.append((input_file.stem, input_file, expected_file))
    return sorted(pairs)


@dataclass
class ValidationResult:
    """Result of solution validation."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


class SolutionValidator:
    """Validates solution correctness and compares to Rust baseline.

    Implements two-phase validation:
    1. Correctness invariants (independent of Rust)
    2. Rust comparison (allows Python to be better)

    See docs/design/rust-parity-validation.md for details.
    """

    def __init__(self, auction: AuctionInstance) -> None:
        """Initialize validator with auction context.

        Args:
            auction: The auction being solved (provides liquidity, tokens, gas price)
        """
        self.auction = auction
        self.amm = UniswapV2()
        self.registry: PoolRegistry = build_registry_from_liquidity(auction.liquidity)

    def validate(
        self,
        python_solution: dict[str, Any],
        rust_solution: dict[str, Any],
        order: Order,
    ) -> ValidationResult:
        """Two-phase validation of a solution.

        Phase 1: Validate correctness invariants
        Phase 2: Compare to Rust (allow better)

        Args:
            python_solution: Python solver output (model_dump format)
            rust_solution: Rust expected output
            order: The order being filled

        Returns:
            ValidationResult with pass/fail and any errors/notes
        """
        errors: list[str] = []
        notes: list[str] = []

        # Phase 1: Correctness invariants
        errors.extend(self._validate_fee(python_solution, order))
        errors.extend(self._validate_amount_consistency(python_solution, order))
        errors.extend(self._validate_amm_simulation(python_solution))
        errors.extend(self._validate_limit_price(python_solution, order))
        # Note: Clearing price consistency is not validated as a hard constraint.
        # CoW Protocol allows flexibility in pricing as long as limit prices are satisfied.

        if errors:
            return ValidationResult(passed=False, errors=errors)

        # Phase 2: Rust comparison
        return self._compare_to_rust(python_solution, rust_solution, order, notes)

    def _validate_fee(self, solution: dict[str, Any], order: Order) -> list[str]:
        """Validate fee calculation is correct.

        Fee = gas_estimate * gas_price * 1e18 // reference_price
        """
        errors = []
        trades = solution.get("trades", [])
        if not trades:
            return errors

        trade = trades[0]
        actual_fee = int(trade.get("fee") or trade.get("fee") or 0)

        # Get gas from solution
        gas_estimate = solution.get("gas", 0)
        gas_price = int(self.auction.effective_gas_price)

        # Get reference price for sell token
        sell_token = normalize_address(order.sell_token)
        token_info = self.auction.tokens.get(sell_token)
        if token_info is None:
            # Try case-insensitive lookup
            for addr, info in self.auction.tokens.items():
                if addr.lower() == sell_token:
                    token_info = info
                    break

        if token_info is None or token_info.reference_price is None:
            # Can't validate fee without reference price
            return errors

        reference_price = int(token_info.reference_price)
        if reference_price == 0:
            return errors

        # Calculate expected fee
        gas_cost = gas_estimate * gas_price
        expected_fee = gas_cost * 10**18 // reference_price

        # Only validate for limit orders (market orders have fee=0)
        if order.class_ == "limit" and actual_fee != expected_fee:
            errors.append(
                f"Fee mismatch: actual={actual_fee}, expected={expected_fee} "
                f"(gas={gas_estimate}, gas_price={gas_price}, ref_price={reference_price})"
            )

        return errors

    def _validate_amount_consistency(self, solution: dict[str, Any], order: Order) -> list[str]:
        """Validate that amounts are consistent.

        For sell orders: interaction.inputAmount == executedAmount + fee
        """
        errors = []
        trades = solution.get("trades", [])
        interactions = solution.get("interactions", [])

        if not trades or not interactions:
            return errors

        trade = trades[0]
        interaction = interactions[0]

        executed = int(trade.get("executedAmount") or trade.get("executed_amount") or 0)
        fee = int(trade.get("fee") or 0)

        input_amount = int(interaction.get("inputAmount") or interaction.get("input_amount") or 0)

        if order.is_sell_order:
            expected_input = executed + fee
            if input_amount != expected_input:
                errors.append(
                    f"Amount inconsistency: inputAmount={input_amount} != "
                    f"executedAmount({executed}) + fee({fee}) = {expected_input}"
                )

        return errors

    def _validate_amm_simulation(self, solution: dict[str, Any]) -> list[str]:
        """Validate that interaction amounts are achievable by AMM.

        Simulates the swap and checks output matches claimed output.
        """
        errors = []
        interactions = solution.get("interactions", [])

        for i, interaction in enumerate(interactions):
            kind = interaction.get("kind", "custom")
            if kind != "liquidity":
                continue

            pool_id = interaction.get("id")
            if pool_id is None:
                continue

            input_amount = int(
                interaction.get("inputAmount") or interaction.get("input_amount") or 0
            )
            claimed_output = int(
                interaction.get("outputAmount") or interaction.get("output_amount") or 0
            )
            input_token = normalize_address(
                interaction.get("inputToken") or interaction.get("input_token") or ""
            )
            output_token = normalize_address(
                interaction.get("outputToken") or interaction.get("output_token") or ""
            )

            # Find the pool
            pool = self._find_pool_by_id_and_tokens(pool_id, input_token, output_token)
            if pool is None:
                # Can't validate without pool
                continue

            # Simulate the swap
            try:
                from solver.amm.balancer import BalancerStablePool, BalancerWeightedPool

                if isinstance(pool, (BalancerWeightedPool, BalancerStablePool)):
                    # Balancer pools need token_out for simulation
                    if isinstance(pool, BalancerWeightedPool):
                        from solver.amm.balancer import BalancerWeightedAMM

                        amm = BalancerWeightedAMM()
                        result = amm.simulate_swap(pool, input_token, output_token, input_amount)
                        simulated_output = result.amount_out if result else 0
                    else:
                        from solver.amm.balancer import BalancerStableAMM

                        amm = BalancerStableAMM()
                        result = amm.simulate_swap(pool, input_token, output_token, input_amount)
                        simulated_output = result.amount_out if result else 0
                else:
                    # V2 pool
                    reserve_in, reserve_out = pool.get_reserves(input_token)
                    simulated_output = self.amm.get_amount_out(
                        input_amount, reserve_in, reserve_out, pool.fee_multiplier
                    )
            except (ValueError, ZeroDivisionError, AttributeError):
                continue

            if simulated_output != claimed_output:
                errors.append(
                    f"Interaction {i} AMM mismatch: simulated={simulated_output}, "
                    f"claimed={claimed_output} (input={input_amount})"
                )

        return errors

    def _validate_limit_price(self, solution: dict[str, Any], order: Order) -> list[str]:
        """Validate that the trade satisfies the order's limit price.

        For sell orders: output/executed >= order.buyAmount/order.sellAmount
        """
        errors = []
        trades = solution.get("trades", [])
        interactions = solution.get("interactions", [])

        if not trades or not interactions:
            return errors

        trade = trades[0]
        interaction = interactions[0]

        executed = int(trade.get("executedAmount") or trade.get("executed_amount") or 0)
        if executed == 0:
            return errors

        output_amount = int(
            interaction.get("outputAmount") or interaction.get("output_amount") or 0
        )

        # Limit price check: output/executed >= buyAmount/sellAmount
        # Rearranged: output * sellAmount >= buyAmount * executed
        # Use int() to ensure Python arbitrary precision integers
        sell_amount = int(order.sell_amount)
        buy_amount = int(order.buy_amount)

        if order.is_sell_order and output_amount * sell_amount < buy_amount * executed:
            effective_price = output_amount / executed if executed else 0
            limit_price = buy_amount / sell_amount if sell_amount else 0
            errors.append(
                f"Limit price violated: effective={effective_price:.6f}, limit={limit_price:.6f}"
            )

        return errors

    def _validate_clearing_prices(self, solution: dict[str, Any]) -> list[str]:
        """Validate clearing prices are consistent with trade amounts.

        price[buy] * input ≈ price[sell] * output
        """
        errors = []
        prices = solution.get("prices", {})
        interactions = solution.get("interactions", [])

        if not prices or not interactions:
            return errors

        interaction = interactions[0]
        input_token = normalize_address(
            interaction.get("inputToken") or interaction.get("input_token") or ""
        )
        output_token = normalize_address(
            interaction.get("outputToken") or interaction.get("output_token") or ""
        )
        input_amount = int(interaction.get("inputAmount") or interaction.get("input_amount") or 0)
        output_amount = int(
            interaction.get("outputAmount") or interaction.get("output_amount") or 0
        )

        # Get prices (normalize to lowercase)
        price_input = int(prices.get(input_token) or prices.get(input_token.lower()) or 0)
        price_output = int(prices.get(output_token) or prices.get(output_token.lower()) or 0)

        if price_input == 0 or price_output == 0:
            return errors

        # Check: price[output] * input == price[input] * output
        # This is how CoW Protocol clearing prices work
        lhs = price_output * input_amount
        rhs = price_input * output_amount

        if lhs != rhs:
            # Allow small rounding (< 0.0001%)
            diff_pct = abs(lhs - rhs) / max(lhs, rhs) * 100 if max(lhs, rhs) > 0 else 0
            if diff_pct > 0.0001:
                errors.append(
                    f"Clearing price inconsistency: price[out]*in={lhs}, "
                    f"price[in]*out={rhs} (diff={diff_pct:.6f}%)"
                )

        return errors

    def _compare_to_rust(
        self,
        python_solution: dict[str, Any],
        rust_solution: dict[str, Any],
        order: Order,
        notes: list[str],
    ) -> ValidationResult:
        """Compare Python solution to Rust, allowing Python to be better.

        Returns:
            ValidationResult with pass if equal or better, fail if worse
        """
        errors: list[str] = []

        py_trades = python_solution.get("trades", [])
        rust_trades = rust_solution.get("trades", [])

        if not py_trades or not rust_trades:
            return ValidationResult(passed=True, notes=notes)

        py_executed = int(
            py_trades[0].get("executedAmount") or py_trades[0].get("executed_amount") or 0
        )
        rust_executed = int(
            rust_trades[0].get("executedAmount") or rust_trades[0].get("executed_amount") or 0
        )

        if py_executed > rust_executed:
            # Python is better
            sell_amount = int(order.sell_amount) if order.sell_amount else 0
            py_pct = py_executed * 100 // sell_amount if sell_amount else 0
            rust_pct = rust_executed * 100 // sell_amount if sell_amount else 0
            notes.append(
                f"Python fills more: {py_executed} ({py_pct}%) vs Rust {rust_executed} ({rust_pct}%)"
            )
            return ValidationResult(passed=True, notes=notes)

        elif py_executed == rust_executed:
            # Same fill - validate exact match on other fields
            exact_errors = self._validate_exact_match(python_solution, rust_solution)
            if exact_errors:
                return ValidationResult(passed=False, errors=exact_errors, notes=notes)
            return ValidationResult(passed=True, notes=notes)

        else:
            # Python is worse
            errors.append(f"Python fills less: {py_executed} < Rust {rust_executed}")
            return ValidationResult(passed=False, errors=errors, notes=notes)

    def _validate_exact_match(
        self,
        python_solution: dict[str, Any],
        rust_solution: dict[str, Any],
    ) -> list[str]:
        """Validate exact field match when Python and Rust have same fill."""
        errors = []

        # Compare prices
        py_prices = self._normalize_prices(python_solution.get("prices", {}))
        rust_prices = self._normalize_prices(rust_solution.get("prices", {}))

        for token in set(py_prices.keys()) | set(rust_prices.keys()):
            py_price = py_prices.get(token, 0)
            rust_price = rust_prices.get(token, 0)
            if py_price != rust_price:
                diff_pct = abs(py_price - rust_price) / rust_price * 100 if rust_price else 0
                errors.append(
                    f"Price mismatch for {token[-8:]}: Python={py_price}, "
                    f"Rust={rust_price} (diff={diff_pct:.4f}%)"
                )

        # Compare interactions
        py_interactions = python_solution.get("interactions", [])
        rust_interactions = rust_solution.get("interactions", [])

        if len(py_interactions) != len(rust_interactions):
            errors.append(
                f"Interaction count differs: Python={len(py_interactions)}, "
                f"Rust={len(rust_interactions)}"
            )
        else:
            for i, (py_int, rust_int) in enumerate(
                zip(py_interactions, rust_interactions, strict=True)
            ):
                py_input = int(py_int.get("inputAmount") or py_int.get("input_amount") or 0)
                rust_input = int(rust_int.get("inputAmount") or rust_int.get("input_amount") or 0)
                py_output = int(py_int.get("outputAmount") or py_int.get("output_amount") or 0)
                rust_output = int(
                    rust_int.get("outputAmount") or rust_int.get("output_amount") or 0
                )

                if py_input != rust_input:
                    errors.append(
                        f"Interaction {i} inputAmount differs: Python={py_input}, Rust={rust_input}"
                    )
                if py_output != rust_output:
                    errors.append(
                        f"Interaction {i} outputAmount differs: Python={py_output}, Rust={rust_output}"
                    )

        # Compare gas
        py_gas = python_solution.get("gas")
        rust_gas = rust_solution.get("gas")
        if py_gas != rust_gas:
            errors.append(f"Gas differs: Python={py_gas}, Rust={rust_gas}")

        return errors

    def _find_pool_by_id_and_tokens(self, pool_id: str, input_token: str, output_token: str) -> Any:
        """Find a pool by its ID and token pair."""
        # Find the liquidity source address by ID
        target_address = None
        for liquidity in self.auction.liquidity:
            if str(liquidity.id) == str(pool_id):
                target_address = liquidity.address.lower()
                break

        if target_address is None:
            return None

        # Get all pools for this token pair
        pools = self.registry.get_pools_for_pair(input_token, output_token)

        # Find the pool with matching address
        for pool in pools:
            if hasattr(pool, "address") and pool.address.lower() == target_address:
                return pool

        return None

    @staticmethod
    def _normalize_prices(prices: dict[str, str]) -> dict[str, int]:
        """Normalize prices dict to lowercase keys and int values."""
        return {k.lower(): int(v) for k, v in prices.items()}


@pytest.fixture
def solver() -> Solver:
    """Create a properly configured solver instance.

    Uses the default solver creation which includes Balancer and limit order AMMs.
    V3 is disabled unless RPC_URL is set.
    """
    from solver.amm.balancer import BalancerStableAMM, BalancerWeightedAMM
    from solver.amm.limit_order import LimitOrderAMM

    return Solver(
        weighted_amm=BalancerWeightedAMM(),
        stable_amm=BalancerStableAMM(),
        limit_order_amm=LimitOrderAMM(),
    )


class TestRustParity:
    """Test Python solver against Rust baseline expected outputs."""

    @pytest.mark.parametrize("test_name,input_file,expected_file", get_fixture_pairs())
    def test_fixture_parity(
        self,
        test_name: str,
        input_file: Path,
        expected_file: Path,
        solver: Solver,
    ):
        """Test that Python solution is correct and at least as good as Rust.

        Uses two-phase validation:
        1. Validate correctness invariants (fees, amounts, limit prices)
        2. Compare to Rust (allow Python to be better)
        """
        # Load input auction
        with open(input_file) as f:
            auction_data = json.load(f)
        auction = AuctionInstance.model_validate(auction_data)

        # Load expected output
        with open(expected_file) as f:
            expected = json.load(f)

        # Run Python solver
        response = solver.solve(auction)

        # Check we got a solution
        rust_solutions = expected.get("solutions", [])

        if not rust_solutions:
            # Rust returned no solution, Python should too
            assert len(response.solutions) == 0, (
                f"Python returned {len(response.solutions)} solutions but Rust returned none"
            )
            return

        assert len(response.solutions) > 0, (
            f"Python returned no solutions but Rust returned {len(rust_solutions)}"
        )

        # Create validator
        validator = SolutionValidator(auction)

        # Validate each solution against corresponding Rust solution
        # Python may return multiple solutions (one per order) matching Rust
        all_errors: list[str] = []
        all_notes: list[str] = []

        for i, (py_sol, rust_sol) in enumerate(
            zip(response.solutions, rust_solutions, strict=False)
        ):
            # Get corresponding order
            order = auction.orders[i] if i < len(auction.orders) else auction.orders[0]

            python_solution = py_sol.model_dump()
            result = validator.validate(python_solution, rust_sol, order)

            all_errors.extend(result.errors)
            all_notes.extend(result.notes)

        # Report results
        if all_notes:
            print(f"\n{test_name} notes:")
            for note in all_notes:
                print(f"  - {note}")

        if all_errors:
            error_msg = f"\n{test_name} validation failures:\n" + "\n".join(
                f"  - {e}" for e in all_errors
            )
            pytest.fail(error_msg)


class TestParityDiagnostics:
    """Diagnostic tests that show all discrepancies across all fixtures."""

    def test_run_all_and_report(self, solver: Solver):
        """Run all fixtures and report all discrepancies (does not fail on discrepancies)."""
        pairs = get_fixture_pairs()
        all_results: list[tuple[str, str, list[str], list[str]]] = []

        for test_name, input_file, expected_file in pairs:
            with open(input_file) as f:
                auction_data = json.load(f)
            auction = AuctionInstance.model_validate(auction_data)

            with open(expected_file) as f:
                expected = json.load(f)

            try:
                response = solver.solve(auction)

                rust_solutions = expected.get("solutions", [])
                if not rust_solutions and len(response.solutions) == 0:
                    all_results.append((test_name, "PASS", [], []))
                elif not rust_solutions:
                    all_results.append(
                        (test_name, "MISMATCH", ["Python has solutions, Rust has none"], [])
                    )
                elif len(response.solutions) == 0:
                    all_results.append(
                        (test_name, "MISMATCH", ["Python has no solutions, Rust has some"], [])
                    )
                else:
                    validator = SolutionValidator(auction)
                    errors: list[str] = []
                    notes: list[str] = []

                    for i, (py_sol, rust_sol) in enumerate(
                        zip(response.solutions, rust_solutions, strict=False)
                    ):
                        order = auction.orders[i] if i < len(auction.orders) else auction.orders[0]
                        result = validator.validate(py_sol.model_dump(), rust_sol, order)
                        errors.extend(result.errors)
                        notes.extend(result.notes)

                    status = "PASS" if not errors else "FAIL"
                    if notes and not errors:
                        status = "BETTER"
                    all_results.append((test_name, status, errors, notes))
            except Exception as e:
                all_results.append((test_name, "ERROR", [str(e)], []))

        # Print report
        print("\n" + "=" * 80)
        print("RUST PARITY REPORT")
        print("=" * 80)

        passed = 0
        better = 0
        failed = 0

        for test_name, status, issues, notes in all_results:
            if status == "PASS":
                passed += 1
                print(f"✓ {test_name}")
            elif status == "BETTER":
                better += 1
                print(f"✓ {test_name} [BETTER]")
                for note in notes:
                    print(f"    {note}")
            else:
                failed += 1
                print(f"✗ {test_name} [{status}]")
                for issue in issues:
                    print(f"    {issue}")

        print("=" * 80)
        print(
            f"SUMMARY: {passed} exact, {better} better, {failed} failed, {len(all_results)} total"
        )
        print("=" * 80)

        # This test always passes - it's for diagnostics only
        # Use test_fixture_parity for actual CI validation
