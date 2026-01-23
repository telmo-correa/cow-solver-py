"""Tests for Python/Rust solver parity using extracted Rust test fixtures.

These fixtures are extracted from the Rust baseline solver's test cases
and validate that Python produces identical solutions.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from solver.models.auction import AuctionInstance
from solver.solver import Solver

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


def normalize_address(addr: str) -> str:
    """Normalize an address to lowercase."""
    return addr.lower()


def normalize_prices(prices: dict[str, str]) -> dict[str, int]:
    """Normalize prices dict to lowercase keys and int values."""
    return {normalize_address(k): int(v) for k, v in prices.items()}


def normalize_trade(trade: dict[str, Any]) -> dict[str, Any]:
    """Normalize a trade for comparison.

    Handles both snake_case (Python model_dump) and camelCase (Rust format).
    """
    # Handle both naming conventions
    executed_amount = trade.get("executedAmount") or trade.get("executed_amount")

    # Handle enum values
    kind = trade["kind"]
    if hasattr(kind, "value"):
        kind = kind.value

    return {
        "kind": kind,
        "order": trade["order"].lower() if isinstance(trade["order"], str) else trade["order"],
        "executedAmount": int(executed_amount) if executed_amount else None,
    }


def normalize_interaction(interaction: dict[str, Any]) -> dict[str, Any]:
    """Normalize an interaction for comparison.

    Handles both Python model_dump() format (snake_case) and Rust format (camelCase).
    """
    kind = interaction.get("kind", "custom")

    # Handle liquidity interactions (both snake_case and camelCase)
    if kind == "liquidity":
        # Handle both snake_case (Python model_dump) and camelCase (Rust format)
        input_token = interaction.get("inputToken") or interaction.get("input_token")
        output_token = interaction.get("outputToken") or interaction.get("output_token")
        input_amount = interaction.get("inputAmount") or interaction.get("input_amount")
        output_amount = interaction.get("outputAmount") or interaction.get("output_amount")

        return {
            "kind": kind,
            "id": interaction.get("id"),
            "inputToken": normalize_address(input_token) if input_token else None,
            "outputToken": normalize_address(output_token) if output_token else None,
            "inputAmount": int(input_amount) if input_amount else 0,
            "outputAmount": int(output_amount) if output_amount else 0,
            "internalize": interaction.get("internalize", False),
        }

    # Handle custom interactions (Python format)
    if kind == "custom":
        inputs = interaction.get("inputs", [])
        outputs = interaction.get("outputs", [])

        # Extract token amounts from inputs/outputs
        input_token = inputs[0]["token"] if inputs else None
        output_token = outputs[0]["token"] if outputs else None
        input_amount = int(inputs[0]["amount"]) if inputs else 0
        output_amount = int(outputs[0]["amount"]) if outputs else 0

        return {
            "kind": kind,
            "id": None,  # Custom interactions don't have IDs
            "inputToken": normalize_address(input_token) if input_token else None,
            "outputToken": normalize_address(output_token) if output_token else None,
            "inputAmount": input_amount,
            "outputAmount": output_amount,
            "internalize": interaction.get("internalize", False),
        }

    # Unknown kind
    return {"kind": kind, "error": "unknown interaction kind"}


def compare_solutions(python_solution: dict, rust_expected: dict) -> list[str]:
    """Compare Python solution against Rust expected output.

    Returns list of discrepancy messages (empty if solutions match).
    """
    discrepancies = []

    # Compare prices
    py_prices = normalize_prices(python_solution.get("prices", {}))
    rust_prices = normalize_prices(rust_expected.get("prices", {}))

    if set(py_prices.keys()) != set(rust_prices.keys()):
        discrepancies.append(
            f"Price tokens differ: Python={set(py_prices.keys())}, Rust={set(rust_prices.keys())}"
        )
    else:
        for token in py_prices:
            py_price = py_prices[token]
            rust_price = rust_prices[token]
            if py_price != rust_price:
                diff_pct = (
                    abs(py_price - rust_price) / rust_price * 100 if rust_price else float("inf")
                )
                discrepancies.append(
                    f"Price mismatch for {token[-8:]}: Python={py_price}, Rust={rust_price} "
                    f"(diff={py_price - rust_price}, {diff_pct:.4f}%)"
                )

    # Compare trades
    py_trades = [normalize_trade(t) for t in python_solution.get("trades", [])]
    rust_trades = [normalize_trade(t) for t in rust_expected.get("trades", [])]

    if len(py_trades) != len(rust_trades):
        discrepancies.append(
            f"Trade count differs: Python={len(py_trades)}, Rust={len(rust_trades)}"
        )
    else:
        for i, (py_trade, rust_trade) in enumerate(zip(py_trades, rust_trades, strict=True)):
            if py_trade["executedAmount"] != rust_trade["executedAmount"]:
                discrepancies.append(
                    f"Trade {i} executedAmount differs: Python={py_trade['executedAmount']}, "
                    f"Rust={rust_trade['executedAmount']}"
                )

    # Compare interactions
    py_interactions = [normalize_interaction(i) for i in python_solution.get("interactions", [])]
    rust_interactions = [normalize_interaction(i) for i in rust_expected.get("interactions", [])]

    if len(py_interactions) != len(rust_interactions):
        discrepancies.append(
            f"Interaction count differs: Python={len(py_interactions)}, Rust={len(rust_interactions)}"
        )
    else:
        for i, (py_int, rust_int) in enumerate(
            zip(py_interactions, rust_interactions, strict=True)
        ):
            if py_int["inputAmount"] != rust_int["inputAmount"]:
                discrepancies.append(
                    f"Interaction {i} inputAmount differs: Python={py_int['inputAmount']}, "
                    f"Rust={rust_int['inputAmount']}"
                )
            if py_int["outputAmount"] != rust_int["outputAmount"]:
                diff = py_int["outputAmount"] - rust_int["outputAmount"]
                diff_pct = (
                    abs(diff) / rust_int["outputAmount"] * 100
                    if rust_int["outputAmount"]
                    else float("inf")
                )
                discrepancies.append(
                    f"Interaction {i} outputAmount differs: Python={py_int['outputAmount']}, "
                    f"Rust={rust_int['outputAmount']} (diff={diff}, {diff_pct:.6f}%)"
                )

    # Compare gas
    py_gas = python_solution.get("gas")
    rust_gas = rust_expected.get("gas")
    if py_gas != rust_gas:
        discrepancies.append(f"Gas differs: Python={py_gas}, Rust={rust_gas}")

    return discrepancies


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
        """Test that Python solution matches Rust expected output."""
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

        # Compare first solution
        python_solution = response.solutions[0].model_dump()
        rust_expected = rust_solutions[0]

        discrepancies = compare_solutions(python_solution, rust_expected, test_name)

        if discrepancies:
            error_msg = f"\n{test_name} parity issues:\n" + "\n".join(
                f"  - {d}" for d in discrepancies
            )
            pytest.fail(error_msg)


class TestParityDiagnostics:
    """Diagnostic tests that show all discrepancies across all fixtures."""

    def test_run_all_and_report(self, solver: Solver):
        """Run all fixtures and report all discrepancies (does not fail on discrepancies)."""
        pairs = get_fixture_pairs()
        all_results = []

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
                    all_results.append((test_name, "PASS", []))
                elif not rust_solutions:
                    all_results.append(
                        (test_name, "MISMATCH", ["Python has solutions, Rust has none"])
                    )
                elif len(response.solutions) == 0:
                    all_results.append(
                        (test_name, "MISMATCH", ["Python has no solutions, Rust has some"])
                    )
                else:
                    python_solution = response.solutions[0].model_dump()
                    discrepancies = compare_solutions(python_solution, rust_solutions[0], test_name)
                    status = "PASS" if not discrepancies else "MISMATCH"
                    all_results.append((test_name, status, discrepancies))
            except Exception as e:
                all_results.append((test_name, "ERROR", [str(e)]))

        # Print report
        print("\n" + "=" * 80)
        print("RUST PARITY REPORT")
        print("=" * 80)

        passed = 0
        failed = 0

        for test_name, status, issues in all_results:
            if status == "PASS":
                passed += 1
                print(f"✓ {test_name}")
            else:
                failed += 1
                print(f"✗ {test_name} [{status}]")
                for issue in issues:
                    print(f"    {issue}")

        print("=" * 80)
        print(f"SUMMARY: {passed} passed, {failed} failed, {len(all_results)} total")
        print("=" * 80)

        # This test always passes - it's for diagnostics only
        # Use test_fixture_parity for actual CI validation
