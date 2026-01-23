#!/usr/bin/env python3
"""Validate Rust fixtures work with Python solver.

This script tests each extracted Rust fixture against the Python solver
and compares the results with the expected output.

Usage:
    python scripts/validate_rust_fixtures.py
"""

import json
import subprocess
from pathlib import Path

FIXTURES_DIR = Path("tests/fixtures/auctions/benchmark_rust")


def compare_solutions(python_sol: dict, expected_sol: dict) -> tuple[bool, list[str]]:
    """Compare Python solution with expected.

    Returns (match, differences) tuple.
    """
    differences = []

    if not python_sol.get("solutions") and expected_sol.get("solutions"):
        return False, ["Python returned no solutions"]

    if not expected_sol.get("solutions"):
        return True, []  # Both have no solutions

    py_sol = python_sol["solutions"][0]
    exp_sol = expected_sol["solutions"][0]

    # Compare prices (ignore order of keys)
    py_prices = py_sol.get("prices", {})
    exp_prices = exp_sol.get("prices", {})

    for token, exp_price in exp_prices.items():
        token_lower = token.lower()
        py_price = py_prices.get(token) or py_prices.get(token_lower)
        if py_price != exp_price:
            differences.append(f"Price mismatch for {token[:10]}...: {py_price} vs {exp_price}")

    # Compare gas
    if py_sol.get("gas") != exp_sol.get("gas"):
        differences.append(f"Gas mismatch: {py_sol.get('gas')} vs {exp_sol.get('gas')}")

    # Compare interaction amounts
    if py_sol.get("interactions") and exp_sol.get("interactions"):
        py_int = py_sol["interactions"][0]
        exp_int = exp_sol["interactions"][0]
        if py_int.get("outputAmount") != exp_int.get("outputAmount"):
            differences.append(
                f"Output amount mismatch: {py_int.get('outputAmount')} vs {exp_int.get('outputAmount')}"
            )

    return len(differences) == 0, differences


def test_fixture(fixture_path: Path) -> dict:
    """Test a single fixture against Python solver.

    Returns result dict with status and details.
    """
    expected_path = fixture_path.with_name(fixture_path.stem + "_expected.json")

    result = {
        "fixture": fixture_path.name,
        "status": "unknown",
        "details": [],
    }

    if not expected_path.exists():
        result["status"] = "skip"
        result["details"].append("No expected file")
        return result

    # Load expected output
    with open(expected_path) as f:
        expected = json.load(f)

    # Call Python solver (assumes it's running on port 8000)
    try:
        proc = subprocess.run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                "http://localhost:8000/mainnet/mainnet",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                f"@{fixture_path}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if proc.returncode != 0:
            result["status"] = "error"
            result["details"].append(f"curl failed: {proc.stderr}")
            return result

        python_output = json.loads(proc.stdout)

    except json.JSONDecodeError as e:
        result["status"] = "error"
        result["details"].append(f"Invalid JSON response: {e}")
        return result
    except subprocess.TimeoutExpired:
        result["status"] = "error"
        result["details"].append("Request timed out")
        return result
    except Exception as e:
        result["status"] = "error"
        result["details"].append(str(e))
        return result

    # Compare results
    match, differences = compare_solutions(python_output, expected)

    if match:
        result["status"] = "pass"
    elif not python_output.get("solutions"):
        result["status"] = "no_solution"
        result["details"].append("Python returned no solutions")
    else:
        result["status"] = "mismatch"
        result["details"] = differences

    return result


def main():
    """Run validation on all fixtures."""
    fixtures = sorted(FIXTURES_DIR.glob("*.json"))
    fixtures = [
        f for f in fixtures if not f.name.endswith("_expected.json") and f.name != "index.json"
    ]

    print(f"Validating {len(fixtures)} fixtures...")
    print("=" * 60)

    results = {"pass": [], "mismatch": [], "no_solution": [], "error": [], "skip": []}

    for fixture in fixtures:
        result = test_fixture(fixture)
        results[result["status"]].append(result)
        status_icon = {"pass": "✓", "mismatch": "~", "no_solution": "○", "error": "✗", "skip": "-"}
        icon = status_icon.get(result["status"], "?")
        print(f"{icon} {fixture.name}: {result['status']}")
        if result["details"]:
            for detail in result["details"][:2]:
                print(f"    {detail}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Pass:        {len(results['pass'])}")
    print(f"  Mismatch:    {len(results['mismatch'])}")
    print(f"  No solution: {len(results['no_solution'])}")
    print(f"  Error:       {len(results['error'])}")
    print(f"  Skip:        {len(results['skip'])}")

    # Create compatibility report
    compatible = [r["fixture"] for r in results["pass"]]
    report = {
        "total": len(fixtures),
        "compatible": len(compatible),
        "compatible_fixtures": compatible,
        "partial_match": [r["fixture"] for r in results["mismatch"]],
        "no_solution": [r["fixture"] for r in results["no_solution"]],
    }

    report_path = FIXTURES_DIR / "compatibility.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nCompatibility report: {report_path}")


if __name__ == "__main__":
    main()
