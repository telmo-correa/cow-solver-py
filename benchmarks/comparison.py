"""Solution comparison utilities for benchmark analysis.

This module provides functions to compare solutions from Python and Rust solvers,
detecting matches, improvements, and regressions.
"""

from typing import Any

from solver.models.solution import (
    CustomInteraction,
    LiquidityInteraction,
    Solution,
    SolverResponse,
)


def extract_output_amounts(solution: Solution) -> dict[str, int]:
    """Extract output amounts from a solution's interactions by token.

    Returns a dict mapping token address (lowercased) to total output amount.
    Works with both CustomInteraction (Python) and LiquidityInteraction (Rust).
    """
    outputs: dict[str, int] = {}

    for interaction in solution.interactions:
        if isinstance(interaction, LiquidityInteraction):
            # Rust solver uses LiquidityInteraction with output_amount
            token = interaction.output_token.lower()
            amount = int(interaction.output_amount)
            outputs[token] = outputs.get(token, 0) + amount
        elif isinstance(interaction, CustomInteraction):
            # Python solver uses CustomInteraction with outputs list
            for output in interaction.outputs:
                token = output.token.lower()
                amount = int(output.amount)
                outputs[token] = outputs.get(token, 0) + amount

    return outputs


def extract_executed_amounts(solution: Solution) -> dict[str, int]:
    """Extract executed amounts from a solution's trades by order UID.

    Returns a dict mapping order UID (lowercased) to executed amount.
    """
    executed: dict[str, int] = {}
    for trade in solution.trades:
        uid = trade.order.lower()
        amount = int(trade.executed_amount)
        executed[uid] = executed.get(uid, 0) + amount
    return executed


def calculate_fill_ratios(
    python_response: SolverResponse,
    rust_response: SolverResponse,
    orders: list[dict[str, Any]],
) -> tuple[float, float]:
    """Calculate average fill ratios for Python and Rust solutions.

    Args:
        python_response: Response from Python solver (must have solutions)
        rust_response: Response from Rust solver (must have solutions)
        orders: List of order dicts from auction JSON

    Returns:
        Tuple of (python_avg_ratio, rust_avg_ratio)
    """
    python_executed = extract_executed_amounts(python_response.solutions[0])
    rust_executed = extract_executed_amounts(rust_response.solutions[0])

    total_python_ratio = 0.0
    total_rust_ratio = 0.0
    order_count = 0

    for order in orders:
        uid = order.get("uid", "").lower()
        kind = order.get("kind", "sell")
        sell_amount = int(order.get("sellAmount", "0"))
        buy_amount = int(order.get("buyAmount", "0"))

        python_exec = python_executed.get(uid, 0)
        rust_exec = rust_executed.get(uid, 0)

        # Calculate fill ratios based on order type
        if kind == "sell":
            if sell_amount > 0:
                python_ratio = python_exec / sell_amount
                rust_ratio = rust_exec / sell_amount
            else:
                continue
        else:
            if buy_amount > 0:
                python_ratio = python_exec / buy_amount
                rust_ratio = rust_exec / buy_amount
            else:
                continue

        total_python_ratio += python_ratio
        total_rust_ratio += rust_ratio
        order_count += 1

    if order_count == 0:
        return 0.0, 0.0

    return total_python_ratio / order_count, total_rust_ratio / order_count


def check_improvement(
    python_response: SolverResponse | None,
    rust_response: SolverResponse | None,
    auction_json: dict[str, Any],
) -> dict[str, Any]:
    """Check if Python's solution is an improvement over Rust's.

    For partial fills, Python may execute more of an order while still
    respecting the limit price. This is an improvement, not a mismatch.

    Args:
        python_response: Response from Python solver
        rust_response: Response from Rust solver
        auction_json: Original auction JSON with orders

    Returns:
        dict with:
        - is_improvement: True if Python fills more while respecting limits
        - is_regression: True if Python fills less or violates limits
        - python_fill_ratio: Fill ratio for Python (0.0-1.0)
        - rust_fill_ratio: Fill ratio for Rust (0.0-1.0)
        - improvement_pct: Percentage improvement (positive = Python better)
    """
    result: dict[str, Any] = {
        "is_improvement": False,
        "is_regression": False,
        "python_fill_ratio": None,
        "rust_fill_ratio": None,
        "improvement_pct": None,
    }

    # Need both responses with solutions
    if not python_response or not python_response.solutions:
        return result
    if not rust_response or not rust_response.solutions:
        return result

    orders = auction_json.get("orders", [])
    if not orders:
        return result

    avg_python_ratio, avg_rust_ratio = calculate_fill_ratios(python_response, rust_response, orders)

    result["python_fill_ratio"] = avg_python_ratio
    result["rust_fill_ratio"] = avg_rust_ratio

    # Calculate improvement percentage
    if avg_rust_ratio > 0:
        improvement = (avg_python_ratio - avg_rust_ratio) / avg_rust_ratio * 100
        result["improvement_pct"] = improvement

        # Threshold for considering it an improvement (> 1% more fill)
        if improvement > 1.0:
            result["is_improvement"] = True
        elif improvement < -1.0:
            result["is_regression"] = True

    return result


def _compare_output_tokens(
    python_outputs: dict[str, int],
    rust_outputs: dict[str, int],
) -> tuple[list[str], list[str]]:
    """Compare output tokens between solutions.

    Returns:
        Tuple of (differences, notes) lists
    """
    differences = []
    notes = []

    common_tokens = set(python_outputs.keys()) & set(rust_outputs.keys())
    python_only_tokens = set(python_outputs.keys()) - set(rust_outputs.keys())
    rust_only_tokens = set(rust_outputs.keys()) - set(python_outputs.keys())

    # Compare common tokens (must match exactly)
    for token in common_tokens:
        py_amt = python_outputs[token]
        rs_amt = rust_outputs[token]
        if py_amt != rs_amt:
            if rs_amt > 0:
                rel_diff = abs(py_amt - rs_amt) / rs_amt * 100
                differences.append(
                    f"{token[:10]}...: Python={py_amt}, Rust={rs_amt} ({rel_diff:.2f}% diff)"
                )
            else:
                differences.append(f"{token[:10]}...: Python={py_amt}, Rust={rs_amt}")

    # Note tokens only in one solver (likely intermediate tokens in multi-hop)
    if rust_only_tokens:
        notes.append(f"Rust has {len(rust_only_tokens)} extra token(s) (likely intermediate)")
    if python_only_tokens:
        notes.append(f"Python has {len(python_only_tokens)} extra token(s)")

    return differences, notes


def compare_solutions(
    python_response: SolverResponse | None,
    rust_response: SolverResponse | None,
    auction_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare solutions from Python and Rust solvers.

    Args:
        python_response: Response from Python solver
        rust_response: Response from Rust solver
        auction_json: Optional auction JSON for improvement detection

    Returns:
        dict with:
        - match: bool indicating if solutions are equivalent
        - improvement: bool indicating if Python is better
        - regression: bool indicating if Python is worse
        - python_outputs: dict of token -> amount for Python solution
        - rust_outputs: dict of token -> amount for Rust solution
        - diff: description of any differences
        - improvement_info: dict with fill ratio details (if applicable)
    """
    result: dict[str, Any] = {
        "match": False,
        "improvement": False,
        "regression": False,
        "python_outputs": {},
        "rust_outputs": {},
        "diff": "",
        "improvement_info": {},
    }

    # Handle missing responses
    if not python_response or not python_response.solutions:
        if not rust_response or not rust_response.solutions:
            result["match"] = True
            result["diff"] = "Both solvers returned no solutions"
            return result
        result["diff"] = "Python returned no solutions, Rust did"
        result["rust_outputs"] = extract_output_amounts(rust_response.solutions[0])
        return result

    if not rust_response or not rust_response.solutions:
        result["diff"] = "Rust returned no solutions, Python did"
        result["python_outputs"] = extract_output_amounts(python_response.solutions[0])
        return result

    # Extract outputs
    python_outputs = extract_output_amounts(python_response.solutions[0])
    rust_outputs = extract_output_amounts(rust_response.solutions[0])
    result["python_outputs"] = python_outputs
    result["rust_outputs"] = rust_outputs

    if not python_outputs and not rust_outputs:
        result["match"] = True
        result["diff"] = "Both have no interaction outputs"
        return result

    # Compare outputs
    differences, notes = _compare_output_tokens(python_outputs, rust_outputs)

    # Build result
    if differences:
        result["diff"] = _build_diff_message(
            differences, notes, python_response, rust_response, auction_json, result
        )
    else:
        result["match"] = True
        result["diff"] = (
            f"Common outputs match [{', '.join(notes)}]" if notes else "Solutions match"
        )

    return result


def _build_diff_message(
    differences: list[str],
    notes: list[str],
    python_response: SolverResponse,
    rust_response: SolverResponse,
    auction_json: dict[str, Any] | None,
    result: dict[str, Any],
) -> str:
    """Build the diff message, checking for improvements if auction context available."""
    if auction_json:
        improvement_info = check_improvement(python_response, rust_response, auction_json)
        result["improvement_info"] = improvement_info

        if improvement_info.get("is_improvement"):
            result["improvement"] = True
            py_ratio = improvement_info.get("python_fill_ratio", 0)
            rs_ratio = improvement_info.get("rust_fill_ratio", 0)
            pct = improvement_info.get("improvement_pct", 0)
            msg = f"Python fills {py_ratio:.1%} vs Rust {rs_ratio:.1%} (+{pct:.1f}% improvement)"
            return f"{msg} [{', '.join(notes)}]" if notes else msg
        elif improvement_info.get("is_regression"):
            result["regression"] = True

    base_diff = "; ".join(differences)
    return f"{base_diff} [{', '.join(notes)}]" if notes else base_diff
