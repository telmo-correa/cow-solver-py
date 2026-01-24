#!/usr/bin/env python3
"""Safe math linting script for CoW Solver.

This script scans the codebase for unsafe math patterns that could cause
precision issues in financial calculations. It should be run as part of CI
to prevent regressions.

Usage:
    python scripts/check_safe_math.py [--verbose] [--include-tests]

Exit codes:
    0 - No issues found
    1 - Issues found (with details printed)
"""

import argparse
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Issue:
    """A detected unsafe math pattern."""

    file: Path
    line_num: int
    line: str
    pattern: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    suggestion: str | None = None


# Directories to scan
SCAN_DIRS = ["solver"]

# Files/directories to completely skip
SKIP_PATHS = [
    "__pycache__",
    "research/",  # Research code is more relaxed
]

# Allowlist: specific files where certain patterns are acceptable
# Format: {file_pattern: [list of allowed pattern names]}
ALLOWLIST = {
    # Logging-only divisions are OK in these files
    "double_auction/core.py": [
        "decimal_division_logging",
        "float_logging",
        "float_conversion_logging",
    ],
    "double_auction/hybrid.py": [
        "decimal_division_logging",
        "float_logging",
        "float_conversion_logging",
    ],
    "settlement.py": ["float_logging", "float_conversion_logging"],  # Diagnostic logging
    "ebbo.py": ["decimal_division_logging", "float_logging"],  # Post-check logging only
    # API return values (not used in comparisons)
    "router.py": ["decimal_division_api"],  # Returns Decimal for API compatibility
    "pool.py": ["float_division_property"],  # Fee property returns fraction
    "pools.py": ["float_division_property"],  # Fee property returns fraction
    "auction.py": ["decimal_division_api"],  # limit_price property
    "fixed_point.py": ["decimal_division_api"],  # to_decimal() method
    # Validation helpers that calculate max deviation bounds
    "parsing.py": ["decimal_division_bounds"],  # Calculates max_deviation bound
    # Fill ratio diagnostics (not used in financial decisions)
    "base.py": ["float_division_property"],  # sell_fill_ratio, buy_fill_ratio
    # High-precision Decimal context usage (divisions wrapped in localcontext)
    # These use _DECIMAL_HIGH_PREC_CONTEXT with prec=78 for exact arithmetic
    "pricing.py": ["decimal_high_precision"],  # All divisions in high-prec context
    "unified_cow.py": ["decimal_high_precision"],  # All divisions in high-prec context
    "ebbo_bounds.py": ["decimal_high_precision"],  # All divisions in high-prec context
}


def should_skip_file(path: Path) -> bool:
    """Check if file should be completely skipped."""
    path_str = str(path)
    return any(skip in path_str for skip in SKIP_PATHS)


def is_allowlisted(path: Path, pattern_name: str) -> bool:
    """Check if a pattern is allowlisted for this file."""
    path_str = str(path)
    for file_pattern, allowed in ALLOWLIST.items():
        if file_pattern in path_str and pattern_name in allowed:
            return True
    return False


class DocstringTracker:
    """Track docstring state across multiple lines."""

    def __init__(self) -> None:
        self.in_docstring = False
        self.docstring_char: str | None = None

    def process_line(self, line: str) -> tuple[str, bool]:
        """Process a line and return (stripped_line, is_in_docstring).

        Returns the line with strings/comments removed and whether
        the line is entirely within a docstring.
        """
        result = []
        i = 0
        line_start_in_docstring = self.in_docstring

        while i < len(line):
            # Check for triple quotes
            if line[i : i + 3] in ('"""', "'''"):
                if not self.in_docstring:
                    self.in_docstring = True
                    self.docstring_char = line[i : i + 3]
                    result.append("   ")
                    i += 3
                    continue
                elif line[i : i + 3] == self.docstring_char:
                    self.in_docstring = False
                    self.docstring_char = None
                    result.append("   ")
                    i += 3
                    continue

            if self.in_docstring:
                result.append(" ")
                i += 1
                continue

            char = line[i]

            # Check for single-line comment
            if char == "#":
                # Rest of line is comment
                result.append(" " * (len(line) - i))
                break

            # Check for string literals (not docstrings)
            if char in ('"', "'") and (i == 0 or line[i - 1] != "\\"):
                # Find the closing quote
                quote_char = char
                result.append(" ")
                i += 1
                while i < len(line):
                    if line[i] == quote_char and line[i - 1] != "\\":
                        result.append(" ")
                        i += 1
                        break
                    result.append(" ")
                    i += 1
                continue

            result.append(char)
            i += 1

        # Line is "in docstring" if it started in one and stayed in one
        entirely_in_docstring = line_start_in_docstring and self.in_docstring

        return "".join(result), entirely_in_docstring


def check_float_division(
    path: Path, lines: list[str], tracker: DocstringTracker
) -> Iterator[Issue]:
    """Check for true division that isn't floor division."""
    for i, original_line in enumerate(lines, 1):
        # Skip pure comment lines
        if original_line.strip().startswith("#"):
            tracker.process_line(original_line)  # Keep tracker in sync
            continue

        # Skip import lines
        stripped = original_line.strip()
        if stripped.startswith(("from ", "import ")):
            tracker.process_line(original_line)
            continue

        # Remove comments and strings for pattern matching
        line, in_docstring = tracker.process_line(original_line)

        # Skip lines entirely within docstrings
        if in_docstring:
            continue

        # Skip if no division at all
        if "/" not in line:
            continue

        # Skip if only floor division
        if "//" in line:
            temp = line.replace("//", "XX")
            if "/" not in temp:
                continue

        # Look for actual division operator (not in URLs, paths, etc.)
        # Pattern: word/number or number/word or word/word with spaces around
        division_pattern = re.compile(r"(\w+|\))\s*/\s*(\w+|\()")

        for _match in division_pattern.finditer(line):
            # Get context to determine if it's financial (from stripped line, not original)
            context = line.lower()

            # Skip if it's clearly a URL or file path
            if "http" in context or "path" in context or "file" in context:
                continue

            # Skip if Fraction is used (exact rational arithmetic)
            if "Fraction(" in line:
                continue

            # Skip if using high-precision context (localcontext)
            if "localcontext" in line:
                continue

            # Check if it's Decimal division
            if "Decimal" in line:
                if is_allowlisted(path, "decimal_division_logging"):
                    continue
                if is_allowlisted(path, "decimal_division_api"):
                    continue
                if is_allowlisted(path, "decimal_division_bounds"):
                    continue
                # Skip files using high-precision Decimal context
                if is_allowlisted(path, "decimal_high_precision"):
                    continue
                yield Issue(
                    file=path,
                    line_num=i,
                    line=original_line.rstrip(),
                    pattern="Decimal division",
                    severity="HIGH",
                    message="Decimal division - use cross-multiplication for comparisons",
                    suggestion="a/b >= c/d should be a*d >= c*b",
                )
            elif "float" not in context and "log" not in context:
                # Check for allowlisted property patterns
                if is_allowlisted(path, "float_division_property"):
                    continue
                # Check for high-precision Decimal context usage
                if is_allowlisted(path, "decimal_high_precision"):
                    continue
                # Regular division on what might be integers
                financial_keywords = ["amount", "price", "fee", "balance", "rate", "fill"]
                if any(kw in context for kw in financial_keywords):
                    yield Issue(
                        file=path,
                        line_num=i,
                        line=original_line.rstrip(),
                        pattern="float division",
                        severity="HIGH",
                        message="Division on financial values - may lose precision",
                        suggestion="Use Fraction or integer cross-multiplication",
                    )


def check_tolerance_patterns(
    path: Path, lines: list[str], tracker: DocstringTracker
) -> Iterator[Issue]:
    """Check for tolerance-based comparisons (excluding zero tolerance)."""
    for i, original_line in enumerate(lines, 1):
        # Skip comments
        if original_line.strip().startswith("#"):
            tracker.process_line(original_line)
            continue

        line, in_docstring = tracker.process_line(original_line)

        # Skip lines entirely within docstrings
        if in_docstring:
            continue

        # Pattern 1: abs(something) < number (non-zero)
        abs_pattern = re.compile(r"abs\s*\([^)]+\)\s*[<>]=?\s*(\d+\.?\d*)")
        match = abs_pattern.search(line)
        if match:
            tolerance_value = match.group(1)
            # Skip if tolerance is 0 or 1 (1 wei is acceptable)
            if tolerance_value not in ("0", "1", "0.0", "1.0"):
                yield Issue(
                    file=path,
                    line_num=i,
                    line=original_line.rstrip(),
                    pattern="abs() tolerance",
                    severity="CRITICAL",
                    message=f"Tolerance-based comparison (tolerance={tolerance_value})",
                    suggestion="Use exact integer comparison or cross-multiplication",
                )

        # Pattern 2: multiplying by 0.99x or 1.00x (percentage tolerance)
        pct_pattern = re.compile(r"\*\s*(0\.9[0-9]+|1\.0[0-9]+)")
        match = pct_pattern.search(line)
        if match:
            # Skip if it's in a comment context
            if "tolerance" not in line.lower() or "no tolerance" in line.lower():
                continue
            yield Issue(
                file=path,
                line_num=i,
                line=original_line.rstrip(),
                pattern="percentage tolerance",
                severity="CRITICAL",
                message=f"Percentage-based tolerance multiplier: {match.group(1)}",
                suggestion="Use exact comparison instead of percentage tolerance",
            )


def check_float_conversion(
    path: Path, lines: list[str], tracker: DocstringTracker
) -> Iterator[Issue]:
    """Check for float() on financial values."""
    for i, original_line in enumerate(lines, 1):
        if original_line.strip().startswith("#"):
            tracker.process_line(original_line)
            continue

        line, in_docstring = tracker.process_line(original_line)

        # Skip lines entirely within docstrings
        if in_docstring:
            continue

        if "float(" not in line:
            continue

        context = line.lower()

        # Skip if it's for logging (check original line for variable names like deficit_pct)
        if (
            "log" in context or "debug" in context or "info" in context or "warning" in context
        ) and is_allowlisted(path, "float_logging"):
            continue
        # Also skip if variable name suggests logging/diagnostics
        if ("_pct" in line or "deficit" in line or "surplus" in line) and is_allowlisted(
            path, "float_logging"
        ):
            continue

        # Skip float conversion used for logging/display (not in comparisons)
        if is_allowlisted(path, "float_conversion_logging"):
            continue

        # Skip float("inf") sentinels
        if 'float("inf")' in line or "float('inf')" in line:
            continue

        # Check for financial context
        financial_keywords = ["amount", "price", "fee", "balance", "rate", "fill", "value"]
        if any(kw in context for kw in financial_keywords):
            yield Issue(
                file=path,
                line_num=i,
                line=original_line.rstrip(),
                pattern="float conversion",
                severity="HIGH",
                message="float() on potential financial value",
                suggestion="Use SafeInt or keep as integer ratio",
            )


def check_cross_multiplication(
    path: Path, lines: list[str], tracker: DocstringTracker
) -> Iterator[Issue]:
    """Check for cross-multiplication without SafeInt protection."""
    # Pattern: a * b >= c * d (or similar comparisons)
    pattern = re.compile(r"(\w+)\s*\*\s*(\w+)\s*([<>=!]+)\s*(\w+)\s*\*\s*(\w+)")

    # Pre-scan: check if variables are pre-wrapped with SafeInt in function scope
    # Look for patterns like "var_a = S(..." which indicates var_a is SafeInt
    safeint_vars: set[str] = set()
    safeint_pattern = re.compile(r"(\w+)\s*=\s*S\(")
    for line in lines:
        for m in safeint_pattern.finditer(line):
            safeint_vars.add(m.group(1))
    # Also check tuple unpacking: "a, b = S(...), S(...)"
    tuple_pattern = re.compile(r"(\w+),\s*(\w+)\s*=\s*S\(")
    for line in lines:
        for m in tuple_pattern.finditer(line):
            safeint_vars.add(m.group(1))
            safeint_vars.add(m.group(2))

    for i, original_line in enumerate(lines, 1):
        if original_line.strip().startswith("#"):
            tracker.process_line(original_line)
            continue

        line, in_docstring = tracker.process_line(original_line)

        # Skip lines entirely within docstrings
        if in_docstring:
            continue

        match = pattern.search(line)

        if match:
            # Check if SafeInt is used directly
            if "S(" in line:
                continue

            # Check if variables are pre-wrapped SafeInt
            vars_found = [match.group(j) for j in range(1, 6) if match.group(j)]
            if all(v in safeint_vars for v in vars_found if v.isidentifier()):
                continue

            financial_patterns = ["amount", "price", "fee", "balance", "fill", "buy", "sell"]

            is_financial = any(
                any(fp in v.lower() for fp in financial_patterns) for v in vars_found
            )

            if is_financial:
                yield Issue(
                    file=path,
                    line_num=i,
                    line=original_line.rstrip(),
                    pattern="unprotected cross-multiplication",
                    severity="HIGH",
                    message="Cross-multiplication without SafeInt overflow protection",
                    suggestion="Use S(a) * S(b) >= S(c) * S(d)",
                )


def scan_file(path: Path) -> list[Issue]:
    """Scan a single file for unsafe math patterns."""
    if should_skip_file(path):
        return []

    try:
        content = path.read_text()
        lines = content.split("\n")
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}", file=sys.stderr)
        return []

    issues = []

    # Each check needs its own tracker since they iterate independently
    issues.extend(check_float_division(path, lines, DocstringTracker()))
    issues.extend(check_tolerance_patterns(path, lines, DocstringTracker()))
    issues.extend(check_float_conversion(path, lines, DocstringTracker()))
    issues.extend(check_cross_multiplication(path, lines, DocstringTracker()))

    return issues


def scan_tests(base_dir: Path) -> list[Issue]:
    """Scan test files for tolerance patterns that should be documented."""
    issues = []
    test_dir = base_dir / "tests"

    if not test_dir.exists():
        return issues

    for py_file in test_dir.rglob("*.py"):
        try:
            lines = py_file.read_text().split("\n")
        except Exception:
            continue

        tracker = DocstringTracker()
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("#"):
                tracker.process_line(line)
                continue

            clean, in_docstring = tracker.process_line(line)

            # Skip lines entirely within docstrings
            if in_docstring:
                continue

            # Check for tolerance patterns
            if re.search(r"abs\s*\([^)]+\)\s*[<>]=?\s*(\d+\.?\d*)", clean):
                match = re.search(r"[<>]=?\s*(\d+\.?\d*)", clean)
                if match and match.group(1) not in ("0", "1", "0.0", "1.0"):
                    # Check if documented
                    has_doc = (
                        "#" in line
                        or (i > 1 and "#" in lines[i - 2])
                        or (i > 2 and '"""' in lines[i - 3])
                    )
                    if not has_doc:
                        issues.append(
                            Issue(
                                file=py_file,
                                line_num=i,
                                line=line.rstrip(),
                                pattern="undocumented test tolerance",
                                severity="MEDIUM",
                                message="Test uses tolerance without documentation",
                                suggestion="Add comment explaining why tolerance is acceptable",
                            )
                        )

    return issues


def print_report(issues: list[Issue], verbose: bool) -> None:
    """Print the audit report."""
    if not issues:
        print("✓ No unsafe math patterns found!")
        return

    by_severity = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
    for issue in issues:
        by_severity[issue.severity].append(issue)

    print(f"\n{'=' * 70}")
    print("SAFE MATH AUDIT RESULTS")
    print(f"{'=' * 70}")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = len(by_severity[sev])
        if count > 0:
            print(f"  {sev:10} {count:4}")
    print(f"  {'TOTAL':10} {len(issues):4}")
    print(f"{'=' * 70}\n")

    for severity in ["CRITICAL", "HIGH", "MEDIUM"]:
        items = by_severity[severity]
        if not items:
            continue

        print(f"\n[{severity}] {len(items)} issue(s):\n")
        for issue in items:
            rel_path = (
                issue.file.relative_to(Path.cwd())
                if issue.file.is_relative_to(Path.cwd())
                else issue.file
            )
            print(f"  {rel_path}:{issue.line_num}")
            print(f"    {issue.pattern}: {issue.message}")
            if verbose:
                print(f"    > {issue.line.strip()[:70]}")
                if issue.suggestion:
                    print(f"    Suggestion: {issue.suggestion}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Safe math linter for CoW Solver")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--include-tests", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    issues = []

    # Scan main code
    for scan_dir in SCAN_DIRS:
        dir_path = base_dir / scan_dir
        if dir_path.exists():
            for py_file in dir_path.rglob("*.py"):
                issues.extend(scan_file(py_file))

    # Scan tests if requested
    if args.include_tests:
        issues.extend(scan_tests(base_dir))

    print_report(issues, args.verbose)

    # Exit code based on severity
    has_critical = any(i.severity == "CRITICAL" for i in issues)
    has_high = any(i.severity == "HIGH" for i in issues)

    if has_critical:
        print("❌ CRITICAL issues found - must fix before commit")
        sys.exit(1)
    elif has_high:
        print("⚠ HIGH severity issues found - should fix")
        sys.exit(1)
    else:
        print("✓ No blocking issues")
        sys.exit(0)


if __name__ == "__main__":
    main()
