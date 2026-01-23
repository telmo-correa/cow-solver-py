#!/usr/bin/env python3
"""Extract test fixtures from Rust baseline solver test files.

This script parses Rust test files from the cow-services repository and
extracts the JSON fixtures (input auctions and expected solutions) for
use in Python benchmarks.

Usage:
    python scripts/extract_rust_fixtures.py

The script reads from:
    /Users/telmo/project/cow-services/crates/solvers/src/tests/cases/

And writes to:
    tests/fixtures/auctions/benchmark_rust/
"""

import json
import re
from pathlib import Path

RUST_TESTS_DIR = Path("/Users/telmo/project/cow-services/crates/solvers/src/tests/cases")
OUTPUT_DIR = Path("tests/fixtures/auctions/benchmark_rust")


def clean_rust_json(rust_json: str) -> str:
    """Clean Rust JSON syntax to valid JSON.

    Handles:
    - Line continuations (backslash at end of line)
    - Trailing commas
    - Rust comments
    - Rust string literals
    """
    # Remove line continuations (backslash + newline + whitespace)
    cleaned = re.sub(r"\\\n\s*", "", rust_json)

    # Remove Rust-style comments (// ...)
    cleaned = re.sub(r"//[^\n]*", "", cleaned)

    # Remove trailing commas before } or ]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

    return cleaned


def extract_json_block(text: str, start_pattern: str) -> str | None:
    """Extract a JSON block starting after a pattern.

    Uses bracket matching to find the complete JSON object.
    """
    match = re.search(start_pattern, text)
    if not match:
        return None

    # Find opening brace
    pos = match.end()
    while pos < len(text) and text[pos] != "{":
        pos += 1

    if pos >= len(text):
        return None

    # Match braces to find closing brace
    depth = 0
    start = pos
    for i in range(pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def extract_test_fixtures(content: str, test_name: str) -> dict | None:
    """Extract input and expected output from a test function.

    Returns dict with 'input' and 'expected' keys, or None if extraction fails.
    """
    # Find the test function
    pattern = rf"async fn {test_name}\s*\(\s*\)"
    match = re.search(pattern, content)
    if not match:
        return None

    # Get content from test function start to next function or end
    func_start = match.start()
    next_func = re.search(r"\n#\[tokio::test\]", content[func_start + 1 :])
    func_end = func_start + 1 + next_func.start() if next_func else len(content)

    func_content = content[func_start:func_end]

    # Extract input JSON from solve(json!({...}))
    input_json = extract_json_block(func_content, r"\.solve\(json!\(")
    if not input_json:
        return None

    # Extract expected JSON from assert_eq!(solution, json!({...}))
    expected_json = extract_json_block(func_content, r"assert_eq!\(\s*solution,\s*json!\(")
    if not expected_json:
        return None

    # Clean and parse JSON
    try:
        input_cleaned = clean_rust_json(input_json)
        expected_cleaned = clean_rust_json(expected_json)

        input_data = json.loads(input_cleaned)
        expected_data = json.loads(expected_cleaned)

        return {"input": input_data, "expected": expected_data}
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse error in {test_name}: {e}")
        return None


def find_test_functions(content: str) -> list[str]:
    """Find all test function names in a Rust file."""
    matches = re.findall(r"#\[tokio::test\]\s*async fn (\w+)\s*\(\s*\)", content)
    return matches


def process_rust_file(filepath: Path) -> list[tuple[str, dict]]:
    """Process a Rust test file and extract all fixtures.

    Returns list of (test_name, fixture_data) tuples.
    """
    content = filepath.read_text()
    test_names = find_test_functions(content)

    fixtures = []
    for name in test_names:
        fixture = extract_test_fixtures(content, name)
        if fixture:
            fixtures.append((name, fixture))

    return fixtures


def main():
    """Extract all Rust test fixtures."""
    print(f"Extracting fixtures from: {RUST_TESTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all Rust test files
    test_files = sorted(RUST_TESTS_DIR.glob("*.rs"))
    if not test_files:
        print(f"No .rs files found in {RUST_TESTS_DIR}")
        return

    total_fixtures = 0
    all_fixtures: list[tuple[str, str, dict]] = []  # (file, test, fixture)

    for filepath in test_files:
        if filepath.name == "mod.rs":
            continue

        print(f"\nProcessing {filepath.name}...")
        fixtures = process_rust_file(filepath)

        for test_name, fixture in fixtures:
            # Generate output filename
            base_name = filepath.stem
            if test_name == "test":
                # For files with single "test" function, use file name
                output_name = f"{base_name}.json"
            else:
                output_name = f"{base_name}_{test_name}.json"

            all_fixtures.append((filepath.name, test_name, fixture))

            # Write fixture file (input only, expected in separate file)
            input_path = OUTPUT_DIR / output_name
            input_path.write_text(json.dumps(fixture["input"], indent=2) + "\n")

            # Write expected output
            expected_path = OUTPUT_DIR / output_name.replace(".json", "_expected.json")
            expected_path.write_text(json.dumps(fixture["expected"], indent=2) + "\n")

            print(f"  ✓ {test_name} -> {output_name}")
            total_fixtures += 1

    print(f"\n{'=' * 60}")
    print(f"Extracted {total_fixtures} fixtures from {len(test_files) - 1} files")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create an index file
    index = {
        "description": "Rust baseline solver test fixtures",
        "source": str(RUST_TESTS_DIR),
        "fixtures": [
            {
                "file": f,
                "test": t,
                "input": f"{Path(f).stem}_{t}.json" if t != "test" else f"{Path(f).stem}.json",
            }
            for f, t, _ in all_fixtures
        ],
    }
    index_path = OUTPUT_DIR / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Created index file: {index_path}")


if __name__ == "__main__":
    main()
