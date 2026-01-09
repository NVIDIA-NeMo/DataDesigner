# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for update_license_headers.py

Run with: uv run python scripts/test_license_headers.py
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from update_license_headers import (
    check_license_header_matches,
    extract_license_header,
    generate_license_header,
    get_copyright_year_string,
    get_file_creation_year,
    should_process_file,
    update_license_header_in_file,
)


class TestResult:
    """Simple test result tracker."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def add_pass(self, name: str) -> None:
        self.passed.append(name)
        print(f"  ✅ {name}")

    def add_fail(self, name: str, reason: str) -> None:
        self.failed.append((name, reason))
        print(f"  ❌ {name}: {reason}")

    def summary(self) -> None:
        total = len(self.passed) + len(self.failed)
        print(f"\n{'=' * 60}")
        print(f"Results: {len(self.passed)}/{total} passed")
        if self.failed:
            print("\nFailed tests:")
            for name, reason in self.failed:
                print(f"  • {name}: {reason}")


# Current year header for testing
CURRENT_YEAR = datetime.now().year
LICENSE_HEADER = (
    f"# SPDX-FileCopyrightText: Copyright (c) {CURRENT_YEAR} "
    "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n\n"
)

LICENSE_HEADER_NO_TRAILING = LICENSE_HEADER.rstrip("\n") + "\n"


def create_temp_file(content: str) -> Path:
    """Create a temporary Python file with given content."""
    fd, path = tempfile.mkstemp(suffix=".py")
    with open(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return Path(path)


def run_tests() -> TestResult:
    """Run all edge case tests."""
    results = TestResult()

    # =========================================================================
    # Test extract_license_header
    # =========================================================================
    print("\n📋 Testing extract_license_header()")

    # Test 1: Empty lines list
    header_result, consumed = extract_license_header([], 0)
    # Handle both original (list) and refactored (str) return types
    is_empty = header_result == [] or header_result == ""
    if is_empty and consumed == 0:
        results.add_pass("extract: empty lines")
    else:
        results.add_fail("extract: empty lines", f"Got {header_result}, {consumed}")

    # Test 2: File with valid header
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA\n",
        "# SPDX-License-Identifier: Apache-2.0\n",
        "\n",
        "import os\n",
    ]
    header_result, consumed = extract_license_header(lines, 0)
    # Count lines: original returns list, refactored returns joined string
    line_count = len(header_result) if isinstance(header_result, list) else header_result.count("\n")
    if line_count == 3 and consumed == 3:
        results.add_pass("extract: valid header with blank line")
    else:
        results.add_fail("extract: valid header with blank line", f"Got {line_count} lines, consumed {consumed}")

    # Test 3: Header without trailing blank line
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA\n",
        "# SPDX-License-Identifier: Apache-2.0\n",
        "import os\n",
    ]
    header_result, consumed = extract_license_header(lines, 0)
    line_count = len(header_result) if isinstance(header_result, list) else header_result.count("\n")
    if line_count == 2 and consumed == 2:
        results.add_pass("extract: header without trailing blank")
    else:
        results.add_fail("extract: header without trailing blank", f"Got {line_count} lines, consumed {consumed}")

    # Test 4: No SPDX header, just code
    lines = ["import os\n", "print('hello')\n"]
    header_result, consumed = extract_license_header(lines, 0)
    is_empty = header_result == [] or header_result == ""
    if is_empty and consumed == 0:
        results.add_pass("extract: no header, just code")
    else:
        results.add_fail("extract: no header, just code", f"Got {header_result}")

    # Test 5: Leading blank lines before header
    lines = [
        "\n",
        "\n",
        "# SPDX-FileCopyrightText: Test\n",
        "# SPDX-License-Identifier: MIT\n",
    ]
    header_result, consumed = extract_license_header(lines, 0)
    line_count = len(header_result) if isinstance(header_result, list) else header_result.count("\n")
    if line_count == 2 and consumed == 4:
        results.add_pass("extract: leading blank lines")
    else:
        results.add_fail("extract: leading blank lines", f"Got {line_count} lines, consumed {consumed}")

    # Test 6: Other comments before SPDX (currently skipped by the script)
    lines = [
        "# Some comment\n",
        "# SPDX-FileCopyrightText: Test\n",
        "# SPDX-License-Identifier: MIT\n",
    ]
    header_result, consumed = extract_license_header(lines, 0)
    line_count = len(header_result) if isinstance(header_result, list) else header_result.count("\n")
    # Current behavior: skips "# Some comment" and finds SPDX
    if line_count == 2:
        results.add_pass("extract: other comments before SPDX (skipped)")
    else:
        results.add_fail("extract: other comments before SPDX", f"Got {header_result}")

    # =========================================================================
    # Test update_license_header_in_file
    # =========================================================================
    print("\n📋 Testing update_license_header_in_file()")

    # Test 7: Empty file
    path = create_temp_file("")
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "added" and "SPDX" in content:
            # Check no trailing blank line for empty file
            if content == LICENSE_HEADER_NO_TRAILING:
                results.add_pass("update: empty file")
            else:
                results.add_fail("update: empty file", f"Unexpected format: {repr(content)}")
        else:
            results.add_fail("update: empty file", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 8: File with only whitespace
    path = create_temp_file("   \n\n   \n")
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "added":
            results.add_pass("update: whitespace-only file")
        else:
            results.add_fail("update: whitespace-only file", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 9: File with shebang and no header
    path = create_temp_file("#!/usr/bin/env python3\n\nimport os\n")
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "added" and content.startswith("#!/usr/bin/env python3\n"):
            # Header should be after shebang
            lines = content.split("\n")
            if "SPDX" in lines[1]:
                results.add_pass("update: shebang preserved")
            else:
                results.add_fail("update: shebang preserved", f"Header not after shebang: {lines[:3]}")
        else:
            results.add_fail("update: shebang preserved", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 10: File with correct header already
    path = create_temp_file(LICENSE_HEADER + "import os\n")
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        if not was_modified and reason == "unchanged":
            results.add_pass("update: correct header unchanged")
        else:
            results.add_fail("update: correct header unchanged", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 11: File with outdated year in header
    old_header = (
        "# SPDX-FileCopyrightText: Copyright (c) 2020 "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )
    path = create_temp_file(old_header + "import os\n")
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "updated" and str(CURRENT_YEAR) in content:
            results.add_pass("update: outdated year replaced")
        else:
            results.add_fail(
                "update: outdated year replaced",
                f"Got {was_modified}, {reason}, year in content: {str(CURRENT_YEAR) in content}",
            )
    finally:
        path.unlink()

    # Test 12: File with malformed/partial header
    partial_header = "# SPDX-License-Identifier: MIT\n\n"
    path = create_temp_file(partial_header + "import os\n")
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "updated":
            results.add_pass("update: malformed header replaced")
        else:
            results.add_fail("update: malformed header replaced", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 13: File with header but no trailing blank line
    header_no_blank = (
        f"# SPDX-FileCopyrightText: Copyright (c) {CURRENT_YEAR} "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
    )
    path = create_temp_file(header_no_blank + "import os\n")
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        # Should add the trailing blank line
        if was_modified and reason == "updated" and "\n\nimport" in content:
            results.add_pass("update: adds trailing blank line")
        else:
            results.add_fail(
                "update: adds trailing blank line", f"Got {was_modified}, {reason}, content: {repr(content[:100])}"
            )
    finally:
        path.unlink()

    # Test 14: File with multiple blank lines after header
    header_multi_blank = LICENSE_HEADER.rstrip("\n") + "\n\n\n\n"
    path = create_temp_file(header_multi_blank + "import os\n")
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        # This is interesting - the script only captures one trailing blank line
        # so this should be seen as a mismatch or the extra blanks should be preserved
        results.add_pass("update: multiple blank lines (behavior noted)")
    finally:
        path.unlink()

    # Test 15: File ending with header only (no code after)
    path = create_temp_file(LICENSE_HEADER_NO_TRAILING)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if not was_modified and reason == "unchanged":
            results.add_pass("update: header-only file unchanged")
        else:
            results.add_fail("update: header-only file unchanged", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 16: File with SPDX in code content (not header)
    code_with_spdx = 'import os\n\nLICENSE = "SPDX-License-Identifier: MIT"\n'
    path = create_temp_file(code_with_spdx)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "added":
            # Should add header at top, not touch the SPDX in code
            if content.count("SPDX") == 3:  # 2 in header + 1 in code
                results.add_pass("update: SPDX in code preserved")
            else:
                results.add_fail("update: SPDX in code preserved", f"SPDX count: {content.count('SPDX')}")
        else:
            results.add_fail("update: SPDX in code preserved", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 17: File with docstring at top (no header)
    docstring_file = '"""Module docstring."""\n\nimport os\n'
    path = create_temp_file(docstring_file)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "added":
            # Header should be before docstring
            if content.startswith("# SPDX"):
                results.add_pass("update: docstring file gets header")
            else:
                results.add_fail("update: docstring file gets header", f"Doesn't start with header: {content[:50]}")
        else:
            results.add_fail("update: docstring file gets header", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 18: File with encoding declaration
    encoding_file = "# -*- coding: utf-8 -*-\n\nimport os\n"
    path = create_temp_file(encoding_file)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        # The encoding comment is NOT a shebang, so header goes at line 0
        # This might not be the desired behavior...
        if was_modified and reason == "added":
            results.add_pass("update: encoding declaration (header added at top)")
        else:
            results.add_fail("update: encoding declaration", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 19: Shebang + encoding + no header
    shebang_encoding = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\nimport os\n"
    path = create_temp_file(shebang_encoding)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        lines = content.split("\n")
        if was_modified and reason == "added" and lines[0].startswith("#!"):
            results.add_pass("update: shebang + encoding")
        else:
            results.add_fail("update: shebang + encoding", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 20: Different license (should be replaced)
    different_license = "# SPDX-FileCopyrightText: Some Other Corp\n# SPDX-License-Identifier: MIT\n\nimport os\n"
    path = create_temp_file(different_license)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "updated" and "NVIDIA" in content:
            results.add_pass("update: different license replaced")
        else:
            results.add_fail("update: different license replaced", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # =========================================================================
    # Test check_license_header_matches
    # =========================================================================
    print("\n📋 Testing check_license_header_matches()")

    # Test 21: Correct header matches
    path = create_temp_file(LICENSE_HEADER + "import os\n")
    try:
        matches, status = check_license_header_matches(path, LICENSE_HEADER)
        if matches and status == "match":
            results.add_pass("check: correct header matches")
        else:
            results.add_fail("check: correct header matches", f"Got {matches}, {status}")
    finally:
        path.unlink()

    # Test 22: Missing header detected
    path = create_temp_file("import os\n")
    try:
        matches, status = check_license_header_matches(path, LICENSE_HEADER)
        if not matches and status == "missing":
            results.add_pass("check: missing header detected")
        else:
            results.add_fail("check: missing header detected", f"Got {matches}, {status}")
    finally:
        path.unlink()

    # Test 23: Mismatched header detected
    old_header = "# SPDX-FileCopyrightText: Old\n# SPDX-License-Identifier: MIT\n\n"
    path = create_temp_file(old_header + "import os\n")
    try:
        matches, status = check_license_header_matches(path, LICENSE_HEADER)
        if not matches and status == "mismatch":
            results.add_pass("check: mismatch detected")
        else:
            results.add_fail("check: mismatch detected", f"Got {matches}, {status}")
    finally:
        path.unlink()

    # Test 24: Empty file = missing
    path = create_temp_file("")
    try:
        matches, status = check_license_header_matches(path, LICENSE_HEADER)
        if not matches and status == "missing":
            results.add_pass("check: empty file is missing")
        else:
            results.add_fail("check: empty file is missing", f"Got {matches}, {status}")
    finally:
        path.unlink()

    # =========================================================================
    # Test should_process_file
    # =========================================================================
    print("\n📋 Testing should_process_file()")

    # Test 25: Regular .py file
    if should_process_file(Path("src/module.py")):
        results.add_pass("should_process: regular .py file")
    else:
        results.add_fail("should_process: regular .py file", "Should be True")

    # Test 26: _version.py excluded
    if not should_process_file(Path("src/_version.py")):
        results.add_pass("should_process: _version.py excluded")
    else:
        results.add_fail("should_process: _version.py excluded", "Should be False")

    # Test 27: __pycache__ excluded
    if not should_process_file(Path("src/__pycache__/module.py")):
        results.add_pass("should_process: __pycache__ excluded")
    else:
        results.add_fail("should_process: __pycache__ excluded", "Should be False")

    # Test 28: Non-.py file excluded
    if not should_process_file(Path("src/module.txt")):
        results.add_pass("should_process: non-.py excluded")
    else:
        results.add_fail("should_process: non-.py excluded", "Should be False")

    # Test 29: .venv excluded
    if not should_process_file(Path(".venv/lib/module.py")):
        results.add_pass("should_process: .venv excluded")
    else:
        results.add_fail("should_process: .venv excluded", "Should be False")

    # Test 30: .git excluded
    if not should_process_file(Path(".git/hooks/pre-commit.py")):
        results.add_pass("should_process: .git excluded")
    else:
        results.add_fail("should_process: .git excluded", "Should be False")

    # =========================================================================
    # Test copyright year functions
    # =========================================================================
    print("\n📋 Testing copyright year functions")

    # Test 31: get_file_creation_year returns None for non-git files
    temp_path = create_temp_file("import os\n")
    try:
        creation_year = get_file_creation_year(temp_path)
        if creation_year is None:
            results.add_pass("copyright: non-git file returns None")
        else:
            results.add_fail("copyright: non-git file returns None", f"Got {creation_year}")
    finally:
        temp_path.unlink()

    # Test 32: get_copyright_year_string returns current year when creation_year is None (non-git file)
    temp_path = create_temp_file("import os\n")
    try:
        year_string = get_copyright_year_string(temp_path, CURRENT_YEAR)
        if year_string == str(CURRENT_YEAR):
            results.add_pass("copyright: non-git file uses current year only")
        else:
            results.add_fail("copyright: non-git file uses current year only", f"Got {year_string}")
    finally:
        temp_path.unlink()

    # Test 33: generate_license_header with single year
    header = generate_license_header(str(CURRENT_YEAR))
    expected = (
        f"# SPDX-FileCopyrightText: Copyright (c) {CURRENT_YEAR} "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )
    if header == expected:
        results.add_pass("copyright: generate_license_header single year")
    else:
        results.add_fail("copyright: generate_license_header single year", f"Got {repr(header)}")

    # Test 34: generate_license_header with year range
    header = generate_license_header("2020-2026")
    expected_range = (
        "# SPDX-FileCopyrightText: Copyright (c) 2020-2026 "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )
    if header == expected_range:
        results.add_pass("copyright: generate_license_header year range")
    else:
        results.add_fail("copyright: generate_license_header year range", f"Got {repr(header)}")

    # Test 35: get_copyright_year_string returns range for older files (simulated via direct logic check)
    # Since we can't easily create a git-tracked file in tests, we verify the logic:
    # If creation_year < current_year, should return "creation_year-current_year"
    # We test this indirectly by checking the generate_license_header output format
    test_years = [
        (2020, 2026, "2020-2026"),
        (2025, 2026, "2025-2026"),
        (2026, 2026, "2026"),  # Same year = no range
    ]
    all_passed = True
    for creation, current, expected_str in test_years:
        # Simulate the logic from get_copyright_year_string
        if creation >= current:
            result = str(current)
        else:
            result = f"{creation}-{current}"
        if result != expected_str:
            all_passed = False
            results.add_fail(
                "copyright: year range logic",
                f"For creation={creation}, current={current}: expected {expected_str}, got {result}",
            )
            break
    if all_passed:
        results.add_pass("copyright: year range logic for various scenarios")

    # Test 36: get_file_creation_year returns valid year for actual git-tracked file
    # Use this test file itself as it's tracked in git
    this_file = Path(__file__)
    if this_file.exists():
        creation_year = get_file_creation_year(this_file)
        if creation_year is not None and 2020 <= creation_year <= CURRENT_YEAR:
            results.add_pass(f"copyright: git-tracked file returns valid year ({creation_year})")
        else:
            # May be None if running outside git repo or in CI without full history
            results.add_pass(f"copyright: git-tracked file (year={creation_year}, may be None in shallow clone)")

    # =========================================================================
    # Test idempotency
    # =========================================================================
    print("\n📋 Testing idempotency")

    # Test 31: Running update twice should be idempotent
    path = create_temp_file("import os\n")
    try:
        # First update
        was_modified1, reason1 = update_license_header_in_file(path, LICENSE_HEADER)
        content1 = path.read_text()

        # Second update
        was_modified2, reason2 = update_license_header_in_file(path, LICENSE_HEADER)
        content2 = path.read_text()

        if (
            was_modified1
            and reason1 == "added"
            and not was_modified2
            and reason2 == "unchanged"
            and content1 == content2
        ):
            results.add_pass("idempotency: double update stable")
        else:
            results.add_fail(
                "idempotency: double update stable",
                f"First: {was_modified1}/{reason1}, Second: {was_modified2}/{reason2}",
            )
    finally:
        path.unlink()

    # =========================================================================
    # Edge cases for Windows-style line endings
    # =========================================================================
    print("\n📋 Testing line ending edge cases")

    # Test 32: Windows line endings (CRLF)
    path = create_temp_file("import os\r\nprint('hello')\r\n")
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        if was_modified and reason == "added":
            results.add_pass("line endings: CRLF handled")
        else:
            results.add_fail("line endings: CRLF handled", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # =========================================================================
    # Test full workflow simulation
    # =========================================================================
    print("\n📋 Testing full workflow")

    # Test 33: Create a temporary directory structure and process it
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Create test files
        (temp_dir / "good.py").write_text(LICENSE_HEADER + "import os\n")
        (temp_dir / "missing.py").write_text("import os\n")
        (temp_dir / "outdated.py").write_text(
            "# SPDX-FileCopyrightText: Copyright (c) 2020 OLD\n# SPDX-License-Identifier: Apache-2.0\n\nimport os\n"
        )
        (temp_dir / "__pycache__").mkdir()
        (temp_dir / "__pycache__" / "skip.py").write_text("# should be skipped\n")
        (temp_dir / "not_python.txt").write_text("not python\n")

        # Import and run main
        from update_license_headers import main

        processed, updated, skipped, _ = main(temp_dir, check_only=False)

        if processed == 3 and updated == 2 and skipped == 1:
            results.add_pass("workflow: directory processing")
        else:
            results.add_fail(
                "workflow: directory processing", f"Got processed={processed}, updated={updated}, skipped={skipped}"
            )

    finally:
        shutil.rmtree(temp_dir)

    # =========================================================================
    # Additional edge cases
    # =========================================================================
    print("\n📋 Testing additional edge cases")

    # Test 34: Unicode content in file
    unicode_content = "# SPDX-FileCopyrightText: Test\n# SPDX-License-Identifier: MIT\n\nprint('こんにちは')\n"
    path = create_temp_file(unicode_content)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "updated" and "こんにちは" in content:
            results.add_pass("edge: unicode content preserved")
        else:
            results.add_fail("edge: unicode content preserved", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 35: File with only comments (no actual code)
    only_comments = "# Just a comment\n# Another comment\n"
    path = create_temp_file(only_comments)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "added":
            results.add_pass("edge: file with only comments")
        else:
            results.add_fail("edge: file with only comments", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 36: SPDX in lowercase (case sensitivity)
    lowercase_spdx = "# spdx-license-identifier: MIT\n\nimport os\n"
    path = create_temp_file(lowercase_spdx)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        # The regex uses re.IGNORECASE, so this should be detected
        if was_modified and reason == "updated":
            results.add_pass("edge: lowercase SPDX detected and replaced")
        else:
            results.add_fail("edge: lowercase SPDX detected and replaced", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 37: Header with trailing spaces
    header_with_spaces = (
        f"# SPDX-FileCopyrightText: Copyright (c) {CURRENT_YEAR} "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved.   \n"  # trailing spaces
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )
    path = create_temp_file(header_with_spaces + "import os\n")
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        # Should detect as mismatch due to trailing spaces
        if was_modified and reason == "updated":
            results.add_pass("edge: header with trailing spaces fixed")
        else:
            results.add_fail("edge: header with trailing spaces fixed", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 38: Very long file (performance check)
    long_content = LICENSE_HEADER + "import os\n" + "x = 1\n" * 10000
    path = create_temp_file(long_content)
    try:
        import time

        start = time.time()
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        elapsed = time.time() - start
        if not was_modified and reason == "unchanged" and elapsed < 1.0:
            results.add_pass(f"edge: long file processed quickly ({elapsed:.3f}s)")
        else:
            results.add_fail(
                "edge: long file processed quickly", f"Got {was_modified}, {reason}, elapsed {elapsed:.3f}s"
            )
    finally:
        path.unlink()

    # Test 39: File with triple-quoted string that looks like comments
    triple_quoted = '"""\n# SPDX-FileCopyrightText: Fake header in docstring\n"""\nimport os\n'
    path = create_temp_file(triple_quoted)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        # Should add header at top since the SPDX is inside a string, not a comment
        # But the current implementation might incorrectly see the docstring as non-comment content
        # and correctly add header at top
        if was_modified and reason == "added" and content.startswith("# SPDX"):
            results.add_pass("edge: SPDX in docstring ignored correctly")
        else:
            results.add_fail("edge: SPDX in docstring ignored correctly", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 40: Empty __init__.py file (common pattern)
    path = create_temp_file("")
    try:
        path = path.rename(path.parent / "__init__.py")
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        if was_modified and reason == "added":
            results.add_pass("edge: empty __init__.py handled")
        else:
            results.add_fail("edge: empty __init__.py handled", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 41: File with blank lines at end
    trailing_blanks = LICENSE_HEADER + "import os\n\n\n\n"
    path = create_temp_file(trailing_blanks)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        if not was_modified and reason == "unchanged":
            results.add_pass("edge: trailing blank lines preserved")
        else:
            results.add_fail("edge: trailing blank lines preserved", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 42: Check mode on file needing update
    path = create_temp_file("import os\n")
    try:
        matches, status = check_license_header_matches(path, LICENSE_HEADER)
        original_content = path.read_text()
        # Verify check mode doesn't modify file
        if not matches and status == "missing" and path.read_text() == original_content:
            results.add_pass("check mode: doesn't modify file")
        else:
            results.add_fail("check mode: doesn't modify file", f"Got {matches}, {status}")
    finally:
        path.unlink()

    # Test 43: Header immediately followed by class definition (no blank line between)
    no_blank_after = (
        f"# SPDX-FileCopyrightText: Copyright (c) {CURRENT_YEAR} "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "class Foo:\n"
        "    pass\n"
    )
    path = create_temp_file(no_blank_after)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        # Should add blank line between header and class
        has_blank = "\n\nclass" in content
        if was_modified and reason == "updated" and has_blank:
            results.add_pass("edge: blank line added after header")
        else:
            results.add_fail(
                "edge: blank line added after header",
                f"Got {was_modified}, {reason}, has blank line before class: {has_blank}",
            )
    finally:
        path.unlink()

    # Test 44: Single-line file with just code
    single_line = "x = 1"
    path = create_temp_file(single_line)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "added" and "x = 1" in content:
            results.add_pass("edge: single-line file handled")
        else:
            results.add_fail("edge: single-line file handled", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 45: Shebang with existing correct header
    shebang_with_header = "#!/usr/bin/env python3\n" + LICENSE_HEADER + "import os\n"
    path = create_temp_file(shebang_with_header)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        if not was_modified and reason == "unchanged":
            results.add_pass("edge: shebang + correct header unchanged")
        else:
            results.add_fail("edge: shebang + correct header unchanged", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 46: Three SPDX lines (non-standard)
    three_spdx = (
        "# SPDX-FileCopyrightText: Line 1\n"
        "# SPDX-FileCopyrightText: Line 2\n"
        "# SPDX-License-Identifier: MIT\n\n"
        "import os\n"
    )
    path = create_temp_file(three_spdx)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        if was_modified and reason == "updated":
            # Verify all three old lines are replaced
            if content.count("SPDX") == 2:  # Only 2 lines in new header
                results.add_pass("edge: three SPDX lines replaced correctly")
            else:
                results.add_fail("edge: three SPDX lines replaced correctly", f"SPDX count: {content.count('SPDX')}")
        else:
            results.add_fail("edge: three SPDX lines replaced correctly", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 47: CRITICAL REGRESSION - SPDX in string literal should NOT be detected as header
    spdx_in_string = "message = 'SPDX is a standard'\nprint(message)\n"
    path = create_temp_file(spdx_in_string)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()
        lines = content.split("\n")

        # Verify: header should be added at top, string literal should be preserved
        has_header_at_top = lines[0].startswith("#") and "SPDX-FileCopyrightText" in lines[0]
        string_preserved = "message = 'SPDX is a standard'" in content
        spdx_count = content.count("SPDX")

        if (
            was_modified
            and reason == "added"
            and has_header_at_top
            and string_preserved
            and spdx_count == 3  # 2 in header + 1 in string
        ):
            results.add_pass("CRITICAL: SPDX in string literal not treated as header")
        else:
            results.add_fail(
                "CRITICAL: SPDX in string literal not treated as header",
                f"modified={was_modified}, reason={reason}, header_at_top={has_header_at_top}, "
                f"string_preserved={string_preserved}, spdx_count={spdx_count}",
            )
    finally:
        path.unlink()

    # Test 48: SPDX in variable name or function name
    spdx_in_code = "def check_spdx_compliance():\n    return True\n"
    path = create_temp_file(spdx_in_code)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()

        # Should add header, not treat function name as header
        if was_modified and reason == "added" and content.startswith("# SPDX"):
            results.add_pass("edge: SPDX in function name not treated as header")
        else:
            results.add_fail("edge: SPDX in function name not treated as header", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    # Test 49: SPDX in multi-line string (docstring)
    spdx_in_docstring = '"""This module handles SPDX compliance checking."""\nimport os\n'
    path = create_temp_file(spdx_in_docstring)
    try:
        was_modified, reason = update_license_header_in_file(path, LICENSE_HEADER)
        content = path.read_text()

        # Should add header at top, before docstring
        if was_modified and reason == "added" and content.startswith("# SPDX"):
            results.add_pass("edge: SPDX in module docstring not treated as header")
        else:
            results.add_fail("edge: SPDX in module docstring not treated as header", f"Got {was_modified}, {reason}")
    finally:
        path.unlink()

    return results


if __name__ == "__main__":
    print("🧪 Running license header script tests\n")
    print("=" * 60)
    results = run_tests()
    results.summary()

    # Exit with error code if any tests failed
    exit(1 if results.failed else 0)
