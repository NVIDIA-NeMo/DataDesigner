# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SKIP_PATTERNS = frozenset(
    [
        "__pycache__",
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".egg-info",
        ".git",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
    ]
)

SKIP_FILES = frozenset(["_version.py"])


@dataclass
class HeaderAnalysis:
    """Result of analyzing a file's license header."""

    lines: list[str]  # All file lines
    header_start: int  # Where header starts (or should be inserted)
    existing_header: str  # Existing header content (empty if none)
    header_end: int  # Line index after header ends
    has_content_after: bool  # Whether there's code after header position

    @property
    def has_header(self) -> bool:
        return bool(self.existing_header)


def extract_license_header(lines: list[str], start_idx: int) -> tuple[str, int]:
    """Extract existing SPDX license header from file lines.

    Args:
        lines: List of lines from the file (with line endings preserved)
        start_idx: Index to start looking for the header

    Returns:
        Tuple of (header_content, num_lines_consumed)
    """
    header_lines: list[str] = []
    end_idx = start_idx

    for i in range(start_idx, min(start_idx + 10, len(lines))):
        line = lines[i]
        stripped = line.rstrip("\n\r")

        if re.search(r"SPDX", line, re.IGNORECASE):
            header_lines.append(line)
            end_idx = i + 1
        elif header_lines:
            # We've started collecting; check for trailing blank and stop
            if stripped == "":
                header_lines.append(line)
                end_idx = i + 1
            break
        elif stripped == "" or stripped.startswith("#"):
            # Skip leading blank lines and non-SPDX comments
            continue
        else:
            # Hit code before finding SPDX header
            break

    return "".join(header_lines), end_idx - start_idx


def _analyze_file_header(lines: list[str]) -> HeaderAnalysis:
    """Analyze a file to find its current header state."""
    if not lines:
        return HeaderAnalysis(
            lines=lines,
            header_start=0,
            existing_header="",
            header_end=0,
            has_content_after=False,
        )

    # Header goes after shebang if present
    insert_pos = 1 if lines[0].startswith("#!") else 0

    # Skip blank lines to find where header actually starts
    header_search_start = insert_pos
    while header_search_start < len(lines) and lines[header_search_start].strip() == "":
        header_search_start += 1

    # Extract existing header
    existing_header, num_lines = extract_license_header(lines, header_search_start)
    header_end = header_search_start + num_lines

    # Check if there's content after the header position
    remaining = lines[header_end:] if existing_header else lines[insert_pos:]
    has_content = any(line.strip() for line in remaining)

    return HeaderAnalysis(
        lines=lines,
        header_start=header_search_start if existing_header else insert_pos,
        existing_header=existing_header,
        header_end=header_end,
        has_content_after=has_content,
    )


def _format_header(license_header: str, has_content_after: bool) -> str:
    """Format header with or without trailing blank line based on context."""
    if has_content_after:
        return license_header
    return license_header.rstrip("\n") + "\n"


def update_license_header_in_file(file_path: Path, license_header: str) -> tuple[bool, str]:
    """Update license header in a single file.

    Returns:
        Tuple of (was_modified, reason) where reason is:
        - "added" - header was added (none existed)
        - "updated" - header was replaced (different existed)
        - "unchanged" - header unchanged (identical existed)
        - "error" - an error occurred
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)

        if not lines:
            file_path.write_text(_format_header(license_header, False), encoding="utf-8")
            return True, "added"

        analysis = _analyze_file_header(lines)
        expected_header = _format_header(license_header, analysis.has_content_after)

        if analysis.has_header:
            if analysis.existing_header == expected_header:
                return False, "unchanged"

            # Replace existing header
            new_lines = lines[: analysis.header_start]
            new_lines.append(expected_header)
            new_lines.extend(lines[analysis.header_end :])
            file_path.write_text("".join(new_lines), encoding="utf-8")
            return True, "updated"

        # No existing header - add one
        if analysis.has_content_after:
            lines.insert(analysis.header_start, expected_header)
            file_path.write_text("".join(lines), encoding="utf-8")
        else:
            file_path.write_text(expected_header, encoding="utf-8")
        return True, "added"

    except (UnicodeDecodeError, PermissionError) as e:
        print(f"  ⏭️  Skipped {file_path} ({e})")
        return False, "error"


def check_license_header_matches(file_path: Path, license_header: str) -> tuple[bool, str]:
    """Check if file has the expected license header.

    Returns:
        Tuple of (header_matches, status) where status is:
        - "match" - header exists and matches
        - "missing" - no header found
        - "mismatch" - header exists but differs
        - "error" - couldn't read file
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)

        if not lines:
            return False, "missing"

        analysis = _analyze_file_header(lines)

        if not analysis.has_header:
            return False, "missing"

        expected_header = _format_header(license_header, analysis.has_content_after)
        if analysis.existing_header == expected_header:
            return True, "match"

        return False, "mismatch"

    except (UnicodeDecodeError, PermissionError):
        return False, "error"


def should_process_file(file_path: Path) -> bool:
    """Determine if a file should be processed for license headers."""
    if file_path.suffix != ".py":
        return False

    if file_path.name in SKIP_FILES:
        return False

    file_str = str(file_path)
    return not any(pattern in file_str for pattern in SKIP_PATTERNS)


def main(path: Path, check_only: bool = False) -> tuple[int, int, int, list[Path]]:
    """Process all Python files in a directory."""
    current_year = datetime.now().year
    license_header = (
        f"# SPDX-FileCopyrightText: Copyright (c) {current_year} "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )

    processed = updated = skipped = 0
    files_needing_update: list[Path] = []

    for file_path in path.glob("**/*.py"):
        if not file_path.is_file() or not should_process_file(file_path):
            continue

        processed += 1

        if check_only:
            matches, _ = check_license_header_matches(file_path, license_header)
            if matches:
                skipped += 1
            else:
                files_needing_update.append(file_path)
                updated += 1
        else:
            was_modified, reason = update_license_header_in_file(file_path, license_header)
            if was_modified:
                action = "Added header to" if reason == "added" else "Updated header in"
                print(f"  {'✏️' if reason == 'added' else '🔄'} {action} {file_path}")
                updated += 1
            else:
                skipped += 1

    return processed, updated, skipped, files_needing_update


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add or check license headers in Python files")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if all files have correct license headers without modifying files",
    )
    args = parser.parse_args()

    repo_path = Path(__file__).parent.parent
    all_files_needing_update: list[Path] = []
    total_processed = total_updated = total_skipped = 0

    for folder in ["src", "tests", "scripts", "e2e_tests"]:
        folder_path = repo_path / folder
        if not folder_path.exists():
            continue

        action = "Checking" if args.check else "Processing"
        print(f"\n📂 {action} {folder}/")

        processed, updated, skipped, files_needing_update = main(folder_path, check_only=args.check)

        total_processed += processed
        total_updated += updated
        total_skipped += skipped
        all_files_needing_update.extend(files_needing_update)

        if args.check:
            print(f"   ❌ Need update: {updated}")
            print(f"   ✅ Up to date: {skipped}")
        else:
            print(f"   ✏️  Updated: {updated}")
            print(f"   ⏭️  Skipped: {skipped}")

    print("\n" + "=" * 80)
    print(f"📊 Summary: {total_processed} files processed")

    if args.check:
        print(f"   ❌ Need update: {total_updated}")
        print(f"   ✅ Up to date: {total_skipped}")

        if all_files_needing_update:
            print(f"\n❌ {len(all_files_needing_update)} file(s) need license header updates:")
            for file_path in all_files_needing_update:
                print(f"   • {file_path}")
            print("💡 Run 'make update-license-headers' to fix")
            sys.exit(1)
        else:
            print("\n🎉 All files have correct license headers!")
    else:
        print(f"   ✏️  Updated: {total_updated}")
        print(f"   ⏭️  Skipped: {total_skipped}")
        print("\n✅ Done!")

    sys.exit(0)
