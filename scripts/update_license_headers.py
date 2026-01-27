# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import subprocess
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

# Skip auto-generated version files (hatch-vcs generates these at build time)
SKIP_FILES = frozenset(["_version.py"])

# File extensions to process for license headers
SUPPORTED_EXTENSIONS = frozenset([".py", ".sh"])

# Maximum number of lines to search for SPDX license header
MAX_HEADER_SEARCH_LINES = 10


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

    This function searches for SPDX tags in comment lines and extracts the complete
    header including any trailing blank line. It handles three states:
    1. Searching: Looking for SPDX in comment lines, skipping blank lines and non-SPDX comments
    2. Collecting: Found SPDX, gathering all SPDX comment lines
    3. Done: Collected header plus optional trailing blank line, or hit non-header content

    Args:
        lines: List of lines from the file (with line endings preserved)
        start_idx: Index to start looking for the header

    Returns:
        Tuple of (header_content, num_lines_consumed)
    """
    header_lines: list[str] = []
    end_idx = start_idx

    for i in range(start_idx, min(start_idx + MAX_HEADER_SEARCH_LINES, len(lines))):
        line = lines[i]
        stripped = line.rstrip("\n\r")

        # State: Collecting or searching for SPDX tags
        if re.search(r"SPDX", line, re.IGNORECASE):
            # Only treat as header if it's in a comment line
            if stripped.startswith("#"):
                header_lines.append(line)
                end_idx = i + 1
            else:
                # SPDX found in code/string literal, not a valid header
                break
        elif header_lines:
            # State: We've started collecting header lines
            # Check for trailing blank line after SPDX lines, then stop
            if stripped == "":
                header_lines.append(line)
                end_idx = i + 1
            break
        elif stripped == "" or stripped.startswith("#"):
            # State: Still searching - skip leading blank lines and non-SPDX comments
            continue
        else:
            # State: Hit actual code before finding any SPDX header
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
    """Format header with or without trailing blank line based on context.

    When content follows the header, preserve the double newline for separation.
    When the file is header-only, strip extra newlines to leave just one final newline.
    """
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
    if file_path.suffix not in SUPPORTED_EXTENSIONS:
        return False

    if file_path.name in SKIP_FILES:
        return False

    file_str = str(file_path)
    return not any(pattern in file_str for pattern in SKIP_PATTERNS)


def get_file_creation_year(file_path: Path) -> int | None:
    """Get the year when the file was first committed to git.

    Returns:
        The year of the first commit, or None if not tracked by git.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--diff-filter=A", "--follow", "--format=%ai", "--", str(file_path)],
            capture_output=True,
            text=True,
            cwd=file_path.parent,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Output format: "2025-01-15 10:30:00 -0800"
            # Take the last line (earliest commit due to --follow)
            lines = result.stdout.strip().split("\n")
            first_commit_date = lines[-1].split()[0]  # Get "YYYY-MM-DD"
            return int(first_commit_date.split("-")[0])
    except (subprocess.SubprocessError, ValueError, IndexError):
        pass
    return None


def get_copyright_year_string(file_path: Path, current_year: int) -> str:
    """Generate the copyright year string for a file.

    Returns:
        - Just the current year if file was created this year (e.g., "2026")
        - A range if file was created in a previous year (e.g., "2025-2026")
    """
    creation_year = get_file_creation_year(file_path)

    if creation_year is None or creation_year >= current_year:
        return str(current_year)

    return f"{creation_year}-{current_year}"


def generate_license_header(copyright_year: str) -> str:
    """Generate the license header with the given copyright year."""
    return (
        f"# SPDX-FileCopyrightText: Copyright (c) {copyright_year} "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )


def main(path: Path, check_only: bool = False) -> tuple[int, int, int, list[Path]]:
    """Process all supported files in a directory."""
    current_year = datetime.now().year

    processed = updated = skipped = 0
    files_needing_update: list[Path] = []

    for ext in SUPPORTED_EXTENSIONS:
        for file_path in path.glob(f"**/*{ext}"):
            if not file_path.is_file() or not should_process_file(file_path):
                continue

            processed += 1

            copyright_year = get_copyright_year_string(file_path, current_year)
            license_header = generate_license_header(copyright_year)

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
    parser = argparse.ArgumentParser(description="Add or check license headers in Python and shell files")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if all files have correct license headers without modifying files",
    )
    args = parser.parse_args()

    repo_path = Path(__file__).parent.parent
    all_files_needing_update: list[Path] = []
    total_processed = total_updated = total_skipped = 0

    # Process root-level directories
    for folder in ["scripts", "tests_e2e"]:
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

    # Process packages directory structure
    packages_path = repo_path / "packages"
    if packages_path.exists():
        for package_dir in sorted(packages_path.iterdir()):
            if not package_dir.is_dir():
                continue

            # Process src/ and tests/ within each package
            for subfolder in ["src", "tests"]:
                folder_path = package_dir / subfolder
                if not folder_path.exists():
                    continue

                action = "Checking" if args.check else "Processing"
                relative_path = folder_path.relative_to(repo_path)
                print(f"\n📂 {action} {relative_path}/")

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
