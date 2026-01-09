# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


def extract_license_header(lines: list[str], start_idx: int) -> tuple[list[str], int]:
    """Extract existing SPDX license header lines from file.

    Args:
        lines: List of lines from the file (with line endings preserved)
        start_idx: Index to start looking for the header

    Returns:
        Tuple of (header_lines, num_lines_consumed) where header_lines includes
        the SPDX comment lines and any trailing blank line.
    """
    header_lines: list[str] = []
    consumed = 0

    for i in range(start_idx, min(start_idx + 10, len(lines))):
        line = lines[i]
        stripped = line.rstrip("\n\r")

        if re.search(r"SPDX", line, re.IGNORECASE):
            header_lines.append(line)
            consumed = i - start_idx + 1
        elif header_lines:
            # We've collected header lines; check for trailing blank line
            if stripped == "":
                header_lines.append(line)
                consumed = i - start_idx + 1
            break
        elif stripped == "":
            # Skip leading blank lines before header
            continue
        elif stripped.startswith("#"):
            # Skip other comments before SPDX lines
            continue
        else:
            # Hit non-comment content before finding SPDX header
            break

    return header_lines, consumed


def update_license_header_in_file(file_path: Path, license_header: str) -> tuple[bool, str]:
    """Update license header in a single file.

    Args:
        file_path: Path to the file to update
        license_header: The license header to add/update

    Returns:
        Tuple of (was_modified, reason) where reason is one of:
        - "added" - header was added (none existed)
        - "updated" - header was replaced (different existed)
        - "unchanged" - header unchanged (identical existed)
        - "error" - an error occurred
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)

        if not lines:
            # Empty file, just add header (without trailing blank line)
            header_only = license_header.rstrip("\n") + "\n"
            file_path.write_text(header_only, encoding="utf-8")
            return True, "added"

        # Find where header should be (after shebang if present)
        insert_pos = 0
        if lines[0].startswith("#!"):
            insert_pos = 1

        # Find where to look for existing header (skip blank lines after shebang)
        header_search_start = insert_pos
        while header_search_start < len(lines) and lines[header_search_start].strip() == "":
            header_search_start += 1

        # Extract existing header if present
        existing_header_lines, num_header_lines = extract_license_header(lines, header_search_start)
        existing_header = "".join(existing_header_lines)

        if existing_header:
            # Calculate the range to replace (from header_search_start to end of header)
            header_end = header_search_start + num_header_lines
            remaining_lines = lines[header_end:]

            # Only include trailing blank line if there's content after the header
            has_content_after = any(line.strip() for line in remaining_lines)
            header_to_use = license_header if has_content_after else license_header.rstrip("\n") + "\n"

            # Compare exactly - header must match including whitespace
            if existing_header == header_to_use:
                return False, "unchanged"

            # Build new content: before header + new header + after header
            new_lines = lines[:header_search_start]
            new_lines.append(header_to_use)
            new_lines.extend(remaining_lines)

            file_path.write_text("".join(new_lines), encoding="utf-8")
            return True, "updated"

        # No existing header found, add one
        remaining_lines = lines[insert_pos:]
        has_content_after = any(line.strip() for line in remaining_lines)

        if has_content_after:
            # File has content, add header with blank line separator
            lines.insert(insert_pos, license_header)
            file_path.write_text("".join(lines), encoding="utf-8")
        else:
            # File has no real content, just write header without trailing blank line
            header_only = license_header.rstrip("\n") + "\n"
            file_path.write_text(header_only, encoding="utf-8")

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

        # Find where header should be
        start_idx = 0
        if lines[0].startswith("#!"):
            start_idx = 1

        # Skip blank lines
        while start_idx < len(lines) and lines[start_idx].strip() == "":
            start_idx += 1

        existing_header_lines, num_header_lines = extract_license_header(lines, start_idx)
        existing_header = "".join(existing_header_lines)

        if not existing_header:
            return False, "missing"

        # Determine expected header format based on whether there's content after
        header_end = start_idx + num_header_lines
        remaining_lines = lines[header_end:]
        has_content_after = any(line.strip() for line in remaining_lines)
        expected_header = license_header if has_content_after else license_header.rstrip("\n") + "\n"

        # Compare exactly - header must match including whitespace
        if existing_header == expected_header:
            return True, "match"

        return False, "mismatch"

    except (UnicodeDecodeError, PermissionError):
        return False, "error"


def should_process_file(file_path: Path) -> bool:
    """Determine if a file should be processed for license headers."""
    # Skip certain files
    skip_patterns = [
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

    # Skip if file path contains any skip patterns
    file_str = str(file_path)
    for pattern in skip_patterns:
        if pattern in file_str:
            return False

    # Only process Python files
    if file_path.suffix != ".py":
        return False

    # Skip certain specific files
    skip_files = ["_version.py"]

    if file_path.name in skip_files:
        return False

    return True


def main(path: Path, check_only: bool = False) -> tuple[int, int, int, list[Path]]:
    current_year = datetime.now().year
    LICENSE_HEADER = (
        f"# SPDX-FileCopyrightText: Copyright (c) {current_year} "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )

    # File patterns to process
    patterns = ["**/*.py"]

    processed_files = 0
    updated_files = 0
    skipped_files = 0
    files_needing_update: list[Path] = []

    for pattern in patterns:
        for file_path in path.glob(pattern):
            # Skip if not a file
            if not file_path.is_file():
                continue

            # Skip if file shouldn't be processed
            if not should_process_file(file_path):
                continue

            processed_files += 1

            if check_only:
                # Check mode - verify headers exist and match
                matches, status = check_license_header_matches(file_path, LICENSE_HEADER)
                if matches:
                    skipped_files += 1
                else:
                    files_needing_update.append(file_path)
                    updated_files += 1
            else:
                # Update mode - add or replace headers as needed
                was_modified, reason = update_license_header_in_file(file_path, LICENSE_HEADER)
                if was_modified:
                    if reason == "added":
                        print(f"  ✏️  Added header to {file_path}")
                    elif reason == "updated":
                        print(f"  🔄 Updated header in {file_path}")
                    updated_files += 1
                else:
                    skipped_files += 1

    return processed_files, updated_files, skipped_files, files_needing_update


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
    total_processed = 0
    total_updated = 0
    total_skipped = 0

    for folder in ["src", "tests", "scripts", "e2e_tests"]:
        folder_path = repo_path / folder
        if not folder_path.exists():
            continue

        if args.check:
            print(f"\n📂 Checking {folder}/")
        else:
            print(f"\n📂 Processing {folder}/")

        processed_files, updated_files, skipped_files, files_needing_update = main(folder_path, check_only=args.check)

        total_processed += processed_files
        total_updated += updated_files
        total_skipped += skipped_files
        all_files_needing_update.extend(files_needing_update)

        if args.check:
            print(f"   ❌ Need update: {updated_files}")
            print(f"   ✅ Up to date: {skipped_files}")
        else:
            print(f"   ✏️  Updated: {updated_files}")
            print(f"   ⏭️  Skipped: {skipped_files}")

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
            sys.exit(0)
    else:
        print(f"   ✏️  Updated: {total_updated}")
        print(f"   ⏭️  Skipped: {total_skipped}")
        print("\n✅ Done!")
        sys.exit(0)
