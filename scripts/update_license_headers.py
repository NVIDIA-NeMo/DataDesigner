# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


def add_license_header_to_file(file_path: Path, license_header: str) -> bool:
    """Add license header to a single file. Returns True if header was added."""
    try:
        # Read file content
        content = file_path.read_text(encoding="utf-8")

        # Check if license header already exists
        if has_license_header(content):
            return False

        # Handle shebang lines
        lines = content.splitlines(keepends=True)
        insert_pos = 0

        # If file starts with shebang, insert after it
        if lines and lines[0].startswith("#!"):
            insert_pos = 1
            # Add empty line after shebang if there isn't one
            if len(lines) > 1 and not lines[1].strip() == "":
                license_header += "\n"

        # Insert license header
        if insert_pos < len(lines):
            lines.insert(insert_pos, license_header)
        else:
            lines.append(license_header)

        # Write back to file
        file_path.write_text("".join(lines), encoding="utf-8")
        return True

    except (UnicodeDecodeError, PermissionError) as e:
        print(f"  ⏭️  Skipped {file_path} ({e})")
        return False


def has_license_header(file_content: str) -> bool:
    """Check if file already has a license header."""
    lines = file_content.splitlines()
    if not lines:
        return False

    # Check first few lines for license header patterns
    first_lines = lines[:10]  # Check first 10 lines
    license_pattern = r"SPDX\-License\-Identifier"

    for line in first_lines:
        if re.search(license_pattern, line, re.IGNORECASE):
            return True

    return False


def should_add_license_header(file_path: Path) -> bool:
    """Determine if a file should have a license header added."""
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

    # Allow __init__.py files that are not in the root of the SDK
    if file_path.name in skip_files:
        return False

    return True


def check_license_header(file_path: Path) -> bool:
    """Check if file has proper license header. Returns True if header exists."""
    try:
        content = file_path.read_text(encoding="utf-8")
        return has_license_header(content)
    except (UnicodeDecodeError, PermissionError):
        return False


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
    missing_headers: list[Path] = []

    for pattern in patterns:
        for file_path in path.glob(pattern):
            # Skip if not a file
            if not file_path.is_file():
                continue

            # Skip if file shouldn't have license header
            if not should_add_license_header(file_path):
                continue

            processed_files += 1

            if check_only:
                # Check mode - only verify headers exist
                if not check_license_header(file_path):
                    missing_headers.append(file_path)
                    updated_files += 1
                else:
                    skipped_files += 1
            else:
                # Update mode - add missing headers
                if add_license_header_to_file(file_path, LICENSE_HEADER):
                    print(f"  ✏️  Added header to {file_path}")
                    updated_files += 1
                else:
                    skipped_files += 1

    return processed_files, updated_files, skipped_files, missing_headers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add or check license headers in Python files")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if all files have license headers without modifying files",
    )
    args = parser.parse_args()

    repo_path = Path(__file__).parent.parent
    all_missing_headers: list[Path] = []
    total_processed = 0
    total_updated = 0
    total_skipped = 0

    for folder in ["src", "tests", "scripts"]:
        folder_path = repo_path / folder
        if not folder_path.exists():
            continue

        if args.check:
            print(f"\n📂 Checking {folder}/")
        else:
            print(f"\n📂 Processing {folder}/")

        processed_files, updated_files, skipped_files, missing_headers = main(folder_path, check_only=args.check)

        total_processed += processed_files
        total_updated += updated_files
        total_skipped += skipped_files
        all_missing_headers.extend(missing_headers)

        if args.check:
            print(f"   ❌ Missing: {updated_files}")
            print(f"   ✅ Present: {skipped_files}")
        else:
            print(f"   ✏️  Updated: {updated_files}")
            print(f"   ⏭️  Skipped: {skipped_files}")

    print("\n" + "=" * 80)
    print(f"📊 Summary: {total_processed} files processed")

    if args.check:
        print(f"   ❌ Missing headers: {total_updated}")
        print(f"   ✅ Has headers: {total_skipped}")

        if all_missing_headers:
            print(f"\n❌ {len(all_missing_headers)} file(s) missing license headers:")
            for file_path in all_missing_headers:
                print(f"   • {file_path}")
            print("💡 Run 'make update-license-headers' to fix")
            sys.exit(1)
        else:
            print("\n🎉 All files have proper license headers!")
            sys.exit(0)
    else:
        print(f"   ✏️  Updated: {total_updated}")
        print(f"   ⏭️  Skipped: {total_skipped}")
        print("\n✅ Done!")
        sys.exit(0)
